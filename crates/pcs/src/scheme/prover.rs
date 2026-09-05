use crate::challenge::Transcript;
use crate::fields::scalar::F162;
use crate::fold::fold_witness;
use crate::key::{AuxData, CommitmentKey};
use crate::labrador;
use crate::recursion;
use std::sync::Arc;
use std::time::Instant;
use super::*;

// =============================================================================================
// the prover
// =============================================================================================

/// The prover: the public matrix and the workspace the commitment and the fold pass between them.
///
/// The 85 MB of witness transform a commitment leaves behind is one `mmap` and 20 736 first
/// touches, ~20 ms of page faults the kernel charges to whoever writes the pages first — more
/// than the commitment itself. [`Prover::new`] pays that once, and [`Prover::fold`] hands the
/// buffer back, so a second [`commit`](Prover::commit) allocates nothing.
pub struct Prover {
    params: Params,
    key: Arc<CommitmentKey>,
    setup: Option<Arc<recursion::setup::Setup>>,
    workspace: Option<AuxData>,
}

impl Prover {
    /// Allocate the workspace and warm it up: one commitment and one fold over a zero witness,
    /// which touches every page of the workspace and runs every kernel and lazy table once.
    pub fn new(pp: &PublicParameters) -> Prover {
        let mut prover = Prover {
            params: pp.params.clone(),
            key: pp.key.clone(),
            setup: pp.recursion.clone(),
            workspace: None,
        };
        let verifier = Verifier::new(pp);
        let witness = Witness {
            params: pp.params.clone(),
            elements: vec![F162::ZERO; pp.params.witness_len()],
        };
        let (commitment, opening) = prover.commit(&witness);
        let mut transcript = Transcript::new(b"bin-ntt/warm-up");
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let row_evaluation = witness.row_evaluate(&point);
        let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
        std::hint::black_box(prover.fold(opening, &challenges));
        if !prover.params.recursion() && prover.params.dropped_bits() == 0 {
            std::hint::black_box(verifier.fold_commitment(&commitment, &challenges));
        }
        prover
    }

    pub fn commit(&mut self, witness: &Witness) -> (Commitment, CommitmentOpening) {
        assert_eq!(
            witness.params, self.params,
            "the witness was built for other parameters"
        );
        let mut aux = self.workspace.take().unwrap_or_else(|| {
            AuxData::new(self.key.len_ring(), self.params.columns(), self.key.limbs())
        });
        let matrix = self
            .key
            .commit_into_aux(&witness.elements, self.params.columns(), &mut aux);
        let columns = matrix.cols();
        let primes = self.params.primes();
        let (value, residues) = match &self.setup {
            None if self.params.dropped_bits() > 0 => {
                let dropped = crate::bd::drop_bits(&matrix, &primes, self.params.dropped_bits());
                (CommitmentValue::Dropped(Arc::new(dropped)), None)
            }
            None => (CommitmentValue::Matrix(matrix), None),
            Some(setup) => {
                let residues = recursion::limbs::residues(&matrix, &primes);
                let parts: Vec<&[i16]> = (0..setup.residues.len())
                    .map(|k| residues.flat(k))
                    .collect();
                let t_y = Arc::new(setup.key_y.commit_blocks(&parts));
                (CommitmentValue::Recursive(t_y), Some(residues))
            }
        };
        (
            Commitment {
                primes,
                columns,
                value,
            },
            CommitmentOpening { aux, residues },
        )
    }

    /// `T_u = Com_{H_u}(lift(u))`, the message the folding challenges are derived from when the
    /// left expansion is not sent.
    pub fn commit_left_expansion(&self, row: &RowEvaluation) -> LeftExpansionCommitment {
        let setup = self.setup.as_ref().expect("recursion is off");
        let mut u = vec![0i16; setup.ranks[recursion::U] * labrador::N];
        let lifts: Vec<[recursion::Poly; recursion::CHUNKS]> = row
            .values
            .iter()
            .map(|x| recursion::chunk::chunks(&recursion::binary::lift(x)))
            .collect();
        for b in 0..recursion::CHUNKS {
            for (j, l) in lifts.iter().enumerate() {
                let at = (b * setup.r + j) * labrador::N;
                u[at..at + labrador::N].copy_from_slice(&l[b]);
            }
        }
        LeftExpansionCommitment {
            t_u: Arc::new(setup.key_u.commit_blocks(&[&u])),
        }
    }

    pub fn fold(
        &mut self,
        opening: CommitmentOpening,
        challenges: &FoldingChallenges,
    ) -> FoldedWitness {
        let elements = fold_witness(
            &opening.aux,
            &challenges.challenges,
            self.key.len_ring() / 32,
            self.key.prime(0),
        );
        self.workspace = Some(opening.aux);
        FoldedWitness { elements }
    }

    /// The fold, the encoding of [`crate::recursion`], `T_R`, the exact norms, the mask scalars,
    /// and one LaBRADOR proof of the whole relation.
    pub fn prove_opening(
        &mut self,
        transcript: &mut Transcript,
        opening: CommitmentOpening,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        left: &LeftExpansionCommitment,
        row: &RowEvaluation,
        claimed_value: &F162,
        commitment: &Commitment,
    ) -> Result<OpeningProof, OpeningError> {
        let setup = self.setup.clone().expect("recursion is off");
        let CommitmentOpening { aux, residues } = opening;
        let residues = residues.expect("the opening holds no residues");
        let mut timings = OpeningTimings::default();
        let clock = Instant::now();
        let folded = self.fold(
            CommitmentOpening {
                aux,
                residues: None,
            },
            challenges,
        );
        timings.fold = clock.elapsed();
        let normsq: u64 = folded
            .elements
            .iter()
            .flat_map(|e| e.v)
            .map(|x| (x as i64 * x as i64) as u64)
            .sum();
        let cap = setup.fold_cap as u64;
        if std::env::var_os("GADGET_STATS").is_some() {
            eprintln!(
                "gadget-stats fold normsq {normsq} cap {cap} fill {:.3}",
                normsq as f64 / cap as f64
            );
        }
        if normsq > cap {
            return Err(OpeningError::FoldTooLong { normsq, cap });
        }

        let clock = Instant::now();
        let instance = recursion::Instance::new(
            &setup,
            &residues,
            &folded,
            row,
            challenges,
            point,
            claimed_value,
        )
        .map_err(OpeningError::GadgetOverflow)?;
        timings.encoding = clock.elapsed();
        let clock = Instant::now();
        let witness = instance.witness();
        timings.witness = clock.elapsed();
        let norms: Vec<u64> = instance.vectors.iter().map(|v| v.betasq()).collect();
        let rest: Vec<&[i16]> = setup
            .rest
            .iter()
            .map(|&i| witness.vectors[i].as_slice())
            .collect();
        let clock = Instant::now();
        let t_r = Arc::new(setup.key_r.commit_blocks(&rest));
        timings.t_r = clock.elapsed();

        absorb_opening(transcript, claimed_value, &t_r, &norms);
        let clock = Instant::now();
        let masks = recursion::statement::Masks::squeeze(&setup, transcript);
        timings.masks = clock.elapsed();
        let digest = statement_digest(transcript);
        let clock = Instant::now();
        let phi = recursion::statement::ProofPhi::new(&setup, &instance);
        timings.phi = clock.elapsed();
        let phi_bytes = phi.footprint();
        let opening = recursion::statement::Opening {
            t_y: commitment.t_y(),
            t_u: &left.t_u,
            t_r: &t_r,
            norms: &norms,
        };
        let clock = Instant::now();
        let statement =
            recursion::statement::build(&setup, &instance, &phi, opening, masks, digest);
        timings.statement = clock.elapsed();
        let clock = Instant::now();
        let proof = labrador::prove(&statement, &labrador::Witness::new(witness.vectors))
            .map_err(OpeningError::Labrador)?;
        timings.labrador = clock.elapsed();
        Ok(OpeningProof {
            t_r,
            norms,
            proof,
            timings,
            phi_bytes,
        })
    }
}

//! The recursive opening: an honest round, the two sides agreeing on the statement, the tampers
//! the relation has to catch, and the retry the fold's norm cap makes necessary.
use std::sync::Arc;
use std::time::Instant;

use bin_ntt::labrador;
use bin_ntt::recursion::setup::Setup;
use bin_ntt::recursion::statement::{build, Masks, Opening, ProofPhi};
use bin_ntt::recursion::Instance;
use bin_ntt::{
    Commitment, EvaluationPoint, FoldingChallenges, LeftExpansionCommitment, Modulus, OpeningError,
    OpeningProof, Params, Prover, PublicParameters, RowEvaluation, Transcript, Verifier, Witness,
    F162,
};

const MATRIX_SEED: [u8; 32] = [17u8; 32];
const WITNESS_SEED: [u8; 32] = [29u8; 32];

fn small(extra: Vec<Modulus>) -> Params {
    Params::new(9, 2, extra, true).unwrap()
}

fn small_based(base: Modulus, extra: Vec<Modulus>) -> Params {
    Params::with_base(9, 2, base, extra, true).unwrap()
}

/// A round driven up to the point where the opening is due.
struct Round {
    setup: Arc<Setup>,
    prover: Prover,
    verifier: Verifier,
    witness: Witness,
    commitment: Commitment,
    opening: Option<bin_ntt::CommitmentOpening>,
    transcript: Transcript,
    point: EvaluationPoint,
    claim: F162,
    row: RowEvaluation,
    left: LeftExpansionCommitment,
    challenges: FoldingChallenges,
}

impl Round {
    fn new(params: &Params, domain: &[u8]) -> Round {
        let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
        let setup = pp.recursion().expect("recursion is on").clone();
        let mut prover = Prover::new(&pp);
        let verifier = Verifier::new(&pp);
        let witness = Witness::random(params, WITNESS_SEED);
        let (commitment, opening) = prover.commit(&witness);
        let mut transcript = Transcript::new(domain);
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let claim = witness.mle_evaluate(&point);
        let row = witness.row_evaluate(&point);
        let left = prover.commit_left_expansion(&row);
        let challenges = verifier.derive_folding_challenges(&mut transcript, &left);
        Round {
            setup,
            prover,
            verifier,
            witness,
            commitment,
            opening: Some(opening),
            transcript,
            point,
            claim,
            row,
            left,
            challenges,
        }
    }

    fn prove(&mut self) -> Result<OpeningProof, OpeningError> {
        let opening = self.opening.take().expect("one opening per round");
        let mut transcript = self.transcript.clone();
        self.prover.prove_opening(
            &mut transcript,
            opening,
            &self.challenges,
            &self.point,
            &self.left,
            &self.row,
            &self.claim,
            &self.commitment,
        )
    }

    fn verify(&self, proof: &OpeningProof) -> bool {
        self.verify_with(&self.commitment, &self.left, &self.challenges, proof)
    }

    fn verify_with(
        &self,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
    ) -> bool {
        let mut transcript = self.transcript.clone();
        self.verifier
            .verify_opening(
                &mut transcript,
                commitment,
                left,
                &self.point,
                &self.claim,
                challenges,
                proof,
            )
            .is_ok()
    }
}

// =============================================================================================
// the honest round
// =============================================================================================

#[test]
fn an_honest_recursive_round_is_accepted() {
    for extra in [
        vec![Modulus::Q9721_FS_S],
        vec![Modulus::Q4861_Q_S],
        vec![Modulus::Q19441_FS_L],
    ] {
        let mut round = Round::new(&small(extra.clone()), b"bin-ntt/test/opening");
        let proof = round.prove().expect("the honest fold is within its cap");
        assert!(round.verify(&proof), "{extra:?}");
        assert_eq!(proof.norms().len(), round.setup.caps.len());
        assert!(proof
            .norms()
            .iter()
            .zip(&round.setup.caps)
            .all(|(n, c)| n <= c));
    }
}

/// The stage timings the proof carries account for the run that produced it: they are the real
/// wall clock of one proving and one verification, so each part is under its whole and the parts
/// are most of it.
#[test]
fn the_stage_timings_account_for_the_run() {
    let params = small(vec![Modulus::Q9721_FS_S]);
    let mut round = Round::new(&params, b"bin-ntt/test/opening/timings");
    let whole = Instant::now();
    let proof = round.prove().expect("the honest fold is within its cap");
    let prove = whole.elapsed();
    let t = proof.timings();
    let stages =
        t.fold + t.encoding + t.witness + t.t_r + t.masks + t.phi + t.statement + t.labrador;
    assert!(
        stages <= prove,
        "the stages outlast the proving they were measured inside"
    );
    assert!(
        stages.as_secs_f64() > 0.5 * prove.as_secs_f64(),
        "the stages are most of proving"
    );

    let mut transcript = round.transcript.clone();
    let whole = Instant::now();
    let v = round
        .verifier
        .verify_opening(
            &mut transcript,
            &round.commitment,
            &round.left,
            &round.point,
            &round.claim,
            &round.challenges,
            &proof,
        )
        .expect("the honest proof verifies");
    let verify = whole.elapsed();
    assert!(v.layout + v.bound + v.phi + v.statement <= v.rebuild);
    assert!(v.rebuild + v.labrador <= verify);
}

/// A recursive round over a quadratic-slot base and over one above `2^14`: the fold runs in that
/// limb's domain and comes back through its own inverse transform, and the relation the recursion
/// encodes is the same one. At this shape there are four challenges, so the fold's norm is far
/// from concentrated and the cap of the plan's D5 is hit often; the loop is the protocol's own
/// retry with fresh challenges.
#[test]
fn a_recursive_round_takes_any_base() {
    for (base, extra) in [
        (Modulus::Q2917_Q_S, vec![Modulus::Q9721_FS_S]),
        (Modulus::Q17497_FS_L, vec![Modulus::Q3889_FS_S]),
    ] {
        let params = small_based(base, extra.clone());
        assert_eq!(params.primes()[0], base.prime());
        let mut accepted = false;
        for attempt in 0..8u8 {
            let mut round = Round::new(
                &params,
                &[b"bin-ntt/test/opening/base/"[..].to_vec(), vec![attempt]].concat(),
            );
            let Ok(proof) = round.prove() else { continue };
            assert!(round.verify(&proof), "base {base:?}");
            assert!(proof
                .norms()
                .iter()
                .zip(&round.setup.caps)
                .all(|(n, c)| n <= c));
            accepted = true;
            break;
        }
        assert!(
            accepted,
            "no attempt cleared the fold cap for base {base:?}"
        );
    }
}

/// The non-recursive round is untouched by the flag.
#[test]
fn an_honest_plain_round_is_accepted() {
    let params = Params::new(9, 2, vec![Modulus::Q9721_FS_S], false).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/opening/plain");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claim = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
    let folded = prover.fold(opening, &challenges);
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row = verifier.fold_row_evaluation(&row, &challenges);
    verifier
        .verify_evaluation(&point, &claim, &row)
        .expect("the claim");
    verifier
        .verify_folded_opening(&folded_commitment, &folded, &point, &folded_row)
        .expect("the opening");
}

// =============================================================================================
// the two sides build the same statement
// =============================================================================================

#[test]
fn the_prover_and_the_verifier_encode_the_same_relation() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/same",
    );
    let residues = round.opening.as_ref().unwrap().residues().unwrap().clone();
    let proof = round.prove().expect("honest opening");
    let folded = {
        let (_, opening) = round.prover.commit(&round.witness);
        round.prover.fold(opening, &round.challenges)
    };
    let full = Instance::new(
        &round.setup,
        &residues,
        &folded,
        &round.row,
        &round.challenges,
        &round.point,
        &round.claim,
    );
    let layout = Instance::layout(&round.setup, &round.challenges, &round.point, &round.claim);

    let (a, b) = (full.statement(), layout.statement());
    assert_eq!(a.constraints, b.constraints);
    assert_eq!(a.residues, b.residues);
    assert_eq!(a.rest, b.rest);
    for (x, y) in a.vectors.iter().zip(&b.vectors) {
        assert_eq!(
            (&x.name, x.n, x.cap_betasq, x.binary),
            (&y.name, y.n, y.cap_betasq, y.binary)
        );
    }

    let masks = || Masks::squeeze(&round.setup, &mut round.transcript.clone());
    let opening = || Opening {
        t_y: round.commitment.t_y(),
        t_u: round.left.t_u(),
        t_r: proof.t_r(),
        norms: proof.norms(),
    };
    let one = build(
        &round.setup,
        &full,
        &ProofPhi::new(&round.setup, &full),
        opening(),
        masks(),
        [7u8; 32],
    );
    let two = build(
        &round.setup,
        &layout,
        &ProofPhi::new(&round.setup, &layout),
        opening(),
        masks(),
        [7u8; 32],
    );
    assert_eq!(one.content_digest(), two.content_digest());
    assert!(
        labrador::verify(&two, proof.handle()).is_err(),
        "the digest is not the round's"
    );
}

// =============================================================================================
// tampering
// =============================================================================================

#[test]
fn a_wrong_norm_is_rejected() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/norm",
    );
    let proof = round.prove().expect("honest opening");
    assert!(round.verify(&proof));

    let mut lowered = round.prove_again();
    lowered.norms_mut()[0] -= 1;
    assert!(!round.verify(&lowered), "a norm the witness exceeds");

    let mut raised = round.prove_again();
    raised.norms_mut()[0] = round.setup.caps[0] + 1;
    assert!(!round.verify(&raised), "a norm above its cap");
}

#[test]
fn a_modified_left_expansion_commitment_is_rejected() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/left",
    );
    let proof = round.prove().expect("honest opening");
    let other = {
        let mut row = round.row.clone();
        row.values_mut()[0] += F162::ONE;
        round.prover.commit_left_expansion(&row)
    };
    assert_ne!(other.t_u(), round.left.t_u());
    assert!(!round.verify_with(&round.commitment, &other, &round.challenges, &proof));
}

/// `T_Y` is what binds the residues the proof opens, so a proof made for one set of residues
/// does not verify against the commitment to another.
#[test]
fn a_commitment_to_other_residues_is_rejected() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/residue",
    );
    let proof = round.prove().expect("honest opening");
    let mut other = round.witness.elements().to_vec();
    other[0] += F162::ONE;
    let other = Witness::from_elements(&small(vec![Modulus::Q9721_FS_S]), other).unwrap();
    let (commitment, opening) = round.prover.commit(&other);
    assert_ne!(commitment.t_y(), round.commitment.t_y());
    assert_ne!(
        opening.residues().unwrap().vectors[0][0],
        round
            .opening
            .as_ref()
            .map_or([0i16; 64], |o| o.residues().unwrap().vectors[0][0])
    );
    assert!(!round.verify_with(&commitment, &round.left, &round.challenges, &proof));
}

/// A coefficient at a position the encoding claims is zero is seen by nothing but the mask
/// constraints, so it is the one thing that tests them.
#[test]
fn a_coefficient_at_a_zero_position_is_caught_by_the_masks() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/zero",
    );
    let residues = round.opening.as_ref().unwrap().residues().unwrap().clone();
    let opening = round.opening.take().unwrap();
    let folded = round.prover.fold(opening, &round.challenges);
    let instance = Instance::new(
        &round.setup,
        &residues,
        &folded,
        &round.row,
        &round.challenges,
        &round.point,
        &round.claim,
    );
    let setup = &round.setup;
    let phi = ProofPhi::new(setup, &instance);
    let masks = || Masks::squeeze(setup, &mut round.transcript.clone());

    let mut witness = instance.witness();
    let padded = (0..setup.used.len())
        .find(|&i| setup.used[i] < setup.ranks[i])
        .expect("some vector is padded");
    witness.vectors[padded][setup.used[padded] * labrador::N] = 1;
    let norms: Vec<u64> = witness
        .vectors
        .iter()
        .map(|v| v.iter().map(|&x| (x as i64 * x as i64) as u64).sum())
        .collect();
    let rest: Vec<&[i16]> = setup
        .rest
        .iter()
        .map(|&i| witness.vectors[i].as_slice())
        .collect();
    let t_r = Arc::new(setup.key_r.commit_blocks(&rest));
    let statement = build(
        setup,
        &instance,
        &phi,
        Opening {
            t_y: round.commitment.t_y(),
            t_u: round.left.t_u(),
            t_r: &t_r,
            norms: &norms,
        },
        masks(),
        [3u8; 32],
    );
    let err = labrador::prove_verified(&statement, &labrador::Witness::new(witness.vectors))
        .expect_err("a coefficient outside its support");
    assert!(err.contains("simple_verify"), "{err}");
}

#[test]
fn a_wrong_challenge_set_is_rejected() {
    let mut round = Round::new(
        &small(vec![Modulus::Q9721_FS_S]),
        b"bin-ntt/test/opening/challenges",
    );
    let proof = round.prove().expect("honest opening");
    let mut other = Transcript::new(b"bin-ntt/test/opening/challenges/other");
    let challenges = round
        .verifier
        .derive_folding_challenges(&mut other, &round.left);
    assert!(!round.verify_with(&round.commitment, &round.left, &challenges, &proof));
}

// =============================================================================================
// the norm cap and its retry
// =============================================================================================

/// The cap of the plan's D5 is the 95th percentile of the honest fold, so the prover has to be
/// able to say no and be given fresh challenges. Four challenges put the fold far from
/// concentrated, so both outcomes are common; the loop is the protocol's own retry.
#[test]
fn a_long_fold_is_refused_and_the_retry_succeeds() {
    let params = small(vec![Modulus::Q9721_FS_S]);
    let (mut refused, mut accepted) = (false, false);
    for tag in 0..32u8 {
        let mut round = Round::new(&params, &[b"cap/"[..].to_vec(), vec![tag]].concat());
        match round.prove() {
            Err(OpeningError::FoldTooLong { normsq, cap }) => {
                assert!(normsq > cap);
                refused = true;
            }
            Ok(proof) => {
                assert!(round.verify(&proof));
                accepted = true;
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
        if refused && accepted {
            return;
        }
    }
    panic!("the cap never fired both ways: refused {refused}, accepted {accepted}");
}

impl Round {
    /// Another proof of the same round, for a test that tampers with the result.
    fn prove_again(&mut self) -> OpeningProof {
        let (_, opening) = self.prover.commit(&self.witness);
        let mut transcript = self.transcript.clone();
        self.prover
            .prove_opening(
                &mut transcript,
                opening,
                &self.challenges,
                &self.point,
                &self.left,
                &self.row,
                &self.claim,
                &self.commitment,
            )
            .expect("honest opening")
    }
}

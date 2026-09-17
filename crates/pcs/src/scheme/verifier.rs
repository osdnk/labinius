use crate::challenge::{
    sample_short_challenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use crate::eval;
use crate::fields::scalar::F162;
use crate::fold::{a_times_v_forward, fold_columns_slots};
use crate::key::CommitmentKey;
use crate::labrador;
use crate::recursion;
use crate::ring::{
    components_of, Batch32, PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs,
    Representation, N162,
};
use crate::simd::transpose32 as tr;
use std::sync::Arc;
use std::time::Instant;
use super::*;

// =============================================================================================
// the verifier
// =============================================================================================

/// The verifier: the public parameters and nothing else.
pub struct Verifier {
    params: Params,
    matrix_seed: [u8; 32],
    key: Arc<CommitmentKey>,
    setup: Option<Arc<recursion::setup::Setup>>,
}

impl Verifier {
    pub fn new(pp: &PublicParameters) -> Verifier {
        Verifier {
            params: pp.params.clone(),
            matrix_seed: pp.matrix_seed,
            key: pp.key.clone(),
            setup: pp.recursion.clone(),
        }
    }

    /// The shape, the moduli, the key seed and — with recursion on — LaBRADOR's modulus, absorbed
    /// before anything the prover chooses.
    fn absorb_parameters(&self, transcript: &mut Transcript) {
        transcript.absorb_bytes(b"labinius/parameters");
        transcript.absorb_u64(self.params.witness_log_len as u64);
        transcript.absorb_u64(self.params.column_log_len as u64);
        for q in self.params.primes() {
            transcript.absorb_u64(q as u64);
        }
        transcript.absorb_u64(u64::from(self.params.recursion()));
        if self.params.dropped_bits() > 0 {
            transcript.absorb_bytes(b"labinius/dropped-bits");
            transcript.absorb_u64(self.params.dropped_bits() as u64);
        }
        if self.params.recursion() {
            transcript.absorb_u64(labrador::logq() as u64);
        }
        transcript.absorb_bytes(&self.matrix_seed);
    }

    /// Absorb the commitment, then derive `p = (p0, p1)`: one uniform `F162` per variable from a
    /// single extendable output of the transcript.
    pub fn derive_evaluation_point(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
    ) -> EvaluationPoint {
        self.absorb_parameters(transcript);
        transcript.absorb_bytes(b"labinius/commitment");
        transcript.absorb_u64(commitment.columns() as u64);
        match commitment.value() {
            CommitmentValue::Matrix(m) => {
                for j in 0..commitment.columns() {
                    transcript.absorb_elements(m.column(j));
                }
            }
            CommitmentValue::Recursive(t) => transcript.absorb_bytes(t.as_bytes()),
            CommitmentValue::Dropped(d) => {
                transcript.absorb_bytes(b"labinius/dropped-commitment");
                transcript.absorb_bytes(&crate::wire::pack_dropped(d));
            }
        }
        let (rows, cols) = (
            self.params.row_log_len() as usize,
            self.params.column_log_len as usize,
        );
        let mut bytes = vec![0u8; 24 * (rows + cols)];
        transcript.fill(b"labinius/evaluation-point", &mut bytes);
        let element = |n: usize| {
            let mut limb = [0u64; 3];
            for (k, l) in limb.iter_mut().enumerate() {
                let o = 24 * n + 8 * k;
                *l = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
            }
            limb[2] &= (1u64 << 34) - 1;
            F162(limb)
        };
        EvaluationPoint {
            p0: (0..rows).map(element).collect(),
            p1: (0..cols).map(|k| element(rows + k)).collect(),
        }
    }

    /// Absorb `u`, then derive the `columns()` challenges — weight 28, canonical bound 12, one
    /// transcript derivation per challenge index.
    pub fn derive_folding_challenges(
        &self,
        transcript: &mut Transcript,
        source: &impl FoldingSource,
    ) -> FoldingChallenges {
        source.absorb(transcript);
        FoldingChallenges {
            challenges: (0..self.params.columns())
                .map(|_| sample_short_challenge(transcript, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
                .collect(),
        }
    }

    /// `sum_j c_j C_j` per modulus. Multiplication by a challenge — an element of the subring
    /// `R_162` — acts on the four components of a commitment alike and slot-wise in the `R_162`
    /// transform, so the whole step is four length-`r` inner products per slot and modulus.
    pub fn fold_commitment(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
    ) -> FoldedCommitment {
        assert_eq!(
            challenges.len(),
            commitment.columns(),
            "one challenge per column"
        );
        let limbs = self.key.limbs();
        let columns = commitment.columns();
        let mut rows: [PowerOfThreeRingElementWithLimbs; 4] =
            core::array::from_fn(|_| PowerOfThreeRingElementWithLimbs::zero(limbs));
        let mut cols: Vec<[*const i16; 4]> = vec![[core::ptr::null(); 4]; columns];
        for k in 0..limbs {
            let q = self.key.prime(k);
            for (j, c) in cols.iter_mut().enumerate() {
                for (t, p) in c.iter_mut().enumerate() {
                    *p = commitment.matrix().get(t, j).limbs[k].v.as_ptr();
                }
            }
            let mut out = [[0i16; N162]; 4];
            fold_columns_slots(q, &challenges.challenges, &cols, &mut out);
            for (t, o) in out.iter().enumerate() {
                rows[t].limbs[k].v = *o;
            }
        }
        FoldedCommitment {
            primes: self.params.primes(),
            rows,
        }
    }

    /// `u' = sum_j u_j (c_j mod 2)` over `F162`.
    pub fn fold_row_evaluation(
        &self,
        row_evaluation: &RowEvaluation,
        challenges: &FoldingChallenges,
    ) -> F162 {
        eval::fold_binary(&row_evaluation.values, &challenges.challenges)
    }

    /// The claim check: `u . eq(p1) == t`.
    pub fn verify_evaluation(
        &self,
        point: &EvaluationPoint,
        claimed_value: &F162,
        row_evaluation: &RowEvaluation,
    ) -> Result<(), VerificationError> {
        if row_evaluation.values.len() != self.params.columns()
            || point.p1.len() != self.params.column_log_len as usize
        {
            return Err(VerificationError::Rejected);
        }
        if eval::claim(&row_evaluation.values, &point.p1) == *claimed_value {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The opening check, recomputed from `v` alone: `v` is centered modulo the base modulus,
    /// `A v` equals the folded commitment on every modulus, and `eq(p0) . (v mod 2) == u'`.
    fn verify_folded_opening(
        &self,
        folded_commitment: &FoldedCommitment,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || folded_commitment.primes != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            if e.representation != Representation::Coefficients {
                return Err(VerificationError::Rejected);
            }
            let (sq, top) = unsafe { crate::simd::norm::normsq_and_max(&e.v) }
                .ok_or(VerificationError::Rejected)?;
            normsq += sq;
            worst = worst.max(top);
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }

        let mut batches: Vec<Batch32> = (0..v.len() / 32)
            .map(|_| Batch32::zero(Representation::Coefficients))
            .collect();
        for (b, batch) in batches.iter_mut().enumerate() {
            let mut src = [tr::ZERO_ROW.as_ptr(); 32];
            for (p, s) in src.iter_mut().enumerate() {
                *s = v[32 * b + p].v.as_ptr();
            }
            unsafe { tr::transpose_into(&src, &tr::IDENTITY, batch) };
        }
        for k in 0..self.key.limbs() {
            let q = self.key.prime(k);
            let y = a_times_v_forward(q, self.key.row(k), &batches);
            let components: [PowerOfThreeRingElement; 4] = components_of(q, &y);
            for (row, c) in components.iter().enumerate() {
                if *c != folded_commitment.rows[row].limbs[k] {
                    return Err(VerificationError::Rejected);
                }
            }
        }

        if eval::binary_check(&point.p0, v, *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    fn verify_folded_opening_bd(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || commitment.moduli() != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let dropped = match commitment.value() {
            CommitmentValue::Dropped(d) => d,
            _ => return Err(VerificationError::Rejected),
        };
        if dropped.dropped_bits() != self.params.dropped_bits() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            if e.representation != Representation::Coefficients {
                return Err(VerificationError::Rejected);
            }
            let (sq, top) = unsafe { crate::simd::norm::normsq_and_max(&e.v) }
                .ok_or(VerificationError::Rejected)?;
            normsq += sq;
            worst = worst.max(top);
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }

        let residual = crate::bd::residual(&self.key, dropped, &challenges.challenges, v)
            .ok_or(VerificationError::Rejected)?;
        if residual > self.params.bd_cap() as u128 {
            return Err(VerificationError::Rejected);
        }

        if eval::binary_check(&point.p0, v, *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The opening check in the mode [`Params::opening`] names, dispatching to the arm the
    /// message carries. What comes back on acceptance is the wall clock of each stage of the
    /// recursive arm, so that a caller reporting a breakdown does not have to verify twice; the
    /// other two arms leave it zero.
    pub fn verify_opening(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        message: OpeningMessage<'_>,
    ) -> Result<VerifyTimings, VerificationError> {
        match (self.params.opening, message) {
            (
                Opening::Clear,
                OpeningMessage::Clear {
                    folded_commitment,
                    folded_witness,
                    folded_row_value,
                },
            ) => self
                .verify_folded_opening(folded_commitment, folded_witness, point, folded_row_value)
                .map(|()| VerifyTimings::default()),
            (
                Opening::BitDropped { .. },
                OpeningMessage::BitDropped {
                    folded_witness,
                    folded_row_value,
                },
            ) => self
                .verify_folded_opening_bd(
                    commitment,
                    challenges,
                    folded_witness,
                    point,
                    folded_row_value,
                )
                .map(|()| VerifyTimings::default()),
            (
                Opening::Recursive,
                OpeningMessage::Recursive {
                    transcript,
                    left,
                    claimed_value,
                    proof,
                },
            ) => self.verify_recursive_opening(
                transcript,
                commitment,
                left,
                point,
                claimed_value,
                challenges,
                proof,
            ),
            _ => Err(VerificationError::Rejected),
        }
    }

    /// The recursive opening check: the announced norms against their caps, the no-wraparound
    /// bound of [`recursion::bound`] against `Q / 2`, and one LaBRADOR verification of the
    /// statement rebuilt from public data alone.
    #[allow(clippy::too_many_arguments)]
    fn verify_recursive_opening(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
    ) -> Result<VerifyTimings, VerificationError> {
        let mut timings = VerifyTimings::default();
        let clock = Instant::now();
        let statement = self.rebuild(
            transcript,
            commitment,
            left,
            point,
            claimed_value,
            challenges,
            proof,
            &mut timings,
        )?;
        timings.rebuild = clock.elapsed();
        let clock = Instant::now();
        labrador::verify(&statement, &proof.proof).map_err(|_| VerificationError::Rejected)?;
        timings.labrador = clock.elapsed();
        Ok(timings)
    }

    /// The statement the recursive arm of [`verify_opening`](Self::verify_opening) hands to
    /// LaBRADOR: the same function of public data that the prover ran, and what a test compares
    /// against.
    pub fn opening_statement(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
    ) -> Result<labrador::Statement, VerificationError> {
        self.rebuild(
            transcript,
            commitment,
            left,
            point,
            claimed_value,
            challenges,
            proof,
            &mut VerifyTimings::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
        timings: &mut VerifyTimings,
    ) -> Result<labrador::Statement, VerificationError> {
        let setup = self.setup.as_ref().ok_or(VerificationError::Rejected)?;
        if proof.norms.len() != setup.caps.len()
            || proof.norms.iter().zip(&setup.caps).any(|(n, c)| n > c)
        {
            return Err(VerificationError::Rejected);
        }
        absorb_opening(transcript, claimed_value, &proof.t_r, &proof.norms);
        let masks = recursion::statement::Masks::squeeze(setup, transcript);
        let digest = statement_digest(transcript);
        let clock = Instant::now();
        let layout = recursion::Instance::layout(setup, challenges, point, claimed_value);
        timings.layout = clock.elapsed();
        let clock = Instant::now();
        let cleared = layout.clears();
        timings.bound = clock.elapsed();
        if !cleared {
            return Err(VerificationError::Rejected);
        }
        let clock = Instant::now();
        let phi = recursion::statement::ProofPhi::new(setup, &layout);
        timings.phi = clock.elapsed();
        let opening = recursion::statement::Opening {
            t_y: commitment.t_y(),
            t_u: &left.t_u,
            t_r: &proof.t_r,
            norms: &proof.norms,
        };
        let clock = Instant::now();
        let statement = recursion::statement::build(setup, &layout, &phi, opening, masks, digest);
        timings.statement = clock.elapsed();
        Ok(statement)
    }
}

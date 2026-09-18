//! The cross-field switch as one call each way: a claim on the *bits* of a `B128` trace at a
//! point of `B128^(7 + l)` becomes the `F162` claim the commitment opens, at the point the
//! switch's sumcheck ends on. The prover and the verifier drive the same [`Transcript`].
use crate::challenge::Transcript;
use crate::fields::crossfield as cf;
use crate::fields::scalar::{B128, F162};
use crate::scheme::{EvaluationPoint, Params};

/// Bits per word: the low seven coordinates of a bit-level point pick the bit within a word.
pub const LOG_WORD: usize = 7;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchProof {
    /// The 128 partial evaluations, one per bit position.
    pub partials: Vec<B128>,
    /// The sumcheck, two `F162` per round.
    pub rounds: Vec<[F162; 2]>,
    /// The value of the lifted trace at the point the switch ends on.
    pub opened: F162,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwitchError(pub &'static str);

impl std::fmt::Display for SwitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cross-field switch: {}", self.0)
    }
}

impl std::error::Error for SwitchError {}

fn absorb_statement(transcript: &mut Transcript, r_lo: &[B128], r_hi: &[B128], claim: B128) {
    transcript.absorb_bytes(b"labinius/switch");
    transcript.absorb_b128(r_lo);
    transcript.absorb_b128(r_hi);
    transcript.absorb_b128(&[claim]);
}

/// Prove that the bits of `trace` evaluate to `claim` at `(r_lo, r_hi)`. The point returned is
/// where the commitment to `Witness::lifted(params, trace)` must open to `proof.opened`.
pub fn prove(
    params: &Params,
    trace: &[B128],
    r_lo: &[B128],
    r_hi: &[B128],
    claim: B128,
    transcript: &mut Transcript,
) -> (SwitchProof, EvaluationPoint) {
    assert_eq!(r_lo.len(), LOG_WORD);
    assert_eq!(r_hi.len(), params.witness_log_len as usize);
    assert_eq!(trace.len(), params.witness_len());
    absorb_statement(transcript, r_lo, r_hi, claim);

    let (partials, eq_hi) = cf::SwitchProver::partial_evals_and_eq(trace, r_hi);
    transcript.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&transcript.sample_f162(b"labinius/switch-batch", LOG_WORD));
    let mut prover = cf::SwitchProver::new(trace, &eq_hi, &batch);
    let mut rounds = Vec::with_capacity(r_hi.len());
    let mut r_pp = Vec::with_capacity(r_hi.len());
    for _ in 0..r_hi.len() {
        let msg = prover.msg();
        transcript.absorb_f162(&msg);
        let r = transcript.sample_f162(b"labinius/switch-round", 1)[0];
        prover.fold(r);
        rounds.push(msg);
        r_pp.push(r);
    }
    let proof = SwitchProof {
        partials,
        rounds,
        opened: prover.final_eval(),
    };
    (proof, EvaluationPoint::msb_first(params, &r_pp))
}

/// The verifier's half: on success, the point at which the commitment must open to the value
/// returned with it.
pub fn verify(
    params: &Params,
    r_lo: &[B128],
    r_hi: &[B128],
    claim: B128,
    proof: &SwitchProof,
    transcript: &mut Transcript,
) -> Result<(EvaluationPoint, F162), SwitchError> {
    let l = params.witness_log_len as usize;
    if r_lo.len() != LOG_WORD || r_hi.len() != l {
        return Err(SwitchError("the point does not match the parameters"));
    }
    if proof.partials.len() != 1 << LOG_WORD || proof.rounds.len() != l {
        return Err(SwitchError("the proof does not match the parameters"));
    }
    absorb_statement(transcript, r_lo, r_hi, claim);

    transcript.absorb_b128(&proof.partials);
    let batch = cf::eq_expand_f162(&transcript.sample_f162(b"labinius/switch-batch", LOG_WORD));
    let mut verifier =
        cf::SwitchVerifier::start(&proof.partials, claim, r_lo, &batch).map_err(SwitchError)?;
    let mut r_pp = Vec::with_capacity(l);
    for &msg in &proof.rounds {
        transcript.absorb_f162(&msg);
        let r = transcript.sample_f162(b"labinius/switch-round", 1)[0];
        verifier.round(msg, r);
        r_pp.push(r);
    }
    verifier
        .finish(r_hi, &r_pp, &batch, proof.opened)
        .map_err(SwitchError)?;
    Ok((EvaluationPoint::msb_first(params, &r_pp), proof.opened))
}

/// The multilinear extension of the trace's bits at `(r_lo, r_hi)`, bit `k` of word `j` at
/// index `k + 128 j`: what an honest claim is.
pub fn bit_mle(trace: &[B128], r_lo: &[B128], r_hi: &[B128]) -> B128 {
    let eq_lo = cf::eq_expand_b128(r_lo);
    let eq_hi = cf::eq_expand_b128(r_hi);
    trace
        .iter()
        .zip(&eq_hi)
        .fold(B128::ZERO, |acc, (&word, &e)| {
            let bits = (0..128)
                .filter(|&k| word.bit(k))
                .fold(B128::ZERO, |s, k| s + eq_lo[k]);
            acc + bits * e
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme::{Opening, SUITES};

    #[test]
    fn round_trip_and_rejection() {
        let suite = SUITES.iter().find(|s| s.name == "sizes").unwrap();
        let params = Params::sized(suite, Opening::Clear);
        let l = params.witness_log_len as usize;
        let mut rng = Transcript::new(b"switch/test");
        let trace = rng.sample_b128(b"trace", 1 << l);
        let r_lo = rng.sample_b128(b"r_lo", LOG_WORD);
        let r_hi = rng.sample_b128(b"r_hi", l);
        let claim = bit_mle(&trace, &r_lo, &r_hi);

        let (proof, point) = prove(
            &params,
            &trace,
            &r_lo,
            &r_hi,
            claim,
            &mut Transcript::new(b"t"),
        );
        let (point2, value) = verify(
            &params,
            &r_lo,
            &r_hi,
            claim,
            &proof,
            &mut Transcript::new(b"t"),
        )
        .unwrap();
        assert_eq!(point, point2);
        assert_eq!(value, proof.opened);
        assert_eq!(
            value,
            crate::scheme::Witness::lifted(&params, &trace)
                .unwrap()
                .mle_evaluate(&point)
        );

        let wrong = claim + B128::ONE;
        assert!(verify(
            &params,
            &r_lo,
            &r_hi,
            wrong,
            &proof,
            &mut Transcript::new(b"t")
        )
        .is_err());
        let mut tampered = proof.clone();
        tampered.opened = tampered.opened + F162::ONE;
        assert!(verify(
            &params,
            &r_lo,
            &r_hi,
            claim,
            &tampered,
            &mut Transcript::new(b"t")
        )
        .is_err());
    }
}

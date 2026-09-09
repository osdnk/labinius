//! The tensor commitment of [`crate::tensor`] over the Brakedown code, which is what the Brakedown
//! paper's own polynomial commitment is.
//!
//! The code is linear-time to encode and its provable relative distance is small, `beta/r` between
//! 0.02 and 0.07 across the six specs, so the query count of the unique-decoding analysis is in
//! the thousands and the opened columns are what the proof weighs. The commitment is one Merkle
//! root; the verifier's own work is two encodings of length `k` and the column checks.
//!
//! `spec` indexes the six parameter rows of Figure 2 of GLSTW21, which is what fixes the rate and
//! the provable distance and therefore both the query count and the balanced split.
use super::codes::brakedown_code::{self, BrakedownCode};
use super::codes::LinearCode;
use super::tensor::{self, Tensor};
use super::{median_of, once, Row};

const CODE_SEED: u64 = 0x3B;

const POINT_SEED: u64 = 0x1D;

const PROBE_K: usize = 1 << 12;

/// The shape is read off a probe of the code itself rather than off the spec, so the `delta` and
/// the rate the split is chosen from are exactly the ones [`Tensor`] then asserts against.
fn code(log_len: usize, spec: usize, security_bits: usize) -> BrakedownCode {
    let row = brakedown_code::SPEC[spec];
    let probe = BrakedownCode::new(PROBE_K, row, CODE_SEED);
    let log_k = tensor::balanced_log_k(
        log_len,
        probe.relative_distance(),
        probe.rate(),
        security_bits,
    );
    BrakedownCode::new(1 << log_k, row, CODE_SEED)
}

pub fn run(log_len: usize, spec: usize, security_bits: usize, u64s: &[u64]) -> Row {
    let code = code(log_len, spec, security_bits);
    let tensor = Tensor::new(&code, log_len, security_bits);
    let values = tensor::elements(log_len, u64s);
    let point = tensor::eval_point(log_len, POINT_SEED);

    let (commit_ms, commitment) = median_of(3, || tensor::commit(&tensor, &values));

    let (_, (proof, claim, timing)) = once(|| tensor::prove(&tensor, &values, &point, None));
    let (verify_ms, verified) = median_of(3, || tensor::verify(&tensor, &proof, &point, claim));
    verified.expect("the honest opening verifies");

    let (tampered, tampered_claim, _) = tensor::prove(&tensor, &values, &point, Some(0));
    assert!(
        tensor::verify(&tensor, &tampered, &point, tampered_claim).is_err(),
        "an opening of a matrix the root does not bind must be rejected"
    );

    Row {
        scheme: "Brakedown tensor",
        rate: format!("{:.3}", code.rate()),
        target: format!("{security_bits}"),
        security: format!(
            "unique decoding to d/3 at delta {:.3}, {} queries, no grinding",
            code.relative_distance(),
            tensor.n_queries()
        ),
        claim: "element-MLE",
        commit_ms,
        open_ms: timing.opening,
        verify_ms,
        commitment: commitment.len(),
        proof: proof.len() - commitment.len(),
    }
}

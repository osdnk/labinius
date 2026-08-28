//! NTT over R_q = Z_q[X] / (X^648 - X^324 + 1), the 1944-th cyclotomic ring (1944 = 2^3 * 3^5),
//! for q in {3889, 9721} (q = 1 mod 1944, so R_q splits into 648 linear factors) and for
//! q in {2917, 4861, 12637} (q = 1 mod 972 only, so it splits into 324 quadratic factors),
//! specialised for binary (0/1) input polynomials and vectorised with AVX-512 across many
//! polynomials. A commitment is taken over a list of those primes as limbs (`api`).
//!
//! Module map
//! - `params`  : all ring constants (roots, twiddles, Montgomery forms), computed at compile time.
//! - `types`   : `RingElement` (rokoko-style), `Batch32`.
//! - `scalar`  : exact reference implementation (schoolbook product mod Phi_1944, NTT).
//! - `rng`     : tiny deterministic RNG (no external crates).
//! - `simd`    : the AVX-512 kernels.
//! - `api`     : the public API (`AdditionalLimb`, `CommitmentKey`, `PowerOfThreeRingElement`,
//!               the height-4 view).
//! - `challenge`: short (fixed-weight ternary) challenges over `R_162` and the blake3 transcript.
//! - `fold`    : the folding step `v = sum_j c_j W_j` in the NTT domain, on top of a commitment.
//! - `eval`    : the left-expansion over `F162`, the binary side of the fold, and the verifier.
#![allow(clippy::needless_range_loop)]

pub mod api;
pub mod challenge;
pub mod eval;
pub mod f162;
pub mod fold;
pub mod params;
pub mod rng;
pub mod scalar;
pub mod simd;
pub mod types;

pub use api::*;
pub use fold::{fold, fold_checked, fold_with, FoldOutput, FoldTimings};
pub use eval::{
    check_claim, components_mod_2, eq_table, evaluate_mle, fold_binary, left_expand, sample_point,
    verify_binary, verify_fold, EvalPoint, LeftExpansion, RawCommitments, Verifier,
};
pub use challenge::{
    canonical_inf_norm_sq, sample_attempt, sample_short_challenge, ShortChallenge, Transcript,
    DEFAULT_BOUND, DEFAULT_WEIGHT, MAX_WEIGHT,
};
pub use params::{N, QS};
pub use types::*;

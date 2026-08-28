//! NTT over R_q = Z_q[X] / (X^648 - X^324 + 1), the 1944-th cyclotomic ring (1944 = 2^3 * 3^5),
//! for q in {3889, 9721} (both q = 1 mod 1944, so R_q splits into 648 linear factors), specialised
//! for binary (0/1) input polynomials and vectorised with AVX-512 across many polynomials.
//!
//! Module map
//! - `params`  : all ring constants (roots, twiddles, Montgomery forms), computed at compile time.
//! - `types`   : `BinaryPoly`, `BinaryBatch32`, `RingElement` (rokoko-style), `Batch32`.
//! - `scalar`  : exact reference implementation (lift, schoolbook product mod Phi_1944, NTT).
//! - `rng`     : tiny deterministic RNG for tests/benches (no external crates).
//! - `perf`    : perf_event_open wrapper (cycles, instructions, uops per port) for the bench.
//! - `simd`    : the AVX-512 kernels (one module per variant).
//! - `api`     : the public API (`CommitmentKey`, `PowerOfThreeRingElement`, the height-4 view).
#![allow(clippy::needless_range_loop)]

pub mod api;
pub mod f162;
pub mod params;
pub mod perf;
pub mod rng;
pub mod scalar;
pub mod simd;
pub mod types;

pub use api::*;
pub use params::{N, QS};
pub use types::*;

//! A commitment, a fold and their verifier over `R_648 = Z_q[X]/(X^648 - X^324 + 1)`, the
//! 1944-th cyclotomic ring (1944 = 2^3 * 3^5), for binary witnesses held as elements of
//! `F162 = GF(2)[x]/(x^162 + x^81 + 1)`.
//!
//! The witness is committed modulo the base modulus 3889 and any of
//! `2917, 4861, 9721, 12637, 17497, 19441`,
//! folded against short binary challenges of the subring `R_162 = Z_q[Z]/Phi_243(Z)`, and the
//! folded opening is checked against the multilinear extension of the same witness over `F162`.
//! Everything is one AVX-512 thread; see [`scheme`] for the round and `src/main.rs` for the
//! reference usage.
//!
//! Module map
//! - `scheme`  : the public surface, re-exported here.
//! - `params`  : all ring constants (roots, twiddles, Montgomery forms), computed at compile time.
//! - `types`   : `RingElement` (`RingElement648` in the surface), `Batch32`.
//! - `scalar`  : exact reference implementation (schoolbook product mod Phi_1944, NTT).
//! - `rng`     : tiny deterministic RNG (no external crates).
//! - `simd`    : the AVX-512 kernels.
//! - `api`     : the commitment key, the commitment and its height-4 view over `R_162`.
//! - `challenge`: short (fixed-weight binary) challenges over `R_162` and the blake3 transcript.
//! - `fields`  : the binary fields `B128` and `F162`, their AVX-512 kernels and the
//!   cross-field switch, inlined from `bin-fields`.
//! - `fold`    : the folding step `v = sum_j c_j W_j` in the NTT domain, on top of a commitment.
//! - `eval`    : the binary shadow of the fold over `F162`.
//! - `wire`    : the serialisation of the clear-text round: bit-packing for the uniform objects,
//!   a static rANS for the folded witness.
#![allow(clippy::needless_range_loop)]

pub mod api;
pub mod challenge;
pub mod eval;
pub mod f162;
pub mod fields;
pub mod fold;
pub mod keccak;
pub mod labrador;
pub mod params;
pub mod recursion;
pub mod rng;
pub mod scalar;
pub mod scheme;
pub mod simd;
pub mod types;
pub mod wire;

pub use api::Modulus;
pub use api::PowerOfThreeRingElement as RingElement162;
pub use fields::scalar::F162;
pub use challenge::Transcript;
pub use scheme::{
    Commitment, CommitmentOpening, CommitmentValue, EvaluationPoint, FoldedCommitment,
    FoldedWitness, FoldingChallenges, FoldingSource, LeftExpansionCommitment, OpeningError,
    OpeningProof, OpeningTimings, ParamError, Params, Prover, PublicParameters, RowEvaluation,
    VerificationError, Verifier, VerifyTimings, Witness, WitnessError,
};
pub use types::RingElement as RingElement648;

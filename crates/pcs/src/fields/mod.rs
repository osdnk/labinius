//! The binary fields the scheme runs over, inlined from the `bin-fields` library (rev `84ee9ce`).
//!
//! [`scalar`] holds the two scalar types: `B128 = GF(2^128)` in the GHASH basis, binius64's field,
//! and `F162 = GF(2)[X]/(X^162 + X^81 + 1)`, this crate's, three `u64` limbs with the top one 34
//! bits wide. [`f162`] is the AVX-512 kernel over `F162` in the word-sliced layout — 8 elements
//! per `__m512i` triple, `clmul` products accumulated unreduced and reduced once at the end — and
//! [`sumcheck`] the SoA polynomial and the round it drives. [`crossfield`] is the switch that
//! turns a `B128` evaluation claim on a packed trace into an `F162` one on the same trace lifted
//! by the `beta` basis.

pub mod bitmat;
pub mod crossfield;
pub mod f162;
pub mod scalar;
pub mod sumcheck;

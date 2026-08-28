//! The `bin_fields::scalar::F162` input front end: how a stream of F162 elements is read as ring
//! elements of R_q = Z_q[X]/(X^648 - X^324 + 1), plus the scalar reference for the SIMD slicer.
//!
//! Four consecutive elements a, b, c, d (indices 4r..4r+3 of the stream) form ring element r by
//! plain interleaving:
//!
//! ```text
//!     coefficient of X^{4m+k} of ring element r  =  bit m of element 4r + k,   m in 0..162,
//! ```
//!
//! i.e. coefficient `c` of ring element r is bit `c / 4` of the F162 element `4r + (c mod 4)`.
//! All coefficients are 0/1, so the binary kernel applies verbatim; the only new work is the
//! bit-slicing front end ([`crate::simd::transpose_f162`]).
use crate::params::N;
use crate::rng::Rng;
use crate::simd::transpose_f162::BinaryIndex32;
use bin_fields::scalar::F162;

/// Number of significant bits of an `F162` (limbs 0 and 1 full, limb 2 holds bits 128..161).
pub const BITS: usize = 162;

/// The byte stride of `F162` inside a slice. Asserted by [`assert_layout`].
pub const STRIDE: usize = 24;

/// Bit m of an `F162` (m < 162), as 0 or 1.
#[inline(always)]
pub fn bit(x: &F162, m: usize) -> u32 {
    ((x.0[m >> 6] >> (m & 63)) & 1) as u32
}

/// The lift semantics: coefficient 4m + k of the ring element is bit m of `q[k]`.
pub fn lift4(q: &[F162; 4]) -> [u32; N] {
    let mut c = [0u32; N];
    for m in 0..BITS {
        for k in 0..4 {
            c[4 * m + k] = bit(&q[k], m);
        }
    }
    c
}

/// Ring element `r` of a stream (elements 4r..4r+3).
#[inline]
pub fn lift_elem(elems: &[F162], r: usize) -> [u32; N] {
    let q: &[F162; 4] = elems[4 * r..4 * r + 4].try_into().unwrap();
    lift4(q)
}

/// The inverse of [`lift4`] for testing: pack 648 binary coefficients back into four `F162`.
pub fn pack4(c: &[u32; N]) -> [F162; 4] {
    let mut q = [F162([0; 3]); 4];
    for m in 0..BITS {
        for k in 0..4 {
            debug_assert!(c[4 * m + k] < 2);
            q[k].0[m >> 6] |= (c[4 * m + k] as u64) << (m & 63);
        }
    }
    q
}

/// `F162::random(&mut rng)` — a uniform 162-bit element (the top 30 bits of limb 2 are zero).
pub trait RandomF162: Sized {
    fn random(rng: &mut Rng) -> Self;
}

impl RandomF162 for F162 {
    fn random(rng: &mut Rng) -> Self {
        F162([rng.next_u64(), rng.next_u64(), rng.next_u64() & ((1u64 << (BITS - 128)) - 1)])
    }
}

pub fn random_elems(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

/// Panics unless `F162` really is 24 contiguous bytes (the SIMD slicer addresses a `&[F162]` by
/// raw stride).
pub fn assert_layout() {
    assert_eq!(core::mem::size_of::<F162>(), STRIDE, "F162 is not 24 bytes");
    assert_eq!(core::mem::align_of::<F162>(), 8);
}

/// Scalar reference for [`crate::simd::transpose_f162::slice_f162`]: the `vpermb` index rows of
/// 32 ring elements (128 `F162`). `rows[i][2p] = n`, `rows[i][2p+1] = n + 16`, where n is the
/// nibble (c_i, c_{i+162}, c_{i+324}, c_{i+486}) of ring element p.
pub fn index_rows_scalar(elems: &[F162; 128]) -> BinaryIndex32 {
    let mut r = BinaryIndex32::zero();
    for p in 0..32 {
        let q: &[F162; 4] = elems[4 * p..4 * p + 4].try_into().unwrap();
        for i in 0..162 {
            let mut n = 0u8;
            for j in 0..4 {
                let c = i + 162 * j;
                n |= (bit(&q[c & 3], c >> 2) as u8) << j;
            }
            r.rows[i][2 * p] = n;
            r.rows[i][2 * p + 1] = n + 16;
        }
    }
    r
}

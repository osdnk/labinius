//! Drivers for the `F162` front end: `&[F162]` -> `Batch32` in the NTT domain, via
//! [`crate::simd::transpose_f162`] and the hand-written binary kernel
//! ([`crate::simd::vertical_bin_asm`], bit-identical to `vertical_bin` and ~6 % faster).
//!
//! `elems.len()` must be `128 * out.len()`: 128 `F162` = 32 ring elements = one `Batch32`.
use crate::simd::transpose::BinaryIndex32;
use crate::simd::transpose_f162::slice_f162_into;
use crate::simd::vertical_bin_asm as vb;
use crate::types::{Batch32, Representation};
use bin_fields::scalar::F162;
use core::arch::x86_64::_mm_sfence;

#[inline(always)]
unsafe fn chunk128(elems: &[F162], b: usize) -> &[F162; 128] {
    &*(elems.as_ptr().add(128 * b) as *const [F162; 128])
}

fn check(elems: &[F162], batches: usize) {
    assert_eq!(core::mem::size_of::<F162>(), 24, "F162 is not 24 bytes");
    assert_eq!(elems.len(), 128 * batches, "128 F162 per output batch");
}

/// Materialised: the last level of the kernel writes with non-temporal stores, which hides the
/// 340 MB output stream.
pub fn ntt_f162<const Q: u16>(elems: &[F162], out: &mut [Batch32]) {
    check(elems, out.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for (b, o) in out.iter_mut().enumerate() {
            slice_f162_into(chunk128(elems, b), &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, o);
        }
        _mm_sfence();
    }
}

/// Both primes from one slicing pass (the index rows are q-independent).
pub fn ntt_f162_2q<const QA: u16, const QB: u16>(
    elems: &[F162],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(out_a.len(), out_b.len());
    check(elems, out_a.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..out_a.len() {
            slice_f162_into(chunk128(elems, b), &mut idx);
            vb::ntt_bin_batch32_nt::<QA>(&idx, out_a.get_unchecked_mut(b));
            vb::ntt_bin_batch32_nt::<QB>(&idx, out_b.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// Each finished batch is handed to `f` instead of being materialised (regular stores: the
/// consumer reads the batch straight out of L1).
pub fn ntt_f162_stream<const Q: u16>(elems: &[F162], mut f: impl FnMut(usize, &Batch32)) {
    assert_eq!(elems.len() % 128, 0);
    let mut buf = Batch32::zero(Representation::Ntt);
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..elems.len() / 128 {
            slice_f162_into(chunk128(elems, b), &mut idx);
            vb::ntt_bin_batch32::<Q>(&idx, &mut buf);
            f(b, &buf);
        }
    }
}

/// Slicing only, for the component benchmark.
pub fn transpose_f162_all(elems: &[F162], idx: &mut [BinaryIndex32]) {
    check(elems, idx.len());
    unsafe {
        for (b, o) in idx.iter_mut().enumerate() {
            slice_f162_into(chunk128(elems, b), o);
        }
    }
}

/// One index-row buffer per batch is reused; a software prefetch of the next batch's 48 input
/// lines runs ahead of the slicer.
pub fn ntt_f162_pf<const Q: u16, const DIST: usize>(elems: &[F162], out: &mut [Batch32]) {
    check(elems, out.len());
    let nb = out.len();
    let mut idx = BinaryIndex32::zero();
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        for b in 0..nb {
            if b + DIST < nb {
                let p = elems.as_ptr().add(128 * (b + DIST)) as *const i8;
                let mut l = 0;
                while l < 3072 {
                    _mm_prefetch(p.add(l), _MM_HINT_T0);
                    l += 64;
                }
            }
            slice_f162_into(chunk128(elems, b), &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, out.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

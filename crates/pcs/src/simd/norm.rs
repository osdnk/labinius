use crate::params::N;
use core::arch::x86_64::*;

const BLOCK: usize = 32;

/// `(‖a‖^2, max |a_i|)`, or `None` when a lane leaves `i32`: the addends are squares, so a wrapped
/// lane is negative, and one add cannot wrap twice.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn normsq_and_max(a: &[i16; N]) -> Option<(u64, i32)> {
    let p = a.as_ptr();
    let mut acc = _mm512_setzero_si512();
    let mut peak = _mm512_setzero_si512();
    let mut wrapped = 0u16;

    let full = N / BLOCK;
    for i in 0..full {
        let x = _mm512_loadu_si512(p.add(i * BLOCK) as *const __m512i);
        peak = _mm512_max_epi16(peak, _mm512_abs_epi16(x));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x, x));
        wrapped |= _mm512_movepi32_mask(acc);
    }

    let rest = N - full * BLOCK;
    if rest != 0 {
        let mask = (1u32 << rest) - 1;
        let x = _mm512_maskz_loadu_epi16(mask, p.add(full * BLOCK));
        peak = _mm512_max_epi16(peak, _mm512_abs_epi16(x));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x, x));
        wrapped |= _mm512_movepi32_mask(acc);
    }

    if wrapped != 0 {
        return None;
    }

    let lo = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<0>(acc));
    let hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(acc));
    let plo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<0>(peak));
    let phi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(peak));
    Some((
        _mm512_reduce_add_epi64(_mm512_add_epi64(lo, hi)) as u64,
        _mm512_reduce_max_epi32(_mm512_max_epi32(plo, phi)),
    ))
}

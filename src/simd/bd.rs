use crate::api::barrett31;
use crate::params::N;
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn combine<const Q: u16>(
    top: &[u16],
    digits: &[&[u16]],
    shift: u32,
    primes: &[u16],
    at: usize,
    k: __mmask16,
) -> __m512i {
    let load = |s: &[u16]| {
        _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(
            k,
            s.as_ptr().add(at) as *const i16,
        ))
    };
    let last = digits.len() - 1;
    let mut inner = load(digits[last]);
    for m in (0..last).rev() {
        let w = _mm512_set1_epi32(primes[m + 1] as i32);
        inner = barrett31::<Q>(_mm512_add_epi32(
            load(digits[m]),
            _mm512_mullo_epi32(inner, w),
        ));
    }
    let t = _mm512_sllv_epi32(load(top), _mm512_set1_epi32(shift as i32));
    let r = barrett31::<Q>(_mm512_add_epi32(
        t,
        _mm512_mullo_epi32(inner, _mm512_set1_epi32(primes[0] as i32)),
    ));
    let hi = _mm512_cmpgt_epi32_mask(r, _mm512_set1_epi32((Q as i32 - 1) / 2));
    _mm512_mask_sub_epi32(r, hi, r, _mm512_set1_epi32(Q as i32))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn residues<const Q: u16>(
    top: &[u16],
    digits: &[&[u16]],
    shift: u32,
    primes: &[u16],
    out: &mut [i16],
) {
    let n = top.len();
    for b in (0..n).step_by(16) {
        let k: __mmask16 = if b + 16 <= n {
            !0
        } else {
            (1u16 << (n - b)) - 1
        };
        let r = combine::<Q>(top, digits, shift, primes, b, k);
        _mm512_mask_cvtepi32_storeu_epi16(out.as_mut_ptr().add(b) as *mut i16, k, r);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn residue_quad<const Q: u16>(
    top: &[u16],
    digits: &[&[u16]],
    shift: u32,
    primes: &[u16],
    at: [usize; 4],
    out: &mut [u64; N],
) {
    let mask = _mm512_set1_epi32(0xFFFF);
    let ilo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let ihi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    for b in (0..N).step_by(16) {
        let k: __mmask16 = if b + 16 <= N {
            !0
        } else {
            (1u16 << (N - b)) - 1
        };
        let r: [__m512i; 4] =
            core::array::from_fn(|c| combine::<Q>(top, digits, shift, primes, at[c] + b, k));
        let d01 = _mm512_or_si512(_mm512_and_si512(r[0], mask), _mm512_slli_epi32::<16>(r[1]));
        let d23 = _mm512_or_si512(_mm512_and_si512(r[2], mask), _mm512_slli_epi32::<16>(r[3]));
        let p = out.as_mut_ptr().add(b) as *mut i64;
        _mm512_mask_storeu_epi64(p, k as __mmask8, _mm512_permutex2var_epi32(d01, ilo, d23));
        _mm512_mask_storeu_epi64(
            p.add(8),
            (k >> 8) as __mmask8,
            _mm512_permutex2var_epi32(d01, ihi, d23),
        );
    }
}

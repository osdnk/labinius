use crate::params::N;
use crate::ring::Batch32;
use core::arch::x86_64::*;

pub const IDENTITY: [u16; N] = {
    let mut t = [0u16; N];
    let mut i = 0;
    while i < N {
        t[i] = i as u16;
        i += 1;
    }
    t
};

pub static ZERO_ROW: [i16; N] = [0i16; N];

#[inline(always)]
unsafe fn tr32(a: &mut [__m512i; 32]) {
    for k in 0..16 {
        let (x, y) = (a[2 * k], a[2 * k + 1]);
        a[2 * k] = _mm512_unpacklo_epi16(x, y);
        a[2 * k + 1] = _mm512_unpackhi_epi16(x, y);
    }
    for k in 0..8 {
        let b = 4 * k;
        let (x0, x1, x2, x3) = (a[b], a[b + 1], a[b + 2], a[b + 3]);
        a[b] = _mm512_unpacklo_epi32(x0, x2);
        a[b + 1] = _mm512_unpackhi_epi32(x0, x2);
        a[b + 2] = _mm512_unpacklo_epi32(x1, x3);
        a[b + 3] = _mm512_unpackhi_epi32(x1, x3);
    }
    for k in 0..4 {
        let b = 8 * k;
        let mut t = [_mm512_setzero_si512(); 8];
        for j in 0..4 {
            t[2 * j] = _mm512_unpacklo_epi64(a[b + j], a[b + j + 4]);
            t[2 * j + 1] = _mm512_unpackhi_epi64(a[b + j], a[b + j + 4]);
        }
        a[b..b + 8].copy_from_slice(&t);
    }
    let mut t = [_mm512_setzero_si512(); 32];
    for j in 0..8 {
        let (a0, a1, a2, a3) = (a[j], a[j + 8], a[j + 16], a[j + 24]);
        let u0 = _mm512_shuffle_i64x2::<0x88>(a0, a1);
        let u1 = _mm512_shuffle_i64x2::<0x88>(a2, a3);
        let u2 = _mm512_shuffle_i64x2::<0xDD>(a0, a1);
        let u3 = _mm512_shuffle_i64x2::<0xDD>(a2, a3);
        t[j] = _mm512_shuffle_i64x2::<0x88>(u0, u1);
        t[j + 8] = _mm512_shuffle_i64x2::<0x88>(u2, u3);
        t[j + 16] = _mm512_shuffle_i64x2::<0xDD>(u0, u1);
        t[j + 24] = _mm512_shuffle_i64x2::<0xDD>(u2, u3);
    }
    a.copy_from_slice(&t);
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn transpose_into(src: &[*const i16; 32], rows: &[u16], dst: &mut Batch32) {
    let n = rows.len();
    let mut off = 0;
    while off < n {
        let w = (n - off).min(32);
        let mut a = [_mm512_setzero_si512(); 32];
        if w == 32 {
            for i in 0..32 {
                a[i] = _mm512_loadu_si512(src[i].add(off) as *const __m512i);
            }
        } else {
            let m = ((1u32 << w) - 1) as __mmask32;
            for i in 0..32 {
                a[i] = _mm512_maskz_loadu_epi16(m, src[i].add(off));
            }
        }
        tr32(&mut a);
        for j in 0..w {
            _mm512_store_si512(
                dst.v[rows[off + j] as usize].as_mut_ptr() as *mut __m512i,
                a[j],
            );
        }
        off += 32;
    }
}

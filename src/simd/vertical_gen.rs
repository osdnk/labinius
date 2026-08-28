//! Generic-input forward NTT on the vertical `Batch32` layout (32 polynomials, one 512-bit vector
//! per coefficient index), in place, `Coefficients -> Ntt`, input lanes `|x| <= q`.
//!
//! # Structure
//!
//! The 7 levels of the tree (see `params`) run as five passes; after the first one everything is
//! depth-first inside a 162-block (10 KB) so the working set stays in L1.
//!
//! | pass | levels | group of vectors                                    | l/s | butterflies |
//! |------|--------|-----------------------------------------------------|-----|-------------|
//! | A    | 0 + 1  | `i, i+162, i+324, i+486` (i < 162)                  | 4   | 4 x radix-2 |
//! | B    | 2 + 3  | `j+27a`, `81+j+27a` (j < 27, a < 3), per 162-block  | 6   | 3 x radix-2, 2 x radix-3 |
//! | C4   | 4      | `i, i+9, i+18` (i < 9), per 27-block                | 3   | 1 x radix-3 |
//! | C5   | 5      | `i, i+3, i+6`, per 9-block                          | 3   | 1 x radix-3 |
//! | D    | 6      | `3g, 3g+1, 3g+2`, per 3-block                       | 3   | 1 x radix-3 |
//!
//! Pass A fuses levels 0 and 1 into one radix-4 pass (level 0 is the Phi_6 split
//! `t = zeta6*a1; y0 = a0 + t; y1 = a0 + a1 - t`, level 1 the two radix-2 halves): 162 iterations
//! of 4 loads, 4 Montgomery multiplications, 4 stores.
//!
//! Deeper fusion was implemented and measured (`ntt_gen_batch32_plan`, `ntt_gen_batch32_r27`, and
//! the bench): fusing levels 4+5 into a radix-9 pass over 9 registers costs 3.5% (438 vs 423
//! cycles/poly at q = 3889), 5+6 costs 3.8%, and the full radix-27 tail costs 6.3%. They do cut L1
//! traffic, but the kernel is port-0-throughput-bound, not load/store-bound, and the longer
//! dependency chains inside a fused group only reduce the number of independent butterflies in
//! flight. Splitting pass B into separate level-2 and level-3 passes also loses (427 vs 423), so
//! B stays fused. The streaming driver `ntt_gen_batches` prefetches the next batch into L2 one
//! cache line per butterfly of levels 4-6, which is worth 23% on the 2^18-polynomial case.
//!
//! # Arithmetic and bounds
//!
//! Twiddle multiplication is the 3-uop signed Montgomery form `mullo(a,w') / mulhi(a,w) /
//! mulhi(m,q) / sub` with `|mont(a,w)| <= |a| q/2^17 + q/2`; twiddle constants live in memory as
//! duplicated `u32` so the broadcast is a pure load (`vpbroadcastd`).  With `c = q/2^17` and all
//! bounds written as multiples of q (input `|x| <= q`), one radix-3 level maps a bound A to
//! `A + 2(1/2 + cA)` on the untwiddled path, so the a0 input needs an occasional `barrett`
//! (`|barrett(x)| <= 0.899q` for 3889, `0.809q` for 9721).  Placement (`BAR_A`, `BAR_L`) and the
//! resulting per-level bounds:
//!
//! ```text
//!                       q = 3889 (budget 8.4276 q)      q = 9721 (budget 3.3708 q)
//!   level 0  (in pass A)   2.5297                          2.5742  (barrett on the a0 half)
//!   level 1  (in pass A)   3.1047                          2.1909
//!   level 2                3.6969                          2.8535
//!   level 3                4.9163                          2.2323  (barrett a0)
//!   level 4                6.2080                          2.1401  (barrett a0)
//!   level 5                2.2676  (barrett a0)            2.1265  (barrett a0)
//!   level 6                3.4022  (output)                2.1244  (output, barrett a0)
//! ```
//!
//! Output bound: `|v| <= 3.4022 q = 13231` for q = 3889 and `|v| <= 2.1244 q = 20652` for q = 9721
//! (`OUTPUT_BOUND`), verified against an exact i32 shadow model in `tests/vertical_gen.rs`.
use crate::params::{barrett_v, Params};
use crate::types::{Batch32, Representation};
use core::arch::x86_64::*;

/// A 16-bit constant duplicated into a u32 so that `vpbroadcastd m32` fills a whole zmm with it.
const fn dup(w: i16) -> u32 {
    let u = w as u16 as u32;
    u | (u << 16)
}

/// Compile-time twiddle tables, all in the duplicated-u32 broadcast form.
///
/// Radix-2 levels store `[zeta', zeta]` per sub-ring, radix-3 levels store
/// `[zeta', zeta, (zeta^2)', zeta^2]` (primed = the `mont_pre` companion).
pub struct Tw<const Q: u16>;

impl<const Q: u16> Tw<Q> {
    const fn pair(x: u16) -> [u32; 2] {
        let w = Params::<Q>::to_mont(x);
        [dup(Params::<Q>::mont_pre(w)), dup(w)]
    }
    const fn r2<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let p = Self::pair(Params::<Q>::zeta(level, k));
            t[2 * k] = p[0];
            t[2 * k + 1] = p[1];
            k += 1;
        }
        t
    }
    const fn r3<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let z = Params::<Q>::zeta(level, k);
            let z2 = (z as u64 * z as u64 % Q as u64) as u16;
            let a = Self::pair(z);
            let b = Self::pair(z2);
            t[4 * k] = a[0];
            t[4 * k + 1] = a[1];
            t[4 * k + 2] = b[0];
            t[4 * k + 3] = b[1];
            k += 1;
        }
        t
    }
    pub const Z6: [u32; 2] = Self::pair(Params::<Q>::ZETA6);
    pub const OM: [u32; 2] = Self::pair(Params::<Q>::OMEGA);
    pub const QD: u32 = dup(Q as i16);
    pub const BV: u32 = dup(barrett_v(Q));
    pub const L1: [u32; 4] = Self::r2::<4>(1, 2);
    pub const L2: [u32; 8] = Self::r2::<8>(2, 4);
    pub const L3: [u32; 32] = Self::r3::<32>(3, 8);
    pub const L4: [u32; 96] = Self::r3::<96>(4, 24);
    pub const L5: [u32; 288] = Self::r3::<288>(5, 72);
    pub const L6: [u32; 864] = Self::r3::<864>(6, 216);

    /// Barrett the level-0 output `a0 + a1 - zeta6*a1` inside pass A (only 9721 needs it).
    pub const BAR_A: bool = Q == 9721;
    /// `BAR_L[l]` = barrett the untwiddled a0 input of level l (l = 2..6).
    pub const BAR_L: [bool; 7] = if Q == 9721 {
        [false, false, false, true, true, true, true]
    } else {
        [false, false, false, false, false, true, false]
    };
    /// Proven output bound, as `ceil(bound * q)`; see the module comment.
    pub const OUTPUT_BOUND: i32 = if Q == 9721 { 20652 } else { 13231 };
}

#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    _mm512_set1_epi32(p.read() as i32)
}

#[inline(always)]
unsafe fn ld(p: *const __m512i, j: usize) -> __m512i {
    _mm512_load_si512(p.add(j))
}

#[inline(always)]
unsafe fn st(p: *mut __m512i, j: usize, x: __m512i) {
    _mm512_store_si512(p.add(j), x);
}

/// Signed Montgomery twiddle product: 3 multiply uops, `|r| <= |a| q/2^17 + q/2 < q`.
#[inline(always)]
unsafe fn mont(a: __m512i, wp: __m512i, w: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = _mm512_mulhi_epi16(a, w);
    _mm512_sub_epi16(hi, _mm512_mulhi_epi16(m, q))
}

/// `vpmulhrsw`-based partial reduction: 2 multiply uops, `|r| <= 0.899q / 0.809q`.
#[inline(always)]
unsafe fn barrett(a: __m512i, v: __m512i, q: __m512i) -> __m512i {
    _mm512_sub_epi16(a, _mm512_mullo_epi16(_mm512_mulhrs_epi16(a, v), q))
}

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(a, b)
}

#[inline(always)]
unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
    _mm512_sub_epi16(a, b)
}

/// Radix-3 butterfly `(a0, a1, a2) -> (a0+t1+t2, a0-t2+u, a0-t1-u)`, `t1 = zeta a1`,
/// `t2 = zeta^2 a2`, `u = omega (t1-t2)`: 9 multiply uops + 7 adds.
#[inline(always)]
unsafe fn r3(
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
    omp: __m512i,
    om: __m512i,
    q: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let t1 = mont(a1, bc(tw), bc(tw.add(1)), q);
    let t2 = mont(a2, bc(tw.add(2)), bc(tw.add(3)), q);
    let u = mont(sub(t1, t2), omp, om, q);
    (add(a0, add(t1, t2)), add(sub(a0, t2), u), sub(sub(a0, t1), u))
}

/// Levels 0 and 1 fused into one radix-4 pass over the 648 vectors.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_a<const Q: u16>(p: *mut __m512i) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let z6p = bc(Tw::<Q>::Z6.as_ptr());
    let z6 = bc(Tw::<Q>::Z6.as_ptr().add(1));
    let l1 = Tw::<Q>::L1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    for i in 0..162 {
        let a0 = ld(p, i);
        let a1 = ld(p, i + 324);
        let b0 = ld(p, i + 162);
        let b1 = ld(p, i + 486);
        let t = mont(a1, z6p, z6, q);
        let s = mont(b1, z6p, z6, q);
        let c0 = add(a0, t);
        let c1 = add(b0, s);
        let mut c2 = sub(add(a0, a1), t);
        let c3 = sub(add(b0, b1), s);
        if Tw::<Q>::BAR_A {
            c2 = barrett(c2, bv, q);
        }
        let u = mont(c1, zap, za, q);
        let w = mont(c3, zbp, zb, q);
        st(p, i, add(c0, u));
        st(p, i + 162, sub(c0, u));
        st(p, i + 324, add(c2, w));
        st(p, i + 486, sub(c2, w));
    }
}

/// Levels 2 and 3 for one 162-block: 27 groups of 6 vectors, 3 radix-2 + 2 radix-3 each.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_b<const Q: u16>(p: *mut __m512i, blk: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let l2 = Tw::<Q>::L2.as_ptr().add(2 * blk);
    let (zp, z) = (bc(l2), bc(l2.add(1)));
    let ta = Tw::<Q>::L3.as_ptr().add(8 * blk);
    let tb = ta.add(4);
    let base = blk * 162;
    for j in 0..27 {
        let b = base + j;
        let x0 = ld(p, b);
        let x1 = ld(p, b + 27);
        let x2 = ld(p, b + 54);
        let t0 = mont(ld(p, b + 81), zp, z, q);
        let t1 = mont(ld(p, b + 108), zp, z, q);
        let t2 = mont(ld(p, b + 135), zp, z, q);
        let mut n0 = add(x0, t0);
        let n1 = add(x1, t1);
        let n2 = add(x2, t2);
        let mut m0 = sub(x0, t0);
        let m1 = sub(x1, t1);
        let m2 = sub(x2, t2);
        if Tw::<Q>::BAR_L[3] {
            n0 = barrett(n0, bv, q);
            m0 = barrett(m0, bv, q);
        }
        let (y0, y1, y2) = r3(n0, n1, n2, ta, omp, om, q);
        let (w0, w1, w2) = r3(m0, m1, m2, tb, omp, om, q);
        st(p, b, y0);
        st(p, b + 27, y1);
        st(p, b + 54, y2);
        st(p, b + 81, w0);
        st(p, b + 108, w1);
        st(p, b + 135, w2);
    }
}

/// Levels 4 and 5 for one 27-block (level-4 sub-ring `k4`): 3 groups of 9 vectors, 6 radix-3 each
/// (3 "columns" for level 4, then 3 "rows" for level 5).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_c<const Q: u16>(p: *mut __m512i, k4: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t4 = Tw::<Q>::L4.as_ptr().add(4 * k4);
    let t5 = Tw::<Q>::L5.as_ptr().add(12 * k4);
    let base = k4 * 27;
    for j in 0..3 {
        let b = base + j;
        let mut v = [_mm512_setzero_si512(); 9];
        for a in 0..3 {
            let (mut c0, c1, c2) = (ld(p, b + 3 * a), ld(p, b + 3 * a + 9), ld(p, b + 3 * a + 18));
            if Tw::<Q>::BAR_L[4] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, c1, c2, t4, omp, om, q);
            v[a] = y0;
            v[3 + a] = y1;
            v[6 + a] = y2;
        }
        for bb in 0..3 {
            let mut c0 = v[3 * bb];
            if Tw::<Q>::BAR_L[5] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, v[3 * bb + 1], v[3 * bb + 2], t5.add(4 * bb), omp, om, q);
            st(p, b + 9 * bb, y0);
            st(p, b + 9 * bb + 3, y1);
            st(p, b + 9 * bb + 6, y2);
        }
    }
}

/// Level 6 for one 27-block: 3 groups of 9 vectors, 3 radix-3 butterflies each (m = 1, so the
/// twiddles change per butterfly).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_d<const Q: u16, const PF: bool>(p: *mut __m512i, k4: usize, pf: *const i8) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t6 = Tw::<Q>::L6.as_ptr().add(36 * k4);
    let base = k4 * 27;
    for g in 0..9 {
        if PF {
            _mm_prefetch(pf.add((base + 18 + g) * 64), _MM_HINT_T1);
        }
        let b = base + 3 * g;
        let mut c0 = ld(p, b);
        if Tw::<Q>::BAR_L[6] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, ld(p, b + 1), ld(p, b + 2), t6.add(4 * g), omp, om, q);
        st(p, b, y0);
        st(p, b + 1, y1);
        st(p, b + 2, y2);
    }
}

/// Level 4 alone for one 27-block: 3 groups of 3 independent butterflies (unfused variant).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_c4<const Q: u16, const PF: bool>(p: *mut __m512i, k4: usize, pf: *const i8) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t4 = Tw::<Q>::L4.as_ptr().add(4 * k4);
    let base = k4 * 27;
    for c in 0..3 {
        for i in 3 * c..3 * c + 3 {
            if PF {
                _mm_prefetch(pf.add((base + i) * 64), _MM_HINT_T1);
            }
            let b = base + i;
            let mut c0 = ld(p, b);
            if Tw::<Q>::BAR_L[4] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, ld(p, b + 9), ld(p, b + 18), t4, omp, om, q);
            st(p, b, y0);
            st(p, b + 9, y1);
            st(p, b + 18, y2);
        }
    }
}

/// Level 5 alone for one 27-block: 3 groups of 9 vectors, 3 butterflies each (unfused variant).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_c5<const Q: u16, const PF: bool>(p: *mut __m512i, k4: usize, pf: *const i8) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t5 = Tw::<Q>::L5.as_ptr().add(12 * k4);
    let base = k4 * 27;
    for bb in 0..3 {
        for j in 0..3 {
            if PF {
                _mm_prefetch(pf.add((base + 9 + 3 * bb + j) * 64), _MM_HINT_T1);
            }
            let b = base + 9 * bb + j;
            let mut c0 = ld(p, b);
            if Tw::<Q>::BAR_L[5] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, ld(p, b + 3), ld(p, b + 6), t5.add(4 * bb), omp, om, q);
            st(p, b, y0);
            st(p, b + 3, y1);
            st(p, b + 6, y2);
        }
    }
}

/// Level 2 alone for one 162-block (unfused variant): 3 independent butterflies per group.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_l2<const Q: u16>(p: *mut __m512i, blk: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let l2 = Tw::<Q>::L2.as_ptr().add(2 * blk);
    let (zp, z) = (bc(l2), bc(l2.add(1)));
    let base = blk * 162;
    for c in 0..27 {
        for i in 3 * c..3 * c + 3 {
            let b = base + i;
            let x = ld(p, b);
            let t = mont(ld(p, b + 81), zp, z, q);
            st(p, b, add(x, t));
            st(p, b + 81, sub(x, t));
        }
    }
}

/// Level 3 alone for one 162-block (unfused variant).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_l3<const Q: u16>(p: *mut __m512i, blk: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    for h in 0..2 {
        let tw = Tw::<Q>::L3.as_ptr().add(8 * blk + 4 * h);
        let base = blk * 162 + 81 * h;
        for c in 0..9 {
            for j in 3 * c..3 * c + 3 {
                let b = base + j;
                let mut c0 = ld(p, b);
                if Tw::<Q>::BAR_L[3] {
                    c0 = barrett(c0, bv, q);
                }
                let (y0, y1, y2) = r3(c0, ld(p, b + 27), ld(p, b + 54), tw, omp, om, q);
                st(p, b, y0);
                st(p, b + 27, y1);
                st(p, b + 54, y2);
            }
        }
    }
}

/// Levels 5 and 6 fused over one 9-block (level-5 sub-ring `k5`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_l56<const Q: u16>(p: *mut __m512i, k5: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t5 = Tw::<Q>::L5.as_ptr().add(4 * k5);
    let t6 = Tw::<Q>::L6.as_ptr().add(12 * k5);
    let base = 9 * k5;
    let mut v = [_mm512_setzero_si512(); 9];
    for j in 0..3 {
        let mut c0 = ld(p, base + j);
        if Tw::<Q>::BAR_L[5] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, ld(p, base + j + 3), ld(p, base + j + 6), t5, omp, om, q);
        v[j] = y0;
        v[3 + j] = y1;
        v[6 + j] = y2;
    }
    for g in 0..3 {
        let mut c0 = v[3 * g];
        if Tw::<Q>::BAR_L[6] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, v[3 * g + 1], v[3 * g + 2], t6.add(4 * g), omp, om, q);
        st(p, base + 3 * g, y0);
        st(p, base + 3 * g + 1, y1);
        st(p, base + 3 * g + 2, y2);
    }
}

/// Levels 4, 5 and 6 in one radix-27 pass (27 vectors live).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_l456<const Q: u16>(p: *mut __m512i, k4: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t4 = Tw::<Q>::L4.as_ptr().add(4 * k4);
    let t5 = Tw::<Q>::L5.as_ptr().add(12 * k4);
    let t6 = Tw::<Q>::L6.as_ptr().add(36 * k4);
    let base = k4 * 27;
    let mut v = [_mm512_setzero_si512(); 27];
    for i in 0..9 {
        let mut c0 = ld(p, base + i);
        if Tw::<Q>::BAR_L[4] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, ld(p, base + i + 9), ld(p, base + i + 18), t4, omp, om, q);
        v[i] = y0;
        v[9 + i] = y1;
        v[18 + i] = y2;
    }
    for bb in 0..3 {
        for j in 0..3 {
            let mut c0 = v[9 * bb + j];
            if Tw::<Q>::BAR_L[5] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) =
                r3(c0, v[9 * bb + j + 3], v[9 * bb + j + 6], t5.add(4 * bb), omp, om, q);
            v[9 * bb + j] = y0;
            v[9 * bb + j + 3] = y1;
            v[9 * bb + j + 6] = y2;
        }
    }
    for g in 0..9 {
        let mut c0 = v[3 * g];
        if Tw::<Q>::BAR_L[6] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, v[3 * g + 1], v[3 * g + 2], t6.add(4 * g), omp, om, q);
        st(p, base + 3 * g, y0);
        st(p, base + 3 * g + 1, y1);
        st(p, base + 3 * g + 2, y2);
    }
}

/// Structural variants, measured by `bench_vertical_gen`; `PLAN` selects one.
///
/// # Safety
/// See `ntt_gen_batch32`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn ntt_gen_batch32_plan<const Q: u16, const PLAN: u32>(b: &mut Batch32) {
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        if PLAN >= 2 {
            pass_l2::<Q>(p, blk);
            pass_l3::<Q>(p, blk);
        } else {
            pass_b::<Q>(p, blk);
        }
        for k4 in 6 * blk..6 * blk + 6 {
            match PLAN % 2 {
                0 => {
                    pass_c::<Q>(p, k4);
                    pass_d::<Q, false>(p, k4, core::ptr::null());
                }
                _ => {
                    pass_c4::<Q, false>(p, k4, core::ptr::null());
                    if PLAN >= 4 {
                        for k5 in 3 * k4..3 * k4 + 3 {
                            pass_l56::<Q>(p, k5);
                        }
                    } else {
                        pass_c5::<Q, false>(p, k4, core::ptr::null());
                        pass_d::<Q, false>(p, k4, core::ptr::null());
                    }
                }
            }
        }
    }
    b.representation = Representation::Ntt;
}

/// Variant with the whole tail (levels 4-6) as one radix-27 pass.
///
/// # Safety
/// See `ntt_gen_batch32`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn ntt_gen_batch32_r27<const Q: u16>(b: &mut Batch32) {
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        pass_b::<Q>(p, blk);
        for k4 in 6 * blk..6 * blk + 6 {
            pass_l456::<Q>(p, k4);
        }
    }
    b.representation = Representation::Ntt;
}

/// Forward NTT of a batch of 32 polynomials in place: `Coefficients -> Ntt` (tree order).
///
/// Requires `|b.v[j][p]| <= q`. Output satisfies `|b.v[j][p]| <= Tw::<Q>::OUTPUT_BOUND`.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn ntt_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Coefficients);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        pass_b::<Q>(p, blk);
        for k4 in 6 * blk..6 * blk + 6 {
            pass_c4::<Q, false>(p, k4, core::ptr::null());
            pass_c5::<Q, false>(p, k4, core::ptr::null());
            pass_d::<Q, false>(p, k4, core::ptr::null());
        }
    }
    b.representation = Representation::Ntt;
}

/// Same kernel with the next batch prefetched into L2, one cache line per butterfly of levels
/// 4-6 (27 per 27-block, 648 per batch). Bursting the same prefetches (162 at the head of each
/// 162-block) is 18% *slower* than not prefetching at all: they overrun the fill buffers.
///
/// # Safety
/// See `ntt_gen_batch32`; `next` must be a valid `Batch32` or null.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn ntt_gen_batch32_pf<const Q: u16>(b: &mut Batch32, next: *const i8) {
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        pass_b::<Q>(p, blk);
        for k4 in 6 * blk..6 * blk + 6 {
            if next.is_null() {
                pass_c4::<Q, false>(p, k4, next);
                pass_c5::<Q, false>(p, k4, next);
                pass_d::<Q, false>(p, k4, next);
            } else {
                pass_c4::<Q, true>(p, k4, next);
                pass_c5::<Q, true>(p, k4, next);
                pass_d::<Q, true>(p, k4, next);
            }
        }
    }
    b.representation = Representation::Ntt;
}

/// Driver: forward NTT of many batches, in place, prefetching one batch ahead (worth 23% on a
/// 340 MB working set: 527 vs 693 cycles/poly at q = 3889).
pub fn ntt_gen_batches<const Q: u16>(bs: &mut [Batch32]) {
    let n = bs.len();
    let base = bs.as_mut_ptr();
    for i in 0..n {
        unsafe {
            let next = if i + 1 < n { base.add(i + 1) as *const i8 } else { core::ptr::null() };
            ntt_gen_batch32_pf::<Q>(&mut *base.add(i), next);
        }
    }
}

//! Generic-input NTT on the *quadratic-slot* tree (q in `params::QS_QUAD`), vertical `Batch32`
//! layout, in place: `Coefficients -> Ntt`, input lanes `|x| <= q`.
//!
//! The counterpart of [`crate::simd::ntt::gen_small`] for the tree that does not split completely
//! (see the quadratic-slot section of `params`); it is what a challenge transform and a folded
//! witness over these limbs need. Three passes, depth-first inside a 162-block after the first so
//! the working set stays in L1:
//!
//! | pass | levels | group of vectors                                        | butterflies         |
//! |------|--------|---------------------------------------------------------|---------------------|
//! | A    | 0 + 1  | `i, i+162, i+324, i+486` (i < 162)                       | 4 x radix-2         |
//! | B    | 2 + 3  | `54s + i0 + 18a` (s, a < 3), per 162-block              | 3 + 3 x radix-3     |
//! | C    | 4 + 5  | one 18-block, two passes                                | 6 + 6 x radix-3     |
//!
//! Pass A fuses the Phi_6 split with the one radix-2 level into a radix-4 pass, exactly as
//! `ntt::gen_small` does. Pass B fuses levels 2 and 3 over groups of 9: the three level-2 triples
//! `(i, i+54, i+108)` at `i = i0, i0+18, i0+36` deliver exactly one level-3 butterfly to each of
//! the three 54-blocks.
//!
//! Levels 4 and 5 are *not* fused, although an 18-block does fit in registers: keeping the 18
//! values live across both levels costs 14 % (436 against 382 cycles per ring element at q = 2917, 521
//! against 481 at 12637) — the register allocator spends the saved loads and stores on moves,
//! and this kernel has no port-0 slack to hide them in. Splitting pass B the same way gains
//! nothing (379 against 382, at 80 more instructions), so B stays fused.
//!
//! **Multiply count.** 648 + 4 x 216 x 3 = 3240 Montgomery products per batch of 32, against the
//! splitting kernel's 3564: the radix-2 level that disappears took 324 with it.
//!
//! ## Bounds and reduction schedule
//!
//! `|mont(a, w)| <= |a| q/2^17 + q/2` and the reduction is the shuffle-port **lookup Barrett** of
//! [`crate::simd::ntt::bin_asm::barrett_lut_i16`] (`|r| <= q/2 + 2^10`, 2 port-5 + 3 flexible
//! uops and *no* multiply-port slot, which is what this kernel is short of). [`gen_flags`] is a
//! `const` search over the flag set — reduce the loaded `a1`, `b1` of pass A, reduce its
//! level-0 output `a0 + a1 - t`, reduce the untwiddled `a0` input of each of levels 2..5 — for
//! the cheapest placement that keeps every intermediate inside i16, and [`gen_model`] replays it:
//!
//! ```text
//!                   q = 2917 (budget 11.23 q)   q = 4861 (6.74 q)   q = 12637 (2.59 q)
//!   reductions      none                        level 4             pass-A inputs and
//!                                                                   output, levels 2..5
//!   after levels 0+1   3.078                       3.131               2.205
//!   after level 2      4.215                       4.364               1.994
//!   after level 3      5.402                       5.687               1.954
//!   after level 4      6.643                       2.107               1.946
//!   after level 5      7.939  (output)             3.263  (output)     1.944  (output)
//! ```
//!
//! For 12637 the pass-A reduction is not optional in any placement: `a0 + a1 - t` with `|x| <= q`
//! reaches 2.60 q = 32811 and leaves i16 before anything can be done about it, so the two loaded
//! `a1` values are reduced as they arrive (which also tightens everything downstream).
use crate::params::*;
use crate::simd::ntt::bar_switch;
use crate::simd::ntt::bin_quad::{barrett_lut_max, lut_byte};
use crate::ring::{Batch32, Representation};
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// bounds
// ---------------------------------------------------------------------------------------------

const fn mont_bound(b: i32, q: u16) -> i32 {
    ((b as i64 * q as i64) >> 17) as i32 + (q as i32 + 1) / 2
}

/// The kernel's schedule replayed on bounds: `[after levels 0+1, after level 2, .., after level
/// 5]` and the largest intermediate ever formed. `bi` reduces the loaded `a1`, `b1` of pass A,
/// `ba` its `a0 + a1 - t` output, `bl[l]` the untwiddled `a0` input of level `2 + l`.
pub const fn gen_model(q: u16, bi: bool, ba: bool, bl: [bool; 4]) -> ([i32; 5], i32) {
    let r = barrett_lut_max(q);
    let mut v = [q as i32; N];
    let mut lm = [0i32; 5];
    let mut peak = 0i32;
    let mut i = 0;
    while i < 162 {
        let a0 = v[i];
        let b0 = v[i + 162];
        let a1 = if bi { r } else { v[i + 324] };
        let b1 = if bi { r } else { v[i + 486] };
        let t = mont_bound(a1, q);
        let s = mont_bound(b1, q);
        let c0 = a0 + t;
        let c1 = b0 + s;
        let mut c2 = a0 + a1 + t;
        let mut c3 = b0 + b1 + s;
        if c0 > peak {
            peak = c0;
        }
        if c2 > peak {
            peak = c2;
        }
        if c1 > peak {
            peak = c1;
        }
        if c3 > peak {
            peak = c3;
        }
        if ba {
            c2 = r;
            c3 = r;
        }
        let u = mont_bound(c1, q);
        let w = mont_bound(c3, q);
        if c0 + u > peak {
            peak = c0 + u;
        }
        if c2 + w > peak {
            peak = c2 + w;
        }
        v[i] = c0 + u;
        v[i + 162] = c0 + u;
        v[i + 324] = c2 + w;
        v[i + 486] = c2 + w;
        i += 1;
    }
    let mut i = 0;
    while i < N {
        if v[i] > lm[0] {
            lm[0] = v[i];
        }
        i += 1;
    }
    let mut l = 0;
    while l < 4 {
        let blk = [162usize, 54, 18, 6][l];
        let m = blk / 3;
        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < m {
                let (i0, i1, i2) = (base + i, base + i + m, base + i + 2 * m);
                let b0 = if bl[l] { r } else { v[i0] };
                let t1 = mont_bound(v[i1], q);
                let t2 = mont_bound(v[i2], q);
                let uu = mont_bound(t1 + t2, q);
                if t1 + t2 > peak {
                    peak = t1 + t2;
                }
                if b0 + t1 + t2 > peak {
                    peak = b0 + t1 + t2;
                }
                if b0 + t2 + uu > peak {
                    peak = b0 + t2 + uu;
                }
                if b0 + t1 + uu > peak {
                    peak = b0 + t1 + uu;
                }
                v[i0] = b0 + t1 + t2;
                v[i1] = b0 + t2 + uu;
                v[i2] = b0 + t1 + uu;
                i += 1;
            }
            base += blk;
        }
        let mut i = 0;
        while i < N {
            if v[i] > lm[l + 1] {
                lm[l + 1] = v[i];
            }
            i += 1;
        }
        l += 1;
    }
    (lm, peak)
}

/// The cheapest reduction placement that keeps every intermediate inside i16, by exhaustive
/// `const` search over the 64 flag combinations (cost = reductions per batch: 324 for each of the
/// pass-A flags, 216 per radix-3 level). Returns `(bar_in, bar_a, bar_level[2..=5])`.
const fn gen_flags_search(q: u16) -> (bool, bool, [bool; 4]) {
    let mut best = (false, false, [false; 4]);
    let mut best_cost = i32::MAX;
    let mut best_out = i32::MAX;
    let mut mask = 0usize;
    while mask < 64 {
        let bi = mask & 1 != 0;
        let ba = mask & 2 != 0;
        let bl = [mask & 4 != 0, mask & 8 != 0, mask & 16 != 0, mask & 32 != 0];
        let mut cost = 0i32;
        if bi {
            cost += 324;
        }
        if ba {
            cost += 324;
        }
        let mut l = 0;
        while l < 4 {
            if bl[l] {
                cost += 216;
            }
            l += 1;
        }
        let (lm, peak) = gen_model(q, bi, ba, bl);
        if peak <= 32767 && (cost < best_cost || (cost == best_cost && lm[4] < best_out)) {
            best = (bi, ba, bl);
            best_cost = cost;
            best_out = lm[4];
        }
        mask += 1;
    }
    best
}

/// The search's answer and the bound it implies, per prime, evaluated once.
const GEN_SCHED: [((bool, bool, [bool; 4]), i32); 3] = [
    gen_sched(QS_QUAD[0]),
    gen_sched(QS_QUAD[1]),
    gen_sched(QS_QUAD[2]),
];

const fn gen_sched(q: u16) -> ((bool, bool, [bool; 4]), i32) {
    let f = gen_flags_search(q);
    (f, gen_model(q, f.0, f.1, f.2).0[4])
}

const fn qi(q: u16) -> usize {
    if q == QS_QUAD[0] {
        0
    } else if q == QS_QUAD[1] {
        1
    } else {
        2
    }
}

/// The reduction placement this kernel uses for `q`.
pub const fn gen_flags(q: u16) -> (bool, bool, [bool; 4]) {
    GEN_SCHED[qi(q)].0
}

/// Declared output bound: max |lane| of [`ntt_quad_gen_batch32`], per prime.
pub const fn output_bound(q: u16) -> i32 {
    GEN_SCHED[qi(q)].1
}

const _: () = {
    let f = gen_flags(2917);
    assert!(gen_model(2917, f.0, f.1, f.2).1 <= 32767);
    let f = gen_flags(4861);
    assert!(gen_model(4861, f.0, f.1, f.2).1 <= 32767);
    let f = gen_flags(12637);
    assert!(gen_model(12637, f.0, f.1, f.2).1 <= 32767);
};

// ---------------------------------------------------------------------------------------------
// tables
// ---------------------------------------------------------------------------------------------

const fn dup(w: i16) -> u32 {
    let u = w as u16 as u32;
    u | (u << 16)
}

/// The four 512-bit constants of the lookup Barrett, 64-byte aligned so they can be `vmovdqa64`
/// memory operands.
#[repr(C, align(64))]
pub struct Cv(pub [[i16; 32]; 4]);

/// Compile-time twiddle tables in the duplicated-u32 broadcast form: radix-2 levels store
/// `[zeta', zeta]` per sub-ring, radix-3 levels `[zeta', zeta, (zeta^2)', zeta^2]`.
pub struct TwQ<const Q: u16>;

impl<const Q: u16> TwQ<Q> {
    pub const fn pair(x: u16) -> [u32; 2] {
        let w = Params::<Q>::to_mont(x);
        [dup(Params::<Q>::mont_pre(w)), dup(w)]
    }
    const fn r2<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let p = Self::pair(ParamsQ::<Q>::zeta(level, k));
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
            let z = ParamsQ::<Q>::zeta(level, k);
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
    pub const Z6: [u32; 2] = Self::pair(ParamsQ::<Q>::ZETA6);
    pub const OM: [u32; 2] = Self::pair(ParamsQ::<Q>::OMEGA);
    pub const QD: u32 = dup(Q as i16);
    pub const L1: [u32; 4] = Self::r2::<4>(1, 2);
    pub const L2: [u32; 16] = Self::r3::<16>(2, 4);
    pub const L3: [u32; 48] = Self::r3::<48>(3, 12);
    pub const L4: [u32; 144] = Self::r3::<144>(4, 36);
    pub const L5: [u32; 432] = Self::r3::<432>(5, 108);

    /// The four 512-bit constants of the lookup Barrett: the `vpmultishiftqb` control, the
    /// byte-split `-k q` correction table and the two index fix-up masks.
    pub const CV: Cv = {
        let mut cv = [[0i16; 32]; 4];
        let mut i = 0;
        while i < 32 {
            cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
            let (b0, b1) = (lut_byte(2 * i, Q), lut_byte(2 * i + 1, Q));
            cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
            cv[2][i] = 0x1f1f;
            cv[3][i] = 0x2000;
            i += 1;
        }
        Cv(cv)
    };

    pub const BAR_IN: bool = gen_flags(Q).0;
    pub const BAR_A: bool = gen_flags(Q).1;
    pub const BAR_L: [bool; 4] = gen_flags(Q).2;
    pub const OUTPUT_BOUND: i32 = output_bound(Q);
}

// ---------------------------------------------------------------------------------------------
// arithmetic
// ---------------------------------------------------------------------------------------------

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
#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(a, b)
}
#[inline(always)]
unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
    _mm512_sub_epi16(a, b)
}

/// Signed Montgomery twiddle product: 3 multiply uops, `|r| <= |a| q/2^17 + q/2`.
#[inline(always)]
unsafe fn mont(a: __m512i, wp: __m512i, w: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = _mm512_mulhi_epi16(a, w);
    _mm512_sub_epi16(hi, _mm512_mulhi_epi16(m, q))
}

struct C {
    q: __m512i,
    om: __m512i,
    omp: __m512i,
    ms: __m512i,
    corr: __m512i,
    andm: __m512i,
    orm: __m512i,
}

impl C {
    #[inline(always)]
    unsafe fn new<const Q: u16>() -> C {
        let cv = TwQ::<Q>::CV.0.as_ptr() as *const __m512i;
        C {
            q: _mm512_set1_epi32(TwQ::<Q>::QD as i32),
            om: bc(TwQ::<Q>::OM.as_ptr().add(1)),
            omp: bc(TwQ::<Q>::OM.as_ptr()),
            ms: _mm512_load_si512(cv),
            corr: _mm512_load_si512(cv.add(1)),
            andm: _mm512_load_si512(cv.add(2)),
            orm: _mm512_load_si512(cv.add(3)),
        }
    }
}

/// The shuffle-port lookup Barrett: no multiply-port slot, `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn barrett_lut(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
}

/// Radix-3 butterfly `(a0, a1, a2) -> (a0+t1+t2, a0-t2+u, a0-t1-u)`, `t1 = zeta a1`,
/// `t2 = zeta^2 a2`, `u = omega (t1-t2)`: 9 multiply uops + 7 adds.
#[inline(always)]
unsafe fn r3<const BAR: bool>(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let t1 = mont(a1, bc(tw), bc(tw.add(1)), c.q);
    let t2 = mont(a2, bc(tw.add(2)), bc(tw.add(3)), c.q);
    let u = mont(sub(t1, t2), c.omp, c.om, c.q);
    let a0 = if BAR { barrett_lut(a0, c) } else { a0 };
    (
        add(a0, add(t1, t2)),
        add(sub(a0, t2), u),
        sub(sub(a0, t1), u),
    )
}

// ---------------------------------------------------------------------------------------------
// passes
// ---------------------------------------------------------------------------------------------

/// Levels 0 and 1 fused into one radix-4 pass over the 648 vectors.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_a<const Q: u16>(p: *mut __m512i, c: &C) {
    let z6p = bc(TwQ::<Q>::Z6.as_ptr());
    let z6 = bc(TwQ::<Q>::Z6.as_ptr().add(1));
    let l1 = TwQ::<Q>::L1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    for i in 0..162 {
        let a0 = ld(p, i);
        let b0 = ld(p, i + 162);
        let mut a1 = ld(p, i + 324);
        let mut b1 = ld(p, i + 486);
        if TwQ::<Q>::BAR_IN {
            a1 = barrett_lut(a1, c);
            b1 = barrett_lut(b1, c);
        }
        let t = mont(a1, z6p, z6, c.q);
        let s = mont(b1, z6p, z6, c.q);
        let c0 = add(a0, t);
        let c1 = add(b0, s);
        let mut c2 = sub(add(a0, a1), t);
        let mut c3 = sub(add(b0, b1), s);
        if TwQ::<Q>::BAR_A {
            c2 = barrett_lut(c2, c);
            c3 = barrett_lut(c3, c);
        }
        let u = mont(c1, zap, za, c.q);
        let w = mont(c3, zbp, zb, c.q);
        st(p, i, add(c0, u));
        st(p, i + 162, sub(c0, u));
        st(p, i + 324, add(c2, w));
        st(p, i + 486, sub(c2, w));
    }
}

/// Levels 2 and 3 for one 162-block: 18 groups of 9 vectors, 3 radix-3 butterflies each level.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_b<const Q: u16>(p: *mut __m512i, c: &C, blk: usize) {
    let t2 = TwQ::<Q>::L2.as_ptr().add(4 * blk);
    let base = 162 * blk;
    for i0 in 0..18 {
        let mut y = [_mm512_setzero_si512(); 9];
        for a in 0..3 {
            let b = base + i0 + 18 * a;
            let (u0, u1, u2) =
                bar_switch!(r3, TwQ::<Q>::BAR_L[0], c, ld(p, b), ld(p, b + 54), ld(p, b + 108), t2);
            y[a] = u0;
            y[3 + a] = u1;
            y[6 + a] = u2;
        }
        for s in 0..3 {
            let tw = TwQ::<Q>::L3.as_ptr().add(4 * (3 * blk + s));
            let (v0, v1, v2) =
                bar_switch!(r3, TwQ::<Q>::BAR_L[1], c, y[3 * s], y[3 * s + 1], y[3 * s + 2], tw);
            let b = base + 54 * s + i0;
            st(p, b, v0);
            st(p, b + 18, v1);
            st(p, b + 36, v2);
        }
    }
}

/// Levels 4 and 5 for one 18-block, register-resident.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_l4<const Q: u16>(p: *mut __m512i, c: &C, k4: usize) {
    let t4 = TwQ::<Q>::L4.as_ptr().add(4 * k4);
    let base = 18 * k4;
    for i in 0..6 {
        let b = base + i;
        let (y0, y1, y2) =
            bar_switch!(r3, TwQ::<Q>::BAR_L[2], c, ld(p, b), ld(p, b + 6), ld(p, b + 12), t4);
        st(p, b, y0);
        st(p, b + 6, y1);
        st(p, b + 12, y2);
    }
}

/// Level 5 alone for one 18-block (unfused variant).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_l5<const Q: u16>(p: *mut __m512i, c: &C, k4: usize) {
    let base = 18 * k4;
    for g in 0..3 {
        let t5 = TwQ::<Q>::L5.as_ptr().add(4 * (3 * k4 + g));
        for i in 0..2 {
            let b = base + 6 * g + i;
            let (y0, y1, y2) =
                bar_switch!(r3, TwQ::<Q>::BAR_L[3], c, ld(p, b), ld(p, b + 2), ld(p, b + 4), t5);
            st(p, b, y0);
            st(p, b + 2, y1);
            st(p, b + 4, y2);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------------------------

/// Forward quadratic-slot NTT of a batch of 32 polynomials in place: `Coefficients -> Ntt`, rows
/// `2j`, `2j+1` holding `a mod (X^2 - psi'^QUAD_SLOT_EXP[j])`.
///
/// Requires `|b.v[j][p]| <= q`; the output satisfies `|b.v[j][p]| <= TwQ::<Q>::OUTPUT_BOUND`.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn ntt_quad_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Coefficients);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    let c = C::new::<Q>();
    pass_a::<Q>(p, &c);
    for blk in 0..4 {
        pass_b::<Q>(p, &c, blk);
        for k4 in 9 * blk..9 * blk + 9 {
            pass_l4::<Q>(p, &c, k4);
            pass_l5::<Q>(p, &c, k4);
        }
    }
    b.representation = Representation::Ntt;
}

// =============================================================================================
// the inverse transform
// =============================================================================================
//
// `Ntt -> Coefficients` on the same tree, the five passes in the opposite order — level 5 and
// level 4 per 18-block, then levels 3 + 2 per 162-block, then levels 1 + 0 over the whole batch —
// with Gentleman-Sande butterflies, the transposes of the forward ones, which cost exactly what
// the forward ones cost: `u = omega (y2 - y1)`, then `(y0+y1+y2, (y0-y1+u) zeta^-1,
// (y0-y2-u) zeta^-2)`, 3 Montgomery products and 7 add/sub for radix 3, and
// `(y0+y1, (y0-y1) zeta^-1)` for radix 2.
//
// The per-level `1/3` and `1/2` are *not* applied: every value reaching level 0 is the true one
// times `2 * 3^4 = 162` — one factor 3 less than the splitting tree's 324, because the tree has
// one radix-2 level fewer — and the whole `1/324` corrected by the Phi_6 determinant sits in the
// three level-0 constants `TwQI::KA`, `KB`, `KC`, exactly as `crate::simd::ntt::gen_small` folds
// `1/648` into its own. The output is fully reduced and centered in `[-(q-1)/2, (q-1)/2]`.
//
// The reduction is again the shuffle-port lookup Barrett — this kernel has no multiply-port slack
// either — and its placement is a `const` search over one flag per level, `inv_flags`, for the
// cheapest schedule that keeps every intermediate inside i16 and leaves level 0 inside `3q/2`,
// which is what the two-conditional centering needs. `inv_model` replays it, and `inv_bound` is
// the per-level table it produces; `tests/quad.rs` replays the same schedule on data.
//
//     q         input     reduced levels               reductions   cycles/poly (forward)
//     2917      7.94 q    the level-5 inputs, 5, 3           1080     540 (393)
//     4861      3.26 q    the level-5 inputs, 4, 2           1080     540 (406)
//     12637     1.94 q    the level-5 inputs, every level    1836     602 (469)
//
// The declared input is the widest transform of this tree the crate produces — the forward
// kernel's own output, which is wider than the binary kernel's — so the three loaded values of a
// level-5 butterfly are reduced in every schedule: `y1 + y2` alone is 16 q at 2917 and leaves i16
// before anything can be done about it. What the inverse pays over its forward is what the linear
// pair pays: the Phi_6 inverse is a general 2x2 (3 Montgomery products per level-0 butterfly
// against pass A's 1), the untwiddled sums have to be reduced level after level because a sum
// cannot absorb a constant the way a Montgomery product does, and the 648 outputs are centered.

/// `|barrett_lut(a)| <= barrett_lut_max(q)`, and the reduction never grows a lane.
const fn red_bound(b: i32, q: u16) -> i32 {
    let r = barrett_lut_max(q);
    if r < b {
        r
    } else {
        b
    }
}

/// Declared input bound of [`intt_quad_gen_batch32`]: the largest transform of this tree the
/// crate produces, the wider of the binary kernel's lazily reduced output and this module's own
/// (7.94 q / 3.26 q / 1.94 q — the generic one, in all three cases). The fold's centered
/// `(q-1)/2` is far inside it.
pub const fn in_bound(q: u16) -> i32 {
    let bin = crate::simd::ntt::bin_quad::output_bound(q);
    let gen = output_bound(q);
    if bin > gen {
        bin
    } else {
        gen
    }
}

/// The inverse schedule replayed on bounds, position by position, with exactly the kernel's
/// reduction placement: `[after level 5, 4, 3, 2, 1, 0 before centering]` and the largest
/// intermediate the pass ever forms (which is what has to stay inside i16).
///
/// `f = [inputs of level 5, sums of level 5, of level 4, of level 3, of level 2, of level 1]`.
pub const fn inv_model(q: u16, f: [bool; 6]) -> ([i32; 6], i32) {
    let mut v = [in_bound(q); N];
    let mut lm = [0i32; 6];
    let mut peak = 0i32;
    macro_rules! ir3 {
        ($i0:expr, $i1:expr, $i2:expr, $bar:expr, $inb:expr) => {{
            let (mut y0, mut y1, mut y2) = (v[$i0], v[$i1], v[$i2]);
            if $inb {
                y0 = red_bound(y0, q);
                y1 = red_bound(y1, q);
                y2 = red_bound(y2, q);
            }
            let u = mont_bound(y1 + y2, q);
            let s = y0 + y1 + y2;
            let x1 = y0 + y1 + u;
            let x2 = y0 + y2 + u;
            if y1 + y2 > peak {
                peak = y1 + y2;
            }
            if s > peak {
                peak = s;
            }
            if x1 > peak {
                peak = x1;
            }
            if x2 > peak {
                peak = x2;
            }
            v[$i0] = if $bar { red_bound(s, q) } else { s };
            v[$i1] = mont_bound(x1, q);
            v[$i2] = mont_bound(x2, q);
        }};
    }
    macro_rules! ir2 {
        ($i0:expr, $i1:expr, $bar:expr) => {{
            let s = v[$i0] + v[$i1];
            if s > peak {
                peak = s;
            }
            v[$i0] = if $bar { red_bound(s, q) } else { s };
            v[$i1] = mont_bound(s, q);
        }};
    }
    macro_rules! level_max {
        ($l:expr) => {{
            let mut i = 0;
            while i < N {
                if v[i] > lm[$l] {
                    lm[$l] = v[i];
                }
                i += 1;
            }
        }};
    }
    let mut k4 = 0;
    while k4 < 36 {
        let base = 18 * k4;
        let mut g = 0;
        while g < 3 {
            let mut i = 0;
            while i < 2 {
                let b = base + 6 * g + i;
                ir3!(b, b + 2, b + 4, f[1], f[0]);
                i += 1;
            }
            g += 1;
        }
        k4 += 1;
    }
    level_max!(0);
    let mut k4 = 0;
    while k4 < 36 {
        let base = 18 * k4;
        let mut i = 0;
        while i < 6 {
            ir3!(base + i, base + i + 6, base + i + 12, f[2], false);
            i += 1;
        }
        k4 += 1;
    }
    level_max!(1);
    let mut blk = 0;
    while blk < 4 {
        let base = 162 * blk;
        let mut i0 = 0;
        while i0 < 18 {
            let mut s = 0;
            while s < 3 {
                let b = base + 54 * s + i0;
                ir3!(b, b + 18, b + 36, f[3], false);
                s += 1;
            }
            i0 += 1;
        }
        blk += 1;
    }
    level_max!(2);
    let mut blk = 0;
    while blk < 4 {
        let base = 162 * blk;
        let mut i0 = 0;
        while i0 < 18 {
            let mut a = 0;
            while a < 3 {
                let b = base + i0 + 18 * a;
                ir3!(b, b + 54, b + 108, f[4], false);
                a += 1;
            }
            i0 += 1;
        }
        blk += 1;
    }
    level_max!(3);
    let mut i = 0;
    while i < 162 {
        ir2!(i, i + 162, f[5]);
        ir2!(i + 324, i + 486, f[5]);
        i += 1;
    }
    level_max!(4);
    // level 0: a1 = mont(Y0-Y1), a0 = mont(Y0+Y1) + mont(Y0-Y1); then centered.
    let mut i = 0;
    while i < 324 {
        let s = v[i] + v[i + 324];
        if s > peak {
            peak = s;
        }
        let m = mont_bound(s, q);
        if 2 * m > peak {
            peak = 2 * m;
        }
        if 2 * m > lm[5] {
            lm[5] = 2 * m;
        }
        i += 1;
    }
    (lm, peak)
}

/// Reductions one flag buys, per batch of 32: 648 for the three loaded inputs of the 216
/// level-5 butterflies, 216 for the untwiddled output of each radix-3 level, 324 for the 324
/// radix-2 ones of level 1.
const fn inv_cost(f: [bool; 6]) -> i32 {
    let mut c = 0;
    if f[0] {
        c += 648;
    }
    let mut l = 1;
    while l < 5 {
        if f[l] {
            c += 216;
        }
        l += 1;
    }
    if f[5] {
        c += 324;
    }
    c
}

/// The cheapest reduction placement that keeps every intermediate inside i16 and level 0 inside
/// `3q/2` — the range the two-conditional centering of [`centre`] inverts — by exhaustive `const`
/// search over the 64 flag combinations.
const fn inv_flags_search(q: u16) -> [bool; 6] {
    let mut best = [true; 6];
    let mut best_cost = i32::MAX;
    let mut mask = 0usize;
    while mask < 64 {
        let mut f = [false; 6];
        let mut l = 0;
        while l < 6 {
            f[l] = mask & (1 << l) != 0;
            l += 1;
        }
        let (lm, peak) = inv_model(q, f);
        let cost = inv_cost(f);
        if peak <= 32767 && lm[5] <= q as i32 + (q as i32 - 1) / 2 && cost < best_cost {
            best = f;
            best_cost = cost;
        }
        mask += 1;
    }
    best
}

const INV_SCHED: [([bool; 6], [i32; 6]); 3] = [
    inv_sched(QS_QUAD[0]),
    inv_sched(QS_QUAD[1]),
    inv_sched(QS_QUAD[2]),
];

const fn inv_sched(q: u16) -> ([bool; 6], [i32; 6]) {
    let f = inv_flags_search(q);
    (f, inv_model(q, f).0)
}

/// The reduction placement [`intt_quad_gen_batch32`] uses for `q`.
pub const fn inv_flags(q: u16) -> [bool; 6] {
    INV_SCHED[qi(q)].0
}

/// `[after level 5, 4, 3, 2, 1, 0 before centering]` for `q`.
pub const fn inv_bound(q: u16) -> [i32; 6] {
    INV_SCHED[qi(q)].1
}

const _: () = {
    let mut i = 0;
    while i < 3 {
        let q = QS_QUAD[i];
        let (lm, peak) = inv_model(q, inv_flags(q));
        assert!(peak <= 32767);
        assert!(lm[5] <= q as i32 + (q as i32 - 1) / 2);
        i += 1;
    }
};

/// Compile-time inverse-twiddle tables, the level-0 recombination constants and the centering
/// pair, in the same duplicated-u32 broadcast form and the same per-level layout as [`TwQ`], so
/// a Gentleman-Sande butterfly reads its constants exactly where the forward one does.
pub struct TwQI<const Q: u16>;

impl<const Q: u16> TwQI<Q> {
    const fn inv(x: u16) -> u16 {
        inv_mod(x as u64, Q as u64) as u16
    }
    const fn r2i<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let p = TwQ::<Q>::pair(Self::inv(ParamsQ::<Q>::zeta(level, k)));
            t[2 * k] = p[0];
            t[2 * k + 1] = p[1];
            k += 1;
        }
        t
    }
    const fn r3i<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let z = Self::inv(ParamsQ::<Q>::zeta(level, k));
            let z2 = (z as u64 * z as u64 % Q as u64) as u16;
            let a = TwQ::<Q>::pair(z);
            let b = TwQ::<Q>::pair(z2);
            t[4 * k] = a[0];
            t[4 * k + 1] = a[1];
            t[4 * k + 2] = b[0];
            t[4 * k + 3] = b[1];
            k += 1;
        }
        t
    }
    pub const IL1: [u32; 4] = Self::r2i::<4>(1, 2);
    pub const IL2: [u32; 16] = Self::r3i::<16>(2, 4);
    pub const IL3: [u32; 48] = Self::r3i::<48>(3, 12);
    pub const IL4: [u32; 144] = Self::r3i::<144>(4, 36);
    pub const IL5: [u32; 432] = Self::r3i::<432>(5, 108);

    /// `d = (2 zeta6 - 1)^-1`, the determinant of the Phi_6 split.
    const DET: u16 = Self::inv(((2 * ParamsQ::<Q>::ZETA6 as u32 + Q as u32 - 1) % Q as u32) as u16);
    /// The whole normalisation, folded into the three level-0 constants. Levels 5..1 run
    /// un-normalised, so every value reaching level 0 carries the factor `2 * 3^4 = 162`; with
    /// `Y = 162 y`, `a1 = d (y0 - y1)` and `a0 = (y0+y1)/2 - a1/2` (using
    /// `zeta6 + zeta6^-1 = 1`) become
    ///
    /// ```text
    ///     a1 = KA (Y0 - Y1),   a0 = KB (Y0 + Y1) + KC (Y0 - Y1)
    ///     KA = d / 162,        KB = 1 / 324,      KC = -d / 324 = -KA / 2.
    /// ```
    pub const KA: [u32; 2] =
        TwQ::<Q>::pair((Self::DET as u64 * inv_mod(162, Q as u64) % Q as u64) as u16);
    pub const KB: [u32; 2] = TwQ::<Q>::pair(inv_mod(324, Q as u64) as u16);
    pub const KC: [u32; 2] = TwQ::<Q>::pair(
        ((Q as u64 - Self::DET as u64 * inv_mod(324, Q as u64) % Q as u64) % Q as u64) as u16,
    );
    /// `(q-1)/2` and `-(q-1)/2`, the centering constants of the output.
    pub const HALF: u32 = dup(((Q - 1) / 2) as i16);
    pub const NHALF: u32 = dup(-(((Q - 1) / 2) as i16));

    /// The reduction placement, `[level-5 inputs, sums of levels 5, 4, 3, 2, 1]`.
    pub const BAR: [bool; 6] = inv_flags(Q);
    pub const IN_BOUND: i32 = in_bound(Q);
    pub const OUT_BOUND: i32 = ((Q - 1) / 2) as i32;
}

/// Inverse radix-3 (Gentleman-Sande) butterfly, the exact transpose of [`r3`] with the level's
/// normalisation deferred: `u = omega (y2 - y1)`, `(y0+y1+y2, (y0-y1+u) zeta^-1,
/// (y0-y2-u) zeta^-2)` = `3 * (a0, a1, a2)`. `IN` reduces the three loaded values, `BAR` the
/// untwiddled sum, both where the bound search asked for them.
#[inline(always)]
unsafe fn ir3<const IN: bool, const BAR: bool>(
    c: &C,
    y0: __m512i,
    y1: __m512i,
    y2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let (y0, y1, y2) = if IN {
        (barrett_lut(y0, c), barrett_lut(y1, c), barrett_lut(y2, c))
    } else {
        (y0, y1, y2)
    };
    let u = mont(sub(y2, y1), c.omp, c.om, c.q);
    let s = add(y0, add(y1, y2));
    let a1 = mont(add(sub(y0, y1), u), bc(tw), bc(tw.add(1)), c.q);
    let a2 = mont(sub(sub(y0, y2), u), bc(tw.add(2)), bc(tw.add(3)), c.q);
    (if BAR { barrett_lut(s, c) } else { s }, a1, a2)
}

/// Inverse radix-2 butterfly, normalisation deferred: `(y0+y1, (y0-y1) zeta^-1)` = `2 (a0, a1)`.
#[inline(always)]
unsafe fn ir2<const BAR: bool>(
    c: &C,
    y0: __m512i,
    y1: __m512i,
    zp: __m512i,
    z: __m512i,
) -> (__m512i, __m512i) {
    let s = add(y0, y1);
    (
        if BAR { barrett_lut(s, c) } else { s },
        mont(sub(y0, y1), zp, z, c.q),
    )
}

/// Exact centered representative for `|x| <= 3q/2`: one conditional subtract and one conditional
/// add of q, which is all the output needs.
#[inline(always)]
unsafe fn centre(x: __m512i, q: __m512i, half: __m512i, nhalf: __m512i) -> __m512i {
    let hi = _mm512_cmpgt_epi16_mask(x, half);
    let x = _mm512_mask_sub_epi16(x, hi, x, q);
    let lo = _mm512_cmplt_epi16_mask(x, nhalf);
    _mm512_mask_add_epi16(x, lo, x, q)
}

/// Level 5 (the first inverse level) for one 18-block: 3 groups of 6 vectors, 2 butterflies each.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_l5<const Q: u16>(p: *mut __m512i, c: &C, k4: usize) {
    let base = 18 * k4;
    for g in 0..3 {
        let t5 = TwQI::<Q>::IL5.as_ptr().add(4 * (3 * k4 + g));
        for i in 0..2 {
            let b = base + 6 * g + i;
            let (y0, y1, y2) = (ld(p, b), ld(p, b + 2), ld(p, b + 4));
            let (s, a1, a2) = match (TwQI::<Q>::BAR[0], TwQI::<Q>::BAR[1]) {
                (false, false) => ir3::<false, false>(c, y0, y1, y2, t5),
                (false, true) => ir3::<false, true>(c, y0, y1, y2, t5),
                (true, false) => ir3::<true, false>(c, y0, y1, y2, t5),
                (true, true) => ir3::<true, true>(c, y0, y1, y2, t5),
            };
            st(p, b, s);
            st(p, b + 2, a1);
            st(p, b + 4, a2);
        }
    }
}

/// Level 4 for one 18-block: 6 butterflies, one per position class of the 6-block.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_l4<const Q: u16>(p: *mut __m512i, c: &C, k4: usize) {
    let t4 = TwQI::<Q>::IL4.as_ptr().add(4 * k4);
    let base = 18 * k4;
    for i in 0..6 {
        let b = base + i;
        let (y0, y1, y2) = (ld(p, b), ld(p, b + 6), ld(p, b + 12));
        let (s, a1, a2) = if TwQI::<Q>::BAR[2] {
            ir3::<false, true>(c, y0, y1, y2, t4)
        } else {
            ir3::<false, false>(c, y0, y1, y2, t4)
        };
        st(p, b, s);
        st(p, b + 6, a1);
        st(p, b + 12, a2);
    }
}

/// Levels 3 and 2 for one 162-block, the mirror of [`pass_b`]: 18 groups of 9 vectors, 3 inverse
/// radix-3 butterflies (level 3) followed by 3 more (level 2).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_b<const Q: u16>(p: *mut __m512i, c: &C, blk: usize) {
    let t2 = TwQI::<Q>::IL2.as_ptr().add(4 * blk);
    let base = 162 * blk;
    for i0 in 0..18 {
        let mut y = [_mm512_setzero_si512(); 9];
        for s in 0..3 {
            let tw = TwQI::<Q>::IL3.as_ptr().add(4 * (3 * blk + s));
            let b = base + 54 * s + i0;
            let (u0, u1, u2) = if TwQI::<Q>::BAR[3] {
                ir3::<false, true>(c, ld(p, b), ld(p, b + 18), ld(p, b + 36), tw)
            } else {
                ir3::<false, false>(c, ld(p, b), ld(p, b + 18), ld(p, b + 36), tw)
            };
            y[3 * s] = u0;
            y[3 * s + 1] = u1;
            y[3 * s + 2] = u2;
        }
        for a in 0..3 {
            let (v0, v1, v2) = if TwQI::<Q>::BAR[4] {
                ir3::<false, true>(c, y[a], y[3 + a], y[6 + a], t2)
            } else {
                ir3::<false, false>(c, y[a], y[3 + a], y[6 + a], t2)
            };
            let b = base + i0 + 18 * a;
            st(p, b, v0);
            st(p, b + 54, v1);
            st(p, b + 108, v2);
        }
    }
}

/// Levels 1 and 0 fused into one radix-4 pass over the 648 vectors, the mirror of [`pass_a`]:
/// the two inverse radix-2 butterflies of level 1, then the two Phi_6 recombinations, which
/// carry the whole normalisation ([`TwQI::KA`]) and center their four outputs.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_a<const Q: u16>(p: *mut __m512i, c: &C) {
    let half = _mm512_set1_epi32(TwQI::<Q>::HALF as i32);
    let nhalf = _mm512_set1_epi32(TwQI::<Q>::NHALF as i32);
    let l1 = TwQI::<Q>::IL1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    let ka = TwQI::<Q>::KA.as_ptr();
    let kb = TwQI::<Q>::KB.as_ptr();
    let kc = TwQI::<Q>::KC.as_ptr();
    for i in 0..162 {
        let ((c0, c1), (c2, c3)) = if TwQI::<Q>::BAR[5] {
            (
                ir2::<true>(c, ld(p, i), ld(p, i + 162), zap, za),
                ir2::<true>(c, ld(p, i + 324), ld(p, i + 486), zbp, zb),
            )
        } else {
            (
                ir2::<false>(c, ld(p, i), ld(p, i + 162), zap, za),
                ir2::<false>(c, ld(p, i + 324), ld(p, i + 486), zbp, zb),
            )
        };
        let d0 = sub(c0, c2);
        let a1 = mont(d0, bc(ka), bc(ka.add(1)), c.q);
        let a0 = add(
            mont(add(c0, c2), bc(kb), bc(kb.add(1)), c.q),
            mont(d0, bc(kc), bc(kc.add(1)), c.q),
        );
        let d1 = sub(c1, c3);
        let b1 = mont(d1, bc(ka), bc(ka.add(1)), c.q);
        let b0 = add(
            mont(add(c1, c3), bc(kb), bc(kb.add(1)), c.q),
            mont(d1, bc(kc), bc(kc.add(1)), c.q),
        );
        st(p, i, centre(a0, c.q, half, nhalf));
        st(p, i + 162, centre(b0, c.q, half, nhalf));
        st(p, i + 324, centre(a1, c.q, half, nhalf));
        st(p, i + 486, centre(b1, c.q, half, nhalf));
    }
}

/// Inverse quadratic-slot NTT of a batch of 32 polynomials in place: `Ntt -> Coefficients`, the
/// exact inverse of [`ntt_quad_gen_batch32`].
///
/// Requires `|b.v[j][p]| <= in_bound(Q)`, the binary kernel's lazily reduced output — every
/// transform of this tree the crate produces. The output is **fully reduced and centered**,
/// `|b.v[j][p]| <= (q-1)/2`.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn intt_quad_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Ntt);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    let c = C::new::<Q>();
    for blk in 0..4 {
        for k4 in 9 * blk..9 * blk + 9 {
            ipass_l5::<Q>(p, &c, k4);
            ipass_l4::<Q>(p, &c, k4);
        }
        ipass_b::<Q>(p, &c, blk);
    }
    ipass_a::<Q>(p, &c);
    b.representation = Representation::Coefficients;
}

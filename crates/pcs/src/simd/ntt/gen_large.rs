//! Generic-input NTT and its inverse on the vertical `Batch32` layout for `q` in [`QS_LARGE`]:
//! the same five passes, the same tree and the same twiddle tables as
//! [`crate::simd::ntt::gen_small`], with the reduction schedule the 1.87 / 1.69 head-room of these
//! primes forces.
//!
//! # What changes
//!
//! Three things, all of them consequences of `2^15 / q` (see
//! [`crate::simd::ntt::bin_large`], which makes the same argument for the binary kernel):
//!
//! * **The reduction is the lookup Barrett.** `round(2^15/q)` is 2 here, so the two-multiply
//!   `vpmulhrsw` estimate of [`crate::params::barrett_i16`] is worthless; the shuffle-port
//!   `vpmultishiftqb` + `vpermb` form is not, and costs no multiply-port slot.
//! * **A radix-3 butterfly reduces all of `a0`, `t1`, `t2` and `u`,** not just the untwiddled
//!   `a0`. Three terms of `q/2` are already `1.5 q`, so nothing bigger than a reduced value may
//!   enter a sum. A radix-2 one reduces its untwiddled inputs, and the forward level 2 reduces
//!   the three it does not multiply.
//! * **The Phi_6 split is two products, not one plus two adds.** `y1 = a0 + a1 - zeta6 a1` is
//!   `2 q + 0.65 q` at the declared input bound `|x| <= q`; written as `a0 + zeta6^-1 a1` — the
//!   same value, `zeta6^-1 = 1 - zeta6` — it is `1.65 q`, which fits.
//!
//! The forward kernel additionally reduces the three outputs of level 6, so its output is
//! `|x| <= q/2 + 2^10` rather than the 2.1-3.4 q of the small primes. That is what
//! [`crate::fold::center_batch`] needs: it shifts by `K q` into an unsigned window and wants
//! `K q + bound <= 2^16`, and at `q = 19441` even `K = 2` does not fit, so the bound has to be
//! below `q` and `K` has to be 1.
//!
//! # Bounds
//!
//! [`fwd_model`] and [`inv_model`] replay the schedules on bounds and are asserted below; the i32
//! shadow in `tests/vertical_gen_large.rs` replays them on the data. `r` is
//! [`vl::barrett_lut_max`], `0.527 q` for both primes.
//!
//! ```text
//!   forward (input |x| <= q)              inverse (input |x| <= (q-1)/2)
//!     levels 0+1   1.27 q                   level 6   1.50 q
//!     levels 2+3   1.58 q                   level 5   1.58 q
//!     level 4      1.58 q                   level 4   1.58 q
//!     level 5      1.58 q                   level 3   1.58 q
//!     level 6      0.53 q  (output)         level 2   1.05 q
//!                                           level 1   1.05 q
//!                                           level 0   1.31 q -> centered
//! ```
//!
//! Neither kernel is on a hot path — the forward one runs on the `len_ring/32` batches of a
//! folded witness once per verification, the inverse on the residues of a commitment's columns —
//! so the schedule is the uniform one above rather than the cheapest that fits, and the recursions
//! prove it rather than search for it.
use crate::params::*;
use crate::simd::ntt::bin_large as vl;
use crate::simd::ntt::gen_small::{Tw, TwI};
use crate::ring::{Batch32, Representation};
use core::arch::x86_64::*;

// =============================================================================================
// bounds
// =============================================================================================

/// `|mont(a, w)| <= |a| q / 2^17 + q/2`.
const fn mont_bound(b: i32, q: u16) -> i32 {
    ((b as i64 * q as i64) >> 17) as i32 + (q as i32 + 1) / 2
}

/// A reduced value: the lookup Barrett never makes a lane bigger than it was.
const fn red_bound(b: i32, q: u16) -> i32 {
    let r = vl::barrett_lut_max(q);
    if r < b {
        r
    } else {
        b
    }
}

/// The forward schedule replayed on bounds: `[after levels 0+1, 2+3, 4, 5, 6]` and the largest
/// intermediate ever formed.
pub const fn fwd_model(q: u16) -> ([i32; 5], i32) {
    let mut peak = 0i32;
    macro_rules! pk {
        ($x:expr) => {{
            let x = $x;
            if x > peak {
                peak = x;
            }
            x
        }};
    }
    // one radix-3 butterfly on a bound, with everything reduced
    macro_rules! r3 {
        ($a0:expr) => {{
            let t = red_bound(pk!(mont_bound($a0, q)), q);
            let u = red_bound(pk!(mont_bound(pk!(2 * t), q)), q);
            let a0 = red_bound($a0, q);
            pk!(a0 + 2 * t);
            pk!(a0 + t + u)
        }};
    }
    let mut lm = [0i32; 5];
    // pass A: level 0 as two products, then the radix-2 of level 1 on a reduced `a0`.
    let c = pk!(q as i32 + mont_bound(q as i32, q));
    lm[0] = pk!(red_bound(c, q) + mont_bound(c, q));
    // pass B: level 2 reduces the three inputs it does not multiply, then level 3.
    let n = pk!(red_bound(lm[0], q) + mont_bound(lm[0], q));
    lm[1] = r3!(n);
    lm[2] = r3!(lm[1]);
    lm[3] = r3!(lm[2]);
    lm[4] = red_bound(r3!(lm[3]), q);
    (lm, peak)
}

/// The inverse schedule replayed on bounds: `[after levels 6, 5, 4, 3, 2, 1, 0]` and the peak.
pub const fn inv_model(q: u16) -> ([i32; 7], i32) {
    let mut peak = 0i32;
    macro_rules! pk {
        ($x:expr) => {{
            let x = $x;
            if x > peak {
                peak = x;
            }
            x
        }};
    }
    // one Gentleman-Sande radix-3 on a bound: the three inputs and `u` are reduced, and the
    // untwiddled sum is left for the next level's input reduction.
    macro_rules! ir3 {
        ($b:expr) => {{
            let y = red_bound($b, q);
            let u = red_bound(pk!(mont_bound(pk!(2 * y), q)), q);
            let s = pk!(3 * y);
            let a = pk!(mont_bound(pk!(2 * y + u), q));
            if s > a {
                s
            } else {
                a
            }
        }};
    }
    macro_rules! ir2 {
        ($b:expr) => {{
            let y = red_bound($b, q);
            let s = pk!(2 * y);
            let a = pk!(mont_bound(s, q));
            if s > a {
                s
            } else {
                a
            }
        }};
    }
    let mut lm = [0i32; 7];
    lm[0] = ir3!((q as i32 - 1) / 2);
    lm[1] = ir3!(lm[0]);
    lm[2] = ir3!(lm[1]);
    lm[3] = ir3!(lm[2]);
    lm[4] = ir2!(lm[3]);
    lm[5] = ir2!(lm[4]);
    // level 0: a1 = mont(Y0 - Y1), a0 = mont(Y0 + Y1) + mont(Y0 - Y1), then centered.
    let y = red_bound(lm[5], q);
    lm[6] = pk!(2 * mont_bound(pk!(2 * y), q));
    (lm, peak)
}

/// Declared output bound of [`ntt_gen_batch32`]: level 6 reduces its three outputs, so this is
/// the lookup Barrett's own bound, below `q`.
pub const fn output_bound(q: u16) -> i32 {
    fwd_model(q).0[4]
}

/// Declared input bound of [`intt_gen_batch32`]: a fully reduced centered transform, which is
/// what `recursion::limbs` hands it.
pub const fn in_bound(q: u16) -> i32 {
    ((q - 1) / 2) as i32
}

const _: () = {
    let mut i = 0;
    while i < 2 {
        let q = QS_LARGE[i];
        assert!(fwd_model(q).1 <= 32767);
        assert!(inv_model(q).1 <= 32767);
        assert!(output_bound(q) < q as i32);
        i += 1;
    }
};

// =============================================================================================
// constants
// =============================================================================================

const fn dup(w: i16) -> u32 {
    let u = w as u16 as u32;
    u | (u << 16)
}

const fn pair<const Q: u16>(x: u16) -> [u32; 2] {
    let w = Params::<Q>::to_mont(x);
    [dup(Params::<Q>::mont_pre(w)), dup(w)]
}

/// What this kernel needs on top of [`Tw`]: the inverse sixth root, which turns the Phi_6 split
/// into two products, and the four lookup-Barrett vectors.
pub struct TwL<const Q: u16>;

impl<const Q: u16> TwL<Q> {
    /// `zeta6^-1 = 1 - zeta6`.
    pub const Z6I: [u32; 2] =
        pair::<Q>(((1 + Q as u32 - Params::<Q>::ZETA6 as u32) % Q as u32) as u16);
    /// `[ms, corr, and, or]`: the `vpmultishiftqb` control, the byte-split `-k q` table and the
    /// index fix-up masks of the lookup Barrett.
    pub const CV: [[i16; 32]; 4] = {
        let mut cv = [[0i16; 32]; 4];
        let mut i = 0;
        while i < 32 {
            cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
            let (b0, b1) = (vl::lut_byte(2 * i, Q), vl::lut_byte(2 * i + 1, Q));
            cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
            cv[2][i] = 0x1f1f;
            cv[3][i] = 0x2000;
            i += 1;
        }
        cv
    };
}

// =============================================================================================
// arithmetic
// =============================================================================================

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

#[inline(always)]
unsafe fn mont(a: __m512i, wp: __m512i, w: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = _mm512_mulhi_epi16(a, w);
    _mm512_sub_epi16(hi, _mm512_mulhi_epi16(m, q))
}

/// The constants every pass carries: `q`, omega and its companion, and the lookup Barrett's four.
struct C {
    q: __m512i,
    omp: __m512i,
    om: __m512i,
    ms: __m512i,
    corr: __m512i,
    andm: __m512i,
    orm: __m512i,
}

impl C {
    #[inline(always)]
    unsafe fn new<const Q: u16>() -> C {
        // the four vectors come out of an associated const, whose promoted static is only
        // i16-aligned.
        let cvp = TwL::<Q>::CV.as_ptr() as *const __m512i;
        C {
            q: _mm512_set1_epi32(Tw::<Q>::QD as i32),
            omp: bc(Tw::<Q>::OM.as_ptr()),
            om: bc(Tw::<Q>::OM.as_ptr().add(1)),
            ms: _mm512_loadu_si512(cvp),
            corr: _mm512_loadu_si512(cvp.add(1)),
            andm: _mm512_loadu_si512(cvp.add(2)),
            orm: _mm512_loadu_si512(cvp.add(3)),
        }
    }
}

/// The shuffle-port lookup Barrett: 2 port-5 + 3 flexible uops, `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn red(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
}

/// Radix-3 butterfly `(a0, a1, a2) -> (a0+t1+t2, a0-t2+u, a0-t1-u)` with every term reduced.
#[inline(always)]
unsafe fn r3(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let t1 = red(mont(a1, bc(tw), bc(tw.add(1)), c.q), c);
    let t2 = red(mont(a2, bc(tw.add(2)), bc(tw.add(3)), c.q), c);
    let u = red(mont(sub(t1, t2), c.omp, c.om, c.q), c);
    let a0 = red(a0, c);
    (
        add(a0, add(t1, t2)),
        add(sub(a0, t2), u),
        sub(sub(a0, t1), u),
    )
}

/// Inverse radix-3 (Gentleman-Sande), the transpose of [`r3`], with the three inputs and `u`
/// reduced and the normalisation deferred to level 0.
#[inline(always)]
unsafe fn ir3(
    c: &C,
    y0: __m512i,
    y1: __m512i,
    y2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let (y0, y1, y2) = (red(y0, c), red(y1, c), red(y2, c));
    let u = red(mont(sub(y2, y1), c.omp, c.om, c.q), c);
    let s = add(y0, add(y1, y2));
    let a1 = mont(add(sub(y0, y1), u), bc(tw), bc(tw.add(1)), c.q);
    let a2 = mont(sub(sub(y0, y2), u), bc(tw.add(2)), bc(tw.add(3)), c.q);
    (s, a1, a2)
}

/// Inverse radix-2, normalisation deferred: `(y0+y1, (y0-y1) zeta^-1)`.
#[inline(always)]
unsafe fn ir2(c: &C, y0: __m512i, y1: __m512i, zp: __m512i, z: __m512i) -> (__m512i, __m512i) {
    let (y0, y1) = (red(y0, c), red(y1, c));
    (add(y0, y1), mont(sub(y0, y1), zp, z, c.q))
}

/// Exact centered representative for `|x| <= 3q/2`.
#[inline(always)]
unsafe fn centre(x: __m512i, q: __m512i, half: __m512i, nhalf: __m512i) -> __m512i {
    let hi = _mm512_cmpgt_epi16_mask(x, half);
    let x = _mm512_mask_sub_epi16(x, hi, x, q);
    let lo = _mm512_cmplt_epi16_mask(x, nhalf);
    _mm512_mask_add_epi16(x, lo, x, q)
}

// =============================================================================================
// forward
// =============================================================================================

/// Levels 0 and 1: the Phi_6 split as the two products `a0 + zeta6 a1` and `a0 + zeta6^-1 a1`,
/// then the radix-2 of level 1 on a reduced untwiddled input.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_a<const Q: u16>(p: *mut __m512i) {
    let c = C::new::<Q>();
    let z6p = bc(Tw::<Q>::Z6.as_ptr());
    let z6 = bc(Tw::<Q>::Z6.as_ptr().add(1));
    let z6ip = bc(TwL::<Q>::Z6I.as_ptr());
    let z6i = bc(TwL::<Q>::Z6I.as_ptr().add(1));
    let l1 = Tw::<Q>::L1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    for i in 0..162 {
        let a0 = ld(p, i);
        let a1 = ld(p, i + 324);
        let b0 = ld(p, i + 162);
        let b1 = ld(p, i + 486);
        let c0 = red(add(a0, mont(a1, z6p, z6, c.q)), &c);
        let c1 = add(b0, mont(b1, z6p, z6, c.q));
        let c2 = red(add(a0, mont(a1, z6ip, z6i, c.q)), &c);
        let c3 = add(b0, mont(b1, z6ip, z6i, c.q));
        let u = mont(c1, zap, za, c.q);
        let w = mont(c3, zbp, zb, c.q);
        st(p, i, add(c0, u));
        st(p, i + 162, sub(c0, u));
        st(p, i + 324, add(c2, w));
        st(p, i + 486, sub(c2, w));
    }
}

/// Levels 2 and 3 for one 162-block: 27 groups of 6 vectors, 3 radix-2 then 2 radix-3.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_b<const Q: u16>(p: *mut __m512i, blk: usize) {
    let c = C::new::<Q>();
    let l2 = Tw::<Q>::L2.as_ptr().add(2 * blk);
    let (zp, z) = (bc(l2), bc(l2.add(1)));
    let ta = Tw::<Q>::L3.as_ptr().add(8 * blk);
    let tb = ta.add(4);
    let base = blk * 162;
    for j in 0..27 {
        let b = base + j;
        let x0 = red(ld(p, b), &c);
        let x1 = red(ld(p, b + 27), &c);
        let x2 = red(ld(p, b + 54), &c);
        let t0 = mont(ld(p, b + 81), zp, z, c.q);
        let t1 = mont(ld(p, b + 108), zp, z, c.q);
        let t2 = mont(ld(p, b + 135), zp, z, c.q);
        let (y0, y1, y2) = r3(&c, add(x0, t0), add(x1, t1), add(x2, t2), ta);
        let (w0, w1, w2) = r3(&c, sub(x0, t0), sub(x1, t1), sub(x2, t2), tb);
        st(p, b, y0);
        st(p, b + 27, y1);
        st(p, b + 54, y2);
        st(p, b + 81, w0);
        st(p, b + 108, w1);
        st(p, b + 135, w2);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_c4<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t4 = Tw::<Q>::L4.as_ptr().add(4 * k4);
    let base = k4 * 27;
    for i in 0..9 {
        let b = base + i;
        let (y0, y1, y2) = r3(&c, ld(p, b), ld(p, b + 9), ld(p, b + 18), t4);
        st(p, b, y0);
        st(p, b + 9, y1);
        st(p, b + 18, y2);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_c5<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t5 = Tw::<Q>::L5.as_ptr().add(12 * k4);
    let base = k4 * 27;
    for bb in 0..3 {
        for j in 0..3 {
            let b = base + 9 * bb + j;
            let (y0, y1, y2) = r3(&c, ld(p, b), ld(p, b + 3), ld(p, b + 6), t5.add(4 * bb));
            st(p, b, y0);
            st(p, b + 3, y1);
            st(p, b + 6, y2);
        }
    }
}

/// Level 6, whose three outputs are reduced so that the transform leaves `|x| < q`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn pass_d<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t6 = Tw::<Q>::L6.as_ptr().add(36 * k4);
    let base = k4 * 27;
    for g in 0..9 {
        let b = base + 3 * g;
        let (y0, y1, y2) = r3(&c, ld(p, b), ld(p, b + 1), ld(p, b + 2), t6.add(4 * g));
        st(p, b, red(y0, &c));
        st(p, b + 1, red(y1, &c));
        st(p, b + 2, red(y2, &c));
    }
}

/// Forward NTT of a batch of 32 polynomials in place: `Coefficients -> Ntt` (tree order).
///
/// Requires `|b.v[j][p]| <= q`. Output satisfies `|b.v[j][p]| <= output_bound(Q) < q`.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn ntt_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Coefficients);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        pass_b::<Q>(p, blk);
        for k4 in 6 * blk..6 * blk + 6 {
            pass_c4::<Q>(p, k4);
            pass_c5::<Q>(p, k4);
            pass_d::<Q>(p, k4);
        }
    }
    b.representation = Representation::Ntt;
}

// =============================================================================================
// inverse
// =============================================================================================

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_d<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t6 = TwI::<Q>::IL6.as_ptr().add(36 * k4);
    let base = k4 * 27;
    for g in 0..9 {
        let b = base + 3 * g;
        let (s, a1, a2) = ir3(&c, ld(p, b), ld(p, b + 1), ld(p, b + 2), t6.add(4 * g));
        st(p, b, s);
        st(p, b + 1, a1);
        st(p, b + 2, a2);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_c5<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t5 = TwI::<Q>::IL5.as_ptr().add(12 * k4);
    let base = k4 * 27;
    for bb in 0..3 {
        for j in 0..3 {
            let b = base + 9 * bb + j;
            let (s, a1, a2) = ir3(&c, ld(p, b), ld(p, b + 3), ld(p, b + 6), t5.add(4 * bb));
            st(p, b, s);
            st(p, b + 3, a1);
            st(p, b + 6, a2);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_c4<const Q: u16>(p: *mut __m512i, k4: usize) {
    let c = C::new::<Q>();
    let t4 = TwI::<Q>::IL4.as_ptr().add(4 * k4);
    let base = k4 * 27;
    for i in 0..9 {
        let b = base + i;
        let (s, a1, a2) = ir3(&c, ld(p, b), ld(p, b + 9), ld(p, b + 18), t4);
        st(p, b, s);
        st(p, b + 9, a1);
        st(p, b + 18, a2);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_b<const Q: u16>(p: *mut __m512i, blk: usize) {
    let c = C::new::<Q>();
    let l2 = TwI::<Q>::IL2.as_ptr().add(2 * blk);
    let (zp, z) = (bc(l2), bc(l2.add(1)));
    let ta = TwI::<Q>::IL3.as_ptr().add(8 * blk);
    let tb = ta.add(4);
    let base = blk * 162;
    for j in 0..27 {
        let b = base + j;
        let (n0, n1, n2) = ir3(&c, ld(p, b), ld(p, b + 27), ld(p, b + 54), ta);
        let (m0, m1, m2) = ir3(&c, ld(p, b + 81), ld(p, b + 108), ld(p, b + 135), tb);
        let (s0, t0) = ir2(&c, n0, m0, zp, z);
        let (s1, t1) = ir2(&c, n1, m1, zp, z);
        let (s2, t2) = ir2(&c, n2, m2, zp, z);
        st(p, b, s0);
        st(p, b + 27, s1);
        st(p, b + 54, s2);
        st(p, b + 81, t0);
        st(p, b + 108, t1);
        st(p, b + 135, t2);
    }
}

/// Levels 1 and 0: the two inverse radix-2 butterflies of level 1, then the Phi_6 recombination,
/// which carries the whole `1/648` normalisation ([`TwI::KA`]) and centers its four outputs.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn ipass_a<const Q: u16>(p: *mut __m512i) {
    let c = C::new::<Q>();
    let half = _mm512_set1_epi32(TwI::<Q>::HALF as i32);
    let nhalf = _mm512_set1_epi32(TwI::<Q>::NHALF as i32);
    let l1 = TwI::<Q>::IL1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    let ka = TwI::<Q>::KA.as_ptr();
    let kb = TwI::<Q>::KB.as_ptr();
    let kc = TwI::<Q>::KC.as_ptr();
    for i in 0..162 {
        let (c0, c1) = ir2(&c, ld(p, i), ld(p, i + 162), zap, za);
        let (c2, c3) = ir2(&c, ld(p, i + 324), ld(p, i + 486), zbp, zb);
        let (c0, c2) = (red(c0, &c), red(c2, &c));
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

/// Inverse NTT of a batch of 32 polynomials in place: `Ntt -> Coefficients`, the exact inverse of
/// [`ntt_gen_batch32`].
///
/// Requires `|b.v[j][p]| <= in_bound(Q) = (q-1)/2`, which is what a fully reduced centered
/// transform gives. The output is fully reduced and centered.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn intt_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Ntt);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    for blk in 0..4 {
        for k4 in 6 * blk..6 * blk + 6 {
            ipass_d::<Q>(p, k4);
            ipass_c5::<Q>(p, k4);
            ipass_c4::<Q>(p, k4);
        }
        ipass_b::<Q>(p, blk);
    }
    ipass_a::<Q>(p);
    b.representation = Representation::Coefficients;
}

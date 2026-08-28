//! Generic-input forward NTT in Gregor Seiler's "horizontal" layout: 4 polynomials per batch, one
//! polynomial per 128-bit lane, 81 zmm registers (5 KB) holding the 4 whole polynomials.
//!
//! # Layout
//!
//! `HBatch4::v[r][8*p + j]` = coefficient `r + 81*j` of polynomial `p` (`p` in 0..4, `j` in 0..8,
//! `r` in 0..81). One 512-bit vector is one `r`; inside it, lane group `8p..8p+7` is polynomial
//! `p` and the 8 lanes of that group are the 8 coefficients of stride 81.
//!
//! # How the tree of `params` maps onto it
//!
//! Coefficient index `i = r + 81*j`, so the three radix-2-type levels act on `j` (in-lane) and the
//! four radix-3 levels act on `r` (register-to-register, no shuffles at all):
//!
//! * level 0 (Phi_6 split, `i` vs `i+324`): pairs `(j, j+4)`; `u[j] = a[j|4]`, `v[j] = a[j&3]`
//!   (two `vpshufd`), `out = v + c0*u` with `c0[j] = zeta6` for `j < 4` and `1 - zeta6 = zeta6^-1`
//!   for `j >= 4`.
//! * level 1 (radix 2 inside each 324-block): pairs `(j, j+2)`, `u[j] = a[j|2]`, `v[j] = a[j&!2]`
//!   (two `vpshufd`), `out = v + c1*u`, `c1[j] = +-ZETA_L1[j/4]` (`+` for `j%4 < 2`).
//! * level 2 (radix 2 inside each 162-block): pairs `(j, j+1)`, `u[j] = a[j|1]`, `v[j] = a[j&!1]`
//!   (two `vpshufb` with a constant control), `out = v + c2*u`, `c2[j] = +-ZETA_L2[j/2]`.
//!
//! After level 2 lane position `j = 4*s0 + 2*s1 + s2` holds the level-3 sub-ring `k3 = j`, i.e. the
//! 81-block at tree offset `81*j`, and `r` is the position inside that block. Levels 3..6 are then
//! plain radix-3 butterflies on `r` with strides 27, 9, 3, 1; the sub-ring index of register `r` in
//! lane position `j` at level `l` is `k_l = j * 3^(l-3) + r / DEGREE[l]`, so the twiddle vector
//! depends on `j` (lane) and on `r / DEGREE[l]` only: 1 + 3 + 9 + 27 = 40 constant 512-bit
//! (zeta, zeta^2) pairs, loaded once per pass.
//!
//! # Slot order
//!
//! The output is in TREE order: `v[r][8p + j]` = tree slot `81*j + r` of polynomial `p`, i.e.
//! `a_p(psi^SLOT_EXP[81*j + r])`. `HBatch4::get(p)` performs exactly that permutation.
//!
//! # Bounds (units of q; see `BOUNDS`, computed at compile time by the same recursion)
//!
//! `M(B) = B*q/2^17 + 1/2` is the Montgomery-product bound for an input of size `B*q`.
//!
//! | after level | 3889   | 9721 (barrett on a0 at levels 3..6) |
//! |-------------|--------|--------|
//! | input       | 1.000  | 1.000  |
//! | 0           | 1.530  | 1.574  |
//! | 1           | 2.075  | 2.191  |
//! | 2           | 2.637  | 2.854  |
//! | 3           | 3.794  | 2.233  |
//! | 4           | 5.019  | 2.141  |
//! | 5           | 6.317  | 2.126  |
//! | 6 (output)  | 7.692  | 2.125  |
//!
//! q = 3889 needs no reduction anywhere (7.692q = 29915 < 2^15, budget 8.42q); q = 9721 needs one
//! `barrett` (vpmulhrsw form, 2 multiply uops) on the untwiddled `a0` input of each radix-3 level,
//! which is the cheapest placement that has a fixed point (~2.13q) below the 3.37q budget.
//!
//! # Schedule and cost
//!
//! Five passes over the 81 registers (5 KB, L1-resident throughout): levels 0-2 fused per register,
//! then one pass per radix-3 level with three independent triples per iteration. Fusing radix-3
//! levels into one pass is slower (~5%): the second level waits for the whole first stage, whereas
//! separate passes keep three independent 24-cycle butterfly chains in flight, and the extra L1
//! round trip costs only load/store uops, which do not compete for p0/p5.
//!
//! Per polynomial (static count, q = 3889): 425 multiply uops (p0 only), 124 shuffles (p5 only),
//! 392 adds/subtracts (p0 or p5), 232 loads/stores/prefetches; 941 ALU uops, so >= 471 cycles on
//! the two 512-bit ALU ports and >= 425 if the multiplies alone were the constraint. Measured
//! (i7-11850H, one core, L1/L2-resident): 541 cycles, 1294 instructions, 1285 uops, p0 511, p5 454
//! (q = 3889); 589 cycles, 1382 instructions, p0 559, p5 493 (q = 9721, the Barretts add 54 p0
//! uops). p0 is 94% occupied; the gap to 471 is the port assignment of the flexible adds.
use crate::params::*;
use crate::types::{BinaryPoly, Representation, RingElement};
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// Layout type
// ---------------------------------------------------------------------------------------------

/// 4 ring elements in the horizontal layout: `v[r][8*p + j]` = coefficient `r + 81*j` of poly `p`
/// (in `Coefficients`), tree slot `81*j + r` of poly `p` (in `Ntt`). 64-byte aligned, 5184 bytes.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct HBatch4 {
    pub v: [[i16; 32]; 81],
}

impl Default for HBatch4 {
    fn default() -> Self {
        Self::zero()
    }
}

impl HBatch4 {
    pub const POLYS: usize = 4;

    pub fn zero() -> Self {
        HBatch4 { v: [[0i16; 32]; 81] }
    }

    /// Element `p` in tree order (`Ntt`): `out.v[81*j + r] = self.v[r][8*p + j]`.
    pub fn get(&self, p: usize) -> RingElement {
        self.get_as(p, Representation::Ntt)
    }

    /// Element `p` with an explicit representation tag. Both directions use the same index map:
    /// coefficient `r + 81*j` sits where tree slot `81*j + r` ends up.
    pub fn get_as(&self, p: usize, representation: Representation) -> RingElement {
        let mut e = RingElement::zero(representation);
        for r in 0..81 {
            for j in 0..8 {
                e.v[r + 81 * j] = self.v[r][8 * p + j];
            }
        }
        e
    }

    pub fn set(&mut self, p: usize, e: &RingElement) {
        for r in 0..81 {
            for j in 0..8 {
                self.v[r][8 * p + j] = e.v[r + 81 * j];
            }
        }
    }

    pub fn from_elements(es: &[RingElement; 4]) -> Self {
        let mut b = Self::zero();
        for (p, e) in es.iter().enumerate() {
            debug_assert_eq!(e.representation, Representation::Coefficients);
            b.set(p, e);
        }
        b
    }

    pub fn from_binary(polys: &[BinaryPoly; 4]) -> Self {
        let mut b = Self::zero();
        for (p, poly) in polys.iter().enumerate() {
            for r in 0..81 {
                for j in 0..8 {
                    b.v[r][8 * p + j] = poly.coeff(r + 81 * j) as i16;
                }
            }
        }
        b
    }
}

// ---------------------------------------------------------------------------------------------
// Compile-time bound model (units of q, fixed point with scale BSCALE)
// ---------------------------------------------------------------------------------------------

/// Fixed-point scale of the bound model: a bound `b` means `|x| <= b * q / BSCALE`.
pub const BSCALE: u64 = 1 << 20;

/// `|mont(a, w)| < |a|*q/2^17 + q/2` for `|w| <= q/2`: the output bound of a twiddle product whose
/// input is bounded by `b`.
const fn mont_bound(b: u64, q: u16) -> u64 {
    b * q as u64 / (1u64 << 17) + BSCALE / 2
}

/// Exact max `|barrett_i16(a, q)|` over all i16 `a` (the `vpmulhrsw` form), in bound units.
const fn barrett_bound(q: u16) -> u64 {
    let v = barrett_v(q) as i32;
    let mut a = -32768i32;
    let mut mx = 0i32;
    while a < 32768 {
        let t = (a * v * 2 + (1 << 15)) >> 16;
        let mut r = a - t * q as i32;
        if r < 0 {
            r = -r;
        }
        if r > mx {
            mx = r;
        }
        a += 1;
    }
    (mx as u64 * BSCALE).div_ceil(q as u64)
}

/// `bounds[l]` = bound on every lane after level `l-1` (so `bounds[0]` = input bound = q,
/// `bounds[7]` = output bound), for the variant that does / does not `barrett` the `a0` input of
/// every radix-3 level.
pub const fn bounds_for(q: u16, bar: bool) -> [u64; 8] {
    let mut b = [0u64; 8];
    b[0] = BSCALE;
    let mut l = 0;
    // levels 0, 1, 2: out = v + c*u, both bounded by b[l].
    while l < 3 {
        b[l + 1] = b[l] + mont_bound(b[l], q);
        l += 1;
    }
    // levels 3..6: t1, t2 = mont(a_i, zeta^i), u = mont(t1 - t2, omega),
    // y0 = a0 + t1 + t2, y1 = a0 - t2 + u, y2 = a0 - t1 - u.
    while l < 7 {
        let m = mont_bound(b[l], q);
        let u = mont_bound(2 * m, q);
        let a0 = if bar { barrett_bound(q) } else { b[l] };
        let y0 = a0 + 2 * m;
        let y1 = a0 + m + u;
        b[l + 1] = if y0 > y1 { y0 } else { y1 };
        l += 1;
    }
    b
}

const fn fits_i16(q: u16, b: u64) -> bool {
    b * q as u64 <= 32767 * BSCALE
}

/// Per-prime compile-time constants: the twiddle tables and the bound decisions.
pub struct HTables<const Q: u16>;

impl<const Q: u16> HTables<Q> {
    /// Whether the radix-3 levels must `barrett` their `a0` input to stay inside i16.
    pub const BARRETT: bool = !fits_i16(Q, bounds_for(Q, false)[7]);
    /// Per-level bounds in units of q, scale `BSCALE`.
    pub const BOUNDS: [u64; 8] = bounds_for(Q, Self::BARRETT);
    /// Output bound as an absolute value: `|out| <= OUT_ABS`.
    pub const OUT_ABS: i32 = (Self::BOUNDS[7] * Q as u64).div_ceil(BSCALE) as i32;
    const _CHECK: () = assert!(fits_i16(Q, Self::BOUNDS[7]), "output does not fit in i16");
    /// The twiddle / shuffle constants.
    pub const T: Tables = build_tables::<Q>();
}

/// Output bound of `ntt_gen_hbatch4::<Q>`: every lane satisfies `|v| <= out_bound::<Q>()`.
pub fn out_bound<const Q: u16>() -> i32 {
    () = HTables::<Q>::_CHECK;
    HTables::<Q>::OUT_ABS
}

/// Per-level bounds in units of q (scale `BSCALE`), `[input, after L0, ..., after L6]`.
pub fn level_bounds<const Q: u16>() -> [u64; 8] {
    HTables::<Q>::BOUNDS
}

/// Whether this prime needs the per-level `barrett`.
pub fn uses_barrett<const Q: u16>() -> bool {
    HTables::<Q>::BARRETT
}

// ---------------------------------------------------------------------------------------------
// Constant tables
// ---------------------------------------------------------------------------------------------

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct V16(pub [i16; 32]);

/// A per-lane twiddle vector in Montgomery form together with its `q^-1` companion, so that a
/// twiddle product is mullo + mulhi + mulhi + sub (3 multiply uops).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Tw {
    pub w: V16,
    pub wp: V16,
}

const ZERO_TW: Tw = Tw { w: V16([0; 32]), wp: V16([0; 32]) };

/// All compile-time vectors of the kernel, in one aligned block.
#[repr(C, align(64))]
pub struct Tables {
    /// Level 0: `zeta6` in `j < 4`, `zeta6^-1 = 1 - zeta6` in `j >= 4`.
    pub l0: Tw,
    /// Level 1: `+-ZETA_L1[j/4]`.
    pub l1: Tw,
    /// Level 2: `+-ZETA_L2[j/2]`.
    pub l2: Tw,
    /// `omega` (all lanes).
    pub om: Tw,
    /// Radix-3 levels: `[level][r / DEGREE[level]][0] = zeta`, `[..][1] = zeta^2`.
    pub l3: [[Tw; 2]; 1],
    pub l4: [[Tw; 2]; 3],
    pub l5: [[Tw; 2]; 9],
    pub l6: [[Tw; 2]; 27],
    /// `round(2^15/q)` for the `vpmulhrsw` Barrett, and `q`.
    pub bar: V16,
    pub q: V16,
    /// `vpshufb` controls for level 2: `u[j] = a[j|1]`, `v[j] = a[j & !1]`.
    pub sh_odd: V16,
    pub sh_even: V16,
}

const fn mk_tw<const Q: u16>(vals: [u16; 8]) -> Tw {
    let mut w = [0i16; 32];
    let mut wp = [0i16; 32];
    let mut i = 0;
    while i < 32 {
        let x = Params::<Q>::to_mont(vals[i & 7]);
        w[i] = x;
        wp[i] = Params::<Q>::mont_pre(x);
        i += 1;
    }
    Tw { w: V16(w), wp: V16(wp) }
}

const fn negq(x: u16, q: u16) -> u16 {
    if x == 0 {
        0
    } else {
        q - x
    }
}

/// Twiddles of radix-3 level `level` for the sub-block `b` (= `r / DEGREE[level]`), lane `j`
/// carrying sub-ring `k = j*nb + b`; `sq` selects `zeta^2`.
const fn r3_vals<const Q: u16>(level: usize, nb: usize, b: usize, sq: bool) -> [u16; 8] {
    let mut o = [0u16; 8];
    let mut j = 0;
    while j < 8 {
        let z = Params::<Q>::zeta(level, j * nb + b) as u64;
        o[j] = if sq { (z * z % Q as u64) as u16 } else { z as u16 };
        j += 1;
    }
    o
}

const fn r3_level<const Q: u16, const NB: usize>(level: usize) -> [[Tw; 2]; NB] {
    let mut out = [[ZERO_TW; 2]; NB];
    let mut b = 0;
    while b < NB {
        out[b][0] = mk_tw::<Q>(r3_vals::<Q>(level, NB, b, false));
        out[b][1] = mk_tw::<Q>(r3_vals::<Q>(level, NB, b, true));
        b += 1;
    }
    out
}

const fn build_tables<const Q: u16>() -> Tables {
    let z6 = Params::<Q>::ZETA6;
    let z6i = (1 + Q - z6) % Q; // 1 - zeta6 = zeta6^-1

    let mut v0 = [0u16; 8];
    let mut v1 = [0u16; 8];
    let mut v2 = [0u16; 8];
    let mut j = 0;
    while j < 8 {
        v0[j] = if j < 4 { z6 } else { z6i };
        let z1 = Params::<Q>::ZETA_L1[j / 4];
        v1[j] = if j % 4 < 2 { z1 } else { negq(z1, Q) };
        let z2 = Params::<Q>::ZETA_L2[j / 2];
        v2[j] = if j % 2 == 0 { z2 } else { negq(z2, Q) };
        j += 1;
    }

    let mut bar = [0i16; 32];
    let mut qv = [0i16; 32];
    let mut sh_odd = [0i16; 32];
    let mut sh_even = [0i16; 32];
    let mut i = 0;
    while i < 32 {
        bar[i] = barrett_v(Q);
        qv[i] = Q as i16;
        i += 1;
    }
    // vpshufb controls: byte 2j/2j+1 of each 128-bit lane takes bytes 2s/2s+1 with s = j|1 / j&!1.
    let mut b = 0;
    while b < 32 {
        let j = b % 8; // 16-bit element inside the 128-bit lane
        let so = (j | 1) as i16;
        let se = (j & !1) as i16;
        // sh_*[b] is a 16-bit slot = two control bytes (low = byte 2j, high = byte 2j+1).
        sh_odd[b] = (2 * so) | ((2 * so + 1) << 8);
        sh_even[b] = (2 * se) | ((2 * se + 1) << 8);
        b += 1;
    }

    Tables {
        l0: mk_tw::<Q>(v0),
        l1: mk_tw::<Q>(v1),
        l2: mk_tw::<Q>(v2),
        om: mk_tw::<Q>([Params::<Q>::OMEGA; 8]),
        l3: r3_level::<Q, 1>(3),
        l4: r3_level::<Q, 3>(4),
        l5: r3_level::<Q, 9>(5),
        l6: r3_level::<Q, 27>(6),
        bar: V16(bar),
        q: V16(qv),
        sh_odd: V16(sh_odd),
        sh_even: V16(sh_even),
    }
}

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    &HTables::<Q>::T
}

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------

#[inline(always)]
unsafe fn ldv(p: *const V16) -> __m512i {
    _mm512_load_si512(p as *const __m512i)
}

/// Signed Montgomery twiddle product with a precomputed companion: `a * x mod q` in `(-q, q)`,
/// 3 multiply uops.
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = _mm512_mulhi_epi16(a, w);
    _mm512_sub_epi16(hi, _mm512_mulhi_epi16(m, q))
}

/// `vpmulhrsw` Barrett: 2 multiply uops, `|r| < q` and `r = a mod q`.
#[inline(always)]
unsafe fn barrett(a: __m512i, bv: __m512i, q: __m512i) -> __m512i {
    let t = _mm512_mulhrs_epi16(a, bv);
    _mm512_sub_epi16(a, _mm512_mullo_epi16(t, q))
}

/// Levels 0, 1, 2 of one register: 21 ALU uops (6 shuffles on p5, 9 multiplies on p0, 3 Montgomery
/// subtractions + 3 adds).
#[inline(always)]
unsafe fn lvl012<const Q: u16>(x: __m512i, t: &Tables, q: __m512i) -> __m512i {
    // level 0: u[j] = a[j|4] (vpshufd 0xEE), v[j] = a[j&3] (vpshufd 0x44)
    let u = _mm512_shuffle_epi32::<0xEE>(x);
    let v = _mm512_shuffle_epi32::<0x44>(x);
    let x = _mm512_add_epi16(v, mont(u, ldv(&t.l0.w), ldv(&t.l0.wp), q));
    // level 1: u[j] = a[j|2] (vpshufd 0xF5), v[j] = a[j&!2] (vpshufd 0xA0)
    let u = _mm512_shuffle_epi32::<0xF5>(x);
    let v = _mm512_shuffle_epi32::<0xA0>(x);
    let x = _mm512_add_epi16(v, mont(u, ldv(&t.l1.w), ldv(&t.l1.wp), q));
    // level 2: u[j] = a[j|1], v[j] = a[j&!1] (vpshufb with constant controls)
    let u = _mm512_shuffle_epi8(x, ldv(&t.sh_odd));
    let v = _mm512_shuffle_epi8(x, ldv(&t.sh_even));
    _mm512_add_epi16(v, mont(u, ldv(&t.l2.w), ldv(&t.l2.wp), q))
}

/// The loop-invariant vectors of one radix-3 pass: zeta, zeta^2 (both with their Montgomery
/// companions), omega, the Barrett constant and q, hoisted out of the loop by hand (LLVM otherwise
/// rebuilds them from the constant pool inside the loop).
#[derive(Clone, Copy)]
struct R3C {
    zw: __m512i,
    zwp: __m512i,
    z2w: __m512i,
    z2wp: __m512i,
    ow: __m512i,
    owp: __m512i,
    bar: __m512i,
    q: __m512i,
}

#[inline(always)]
unsafe fn r3c(zt: &[Tw; 2], t: &Tables, q: __m512i) -> R3C {
    R3C {
        zw: ldv(&zt[0].w),
        zwp: ldv(&zt[0].wp),
        z2w: ldv(&zt[1].w),
        z2wp: ldv(&zt[1].wp),
        ow: ldv(&t.om.w),
        owp: ldv(&t.om.wp),
        bar: ldv(&t.bar),
        q,
    }
}

/// One radix-3 butterfly: 19 ALU uops (9 multiplies on p0, 10 adds/subtractions), 21 with the
/// Barrett on `a0`.
#[inline(always)]
unsafe fn r3<const BAR: bool>(
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    c: &R3C,
) -> (__m512i, __m512i, __m512i) {
    let t1 = mont(a1, c.zw, c.zwp, c.q);
    let t2 = mont(a2, c.z2w, c.z2wp, c.q);
    let u = mont(_mm512_sub_epi16(t1, t2), c.ow, c.owp, c.q);
    let a0 = if BAR { barrett(a0, c.bar, c.q) } else { a0 };
    (
        _mm512_add_epi16(_mm512_add_epi16(a0, t1), t2),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// One radix-3 triple `(base, base+m, base+2m)`.
macro_rules! r3_at {
    ($p:expr, $c:expr, $BAR:expr, $base:expr, $m:expr) => {{
        let b = $p.add($base);
        let (y0, y1, y2) = r3::<$BAR>(
            _mm512_load_si512(b),
            _mm512_load_si512(b.add($m)),
            _mm512_load_si512(b.add(2 * $m)),
            $c,
        );
        _mm512_store_si512(b, y0);
        _mm512_store_si512(b.add($m), y1);
        _mm512_store_si512(b.add(2 * $m), y2);
    }};
}

/// One level per pass, three independent triples per iteration: fusing two levels into one pass
/// costs ~5% (the second level waits for the whole first stage, while separate passes keep three
/// independent 24-cycle chains in flight and the extra L1 round trip is free at 5 KB).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_inner<const Q: u16, const BAR: bool>(bat: &mut HBatch4, pf: *const i8) {
    let t = tables::<Q>();
    let q = ldv(&t.q);
    let p = bat.v.as_mut_ptr() as *mut __m512i;

    for r in 0..81 {
        _mm_prefetch::<_MM_HINT_T0>(pf.add(64 * r));
        _mm512_store_si512(p.add(r), lvl012::<Q>(_mm512_load_si512(p.add(r)), t, q));
    }
    let c = r3c(&t.l3[0], t, q);
    for i in (0..27).step_by(3) {
        r3_at!(p, &c, BAR, i, 27);
        r3_at!(p, &c, BAR, i + 1, 27);
        r3_at!(p, &c, BAR, i + 2, 27);
    }
    for b4 in 0..3 {
        let c = r3c(&t.l4[b4], t, q);
        for i in (0..9).step_by(3) {
            r3_at!(p, &c, BAR, 27 * b4 + i, 9);
            r3_at!(p, &c, BAR, 27 * b4 + i + 1, 9);
            r3_at!(p, &c, BAR, 27 * b4 + i + 2, 9);
        }
    }
    for b5 in 0..9 {
        let c = r3c(&t.l5[b5], t, q);
        r3_at!(p, &c, BAR, 9 * b5, 3);
        r3_at!(p, &c, BAR, 9 * b5 + 1, 3);
        r3_at!(p, &c, BAR, 9 * b5 + 2, 3);
    }
    for k in 0..9 {
        for s in 0..3 {
            let c = r3c(&t.l6[3 * k + s], t, q);
            r3_at!(p, &c, BAR, 9 * k + 3 * s, 1);
        }
    }
}

/// Forward NTT of 4 polynomials in place, `Coefficients` -> `Ntt` (tree order, see the module
/// docs). Input lanes must satisfy `|v| <= q`; output lanes satisfy `|v| <= out_bound::<Q>()`.
///
/// # Safety
/// Requires AVX-512 F/BW/VL (the target machine).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_gen_hbatch4<const Q: u16>(b: &mut HBatch4) {
    let pf = b.v.as_ptr() as *const i8;
    if HTables::<Q>::BARRETT {
        ntt_inner::<Q, true>(b, pf);
    } else {
        ntt_inner::<Q, false>(b, pf);
    }
}

/// Driver over many batches (in place).
///
/// # Safety
/// Requires AVX-512 F/BW/VL.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_gen_hbatch4_many<const Q: u16>(bs: &mut [HBatch4]) {
    // Software prefetch of the next batch, one line per register of the first pass: the hardware
    // prefetcher alone leaves the 5 KB first pass exposed to DRAM latency (770 -> 590 cyc/poly).
    let (n, base) = (bs.len(), bs.as_ptr() as *const i8);
    for (i, b) in bs.iter_mut().enumerate() {
        let pf = base.add(core::mem::size_of::<HBatch4>() * if i + 1 < n { i + 1 } else { i });
        if HTables::<Q>::BARRETT {
            ntt_inner::<Q, true>(b, pf);
        } else {
            ntt_inner::<Q, false>(b, pf);
        }
    }
}

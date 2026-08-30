//! Forward NTT for **binary** inputs on the splitting tree for `q` in [`QS_LARGE`], in the
//! vertical batch-of-32 layout.
//!
//! The tree, the lookup tables and the block layout are
//! [`crate::simd::vertical_bin`]'s — 648 linear slots, levels 0, 1, 2 and the level-3 twiddles
//! folded into 96 byte-split 16-entry tables, then four radix-3 levels, handed out as 24 blocks
//! of 27 rows through [`BlockSink`]. What differs is the arithmetic budget.
//!
//! # Why a separate kernel
//!
//! A lane is an i16, so the whole transform lives inside `2^15 / q`: 8.43 at q = 3889 and 3.37 at
//! q = 9721, but only **1.873** at q = 17497 and **1.686** at q = 19441. A radix-3 butterfly
//! forms `a0 + t1 + t2`, and a Montgomery twiddle product never gets below `q/2` — the
//! `mulhi(m, q)` term is a full i16 times `q` — so three terms already cost `1.5 q` before any
//! slack. There is no schedule of the existing kernels, which reduce only the untwiddled `a0`,
//! that fits: at 19441 the un-Barretted level 3 alone leaves `3 q = 58323`.
//!
//! Everything a butterfly adds therefore has to be reduced, not just `a0`, and the reduction has
//! to be the shuffle-port **lookup Barrett** of
//! [`crate::simd::vertical_bin_asm::barrett_lut_i16`] (`vpmultishiftqb` + `vpandd` + `vpord` +
//! `vpermb` + `vpaddw`: 2 port-5 and 3 flexible uops, **not one multiply-port slot**), because
//! the multiply port is what the kernel is already bound by and the two-multiply `vpmulhrsw`
//! Barrett is useless up here anyway — `round(2^15/q)` is 2 for both primes, an estimate with two
//! significant bits.
//!
//! # The unsigned alternative, and why this one
//!
//! The natural other answer is to leave the signed representation: keep the lanes in `[0, q)` as
//! u16, where the head-room is `2^16/q` = 3.75 / 3.37 and a radix-3 sum of three reduced values
//! fits with room. A butterfly is then 3 Shoup products with a conditional subtract
//! (`vpmullw` + `vpmulhuw` + `vpmullw` + `vpsubw` + `vpsubw` + `vpminuw`, 6 uops of which 3 on
//! the multiply port), three negations `q - t` so that the two twiddled outputs stay sums rather
//! than differences, six adds and one unsigned lookup Barrett on `a0`: 33 uops, 9 on port 0 and 2
//! on port 5, against the signed schedule's 34 (q = 17497) and 39 (q = 19441) with the same 9 on
//! port 0 and 6 or 8 on port 5.
//!
//! Measured over the same 216-butterfly level, both written out in intrinsics
//! (`tests/vertical_bin_large.rs::the_unsigned_alternative`), the signed one wins anyway:
//! **3.89 ns per butterfly against the unsigned form's 6.10 at q = 17497, and 4.51 against 6.20
//! at q = 19441** — 1.6x and 1.4x. The uop counts are a wash and the dependency chains are
//! not: a Shoup product is
//! `vpmullw -> vpmullw -> vpsubw -> vpsubw -> vpminuw` deep where a Montgomery one is
//! `vpmullw -> vpmulhw -> vpsubw`, and the unsigned butterfly serialises three of them behind
//! each other through `t1 - t2`.
//!
//! And the unsigned form has no output the rest of the crate can use. `Batch32` is i16, `A` is
//! centered, the base multiplication is `vpdpwssd` on signed pairs, and the fold, the
//! decomposition and every bound in [`crate::simd::commit`] are written on centered lanes; an
//! unsigned transform would have to be centered before any of that, a pass over the 648 output
//! vectors of every batch on top of a per-butterfly loss. Signed it is.
//!
//! # The schedule
//!
//! Per level, the flags are `(a0, t12, u)`: Barrett the untwiddled `a0` input, the two twiddle
//! products `t1`, `t2`, and the `omega (t1 - t2)` of the radix-3 butterfly. [`bin_sched`] tries
//! every one of the 4096 placements over the four radix-3 levels against the exact bound
//! recursion [`bin_model`] and keeps the cheapest that stays inside i16, so the reduction points
//! are a compile-time consequence of `q` and not a table:
//!
//! | after            | q = 17497 | q = 19441 |
//! |------------------|----------:|----------:|
//! | levels 0+1+2     |    1.00 q |    1.00 q |
//! | level 3          |    1.71 q |    1.58 q |
//! | level 4          |    1.71 q |    1.58 q |
//! | level 5          |    1.71 q |    1.58 q |
//! | level 6 (output) |    1.71 q |    1.58 q |
//!
//! There is no `asm!` tail here as there is in [`crate::simd::vertical_bin_asm`]. That kernel
//! hand-schedules levels 4, 5 and 6 into one block per 27-slot block because 27 resident data
//! registers plus its three constants exactly fill the file; this one needs seven constants (the
//! lookup Barrett's four on top of `q`, omega and its companion) and three or four reductions per
//! butterfly, so the register-resident tail does not exist to be written. It is the intrinsics
//! form of the same tree, block for block and sink for sink, which is also what
//! [`crate::simd::vertical_bin_quad`] does at 247 to 291 cycles per ring element.
//!
//! 17497 reduces `a0`, `t1` and `t2` — three lookup Barretts per butterfly, 2592 per batch —
//! and can leave `u` alone because `a0 + t2 + u` at `0.53 q + 0.53 q + 0.64 q` still clears
//! 1.873 q. 19441 cannot (that same sum is 1.77 q against a 1.686 q budget) and pays a fourth,
//! 3456 per batch. The `t1`, `t2` reduction is what earns its keep: without it the level-3 sum
//! `t1 + t2` of two table sums is `2 q`, and no amount of reducing `a0` brings `a0 + t1 + t2`
//! under 2.5 q.
use crate::params::*;
use crate::simd::vertical_bin_asm::barrett_lut_corr;
pub use crate::simd::vertical_bin_asm::BlockSink;
pub use crate::simd::transpose_f162::BinaryIndex32;
use crate::types::*;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// bounds and the reduction schedule
// ---------------------------------------------------------------------------------------------

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`, per prime — an exhaustive sweep, evaluated
/// once (`q/2 + 2^10` is the theoretical bound; the sweep is a little tighter).
const BLM: [i32; 2] = [barrett_lut_sweep(QS_LARGE[0]), barrett_lut_sweep(QS_LARGE[1])];

const fn qi(q: u16) -> usize {
    if q == QS_LARGE[0] {
        0
    } else {
        1
    }
}

/// Does `q` run this kernel rather than [`crate::simd::vertical_bin_asm`]?
pub const fn is_large(q: u16) -> bool {
    q == QS_LARGE[0] || q == QS_LARGE[1]
}

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`.
pub const fn barrett_lut_max(q: u16) -> i32 {
    BLM[qi(q)]
}

const fn barrett_lut_sweep(q: u16) -> i32 {
    let mut m = 0i32;
    let mut a = -32768i32;
    while a < 32768 {
        let r = ((a + barrett_lut_corr(((a >> 11) & 31) as usize, q) as i32) as i16) as i32;
        let r = if r < 0 { -r } else { r };
        if r > m {
            m = r;
        }
        a += 1;
    }
    m
}

/// `|mont(a, w)| <= |a| q / 2^17 + q/2` (`params::mont_mul_i16`, |w| <= q/2).
const fn mont_bound(b: i32, q: u16) -> i32 {
    ((b as i64 * q as i64) >> 17) as i32 + (q as i32 + 1) / 2
}

/// Is flag `i` of level `3 + l` set in the placement mask? `i` = 0 reduces the untwiddled `a0`,
/// 1 the two twiddle products, 2 the `omega (t1 - t2)`.
pub const fn bar_flag(mask: u32, l: usize, i: usize) -> bool {
    (mask >> (3 * l + i)) & 1 == 1
}

/// The kernel's schedule replayed on bounds: the maximum |lane| after the fused lookups and
/// after each of levels 3, 4, 5, 6, and the largest intermediate ever formed (which is what has
/// to stay inside i16).
///
/// Every position of a level carries the same bound — the tree is uniform and the flags are
/// per level — so the recursion is the scalar one and its per-level maximum is exact.
pub const fn bin_model(q: u16, mask: u32) -> ([i32; 5], i32) {
    let r = barrett_lut_max(q);
    let h = (q as i32 + 1) / 2;
    let mut lm = [0i32; 5];
    let mut v = 2 * h;
    let mut peak = v;
    lm[0] = v;
    let mut l = 0;
    while l < 4 {
        // level 3's twiddles are folded into the lookup tables, so its `t1`, `t2` are table sums.
        let mut t = if l == 0 { v } else { mont_bound(v, q) };
        if bar_flag(mask, l, 1) && r < t {
            t = r;
        }
        if 2 * t > peak {
            peak = 2 * t;
        }
        let mut u = mont_bound(2 * t, q);
        if bar_flag(mask, l, 2) && r < u {
            u = r;
        }
        let b0 = if bar_flag(mask, l, 0) && r < v { r } else { v };
        let (o0, o1) = (b0 + 2 * t, b0 + t + u);
        if o0 > peak {
            peak = o0;
        }
        if o1 > peak {
            peak = o1;
        }
        v = if o0 > o1 { o0 } else { o1 };
        lm[l + 1] = v;
        l += 1;
    }
    (lm, peak)
}

/// Lookup Barretts one butterfly of the mask costs.
const fn bar_cost(mask: u32) -> u32 {
    let mut c = 0;
    let mut l = 0;
    while l < 4 {
        c += bar_flag(mask, l, 0) as u32 + 2 * bar_flag(mask, l, 1) as u32
            + bar_flag(mask, l, 2) as u32;
        l += 1;
    }
    c
}

/// The cheapest placement that keeps every intermediate inside i16, and the output bound it
/// leaves. Ties on cost go to the tighter output.
const fn bin_sched(q: u16) -> (u32, i32) {
    let (mut best, mut cost, mut out) = (u32::MAX, u32::MAX, i32::MAX);
    let mut mask = 0u32;
    while mask < 1 << 12 {
        let (lm, peak) = bin_model(q, mask);
        if peak <= 32767 {
            let c = bar_cost(mask);
            if c < cost || (c == cost && lm[4] < out) {
                best = mask;
                cost = c;
                out = lm[4];
            }
        }
        mask += 1;
    }
    assert!(best != u32::MAX, "no i16 schedule for this prime");
    (best, out)
}

const BIN_SCHED: [(u32, i32); 2] = [bin_sched(QS_LARGE[0]), bin_sched(QS_LARGE[1])];

/// Which of `a0`, `(t1, t2)` and `u` each of levels 3, 4, 5, 6 reduces.
pub const fn bar_levels(q: u16) -> u32 {
    BIN_SCHED[qi(q)].0
}

/// Declared output bound: max |lane| of [`ntt_bin_batch32`].
pub const fn output_bound(q: u16) -> i32 {
    BIN_SCHED[qi(q)].1
}

const _: () = {
    let mut i = 0;
    while i < 2 {
        let q = QS_LARGE[i];
        assert!(bin_model(q, bar_levels(q)).1 <= 32767);
        // three reductions per butterfly at 17497, four at 19441, at every one of the four levels
        assert!(bar_cost(bar_levels(q)) == if q == 17497 { 12 } else { 16 });
        i += 1;
    }
};
// Reducing only the untwiddled `a0`, which is all the kernels below 2^14 ever do, does not fit.
const _: () = assert!(bin_model(QS_LARGE[0], 0o1111).1 > 32767);
const _: () = assert!(bin_model(QS_LARGE[1], 0o1111).1 > 32767);

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[((k * 2 + s2) * 3 + r) * 2 + ab]`: the 16 centered i16 values of
    /// `base_k(n) * zeta''_{2k+s2}^r * (ab == 1 ? zeta'_k : 1)`, **byte-split** so that a single
    /// `vpermb` (1 uop, port 5) does the lookup: byte n is the low half of entry n, byte 16+n the
    /// high half.
    lut: [[u8; 64]; 96],
    /// 512-bit constants of the lookup Barrett: `[ms, corr, and, or]` — the `vpmultishiftqb`
    /// control, the byte-split `-k q` table and the index fix-up masks.
    cv: [[i16; 32]; 4],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2), each i16
    /// duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw4: [[u32; 4]; 24],
    tw5: [[u32; 4]; 72],
    tw6: [[u32; 4]; 216],
    /// omega and its companion.
    om: [u32; 2],
    /// q, duplicated.
    qd: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

const fn r3_pair<const Q: u16>(z: u16) -> [u32; 4] {
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let (a, b) = mont_pair::<Q>(z);
    let (c, d) = mont_pair::<Q>(z2);
    [a, b, c, d]
}

/// Byte `u` of the 64-byte `vpermb` correction table: the low halves of `-k(s) q` at u = s < 32,
/// the high halves at u = 32 + s.
pub const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = Params::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    let mut lut = [[0u8; 64]; 96];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        let mut s2 = 0;
        while s2 < 2 {
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            let mut r = 0;
            while r < 3 {
                let f = pow_mod(z3, r as u64, q);
                let mut ab = 0;
                while ab < 2 {
                    let extra = if ab == 0 { 1 } else { zp };
                    let mut n = 0;
                    while n < 16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let e = center(base * f % q * extra % q, q) as u16;
                        let ti = ((k * 2 + s2) * 3 + r) * 2 + ab;
                        lut[ti][n] = e as u8;
                        lut[ti][16 + n] = (e >> 8) as u8;
                        n += 1;
                    }
                    ab += 1;
                }
                r += 1;
            }
            s2 += 1;
        }
        k += 1;
    }

    let mut tw4 = [[0u32; 4]; 24];
    let mut i = 0;
    while i < 24 {
        tw4[i] = r3_pair::<Q>(Params::<Q>::ZETA_L4[i]);
        i += 1;
    }
    let mut tw5 = [[0u32; 4]; 72];
    let mut i = 0;
    while i < 72 {
        tw5[i] = r3_pair::<Q>(Params::<Q>::ZETA_L5[i]);
        i += 1;
    }
    let mut tw6 = [[0u32; 4]; 216];
    let mut i = 0;
    while i < 216 {
        tw6[i] = r3_pair::<Q>(Params::<Q>::ZETA_L6[i]);
        i += 1;
    }
    let (oa, ob) = mont_pair::<Q>(Params::<Q>::OMEGA);

    let mut cv = [[0i16; 32]; 4];
    let mut i = 0;
    while i < 32 {
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that word.
        cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
        let (b0, b1) = (lut_byte(2 * i, Q), lut_byte(2 * i + 1, Q));
        cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[2][i] = 0x1f1f;
        cv[3][i] = 0x2000;
        i += 1;
    }
    Tables { lut, cv, tw4, tw5, tw6, om: [oa, ob], qd: dup(Q as i16) }
}

static T17497: Tables = build_tables::<17497>();
static T19441: Tables = build_tables::<19441>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    if Q == QS_LARGE[0] {
        &T17497
    } else {
        &T19441
    }
}

// ---------------------------------------------------------------------------------------------
// arithmetic helpers
// ---------------------------------------------------------------------------------------------

/// `vpbroadcastd zmm, m32` - one pure load uop. Written as `asm!` because LLVM otherwise
/// "recognises" the duplicated-u32 splat and rebuilds it with vpmovsxwd/vpmovdw/vinserti64x4.
#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// The four twiddle broadcasts of one sub-ring off a single base register.
#[inline(always)]
unsafe fn bc4(p: *const u32) -> (__m512i, __m512i, __m512i, __m512i) {
    let (a, b, c, d): (__m512i, __m512i, __m512i, __m512i);
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{4}]",
        "vpbroadcastd {1}, dword ptr [{4} + 4]",
        "vpbroadcastd {2}, dword ptr [{4} + 8]",
        "vpbroadcastd {3}, dword ptr [{4} + 12]",
        out(zmm_reg) a,
        out(zmm_reg) b,
        out(zmm_reg) c,
        out(zmm_reg) d,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    (a, b, c, d)
}

/// `vpmulhw`. stdarch's `_mm512_mulhi_epi16` is written as sext -> mul -> shr -> trunc; LLVM
/// folds the multiply but leaves a `vpmovsxwd`/`vpmovdw`/`vinserti64x4` round trip on operands it
/// cannot see through (broadcast constants), and rematerialises it inside the hot loops.
#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpmulhw {0}, {1}, {2}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// 3-uop signed Montgomery twiddle multiply: a * x mod q in (-q, q).
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm512_sub_epi16(hi, t)
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

/// The shuffle-port lookup Barrett: 2 port-5 + 3 flexible uops, no multiply-port slot,
/// `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn barrett_lut(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
}

#[inline(always)]
unsafe fn red(a: __m512i, c: &C, on: bool) -> __m512i {
    if on {
        barrett_lut(a, c)
    } else {
        a
    }
}

/// The radix-3 butterfly of a level whose twiddles are already in `t1`, `t2` (level 3, where the
/// tables carry them), reducing whichever of `a0`, `(t1, t2)` and `u` the schedule names.
#[inline(always)]
unsafe fn r3_folded(
    c: &C,
    a0: __m512i,
    t1: __m512i,
    t2: __m512i,
    bar: [bool; 3],
) -> (__m512i, __m512i, __m512i) {
    let t1 = red(t1, c, bar[1]);
    let t2 = red(t2, c, bar[1]);
    let u = red(mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q), c, bar[2]);
    let a0 = red(a0, c, bar[0]);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// The same butterfly with the twiddles `[w, w', w2, w2']` applied first.
#[inline(always)]
unsafe fn r3(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
    bar: [bool; 3],
) -> (__m512i, __m512i, __m512i) {
    let (w1, w1p, w2, w2p) = bc4(tw);
    let t1 = mont(a1, w1, w1p, c.q);
    let t2 = mont(a2, w2, w2p, c.q);
    r3_folded(c, a0, t1, t2, bar)
}

#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m512i {
    _mm512_load_si512(p.add(64 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn ld(p: *const i16, j: usize) -> __m512i {
    _mm512_load_si512(p.add(32 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn st(p: *mut i16, j: usize, v: __m512i) {
    _mm512_store_si512(p.add(32 * j) as *mut __m512i, v);
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);

/// The per-prime schedule as compile-time constants of the kernel: a `const fn` call on its own
/// is not folded before instruction selection, an associated const is.
struct Sched<const Q: u16>;

impl<const Q: u16> Sched<Q> {
    const BAR: [[bool; 3]; 4] = {
        let m = bar_levels(Q);
        let mut b = [[false; 3]; 4];
        let mut l = 0;
        while l < 4 {
            b[l] = [bar_flag(m, l, 0), bar_flag(m, l, 1), bar_flag(m, l, 2)];
            l += 1;
        }
        b
    };
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_core<const Q: u16, S: BlockSink>(input: &BinaryIndex32, sink: &mut S) {
    let t = tables::<Q>();
    let cvp = t.cv.as_ptr() as *const __m512i;
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
        ms: _mm512_load_si512(cvp),
        corr: _mm512_load_si512(cvp.add(1)),
        andm: _mm512_load_si512(cvp.add(2)),
        orm: _mm512_load_si512(cvp.add(3)),
    };
    let bar = Sched::<Q>::BAR;

    // The caller already holds the `vpermb` byte-index rows; the kernel reads them straight.
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;

    for k in 0..4 {
        let lut = t.lut.as_ptr().add(12 * k) as *const u8;
        let l = |s2: usize, r: usize, ab: usize| -> __m512i {
            _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
        };
        let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
        let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
        let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));
        let (l110, l111) = (l(1, 1, 0), l(1, 1, 1));
        let (l120, l121) = (l(1, 2, 0), l(1, 2, 1));

        // levels 0+1+2 (table lookups) fused with level 3 (omega multiply only).
        for i in 0..27 {
            let n0 = ldb(ip, i);
            let n0h = ldb(ip, i + 81);
            let n1 = ldb(ip, i + 27);
            let n1h = ldb(ip, i + 108);
            let n2 = ldb(ip, i + 54);
            let n2h = ldb(ip, i + 135);

            let x = _mm512_permutexvar_epi8(n0, l000);
            let y = _mm512_permutexvar_epi8(n0h, l001);
            let a0 = _mm512_add_epi16(x, y);
            let b0 = _mm512_sub_epi16(x, y);

            let x = _mm512_permutexvar_epi8(n1, l010);
            let y = _mm512_permutexvar_epi8(n1h, l011);
            let a1 = _mm512_add_epi16(x, y);
            let x = _mm512_permutexvar_epi8(n1, l110);
            let y = _mm512_permutexvar_epi8(n1h, l111);
            let b1 = _mm512_sub_epi16(x, y);

            let x = _mm512_permutexvar_epi8(n2, l020);
            let y = _mm512_permutexvar_epi8(n2h, l021);
            let a2 = _mm512_add_epi16(x, y);
            let x = _mm512_permutexvar_epi8(n2, l120);
            let y = _mm512_permutexvar_epi8(n2h, l121);
            let b2 = _mm512_sub_epi16(x, y);

            let (u0, u1, u2) = r3_folded(&c, a0, a1, a2, bar[0]);
            let (v0, v1, v2) = r3_folded(&c, b0, b1, b2, bar[0]);
            st(bp, i, u0);
            st(bp, i + 27, u1);
            st(bp, i + 54, u2);
            st(bp, i + 81, v0);
            st(bp, i + 108, v1);
            st(bp, i + 135, v2);
        }

        // levels 4, 5 and 6, one 27-block at a time: level 4 over the L1 scratch, then levels 5
        // and 6 fused over the nine values of a degree-9 sub-ring, straight to the sink.
        for j in 0..6 {
            let kk = 6 * k + j;
            let op = sink.dst(kk);
            let base = 27 * j;
            let t4 = t.tw4[kk].as_ptr();
            for i in 0..9 {
                let (b0, b1, b2) = (base + i, base + i + 9, base + i + 18);
                let (o0, o1, o2) =
                    r3(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4, bar[1]);
                st(bp, b0, o0);
                st(bp, b1, o1);
                st(bp, b2, o2);
            }
            for g in 0..3 {
                let t5 = t.tw5[3 * kk + g].as_ptr();
                let b = base + 9 * g;
                let mut v = [_mm512_setzero_si512(); 9];
                for i in 0..9 {
                    v[i] = ld(bp, b + i);
                }
                for i in 0..3 {
                    let (o0, o1, o2) = r3(&c, v[i], v[3 + i], v[6 + i], t5, bar[2]);
                    v[i] = o0;
                    v[3 + i] = o1;
                    v[6 + i] = o2;
                }
                for i in 0..3 {
                    let t6 = t.tw6[9 * kk + 3 * g + i].as_ptr();
                    let (o0, o1, o2) =
                        r3(&c, v[3 * i], v[3 * i + 1], v[3 * i + 2], t6, bar[3]);
                    let o = 9 * g + 3 * i;
                    st(op, o, o0);
                    st(op, o + 1, o1);
                    st(op, o + 2, o2);
                }
            }
            sink.block(kk, op);
        }
    }
}

/// Forward NTT of 32 binary polynomials: `out.v[j][p] = a_p(psi^SLOT_EXP[j]) mod q`, lazily
/// reduced (`|lane| <= output_bound(Q)`).
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `out` is 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    let mut sink = crate::simd::vertical_bin_asm::OutSink(out.v.as_mut_ptr() as *mut i16);
    ntt_core::<Q, _>(input, &mut sink);
    out.representation = Representation::Ntt;
}

/// The same transform with the output handed to `sink` 27 rows at a time instead of being written
/// to a `Batch32`, for consumers that want each block while it is still in L1.
///
/// # Safety
/// See [`BlockSink`]: `sink.dst` must give 27 writable 64-byte aligned vectors per block.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32_sink<const Q: u16, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
    ntt_core::<Q, S>(input, sink);
}

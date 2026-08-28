//! Forward NTT for **binary** inputs on the *quadratic-slot* tree (q in `params::QS_QUAD`), in
//! the vertical batch-of-32 layout.
//!
//! For q = 1 mod 972 but not mod 1944 the ring does not split completely: `Phi_1944` factors into
//! 324 irreducible quadratics and the transform ends at `Z_q[X]/(X^2 - psi'^u)` leaves (see the
//! quadratic-slot section of `params`). The tree is the splitting one with its *second* radix-2
//! level removed — Phi_6, one radix-2 level, then four radix-3 levels 162 -> 54 -> 18 -> 6 -> 2 —
//! and the kernel is the splitting binary kernel with the same removal:
//!
//! | phase                | what it does                                                       |
//! |----------------------|--------------------------------------------------------------------|
//! | lookups + levels 2, 3| 9 `vpermb` per group of 9 vectors, then 3 omega-only radix-3 (level 2) and 3 full radix-3 (level 3) |
//! | levels 4 + 5         | two passes over one 18-block of the L1 scratch, straight to the output |
//!
//! Levels 0 and 1 are a linear function of the 4-bit nibble `(b_i, b_{i+162}, b_{i+324},
//! b_{i+486})` of each polynomial, so they are one 16-entry byte-split `vpermb` lookup per output
//! vector on exactly the [`BinaryIndex32`] rows the splitting kernel consumes — the front end
//! ([`crate::simd::transpose_f162`]) is shared verbatim. Level 2 is now radix 3 (triples
//! `(i, i+54, i+108)` inside a 162-block, the role given by `i/54`), and because each level-1
//! value is consumed in exactly one role its level-2 twiddle is folded into the table: 3 tables of
//! 16 entries per 162-block, 12 in all (768 bytes), and level 2 costs only the omega
//! multiplication of its radix-3 butterfly.
//!
//! **Multiply count.** 216 omega products (levels 0-2) + 3 x 216 full radix-3 butterflies =
//! 2160 Montgomery products = 6480 multiply-port uops per batch of 32 — *exactly* what the
//! splitting binary kernel costs, since the radix-2 level that disappears is replaced by nothing
//! and the four radix-3 levels are unchanged in number of butterflies. What does drop is the
//! shuffle port (648 `vpermb` per batch against 1080: only one level of twiddle folding is
//! possible here) and the add count (no two-lookup combines).
//!
//! ## Output layout and the block hook
//!
//! `out.v[2j][p]` and `out.v[2j+1][p]` are the two coefficients of
//! `a_p mod (X^2 - psi'^QUAD_SLOT_EXP[j])`. The kernel finishes the 648 rows as **36 blocks of 18
//! consecutive rows** — block `blk` is rows `18 blk .. 18 blk + 18`, i.e. the 9 quadratic leaves
//! `9 blk .. 9 blk + 9` — each produced from 18 registers by one levels-4+5 tail. [`BlockSink`]
//! hands each block to a consumer while it is still in L1, exactly as
//! [`crate::simd::vertical_bin_asm::BlockSink`] does with its 27-row blocks.
//!
//! ## Bounds (|lane| as a multiple of q; the const recursion [`bin_model`] proves them and the
//! i32 shadow model in `tests/quad.rs` replays the schedule)
//!
//! Table entries are centered, |T| <= q/2, and `|mont(a, w)| <= |a| q/2^17 + q/2`, so a radix-3
//! butterfly adds at most 1.5 q to its untwiddled `a0` input.
//!
//! | after            | q = 2917 | q = 4861 | q = 12637 |
//! |------------------|---------:|---------:|----------:|
//! | levels 0+1+2     |   1.52 q |   1.54 q |    1.60 q |
//! | level 3          |   2.59 q |   2.65 q |    1.88 q |
//! | level 4          |   3.71 q |   3.85 q |    1.93 q |
//! | level 5 (output) |   4.87 q |   5.13 q |    1.94 q |
//!
//! `2^15/q` is 11.23 (2917), 6.74 (4861) and 2.59 (12637), so **2917 and 4861 need no reduction
//! anywhere**; 12637 gets the shuffle-port **lookup Barrett** of
//! [`crate::simd::vertical_bin_asm::barrett_lut_i16`] (`vpmultishiftqb` + `vpandd` + `vpord` +
//! `vpermb` + `vpaddw`: 2 port-5 and 3 flexible uops, not one multiply-port slot, and
//! `|r| <= q/2 + 2^10 = 0.569 q`) on the untwiddled `a0` input of each of levels 3, 4 and 5 —
//! 648 reductions per batch, none of them on the saturated port 0. All three are needed: dropping
//! any one of them overflows i16 (`bin_model` is exhaustive over the flag set).
use crate::params::*;
use crate::simd::transpose;
use crate::simd::vertical_bin_asm::barrett_lut_corr;
pub use crate::simd::transpose::BinaryIndex32;
use crate::types::*;
use bin_fields::scalar::F162;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// bounds
// ---------------------------------------------------------------------------------------------

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`, per prime — an exhaustive sweep, evaluated
/// once (`q/2 + 2^10` is the theoretical bound; the sweep is a little tighter).
const BLM: [i32; 3] = [
    barrett_lut_sweep(QS_QUAD[0]),
    barrett_lut_sweep(QS_QUAD[1]),
    barrett_lut_sweep(QS_QUAD[2]),
];

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`.
pub const fn barrett_lut_max(q: u16) -> i32 {
    if q == QS_QUAD[0] {
        BLM[0]
    } else if q == QS_QUAD[1] {
        BLM[1]
    } else {
        BLM[2]
    }
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

/// The kernel's schedule replayed on bounds, position by position: the maximum |lane| after each
/// of levels 2 (the fused lookups), 3, 4 and 5, and the largest intermediate ever formed (which
/// is what has to stay inside i16). `bar[l]` reduces the untwiddled `a0` input of level `3 + l`
/// with the lookup Barrett.
pub const fn bin_model(q: u16, bar: [bool; 3]) -> ([i32; 4], i32) {
    let r = barrett_lut_max(q);
    let h = (q as i32 + 1) / 2;
    let mut v = [0i32; N];
    let mut lm = [0i32; 4];
    let mut peak = 0i32;
    // fused lookups (|T| <= q/2 in all three roles) + level 2, omega-only radix-3
    let u = mont_bound(2 * h, q);
    let mut k = 0;
    while k < 4 {
        let mut i = 0;
        while i < 54 {
            let b = 162 * k + i;
            v[b] = 3 * h;
            v[b + 54] = 2 * h + u;
            v[b + 108] = 2 * h + u;
            if 3 * h > peak {
                peak = 3 * h;
            }
            if 2 * h + u > peak {
                peak = 2 * h + u;
            }
            i += 1;
        }
        k += 1;
    }
    let mut i = 0;
    while i < N {
        if v[i] > lm[0] {
            lm[0] = v[i];
        }
        i += 1;
    }
    // levels 3, 4, 5: radix-3 on 54-, 18- and 6-blocks
    let mut l = 0;
    while l < 3 {
        let blk = [54usize, 18, 6][l];
        let m = blk / 3;
        let mut base = 0;
        while base < N {
            let mut i = 0;
            while i < m {
                let (i0, i1, i2) = (base + i, base + i + m, base + i + 2 * m);
                let b0 = if bar[l] { r } else { v[i0] };
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

/// Does the un-reduced schedule leave i16? (2917 and 4861: no. 12637: yes.)
pub const fn needs_barrett(q: u16) -> bool {
    bin_model(q, [false; 3]).1 > 32767
}

/// `(reduction flags, output bound)` per prime, evaluated once.
const BIN_SCHED: [([bool; 3], i32); 3] = [
    bin_sched(QS_QUAD[0]),
    bin_sched(QS_QUAD[1]),
    bin_sched(QS_QUAD[2]),
];

const fn bin_sched(q: u16) -> ([bool; 3], i32) {
    let bar = if needs_barrett(q) { [true; 3] } else { [false; 3] };
    (bar, bin_model(q, bar).0[3])
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

/// Which of levels 3, 4 and 5 reduce their untwiddled `a0` input with the lookup Barrett.
pub const fn bar_levels(q: u16) -> [bool; 3] {
    BIN_SCHED[qi(q)].0
}

/// Declared output bound: max |lane| of [`ntt_quad_bin_batch32`], per prime.
pub const fn output_bound(q: u16) -> i32 {
    BIN_SCHED[qi(q)].1
}

const _: () = assert!(bin_model(2917, bar_levels(2917)).1 <= 32767);
const _: () = assert!(bin_model(4861, bar_levels(4861)).1 <= 32767);
const _: () = assert!(bin_model(12637, bar_levels(12637)).1 <= 32767);
// only 12637 needs to reduce, and it needs all three levels
const _: () = assert!(!needs_barrett(2917) && !needs_barrett(4861) && needs_barrett(12637));
const _: () = assert!(bin_model(12637, [false, true, true]).1 > 32767);
const _: () = assert!(bin_model(12637, [true, false, true]).1 > 32767);
const _: () = assert!(bin_model(12637, [true, true, false]).1 > 32767);

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[3 k + r]`: the 16 centered i16 values of `base_k(n) * zeta2_k^r`, r = the level-2 role
    /// `i/54` of the position, **byte-split** so that one `vpermb` (1 uop, port 5) does the
    /// lookup: byte n is the low half of entry n, byte 16+n the high half.
    lut: [[u8; 64]; 12],
    /// 512-bit constants of the lookup Barrett and the omega product, as memory operands:
    /// `[ms, corr, and, or]` — the `vpmultishiftqb` control, the byte-split `-k q` table and the
    /// index fix-up masks (see `vertical_bin_asm::barrett_lut_i16`).
    cv: [[i16; 32]; 4],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2) per sub-ring,
    /// each i16 duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw3: [[u32; 4]; 12],
    tw4: [[u32; 4]; 36],
    tw5: [[u32; 4]; 108],
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

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    // levels 0 + 1 as a function of the nibble, with the level-2 twiddle folded in
    let mut lut = [[0u8; 64]; 12];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
        let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
        let mut r = 0;
        while r < 3 {
            let f = pow_mod(z2, r as u64, q);
            let mut n = 0;
            while n < 16 {
                let n0 = (n & 1) as u64;
                let n1 = ((n >> 1) & 1) as u64;
                let n2 = ((n >> 2) & 1) as u64;
                let n3 = ((n >> 3) & 1) as u64;
                let inner = z1 * ((n1 + ka * n3) % q) % q;
                let t = if s1 == 0 { inner } else { (q - inner) % q };
                let base = ((n0 + ka * n2) % q + t) % q;
                let e = center(base * f % q, q) as u16;
                lut[3 * k + r][n] = e as u8;
                lut[3 * k + r][16 + n] = (e >> 8) as u8;
                n += 1;
            }
            r += 1;
        }
        k += 1;
    }

    let mut tw3 = [[0u32; 4]; 12];
    let mut i = 0;
    while i < 12 {
        tw3[i] = r3_pair::<Q>(ParamsQ::<Q>::ZETA_L3[i]);
        i += 1;
    }
    let mut tw4 = [[0u32; 4]; 36];
    let mut i = 0;
    while i < 36 {
        tw4[i] = r3_pair::<Q>(ParamsQ::<Q>::ZETA_L4[i]);
        i += 1;
    }
    let mut tw5 = [[0u32; 4]; 108];
    let mut i = 0;
    while i < 108 {
        tw5[i] = r3_pair::<Q>(ParamsQ::<Q>::ZETA_L5[i]);
        i += 1;
    }
    let (oa, ob) = mont_pair::<Q>(ParamsQ::<Q>::OMEGA);

    let mut cv = [[0i16; 32]; 4];
    let mut i = 0;
    while i < 32 {
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that word.
        cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
        // byte-split correction table: byte u (u < 32) is the low half of -k(u) q, byte 32 + u
        // its high half.
        let (b0, b1) = (lut_byte(2 * i, Q), lut_byte(2 * i + 1, Q));
        cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[2][i] = 0x1f1f;
        cv[3][i] = 0x2000;
        i += 1;
    }
    Tables { lut, cv, tw3, tw4, tw5, om: [oa, ob], qd: dup(Q as i16) }
}

/// Byte `u` of the 64-byte `vpermb` correction table of the lookup Barrett: the low halves of
/// `-k(s) q` at u = s < 32, the high halves at u = 32 + s.
pub const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

static T2917: Tables = build_tables::<2917>();
static T4861: Tables = build_tables::<4861>();
static T12637: Tables = build_tables::<12637>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    match Q {
        2917 => &T2917,
        4861 => &T4861,
        _ => &T12637,
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

/// The shuffle-port lookup Barrett: 2 port-5 + 3 flexible uops, no multiply-port slot,
/// `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn barrett_lut(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
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

/// Radix-3 butterfly with twiddles from `tw = [w, w', w2, w2']`, optionally reducing `a0`.
#[inline(always)]
unsafe fn r3<const BAR: bool>(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
) -> (__m512i, __m512i, __m512i) {
    let (w1, w1p, w2, w2p) = bc4(tw);
    let t1 = mont(a1, w1, w1p, c.q);
    let t2 = mont(a2, w2, w2p, c.q);
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    let a0 = if BAR { barrett_lut(a0, c) } else { a0 };
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 2).
#[inline(always)]
unsafe fn r3_folded(c: &C, a0: __m512i, t1: __m512i, t2: __m512i) -> (__m512i, __m512i, __m512i) {
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
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

// ---------------------------------------------------------------------------------------------
// where the finished blocks go
// ---------------------------------------------------------------------------------------------

/// Consumer of the transform's output, one **18-row block** at a time.
///
/// The kernel finishes the 648 rows as 36 independent blocks of 18: block `blk` holds rows
/// `18 blk .. 18 blk + 18`, which is the pair of coefficients of each of the 9 quadratic leaves
/// `9 blk .. 9 blk + 9` (leaf j in rows 2j, 2j+1), and is produced out of 18 registers by one
/// levels-4+5 tail. The sink says where those 18 vectors go ([`dst`](BlockSink::dst)) and is
/// handed them the instant they are stored ([`block`](BlockSink::block)), so a consumer can read a
/// block while it is still in L1.
///
/// Both methods are monomorphised into the kernel, so a sink that does nothing costs nothing.
/// A sink whose `block` uses AVX-512 must carry a `#[target_feature]` attribute covering the
/// intrinsics it uses, or it will not be inlined into the kernel.
///
/// # Safety
/// `dst` must return a 64-byte aligned pointer to 1152 writable bytes (18 vectors); the kernel
/// writes them and then calls `block` with the same pointer.
pub trait BlockSink {
    /// Where block `blk` is to be stored.
    unsafe fn dst(&mut self, blk: usize) -> *mut i16;
    /// Called once the 18 vectors of block `blk` are stored at `dst`.
    unsafe fn block(&mut self, blk: usize, dst: *const i16);
}

/// The sink of the plain entry points: block `blk` goes to its own place in the output batch and
/// nothing further happens, so the kernel is exactly the loop it would be without the hook.
pub struct OutSink(pub *mut i16);

impl BlockSink for OutSink {
    #[inline(always)]
    unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
        self.0.add(32 * 18 * blk)
    }
    #[inline(always)]
    unsafe fn block(&mut self, _blk: usize, _dst: *const i16) {}
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_core<const Q: u16, const NT: bool, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
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
    let bar = bar_levels(Q);

    // The caller already holds the `vpermb` byte-index rows; the kernel reads them straight.
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;

    for k in 0..4 {
        let lut = t.lut.as_ptr().add(3 * k) as *const u8;
        let l = |r: usize| -> __m512i { _mm512_load_si512(lut.add(64 * r) as *const __m512i) };
        let (l0, l1, l2) = (l(0), l(1), l(2));

        // lookups (levels 0+1, level-2 twiddle folded in) + level 2 (omega only) + level 3,
        // 9 vectors at a time: the three level-2 triples (i, i+54, i+108) for i = i0, i0+18,
        // i0+36 are exactly the three inputs of one level-3 butterfly in each of the three
        // 54-blocks.
        for i0 in 0..18 {
            let mut y = [_mm512_setzero_si512(); 9];
            for a in 0..3 {
                let i = i0 + 18 * a;
                let x0 = _mm512_permutexvar_epi8(ldb(ip, i), l0);
                let x1 = _mm512_permutexvar_epi8(ldb(ip, i + 54), l1);
                let x2 = _mm512_permutexvar_epi8(ldb(ip, i + 108), l2);
                let (u0, u1, u2) = r3_folded(&c, x0, x1, x2);
                y[a] = u0;
                y[3 + a] = u1;
                y[6 + a] = u2;
            }
            for s in 0..3 {
                let tw = t.tw3[3 * k + s].as_ptr();
                let (v0, v1, v2) = if bar[0] {
                    r3::<true>(&c, y[3 * s], y[3 * s + 1], y[3 * s + 2], tw)
                } else {
                    r3::<false>(&c, y[3 * s], y[3 * s + 1], y[3 * s + 2], tw)
                };
                let b = 54 * s + i0;
                st(bp, b, v0);
                st(bp, b + 18, v1);
                st(bp, b + 36, v2);
            }
        }

        // levels 4 and 5, one 18-block at a time, straight to the sink. The register-resident
        // form (all 18 values live across both levels) measures the same 267 cycles per ring
        // element when it is the only tail in the build, and offering both behind a `const`
        // parameter costs 60: LLVM then schedules the register-resident copy badly. Two passes
        // over the L1-resident block it is.
        for j in 0..9 {
            let kk = 9 * k + j;
            let op = sink.dst(kk);
            let base = 18 * j;
            let t4 = t.tw4[kk].as_ptr();
            for i in 0..6 {
                let (b0, b1, b2) = (base + i, base + i + 6, base + i + 12);
                let (o0, o1, o2) = if bar[1] {
                    r3::<true>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4)
                } else {
                    r3::<false>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4)
                };
                st(bp, b0, o0);
                st(bp, b1, o1);
                st(bp, b2, o2);
            }
            for g in 0..3 {
                let t5 = t.tw5[3 * kk + g].as_ptr();
                let b = 6 * g;
                for i in 0..2 {
                    let (b0, b1, b2) = (base + b + i, base + b + i + 2, base + b + i + 4);
                    let (o0, o1, o2) = if bar[2] {
                        r3::<true>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t5)
                    } else {
                        r3::<false>(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t5)
                    };
                    if NT {
                        _mm512_stream_si512(op.add(32 * (b + i)) as *mut __m512i, o0);
                        _mm512_stream_si512(op.add(32 * (b + i + 2)) as *mut __m512i, o1);
                        _mm512_stream_si512(op.add(32 * (b + i + 4)) as *mut __m512i, o2);
                    } else {
                        st(op, b + i, o0);
                        st(op, b + i + 2, o1);
                        st(op, b + i + 4, o2);
                    }
                }
            }
            sink.block(kk, op);
        }
    }
}

/// Forward quadratic-slot NTT of 32 binary polynomials: rows `2j`, `2j+1` of the output hold
/// `a_p mod (X^2 - psi'^QUAD_SLOT_EXP[j])`, lazily reduced (`|lane| <= output_bound(Q)`).
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `out` is 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_quad_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    ntt_core::<Q, false, _>(input, &mut OutSink(out.v.as_mut_ptr() as *mut i16));
    out.representation = Representation::Ntt;
}

/// Same, but the output is written with non-temporal stores (for output that will not be re-read
/// soon). The caller must `_mm_sfence()` after the last batch.
///
/// # Safety
/// See [`ntt_quad_bin_batch32`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_quad_bin_batch32_nt<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    ntt_core::<Q, true, _>(input, &mut OutSink(out.v.as_mut_ptr() as *mut i16));
    out.representation = Representation::Ntt;
}

/// The same transform with the output handed to `sink` 18 rows at a time instead of being written
/// to a `Batch32`, for consumers that want each block while it is still in L1.
///
/// # Safety
/// See [`BlockSink`]: `sink.dst` must give 18 writable 64-byte aligned vectors per block.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_quad_bin_batch32_sink<const Q: u16, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
    ntt_core::<Q, false, S>(input, sink);
}

// ---------------------------------------------------------------------------------------------
// drivers
// ---------------------------------------------------------------------------------------------

/// Transpose + quadratic-slot NTT of `32 * out.len()` binary polynomials into a materialised
/// `Batch32` array (non-temporal stores: the output of a large run does not fit any cache).
pub fn ntt_quad_bin_polys<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for (b, o) in out.iter_mut().enumerate() {
            let chunk: &[BinaryPoly; 32] =
                &*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]);
            transpose::slice_polys_idx_into(chunk, &mut idx);
            ntt_quad_bin_batch32_nt::<Q>(&idx, o);
        }
        _mm_sfence();
    }
}

/// The `F162` front end: `elems.len()` must be `128 * out.len()` (128 `F162` = 32 ring elements =
/// one `Batch32`). Same slicer as the splitting kernel's driver
/// ([`crate::simd::transpose_f162`]), since the lookup index does not depend on the tree.
pub fn ntt_quad_f162<const Q: u16>(elems: &[F162], out: &mut [Batch32]) {
    drive::<Q, true>(elems, out);
}

/// [`ntt_quad_f162`] with regular stores (cache-resident output).
pub fn ntt_quad_f162_cached<const Q: u16>(elems: &[F162], out: &mut [Batch32]) {
    drive::<Q, false>(elems, out);
}

fn drive<const Q: u16, const NT: bool>(elems: &[F162], out: &mut [Batch32]) {
    assert_eq!(core::mem::size_of::<F162>(), 24, "F162 is not 24 bytes");
    assert_eq!(elems.len(), 128 * out.len(), "128 F162 per output batch");
    let mut idx: Box<[BinaryIndex32; 2]> =
        unsafe { Box::<[BinaryIndex32; 2]>::new_uninit().assume_init() };
    unsafe {
        for (b, o) in out.iter_mut().enumerate() {
            let i = &mut idx[b & 1];
            crate::simd::transpose_f162::slice_f162_into(
                &*(elems.as_ptr().add(128 * b) as *const [F162; 128]),
                i,
            );
            if NT {
                ntt_quad_bin_batch32_nt::<Q>(i, o);
            } else {
                ntt_quad_bin_batch32::<Q>(i, o);
            }
        }
        _mm_sfence();
    }
}

/// Transpose + NTT, handing each finished batch to `f` instead of materialising it.
pub fn ntt_quad_f162_stream<const Q: u16>(elems: &[F162], mut f: impl FnMut(usize, &Batch32)) {
    assert_eq!(elems.len() % 128, 0);
    let mut buf = Batch32::zero(Representation::Ntt);
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..elems.len() / 128 {
            crate::simd::transpose_f162::slice_f162_into(
                &*(elems.as_ptr().add(128 * b) as *const [F162; 128]),
                &mut idx,
            );
            ntt_quad_bin_batch32::<Q>(&idx, &mut buf);
            f(b, &buf);
        }
    }
}

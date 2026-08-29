//! Forward NTT for **binary** inputs in the vertical batch-of-32 layout: the pure-intrinsics
//! **reference** implementation of the split-tree kernel.
//!
//! The production path ([`crate::simd::commit`]) does not call this module; it uses
//! [`crate::simd::vertical_bin_asm`], which is the same tree with levels 4, 5 and 6
//! hand-scheduled in one `asm!` block per 27-block and, for q = 9721, a different reduction
//! schedule. This file keeps the schedule readable — one Rust expression per butterfly, the
//! reduction written where the bound argument needs it — so that the generated kernel has
//! something to be checked against: for q = 3889 the two are bit-identical (same operation order
//! per butterfly, no reduction at all), for q = 9721 they agree modulo q
//! (`tests/vertical_bin.rs`).
//!
//! Levels 0, 1, 2 and the level-3 twiddles are all folded into 16-entry lookup tables indexed by
//! the 4-bit nibble (b_i, b_{i+162}, b_{i+324}, b_{i+486}) of each polynomial; levels 3..6 are
//! radix-3 signed-Montgomery butterflies. The whole batch is done depth-first per 162-block: the
//! table lookups, level 3 and level 4 write a 10 KB stack block, levels 5 and 6 are fused (a
//! degree-9 sub-ring is exactly three degree-3 sub-rings, so its nine values never leave
//! registers) and go straight to the output.
//!
//! ## Instruction selection (port facts measured on this core)
//!
//! `vpermw` zmm is 2 uops (p0 + p5), so a 16-entry i16 lookup done with `vpermw` would cost one
//! port-0 slot per lookup (1080 per batch, ~17% of the port-0 budget). The tables are therefore
//! stored **byte-split** (low halves at byte n, high halves at byte 16+n) and looked up with
//! `vpermb` (1 uop, p5 only) on the byte-index rows `(n, 16+n)` that
//! [`crate::simd::transpose_f162::slice_f162_into`] emits directly (`BinaryIndex32`), so the
//! kernel has no index-expansion prologue at all.
//! `vpbroadcastd zmm, m32` really is a free load (0 p0/p5 uops), so every twiddle is stored as a
//! duplicated u32.
//!
//! ## Bounds (|lane| as a multiple of q; `tests/vertical_bin.rs` checks every output against the
//! declared bound and against `scalar::ntt` of the same lift)
//!
//! Table entries are centered, |T| <= q/2. A twiddle multiplication `mont(a, w, w')` with
//! |w| <= q/2 satisfies |mont| <= |a| q / 2^17 + q/2 + 1 < 0.75 q for any i16 `a`; the radix-3
//! butterfly therefore adds at most 1.5 q to the (untwiddled) `a0` input.
//!
//! | after            | q = 3889 | q = 9721 |
//! |------------------|---------:|---------:|
//! | levels 0+1+2     |   1.00 q |   1.00 q |
//! | level 3          |   3.00 q |   3.00 q |
//! | level 4          |   4.50 q |   2.31 q |
//! | level 5          |   6.00 q |   2.31 q |
//! | level 6 (output) |   7.50 q |   2.31 q |
//!
//! 2^15 / q = 8.42 (3889) and 3.37 (9721), so **q = 3889 needs no Barrett at all**; for q = 9721
//! one two-multiply Barrett ([`params::barrett_i16`]: `vpmulhrsw` + `vpmullw` + `vpsubw`) is
//! applied to the untwiddled `a0` input of levels 4, 5 and 6, which caps it at 0.809 q and hence
//! every output at 0.809 q + 1.5 q = 2.309 q. This is the straightforward schedule, one
//! reduction per butterfly on the one input that grows; the asm kernel spends its port-0 budget
//! differently (lookup Barrett at level 4, none at level 5, `vpmulhrsw` at level 6) and so does
//! not produce the same representatives here.
use crate::params::*;
pub use crate::simd::transpose_f162::BinaryIndex32;
use crate::types::*;
use core::arch::x86_64::*;

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
    /// `vpermb` (1 uop, port 5 - unlike `vpermw`, which is 2 uops and costs a port-0 slot on this
    /// core) does the lookup: byte n is the low half of entry n, byte 16+n the high half.
    lut: [[u8; 64]; 96],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2), each i16
    /// duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw4: [[u32; 4]; 24],
    tw5: [[u32; 4]; 72],
    tw6: [[u32; 4]; 216],
    /// omega and its companion.
    om: [u32; 2],
    /// q and the vpmulhrsw Barrett constant.
    qd: u32,
    bv: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

impl Tables {
    /// `ntt_bin_batch32` output is `a(psi^SLOT_EXP[j]) mod q`.
    pub const fn new<const Q: u16>() -> Tables {
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
            let z = Params::<Q>::ZETA_L4[i];
            let z2 = (z as u64 * z as u64 % q) as u16;
            let (a, b) = mont_pair::<Q>(z);
            let (c, d) = mont_pair::<Q>(z2);
            tw4[i] = [a, b, c, d];
            i += 1;
        }
        let mut tw5 = [[0u32; 4]; 72];
        let mut i = 0;
        while i < 72 {
            let z = Params::<Q>::ZETA_L5[i];
            let z2 = (z as u64 * z as u64 % q) as u16;
            let (a, b) = mont_pair::<Q>(z);
            let (c, d) = mont_pair::<Q>(z2);
            tw5[i] = [a, b, c, d];
            i += 1;
        }
        let mut tw6 = [[0u32; 4]; 216];
        let mut i = 0;
        while i < 216 {
            let z = Params::<Q>::ZETA_L6[i];
            let z2 = (z as u64 * z as u64 % q) as u16;
            let (a, b) = mont_pair::<Q>(z);
            let (c, d) = mont_pair::<Q>(z2);
            tw6[i] = [a, b, c, d];
            i += 1;
        }
        let (oa, ob) = mont_pair::<Q>(Params::<Q>::OMEGA);
        Tables {
            lut,
            tw4,
            tw5,
            tw6,
            om: [oa, ob],
            qd: dup(Q as i16),
            bv: dup(barrett_v(Q)),
        }
    }
}

static T3889: Tables = Tables::new::<3889>();
static T9721: Tables = Tables::new::<9721>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    if Q == 3889 {
        &T3889
    } else {
        &T9721
    }
}

/// Does the 7.5 q growth of the un-Barretted schedule still fit in i16?
pub const fn needs_barrett(q: u16) -> bool {
    15 * (q as u32) >= 2 * 32768
}

/// Declared output bound: max |lane| of `ntt_bin_batch32`, as a multiple of q (numerator / 1000).
pub const fn output_bound_milli_q(q: u16) -> u32 {
    if needs_barrett(q) {
        2310
    } else {
        7500
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

/// `barrett_i16` lane-wise: 2 multiply uops, |r| <= 0.899 q (3889) / 0.809 q (9721).
#[inline(always)]
unsafe fn barrett(a: __m512i, bv: __m512i, q: __m512i) -> __m512i {
    let t = _mm512_mulhrs_epi16(a, bv);
    _mm512_sub_epi16(a, _mm512_mullo_epi16(t, q))
}

struct C {
    q: __m512i,
    bv: __m512i,
    om: __m512i,
    omp: __m512i,
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

/// Radix-3 butterfly with twiddles from `tw = [w, w', w2, w2']`.
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
    let a0 = if BAR { barrett(a0, c.bv, c.q) } else { a0 };
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 3).
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
// the kernel
// ---------------------------------------------------------------------------------------------

/// Forward NTT of 32 binary polynomials: `out.v[j][p] = a_p(psi^SLOT_EXP[j]) mod q`, lazily
/// reduced (see the bound table above).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    let outp = out.v.as_mut_ptr() as *mut i16;
    let t = tables::<Q>();
    let c = C {
        q: bc(&t.qd),
        bv: bc(&t.bv),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
    };
    let bar = needs_barrett(Q);

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

            let (u0, u1, u2) = r3_folded(&c, a0, a1, a2);
            let (v0, v1, v2) = r3_folded(&c, b0, b1, b2);
            st(bp, i, u0);
            st(bp, i + 27, u1);
            st(bp, i + 54, u2);
            st(bp, i + 81, v0);
            st(bp, i + 108, v1);
            st(bp, i + 135, v2);
        }

        // level 4: 6 sub-rings of degree 27, m = 9.
        for j in 0..6 {
            let base = 27 * j;
            let tw = t.tw4[6 * k + j].as_ptr();
            for i in 0..9 {
                let (a0, a1, a2) = (ld(bp, base + i), ld(bp, base + 9 + i), ld(bp, base + 18 + i));
                let (o0, o1, o2) = if bar {
                    r3::<true>(&c, a0, a1, a2, tw)
                } else {
                    r3::<false>(&c, a0, a1, a2, tw)
                };
                st(bp, base + i, o0);
                st(bp, base + 9 + i, o1);
                st(bp, base + 18 + i, o2);
            }
        }

        // levels 5 and 6 fused: each degree-9 sub-ring is exactly three degree-3 sub-rings, so the
        // nine values stay in registers and the level-6 results go straight to the output.
        let op = outp.add(32 * 162 * k);
        for j in 0..18 {
            let base = 9 * j;
            let tw = t.tw5[18 * k + j].as_ptr();
            let mut v = [_mm512_setzero_si512(); 9];
            for i in 0..9 {
                v[i] = ld(bp, base + i);
            }
            for i in 0..3 {
                let (o0, o1, o2) = if bar {
                    r3::<true>(&c, v[i], v[3 + i], v[6 + i], tw)
                } else {
                    r3::<false>(&c, v[i], v[3 + i], v[6 + i], tw)
                };
                v[i] = o0;
                v[3 + i] = o1;
                v[6 + i] = o2;
            }
            for i in 0..3 {
                let tw = t.tw6[54 * k + 3 * j + i].as_ptr();
                let (o0, o1, o2) = if bar {
                    r3::<true>(&c, v[3 * i], v[3 * i + 1], v[3 * i + 2], tw)
                } else {
                    r3::<false>(&c, v[3 * i], v[3 * i + 1], v[3 * i + 2], tw)
                };
                let b = base + 3 * i;
                st(op, b, o0);
                st(op, b + 1, o1);
                st(op, b + 2, o2);
            }
        }
    }
    out.representation = Representation::Ntt;
}

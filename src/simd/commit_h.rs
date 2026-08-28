//! The horizontal-layout Ajtai commitment: `y[j] = sum_i A_i[j] * NTT(w_i)[j] mod q` computed in
//! groups of four ring elements (or eight, or sixteen) that stay in L1 from the `F162` input to
//! the accumulator.
//!
//! # Why groups of four
//!
//! A commitment over 2^16 ring elements needs 2^16 NTT-domain rows `A_i` (85 MB per prime, cold).
//! Materialising the transforms first writes 85 MB of NTT output to DRAM and reads it back on top
//! of the 85 MB of `A`. With one `HBatch4` in flight — 5 KB of transform output, 5 KB of `A` and
//! the 5 KB accumulator, all in L1 — the only DRAM traffic is the 85 MB of `A` plus the 6 MB of
//! `F162` input, and the base multiplication is a handful of uops on data that is already in
//! registers. The price is the horizontal kernel itself: 540 / 587 cycles per polynomial against
//! the vertical binary kernel's 280 / 305, which is what makes this the slower of the two
//! commitments ([`crate::simd::commit`] is the fast one).
//!
//! # Layout
//!
//! `A` is stored per prime as `&[HBatch4]` in exactly the layout the transform produces:
//! `a[g].v[r][8*p + j]` = tree slot `81*j + r` of `A_{4g+p}`, i.e. `HBatch4::set(p, &a_ntt)`.
//! One `HBatch4` of `A` is 5184 bytes, 16384 of them are 84.9 MB.
//!
//! # The base multiplication
//!
//! `vpdpwssd` accumulates *pairs* of adjacent 16-bit lanes into one 32-bit lane. In the transform's
//! lane order `8*p + j` an adjacent pair is `(p, j)` and `(p, j+1)`: the same polynomial at two
//! different slots, which must not be summed. One `vpermb` (a pure port-5 uop, and port 5 is idle
//! in this loop) puts the lanes in the order `4*j + p`, where an adjacent pair is two polynomials
//! at the *same* slot — exactly the sum the commitment wants. Then
//!
//! ```text
//!     acc[r].dword[2j]     += W[81j+r] A^{(0)}[81j+r] + W[81j+r] A^{(1)}[81j+r]
//!     acc[r].dword[2j + 1] += W[81j+r] A^{(2)}[81j+r] + W[81j+r] A^{(3)}[81j+r]
//! ```
//!
//! and `y[81*j + r] = acc[r][2j] + acc[r][2j+1] mod q`. `A` may be stored pre-permuted
//! ([`permute_a_slice`]), which removes the second `vpermb` per register; it is the same 85 MB.
//!
//! # Overflow and the periodic reduction
//!
//! Raw `i32` accumulation, no Montgomery anything: the kernel's output bound is
//! `|W| <= OUT_ABS` (7.69 q for 3889, 2.13 q for 9721) and `A` is centered, `|A| <= (q-1)/2`, so
//! one group adds at most `2 * OUT_ABS * (q-1)/2` per lane. With the accumulator reduced to
//! `|acc| <= 1.5 q` the number of groups that fit under `i32::MAX` is [`CParams::KRED`]
//! (18 for q = 3889, 10 for q = 9721). Every `KRED` groups the accumulator is reduced by a
//! float-assisted exact scheme: `t = round(x / q)` computed in `f32` (the relative error of the
//! conversion and the multiplication is at most `2^-23`, so `|t - x/q| <= 0.5 + 0.1`), then
//! `r = x - t*q` in `i32`, which is *exactly* congruent to `x` mod q with `|r| <= 1.5 q` for any
//! rounding mode. `t*q` cannot overflow because `|t| q <= |x| + 1.5 q` and the accumulator is kept
//! below `i32::MAX - 2q`.
//!
//! # The front end
//!
//! [`hbatch4_from_f162`] turns 16 `F162` into the coefficient-form `HBatch4` of the four ring
//! elements they lift to. Writing `r = 4a + b`, the lift gives
//!
//! ```text
//!     v[4a + b][8p + j] = bit (a + off(b, j)) of F162 element 4p + k(b, j),
//!     k(b, j) = (b + j) mod 4,   off(b, j) = floor(81 j / 4) + [b + (j mod 4) >= 4],
//! ```
//!
//! so for fixed `(p, b, j)` the 21 values `a = 0..21` are 21 *consecutive* bits of one `F162`.
//! Per polynomial: two `vpermi2b` cut the 32 windows `(b, j)` out of the 96 input bytes as dwords,
//! two `vpsrlvd` align them, two more `vpermi2b` group them into the qwords a bit transpose wants,
//! two `vgf2p8affineqb` (the identity matrix as the vector operand) transpose them — giving one
//! byte per output register, bit `j` of byte `r` of polynomial `p` — and two `vpermb` put those
//! bytes in the order that makes the following interleave land in plain register order. Eight
//! `vpunpck` then interleave the four polynomials into 81 32-bit masks, and each output register
//! is one `kmovd` from that scratch plus one zero-masked `vmovdqu16` of the all-ones vector.
//! Per group of four ring elements: 428 instructions, 644 uops (150 of them port-5 only), 89
//! stores. Measured 44 / 43 cycles per ring element (q = 3889 / 9721) — port-5 bound, against a
//! floor of 81 stores = 20 cycles per ring element for writing the 5184-byte batch at all.
//!
//! # Measured (i7-11850H, one core, 2^18 F162 = 2^16 ring elements, 16384 `HBatch4` of `A`)
//!
//! | cycles / ring element        | q = 3889 | q = 9721 |
//! |------------------------------|---------:|---------:|
//! | front end                    |     44.3 |     42.3 |
//! | kernel (L1 resident)         |    546.1 |    593.3 |
//! | basemul + reduction (L1)     |     50.6 |     56.3 |
//! | ... with `A` pre-permuted    |     46.1 |     49.6 |
//! | compute sum                  |    641.0 |    691.9 |
//! | 85 MB `A` stream alone       |    308.3 |    312.0 |
//! | **best measured total**      |**855.9** |**897.0** |
//!
//! i.e. 12.5 / 13.1 ms against a 9.3 / 10.1 ms compute-only time and a 4.4 ms DRAM floor: only
//! about a third of the `A` stream hides behind the transform. The reason is structural — `A` is
//! touched in one burst of 81 cache lines per ~2600-cycle block and the transform in between is
//! an opaque call, so there is nowhere to issue prefetches at the ~1-per-32-cycles rate that would
//! keep 16 line fills in flight; issuing all 81 at once (from the base multiplication, from the
//! front end, or both) recovers only ~45 cycles per ring element.
use crate::f162::bit;
use crate::params::N;
use crate::simd::horizontal_gen::{ntt_gen_hbatch4, HBatch4, HTables};
use bin_fields::scalar::F162;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// Aligned constant tables
// ---------------------------------------------------------------------------------------------

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct A64<T>(pub T);

/// The 8x8 identity bit-matrix in `vgf2p8affineqb` vector form (byte j = 1 << j).
const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

/// `floor(81 j / 4)`: the bit offset of the `j`-th 81-coefficient block inside its `F162`.
pub const fn base_off(j: usize) -> usize {
    81 * j / 4
}
/// Which of the four `F162` of a ring element carries `v[4a+b][8p+j]`.
pub const fn k_of(b: usize, j: usize) -> usize {
    (b + j) & 3
}
/// Bit offset inside that `F162` of `a = 0`.
pub const fn off_of(b: usize, j: usize) -> usize {
    base_off(j) + if b + (j & 3) >= 4 { 1 } else { 0 }
}

/// Byte index of a byte `o` of the 96-byte region of polynomial `p` inside the 128-byte
/// `vpermi2b` window built from loads at `96p` and `96p + 32`.
const fn src_idx(o: usize) -> u8 {
    (if o < 64 { o } else { o + 32 }) as u8
}

/// `vpermi2b` index that cuts the 32-bit windows of `(b, j)` out of the input.
/// `half` = 0 for `j` in 0..4 (dword `4j + b`), 1 for `j` in 4..8.
const fn win_idx(half: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut j = 4 * half;
    while j < 4 * half + 4 {
        let mut b = 0;
        while b < 4 {
            let d = 4 * (j - 4 * half) + b;
            let o0 = 24 * k_of(b, j) + off_of(b, j) / 8;
            let mut u = 0;
            while u < 4 {
                t[4 * d + u] = src_idx(o0 + u);
                u += 1;
            }
            b += 1;
        }
        j += 1;
    }
    t
}

/// Per-dword right shift that aligns each window to its bit offset.
const fn win_shift(half: usize) -> [u32; 16] {
    let mut t = [0u32; 16];
    let mut j = 4 * half;
    while j < 4 * half + 4 {
        let mut b = 0;
        while b < 4 {
            t[4 * (j - 4 * half) + b] = (off_of(b, j) % 8) as u32;
            b += 1;
        }
        j += 1;
    }
    t
}

/// `vpermi2b` index that groups the aligned windows into the qwords the bit transpose consumes:
/// qword `4*tb + b` (`tb` = byte index inside a window) holds `R(b, 7-jj)[tb]` at byte `jj`, so
/// that `vgf2p8affineqb` emits byte `j'` = the 8-bit mask of output register `32*tb + 4*j' + b`.
/// `part` = 0 gives `tb` = 0, 1 (one whole zmm), `part` = 1 gives `tb` = 2 (half a zmm).
const fn q_idx(part: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut tb = 2 * part;
    while tb < 2 * part + (if part == 0 { 2 } else { 1 }) {
        let mut b = 0;
        while b < 4 {
            let mut jj = 0;
            while jj < 8 {
                let j = 7 - jj;
                let src = if j < 4 { 16 * j + 4 * b + tb } else { 64 + 16 * (j - 4) + 4 * b + tb };
                let dst = 32 * (tb - 2 * part) + 8 * b + jj;
                t[dst] = src as u8;
                jj += 1;
            }
            b += 1;
        }
        tb += 1;
    }
    t
}

/// Byte index at which the bit transpose leaves the mask of output register `r`:
/// qword `4*tb + b` of the transpose output, byte `u`, with `r = 32*tb + 4*u + b`.
const fn gf_byte(r: usize) -> usize {
    let tb = r / 32;
    let rem = r % 32;
    32 * tb + 8 * (rem % 4) + rem / 4
}

/// Byte index that the four-way interleave of [`interleave4`] lands at scratch dword `d`.
const fn ilv_byte(d: usize) -> usize {
    16 * ((d >> 2) & 3) + 4 * (d >> 4) + (d & 3)
}

/// One `vpermb` per transpose output that reorders the mask bytes so that the interleave writes
/// the 32-bit masks in plain register order — the expansion loop is then a sequential scan with
/// no index table. `part` = 0 covers registers 0..64, `part` = 1 registers 64..81.
const fn reorder_idx(part: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let n = if part == 0 { 64 } else { 81 - 64 };
    let mut i = 0;
    while i < n {
        t[ilv_byte(i)] = gf_byte(if part == 0 { i } else { i + 64 }) as u8;
        i += 1;
    }
    t
}

/// The lane permutation `8p + j -> 4j + p` that turns a `vpdpwssd` pair into "two polynomials,
/// one slot" (a byte permutation, so one `vpermb`).
const fn perm_lane() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 4 {
        let mut j = 0;
        while j < 8 {
            let dst = 4 * j + p;
            let src = 8 * p + j;
            t[2 * dst] = (2 * src) as u8;
            t[2 * dst + 1] = (2 * src + 1) as u8;
            j += 1;
        }
        p += 1;
    }
    t
}

static WIN_IDX: [A64<[u8; 64]>; 2] = [A64(win_idx(0)), A64(win_idx(1))];
static WIN_SHIFT: [A64<[u32; 16]>; 2] = [A64(win_shift(0)), A64(win_shift(1))];
static Q_IDX: [A64<[u8; 64]>; 2] = [A64(q_idx(0)), A64(q_idx(1))];
static REORDER: [A64<[u8; 64]>; 2] = [A64(reorder_idx(0)), A64(reorder_idx(1))];
static PERM_LANE: A64<[u8; 64]> = A64(perm_lane());

#[inline(always)]
unsafe fn ld<T>(p: *const T) -> __m512i {
    _mm512_load_si512(p as *const __m512i)
}

// ---------------------------------------------------------------------------------------------
// Front end: 16 F162 -> HBatch4 in coefficient form
// ---------------------------------------------------------------------------------------------

/// Scalar reference for [`hbatch4_from_f162`]: coefficient `r + 81 j` of ring element `4g + p` is
/// bit `(r + 81 j) / 4` of `F162` element `16 g + 4 p + ((r + 81 j) mod 4)`.
pub fn hbatch4_from_f162_scalar(elems: &[F162; 16]) -> HBatch4 {
    let mut b = HBatch4::zero();
    for p in 0..4 {
        for r in 0..81 {
            for j in 0..8 {
                let c = r + 81 * j;
                b.v[r][8 * p + j] = bit(&elems[4 * p + (c & 3)], c >> 2) as i16;
            }
        }
    }
    b
}

#[inline(always)]
unsafe fn interleave4(x: &[__m512i; 4], dst: *mut u32) {
    let l01 = _mm512_unpacklo_epi8(x[0], x[1]);
    let h01 = _mm512_unpackhi_epi8(x[0], x[1]);
    let l23 = _mm512_unpacklo_epi8(x[2], x[3]);
    let h23 = _mm512_unpackhi_epi8(x[2], x[3]);
    _mm512_store_si512(dst as *mut __m512i, _mm512_unpacklo_epi16(l01, l23));
    _mm512_store_si512(dst.add(16) as *mut __m512i, _mm512_unpackhi_epi16(l01, l23));
    _mm512_store_si512(dst.add(32) as *mut __m512i, _mm512_unpacklo_epi16(h01, h23));
    _mm512_store_si512(dst.add(48) as *mut __m512i, _mm512_unpackhi_epi16(h01, h23));
}

/// 16 `F162` -> the coefficient-form `HBatch4` of the four ring elements they lift to, issuing one
/// `prefetcht0` per output register from `pf` (null = none) — the expansion loop is the only place
/// in a block, apart from the base multiplication, where the `A` stream can be pulled in ahead of
/// the transform.
///
/// # Safety
/// Requires AVX-512 F/BW/VBMI/VBMI2/GFNI and `elems` to be 16 contiguous `F162` (384 bytes).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn hbatch4_from_f162_pf(elems: &[F162; 16], out: &mut HBatch4, pf: *const i8) {
    let src = elems.as_ptr() as *const u8;
    let ident = _mm512_set1_epi64(GF_IDENT);
    let mut scratch = A64([0u32; 128]);
    let mut xs = [_mm512_setzero_si512(); 4];
    let mut ys = [_mm512_setzero_si512(); 4];
    for p in 0..4 {
        let s0 = _mm512_loadu_si512(src.add(96 * p) as *const __m512i);
        let s1 = _mm512_loadu_si512(src.add(96 * p + 32) as *const __m512i);
        let w0 = _mm512_permutex2var_epi8(s0, ld(&WIN_IDX[0]), s1);
        let w1 = _mm512_permutex2var_epi8(s0, ld(&WIN_IDX[1]), s1);
        let h0 = _mm512_srlv_epi32(w0, ld(&WIN_SHIFT[0]));
        let h1 = _mm512_srlv_epi32(w1, ld(&WIN_SHIFT[1]));
        let gx = _mm512_gf2p8affine_epi64_epi8::<0>(
            ident,
            _mm512_permutex2var_epi8(h0, ld(&Q_IDX[0]), h1),
        );
        let gy = _mm512_gf2p8affine_epi64_epi8::<0>(
            ident,
            _mm512_permutex2var_epi8(h0, ld(&Q_IDX[1]), h1),
        );
        xs[p] = _mm512_permutexvar_epi8(ld(&REORDER[0]), gx);
        ys[p] = _mm512_permutexvar_epi8(ld(&REORDER[1]), gy);
    }
    interleave4(&xs, scratch.0.as_mut_ptr());
    interleave4(&ys, scratch.0.as_mut_ptr().add(64));

    let ones = core::hint::black_box(_mm512_set1_epi16(1));
    let sp = scratch.0.as_ptr();
    let op = out.v.as_mut_ptr() as *mut __m512i;
    for r in 0..81 {
        if !pf.is_null() {
            _mm_prefetch::<_MM_HINT_T0>(pf.add(64 * r));
        }
        let m = *sp.add(r);
        _mm512_store_si512(op.add(r), _mm512_maskz_mov_epi16(m as __mmask32, ones));
    }
}

/// [`hbatch4_from_f162_pf`] with no prefetch.
///
/// # Safety
/// See [`hbatch4_from_f162_pf`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn hbatch4_from_f162_into(elems: &[F162; 16], out: &mut HBatch4) {
    hbatch4_from_f162_pf(elems, out, core::ptr::null())
}

/// [`hbatch4_from_f162_into`] returning a fresh batch.
///
/// # Safety
/// See [`hbatch4_from_f162_into`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn hbatch4_from_f162(elems: &[F162; 16]) -> HBatch4 {
    let mut b = HBatch4::zero();
    hbatch4_from_f162_into(elems, &mut b);
    b
}

// ---------------------------------------------------------------------------------------------
// Accumulator, bounds
// ---------------------------------------------------------------------------------------------

/// The raw `i32` accumulator: `v[r][2j]` and `v[r][2j+1]` together hold `y[81 j + r]`
/// (polynomials 0,1 and 2,3 of each group respectively). 81 vectors, 5184 bytes.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Acc {
    pub v: [[i32; 16]; 81],
}

impl Default for Acc {
    fn default() -> Self {
        Self::zero()
    }
}

impl Acc {
    pub fn zero() -> Self {
        Acc { v: [[0i32; 16]; 81] }
    }
    /// `y[81 j + r] = (v[r][2j] + v[r][2j+1]) mod q`, fully reduced.
    pub fn finish<const Q: u16>(&self) -> [u32; N] {
        let q = Q as i64;
        let mut y = [0u32; N];
        for r in 0..81 {
            for j in 0..8 {
                let s = self.v[r][2 * j] as i64 + self.v[r][2 * j + 1] as i64;
                y[81 * j + r] = s.rem_euclid(q) as u32;
            }
        }
        y
    }
}

/// Bound bookkeeping of the commitment loop.
pub struct CParams<const Q: u16>;

impl<const Q: u16> CParams<Q> {
    /// `|NTT output|` of `horizontal_gen`.
    pub const WMAX: i64 = HTables::<Q>::OUT_ABS as i64;
    /// `|A|` for centered uniform rows.
    pub const AMAX: i64 = ((Q as i64) - 1) / 2;
    /// What one group of four polynomials adds to a lane: two `vpdpwssd` terms.
    pub const STEP: i64 = 2 * Self::WMAX * Self::AMAX;
    /// `|acc|` right after [`reduce_acc`] (any rounding mode; 0.6 q with round-to-nearest).
    pub const RMAX: i64 = 3 * (Q as i64) / 2;
    /// Largest number of groups that may be accumulated between two reductions.
    pub const KRED: usize =
        ((i32::MAX as i64 - 2 * Q as i64 - Self::RMAX) / Self::STEP) as usize;
    /// The accumulator never leaves `[-CAP, CAP]`.
    pub const CAP: i64 = Self::RMAX + Self::KRED as i64 * Self::STEP;
    const _CHECK: () = assert!(Self::CAP + Self::RMAX < i32::MAX as i64, "i32 accumulator overflow");
}

/// Groups between two accumulator reductions for prime `Q` (18 for 3889, 10 for 9721).
pub fn kred<const Q: u16>() -> usize {
    () = CParams::<Q>::_CHECK;
    CParams::<Q>::KRED
}

/// The proved accumulator cap `|acc| <= cap` for prime `Q`.
pub fn acc_cap<const Q: u16>() -> i64 {
    CParams::<Q>::CAP
}

/// What one group of four ring elements can add to an accumulator lane.
pub fn acc_step<const Q: u16>() -> i64 {
    CParams::<Q>::STEP
}

/// `x mod q` in `i32`, exact and `|r| <= 1.5 q`: `t = round(x/q)` through `f32` (the conversion
/// and the multiply each carry at most a `2^-23` relative error, so `|t - x/q| < 1.2` for any
/// rounding mode), then `r = x - t q` exactly. `|t| q <= |x| + 1.5 q < i32::MAX`.
#[inline(always)]
unsafe fn reduce_i32(x: __m512i, invq: __m512, qv: __m512i) -> __m512i {
    let t = _mm512_cvt_roundps_epi32::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(_mm512_cvtepi32_ps(x), invq),
    );
    _mm512_sub_epi32(x, _mm512_mullo_epi32(t, qv))
}

/// Reduce every lane of the accumulator to `|acc| <= 1.5 q`, exactly (`~6` uops per register,
/// run once every [`kred`] groups).
///
/// # Safety
/// Requires AVX-512 F.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn reduce_acc<const Q: u16>(acc: &mut Acc) {
    let qv = _mm512_set1_epi32(Q as i32);
    let invq = _mm512_set1_ps(1.0f32 / Q as f32);
    let p = acc.v.as_mut_ptr() as *mut __m512i;
    for r in 0..81 {
        _mm512_store_si512(p.add(r), reduce_i32(_mm512_load_si512(p.add(r)), invq, qv));
    }
}

// ---------------------------------------------------------------------------------------------
// Base multiplication
// ---------------------------------------------------------------------------------------------

/// `a[g].v[r][8p+j] -> a[g].v[r][4j+p]`: the same 85 MB, stored so the base multiplication needs
/// only one `vpermb` per register instead of two.
pub fn permute_a(a: &HBatch4) -> HBatch4 {
    let mut o = HBatch4::zero();
    for r in 0..81 {
        for p in 0..4 {
            for j in 0..8 {
                o.v[r][4 * j + p] = a.v[r][8 * p + j];
            }
        }
    }
    o
}

pub fn permute_a_slice(a: &mut [HBatch4]) {
    for b in a.iter_mut() {
        *b = permute_a(b);
    }
}

/// `acc += W o A` for one group of four polynomials. `PA` = `a` is already lane-permuted.
///
/// # Safety
/// Requires AVX-512 F/BW/VBMI/VNNI.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn basemul_step<const PA: bool>(w: &HBatch4, a: &HBatch4, acc: &mut Acc) {
    let perm = ld(&PERM_LANE);
    let wp = w.v.as_ptr() as *const __m512i;
    let ap = a.v.as_ptr() as *const __m512i;
    let cp = acc.v.as_mut_ptr() as *mut __m512i;
    for r in 0..81 {
        let ww = _mm512_permutexvar_epi8(perm, _mm512_load_si512(wp.add(r)));
        let av = _mm512_load_si512(ap.add(r));
        let aa = if PA { av } else { _mm512_permutexvar_epi8(perm, av) };
        _mm512_store_si512(cp.add(r), _mm512_dpwssd_epi32(_mm512_load_si512(cp.add(r)), ww, aa));
    }
}

// ---------------------------------------------------------------------------------------------
// The commitment
// ---------------------------------------------------------------------------------------------

/// One block of `G` groups: front end + transform for each, then one fused pass over the 81
/// registers that keeps the accumulator vector live across all `G` products.
#[inline(always)]
unsafe fn block<const Q: u16, const G: usize, const PA: bool, const PF: usize, const PFF: usize>(
    ep: *const F162,
    a: &[HBatch4],
    g0: usize,
    ws: &mut [HBatch4],
    acc: &mut Acc,
    perm: __m512i,
) {
    for gi in 0..G {
        let t = g0 + G * PFF + gi;
        let fpf = if PFF > 0 && t < a.len() { a[t].v.as_ptr() as *const i8 } else { core::ptr::null() };
        hbatch4_from_f162_pf(&*(ep.add(16 * (g0 + gi)) as *const [F162; 16]), &mut ws[gi], fpf);
        ntt_gen_hbatch4::<Q>(&mut ws[gi]);
    }
    let cp = acc.v.as_mut_ptr() as *mut __m512i;
    let nxt = g0 + G * PF;
    let pf = if PF > 0 && nxt < a.len() { a[nxt].v.as_ptr() } else { a[g0].v.as_ptr() } as *const i8;
    for r in 0..81 {
        if PF > 0 {
            _mm_prefetch::<_MM_HINT_T0>(pf.add(64 * r));
        }
        let mut s = _mm512_load_si512(cp.add(r));
        for gi in 0..G {
            let ww = _mm512_permutexvar_epi8(
                perm,
                _mm512_load_si512((ws[gi].v.as_ptr() as *const __m512i).add(r)),
            );
            let av = _mm512_load_si512((a[g0 + gi].v.as_ptr() as *const __m512i).add(r));
            let aa = if PA { av } else { _mm512_permutexvar_epi8(perm, av) };
            s = _mm512_dpwssd_epi32(s, ww, aa);
        }
        _mm512_store_si512(cp.add(r), s);
    }
}

/// The commitment loop.
///
/// # Safety
/// Requires AVX-512 F/BW/VL/VBMI/VBMI2/VNNI/GFNI.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn commit_acc<const Q: u16, const G: usize, const PA: bool, const PF: usize, const PFF: usize>(
    elems: &[F162],
    a: &[HBatch4],
) -> Acc {
    let ngroups = elems.len() / 16;
    assert_eq!(elems.len() % 16, 0, "need a whole number of groups of 4 ring elements");
    assert!(a.len() >= ngroups, "not enough A rows");
    let mut acc = Acc::zero();
    let mut ws: Vec<HBatch4> = (0..G.max(1)).map(|_| HBatch4::zero()).collect();
    let perm = ld(&PERM_LANE);
    let ep = elems.as_ptr();
    let kr = (kred::<Q>() / G).max(1) * G;

    let mut g = 0;
    let mut since = 0;
    while g + G <= ngroups {
        block::<Q, G, PA, PF, PFF>(ep, a, g, &mut ws, &mut acc, perm);
        g += G;
        since += G;
        if since >= kr {
            reduce_acc::<Q>(&mut acc);
            since = 0;
        }
    }
    while g < ngroups {
        block::<Q, 1, PA, PF, PFF>(ep, a, g, &mut ws, &mut acc, perm);
        g += 1;
        since += 1;
        if since >= kred::<Q>() {
            reduce_acc::<Q>(&mut acc);
            since = 0;
        }
    }
    acc
}

/// `y[j] = sum_i A_i[j] * NTT_q(w_i)[j] mod q` over the ring elements the `F162` stream lifts to,
/// with `A` in the transform's own layout (`a[g].v[r][8p+j]` = slot `81 j + r` of `A_{4g+p}`).
/// `elems.len()` must be a multiple of 16 and `a.len() >= elems.len() / 16`.
pub fn commit_h<const Q: u16>(elems: &[F162], a: &[HBatch4]) -> [u32; N] {
    unsafe { commit_acc::<Q, 1, false, 1, 2>(elems, a).finish::<Q>() }
}

/// [`commit_h`] with an explicit configuration: `G` groups (4 G polynomials) per accumulator pass,
/// `PA` = `A` pre-permuted by [`permute_a_slice`], `PF` = software prefetch distance in blocks
/// (0 = none, k = prefetch the `A` of the block k ahead, one line per register) in the base
/// multiplication, `PFF` = the same distance for the prefetches issued from the front end.
pub fn commit_h_var<const Q: u16, const G: usize, const PA: bool, const PF: usize, const PFF: usize>(
    elems: &[F162],
    a: &[HBatch4],
) -> [u32; N] {
    unsafe { commit_acc::<Q, G, PA, PF, PFF>(elems, a).finish::<Q>() }
}

/// The scalar reference: `sum_i A_i[j] * scalar::ntt(lift_elem(elems, i))[j] mod q`.
pub fn commit_h_scalar<const Q: u16>(elems: &[F162], a: &[HBatch4]) -> [u32; N] {
    let q = Q as u64;
    let mut y = [0u64; N];
    for i in 0..elems.len() / 4 {
        let w = crate::scalar::ntt::<Q>(&crate::f162::lift_elem(elems, i));
        let ai = a[i / 4].get(i % 4);
        for j in 0..N {
            y[j] = (y[j] + w[j] as u64 * (ai.v[j] as i64).rem_euclid(q as i64) as u64) % q;
        }
    }
    let mut o = [0u32; N];
    for j in 0..N {
        o[j] = y[j] as u32;
    }
    o
}

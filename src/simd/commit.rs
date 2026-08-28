//! Ajtai commitment in the NTT domain: one row of an inner product,
//!
//! ```text
//!     y[j] = sum_i A_i[j] * NTT_q(w_i)[j]  mod q,      j = 0..648,
//! ```
//!
//! where the `w_i` are the ring elements lifted from a stream of `F162` (`f162::lift4`) and `A` is
//! a fixed row of uniform NTT-domain ring elements.
//!
//! # The layout of A
//!
//! `A` is a `&[Batch32]` in exactly the layout the transform writes: `a[b].v[j][p]` is slot `j` of
//! `A_{32b + p}`, one 64-byte vector per slot per batch, 41472 bytes per batch, 85 MB per prime
//! for 2^16 ring elements. Entries are **centered**, `|A| <= (q-1)/2`. `A` does not fit any cache
//! and is read exactly once per commitment, so its 85 MB at 19.5 GB/s is the DRAM floor of the
//! whole computation (4.4 ms) and every design decision below is about hiding it behind the
//! transform rather than adding traffic of its own.
//!
//! # Raw VNNI accumulation
//!
//! A slot product is never reduced. One `vpdpwssd` per slot per batch does
//! `acc32 += W_j * A_j` on adjacent pairs of 16-bit lanes, so a 32-bit lane carries the running
//! sum of two ring elements' products: one multiply-port uop, three loads and one store per slot
//! per batch, against the 4 multiply-port uops a Montgomery slot product would cost. Nothing but
//! the sum is ever needed, and `sum_i A_i[j] W_i[j]` is congruent mod q whatever representatives
//! the transform leaves, so the only question is overflow.
//!
//! # The exact fold-back
//!
//! The transform's output is lazily reduced to [`w_bound`] (7.5 q for q = 3889, 2.294 q for
//! q = 9721) and `|A| <= (q-1)/2`, so one batch adds at most [`acc_per_batch`] to a lane. Every
//! [`red_period`] batches the accumulator is folded back into `|acc| <= 2^15 (1 + R)`
//! ([`acc_after_reduce`], `R = 2^16 mod q`) by [`reduce_acc_i32`]: three uops per accumulator
//! vector, one of them on the multiply port, amortised over 8 (q = 3889) or 4 (q = 9721) batches.
//! The period is the largest power of two for which `acc_after_reduce + period * acc_per_batch`
//! still fits `i32`, which the `fits` assertion below checks at compile time.
//!
//! # The packed accumulator
//!
//! One `vpdpwssd` per slot leaves 16 i32 lanes per slot: 648 vectors, 41 KB, read and written once
//! per slot per batch. That does not fit L1 next to the A stream — measured 85 cycles per ring
//! element against a 40-cycle uop floor. Folding the 16 lanes of a slot down to 8 and packing two
//! slots into one vector cuts the accumulator to 21.5 KB and its traffic by a third for the same
//! uop count (two `vpmaddwd` and two `vshufi64x2` per two slots instead of two `vpdpwssd`, ALU
//! work in place of loads): 85 -> 58 cycles per ring element, and 58 is what the commitment pays.
//!
//! Layout: the 648 slots are 24 consecutive 27-slot blocks, exactly the granularity of one `asm!`
//! block of the transform. Inside a block, vector `p < 13` carries slot `2p` in lanes 0..8 and
//! slot `2p+1` in lanes 8..16; vector 13 carries the odd slot 26 in lanes 0..8 and a harmless
//! duplicate of it in lanes 8..16 ([`slot_lane`]). Lane `l` of a group therefore sums the ring
//! elements `2l, 2l+1, 2l+16, 2l+17` of every batch — four products per lane per batch, which is
//! why [`red_period`] is half of what the unpacked accumulator would allow.
//!
//! # Consuming the transform one block at a time
//!
//! [`vertical_bin_asm`](crate::simd::vertical_bin_asm) produces the 648 slots as 24 blocks of 27,
//! each written out of registers by one `asm!` block. `Mac` is a [`BlockSink`] that hands the
//! kernel a single 1728-byte scratch for every block and multiplies the block into the
//! accumulator the moment it is stored, while it is still in L1. The alternatives cost, per ring
//! element (q = 3889 / 9721): materialising the whole transform first 874 / 908 cycles — it adds
//! 85 MB of writes and 85 MB of reads and is DRAM-bound; one 41 KB buffer per batch 645 / 687;
//! per block, which is what this module does, 630 / 664.
//!
//! # Prefetching A
//!
//! Per-block consumption is also what makes the A stream hideable. [`mac27`] issues one
//! `prefetcht1` per cache line it will read one batch later, 27 per block, so the 648 lines of the
//! next batch's A are requested at a steady ~1 per 20 cycles across the whole batch instead of in
//! one burst: 630 -> 479 cycles per ring element for q = 3889. The same 648 prefetches issued at
//! once from the batch-fused accumulate loop are worth nothing. Prefetch distances of 1, 2 and 3
//! batches are equal within noise and 6 is worse; `prefetchnta` is a disaster (the A lines have to
//! survive in L2 until the accumulate reads them).
//!
//! # Measured (i7-11850H, one core, 2^18 F162 = 2^16 ring elements, 85 MB of A per prime)
//!
//! A single-prime commitment runs at 7.5 ms / 479 cycles per ring element for q = 3889 and
//! 7.9 ms / 499 for q = 9721; two primes off one slicing pass cost 483 cycles per ring element
//! and prime. Of the 479 cycles, 309 are the front end plus the transform (cache-resident) and 58
//! the base multiplication, leaving 112 of A stream that does not hide behind them; the floor is
//! max(DRAM 4.4 ms, compute 5.9 ms).
use crate::params::*;
use crate::simd::transpose_f162::BinaryIndex32;
use crate::simd::transpose_f162::slice_f162_into;
use crate::simd::vertical_bin_asm::{self as vb, BlockSink};
use crate::types::*;
use bin_fields::scalar::F162;
use core::arch::x86_64::*;

// =============================================================================================
// bounds
// =============================================================================================

/// `R = 2^16 mod q` (3312 for q = 3889, 7210 for q = 9721): the weight the high half of an i32
/// accumulator lane carries into the low half.
pub const fn r16(q: u16) -> i32 {
    (65536 % q as u32) as i32
}

/// Bound on one lane of the transform's output (`vertical_bin_asm`'s declared output bound,
/// 7.5 q for q = 3889 and 2.294 q for q = 9721).
pub const fn w_bound(q: u16) -> i64 {
    (vb::output_bound_milli_q(q) as i64 * q as i64) / 1000
}

/// Bound on one lane of A: the matrix is stored centered.
pub const fn a_bound(q: u16) -> i64 {
    ((q - 1) / 2) as i64
}

/// What one batch adds to one accumulator lane: four products.
pub const fn acc_per_batch(q: u16) -> i64 {
    4 * w_bound(q) * a_bound(q)
}

/// Bound on a lane straight after [`reduce_acc_i32`]: `|l| <= 2^15` and `|h + c| <= 2^15`, so
/// `|l + (h + c) R| <= 2^15 (1 + R)`.
pub const fn acc_after_reduce(q: u16) -> i64 {
    32768 * (1 + r16(q) as i64)
}

/// Batches accumulated between two fold-backs: the largest power of two P with
/// `acc_after_reduce + P * acc_per_batch <= i32::MAX`.
pub const fn red_period(q: u16) -> usize {
    if q == 3889 {
        8
    } else {
        4
    }
}

const fn fits(q: u16) -> bool {
    acc_after_reduce(q) + (red_period(q) as i64) * acc_per_batch(q) <= i32::MAX as i64
}
const _: () = assert!(fits(3889));
const _: () = assert!(fits(9721));

/// The exact fold-back, lane-wise (the scalar model of `reduce_vec`).
///
/// Write the i32 lane as `x = 2^16 h + u`, `h = x >> 16` (arithmetic), `u = x & 0xffff`, and let
/// `l` be `u` read as an i16, i.e. `u = l + 2^16 c` with `c = [bit 15 of x]`. Then
/// `x = 2^16 (h + c) + l`, so `x = l + (h + c) R (mod q)` exactly, and the result satisfies
/// `|l + (h + c) R| <= 2^15 + 2^15 R = acc_after_reduce(q)`.
///
/// In vector form this is `vpmaddwd(acc, [1, R])` — which computes `l + h R`, the two halves of
/// the lane read as the i16 pair `(l, h)` — plus `+ R` on the lanes whose bit 15 is set
/// (`vptestmd` + masked `vpaddd`): three uops per vector, one of them on the multiply port.
#[inline]
pub fn reduce_acc_i32(x: i32, q: u16) -> i32 {
    let l = ((x as u32) as u16) as i16 as i32;
    let h = x >> 16;
    let c = (x >> 15) & 1;
    l + (h + c) * r16(q)
}

/// The vector form of [`reduce_acc_i32`] (three uops, one on the multiply port).
///
/// # Safety
/// AVX-512 F/BW.
#[inline(always)]
pub unsafe fn reduce_vec<const Q: u16>(x: __m512i) -> __m512i {
    let k = _mm512_set1_epi32(1 | (r16(Q) << 16));
    let r = _mm512_madd_epi16(x, k);
    let m = _mm512_test_epi32_mask(x, _mm512_set1_epi32(0x8000));
    _mm512_mask_add_epi32(r, m, r, _mm512_set1_epi32(r16(Q)))
}

// =============================================================================================
// the packed accumulator
// =============================================================================================

/// Accumulator vectors per 27-slot block, and in total (21.5 KB).
pub const ACC_PER_BLK: usize = 14;
pub const ACC_VECS: usize = 24 * ACC_PER_BLK;

#[repr(C, align(64))]
pub struct Acc {
    pub v: [[i32; 16]; ACC_VECS],
}

impl Acc {
    pub fn zero() -> Box<Acc> {
        unsafe {
            let mut b = Box::<Acc>::new_uninit();
            core::ptr::write_bytes(b.as_mut_ptr() as *mut u8, 0, core::mem::size_of::<Acc>());
            b.assume_init()
        }
    }
}

/// Accumulator vector and lane group (0 = lanes 0..8, 1 = lanes 8..16) holding slot `s`.
pub const fn slot_lane(s: usize) -> (usize, usize) {
    let (bl, r) = (s / 27, s % 27);
    if r == 26 {
        (ACC_PER_BLK * bl + 13, 0)
    } else {
        (ACC_PER_BLK * bl + r / 2, r % 2)
    }
}

/// Two 16-lane product vectors folded to 8 lanes each and packed into one vector: lanes 0..8
/// carry `t0`, lanes 8..16 carry `t1` (two `vshufi64x2` and one `vpaddd`).
#[inline(always)]
unsafe fn pack2(t0: __m512i, t1: __m512i) -> __m512i {
    _mm512_add_epi32(
        _mm512_shuffle_i64x2::<0x44>(t0, t1),
        _mm512_shuffle_i64x2::<0xEE>(t0, t1),
    )
}

/// The two slots `w0, w1` times `a0, a1`, folded to 8 lanes each and packed into one vector.
#[inline(always)]
unsafe fn fold_pair(w0: *const i16, w1: *const i16, a0: *const i16, a1: *const i16) -> __m512i {
    let t0 = _mm512_madd_epi16(
        _mm512_load_si512(w0 as *const __m512i),
        _mm512_load_si512(a0 as *const __m512i),
    );
    let t1 = _mm512_madd_epi16(
        _mm512_load_si512(w1 as *const __m512i),
        _mm512_load_si512(a1 as *const __m512i),
    );
    pack2(t0, t1)
}

/// One 27-slot block of one batch into 14 accumulator vectors, and, with `PF`, the 27 A lines the
/// same block of a later batch will read.
///
/// # Safety
/// `w`, `a` and `acc` must be 64-byte aligned; `w` and `a` must cover 27 vectors, `acc`
/// [`ACC_PER_BLK`]. `apf` must be readable for 1728 bytes when `PF` (a prefetch of an unmapped
/// page is harmless on x86, but keep it inside).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac27<const PF: bool>(w: *const i16, a: *const i16, apf: *const i8, acc: *mut i32) {
    for p in 0..13 {
        let s = _mm512_load_si512(acc.add(16 * p) as *const __m512i);
        let d = fold_pair(w.add(64 * p), w.add(64 * p + 32), a.add(64 * p), a.add(64 * p + 32));
        _mm512_store_si512(acc.add(16 * p) as *mut __m512i, _mm512_add_epi32(s, d));
        if PF {
            _mm_prefetch(apf.add(128 * p), _MM_HINT_T1);
            _mm_prefetch(apf.add(128 * p + 64), _MM_HINT_T1);
        }
    }
    let s = _mm512_load_si512(acc.add(16 * 13) as *const __m512i);
    let d = fold_pair(w.add(32 * 26), w.add(32 * 26), a.add(32 * 26), a.add(32 * 26));
    _mm512_store_si512(acc.add(16 * 13) as *mut __m512i, _mm512_add_epi32(s, d));
    if PF {
        _mm_prefetch(apf.add(64 * 26), _MM_HINT_T1);
    }
}

/// One whole batch: the same 24 blocks, for the paths that do not consume the transform block
/// by block.
///
/// # Safety
/// See [`mac27`]; `w` and `a` must cover 648 vectors and `acc` [`ACC_VECS`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_batch<const PF: bool>(
    w: *const i16,
    a: *const i16,
    apf: *const i8,
    acc: *mut i32,
) {
    for bl in 0..24 {
        mac27::<PF>(
            w.add(32 * 27 * bl),
            a.add(32 * 27 * bl),
            apf.add(64 * 27 * bl),
            acc.add(16 * ACC_PER_BLK * bl),
        );
    }
}

/// The periodic fold-back over the whole accumulator.
///
/// # Safety
/// `acc` must be 64-byte aligned and cover [`ACC_VECS`] vectors.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn reduce_acc<const Q: u16>(acc: *mut i32) {
    for j in 0..ACC_VECS {
        let s = _mm512_load_si512(acc.add(16 * j) as *const __m512i);
        _mm512_store_si512(acc.add(16 * j) as *mut __m512i, reduce_vec::<Q>(s));
    }
}

/// Sum of the 8 lanes of every slot, reduced to [0, q). Done once per commitment in scalar code
/// (5184 i64 additions), so its cost is not measurable.
pub fn finish<const Q: u16>(acc: &Acc) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0u32; N];
    for s in 0..N {
        let (v, g) = slot_lane(s);
        let t: i64 = acc.v[v][8 * g..8 * g + 8].iter().map(|&x| x as i64).sum();
        y[s] = t.rem_euclid(q) as u32;
    }
    y
}

// =============================================================================================
// the sink
// =============================================================================================

/// The 1728-byte scratch every block of the transform is written into.
#[repr(C, align(64))]
struct Blk27([i16; 27 * 32]);

/// The multiply-accumulate sink: the kernel writes every 27-slot block into the same L1-resident
/// scratch, and this multiplies it into the accumulator against the batch's A rows (prefetching
/// a later batch's) before the next `asm!` block starts.
struct Mac {
    buf: *mut i16,
    a: *const i16,
    apf: *const i8,
    acc: *mut i32,
}

impl BlockSink for Mac {
    #[inline(always)]
    unsafe fn dst(&mut self, _blk: usize) -> *mut i16 {
        self.buf
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn block(&mut self, blk: usize, w: *const i16) {
        mac27::<true>(
            w,
            self.a.add(32 * 27 * blk),
            self.apf.add(64 * 27 * blk),
            self.acc.add(16 * ACC_PER_BLK * blk),
        );
    }
}

/// The same sink, plus a non-temporal copy of the block to a materialised transform.
///
/// The kernel still writes its 27 vectors into the L1 scratch and [`mac27`] still reads them from
/// there, so the accumulate is unchanged; the extra work is 27 `vmovntdq` per block, 41472 bytes
/// per batch, which leave no cache footprint and are absorbed by the write-combining buffers
/// while the transform of the next block runs.
struct MacKeep {
    buf: *mut i16,
    a: *const i16,
    apf: *const i8,
    acc: *mut i32,
    out: *mut i16,
}

impl BlockSink for MacKeep {
    #[inline(always)]
    unsafe fn dst(&mut self, _blk: usize) -> *mut i16 {
        self.buf
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn block(&mut self, blk: usize, w: *const i16) {
        mac27::<true>(
            w,
            self.a.add(32 * 27 * blk),
            self.apf.add(64 * 27 * blk),
            self.acc.add(16 * ACC_PER_BLK * blk),
        );
        let dst = self.out.add(32 * 27 * blk);
        for i in 0..27 {
            _mm512_stream_si512(
                dst.add(32 * i) as *mut __m512i,
                _mm512_load_si512(w.add(32 * i) as *const __m512i),
            );
        }
    }
}

// =============================================================================================
// entry points
// =============================================================================================

#[inline(always)]
unsafe fn chunk128(elems: &[F162], b: usize) -> &[F162; 128] {
    &*(elems.as_ptr().add(128 * b) as *const [F162; 128])
}

fn check(elems: &[F162], a: &[Batch32]) {
    assert_eq!(core::mem::size_of::<F162>(), 24, "F162 is not 24 bytes");
    assert_eq!(elems.len(), 128 * a.len(), "128 F162 (= 32 ring elements) per A batch");
}

/// Distance, in batches, of the A prefetch.
pub const PF_DIST: usize = 1;

// =============================================================================================
// quadratic-slot limbs
// =============================================================================================
//
// For q in `params::QS_QUAD` the ring does not split completely: the transform ends at 324
// quadratic leaves `Z_q[X]/(X^2 - c_j)`, `c_j = psi'^QUAD_SLOT_EXP[j]`, and leaf j occupies rows
// `2j` (constant term) and `2j+1` (X coefficient) of the 648-row output. A slot product is the
// quadratic product
//
//     (a_0 + a_1 X)(b_0 + b_1 X) = (a_0 b_0 + c_j a_1 b_1) + (a_0 b_1 + a_1 b_0) X,
//
// and the commitment wants its sum over the ring elements, so the three sums
//
//     P_0 = sum_i a_0 b_0,   P_1 = sum_i a_1 b_1,   P_2 = sum_i (a_0 b_1 + a_1 b_0)
//
// are accumulated raw and combined once per leaf at the end:
// `y[2j] = P_0 + c_j P_1`, `y[2j+1] = P_2`.
//
// **Karatsuba.** `P_2` is one `vpmaddwd` instead of two when it is formed as
// `(a_0 + a_1)(b_0 + b_1) - P_0 - P_1`: one `vpaddw` on each side, three multiply-port uops per
// leaf against four. The A side always fits (`|b_0 + b_1| <= q - 1`), but the W side is the
// kernel's *lazily reduced* output, `|a_k| <= output_bound(q)` = 4.87 q / 5.13 q / 1.94 q, so the
// sum fits an i16 lane only for q = 2917 (2 * 4.87 q = 28412 < 2^15; 4861 and 12637 would reach
// 49874 and 49032). Those two therefore accumulate `a_0 b_1` and `a_1 b_0` into `P_2` with two
// `vpmaddwd` — the same three accumulators, the same combine with the Karatsuba correction
// dropped, one more multiply-port uop per leaf. [`karatsuba`] is that condition.
//
// **The packed accumulator** is the splitting one's, per 18-row block: one vector holds `P_0` of a
// leaf in lanes 0..8 and `P_1` in lanes 8..16 (9 per block), and one vector holds the `P_2` of two
// leaves (5 per block, the ninth leaf duplicated into the upper half as slot 26 is in `mac27`),
// so a block costs 14 accumulator vectors exactly as a 27-slot block of the splitting kernel
// does. 36 blocks: 32.3 KB, L1-resident. A lane again carries four products per batch.

use crate::simd::vertical_bin_quad::{self as vq, BlockSink as QBlockSink};

/// Bound on one lane of the quadratic kernel's output ([`vq::output_bound`]).
pub const fn w_bound_quad(q: u16) -> i64 {
    vq::output_bound(q) as i64
}

/// Can the Karatsuba sum `a_0 + a_1` of two output rows live in an i16 lane? (2917: yes.)
pub const fn karatsuba(q: u16) -> bool {
    2 * w_bound_quad(q) <= 32767
}
const _: () = assert!(karatsuba(2917));
const _: () = assert!(!karatsuba(4861) && !karatsuba(12637));

/// What one batch adds to a lane of the `P_0 | P_1` accumulator: four products of `|W| |A|`.
pub const fn acc_per_batch_quad01(q: u16) -> i64 {
    4 * w_bound_quad(q) * a_bound(q)
}

/// What one batch adds to a lane of the `P_2` accumulator: four products of `2|W|` by
/// `|b_0 + b_1| <= q - 1` with Karatsuba, eight of `|W| |A|` without.
pub const fn acc_per_batch_quad2(q: u16) -> i64 {
    if karatsuba(q) {
        4 * (2 * w_bound_quad(q)) * (2 * a_bound(q))
    } else {
        8 * w_bound_quad(q) * a_bound(q)
    }
}

/// The largest power of two P with `acc_after_reduce(q) + P * per <= i32::MAX`.
pub const fn period_for(q: u16, per: i64) -> usize {
    let mut p = 1usize;
    while acc_after_reduce(q) + 2 * (p as i64) * per <= i32::MAX as i64 {
        p *= 2;
    }
    p
}

/// Batches between two fold-backs of the `P_0 | P_1` accumulator (16 / 8 / 2).
pub const fn red_period_quad01(q: u16) -> usize {
    period_for(q, acc_per_batch_quad01(q))
}
/// Batches between two fold-backs of the `P_2` accumulator (4 / 4 / 1).
pub const fn red_period_quad2(q: u16) -> usize {
    period_for(q, acc_per_batch_quad2(q))
}

const fn fits_quad(q: u16) -> bool {
    acc_after_reduce(q) + (red_period_quad01(q) as i64) * acc_per_batch_quad01(q)
        <= i32::MAX as i64
        && acc_after_reduce(q) + (red_period_quad2(q) as i64) * acc_per_batch_quad2(q)
            <= i32::MAX as i64
}
const _: () = assert!(fits_quad(2917) && fits_quad(4861) && fits_quad(12637));

/// Accumulator vectors per 18-row block: 9 for `P_0 | P_1`, 5 for `P_2`.
pub const QACC01_PER_BLK: usize = 9;
pub const QACC2_PER_BLK: usize = 5;
/// Blocks the quadratic kernel hands out (36 x 18 rows = 648).
pub const QBLOCKS: usize = 36;

#[repr(C, align(64))]
pub struct QuadAcc {
    /// `p01[9 blk + j]`: lanes 0..8 are `P_0` of leaf `9 blk + j`, lanes 8..16 its `P_1`.
    pub p01: [[i32; 16]; QBLOCKS * QACC01_PER_BLK],
    /// `p2[5 blk + j/2]`: lanes `8 (j % 2) ..` are `P_2` of leaf `9 blk + j` (leaf 8 in lanes
    /// 0..8, its duplicate in 8..16).
    pub p2: [[i32; 16]; QBLOCKS * QACC2_PER_BLK],
}

impl QuadAcc {
    pub fn zero() -> Box<QuadAcc> {
        unsafe {
            let mut b = Box::<QuadAcc>::new_uninit();
            core::ptr::write_bytes(b.as_mut_ptr() as *mut u8, 0, core::mem::size_of::<QuadAcc>());
            b.assume_init()
        }
    }
}

/// One leaf: the packed `P_0 | P_1` contribution and the `P_2` one, off four loads.
///
/// The four vectors are loaded once and used by both products — with the accumulator stores in
/// between, LLVM has to assume they alias and reloads them, which measures 30 % of the sink.
#[inline(always)]
unsafe fn leaf<const Q: u16>(w: *const i16, a: *const i16) -> (__m512i, __m512i) {
    let w0 = _mm512_load_si512(w as *const __m512i);
    let w1 = _mm512_load_si512(w.add(32) as *const __m512i);
    let a0 = _mm512_load_si512(a as *const __m512i);
    let a1 = _mm512_load_si512(a.add(32) as *const __m512i);
    let p01 = pack2(_mm512_madd_epi16(w0, a0), _mm512_madd_epi16(w1, a1));
    let p2 = if karatsuba(Q) {
        _mm512_madd_epi16(_mm512_add_epi16(w0, w1), _mm512_add_epi16(a0, a1))
    } else {
        _mm512_add_epi32(_mm512_madd_epi16(w0, a1), _mm512_madd_epi16(w1, a0))
    };
    (p01, p2)
}

/// One 18-row block (9 leaves) of one batch into its 14 accumulator vectors, and, with `PF`, the
/// 18 A lines the same block of a later batch will read.
///
/// # Safety
/// `w`, `a`, `acc01` and `acc2` must be 64-byte aligned; `w` and `a` must cover 18 vectors,
/// `acc01` [`QACC01_PER_BLK`] and `acc2` [`QACC2_PER_BLK`]. `apf` must be readable for 1152 bytes
/// when `PF`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_quad18<const Q: u16, const PF: bool>(
    w: *const i16,
    a: *const i16,
    apf: *const i8,
    acc01: *mut i32,
    acc2: *mut i32,
) {
    for m in 0..4 {
        let (j0, j1) = (2 * m, 2 * m + 1);
        let (d0, t0) = leaf::<Q>(w.add(64 * j0), a.add(64 * j0));
        let (d1, t1) = leaf::<Q>(w.add(64 * j1), a.add(64 * j1));
        let s0 = _mm512_load_si512(acc01.add(16 * j0) as *const __m512i);
        let s1 = _mm512_load_si512(acc01.add(16 * j1) as *const __m512i);
        let s2 = _mm512_load_si512(acc2.add(16 * m) as *const __m512i);
        _mm512_store_si512(acc01.add(16 * j0) as *mut __m512i, _mm512_add_epi32(s0, d0));
        _mm512_store_si512(acc01.add(16 * j1) as *mut __m512i, _mm512_add_epi32(s1, d1));
        _mm512_store_si512(
            acc2.add(16 * m) as *mut __m512i,
            _mm512_add_epi32(s2, pack2(t0, t1)),
        );
        if PF {
            for i in 0..4 {
                _mm_prefetch(apf.add(64 * (4 * m + i)), _MM_HINT_T1);
            }
        }
    }
    let (d, t) = leaf::<Q>(w.add(64 * 8), a.add(64 * 8));
    let s0 = _mm512_load_si512(acc01.add(16 * 8) as *const __m512i);
    let s2 = _mm512_load_si512(acc2.add(16 * 4) as *const __m512i);
    _mm512_store_si512(acc01.add(16 * 8) as *mut __m512i, _mm512_add_epi32(s0, d));
    _mm512_store_si512(
        acc2.add(16 * 4) as *mut __m512i,
        _mm512_add_epi32(s2, pack2(t, t)),
    );
    if PF {
        _mm_prefetch(apf.add(64 * 16), _MM_HINT_T1);
        _mm_prefetch(apf.add(64 * 17), _MM_HINT_T1);
    }
}

/// One whole batch through [`mac_quad18`], for the paths that do not consume the transform block
/// by block.
///
/// # Safety
/// See [`mac_quad18`]; `w` and `a` must cover 648 vectors and the accumulators a whole [`QuadAcc`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_quad_batch<const Q: u16, const PF: bool>(
    w: *const i16,
    a: *const i16,
    apf: *const i8,
    acc01: *mut i32,
    acc2: *mut i32,
) {
    for bl in 0..QBLOCKS {
        mac_quad18::<Q, PF>(
            w.add(32 * 18 * bl),
            a.add(32 * 18 * bl),
            apf.add(64 * 18 * bl),
            acc01.add(16 * QACC01_PER_BLK * bl),
            acc2.add(16 * QACC2_PER_BLK * bl),
        );
    }
}

/// The fold-back over both quadratic accumulators.
///
/// # Safety
/// AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn reduce_quad_acc<const Q: u16>(acc: &mut QuadAcc) {
    reduce_quad_part::<Q>(acc.p01.as_mut_ptr() as *mut i32, QBLOCKS * QACC01_PER_BLK);
    reduce_quad_part::<Q>(acc.p2.as_mut_ptr() as *mut i32, QBLOCKS * QACC2_PER_BLK);
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn reduce_quad_part<const Q: u16>(acc: *mut i32, vecs: usize) {
    for j in 0..vecs {
        let p = acc.add(16 * j);
        _mm512_store_si512(
            p as *mut __m512i,
            reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)),
        );
    }
}

/// The three sums per leaf combined into the 648 rows of the commitment, reduced to `[0, q)`:
/// `y[2j] = P_0 + c_j P_1`, `y[2j+1] = P_2` (minus `P_0 + P_1` when the Karatsuba product was
/// accumulated). Scalar, once per commitment.
pub fn finish_quad<const Q: u16>(acc: &QuadAcc) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0u32; N];
    for blk in 0..QBLOCKS {
        for j in 0..QACC01_PER_BLK {
            let leaf = QACC01_PER_BLK * blk + j;
            let v = &acc.p01[QACC01_PER_BLK * blk + j];
            let p0: i64 = v[0..8].iter().map(|&x| x as i64).sum();
            let p1: i64 = v[8..16].iter().map(|&x| x as i64).sum();
            let g = j % 2;
            let w = &acc.p2[QACC2_PER_BLK * blk + j / 2];
            let mut p2: i64 = w[8 * g..8 * g + 8].iter().map(|&x| x as i64).sum();
            if karatsuba(Q) {
                p2 -= p0 + p1;
            }
            let c = ParamsQ::<Q>::LEAF_C[leaf] as i64;
            y[2 * leaf] = (p0 + c % q * (p1 % q)).rem_euclid(q) as u32;
            y[2 * leaf + 1] = p2.rem_euclid(q) as u32;
        }
    }
    y
}

/// The quadratic multiply-accumulate sink: the same L1 scratch for every block, multiplied into
/// the three accumulators against the block's A rows (prefetching a later batch's) the moment the
/// kernel has stored it.
struct MacQ<const Q: u16> {
    buf: *mut i16,
    a: *const i16,
    apf: *const i8,
    acc01: *mut i32,
    acc2: *mut i32,
}

impl<const Q: u16> QBlockSink for MacQ<Q> {
    #[inline(always)]
    unsafe fn dst(&mut self, _blk: usize) -> *mut i16 {
        self.buf
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn block(&mut self, blk: usize, w: *const i16) {
        mac_quad18::<Q, true>(
            w,
            self.a.add(32 * 18 * blk),
            self.apf.add(64 * 18 * blk),
            self.acc01.add(16 * QACC01_PER_BLK * blk),
            self.acc2.add(16 * QACC2_PER_BLK * blk),
        );
    }
}

/// One batch of a quadratic limb: the transform consumed block by block into `acc`, with the A
/// prefetch one batch ahead and the two fold-backs on their own periods.
///
/// # Safety
/// `idx` is the batch's index rows; `a` and `apf` are 648-vector A rows; `buf` is 18 writable
/// 64-byte aligned vectors.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn quad_batch<const Q: u16>(
    idx: &BinaryIndex32,
    a: *const i16,
    apf: *const i8,
    buf: *mut i16,
    p01: *mut i32,
    p2: *mut i32,
    done: usize,
) {
    vq::ntt_quad_bin_batch32_sink::<Q, _>(
        idx,
        &mut MacQ::<Q> { buf, a, apf, acc01: p01, acc2: p2 },
    );
    if done % red_period_quad01(Q) == 0 {
        reduce_quad_part::<Q>(p01, QBLOCKS * QACC01_PER_BLK);
    }
    if done % red_period_quad2(Q) == 0 {
        reduce_quad_part::<Q>(p2, QBLOCKS * QACC2_PER_BLK);
    }
}

// =============================================================================================
// the multi-limb commitment
// =============================================================================================

/// One limb of a commitment: its prime, whether that prime's `R_648` ends in quadratic leaves,
/// and its matrix `A` in the vertical layout.
#[derive(Clone, Copy)]
pub struct Limb<'a> {
    pub q: u16,
    pub quad: bool,
    pub a: &'a [Batch32],
}

enum LimbAcc {
    Split(Box<Acc>),
    Quad(Box<QuadAcc>),
}

impl LimbAcc {
    fn new(l: &Limb) -> LimbAcc {
        if l.quad {
            LimbAcc::Quad(QuadAcc::zero())
        } else {
            LimbAcc::Split(Acc::zero())
        }
    }
    fn clear(&mut self) {
        unsafe {
            match self {
                LimbAcc::Split(a) => core::ptr::write_bytes(
                    a.as_mut() as *mut Acc as *mut u8,
                    0,
                    core::mem::size_of::<Acc>(),
                ),
                LimbAcc::Quad(a) => core::ptr::write_bytes(
                    a.as_mut() as *mut QuadAcc as *mut u8,
                    0,
                    core::mem::size_of::<QuadAcc>(),
                ),
            }
        }
    }
}

/// The accumulators of one limb list, allocated once and reused for every chunk a key commits to
/// (21.5 KB per splitting limb, 32.3 KB per quadratic one; they are cleared, not reallocated).
pub struct Scratch {
    accs: Vec<LimbAcc>,
}

impl Scratch {
    pub fn new(limbs: &[Limb]) -> Scratch {
        Scratch {
            accs: limbs.iter().map(LimbAcc::new).collect(),
        }
    }
}

/// One split limb's batch, with the transform kept when `out` is given.
///
/// # Safety
/// As [`quad_batch`]; `out` is 648 writable vectors.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn split_batch<const Q: u16, const KEEP: bool>(
    idx: &BinaryIndex32,
    a: *const i16,
    apf: *const i8,
    buf: *mut i16,
    acc: *mut i32,
    out: *mut i16,
    done: usize,
) {
    if KEEP {
        vb::ntt_bin_batch32_sink::<Q, _>(
            idx,
            &mut MacKeep { buf, a, apf, acc, out },
        );
    } else {
        vb::ntt_bin_batch32_sink::<Q, _>(idx, &mut Mac { buf, a, apf, acc });
    }
    if done % red_period(Q) == 0 {
        reduce_acc::<Q>(acc);
    }
}

/// The commitment over a list of limbs: one front end (the index rows depend neither on q nor on
/// the tree) and one kernel pass per limb per batch, each with its own accumulator, fold-back
/// period and A prefetch.
///
/// `limbs[0]` is the base limb; `w`, when given, receives the transform of every ring element
/// modulo `limbs[0].q` (non-temporal stores out of that limb's block sink), which is what
/// [`crate::fold`] consumes. `out` receives one 648-row commitment per limb, in `[0, q)`.
pub fn commit_limbs_into(
    elems: &[F162],
    limbs: &[Limb],
    w: Option<&mut [Batch32]>,
    st: &mut Scratch,
    out: &mut [[u32; N]],
) {
    assert!(!limbs.is_empty(), "at least the base limb");
    let nb = limbs[0].a.len();
    check(elems, limbs[0].a);
    for l in limbs {
        assert_eq!(l.a.len(), nb, "every limb's A has one batch per 32 ring elements");
    }
    assert!(
        st.accs.len() == limbs.len() && out.len() == limbs.len(),
        "the scratch and the output do not match this limb list"
    );
    assert!(limbs.len() <= MAX_LIMBS, "at most {MAX_LIMBS} limbs");
    let keep = w.map(|w| {
        assert_eq!(w.len(), nb, "one output batch per A batch");
        w.as_mut_ptr()
    });
    for a in st.accs.iter_mut() {
        a.clear();
    }
    unsafe { commit_limbs_core(elems, limbs, keep, st, out) };
}

/// The base limb and the four [`crate::Modulus`]s: the widest limb list there is.
pub const MAX_LIMBS: usize = 5;

/// One limb, resolved: everything the batch loop needs as plain words, so that the loop does not
/// walk a `Vec` of boxed accumulators per batch.
#[derive(Clone, Copy)]
struct Run {
    q: u16,
    quad: bool,
    a: *const Batch32,
    nb: usize,
    acc: *mut i32,
    acc2: *mut i32,
    keep: *mut Batch32,
}

/// The batch loop of [`commit_limbs_into`], with the whole feature set enabled so that the
/// per-limb entry points inline into it exactly as the two-prime loop they replace did.
///
/// # Safety
/// The arguments are [`commit_limbs_into`]'s, already checked; `keep` is `nb` writable batches.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn commit_limbs_core(
    elems: &[F162],
    limbs: &[Limb],
    keep: Option<*mut Batch32>,
    st: &mut Scratch,
    out: &mut [[u32; N]],
) {
    let nb = limbs[0].a.len();
    let mut plan = [Run {
        q: 0,
        quad: false,
        a: core::ptr::null(),
        nb,
        acc: core::ptr::null_mut(),
        acc2: core::ptr::null_mut(),
        keep: core::ptr::null_mut(),
    }; MAX_LIMBS];
    for (li, (l, acc)) in limbs.iter().zip(st.accs.iter_mut()).enumerate() {
        let (p, p2) = match acc {
            LimbAcc::Split(a) => (a.v.as_mut_ptr() as *mut i32, core::ptr::null_mut()),
            LimbAcc::Quad(a) => (
                a.p01.as_mut_ptr() as *mut i32,
                a.p2.as_mut_ptr() as *mut i32,
            ),
        };
        plan[li] = Run {
            q: l.q,
            quad: l.quad,
            a: l.a.as_ptr(),
            nb,
            acc: p,
            acc2: p2,
            keep: if li == 0 {
                keep.unwrap_or(core::ptr::null_mut())
            } else {
                core::ptr::null_mut()
            },
        };
    }
    let runs = &plan[..limbs.len()];

    let mut idx = BinaryIndex32::zero();
    let mut buf: core::mem::MaybeUninit<Blk27> = core::mem::MaybeUninit::uninit();
    let bp = buf.as_mut_ptr() as *mut i16;
    for b in 0..nb {
        slice_f162_into(chunk128(elems, b), &mut idx);
        for r in runs.iter() {
            let cur = (*r.a.add(b)).v.as_ptr() as *const i16;
            let nxt = (*r.a.add((b + PF_DIST).min(r.nb - 1))).v.as_ptr() as *const i8;
            if r.quad {
                match r.q {
                    2917 => quad_batch::<2917>(&idx, cur, nxt, bp, r.acc, r.acc2, b + 1),
                    4861 => quad_batch::<4861>(&idx, cur, nxt, bp, r.acc, r.acc2, b + 1),
                    12637 => quad_batch::<12637>(&idx, cur, nxt, bp, r.acc, r.acc2, b + 1),
                    _ => unreachable!("no quadratic kernel for q = {}", r.q),
                }
            } else {
                let o = if r.keep.is_null() {
                    core::ptr::null_mut()
                } else {
                    (*r.keep.add(b)).v.as_mut_ptr() as *mut i16
                };
                match (r.q, o.is_null()) {
                    (3889, true) => split_batch::<3889, false>(&idx, cur, nxt, bp, r.acc, o, b + 1),
                    (3889, false) => split_batch::<3889, true>(&idx, cur, nxt, bp, r.acc, o, b + 1),
                    (9721, true) => split_batch::<9721, false>(&idx, cur, nxt, bp, r.acc, o, b + 1),
                    (9721, false) => split_batch::<9721, true>(&idx, cur, nxt, bp, r.acc, o, b + 1),
                    _ => unreachable!("no splitting kernel for q = {}", r.q),
                }
            }
        }
    }
    if keep.is_some() {
        _mm_sfence();
        for b in 0..nb {
            (*keep.unwrap().add(b)).representation = Representation::Ntt;
        }
    }
    for (li, l) in limbs.iter().enumerate() {
        out[li] = match &st.accs[li] {
            LimbAcc::Split(a) => match l.q {
                3889 => finish::<3889>(a),
                9721 => finish::<9721>(a),
                _ => unreachable!(),
            },
            LimbAcc::Quad(a) => match l.q {
                2917 => finish_quad::<2917>(a),
                4861 => finish_quad::<4861>(a),
                12637 => finish_quad::<12637>(a),
                _ => unreachable!(),
            },
        };
    }
}

/// [`commit_limbs_into`] allocating its own scratch and output — one chunk, for tests and callers
/// that commit once.
pub fn commit_limbs(
    elems: &[F162],
    limbs: &[Limb],
    w: Option<&mut [Batch32]>,
) -> Vec<[u32; N]> {
    let mut st = Scratch::new(limbs);
    let mut out = vec![[0u32; N]; limbs.len()];
    commit_limbs_into(elems, limbs, w, &mut st, &mut out);
    out
}

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
//! element (q = 3889 / 9721): materialising the whole transform first ([`commit_unfused`])
//! 874 / 908 cycles — it adds 85 MB of writes and 85 MB of reads and is DRAM-bound; one 41 KB
//! buffer per batch ([`commit_batch_fused`]) 645 / 687; per block 630 / 664.
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
//! [`commit`] runs at 7.5 ms / 479 cycles per ring element for q = 3889 and 7.9 ms / 499 for
//! q = 9721; [`commit_2q`] does both primes off one slicing pass at 483 cycles per ring element
//! and prime. Of the 479 cycles, 309 are the front end plus the transform (cache-resident) and 58
//! the base multiplication, leaving 112 of A stream that does not hide behind them; the floor is
//! max(DRAM 4.4 ms, compute 5.9 ms).
use crate::params::*;
use crate::simd::transpose::BinaryIndex32;
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

#[inline(always)]
unsafe fn reduce_vec<const Q: u16>(x: __m512i) -> __m512i {
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
    _mm512_add_epi32(
        _mm512_shuffle_i64x2::<0x44>(t0, t1),
        _mm512_shuffle_i64x2::<0xEE>(t0, t1),
    )
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
struct Mac<const PF: bool> {
    buf: *mut i16,
    a: *const i16,
    apf: *const i8,
    acc: *mut i32,
}

impl<const PF: bool> BlockSink for Mac<PF> {
    #[inline(always)]
    unsafe fn dst(&mut self, _blk: usize) -> *mut i16 {
        self.buf
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn block(&mut self, blk: usize, w: *const i16) {
        mac27::<PF>(
            w,
            self.a.add(32 * 27 * blk),
            self.apf.add(64 * 27 * blk),
            self.acc.add(16 * ACC_PER_BLK * blk),
        );
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

/// A row pointer of batch `b`, and of batch `b + d` clamped to the last one (the prefetch target).
#[inline(always)]
fn rows_d(a: &[Batch32], b: usize, d: usize) -> (*const i16, *const i8) {
    let cur = a[b].v.as_ptr() as *const i16;
    let nxt = a[(b + d).min(a.len() - 1)].v.as_ptr() as *const i8;
    (cur, nxt)
}

/// Distance, in batches, of the A prefetch that [`commit`] uses.
pub const PF_DIST: usize = 1;

/// The commitment: the transform consumed block by block, with the packed accumulator and the A
/// prefetch.
pub fn commit<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    commit_block_fused::<Q, true, PF_DIST>(elems, a, red_period(Q))
}

/// Both primes off one slicing pass of the input (the index rows do not depend on q).
pub fn commit_2q(elems: &[F162], a3889: &[Batch32], a9721: &[Batch32]) -> ([u32; N], [u32; N]) {
    check(elems, a3889);
    assert_eq!(a3889.len(), a9721.len());
    let mut acc_a = Acc::zero();
    let mut acc_b = Acc::zero();
    let mut idx = BinaryIndex32::zero();
    let mut buf: core::mem::MaybeUninit<Blk27> = core::mem::MaybeUninit::uninit();
    let bp = buf.as_mut_ptr() as *mut i16;
    let (pa, pb) = (acc_a.v.as_mut_ptr() as *mut i32, acc_b.v.as_mut_ptr() as *mut i32);
    unsafe {
        for b in 0..a3889.len() {
            slice_f162_into(chunk128(elems, b), &mut idx);
            let (a, apf) = rows_d(a3889, b, PF_DIST);
            vb::ntt_bin_batch32_sink::<3889, false, _>(
                &idx,
                &mut Mac::<true> { buf: bp, a, apf, acc: pa },
            );
            if (b + 1) % red_period(3889) == 0 {
                reduce_acc::<3889>(pa);
            }
            let (a, apf) = rows_d(a9721, b, PF_DIST);
            vb::ntt_bin_batch32_sink::<9721, false, _>(
                &idx,
                &mut Mac::<true> { buf: bp, a, apf, acc: pb },
            );
            if (b + 1) % red_period(9721) == 0 {
                reduce_acc::<9721>(pb);
            }
        }
    }
    (finish::<3889>(&acc_a), finish::<9721>(&acc_b))
}

/// The transform materialised into `w` first (85 MB for 2^16 elements, non-temporal stores), then
/// read back by a separate multiply-accumulate pass.
pub fn commit_unfused<const Q: u16, const PF: bool>(
    elems: &[F162],
    a: &[Batch32],
    w: &mut [Batch32],
    period: usize,
) -> [u32; N] {
    check(elems, a);
    assert_eq!(w.len(), a.len());
    crate::simd::ntt_f162::ntt_f162::<Q>(elems, w);
    let mut acc = Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    unsafe {
        for b in 0..a.len() {
            let (cur, nxt) = rows_d(a, b, 1);
            mac_batch::<PF>(w[b].v.as_ptr() as *const i16, cur, nxt, ap);
            if (b + 1) % period == 0 {
                reduce_acc::<Q>(ap);
            }
        }
    }
    finish::<Q>(&acc)
}

/// Each batch's transform written to one 41 KB buffer that stays in L1/L2 and multiplied by the
/// batch's A rows immediately after.
pub fn commit_batch_fused<const Q: u16, const PF: bool>(
    elems: &[F162],
    a: &[Batch32],
    period: usize,
) -> [u32; N] {
    check(elems, a);
    let mut acc = Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    let mut buf = Batch32::zero(Representation::Ntt);
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..a.len() {
            slice_f162_into(chunk128(elems, b), &mut idx);
            vb::ntt_bin_batch32::<Q>(&idx, &mut buf);
            let (cur, nxt) = rows_d(a, b, 1);
            mac_batch::<PF>(buf.v.as_ptr() as *const i16, cur, nxt, ap);
            if (b + 1) % period == 0 {
                reduce_acc::<Q>(ap);
            }
        }
    }
    finish::<Q>(&acc)
}

/// The accumulate inserted between the kernel's `asm!` blocks through [`Mac`], with the A prefetch
/// aimed `DIST` batches ahead. `commit_block_fused::<Q, true, PF_DIST>` is [`commit`].
pub fn commit_block_fused<const Q: u16, const PF: bool, const DIST: usize>(
    elems: &[F162],
    a: &[Batch32],
    period: usize,
) -> [u32; N] {
    check(elems, a);
    let mut acc = Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    let mut idx = BinaryIndex32::zero();
    let mut buf: core::mem::MaybeUninit<Blk27> = core::mem::MaybeUninit::uninit();
    let bp = buf.as_mut_ptr() as *mut i16;
    unsafe {
        for b in 0..a.len() {
            slice_f162_into(chunk128(elems, b), &mut idx);
            let (cur, apf) = rows_d(a, b, DIST);
            vb::ntt_bin_batch32_sink::<Q, false, _>(
                &idx,
                &mut Mac::<PF> { buf: bp, a: cur, apf, acc: ap },
            );
            if (b + 1) % period == 0 {
                reduce_acc::<Q>(ap);
            }
        }
    }
    finish::<Q>(&acc)
}

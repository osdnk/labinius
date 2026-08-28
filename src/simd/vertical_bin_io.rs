//! Everything *outside* the binary NTT kernel proper: alternative entry points, output
//! allocation and the batch-to-batch pipeline.
//!
//! `vertical_bin::ntt_bin_polys` runs, per batch of 32 polynomials, `transpose::slice_polys_idx_into`
//! (31 cycles / polynomial) followed by `ntt_bin_batch32_nt` (297 / 330). For 2^18 polynomials the
//! measured total is ~365 / ~391, i.e. ~37 / ~31 cycles more than the cache-resident sum; that
//! remainder is the 23 MB input stream, the 340 MB non-temporal output stream, the page walks on
//! both, and the loop itself. This module attacks exactly that remainder:
//!
//! * [`ntt_bin_polys_2q`] — both primes from one transpose (the index rows are q-independent).
//! * [`ntt_bin_idx`] / [`ntt_bin_nib`] — the producer hands over already-transposed input
//!   ([`BinaryIndex32`], 10 368 B per 32 polynomials, or the nibble form [`BinaryBatch32`],
//!   5 184 B, expanded by a 3-uop-per-row prologue), so the transpose disappears from the NTT at
//!   the price of input bandwidth.
//! * [`Batches`] — an mmap-backed output buffer with `MADV_HUGEPAGE`, which removes ~85 000
//!   4 KB store-TLB entries from the 340 MB output stream.
//! * prefetch variants of every driver, and [`ntt_bin_accumulate`], the streamed form with a
//!   realistic consumer that prefetches the *next* batch's operand while multiplying this one.
use crate::simd::pointwise;
use crate::simd::transpose::{self, BinaryIndex32};
use crate::simd::vertical_bin as vb;
use crate::types::*;
use core::arch::x86_64::*;

pub const N: usize = crate::params::N;

#[inline(always)]
unsafe fn chunk32(polys: &[BinaryPoly], b: usize) -> &[BinaryPoly; 32] {
    &*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32])
}

#[inline(always)]
unsafe fn pf<const LINES: usize>(p: *const u8) {
    let mut i = 0;
    while i < LINES {
        _mm_prefetch::<_MM_HINT_T0>(p.add(64 * i) as *const i8);
        i += 1;
    }
}

// ---------------------------------------------------------------------------------------------
// 1. both primes from one transpose
// ---------------------------------------------------------------------------------------------

/// Transpose once, run both kernels: `out_a` gets `QA`, `out_b` gets `QB`. Saves the 31-cycle
/// transpose on the second prime.
pub fn ntt_bin_polys_2q<const QA: u16, const QB: u16>(
    polys: &[BinaryPoly],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(polys.len(), 32 * out_a.len());
    assert_eq!(out_a.len(), out_b.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..out_a.len() {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32_nt::<QA>(&idx, out_a.get_unchecked_mut(b));
            vb::ntt_bin_batch32_nt::<QB>(&idx, out_b.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// Same, with a prefetch of the next batch's 32 `BinaryPoly` (44 lines) at the head of the loop.
pub fn ntt_bin_polys_2q_pf<const QA: u16, const QB: u16, const DIST: usize>(
    polys: &[BinaryPoly],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(polys.len(), 32 * out_a.len());
    assert_eq!(out_a.len(), out_b.len());
    let nb = out_a.len();
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..nb {
            if b + DIST < nb {
                pf::<44>(polys.as_ptr().add(32 * (b + DIST)) as *const u8);
            }
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32_nt::<QA>(&idx, out_a.get_unchecked_mut(b));
            vb::ntt_bin_batch32_nt::<QB>(&idx, out_b.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

// ---------------------------------------------------------------------------------------------
// 2. producer-side index rows
// ---------------------------------------------------------------------------------------------

/// `polys -> BinaryIndex32` for a whole slice (what a producer would emit instead of `BinaryPoly`).
pub fn transpose_polys(polys: &[BinaryPoly], idx: &mut [BinaryIndex32]) {
    assert_eq!(polys.len(), 32 * idx.len());
    unsafe {
        for (b, o) in idx.iter_mut().enumerate() {
            transpose::slice_polys_idx_into(chunk32(polys, b), o);
        }
    }
}

/// `polys -> BinaryBatch32` (nibble form, 5 184 B per 32) for a whole slice.
pub fn nibble_polys(polys: &[BinaryPoly], nib: &mut [BinaryBatch32]) {
    assert_eq!(polys.len(), 32 * nib.len());
    unsafe {
        for (b, o) in nib.iter_mut().enumerate() {
            *o = transpose::slice_polys(chunk32(polys, b));
        }
    }
}

/// NTT straight from producer-side index rows: no transpose at all.
pub fn ntt_bin_idx<const Q: u16>(idxs: &[BinaryIndex32], out: &mut [Batch32]) {
    assert_eq!(idxs.len(), out.len());
    unsafe {
        for (i, o) in idxs.iter().zip(out.iter_mut()) {
            vb::ntt_bin_batch32_nt::<Q>(i, o);
        }
        _mm_sfence();
    }
}

/// Same with a software prefetch of `LINES` lines of the index rows `DIST` batches ahead.
pub fn ntt_bin_idx_pf<const Q: u16, const DIST: usize, const LINES: usize>(
    idxs: &[BinaryIndex32],
    out: &mut [Batch32],
) {
    assert_eq!(idxs.len(), out.len());
    let nb = idxs.len();
    unsafe {
        for b in 0..nb {
            if b + DIST < nb {
                pf::<LINES>(idxs.get_unchecked(b + DIST).rows.as_ptr() as *const u8);
            }
            vb::ntt_bin_batch32_nt::<Q>(idxs.get_unchecked(b), out.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// Both primes from producer-side index rows (one 85 MB input stream, two 340 MB outputs).
pub fn ntt_bin_idx_2q<const QA: u16, const QB: u16>(
    idxs: &[BinaryIndex32],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(idxs.len(), out_a.len());
    assert_eq!(idxs.len(), out_b.len());
    unsafe {
        for b in 0..idxs.len() {
            let i = idxs.get_unchecked(b);
            vb::ntt_bin_batch32_nt::<QA>(i, out_a.get_unchecked_mut(b));
            vb::ntt_bin_batch32_nt::<QB>(i, out_b.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// `vpermb` index that duplicates byte p of the low half into bytes 2p, 2p+1.
const fn dup_idx() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p] = p as u8;
        t[2 * p + 1] = p as u8;
        p += 1;
    }
    t
}
/// `[0, 16, 0, 16, ...]`, the `+16` of the high-byte index of each pair.
const fn plus16() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p + 1] = 16;
        p += 1;
    }
    t
}

#[repr(C, align(64))]
struct A64<T>(T);
static DUP_IDX: A64<[u8; 64]> = A64(dup_idx());
static PLUS16: A64<[u8; 64]> = A64(plus16());

/// `BinaryBatch32` (32 nibble bytes per row) -> `BinaryIndex32` (`(n, n+16)` byte pairs):
/// per row one 32-byte load, one `vpermb` (p5) and one `vpaddb` (p05) — 162 of each per batch,
/// ~5 uops per polynomial, against half the input bandwidth of the index form.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
pub unsafe fn expand_nibbles(nib: &BinaryBatch32, out: &mut BinaryIndex32) {
    let d = _mm512_load_si512(DUP_IDX.0.as_ptr() as *const __m512i);
    let p16 = _mm512_load_si512(PLUS16.0.as_ptr() as *const __m512i);
    let src = nib.idx.as_ptr() as *const u8;
    let dst = out.rows.as_mut_ptr() as *mut __m512i;
    for i in 0..162 {
        let s = _mm512_castsi256_si512(_mm256_loadu_si256(src.add(32 * i) as *const __m256i));
        let e = _mm512_permutexvar_epi8(d, s);
        _mm512_store_si512(dst.add(i), _mm512_add_epi8(e, p16));
    }
}

/// NTT from the nibble form: expansion prologue per batch, then the kernel.
pub fn ntt_bin_nib<const Q: u16>(nib: &[BinaryBatch32], out: &mut [Batch32]) {
    assert_eq!(nib.len(), out.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for (n, o) in nib.iter().zip(out.iter_mut()) {
            expand_nibbles(n, &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, o);
        }
        _mm_sfence();
    }
}

/// Nibble form, both primes (one expansion for two kernels).
pub fn ntt_bin_nib_2q<const QA: u16, const QB: u16>(
    nib: &[BinaryBatch32],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(nib.len(), out_a.len());
    assert_eq!(nib.len(), out_b.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..nib.len() {
            expand_nibbles(nib.get_unchecked(b), &mut idx);
            vb::ntt_bin_batch32_nt::<QA>(&idx, out_a.get_unchecked_mut(b));
            vb::ntt_bin_batch32_nt::<QB>(&idx, out_b.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

// ---------------------------------------------------------------------------------------------
// 3. out-of-cache overhead: prefetch, store placement, sfence
// ---------------------------------------------------------------------------------------------

/// `ntt_bin_polys` plus a prefetch of the next batch's 32 `BinaryPoly` (44 lines, 2 816 B).
pub fn ntt_bin_polys_pf<const Q: u16, const DIST: usize>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let nb = out.len();
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..nb {
            if b + DIST < nb {
                pf::<44>(polys.as_ptr().add(32 * (b + DIST)) as *const u8);
            }
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, out.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// `ntt_bin_polys` with cached stores in the kernel and a separate non-temporal copy pass
/// (648 loads + 648 `vmovntdq` per batch) instead of streaming from the last level directly.
pub fn ntt_bin_polys_copy<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let mut idx = BinaryIndex32::zero();
    let mut buf = Batch32::zero(Representation::Ntt);
    unsafe {
        for b in 0..out.len() {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32::<Q>(&idx, &mut buf);
            let s = buf.v.as_ptr() as *const __m512i;
            let d = out.get_unchecked_mut(b).v.as_mut_ptr() as *mut __m512i;
            for j in 0..N {
                _mm512_stream_si512(d.add(j), _mm512_load_si512(s.add(j)));
            }
            out.get_unchecked_mut(b).representation = Representation::Ntt;
        }
        _mm_sfence();
    }
}

/// `ntt_bin_polys` with an `sfence` after every batch instead of one at the end.
pub fn ntt_bin_polys_fence<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..out.len() {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, out.get_unchecked_mut(b));
            _mm_sfence();
        }
    }
}

/// `ntt_bin_polys` with cached (non-streaming) stores throughout, for reference.
pub fn ntt_bin_polys_cached<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..out.len() {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32::<Q>(&idx, out.get_unchecked_mut(b));
        }
    }
}

/// Transpose a group of `G` batches, then run the kernel over the group: the index rows for the
/// group (G x 10 368 B) stay in L2 and the two phases each see a longer straight run.
pub fn ntt_bin_polys_grouped<const Q: u16, const G: usize>(
    polys: &[BinaryPoly],
    out: &mut [Batch32],
) {
    assert_eq!(polys.len(), 32 * out.len());
    let nb = out.len();
    let mut idx: Vec<BinaryIndex32> = (0..G).map(|_| BinaryIndex32::zero()).collect();
    unsafe {
        let mut b = 0;
        while b < nb {
            let g = G.min(nb - b);
            for j in 0..g {
                transpose::slice_polys_idx_into(chunk32(polys, b + j), idx.get_unchecked_mut(j));
            }
            for j in 0..g {
                vb::ntt_bin_batch32_nt::<Q>(idx.get_unchecked(j), out.get_unchecked_mut(b + j));
            }
            b += g;
        }
        _mm_sfence();
    }
}


/// `ntt_bin_polys` with the 44-line prefetch of batch `b + DIST` issued *between* the transpose
/// and the kernel: at the loop head the prefetches contend with the transpose's own demand misses
/// for the 12 fill buffers (hardware drops prefetches when they are full), whereas the kernel is
/// ~9 500 cycles of pure compute with no demand traffic at all.
pub fn ntt_bin_polys_pfmid<const Q: u16, const DIST: usize, const HINT: i32>(
    polys: &[BinaryPoly],
    out: &mut [Batch32],
) {
    assert_eq!(polys.len(), 32 * out.len());
    let nb = out.len();
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..nb {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            if b + DIST < nb {
                let p = polys.as_ptr().add(32 * (b + DIST)) as *const u8;
                for i in 0..44 {
                    _mm_prefetch::<HINT>(p.add(64 * i) as *const i8);
                }
            }
            vb::ntt_bin_batch32_nt::<Q>(&idx, out.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// Same for the index-row entry point: `LINES` lines of `idxs[b + DIST]`, hint `HINT`.
pub fn ntt_bin_idx_pfh<const Q: u16, const DIST: usize, const LINES: usize, const HINT: i32>(
    idxs: &[BinaryIndex32],
    out: &mut [Batch32],
) {
    assert_eq!(idxs.len(), out.len());
    let nb = idxs.len();
    unsafe {
        for b in 0..nb {
            if b + DIST < nb {
                let p = idxs.get_unchecked(b + DIST).rows.as_ptr() as *const u8;
                for i in 0..LINES {
                    _mm_prefetch::<HINT>(p.add(64 * i) as *const i8);
                }
            }
            vb::ntt_bin_batch32_nt::<Q>(idxs.get_unchecked(b), out.get_unchecked_mut(b));
        }
        _mm_sfence();
    }
}

/// Diagnostic: every batch is streamed to the *same* `Batch32` — full non-temporal write traffic
/// (340 MB) but no walk through a 340 MB address range, so the store-side TLB and page-walk cost
/// is removed while the DRAM write traffic stays.
pub fn ntt_bin_polys_1out<const Q: u16>(polys: &[BinaryPoly], out: &mut Batch32) {
    assert_eq!(polys.len() % 32, 0);
    let mut idx = BinaryIndex32::zero();
    unsafe {
        for b in 0..polys.len() / 32 {
            transpose::slice_polys_idx_into(chunk32(polys, b), &mut idx);
            vb::ntt_bin_batch32_nt::<Q>(&idx, out);
        }
        _mm_sfence();
    }
}

// ---------------------------------------------------------------------------------------------
// mmap-backed output buffer, optionally on transparent huge pages
// ---------------------------------------------------------------------------------------------

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
const PROT_RW: i32 = 3;
const MAP_PRIVATE_ANON: i32 = 0x22;
const MADV_HUGEPAGE: i32 = 14;
const MADV_NOHUGEPAGE: i32 = 15;
const HP: usize = 2 << 20;

/// A `[Batch32]` in its own anonymous mapping, with `MADV_HUGEPAGE` or `MADV_NOHUGEPAGE` applied
/// and every page pre-faulted, so the benchmark measures the stream and not the page faults.
pub struct Batches {
    base: *mut u8,
    map_len: usize,
    ptr: *mut Batch32,
    len: usize,
}

impl Batches {
    pub fn new(len: usize, huge: bool) -> Self {
        let need = len * core::mem::size_of::<Batch32>();
        let map_len = (need + 2 * HP - 1) & !(HP - 1);
        unsafe {
            let base = mmap(core::ptr::null_mut(), map_len, PROT_RW, MAP_PRIVATE_ANON, -1, 0);
            assert!((base as isize) > 0, "mmap failed");
            let a = ((base as usize + HP - 1) & !(HP - 1)) as *mut u8;
            let alen = map_len - (a as usize - base as usize);
            madvise(a, alen & !(HP - 1), if huge { MADV_HUGEPAGE } else { MADV_NOHUGEPAGE });
            let mut o = 0;
            while o < need {
                core::ptr::write_volatile(a.add(o), 0u8);
                o += 4096;
            }
            Batches { base, map_len, ptr: a as *mut Batch32, len }
        }
    }
}

impl core::ops::Deref for Batches {
    type Target = [Batch32];
    fn deref(&self) -> &[Batch32] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl core::ops::DerefMut for Batches {
    fn deref_mut(&mut self) -> &mut [Batch32] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for Batches {
    fn drop(&mut self) {
        unsafe { munmap(self.base, self.map_len) };
    }
}

/// The same for the producer-side index rows (85 MB for 2^18 polynomials).
pub struct Indices {
    base: *mut u8,
    map_len: usize,
    ptr: *mut BinaryIndex32,
    len: usize,
}

impl Indices {
    pub fn new(len: usize, huge: bool) -> Self {
        let need = len * core::mem::size_of::<BinaryIndex32>();
        let map_len = (need + 2 * HP - 1) & !(HP - 1);
        unsafe {
            let base = mmap(core::ptr::null_mut(), map_len, PROT_RW, MAP_PRIVATE_ANON, -1, 0);
            assert!((base as isize) > 0, "mmap failed");
            let a = ((base as usize + HP - 1) & !(HP - 1)) as *mut u8;
            let alen = map_len - (a as usize - base as usize);
            madvise(a, alen & !(HP - 1), if huge { MADV_HUGEPAGE } else { MADV_NOHUGEPAGE });
            let mut o = 0;
            while o < need {
                core::ptr::write_volatile(a.add(o), 0u8);
                o += 4096;
            }
            Indices { base, map_len, ptr: a as *mut BinaryIndex32, len }
        }
    }
}

impl core::ops::Deref for Indices {
    type Target = [BinaryIndex32];
    fn deref(&self) -> &[BinaryIndex32] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl core::ops::DerefMut for Indices {
    fn deref_mut(&mut self) -> &mut [BinaryIndex32] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for Indices {
    fn drop(&mut self) {
        unsafe { munmap(self.base, self.map_len) };
    }
}

// ---------------------------------------------------------------------------------------------
// 4. the streaming form with a realistic consumer
// ---------------------------------------------------------------------------------------------

/// `y += sum_i a_i o NTT(w_i)`: one `Batch32` of NTT-domain operands per batch, streamed from
/// DRAM (340 MB). `PFD` is the prefetch distance in *batches* for the operand stream (0 = none);
/// the prefetch of `a[b + PFD]` is spread one line per slot over the multiply loop.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn accumulate<const Q: u16, const PFD: usize>(
    w: &Batch32,
    a: *const i16,
    nxt: *const i16,
    acc: &mut [[i32; 16]; N],
) {
    let ones = _mm512_set1_epi16(1);
    for j in 0..N {
        if PFD > 0 && j % 2 == 0 {
            _mm_prefetch::<_MM_HINT_T0>(nxt.add(32 * j) as *const i8);
        }
        let x = _mm512_load_si512(w.v.as_ptr().add(j) as *const __m512i);
        let y = _mm512_load_si512(a.add(32 * j) as *const __m512i);
        let p = pointwise::mont_mul_epi16::<Q>(x, y);
        let s = _mm512_madd_epi16(p, ones);
        let ap = acc.as_mut_ptr().add(j) as *mut __m512i;
        _mm512_storeu_si512(ap, _mm512_add_epi32(_mm512_loadu_si512(ap as *const __m512i), s));
    }
}

/// The streamed driver with the accumulating consumer. `PFD > 0` prefetches the operand batch
/// `PFD` ahead while the current one is being multiplied.
pub fn ntt_bin_accumulate<const Q: u16, const PFD: usize>(
    polys: &[BinaryPoly],
    a: &[Batch32],
    acc: &mut [[i32; 16]; N],
) {
    assert_eq!(polys.len(), 32 * a.len());
    let nb = a.len();
    vb::ntt_bin_stream::<Q>(polys, |b, w| unsafe {
        let nxt = if PFD > 0 && b + PFD < nb {
            a.get_unchecked(b + PFD).v.as_ptr() as *const i16
        } else {
            a.get_unchecked(b).v.as_ptr() as *const i16
        };
        accumulate::<Q, PFD>(w, a.get_unchecked(b).v.as_ptr() as *const i16, nxt, acc);
    });
}

/// Finish: y[j] = (sum of the 16 lanes) * 2^16 mod q, fully reduced.
pub fn finish_accumulator<const Q: u16>(acc: &[[i32; 16]; N]) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0u32; N];
    for j in 0..N {
        let s: i64 = acc[j].iter().map(|&x| x as i64).sum();
        y[j] = ((s.rem_euclid(q) * 65536) % q) as u32;
    }
    y
}

// ---------------------------------------------------------------------------------------------
// 3b. the transpose software-pipelined *into* the kernel's instruction stream
// ---------------------------------------------------------------------------------------------

/// The transpose is port-5 bound (826 p5 uops per batch, 31 cycles / polynomial) and the kernel is
/// port-0 bound (6 480 p0 uops, 94 % busy) with only 1 080 p5-only uops of its own, all of them in
/// the table-lookup loop; its two radix-3 loops issue *no* p5-only uop at all. Running the two back
/// to back therefore pays for both; issuing them in one instruction stream should cost
/// `max(p0, p5, ALU/2)` ~ 279 cycles per polynomial rather than 296 + 31.
///
/// The out-of-order window (352-entry ROB) is far smaller than a batch (23 000 instructions), so
/// the interleaving has to be static: this module holds a private copy of the kernel and of the
/// transpose, cut into 107 independent units, with the units of batch b+1 sprinkled through the
/// four 162-block passes of batch b. The four transpose phases are a chain (P1 -> P2 -> P3 -> P4),
/// so they are laid out in kernel order and only inside the two radix-3 loops, which are the ones
/// with port-5 slack:
///
/// | kernel loop            | iterations | transpose units placed there |
/// |------------------------|-----------:|------------------------------|
/// | block 0, level 4       |         54 | P1 g = 0..3 (8x8 qword + GFNI bit transpose) |
/// | block 0, levels 5+6    |         18 | P2 w = 0..10 (4-way byte interleave) |
/// | block 1, level 4       |         54 | P3 grp = 0..10 (4 x 81 qword transpose) |
/// | block 1, levels 5+6    |         18 | P4 h = 0..17 (mask -> index row) |
/// | block 2, level 4 / 5+6 |    54 / 18 | P4 h = 18..44 / 45..62 |
/// | block 3, level 4       |         54 | P4 h = 63..80 |
///
/// P1's 44 output vectors are spilled to a 2.8 KB scratch (44 stores + 44 loads per batch) because
/// they can no longer live in registers across the kernel.
mod fused {
#![allow(dead_code)]
use crate::params::*;
use crate::simd::transpose::BinaryIndex32;
use crate::types::*;
use core::arch::x86_64::*;

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
                        let v = base * f % q * extra % q;
                        let e = center(v, q) as u16;
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

static T3889: Tables = build_tables::<3889>();
static T9721: Tables = build_tables::<9721>();

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
#[repr(C, align(64))]
struct A64<T>(T);

/// `vpermb` pattern: Y.byte[8B + (7-j)] = Z.byte[8j + B].
const fn byte_perm() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut b = 0;
    while b < 8 {
        let mut j = 0;
        while j < 8 {
            t[8 * b + 7 - j] = (8 * j + b) as u8;
            j += 1;
        }
        b += 1;
    }
    t
}

/// Affine-matrix operand for output row r (0 or 1) of a 32-byte `mask` group: qword q gets
/// bytes 4..7 = the four plane masks of group q&3, byte 3 = 0x00 (q < 4) or 0xFF (q >= 4, the
/// +16), bytes 0..2 = 0x00 so the top three output bits stay clear.
const fn affine_idx(r: usize) -> [u8; 64] {
    let mut t = [32u8; 64];
    let mut q = 0;
    while q < 8 {
        let mut b = 0;
        while b < 4 {
            t[8 * q + 7 - b] = (8 * b + 4 * r + (q & 3)) as u8;
            b += 1;
        }
        if q >= 4 {
            t[8 * q + 3] = 33;
        }
        q += 1;
    }
    t
}

/// `[n_0..n_31 | n_0+16..n_31+16]` -> `[n_0, n_0+16, n_1, n_1+16, ...]`.
const fn interleave_idx() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p] = p as u8;
        t[2 * p + 1] = (32 + p) as u8;
        p += 1;
    }
    t
}

/// Upper half of the `vpermb` source: byte 32 = 0x00, byte 33 = 0xFF.
const fn const_half() -> [u8; 64] {
    let mut t = [0u8; 64];
    t[33] = 0xff;
    t
}

static BYTE_PERM: A64<[u8; 64]> = A64(byte_perm());
static AFFINE_IDX0: A64<[u8; 64]> = A64(affine_idx(0));
static AFFINE_IDX1: A64<[u8; 64]> = A64(affine_idx(1));
static INTERLEAVE: A64<[u8; 64]> = A64(interleave_idx());
static CONST_HALF: A64<[u8; 64]> = A64(const_half());

/// The 8x8 identity bit-matrix in `vgf2p8affineqb` vector form (byte j = 1 << j).
const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

#[inline(always)]
unsafe fn transpose8x8_q(r: [__m512i; 8]) -> [__m512i; 8] {
    let a0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi64(r[6], r[7]);
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b2 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b5 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b6 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    [
        _mm512_shuffle_i64x2::<0x88>(b0, b2),
        _mm512_shuffle_i64x2::<0x88>(b4, b6),
        _mm512_shuffle_i64x2::<0x88>(b1, b3),
        _mm512_shuffle_i64x2::<0x88>(b5, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b2),
        _mm512_shuffle_i64x2::<0xDD>(b4, b6),
        _mm512_shuffle_i64x2::<0xDD>(b1, b3),
        _mm512_shuffle_i64x2::<0xDD>(b5, b7),
    ]
}

#[repr(C, align(64))]
struct Masks([u32; 704]);
#[repr(C, align(64))]
struct Planes([u64; 4 * 88]);

// -------------------------------------------------------------- transpose units (next batch)

#[repr(C, align(64))]
pub struct Scratch {
    scr: [[u8; 64]; 44],
    masks: [u32; 704],
    planes: [u64; 4 * 88],
}

impl Scratch {
    pub fn new() -> Self {
        Scratch { scr: [[0u8; 64]; 44], masks: [0u32; 704], planes: [0u64; 4 * 88] }
    }
}

struct Tr {
    src: *const u8,
    scr: *mut __m512i,
    masks: *mut u32,
    planes: *mut u64,
    rows: *mut __m512i,
}

/// Phase 1, group g: 8x8 qword transpose + `vpermb` + GFNI bit transpose of 8 polynomials,
/// spilled to `scr[11g .. 11g + 11]`.
#[inline(always)]
unsafe fn p1(t: &Tr, g: usize) {
    let perm = _mm512_load_si512(BYTE_PERM.0.as_ptr() as *const __m512i);
    let ident = _mm512_set1_epi64(GF_IDENT);
    let mut rows: [__m512i; 8] = [_mm512_setzero_si512(); 8];
    for j in 0..8 {
        rows[j] = _mm512_loadu_si512(t.src.add((8 * g + j) * 88) as *const __m512i);
    }
    let c = transpose8x8_q(rows);
    for w in 0..8 {
        let y = _mm512_permutexvar_epi8(perm, c[w]);
        _mm512_store_si512(t.scr.add(11 * g + w), _mm512_gf2p8affine_epi64_epi8::<0>(ident, y));
    }
    for j in 0..8 {
        rows[j] = _mm512_maskz_loadu_epi64(0x07, t.src.add((8 * g + j) * 88 + 64) as *const i64);
    }
    let c = transpose8x8_q(rows);
    for w in 0..3 {
        let y = _mm512_permutexvar_epi8(perm, c[w]);
        _mm512_store_si512(t.scr.add(11 * g + 8 + w), _mm512_gf2p8affine_epi64_epi8::<0>(ident, y));
    }
}

/// Phase 2, word w: 4-way byte interleave of the four groups into `masks[64w .. 64w + 64]`.
#[inline(always)]
unsafe fn p2(t: &Tr, w: usize) {
    let a = _mm512_load_si512(t.scr.add(w));
    let b = _mm512_load_si512(t.scr.add(11 + w));
    let cc = _mm512_load_si512(t.scr.add(22 + w));
    let d = _mm512_load_si512(t.scr.add(33 + w));
    let l01 = _mm512_unpacklo_epi8(a, b);
    let h01 = _mm512_unpackhi_epi8(a, b);
    let l23 = _mm512_unpacklo_epi8(cc, d);
    let h23 = _mm512_unpackhi_epi8(cc, d);
    let r0 = _mm512_unpacklo_epi16(l01, l23);
    let r1 = _mm512_unpackhi_epi16(l01, l23);
    let r2 = _mm512_unpacklo_epi16(h01, h23);
    let r3 = _mm512_unpackhi_epi16(h01, h23);
    let s0 = _mm512_shuffle_i64x2::<0x44>(r0, r1);
    let s1 = _mm512_shuffle_i64x2::<0xEE>(r0, r1);
    let s2 = _mm512_shuffle_i64x2::<0x44>(r2, r3);
    let s3 = _mm512_shuffle_i64x2::<0xEE>(r2, r3);
    let p = t.masks.add(64 * w) as *mut __m512i;
    _mm512_store_si512(p, _mm512_shuffle_i64x2::<0x88>(s0, s2));
    _mm512_store_si512(p.add(1), _mm512_shuffle_i64x2::<0xDD>(s0, s2));
    _mm512_store_si512(p.add(2), _mm512_shuffle_i64x2::<0x88>(s1, s3));
    _mm512_store_si512(p.add(3), _mm512_shuffle_i64x2::<0xDD>(s1, s3));
}

/// Phase 3, group grp: the 4 x 81 qword transpose for rows 16grp .. 16grp + 15.
#[inline(always)]
unsafe fn p3(t: &Tr, grp: usize) {
    let m64 = t.masks as *const i64;
    let h = 8 * grp;
    let a = _mm512_loadu_si512(m64.add(h) as *const __m512i);
    let b = _mm512_loadu_si512(m64.add(h + 81) as *const __m512i);
    let cc = _mm512_loadu_si512(m64.add(h + 162) as *const __m512i);
    let d = _mm512_loadu_si512(m64.add(h + 243) as *const __m512i);
    let l01 = _mm512_unpacklo_epi64(a, b);
    let h01 = _mm512_unpackhi_epi64(a, b);
    let l23 = _mm512_unpacklo_epi64(cc, d);
    let h23 = _mm512_unpackhi_epi64(cc, d);
    let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
    let s1 = _mm512_shuffle_i64x2::<0xEE>(l01, l23);
    let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
    let s3 = _mm512_shuffle_i64x2::<0xEE>(h01, h23);
    let p = t.planes.add(4 * h) as *mut __m512i;
    _mm512_store_si512(p, _mm512_shuffle_i64x2::<0x88>(s0, s2));
    _mm512_store_si512(p.add(1), _mm512_shuffle_i64x2::<0xDD>(s0, s2));
    _mm512_store_si512(p.add(2), _mm512_shuffle_i64x2::<0x88>(s1, s3));
    _mm512_store_si512(p.add(3), _mm512_shuffle_i64x2::<0xDD>(s1, s3));
}

/// Phase 4, group h: two `vpermb` index rows out of one 32-byte plane group.
#[inline(always)]
unsafe fn p4(t: &Tr, h: usize) {
    let ident = _mm512_set1_epi64(GF_IDENT);
    let ai0 = _mm512_load_si512(AFFINE_IDX0.0.as_ptr() as *const __m512i);
    let ai1 = _mm512_load_si512(AFFINE_IDX1.0.as_ptr() as *const __m512i);
    let inter = _mm512_load_si512(INTERLEAVE.0.as_ptr() as *const __m512i);
    let ch = _mm512_load_si512(CONST_HALF.0.as_ptr() as *const __m512i);
    let src = _mm512_mask_loadu_epi64(ch, 0x0f, t.planes.add(4 * h) as *const i64);
    let n0 = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(ai0, src));
    let n1 = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(ai1, src));
    _mm512_store_si512(t.rows.add(2 * h), _mm512_permutexvar_epi8(inter, n0));
    _mm512_store_si512(t.rows.add(2 * h + 1), _mm512_permutexvar_epi8(inter, n1));
}

// -------------------------------------------------------------- the fused 162-block pass

#[inline(always)]
unsafe fn kblock<const Q: u16, const K: usize, const BAR: bool, const PIPE: u32>(
    t: &'static Tables,
    c: &C,
    ip: *const u8,
    bp: *mut i16,
    outp: *mut i16,
    tr: &Tr,
) {
    let lut = t.lut.as_ptr().add(12 * K) as *const u8;
    let l = |s2: usize, r: usize, ab: usize| -> __m512i {
        _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
    };
    let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
    let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
    let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));
    let (l110, l111) = (l(1, 1, 0), l(1, 1, 1));
    let (l120, l121) = (l(1, 2, 0), l(1, 2, 1));

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

        let (u0, u1, u2) = r3_folded(c, a0, a1, a2);
        let (v0, v1, v2) = r3_folded(c, b0, b1, b2);
        st(bp, i, u0);
        st(bp, i + 27, u1);
        st(bp, i + 54, u2);
        st(bp, i + 81, v0);
        st(bp, i + 108, v1);
        st(bp, i + 135, v2);
    }

    for j in 0..6 {
        let base = 27 * j;
        let tw = t.tw4[6 * K + j].as_ptr();
        for i in 0..9 {
            let (o0, o1, o2) =
                r3::<BAR>(c, ld(bp, base + i), ld(bp, base + 9 + i), ld(bp, base + 18 + i), tw);
            st(bp, base + i, o0);
            st(bp, base + 9 + i, o1);
            st(bp, base + 18 + i, o2);
            let ii = 9 * j + i;
            match K {
                0 => {
                    if PIPE & 1 != 0 && ii % 13 == 0 && ii / 13 < 4 {
                        p1(tr, ii / 13);
                    }
                }
                1 => {
                    if PIPE & 2 != 0 && ii % 5 == 0 && ii / 5 < 11 {
                        p3(tr, ii / 5);
                    }
                }
                2 => {
                    if PIPE & 2 != 0 && ii % 2 == 0 {
                        p4(tr, 18 + ii / 2);
                    }
                }
                _ => {
                    if PIPE & 2 != 0 && ii % 3 == 0 && ii / 3 < 18 {
                        p4(tr, 63 + ii / 3);
                    }
                }
            }
        }
    }

    let op = outp.add(32 * 162 * K);
    for j in 0..18 {
        let base = 9 * j;
        let tw = t.tw5[18 * K + j].as_ptr();
        let mut v = [_mm512_setzero_si512(); 9];
        for i in 0..9 {
            v[i] = ld(bp, base + i);
        }
        for i in 0..3 {
            let (o0, o1, o2) = r3::<BAR>(c, v[i], v[3 + i], v[6 + i], tw);
            v[i] = o0;
            v[3 + i] = o1;
            v[6 + i] = o2;
        }
        for i in 0..3 {
            let tw = t.tw6[54 * K + 3 * j + i].as_ptr();
            let (o0, o1, o2) = r3::<BAR>(c, v[3 * i], v[3 * i + 1], v[3 * i + 2], tw);
            let b = base + 3 * i;
            _mm512_stream_si512(op.add(32 * b) as *mut __m512i, o0);
            _mm512_stream_si512(op.add(32 * b + 32) as *mut __m512i, o1);
            _mm512_stream_si512(op.add(32 * b + 64) as *mut __m512i, o2);
        }
        match K {
            0 => {
                if PIPE & 1 != 0 && j < 11 {
                    p2(tr, j);
                }
            }
            1 => {
                if PIPE & 2 != 0 {
                    p4(tr, j);
                }
            }
            2 => {
                if PIPE & 2 != 0 {
                    p4(tr, 45 + j);
                }
            }
            _ => {}
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_pipelined<const Q: u16, const PIPE: u32>(
    input: &BinaryIndex32,
    outp: *mut i16,
    next_polys: *const u8,
    next_idx: *mut BinaryIndex32,
    s: &mut Scratch,
) {
    let t = tables::<Q>();
    let c = C { q: bc(&t.qd), bv: bc(&t.bv), om: bc(&t.om[0]), omp: bc(&t.om[1]) };
    let tr = Tr {
        src: next_polys,
        scr: s.scr.as_mut_ptr() as *mut __m512i,
        masks: s.masks.as_mut_ptr(),
        planes: s.planes.as_mut_ptr(),
        rows: (*next_idx).rows.as_mut_ptr() as *mut __m512i,
    };
    if PIPE & 1 == 0 && PIPE != 0 {
        for g in 0..4 {
            p1(&tr, g);
        }
        for w in 0..11 {
            p2(&tr, w);
        }
    }
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;
    const fn bar(q: u16) -> bool {
        needs_barrett(q)
    }
    if bar(Q) {
        kblock::<Q, 0, true, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 1, true, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 2, true, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 3, true, PIPE>(t, &c, ip, bp, outp, &tr);
    } else {
        kblock::<Q, 0, false, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 1, false, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 2, false, PIPE>(t, &c, ip, bp, outp, &tr);
        kblock::<Q, 3, false, PIPE>(t, &c, ip, bp, outp, &tr);
    }
}

/// Driver: one transpose ahead, everything else inside the kernel. `PIPE = false` gives the same
/// code with the transpose units switched off (used to check that the copy of the kernel itself
/// costs nothing).
pub fn drive<const Q: u16, const PIPE: u32>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    assert_eq!(polys.len(), 32 * out.len());
    let nb = out.len();
    if nb == 0 {
        return;
    }
    let mut s = Scratch::new();
    let mut idx = [BinaryIndex32::zero(), BinaryIndex32::zero()];
    unsafe {
        crate::simd::transpose::slice_polys_idx_into(
            &*(polys.as_ptr() as *const [BinaryPoly; 32]),
            &mut idx[0],
        );
        let ip = idx.as_mut_ptr();
        for b in 0..nb {
            let cur = b & 1;
            let nxt = cur ^ 1;
            let src = polys.as_ptr().add(32 * (b + 1).min(nb - 1)) as *const u8;
            if PIPE != 0 {
                ntt_pipelined::<Q, PIPE>(
                    &*ip.add(cur),
                    out.get_unchecked_mut(b).v.as_mut_ptr() as *mut i16,
                    src,
                    ip.add(nxt),
                    &mut s,
                );
            } else {
                ntt_pipelined::<Q, 0>(
                    &*ip.add(cur),
                    out.get_unchecked_mut(b).v.as_mut_ptr() as *mut i16,
                    src,
                    ip.add(nxt),
                    &mut s,
                );
                if b + 1 < nb {
                    crate::simd::transpose::slice_polys_idx_into(
                        &*(polys.as_ptr().add(32 * (b + 1)) as *const [BinaryPoly; 32]),
                        &mut *ip.add(nxt),
                    );
                }
            }
            out.get_unchecked_mut(b).representation = Representation::Ntt;
        }
        _mm_sfence();
    }
}

/// Two primes, one transpose, and that transpose interleaved into the first prime's kernel:
/// the index rows of batch b+1 are built inside the `QA` kernel of batch b, and the `QB` kernel
/// runs on the same rows with no transpose work at all.
pub fn drive2q<const QA: u16, const QB: u16>(
    polys: &[BinaryPoly],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    assert_eq!(polys.len(), 32 * out_a.len());
    assert_eq!(out_a.len(), out_b.len());
    let nb = out_a.len();
    if nb == 0 {
        return;
    }
    let mut s = Scratch::new();
    let mut idx = [BinaryIndex32::zero(), BinaryIndex32::zero()];
    unsafe {
        crate::simd::transpose::slice_polys_idx_into(
            &*(polys.as_ptr() as *const [BinaryPoly; 32]),
            &mut idx[0],
        );
        let ip = idx.as_mut_ptr();
        for b in 0..nb {
            let cur = b & 1;
            let nxt = cur ^ 1;
            let src = polys.as_ptr().add(32 * (b + 1).min(nb - 1)) as *const u8;
            ntt_pipelined::<QA, 3>(
                &*ip.add(cur),
                out_a.get_unchecked_mut(b).v.as_mut_ptr() as *mut i16,
                src,
                ip.add(nxt),
                &mut s,
            );
            ntt_pipelined::<QB, 0>(
                &*ip.add(cur),
                out_b.get_unchecked_mut(b).v.as_mut_ptr() as *mut i16,
                src,
                ip.add(nxt),
                &mut s,
            );
            out_a.get_unchecked_mut(b).representation = Representation::Ntt;
            out_b.get_unchecked_mut(b).representation = Representation::Ntt;
        }
        _mm_sfence();
    }
}
}

/// `ntt_bin_polys` with the transpose of batch b+1 interleaved into the kernel of batch b.
pub fn ntt_bin_polys_pipelined<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    fused::drive::<Q, 3>(polys, out)
}

/// The same private copy of the kernel with the interleaving switched off (control: it must land
/// on the baseline, otherwise the copy itself changed the code generation).
pub fn ntt_bin_polys_pipelined_off<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    fused::drive::<Q, 0>(polys, out)
}

/// Only the second half of the transpose (P3, P4 — the 4 x 81 qword transpose and the
/// mask -> index-row phase, 456 of the 826 port-5 uops) is interleaved; P1 and P2, which need 16
/// live vectors for the 8x8 qword transposes, run as a prologue where they cannot push the
/// kernel's own values out of the register file.
pub fn ntt_bin_polys_pipelined_p34<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32]) {
    fused::drive::<Q, 2>(polys, out)
}

/// Both primes with the shared transpose folded into the first kernel (see [`fused::drive2q`]).
pub fn ntt_bin_polys_2q_pipelined<const QA: u16, const QB: u16>(
    polys: &[BinaryPoly],
    out_a: &mut [Batch32],
    out_b: &mut [Batch32],
) {
    fused::drive2q::<QA, QB>(polys, out_a, out_b)
}

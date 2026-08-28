//! The folding step of `Pi_fold`, on top of the commitment, entirely in the 648-slot NTT domain
//! of `R_648`.
//!
//! # The algebra
//!
//! A witness of `r` chunks `W_0, .., W_{r-1}`, each `len_ring` elements of
//! `R_648 = Z_q[X]/(X^648 - X^324 + 1)`, is committed under one key `A` to the `r` elements
//! `C_j = sum_i A_i W_{j,i}`. A challenge `c_j` is a short ternary element of the subring
//! `R_162 = Z_q[Z]/Phi_243(Z)`, embedded into `R_648` as `c_j(-X^4)` (coefficient of `X^{4m}` is
//! `(-1)^m c_{j,m}`, everything else zero — the embedding of "The lift is a ring extension of
//! degree 4"). The folded witness and its commitment are
//!
//! ```text
//!     v = sum_j c_j W_j   (len_ring elements),        A v = sum_j c_j C_j,
//! ```
//! the second identity by `R_648`-linearity of `A`. Both sides are slot-wise in the NTT domain:
//! `NTT(v)[u] = sum_j NTT(c_j)[u] NTT(W_j)[u]`, one scalar per challenge per slot. Multiplication
//! by a subring element acts on the four `R_162` components of an element alike, so the whole
//! fold is a single length-`r` inner product per slot with no ring multiplication anywhere.
//!
//! # What it costs
//!
//! [`CommitmentKey::commit_with_aux`](crate::CommitmentKey::commit_with_aux) already left `NTT_3889(W)`
//! in memory, so the fold never transforms the witness again: it reads those 85 MB once,
//! which is the DRAM floor of the step. The accumulator is `len_ring/32 x 648` 32-lane i32 groups
//! (663 KB for `len_ring = 256`, L2-resident) and every chunk contributes one `vpmaddwd` per slot
//! vector; the small transform's own lazy reduction (`|W| <= 7.5 q`) survives untouched into the
//! products, and the accumulator is folded back exactly, `x = l + (h + c) R mod q`, every
//! [`FOLD_PERIOD`] chunks.
//!
//! Only the base limb's transform is kept. `v` is inverted back to coefficients modulo
//! `q1 = 3889` — where it becomes a genuine small-integer vector, see [`FoldOutput::max_abs_v`] —
//! and transformed forward again modulo every additional limb (`vertical_gen`, or
//! `vertical_gen_quad` for a quadratic-slot one), which is why an additional limb costs an
//! 8-batch NTT and its own `A v` instead of a second 85 MB stream.
//!
//! ```no_run
//! use bin_ntt::{fold, CommitmentKey, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
//! # use bin_fields::scalar::F162;
//! # let witness: Vec<F162> = Vec::new();
//! let ck = CommitmentKey::random_default(1 << 10, 0xC0FFEE);
//! let (c, aux) = ck.commit_with_aux(&witness, 256);
//! let mut t = Transcript::new(b"bin-ntt/fold");
//! for j in 0..256 {
//!     t.absorb_elements(c.column(j));
//! }
//! let ch: Vec<_> = (0..256)
//!     .map(|_| bin_ntt::sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
//!     .collect();
//! let out = fold::fold(&ck, &aux, &ch);
//! let _ = &out.v; // the amortised witness, 256 ring elements, centered coefficients
//! ```
use crate::api::{
    components_of, AuxData, CommitmentKey, PowerOfThreeRingElementWithLimbs, BASE_PRIME, N162,
    PRIMES,
};
use crate::challenge::ShortChallenge;
use crate::params::{ParamsQ, N, QUAD_SLOTS};
use crate::simd::commit as cm;
use crate::simd::vertical_gen::{self as vg, intt_gen_batch32, ntt_gen_batch32};
use crate::simd::vertical_gen_quad::{self as vgq, ntt_quad_gen_batch32};
use crate::types::{Batch32, Representation, RingElement};
use core::arch::x86_64::*;
use std::time::Instant;

/// The prime the witness transform is kept in, and the one the fold accumulates over: the base
/// limb of every key.
pub const Q1: u16 = BASE_PRIME;
/// The second prime of the default limb list, reached — like every additional limb — by an
/// inverse transform of `v` and a forward one on `len_ring / 32` batches.
pub const Q2: u16 = PRIMES[1];

// =============================================================================================
// the accumulation bound
// =============================================================================================

/// What one chunk adds to one accumulator lane: `|W| <= 7.5 q` (the binary kernel's declared
/// output bound, [`cm::w_bound`]) times `|c| <= (q-1)/2` (a fully reduced centered challenge
/// slot, [`cm::a_bound`]) — one product, unlike the commitment's four, because a lane of this
/// accumulator carries one ring element rather than four.
pub const fn fold_per_chunk(q: u16) -> i64 {
    cm::w_bound(q) * cm::a_bound(q)
}

/// Chunks accumulated between two fold-backs.
///
/// `|acc| <= acc_after_reduce + P * fold_per_chunk` must fit `i32`; for `q = 3889` that is
/// `2^15 (1 + 3312) + P * 29167 * 1944 = 108 592 384 + P * 56 700 648`, so `P = 32` (1 923 013 120)
/// fits and `P = 64` does not.
pub const FOLD_PERIOD: usize = 32;

const fn fits(q: u16, p: usize) -> bool {
    cm::acc_after_reduce(q) + (p as i64) * fold_per_chunk(q) <= i32::MAX as i64
}
const _: () = assert!(fits(Q1, FOLD_PERIOD));
const _: () = assert!(!fits(Q1, 2 * FOLD_PERIOD));

/// `y = A v` accumulates `len_ring/32` batches of `|v| <= (q-1)/2` against `|A| <= (q-1)/2`, four
/// products per lane, and never folds back; 8 batches of `q = 9721` is `8 * 4 * 4860^2 < 2^31`.
const fn av_fits(q: u16, batches: usize) -> bool {
    (batches as i64) * 4 * cm::a_bound(q) * cm::a_bound(q) <= i32::MAX as i64
}
const _: () = assert!(av_fits(Q2, 8));
const _: () = assert!(av_fits(Q1, 8));

// =============================================================================================
// small vector helpers
// =============================================================================================

/// `floor(2^43 / q)`, the Barrett magic of [`barrett29`].
const fn barrett_magic(q: u16) -> u64 {
    (1u64 << 43) / q as u64
}

/// The multiple of `q` added before [`barrett29`] to make a folded-back lane non-negative:
/// the smallest one at least [`cm::acc_after_reduce`], so the sum stays below `2^29`.
const fn shift_up(q: u16) -> i32 {
    let a = cm::acc_after_reduce(q);
    let k = (a + q as i64 - 1) / q as i64;
    (k * q as i64) as i32
}
const _: () = assert!(2 * (shift_up(Q1) as i64) < (1i64 << 29));

/// `p mod q` for `0 <= p < 2^29`, 16 lanes at a time (the same Barrett as
/// `api::decompose_648_to_4x162` uses).
#[inline(always)]
unsafe fn barrett29<const Q: u16>(p: __m512i) -> __m512i {
    let q = _mm512_set1_epi32(Q as i32);
    let mag = _mm512_set1_epi64(barrett_magic(Q) as i64);
    let lo = _mm512_set1_epi64(0xFFFF_FFFFu32 as i64);
    let he = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_and_si512(p, lo), mag));
    let ho = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(p), mag));
    let t = _mm512_or_si512(he, _mm512_slli_epi64::<32>(ho));
    let r = _mm512_sub_epi32(p, _mm512_mullo_epi32(t, q));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, q))
}

/// Declared output bound of the generic-input kernel of `q` — `vertical_gen` for a splitting
/// prime, `vertical_gen_quad` for a quadratic-slot one.
pub const fn gen_bound(q: u16) -> i32 {
    if q == 3889 {
        13231
    } else if q == 9721 {
        20652
    } else {
        vgq::output_bound(q)
    }
}
const _: () = assert!(gen_bound(3889) == vg::Tw::<3889>::OUTPUT_BOUND);
const _: () = assert!(gen_bound(9721) == vg::Tw::<9721>::OUTPUT_BOUND);

/// The shift [`center_epi16`] uses: the smallest power of two `K` with `K q >= gen_bound(q)`
/// (4 for 3889 and 9721 and 4861, 8 for 2917, 2 for 12637).
pub const fn center_k(q: u16) -> u32 {
    let mut k = 1u32;
    while (k * q as u32) < gen_bound(q) as u32 {
        k *= 2;
    }
    k
}

/// `K q + gen_bound(q)` must stay inside a u16 lane, which is what makes the unsigned trick work.
const fn center_fits(q: u16) -> bool {
    center_k(q) * q as u32 + gen_bound(q) as u32 <= 65535
}
const _: () = assert!(center_fits(3889) && center_fits(9721));
const _: () = assert!(center_fits(2917) && center_fits(4861) && center_fits(12637));

/// `x mod q` centered into `[-(q-1)/2, (q-1)/2]` for 32 i16 lanes with `|x| <= K q`: shift by
/// `K q` into `[0, 2 K q) < 2^16`, `log2 K + 1` unsigned conditional subtracts, then the
/// centering subtract.
#[inline(always)]
unsafe fn center_epi16<const Q: u16>(x: __m512i) -> __m512i {
    let q = Q as u32;
    let mut k = center_k(Q);
    let mut v = _mm512_add_epi16(x, _mm512_set1_epi16((k * q) as i16));
    while k >= 1 {
        let s = _mm512_sub_epi16(v, _mm512_set1_epi16((k * q) as i16));
        v = _mm512_min_epu16(v, s);
        k /= 2;
    }
    let hi = _mm512_cmpgt_epu16_mask(v, _mm512_set1_epi16(((q - 1) / 2) as i16));
    _mm512_mask_sub_epi16(v, hi, v, _mm512_set1_epi16(q as i16))
}

/// `max_j max_p |b.v[j][p]|`, 32 lanes at a time.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn max_abs_batch(b: &Batch32) -> i32 {
    let mut m = _mm512_setzero_si512();
    for j in 0..N {
        let x = _mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
        m = _mm512_max_epi16(m, _mm512_abs_epi16(x));
    }
    let mut out = [0i16; 32];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, m);
    out.iter().map(|x| *x as i32).max().unwrap()
}

/// Every slot of a batch fully reduced and centered (`|v| <= (q-1)/2`); the input must satisfy
/// `|v| <= center_k(Q) * q`, which every generic kernel's output bound does.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn center_batch<const Q: u16>(b: &mut Batch32) {
    for j in 0..N {
        let p = b.v[j].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(p, center_epi16::<Q>(_mm512_load_si512(p as *const __m512i)));
    }
}

// =============================================================================================
// the challenges
// =============================================================================================

/// The `r` challenges transformed, packed so that the scalar pair `(c_{2i}[u], c_{2i+1}[u])` is
/// one dword — a `vpbroadcastd` memory operand, and exactly the operand `vpmaddwd` wants against
/// two interleaved witness rows.
pub(crate) struct ChallengeNtt {
    /// `pair[i][u]` = `c_{2i}[u] as u16 | (c_{2i+1}[u] as u16) << 16`.
    pair: Vec<[u32; N]>,
    /// The same transforms per challenge, for the consistency check: `slot[j][u]`.
    pub(crate) slot: Vec<[i16; N]>,
}

/// The `r` challenges embedded as `c(-X^4)` into as many `Batch32` of coefficients as they need.
fn embed(challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let nb = challenges.len().div_ceil(32);
    let mut out: Vec<Batch32> = (0..nb)
        .map(|_| Batch32::zero(Representation::Coefficients))
        .collect();
    for (j, c) in challenges.iter().enumerate() {
        let co = c.coeffs();
        for m in 0..N162 {
            let s = if m % 2 == 0 { co[m] } else { -co[m] };
            out[j / 32].v[4 * m][j % 32] = s as i16;
        }
    }
    out
}

/// Transform the embedded challenges modulo `Q` and fully reduce them to centered slots, which is
/// what the accumulation bound of [`FOLD_PERIOD`] assumes.
pub(crate) fn challenge_ntt<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = slots(challenges);
    unsafe {
        for b in bs.iter_mut() {
            ntt_gen_batch32::<Q>(b);
            center_batch::<Q>(b);
        }
    }
    pack(&bs, challenges.len())
}

/// The same on the quadratic-slot tree: the 648 rows are then the two coefficients of each of the
/// 324 leaves, which is the algebra [`combine_limb`] multiplies in.
pub(crate) fn challenge_ntt_quad<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = slots(challenges);
    unsafe {
        for b in bs.iter_mut() {
            ntt_quad_gen_batch32::<Q>(b);
            center_batch::<Q>(b);
        }
    }
    pack(&bs, challenges.len())
}

/// The challenge transforms of one limb, dispatched on its prime.
pub(crate) fn challenge_ntt_limb(
    q: u16,
    quad: bool,
    challenges: &[ShortChallenge],
) -> ChallengeNtt {
    match (q, quad) {
        (3889, false) => challenge_ntt::<3889>(challenges),
        (9721, false) => challenge_ntt::<9721>(challenges),
        (2917, true) => challenge_ntt_quad::<2917>(challenges),
        (4861, true) => challenge_ntt_quad::<4861>(challenges),
        (12637, true) => challenge_ntt_quad::<12637>(challenges),
        _ => unreachable!("no limb with q = {q}"),
    }
}

fn slots(challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let r = challenges.len();
    assert!(r >= 2 && r % 2 == 0, "the fold pairs the chunks: r must be even");
    embed(challenges)
}

/// The transformed batches read out per challenge and packed into the dword pairs the
/// accumulation wants.
fn pack(bs: &[Batch32], r: usize) -> ChallengeNtt {
    let mut slot = vec![[0i16; N]; r];
    for j in 0..r {
        for u in 0..N {
            slot[j][u] = bs[j / 32].v[u][j % 32];
        }
    }
    let mut pair = vec![[0u32; N]; r / 2];
    for i in 0..r / 2 {
        for u in 0..N {
            pair[i][u] =
                (slot[2 * i][u] as u16) as u32 | (((slot[2 * i + 1][u] as u16) as u32) << 16);
        }
    }
    ChallengeNtt { pair, slot }
}

// =============================================================================================
// the accumulator
// =============================================================================================

/// One 64-byte aligned accumulator vector.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AccVec([i32; 16]);

/// `acc[(b * 648 + u) * 2 + h]`: batch position `b` inside a chunk, slot `u`, half `h`.
///
/// Half 0 is `vpunpcklwd` of the two witness rows, half 1 is `vpunpckhwd`, so lane `t` of half
/// `h` carries ring element [`lane_of`]`(h, t)` of the chunk. The permutation is undone once, when
/// the accumulator is read out.
const fn lane_of(h: usize, t: usize) -> usize {
    8 * (t / 4) + 4 * h + t % 4
}

/// The `vpermi2d` indices that undo [`lane_of`]: `NAT[k]` gathers lanes `16k .. 16k+16` of the
/// natural element order out of (half 0, half 1).
const NAT: [[i32; 16]; 2] = {
    let mut nat = [[0i32; 16]; 2];
    let mut h = 0;
    while h < 2 {
        let mut t = 0;
        while t < 16 {
            let p = lane_of(h, t);
            nat[p / 16][p % 16] = (16 * h + t) as i32;
            t += 1;
        }
        h += 1;
    }
    nat
};

/// Two consecutive chunks into the accumulator of one batch position: 648 slot vectors, one
/// `vpunpck` and one `vpmaddwd` per half.
///
/// The two witness rows and the accumulator are three plain forward streams, which is all the
/// hardware prefetcher needs: one `prefetcht1` per slot aimed one step ahead (as
/// `simd::commit` does for `A`) measured 4.96 ms against 4.26 ms with none — unlike the
/// commitment, this loop has no compute to hide the extra fill-buffer pressure behind.
///
/// # Safety
/// `w0`, `w1` and `acc` cover 648 vectors (`acc` 648 pairs).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_pair(w0: *const i16, w1: *const i16, cp: *const u32, acc: *mut i32) {
    for u in 0..N {
        let a = _mm512_load_si512(w0.add(32 * u) as *const __m512i);
        let b = _mm512_load_si512(w1.add(32 * u) as *const __m512i);
        let c = _mm512_set1_epi32(*cp.add(u) as i32);
        let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c);
        let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c);
        let d = acc.add(32 * u);
        _mm512_store_si512(
            d as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d as *const __m512i), lo),
        );
        _mm512_store_si512(
            d.add(16) as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d.add(16) as *const __m512i), hi),
        );
    }
}

/// The exact fold-back of [`crate::simd::commit`] over one batch position's accumulator.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn fold_back<const Q: u16>(acc: *mut i32, vecs: usize) {
    for j in 0..vecs {
        let p = acc.add(16 * j);
        _mm512_store_si512(
            p as *mut __m512i,
            cm::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)),
        );
    }
}

/// The accumulator of one batch position, folded back a last time, reduced to `[0, q)` and
/// written out in the natural element order as centered slots of `out`.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn drain<const Q: u16>(acc: *const i32, out: &mut Batch32) {
    let up = _mm512_set1_epi32(shift_up(Q));
    let half = _mm512_set1_epi32((Q as i32 - 1) / 2);
    let q = _mm512_set1_epi32(Q as i32);
    let i0 = _mm512_loadu_si512(NAT[0].as_ptr() as *const __m512i);
    let i1 = _mm512_loadu_si512(NAT[1].as_ptr() as *const __m512i);
    for u in 0..N {
        let p = acc.add(32 * u);
        let lo = cm::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i));
        let hi = cm::reduce_vec::<Q>(_mm512_load_si512(p.add(16) as *const __m512i));
        let lo = barrett29::<Q>(_mm512_add_epi32(lo, up));
        let hi = barrett29::<Q>(_mm512_add_epi32(hi, up));
        let n0 = _mm512_permutex2var_epi32(lo, i0, hi);
        let n1 = _mm512_permutex2var_epi32(lo, i1, hi);
        let c0 = _mm512_mask_sub_epi32(n0, _mm512_cmpgt_epi32_mask(n0, half), n0, q);
        let c1 = _mm512_mask_sub_epi32(n1, _mm512_cmpgt_epi32_mask(n1, half), n1, q);
        let d = out.v[u].as_mut_ptr();
        _mm256_storeu_si256(d as *mut __m256i, _mm512_cvtepi32_epi16(c0));
        _mm256_storeu_si256(d.add(16) as *mut __m256i, _mm512_cvtepi32_epi16(c1));
    }
}

/// `v_ntt = sum_j c_j o W_j` modulo `Q1`, centered, one `Batch32` per batch position.
fn accumulate(w: &AuxData, ch: &ChallengeNtt, bpc: usize) -> Vec<Batch32> {
    let pairs = ch.pair.len();
    let mut acc = vec![AccVec([0i32; 16]); bpc * N * 2];
    let base = acc.as_mut_ptr() as *mut i32;
    let row = |j: usize, b: usize| w.batch(j * bpc + b).v.as_ptr() as *const i16;
    unsafe {
        for i in 0..pairs {
            for b in 0..bpc {
                accumulate_pair(
                    row(2 * i, b),
                    row(2 * i + 1, b),
                    ch.pair[i].as_ptr(),
                    base.add(32 * N * b),
                );
            }
            if (2 * (i + 1)) % FOLD_PERIOD == 0 {
                fold_back::<Q1>(base, bpc * N * 2);
            }
        }
        let mut out: Vec<Batch32> = (0..bpc)
            .map(|_| Batch32::zero(Representation::Ntt))
            .collect();
        for b in 0..bpc {
            drain::<Q1>(base.add(32 * N * b), &mut out[b]);
        }
        out
    }
}

// =============================================================================================
// A v
// =============================================================================================

/// `y[u] = sum_i A_i[u] v_i[u] mod q`, the commitment of the folded witness, on the commitment's
/// own packed accumulator: `|v| <= (q-1)/2` and `|A| <= (q-1)/2`, so no fold-back is needed
/// (`av_fits`).
pub(crate) fn a_times_v<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    unsafe {
        for b in 0..a.len() {
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_batch::<false>(v[b].v.as_ptr() as *const i16, ar, ar as *const i8, ap);
        }
    }
    cm::finish::<Q>(&acc)
}

/// The same for a quadratic-slot limb, on the quadratic accumulator of
/// [`crate::simd::commit`]: three sums per leaf, combined into the leaf's two rows at the end.
/// `|v| <= (q-1)/2` and `|A| <= (q-1)/2`, so the fold-back periods are the wider [`av_period`]
/// ones rather than the commitment's.
pub(crate) fn a_times_v_quad<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::QuadAcc::zero();
    let (p01, p2) = (
        acc.p01.as_mut_ptr() as *mut i32,
        acc.p2.as_mut_ptr() as *mut i32,
    );
    unsafe {
        for b in 0..a.len() {
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_quad_batch::<Q, false>(
                v[b].v.as_ptr() as *const i16,
                ar,
                ar as *const i8,
                p01,
                p2,
            );
            if (b + 1) % av_period(Q) == 0 {
                cm::reduce_quad_acc::<Q>(&mut acc);
            }
        }
    }
    cm::finish_quad::<Q>(&acc)
}

/// Batches of `A v` between two fold-backs of a quadratic limb's accumulators: both operands are
/// centered (`(q-1)/2`), so the widest lane grows by `16 ((q-1)/2)^2` per batch (the Karatsuba
/// `P_2`) or `8 ((q-1)/2)^2` (the schoolbook one).
pub const fn av_period(q: u16) -> usize {
    let per = if cm::karatsuba(q) {
        16 * cm::a_bound(q) * cm::a_bound(q)
    } else {
        8 * cm::a_bound(q) * cm::a_bound(q)
    };
    cm::period_for(q, per)
}
const _: () = assert!(av_period(2917) >= 1 && av_period(4861) >= 1 && av_period(12637) >= 1);

/// `sum_j c_j C_j` for one limb, in that limb's slot algebra: a scalar product per slot for a
/// splitting limb, and for a quadratic one the leaf product
/// `(c_0 C_0 + leaf_c c_1 C_1) + (c_0 C_1 + c_1 C_0) X` (`scalar::mul_quad_slots`), summed over
/// the chunks. The verifier's side of `A v = sum_j c_j C_j`.
pub(crate) fn combine_limb(
    q: u16,
    quad: bool,
    ch: &ChallengeNtt,
    cs: &[[u32; N]],
) -> [u32; N] {
    if !quad {
        let m = q as i64;
        let mut out = [0u32; N];
        for u in 0..N {
            let mut s = 0i64;
            for (j, c) in cs.iter().enumerate() {
                s += ch.slot[j][u] as i64 * c[u] as i64;
            }
            out[u] = s.rem_euclid(m) as u32;
        }
        return out;
    }
    match q {
        2917 => combine_quad::<2917>(ch, cs),
        4861 => combine_quad::<4861>(ch, cs),
        12637 => combine_quad::<12637>(ch, cs),
        _ => unreachable!("no quadratic limb with q = {q}"),
    }
}

fn combine_quad<const Q: u16>(ch: &ChallengeNtt, cs: &[[u32; N]]) -> [u32; N] {
    let q = Q as i64;
    let mut out = [0u32; N];
    for j in 0..QUAD_SLOTS {
        let leaf = ParamsQ::<Q>::LEAF_C[j] as i64;
        let (mut s0, mut s1) = (0i64, 0i64);
        for (t, c) in cs.iter().enumerate() {
            let (a0, a1) = (ch.slot[t][2 * j] as i64, ch.slot[t][2 * j + 1] as i64);
            let (b0, b1) = (c[2 * j] as i64, c[2 * j + 1] as i64);
            s0 = (s0 + a0 * b0 + leaf * (a1 * b1 % q)) % q;
            s1 = (s1 + a0 * b1 + a1 * b0) % q;
        }
        out[2 * j] = s0.rem_euclid(q) as u32;
        out[2 * j + 1] = s1.rem_euclid(q) as u32;
    }
    out
}

// =============================================================================================
// the output
// =============================================================================================

/// Wall time of the five stages of one [`fold`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FoldTimings {
    /// Embedding and transforming the `r` challenges modulo the base limb, and modulo every
    /// other limb when the check runs.
    pub challenge_ntt_ms: f64,
    /// The `r`-term slot-wise accumulation over the kept witness — the 85 MB stream.
    pub accumulate_ms: f64,
    /// `vertical_gen::intt_gen_batch32::<3889>` on the `len_ring/32` batches, and reading the
    /// centered coefficients out of the vertical layout.
    pub inverse_ntt_ms: f64,
    /// The forward transform of `v` modulo every additional limb (`vertical_gen` for a splitting
    /// one, `vertical_gen_quad` for a quadratic one).
    pub forward_add_ms: f64,
    /// `A v` for every limb and the four-way decomposition of the result.
    pub y_ms: f64,
    /// Everything.
    pub total_ms: f64,
}

/// The result of one fold.
pub struct FoldOutput {
    /// **The amortised witness**, `len_ring` elements of `R_648` in coefficient form, centered
    /// (`|coefficient| <= (q1-1)/2`, and in fact a few hundred — see [`max_abs_v`](Self::max_abs_v)),
    /// so these are the true integer coefficients of `sum_j c_j W_j`.
    pub v: Vec<RingElement>,
    /// `NTT(v)` for every limb of the key, `len_ring / 32` batches each, centered.
    pub v_ntt: Vec<Vec<Batch32>>,
    /// `A v` as the four `R_162` components over every limb, centered — the same shape as one
    /// column of a [`crate::VerticallyAlignedMatrix`] returned by a commitment.
    pub y: [PowerOfThreeRingElementWithLimbs; 4],
    /// The same `A v` before the decomposition: 648 rows in `[0, q)` per limb.
    pub y_raw: Vec<[u32; N]>,
    /// `max_i max_k |v_i[k]|`, the largest integer coefficient of the folded witness.
    ///
    /// A coefficient of `v` is a sum of `r * w` signed 0/1 terms (`r` challenges of weight `w`,
    /// each hitting one binary coefficient per term), so it has mean zero and standard deviation
    /// about `sqrt(r w / 2)` — 52 for `r = 256`, `w = 21` — and the maximum over the
    /// `648 * len_ring` coefficients lands a few hundred below, two orders under
    /// `q1 / 2 = 1944.5`. That margin is what makes the centered lift of `v mod q1` the true
    /// integer vector, and it is the only place in the fold where the integers matter.
    pub max_abs_v: i32,
    /// Wall time per stage.
    pub timings: FoldTimings,
}

impl FoldOutput {
    /// The four `R_162` components of `v[i]` in coefficient form: component `k` is the
    /// coefficients `4m + k`, `m = 0..162` (the basis `1, X, X^2, X^3` of `R_648` over
    /// `S = Z[Y]/(Y^162 - Y^81 + 1)`, `Y = X^4`).
    pub fn v_components(&self, i: usize) -> [[i16; N162]; 4] {
        let mut out = [[0i16; N162]; 4];
        for m in 0..N162 {
            for k in 0..4 {
                out[k][m] = self.v[i].v[4 * m + k];
            }
        }
        out
    }
}

// =============================================================================================
// the fold
// =============================================================================================

/// The folding step: `v = sum_j c_j W_j` and `A v`, from a witness kept by
/// [`CommitmentKey::commit_with_aux`](crate::CommitmentKey::commit_with_aux).
///
/// `challenges.len()` must be the number of chunks (even). The consistency identity
/// `A v = sum_j c_j C_j` is verified when `debug_assertions` are on; [`fold_checked`] forces it.
pub fn fold(key: &CommitmentKey, aux: &AuxData, challenges: &[ShortChallenge]) -> FoldOutput {
    fold_with(key, aux, challenges, cfg!(debug_assertions))
}

/// [`fold`] with the consistency check forced on (`sum_j c_j C_j == A v` for every limb).
pub fn fold_checked(
    key: &CommitmentKey,
    aux: &AuxData,
    challenges: &[ShortChallenge],
) -> FoldOutput {
    fold_with(key, aux, challenges, true)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The whole step, stage by stage.
pub fn fold_with(
    key: &CommitmentKey,
    aux: &AuxData,
    challenges: &[ShortChallenge],
    check: bool,
) -> FoldOutput {
    let r = aux.chunks();
    assert_eq!(challenges.len(), r, "one challenge per chunk");
    assert_eq!(aux.limbs(), key.limbs(), "the key and the auxiliary data disagree");
    let bpc = aux.batches_per_chunk();
    assert_eq!(bpc, key.len_ring() / 32, "the key and the chunks disagree");
    let mut t = FoldTimings::default();
    let t_all = Instant::now();

    // (a) the challenges, embedded as c(-X^4) and transformed modulo the base limb.
    let t0 = Instant::now();
    let ch1 = challenge_ntt::<Q1>(challenges);
    t.challenge_ntt_ms = ms(t0);

    // (b) the slot-wise inner product over the kept witness.
    let t0 = Instant::now();
    let v1 = accumulate(aux, &ch1, bpc);
    t.accumulate_ms = ms(t0);

    // (c) back to coefficients modulo q1, where v is a small integer vector: one batch kernel
    //     per batch position, whose output is already fully reduced and centered.
    let t0 = Instant::now();
    let mut vb = v1.clone();
    let half = (Q1 as i32 - 1) / 2;
    let mut max_abs = 0i32;
    unsafe {
        for b in vb.iter_mut() {
            intt_gen_batch32::<Q1>(b);
            max_abs = max_abs.max(max_abs_batch(b));
        }
    }
    assert!(
        max_abs <= half,
        "the folded witness does not fit the centered range of q1"
    );
    let v: Vec<RingElement> = (0..32 * bpc).map(|i| vb[i / 32].get(i % 32)).collect();
    t.inverse_ntt_ms = ms(t0);

    // (d) forward again modulo every additional limb (|coefficient| <= (q1-1)/2 < q, so every
    //     generic kernel's input bound holds); modulo q1 the accumulator's own output already is
    //     NTT(v), so it is kept.
    let t0 = Instant::now();
    let mut v_ntt = vec![v1];
    for k in 1..key.limbs() {
        let mut b = vb.clone();
        forward_limb(key.prime(k), key.is_quadratic(k), &mut b);
        v_ntt.push(b);
    }
    t.forward_add_ms = ms(t0);

    // (e) y = A v for every limb, then the four R_162 components.
    let t0 = Instant::now();
    let y_raw: Vec<[u32; N]> = (0..key.limbs())
        .map(|k| a_times_v_limb(key.prime(k), key.is_quadratic(k), key.row(k), &v_ntt[k]))
        .collect();
    let per: Vec<[crate::api::PowerOfThreeRingElement; 4]> = (0..key.limbs())
        .map(|k| components_of(key.prime(k), key.is_quadratic(k), &y_raw[k]))
        .collect();
    let y: [PowerOfThreeRingElementWithLimbs; 4] = core::array::from_fn(|c| {
        PowerOfThreeRingElementWithLimbs {
            limbs: per.iter().map(|d| d[c]).collect(),
        }
    });
    t.y_ms = ms(t0);

    // (f) the linearity identity that validates every step above.
    if check {
        let t0 = Instant::now();
        for k in 0..key.limbs() {
            let (q, quad) = (key.prime(k), key.is_quadratic(k));
            let ch = if k == 0 {
                None
            } else {
                Some(challenge_ntt_limb(q, quad, challenges))
            };
            let want = combine_limb(q, quad, ch.as_ref().unwrap_or(&ch1), &aux.raw[k]);
            for u in 0..N {
                assert_eq!(
                    want[u], y_raw[k][u],
                    "q = {q}: A v != sum_j c_j C_j at row {u}"
                );
            }
        }
        t.challenge_ntt_ms += ms(t0);
    }

    t.total_ms = ms(t_all);
    FoldOutput {
        v,
        v_ntt,
        y,
        y_raw,
        max_abs_v: max_abs,
        timings: t,
    }
}

/// `NTT(v)` for one limb, in place on centered coefficient batches, fully reduced and centered.
pub(crate) fn forward_limb(q: u16, quad: bool, bs: &mut [Batch32]) {
    unsafe {
        for b in bs.iter_mut() {
            b.representation = Representation::Coefficients;
            match (q, quad) {
                (3889, false) => {
                    ntt_gen_batch32::<3889>(b);
                    center_batch::<3889>(b);
                }
                (9721, false) => {
                    ntt_gen_batch32::<9721>(b);
                    center_batch::<9721>(b);
                }
                (2917, true) => {
                    ntt_quad_gen_batch32::<2917>(b);
                    center_batch::<2917>(b);
                }
                (4861, true) => {
                    ntt_quad_gen_batch32::<4861>(b);
                    center_batch::<4861>(b);
                }
                (12637, true) => {
                    ntt_quad_gen_batch32::<12637>(b);
                    center_batch::<12637>(b);
                }
                _ => unreachable!("no limb with q = {q}"),
            }
        }
    }
}

/// `A v` for one limb, dispatched on its prime.
pub(crate) fn a_times_v_limb(q: u16, quad: bool, a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    match (q, quad) {
        (3889, false) => a_times_v::<3889>(a, v),
        (9721, false) => a_times_v::<9721>(a, v),
        (2917, true) => a_times_v_quad::<2917>(a, v),
        (4861, true) => a_times_v_quad::<4861>(a, v),
        (12637, true) => a_times_v_quad::<12637>(a, v),
        _ => unreachable!("no limb with q = {q}"),
    }
}

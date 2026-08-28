//! Bit-slicing 128 `F162` (= 32 ring elements) straight into the `BinaryIndex32` rows the binary
//! NTT kernel consumes. Same machinery as [`crate::simd::transpose`] (GFNI 8x8 bit transposes,
//! a coefficient-major mask array, the GFNI index-row trick), re-planned for the interleaved
//! input: coefficient c of ring element p is bit `c / 4` of `F162` element `4p + (c mod 4)`.
//!
//! Write `M[k][m]` for the 32-bit mask `{ bit m of element 4p + k : p = 0..32 }`, `i = 4t + s`
//! for an output row and `U[m] = (M[0][m], M[1][m])`, `V[m] = (M[2][m], M[3][m])` as u64 pairs.
//! Row i needs the plane masks of the coefficients i, i+162, i+324, i+486, i.e.
//!
//! ```text
//!     s = 0, 1:  (k, m) = (s, t), (s+2, t+40), (s, t+81), (s+2, t+121)
//!     s = 2, 3:  (k, m) = (s, t), (s-2, t+41), (s, t+81), (s-2, t+122)
//! ```
//!
//! so the rows 4t and 4t+1 are exactly the low and high halves of `U[t], V[t+40], U[t+81],
//! V[t+121]`, and the rows 4t+2, 4t+3 those of `V[t], U[t+41], V[t+81], U[t+122]`. Phase 2 emits
//! the (k, k+1)-interleaved u64 planes directly, so phase 3 is the same 4 x 8 qword transpose as
//! in `transpose`, and phase 4 reads one plain 32-byte group per two rows.
//!
//! 1. **qword transpose.** A phase-1 row is a *pair* of ring elements (j, j+8) — 24 qwords,
//!    exactly three 8x8 transposes with no wasted column (a single ring element is 12 qwords and
//!    would waste four of sixteen). Four loads per row, all 24 columns used.
//! 2. **GFNI bit transpose + 8-way interleave.** One `vpermb` + one `vgf2p8affineqb` per column
//!    gives `T.byte[b]` = the 8-element mask of bit 64w + b; a byte/word/dword unpack tree over
//!    the four element groups and the two k's of a pair writes `U` and `V`.
//! 3. **4 x 8 qword transpose** so the four planes of a row pair are 32 contiguous bytes.
//! 4. **mask -> index rows.** One `vpermb` lays the group out as the *matrix* operand of a
//!    `vgf2p8affineqb` whose rows 0..3 are the four planes of row A in qwords 0..3 and of row B
//!    in qwords 4..7; the single affine therefore emits both rows' nibbles, `[n^A_0..n^A_31 |
//!    n^B_0..n^B_31]`. Per row one `vpermb` duplicates each byte and one `vpternlogd` masks the
//!    nibble and sets the +16 of the odd byte — three port-5 uops per two rows where the
//!    one-row-at-a-time form of `transpose` needs four, and no masked load (a merge-masked
//!    `vmovdqu64` costs a p0/p5 uop on top of the load on this core).
use crate::simd::transpose::BinaryIndex32;
use bin_fields::scalar::F162;
use core::arch::x86_64::*;
use core::mem::MaybeUninit;

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

/// Matrix operand of the two-row affine: qword q carries the four plane bytes of element group
/// `q & 3` of row `q >> 2` at the byte positions of the output bits 0..3. Bits 4..7 of the result
/// are masked away afterwards, so the remaining bytes are don't-care.
const fn affine_both() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut q = 0;
    while q < 8 {
        let mut b = 0;
        while b < 4 {
            t[8 * q + 7 - b] = (8 * b + 4 * (q >> 2) + (q & 3)) as u8;
            b += 1;
        }
        q += 1;
    }
    t
}

/// `vpermb` index that duplicates byte `off + p` of the affine output into bytes 2p, 2p+1.
const fn dup_idx(off: u8) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p] = off + p as u8;
        t[2 * p + 1] = off + p as u8;
        p += 1;
    }
    t
}

/// `[0, 16, 0, 16, ...]`: the +16 of the high-byte index of each `(n, n+16)` pair.
const fn plus16() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p + 1] = 16;
        p += 1;
    }
    t
}

static BYTE_PERM: A64<[u8; 64]> = A64(byte_perm());
static AFFINE_BOTH: A64<[u8; 64]> = A64(affine_both());
static DUP_A: A64<[u8; 64]> = A64(dup_idx(0));
static DUP_B: A64<[u8; 64]> = A64(dup_idx(32));
static PLUS16: A64<[u8; 64]> = A64(plus16());

const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

/// Scratch: the plane pairs and the 32-byte row-pair groups. `ag[4t..4t+4]` is the group of the
/// rows 4t, 4t+1 (41 of them), `bg[4t..]` that of the rows 4t+2, 4t+3 (40); both are padded to 48
/// so that phase 4 may read a full 64-byte vector at the last group.
#[repr(C, align(64))]
struct Work {
    u: [u64; 192],
    v: [u64; 192],
    ag: [u64; 4 * 48],
    bg: [u64; 4 * 48],
}

/// The 48 bit-sliced columns `(k, w, g)` of one batch.
#[repr(C, align(64))]
struct Cols([__m512i; 48]);

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

/// `T.byte[b]` = the 8-element mask of bit b of the column's qwords.
#[inline(always)]
unsafe fn bitslice(perm: __m512i, ident: __m512i, col: __m512i) -> __m512i {
    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, col))
}

/// Phase 1: the 48 columns `(k, w, g)`, `cols[(3k + w) * 4 + g]`.
#[inline(always)]
unsafe fn columns(base: *const u8, cp: *mut __m512i) {
    let perm = _mm512_load_si512(BYTE_PERM.0.as_ptr() as *const __m512i);
    let ident = _mm512_set1_epi64(GF_IDENT);
    for blk in 0..2 {
        let b = base.add(1536 * blk);
        let mut r0 = [_mm512_setzero_si512(); 8];
        let mut r1 = [_mm512_setzero_si512(); 8];
        let mut r2 = [_mm512_setzero_si512(); 8];
        for j in 0..8 {
            // row j = ring elements (j, j+8): 24 qwords, four loads.
            let p = b.add(96 * j);
            r0[j] = _mm512_loadu_si512(p as *const __m512i);
            let lo = _mm512_maskz_loadu_epi64(0x0f, p.add(64) as *const i64);
            r1[j] = _mm512_mask_loadu_epi64(lo, 0xf0, p.add(736) as *const i64);
            r2[j] = _mm512_loadu_si512(p.add(800) as *const __m512i);
        }
        let c0 = transpose8x8_q(r0);
        let c1 = transpose8x8_q(r1);
        let c2 = transpose8x8_q(r2);
        for k in 0..4 {
            for w in 0..3 {
                let c = 3 * k + w;
                let g0 = if c < 8 { c0[c] } else { c1[c - 8] };
                let g1 = if c < 4 { c1[c + 4] } else { c2[c - 4] };
                _mm512_store_si512(cp.add(4 * c + 2 * blk), bitslice(perm, ident, g0));
                _mm512_store_si512(cp.add(4 * c + 2 * blk + 1), bitslice(perm, ident, g1));
            }
        }
    }
}

/// Phase 2: writes the 64 u64 `(M_k0[m], M_k1[m])`, m = 64w..64w+64, of one (k pair, w).
#[inline(always)]
unsafe fn interleave8(cp: *const __m512i, k0: usize, w: usize, dst: *mut u64) {
    let a = |g: usize| _mm512_load_si512(cp.add(4 * (3 * k0 + w) + g));
    let e = |g: usize| _mm512_load_si512(cp.add(4 * (3 * (k0 + 1) + w) + g));
    let lab = _mm512_unpacklo_epi8(a(0), a(1));
    let hab = _mm512_unpackhi_epi8(a(0), a(1));
    let lcd = _mm512_unpacklo_epi8(a(2), a(3));
    let hcd = _mm512_unpackhi_epi8(a(2), a(3));
    let lef = _mm512_unpacklo_epi8(e(0), e(1));
    let hef = _mm512_unpackhi_epi8(e(0), e(1));
    let lgh = _mm512_unpacklo_epi8(e(2), e(3));
    let hgh = _mm512_unpackhi_epi8(e(2), e(3));

    let r0 = _mm512_unpacklo_epi16(lab, lcd);
    let r1 = _mm512_unpackhi_epi16(lab, lcd);
    let r2 = _mm512_unpacklo_epi16(hab, hcd);
    let r3 = _mm512_unpackhi_epi16(hab, hcd);
    let q0 = _mm512_unpacklo_epi16(lef, lgh);
    let q1 = _mm512_unpackhi_epi16(lef, lgh);
    let q2 = _mm512_unpacklo_epi16(hef, hgh);
    let q3 = _mm512_unpackhi_epi16(hef, hgh);

    // s0..s3 lane L hold m = 16L..16L+8, s4..s7 lane L hold m = 16L+8..16L+16.
    let s0 = _mm512_unpacklo_epi32(r0, q0);
    let s1 = _mm512_unpackhi_epi32(r0, q0);
    let s2 = _mm512_unpacklo_epi32(r1, q1);
    let s3 = _mm512_unpackhi_epi32(r1, q1);
    let s4 = _mm512_unpacklo_epi32(r2, q2);
    let s5 = _mm512_unpackhi_epi32(r2, q2);
    let s6 = _mm512_unpacklo_epi32(r3, q3);
    let s7 = _mm512_unpackhi_epi32(r3, q3);

    let x0 = _mm512_shuffle_i64x2::<0x44>(s0, s1);
    let x1 = _mm512_shuffle_i64x2::<0xEE>(s0, s1);
    let x2 = _mm512_shuffle_i64x2::<0x44>(s2, s3);
    let x3 = _mm512_shuffle_i64x2::<0xEE>(s2, s3);
    let y0 = _mm512_shuffle_i64x2::<0x44>(s4, s5);
    let y1 = _mm512_shuffle_i64x2::<0xEE>(s4, s5);
    let y2 = _mm512_shuffle_i64x2::<0x44>(s6, s7);
    let y3 = _mm512_shuffle_i64x2::<0xEE>(s6, s7);

    let d = dst as *mut __m512i;
    _mm512_store_si512(d, _mm512_shuffle_i64x2::<0x88>(x0, x2));
    _mm512_store_si512(d.add(2), _mm512_shuffle_i64x2::<0xDD>(x0, x2));
    _mm512_store_si512(d.add(4), _mm512_shuffle_i64x2::<0x88>(x1, x3));
    _mm512_store_si512(d.add(6), _mm512_shuffle_i64x2::<0xDD>(x1, x3));
    _mm512_store_si512(d.add(1), _mm512_shuffle_i64x2::<0x88>(y0, y2));
    _mm512_store_si512(d.add(3), _mm512_shuffle_i64x2::<0xDD>(y0, y2));
    _mm512_store_si512(d.add(5), _mm512_shuffle_i64x2::<0x88>(y1, y3));
    _mm512_store_si512(d.add(7), _mm512_shuffle_i64x2::<0xDD>(y1, y3));
}

/// Phase 3: eight 32-byte row-pair groups from the four plane streams.
#[inline(always)]
unsafe fn gather8(p0: *const i64, p1: *const i64, p2: *const i64, p3: *const i64, dst: *mut u64) {
    let a = _mm512_loadu_si512(p0 as *const __m512i);
    let b = _mm512_loadu_si512(p1 as *const __m512i);
    let c = _mm512_loadu_si512(p2 as *const __m512i);
    let d = _mm512_loadu_si512(p3 as *const __m512i);
    let l01 = _mm512_unpacklo_epi64(a, b);
    let h01 = _mm512_unpackhi_epi64(a, b);
    let l23 = _mm512_unpacklo_epi64(c, d);
    let h23 = _mm512_unpackhi_epi64(c, d);
    let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
    let s1 = _mm512_shuffle_i64x2::<0xEE>(l01, l23);
    let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
    let s3 = _mm512_shuffle_i64x2::<0xEE>(h01, h23);
    let o = dst as *mut __m512i;
    _mm512_store_si512(o, _mm512_shuffle_i64x2::<0x88>(s0, s2));
    _mm512_store_si512(o.add(1), _mm512_shuffle_i64x2::<0xDD>(s0, s2));
    _mm512_store_si512(o.add(2), _mm512_shuffle_i64x2::<0x88>(s1, s3));
    _mm512_store_si512(o.add(3), _mm512_shuffle_i64x2::<0xDD>(s1, s3));
}

/// 128 `F162` (32 ring elements, 3072 contiguous bytes at `base`) -> the kernel's index rows.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
unsafe fn slice_raw(base: *const u8, out: &mut BinaryIndex32) {
    let mut cols: MaybeUninit<Cols> = MaybeUninit::uninit();
    let cp = cols.as_mut_ptr() as *mut __m512i;
    columns(base, cp);

    let mut work: MaybeUninit<Work> = MaybeUninit::uninit();
    let wk = work.as_mut_ptr();
    let u = (*wk).u.as_mut_ptr();
    let v = (*wk).v.as_mut_ptr();
    for w in 0..3 {
        interleave8(cp, 0, w, u.add(64 * w));
        interleave8(cp, 2, w, v.add(64 * w));
    }

    let (ag, bg) = ((*wk).ag.as_mut_ptr(), (*wk).bg.as_mut_ptr());
    for i in 0..6 {
        let t = 8 * i;
        gather8(
            u.add(t) as *const i64,
            v.add(t + 40) as *const i64,
            u.add(t + 81) as *const i64,
            v.add(t + 121) as *const i64,
            ag.add(4 * t),
        );
    }
    for i in 0..5 {
        let t = 8 * i;
        gather8(
            v.add(t) as *const i64,
            u.add(t + 41) as *const i64,
            v.add(t + 81) as *const i64,
            u.add(t + 122) as *const i64,
            bg.add(4 * t),
        );
    }

    let ident = _mm512_set1_epi64(GF_IDENT);
    let am = _mm512_load_si512(AFFINE_BOTH.0.as_ptr() as *const __m512i);
    let da = _mm512_load_si512(DUP_A.0.as_ptr() as *const __m512i);
    let db = _mm512_load_si512(DUP_B.0.as_ptr() as *const __m512i);
    let p16 = _mm512_load_si512(PLUS16.0.as_ptr() as *const __m512i);
    let lo4 = _mm512_set1_epi8(0x0f);
    let op = out.rows.as_mut_ptr() as *mut __m512i;
    for t in 0..41 {
        let src = _mm512_loadu_si512(ag.add(4 * t) as *const __m512i);
        let n = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(am, src));
        let a = _mm512_permutexvar_epi8(da, n);
        let b = _mm512_permutexvar_epi8(db, n);
        _mm512_store_si512(op.add(4 * t), _mm512_ternarylogic_epi32::<0xEA>(a, lo4, p16));
        _mm512_store_si512(op.add(4 * t + 1), _mm512_ternarylogic_epi32::<0xEA>(b, lo4, p16));
    }
    for t in 0..40 {
        let src = _mm512_loadu_si512(bg.add(4 * t) as *const __m512i);
        let n = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(am, src));
        let a = _mm512_permutexvar_epi8(da, n);
        let b = _mm512_permutexvar_epi8(db, n);
        _mm512_store_si512(op.add(4 * t + 2), _mm512_ternarylogic_epi32::<0xEA>(a, lo4, p16));
        _mm512_store_si512(op.add(4 * t + 3), _mm512_ternarylogic_epi32::<0xEA>(b, lo4, p16));
    }
}

/// The `vpermb` index rows of the 32 ring elements formed by `elems` (element 4p + k supplies the
/// coefficients 4m + k of ring element p).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_f162_into(elems: &[F162; 128], out: &mut BinaryIndex32) {
    slice_raw(elems.as_ptr() as *const u8, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_f162(elems: &[F162; 128]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    slice_f162_into(elems, &mut out);
    out
}

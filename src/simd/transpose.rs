//! Bit-slicing transpose: 32 `BinaryPoly` (bit i = coefficient i) into the two kernel-side
//! layouts.
//!
//! * `slice_polys` -> `BinaryBatch32` (`idx[i][p]` = the nibble b_i | b_{i+162}<<1 |
//!   b_{i+324}<<2 | b_{i+486}<<3 of polynomial p), the reference layout of `types.rs`.
//! * `slice_polys_idx` -> [`BinaryIndex32`], the form the NTT kernel actually consumes:
//!   `rows[i][2p] = n`, `rows[i][2p+1] = 16 + n`, i.e. the `vpermb` byte-index pair that reads the
//!   low and high half of entry n of a byte-split 16-entry i16 table. Producing this directly
//!   saves the kernel a 162-row expansion pass (~490 uops per batch).
//!
//! Pipeline (all AVX-512), for a batch of 32 polynomials:
//!
//! 1. **qword transpose + byte transpose + GFNI bit transpose.** For each group g of 8
//!    polynomials an 8x8 qword transpose (24 `vshufi64x2`/`vpunpckq` uops) turns 8 rows
//!    "poly -> its words" into 8 columns "word w -> the 8 polys". One `vpermb` reorders each
//!    column into `Y.byte[8B + 7-j] = byte B of poly j`, and one `vgf2p8affineqb` with the
//!    identity matrix 0x8040201008040201 as the *vector* operand computes, per qword B,
//!    `T.qword[B].byte[t].bit[p] = bit t of byte B of poly p`, i.e. the 8-poly mask of coefficient
//!    64w + 8B + t.
//! 2. **4-way byte interleave** of the four groups (8 `vpunpck` + 8 `vshufi64x2` per 64-bit word)
//!    gives `mask[b]`, a u32 whose bit p is coefficient b of polynomial p, for b in 0..704.
//! 3. **4 x 81 qword transpose** of `mask` viewed as u64s, so that the four planes a row needs
//!    (`mask[2h + 162*plane]`, plane = 0..3) become 32 contiguous bytes.
//! 4. **mask -> index row**, with no mask registers at all: one merge-masked load pulls those 32
//!    bytes in next to two constant bytes (0x00, 0xFF); one `vpermb` lays them out as the operand
//!    of a second `vgf2p8affineqb` (rows 4..7 of the affine matrix are the four planes, row 3 is
//!    0x00/0xFF and adds the +16), which emits `[n_0..n_31 | n_0+16..n_31+16]`; a final `vpermb`
//!    interleaves that into the index row. Two output rows per 32-byte load, four `vpermb` and
//!    two `vgf2p8affineqb` - the previous version needed four `kmovq` (port 5!) plus six port-0/5
//!    uops per two rows.
use crate::types::{BinaryBatch32, BinaryPoly};
use core::arch::x86_64::*;

/// The `vpermb` byte-index rows the binary NTT kernel consumes: `rows[i][2p] = n(i, p)`,
/// `rows[i][2p+1] = 16 + n(i, p)` where `n(i, p)` is the 4-bit nibble
/// b_i | b_{i+162}<<1 | b_{i+324}<<2 | b_{i+486}<<3 of polynomial p.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct BinaryIndex32 {
    pub rows: [[u8; 64]; 162],
}

impl BinaryIndex32 {
    pub fn zero() -> Self {
        BinaryIndex32 { rows: [[0u8; 64]; 162] }
    }
    /// Scalar reference / adapter from the `types.rs` layout.
    pub fn from_nibbles(b: &BinaryBatch32) -> Self {
        let mut r = Self::zero();
        for i in 0..162 {
            for p in 0..32 {
                r.rows[i][2 * p] = b.idx[i][p];
                r.rows[i][2 * p + 1] = b.idx[i][p] + 16;
            }
        }
        r
    }
}

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

/// Phases 1 and 2: `masks[b]` = u32 whose bit p is coefficient b of polynomial p.
#[inline(always)]
unsafe fn bit_transpose(polys: &[BinaryPoly; 32], masks: &mut Masks) {
    let perm = _mm512_load_si512(BYTE_PERM.0.as_ptr() as *const __m512i);
    let ident = _mm512_set1_epi64(GF_IDENT);
    let base = polys.as_ptr() as *const u8;

    let mut t: [[__m512i; 11]; 4] = [[_mm512_setzero_si512(); 11]; 4];
    for g in 0..4 {
        let mut rows: [__m512i; 8] = [_mm512_setzero_si512(); 8];
        for j in 0..8 {
            rows[j] = _mm512_loadu_si512(base.add((8 * g + j) * 88) as *const __m512i);
        }
        let c = transpose8x8_q(rows);
        for w in 0..8 {
            let y = _mm512_permutexvar_epi8(perm, c[w]);
            t[g][w] = _mm512_gf2p8affine_epi64_epi8::<0>(ident, y);
        }
        for j in 0..8 {
            rows[j] = _mm512_maskz_loadu_epi64(0x07, base.add((8 * g + j) * 88 + 64) as *const i64);
        }
        let c = transpose8x8_q(rows);
        for w in 0..3 {
            let y = _mm512_permutexvar_epi8(perm, c[w]);
            t[g][8 + w] = _mm512_gf2p8affine_epi64_epi8::<0>(ident, y);
        }
    }

    for w in 0..11 {
        let l01 = _mm512_unpacklo_epi8(t[0][w], t[1][w]);
        let h01 = _mm512_unpackhi_epi8(t[0][w], t[1][w]);
        let l23 = _mm512_unpacklo_epi8(t[2][w], t[3][w]);
        let h23 = _mm512_unpackhi_epi8(t[2][w], t[3][w]);
        let r0 = _mm512_unpacklo_epi16(l01, l23);
        let r1 = _mm512_unpackhi_epi16(l01, l23);
        let r2 = _mm512_unpacklo_epi16(h01, h23);
        let r3 = _mm512_unpackhi_epi16(h01, h23);
        let s0 = _mm512_shuffle_i64x2::<0x44>(r0, r1);
        let s1 = _mm512_shuffle_i64x2::<0xEE>(r0, r1);
        let s2 = _mm512_shuffle_i64x2::<0x44>(r2, r3);
        let s3 = _mm512_shuffle_i64x2::<0xEE>(r2, r3);
        let p = masks.0.as_mut_ptr().add(64 * w) as *mut __m512i;
        _mm512_store_si512(p, _mm512_shuffle_i64x2::<0x88>(s0, s2));
        _mm512_store_si512(p.add(1), _mm512_shuffle_i64x2::<0xDD>(s0, s2));
        _mm512_store_si512(p.add(2), _mm512_shuffle_i64x2::<0x88>(s1, s3));
        _mm512_store_si512(p.add(3), _mm512_shuffle_i64x2::<0xDD>(s1, s3));
    }
}

/// AVX-512 equivalent of `BinaryBatch32::from_polys_scalar` (the `types.rs` nibble layout).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_polys(polys: &[BinaryPoly; 32]) -> BinaryBatch32 {
    let mut masks = Masks([0u32; 704]);
    bit_transpose(polys, &mut masks);
    let mut out = BinaryBatch32::zero();
    let one = _mm512_set1_epi8(1);
    let two = _mm512_set1_epi8(2);
    let four = _mm512_set1_epi8(4);
    let eight = _mm512_set1_epi8(8);
    let mp = masks.0.as_ptr() as *const u64;
    let op = out.idx.as_mut_ptr() as *mut __m512i;
    for h in 0..81 {
        let k0 = *mp.add(h) as __mmask64;
        let k1 = *mp.add(h + 81) as __mmask64;
        let k2 = *mp.add(h + 162) as __mmask64;
        let k3 = *mp.add(h + 243) as __mmask64;
        let a = _mm512_maskz_mov_epi8(k0, one);
        let b = _mm512_maskz_mov_epi8(k1, two);
        let c = _mm512_maskz_mov_epi8(k2, four);
        let d = _mm512_maskz_mov_epi8(k3, eight);
        let z = _mm512_ternarylogic_epi32::<0xfe>(a, b, c);
        _mm512_store_si512(op.add(h), _mm512_or_si512(z, d));
    }
    out
}

#[repr(C, align(64))]
struct Planes([u64; 4 * 88]);

/// The layout the NTT kernel wants, produced directly (no mask registers, no expansion pass).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_polys_idx_into(polys: &[BinaryPoly; 32], out: &mut BinaryIndex32) {
    let mut masks = Masks([0u32; 704]);
    bit_transpose(polys, &mut masks);

    // phase 3: pl[4h + b] = mask64[h + 81b], so the four planes of rows 2h, 2h+1 are contiguous.
    let mut pl: core::mem::MaybeUninit<Planes> = core::mem::MaybeUninit::uninit();
    let plp = pl.as_mut_ptr() as *mut u64;
    let m64 = masks.0.as_ptr() as *const i64;
    for grp in 0..11 {
        let h = 8 * grp;
        let a = _mm512_loadu_si512(m64.add(h) as *const __m512i);
        let b = _mm512_loadu_si512(m64.add(h + 81) as *const __m512i);
        let c = _mm512_loadu_si512(m64.add(h + 162) as *const __m512i);
        let d = _mm512_loadu_si512(m64.add(h + 243) as *const __m512i);
        let l01 = _mm512_unpacklo_epi64(a, b);
        let h01 = _mm512_unpackhi_epi64(a, b);
        let l23 = _mm512_unpacklo_epi64(c, d);
        let h23 = _mm512_unpackhi_epi64(c, d);
        let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
        let s1 = _mm512_shuffle_i64x2::<0xEE>(l01, l23);
        let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
        let s3 = _mm512_shuffle_i64x2::<0xEE>(h01, h23);
        let p = plp.add(4 * h) as *mut __m512i;
        _mm512_store_si512(p, _mm512_shuffle_i64x2::<0x88>(s0, s2));
        _mm512_store_si512(p.add(1), _mm512_shuffle_i64x2::<0xDD>(s0, s2));
        _mm512_store_si512(p.add(2), _mm512_shuffle_i64x2::<0x88>(s1, s3));
        _mm512_store_si512(p.add(3), _mm512_shuffle_i64x2::<0xDD>(s1, s3));
    }

    // phase 4: two index rows per 32-byte group.
    let ident = _mm512_set1_epi64(GF_IDENT);
    let ai0 = _mm512_load_si512(AFFINE_IDX0.0.as_ptr() as *const __m512i);
    let ai1 = _mm512_load_si512(AFFINE_IDX1.0.as_ptr() as *const __m512i);
    let inter = _mm512_load_si512(INTERLEAVE.0.as_ptr() as *const __m512i);
    let ch = _mm512_load_si512(CONST_HALF.0.as_ptr() as *const __m512i);
    let op = out.rows.as_mut_ptr() as *mut __m512i;
    for h in 0..81 {
        let src = _mm512_mask_loadu_epi64(ch, 0x0f, plp.add(4 * h) as *const i64);
        let n0 = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(ai0, src));
        let n1 = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(ai1, src));
        _mm512_store_si512(op.add(2 * h), _mm512_permutexvar_epi8(inter, n0));
        _mm512_store_si512(op.add(2 * h + 1), _mm512_permutexvar_epi8(inter, n1));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_polys_idx(polys: &[BinaryPoly; 32]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    slice_polys_idx_into(polys, &mut out);
    out
}

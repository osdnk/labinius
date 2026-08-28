//! The public API: a commitment key, a commitment, and the height-4 view of its output over the
//! 243-rd cyclotomic ring.
//!
//! Everything the crate does is reachable from three types.
//!
//! * [`CommitmentKey`] holds the Ajtai matrix `A` for the base prime [`BASE_PRIME`] and for each
//!   [`AdditionalLimb`] the key was built with, in the NTT domain, centered, in the layout the
//!   AVX-512 kernel streams.
//! * [`CommitmentKey::commit`] maps a witness — a plain `&[F162]`, read as binary ring elements
//!   of `R_648 = Z_q[X]/(X^648 - X^324 + 1)` four `F162` at a time — to a
//!   [`VerticallyAlignedMatrix`] of [`PowerOfThreeRingElementWithLimbs`].
//! * [`PowerOfThreeRingElement`] is one element of `R_162 = Z_q[Z]/Phi_243(Z)` in its NTT domain:
//!   162 slots, centered signed residues in `[-(q-1)/2, (q-1)/2]`, slot `s` holding the evaluation at the primitive
//!   243-rd root of unity named by [`POW3_SLOT_EXP`]`[s]`.
//!
//! ```no_run
//! use bin_ntt::{AdditionalLimb::*, CommitmentKey};
//! # use bin_fields::scalar::F162;
//! # let witness: Vec<F162> = Vec::new();
//! let ck = CommitmentKey::random(1 << 18, 0xC0FFEE, &[Q9721, Q2917]);
//! let c = ck.commit(&witness, 1);
//! let slots = &c.get(0, 0).base().v[..];        // component 0 mod 3889, 162 centered slots
//! let other = &c.get(0, 0).additional(1).v[..]; // the same component mod 2917
//! ```
//!
//! # Why the output is a matrix of height 4
//!
//! One commitment is a single ring element `y in R_648`. With `Y = X^4` the ring `R_648` is a free
//! module of rank 4 over `S = Z_q[Y]/(Y^162 - Y^81 + 1)` with basis `1, X, X^2, X^3`, and
//! `S = R_162` under `Z = -Y` (see "The lift is a ring extension of degree 4" in the README). So
//! `y = y_0 + X y_1 + X^2 y_2 + X^3 y_3` with the four `y_k` in `R_162`, and the commitment over
//! `R_648` read in that basis *is* a rank-4 module-SIS commitment over `R_162` whose 4 x 4 blocks
//! are the `Y`-twisted circulants of multiplication by the uniform `A_i`. The four rows of the
//! returned matrix are those four components; the `r` columns are the `r` chunks the witness was
//! split into, each committed under the same key.
//!
//! # The decomposition, in the NTT domain
//!
//! `psi` is the primitive 1944-th root of unity that fixes the slot order of `R_648`
//! ([`crate::params::SLOT_EXP`]), `theta = psi^4` is then a primitive 486-th root of unity and
//! `i = psi^486` a primitive 4th root of unity. Splitting the coefficients of `y` by residue mod 4
//! gives `y_k(Y) = sum_m y[4m + k] Y^m`, and for every unit `u` mod 1944
//!
//! ```text
//!     y(psi^u) = sum_k psi^{uk} y_k(theta^u),        theta^u depends only on v = u mod 486.
//! ```
//!
//! The units mod 1944 fall into 162 classes of four modulo 486, `u = v + 486 t` with `t = 0..3`,
//! so writing `E_t = y(psi^{v + 486 t})` and `Y_k(v) = y_k(theta^v)`,
//!
//! ```text
//!     E_t = sum_k psi^{vk} i^{tk} Y_k(v),            Y_k(v) = 4^-1 psi^{-vk} sum_t i^{-tk} E_t.
//! ```
//!
//! [`decompose_648_to_4x162`] is exactly this map, from the 648 slots of `y` in tree order to the
//! four times 162 slots of the components. Since `i^2 = -1`, the length-4 inverse DFT is a radix-4
//! butterfly with a single product `i (E_1 - E_3)`, so the whole map is 5 modular multiplications
//! per class; done 16 lanes at a time (`vpgatherdd` for the four `E_t`, a 43-bit Barrett for the
//! reduction) it costs ~0.35 us per element and prime. It runs on the *one* output element rather
//! than on the witness, so at `r = 1` it is 3 us of a 15 ms commitment; even at `r = 256` chunks,
//! where it runs 512 times, it is 0.18 ms of 14 ms.
//!
//! # The slot order of `R_162`
//!
//! [`POW3_SLOT_EXP`] lists the 162 values `v` in the order in which they first appear as
//! `SLOT_EXP[j] mod 486` while `j` runs over the 648 slots of `R_648` — the tree order of the
//! big ring, restricted. Slot `s` of a component holds `y_k(theta^{v_s})`, equivalently the
//! component read as a polynomial in `Z = -Y` evaluated at `Z = -theta^{v_s}`, which is a
//! primitive 243-rd root of unity. Outputs are centered into `[-(q-1)/2, (q-1)/2]`; note the `4^-1`
//! factor above, which is already applied.
use crate::params::{
    inv_mod, pow_mod, Params, ParamsQ, CONDUCTOR, CONDUCTOR_QUAD, N, QS, QS_QUAD, QUAD_CLASS_SLOT,
    QUAD_POW3_CLASS, SLOT_EXP,
};
use crate::rng::Rng;
use crate::simd::commit as cm;
use crate::types::{Batch32, Representation};
use bin_fields::scalar::F162;
use core::arch::x86_64::*;
use std::time::Instant;

/// The primes of the default limb list: the base and [`AdditionalLimb::Q9721`]. Both are
/// `1 mod 1944`, so `R_q` splits into 648 linear factors and the transform is complete.
pub const PRIMES: [u16; 2] = QS;

/// The base limb, present in every key: the only prime below `2^13` for which `R_648` splits
/// completely, and the limb the folded witness lives in.
pub const BASE_PRIME: u16 = QS[0];

/// A limb a [`CommitmentKey`] can carry on top of [`BASE_PRIME`].
///
/// `Q9721` is the second splitting prime (648 linear slots, `vertical_bin_asm`); the other three
/// are the *quadratic-slot* primes of [`crate::params::QS_QUAD`] — `q = 1 mod 972` but not mod
/// 1944, so `Phi_1944` factors into 324 irreducible quadratics and the transform ends at
/// `Z_q[X]/(X^2 - psi'^u)` leaves (`vertical_bin_quad`). Both kinds produce the same public
/// output: four elements of `R_162`, 162 slots each, in the [`POW3_SLOT_EXP`] order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AdditionalLimb {
    Q2917,
    Q4861,
    Q9721,
    Q12637,
}

impl AdditionalLimb {
    /// Every limb, cheapest first (the order of the ranking in the README).
    pub const ALL: [AdditionalLimb; 4] = [
        AdditionalLimb::Q2917,
        AdditionalLimb::Q4861,
        AdditionalLimb::Q9721,
        AdditionalLimb::Q12637,
    ];

    /// The prime.
    pub const fn prime(self) -> u16 {
        match self {
            AdditionalLimb::Q2917 => QS_QUAD[0],
            AdditionalLimb::Q4861 => QS_QUAD[1],
            AdditionalLimb::Q9721 => QS[1],
            AdditionalLimb::Q12637 => QS_QUAD[2],
        }
    }

    /// Does `R_648` end in 324 quadratic leaves for this prime (rather than 648 linear slots)?
    pub const fn is_quadratic(self) -> bool {
        !matches!(self, AdditionalLimb::Q9721)
    }

    /// The limb of a prime, if it is one.
    pub fn from_prime(q: u16) -> Option<AdditionalLimb> {
        AdditionalLimb::ALL.into_iter().find(|l| l.prime() == q)
    }
}

/// The default limb list of [`CommitmentKey::random_default`].
pub const DEFAULT_LIMBS: [AdditionalLimb; 1] = [AdditionalLimb::Q9721];

/// Degree of the small ring `R_162 = Z_q[Z]/Phi_243(Z)`; `Phi_243(Z) = Z^162 + Z^81 + 1`.
pub const N162: usize = 162;

/// Order of `theta = psi^4`, the root of unity the small ring's slots evaluate at.
pub const CONDUCTOR162: u32 = CONDUCTOR / 4;

// =============================================================================================
// slot order
// =============================================================================================

const fn pow3_tables() -> ([u16; N162], [[u16; N162]; 4]) {
    let mut v_of = [0u16; N162];
    let mut idx = [[0u16; N162]; 4];
    let mut pos = [u16::MAX; CONDUCTOR162 as usize];
    let mut n = 0usize;
    let mut j = 0usize;
    while j < N {
        let u = SLOT_EXP[j] as usize;
        let v = u % CONDUCTOR162 as usize;
        let t = (u - v) / CONDUCTOR162 as usize;
        if pos[v] == u16::MAX {
            pos[v] = n as u16;
            v_of[n] = v as u16;
            n += 1;
        }
        idx[t][pos[v] as usize] = j as u16;
        j += 1;
    }
    assert!(n == N162);
    (v_of, idx)
}

const POW3: ([u16; N162], [[u16; N162]; 4]) = pow3_tables();

/// `POW3_SLOT_EXP[s] = v_s`: slot `s` of a [`PowerOfThreeRingElement`] holds the value at
/// `theta^{v_s}`, `theta = psi^4` a primitive 486-th root of unity. The order is the order in
/// which the 162 classes `v = u mod 486` first appear in [`SLOT_EXP`] (the tree order of `R_648`).
pub const POW3_SLOT_EXP: [u16; N162] = POW3.0;

/// `SLOT_648[t][s]` = the slot of `R_648` holding `y(psi^{v_s + 486 t})`.
const SLOT_648: [[u16; N162]; 4] = POW3.1;

/// The same table as 32-bit gather offsets, padded to [`PAD`] with 0.
const SLOT_IDX: [[i32; PAD]; 4] = {
    let mut g = [[0i32; PAD]; 4];
    let mut t = 0;
    while t < 4 {
        let mut s = 0;
        while s < N162 {
            g[t][s] = SLOT_648[t][s] as i32;
            s += 1;
        }
        t += 1;
    }
    g
};

const _: () = {
    let mut t = 0;
    while t < 4 {
        let mut s = 0;
        while s < N162 {
            assert!((SLOT_648[t][s] as usize) < N);
            s += 1;
        }
        t += 1;
    }
};

/// [`POW3_SLOT_EXP`] as a slice.
pub fn pow3_slot_exp() -> &'static [u16; N162] {
    &POW3_SLOT_EXP
}

// =============================================================================================
// the decomposition
// =============================================================================================

/// Per-prime constants of the radix-4 recombination.
struct Pow3Consts<const Q: u16>;

impl<const Q: u16> Pow3Consts<Q> {
    const PSI_POW: [u16; CONDUCTOR as usize] = {
        let mut t = [0u16; CONDUCTOR as usize];
        let mut e = 0;
        while e < CONDUCTOR as usize {
            t[e] = pow_mod(Params::<Q>::PSI as u64, e as u64, Q as u64) as u16;
            e += 1;
        }
        t
    };
    /// `i = psi^486`, the primitive 4th root of unity: the only multiplication the length-4
    /// inverse DFT needs, since `i^2 = -1` and `i^3 = -i`.
    const I: u32 = Self::PSI_POW[CONDUCTOR162 as usize] as u32;
    /// `M = floor(2^43 / q)`, the Barrett magic of [`barrett29`].
    const BARRETT_M: u64 = (1u64 << 43) / Q as u64;
    /// `TWIST[k][s] = 4^-1 psi^{-v_s k} mod q`, the whole scaling of component `k` at slot `s`.
    const TWIST: [[u16; PAD]; 4] = {
        let inv4 = inv_mod(4, Q as u64);
        let mut t = [[0u16; PAD]; 4];
        let mut s = 0;
        while s < N162 {
            let v = POW3_SLOT_EXP[s] as usize;
            let mut k = 0;
            while k < 4 {
                let e = (CONDUCTOR as usize - v * k % CONDUCTOR as usize) % CONDUCTOR as usize;
                t[k][s] = (inv4 * Self::PSI_POW[e] as u64 % Q as u64) as u16;
                k += 1;
            }
            s += 1;
        }
        t
    };
}

/// The 162 slots padded to 11 AVX-512 vectors of 16 `u32`; the pad lanes carry zeros.
const PAD: usize = 176;

/// `p mod q` for `p < 2^29`, 16 lanes at a time: `t = (p * M) >> 43` is `floor(p/q)` or one less
/// (`M = floor(2^43/q) < 2^32`, so `p * M < 2^61` and the quotient defect is below `p / 2^43`),
/// which leaves `p - t q` in `[0, 2q)` for one conditional subtract.
#[target_feature(enable = "avx512f")]
unsafe fn barrett29<const Q: u16>(p: __m512i) -> __m512i {
    let q = _mm512_set1_epi32(Q as i32);
    let mag = _mm512_set1_epi64(Pow3Consts::<Q>::BARRETT_M as i64);
    let lo = _mm512_set1_epi64(0xFFFF_FFFFu32 as i64);
    let he = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_and_si512(p, lo), mag));
    let ho = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(p), mag));
    let t = _mm512_or_si512(he, _mm512_slli_epi64::<32>(ho));
    let r = _mm512_sub_epi32(p, _mm512_mullo_epi32(t, q));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, q))
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn recombine<const Q: u16>(y: &[u32; N], out: &mut [PowerOfThreeRingElement; 4]) {
    let q = _mm512_set1_epi32(Q as i32);
    let q2 = _mm512_set1_epi32(2 * Q as i32);
    let iq = _mm512_set1_epi32(Pow3Consts::<Q>::I as i32);
    let tw = &Pow3Consts::<Q>::TWIST;
    for b in 0..PAD / 16 {
        let ld = |t: usize| {
            let ix = _mm512_loadu_si512(SLOT_IDX[t].as_ptr().add(16 * b) as *const __m512i);
            _mm512_i32gather_epi32::<4>(ix, y.as_ptr() as *const i32)
        };
        let (e0, e1, e2, e3) = (ld(0), ld(1), ld(2), ld(3));
        let a = _mm512_add_epi32(e0, e2);
        let d0 = _mm512_sub_epi32(_mm512_add_epi32(e0, q), e2);
        let c = _mm512_add_epi32(e1, e3);
        let d1 = _mm512_sub_epi32(_mm512_add_epi32(e1, q), e3);
        let id = barrett29::<Q>(_mm512_mullo_epi32(d1, iq));
        let m = [
            _mm512_add_epi32(a, c),
            _mm512_sub_epi32(_mm512_add_epi32(d0, q), id),
            _mm512_sub_epi32(_mm512_add_epi32(a, q2), c),
            _mm512_add_epi32(d0, id),
        ];
        let k: __mmask16 = if 16 * b + 16 <= N162 {
            !0
        } else {
            (1u16 << (N162 - 16 * b)) - 1
        };
        for kk in 0..4 {
            let t = _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                tw[kk].as_ptr().add(16 * b) as *const __m256i
            ));
            let r = barrett29::<Q>(_mm512_mullo_epi32(m[kk], t));
            // centre: r in [0, q) -> r - q where r > (q-1)/2, i.e. (-(q-1)/2 ..= (q-1)/2)
            let hi = _mm512_cmpgt_epi32_mask(r, _mm512_set1_epi32((Q as i32 - 1) / 2));
            let r = _mm512_mask_sub_epi32(r, hi, r, q);
            _mm512_mask_cvtepi32_storeu_epi16(out[kk].v.as_mut_ptr().add(16 * b) as *mut _, k, r);
        }
    }
}

const _: () = assert!(4 * (PRIMES[1] as u64) * (PRIMES[1] as u64) < (1u64 << 29));

/// The same map, straight into the four ring elements the API returns.
pub(crate) fn decompose_components<const Q: u16>(y: &[u32; N]) -> [PowerOfThreeRingElement; 4] {
    debug_assert!(
        y.iter().all(|&x| x < Q as u32),
        "decompose wants a fully reduced input"
    );
    let mut out = [PowerOfThreeRingElement::zero(); 4];
    unsafe { recombine::<Q>(y, &mut out) };
    out
}

pub fn decompose_648_to_4x162<const Q: u16>(y: &[u32; N]) -> [[u32; N162]; 4] {
    let c = decompose_components::<Q>(y);
    let mut out = [[0u32; N162]; 4];
    for k in 0..4 {
        for s in 0..N162 {
            out[k][s] = (c[k].v[s] as i32).rem_euclid(Q as i32) as u32;
        }
    }
    out
}

/// The two trees name the 162 `R_162` classes in the same order, so a commitment's four
/// components mean the same thing in every limb.
const _: () = {
    let mut s = 0;
    while s < N162 {
        assert!(QUAD_POW3_CLASS[s] == POW3_SLOT_EXP[s]);
        s += 1;
    }
};

/// The scaling of the quadratic-slot decomposition: `2^-1` on the sum, `2^-1 psi'^{-v_s}` on the
/// difference — the 2-point butterfly of [`crate::scalar::decompose_quad_648_to_4x162`] with its
/// per-class constant precomputed.
struct QuadConsts<const Q: u16>;

impl<const Q: u16> QuadConsts<Q> {
    const INV2: u64 = inv_mod(2, Q as u64);
    const TW: [u16; N162] = {
        let mut t = [0u16; N162];
        let mut s = 0;
        while s < N162 {
            let v = QUAD_POW3_CLASS[s] as u32;
            let e = (CONDUCTOR_QUAD - v) % CONDUCTOR_QUAD;
            t[s] = (Self::INV2 * ParamsQ::<Q>::psi_pow(e) as u64 % Q as u64) as u16;
            s += 1;
        }
        t
    };
}

/// The same four components for a quadratic-slot limb: one 2-point butterfly per class over the
/// two leaves that share it — `Y_0 = (E^+ + E^-)/2`, `Y_2 = (E^+ - E^-)/(2 psi'^v)` and likewise
/// for the `X` rows — in the [`POW3_SLOT_EXP`] order the assertion above pins down, centered.
/// [`crate::scalar::decompose_quad_648_to_4x162`] is the reference `tests/limbs.rs` checks it
/// against.
pub(crate) fn decompose_components_quad<const Q: u16>(
    y: &[u32; N],
) -> [PowerOfThreeRingElement; 4] {
    debug_assert!(
        y.iter().all(|&x| x < Q as u32),
        "decompose wants a fully reduced input"
    );
    let q = Q as u64;
    let half = (Q - 1) / 2;
    let ctr = |x: u64| -> i16 {
        if x as u16 > half {
            x as i16 - Q as i16
        } else {
            x as i16
        }
    };
    let mut out = [PowerOfThreeRingElement::zero(); 4];
    for s in 0..N162 {
        let jp = QUAD_CLASS_SLOT[0][s] as usize;
        let jm = QUAD_CLASS_SLOT[1][s] as usize;
        let tw = QuadConsts::<Q>::TW[s] as u64;
        for k in 0..2 {
            let ep = y[2 * jp + k] as u64;
            let em = y[2 * jm + k] as u64;
            out[k].v[s] = ctr((ep + em) * QuadConsts::<Q>::INV2 % q);
            out[k + 2].v[s] = ctr((ep + q - em) * tw % q);
        }
    }
    out
}

/// The four components of a commitment for any limb, split or quadratic, dispatched on the prime.
pub(crate) fn components_of(q: u16, quad: bool, y: &[u32; N]) -> [PowerOfThreeRingElement; 4] {
    match (q, quad) {
        (3889, false) => decompose_components::<3889>(y),
        (9721, false) => decompose_components::<9721>(y),
        (2917, true) => decompose_components_quad::<2917>(y),
        (4861, true) => decompose_components_quad::<4861>(y),
        (12637, true) => decompose_components_quad::<12637>(y),
        _ => unreachable!("no limb with q = {q}"),
    }
}

// =============================================================================================
// ring elements
// =============================================================================================

/// One element of `R_162 = Z_q[Z]/Phi_243(Z)` for a single prime, in the NTT domain: 162 slots,
/// centered signed residues in `[-(q-1)/2, (q-1)/2]`, slot `s` holding the evaluation at the
/// primitive 243-rd root of unity indexed by [`POW3_SLOT_EXP`]`[s]` (see the module documentation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowerOfThreeRingElement {
    pub v: [i16; N162],
}

impl PowerOfThreeRingElement {
    pub fn zero() -> Self {
        PowerOfThreeRingElement { v: [0i16; N162] }
    }
    /// The canonical non-negative representatives in `[0, q)`.
    pub fn normalized(&self, q: u16) -> [u32; N162] {
        let mut out = [0u32; N162];
        for s in 0..N162 {
            out[s] = (self.v[s] as i32).rem_euclid(q as i32) as u32;
        }
        out
    }
}

impl Default for PowerOfThreeRingElement {
    fn default() -> Self {
        Self::zero()
    }
}

/// One element of `R_162` given by its residues modulo the key's limbs: `limbs[0]` is the residue
/// modulo [`BASE_PRIME`] and `limbs[1 + i]` the one modulo the key's `i`-th
/// [`AdditionalLimb`], in the order the key was built with.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PowerOfThreeRingElementWithLimbs {
    pub limbs: Vec<PowerOfThreeRingElement>,
}

impl PowerOfThreeRingElementWithLimbs {
    /// `n` zero limbs.
    pub fn zero(n: usize) -> Self {
        PowerOfThreeRingElementWithLimbs {
            limbs: vec![PowerOfThreeRingElement::zero(); n],
        }
    }
    /// The residue modulo [`BASE_PRIME`].
    pub fn base(&self) -> &PowerOfThreeRingElement {
        &self.limbs[0]
    }
    /// The residue modulo the key's `i`-th additional limb.
    pub fn additional(&self, i: usize) -> &PowerOfThreeRingElement {
        &self.limbs[1 + i]
    }
    /// Number of limbs (`1 + additional`).
    pub fn len(&self) -> usize {
        self.limbs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.limbs.is_empty()
    }
}

// =============================================================================================
// the matrix
// =============================================================================================

/// A `rows x cols` matrix stored column by column, one column being one commitment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VerticallyAlignedMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> VerticallyAlignedMatrix<T> {
    /// `data` in column-major order: entry `(row, col)` at `data[col * rows + row]`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "column-major data of the wrong length"
        );
        VerticallyAlignedMatrix { rows, cols, data }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) out of range"
        );
        &self.data[col * self.rows + row]
    }
    /// One whole column — one commitment, its `rows` components in order.
    pub fn column(&self, col: usize) -> &[T] {
        assert!(col < self.cols, "column {col} out of range");
        &self.data[col * self.rows..(col + 1) * self.rows]
    }
    pub fn columns(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.rows)
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

// =============================================================================================
// the key
// =============================================================================================

/// `F162` elements per `Batch32` of the key: 128 `F162` = 32 ring elements.
const F162_PER_BATCH: usize = 128;

/// The Ajtai matrix `A` — one row of uniform ring elements, in the NTT domain, centered, for
/// every limb of the key. Opaque: the layout is the vertical one the AVX-512 kernels stream
/// ([`crate::simd::commit`]).
///
/// Limb 0 is [`BASE_PRIME`]; limb `1 + i` is the `i`-th [`AdditionalLimb`] the key was built with.
/// A split limb's rows are the 648 tree-order slots, a quadratic limb's are the 648 rows of the
/// quadratic tree (the two coefficients of each of the 324 leaves); in both cases every row is
/// uniform in `[-(q-1)/2, (q-1)/2]`.
pub struct CommitmentKey {
    a: Vec<Vec<Batch32>>,
    additional: Vec<AdditionalLimb>,
    len_f162: usize,
}

impl CommitmentKey {
    /// A uniformly random key for `len_f162` witness elements over the base limb and
    /// `additional`, deterministically from `seed`. `len_f162` must be a multiple of 128
    /// (= 32 ring elements, one `Batch32` per limb).
    ///
    /// Costs `(1 + additional.len()) * len_f162 / 128 * 41472` bytes: 85 MB per limb for
    /// `len_f162 = 2^18`.
    pub fn random(len_f162: usize, seed: u64, additional: &[AdditionalLimb]) -> Self {
        assert!(
            len_f162 > 0 && len_f162 % F162_PER_BATCH == 0,
            "len_f162 must be a multiple of 128"
        );
        assert_eq!(core::mem::size_of::<F162>(), 24, "F162 is not 24 bytes");
        let mut seen = Vec::new();
        for l in additional {
            assert!(!seen.contains(l), "the limb {l:?} is listed twice");
            seen.push(*l);
        }
        let nb = len_f162 / F162_PER_BATCH;
        let primes: Vec<u16> = core::iter::once(BASE_PRIME)
            .chain(additional.iter().map(|l| l.prime()))
            .collect();
        let a = primes
            .iter()
            .enumerate()
            .map(|(k, &q)| {
                let half = ((q - 1) / 2) as i16;
                let mut rng = Rng::new(seed ^ (0x9E37_79B9_u64.wrapping_mul(k as u64 + 1)));
                (0..nb)
                    .map(|_| {
                        let mut b = Batch32::zero(Representation::Ntt);
                        for j in 0..N {
                            for p in 0..32 {
                                b.v[j][p] = rng.below(q as u32) as i16 - half;
                            }
                        }
                        b
                    })
                    .collect()
            })
            .collect();
        CommitmentKey {
            a,
            additional: additional.to_vec(),
            len_f162,
        }
    }

    /// [`random`](Self::random) with the default limb list ([`DEFAULT_LIMBS`], the base and 9721).
    pub fn random_default(len_f162: usize, seed: u64) -> Self {
        Self::random(len_f162, seed, &DEFAULT_LIMBS)
    }

    /// Length of the key in `F162` elements: the size of one chunk of witness it commits to.
    pub fn len_f162(&self) -> usize {
        self.len_f162
    }

    /// Length of the key in `R_648` ring elements (`len_f162 / 4`).
    pub fn len_ring(&self) -> usize {
        self.len_f162 / 4
    }

    /// Number of limbs, `1 + additional().len()`.
    pub fn limbs(&self) -> usize {
        self.a.len()
    }

    /// The additional limbs, in the order the commitment reports them.
    pub fn additional(&self) -> &[AdditionalLimb] {
        &self.additional
    }

    /// The prime of limb `k` (limb 0 is [`BASE_PRIME`]).
    pub fn prime(&self, k: usize) -> u16 {
        if k == 0 {
            BASE_PRIME
        } else {
            self.additional[k - 1].prime()
        }
    }

    /// Does limb `k` use the quadratic-slot tree?
    pub fn is_quadratic(&self, k: usize) -> bool {
        k > 0 && self.additional[k - 1].is_quadratic()
    }

    /// The matrix itself, for limb `k`, in the vertical layout [`crate::simd::commit`] streams:
    /// `row(k)[b].v[j][p]` is row `j` of `A_{32b + p}` modulo `prime(k)`, centered. Exposed so
    /// that a caller can drive the kernel directly, or check a commitment against
    /// [`crate::scalar`].
    pub fn row(&self, k: usize) -> &[Batch32] {
        &self.a[k]
    }

    /// The limbs in the form the kernel driver takes.
    pub(crate) fn limb_list(&self) -> Vec<cm::Limb<'_>> {
        (0..self.limbs())
            .map(|k| cm::Limb {
                q: self.prime(k),
                quad: self.is_quadratic(k),
                a: &self.a[k],
            })
            .collect()
    }

    /// Bytes of `A` held, over all limbs.
    pub fn bytes(&self) -> usize {
        self.limbs() * (self.len_f162 / F162_PER_BATCH) * core::mem::size_of::<Batch32>()
    }

    /// The four `R_162` components of the raw commitments of one chunk, one entry per limb.
    fn components(&self, raw: &[[u32; N]]) -> [PowerOfThreeRingElementWithLimbs; 4] {
        let mut out: [PowerOfThreeRingElementWithLimbs; 4] =
            core::array::from_fn(|_| PowerOfThreeRingElementWithLimbs {
                limbs: Vec::with_capacity(self.limbs()),
            });
        for k in 0..self.limbs() {
            let d = components_of(self.prime(k), self.is_quadratic(k), &raw[k]);
            for (c, e) in d.into_iter().enumerate() {
                out[c].limbs.push(e);
            }
        }
        out
    }

    /// Commit to `witness` in `r` chunks under the same key.
    ///
    /// `r` must be a power of two and `witness.len()` must be `r * self.len_f162()`. Chunk `c` is
    /// `witness[c * len .. (c + 1) * len]`; it is read as `len / 4` binary ring elements of
    /// `R_648` (`crate::f162`), sliced once, and transformed and multiplied into the inner product
    /// `y = sum_i A_i * NTT(w_i)` once per limb. Each limb's `y` is then split into its four
    /// `R_162` components ([`decompose_648_to_4x162`] for a split limb,
    /// [`crate::scalar::decompose_quad_648_to_4x162`] for a quadratic one), which become column
    /// `c`.
    ///
    /// The returned matrix is 4 x `r`. Reusing one key across `r` chunks is what makes a large `r`
    /// faster: the same `A` is streamed `r` times instead of once, but at `r >= 8` it is small
    /// enough to stay in L3.
    pub fn commit(
        &self,
        witness: &[F162],
        r: usize,
    ) -> VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs> {
        self.commit_timed(witness, r).0
    }

    /// [`commit`](Self::commit), with the wall time of the two phases.
    pub fn commit_timed(
        &self,
        witness: &[F162],
        r: usize,
    ) -> (
        VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
        Timings,
    ) {
        self.check_witness(witness, r);
        let limbs = self.limb_list();
        let mut data = Vec::with_capacity(4 * r);
        let mut t = Timings {
            chunks: r,
            commit_ms: 0.0,
            decompose_ms: 0.0,
            total_ms: 0.0,
        };
        let mut st = cm::Scratch::new(&limbs);
        let mut raw = vec![[0u32; N]; limbs.len()];
        let t_all = Instant::now();
        for c in 0..r {
            let chunk = &witness[c * self.len_f162..(c + 1) * self.len_f162];
            let t0 = Instant::now();
            cm::commit_limbs_into(chunk, &limbs, None, &mut st, &mut raw);
            t.commit_ms += ms(t0);
            let t1 = Instant::now();
            data.extend(self.components(&raw));
            t.decompose_ms += ms(t1);
        }
        t.total_ms = ms(t_all);
        (VerticallyAlignedMatrix::new(4, r, data), t)
    }

    /// [`commit`](Self::commit), returning the auxiliary data a later folding step
    /// ([`crate::fold::fold`]) consumes.
    ///
    /// The matrix is bit-identical to [`commit`](Self::commit)'s; the only difference is that the
    /// transform of every ring element modulo the base limb is also written out — by the same
    /// block sink that feeds the base multiplication, with non-temporal stores — so the 85 MB an
    /// [`AuxData`] holds for a 2^16-element witness cost a few percent rather than the 2.3 ms a
    /// separate cached write of that size would.
    pub fn commit_with_aux(
        &self,
        witness: &[F162],
        r: usize,
    ) -> (
        VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
        AuxData,
    ) {
        let mut aux = AuxData::new(self.len_ring(), r, self.limbs());
        let c = self.commit_into_aux(witness, r, &mut aux);
        (c, aux)
    }

    /// [`commit_with_aux`](Self::commit_with_aux) writing into a buffer the caller already owns.
    ///
    /// The 85 MB an [`AuxData`] holds for a 2^16-element witness is one `mmap` and 20 736 first
    /// touches, ~20 ms of page faults the kernel charges to whoever writes the pages first — more
    /// than the commitment itself. A prover that folds repeatedly allocates one [`AuxData::new`]
    /// and reuses it, and then keeping the witness costs what it should: the non-temporal stores,
    /// which hide behind the transform.
    pub fn commit_into_aux(
        &self,
        witness: &[F162],
        r: usize,
        aux: &mut AuxData,
    ) -> VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs> {
        self.check_witness(witness, r);
        let bpc = self.a[0].len();
        assert!(
            aux.chunks == r && aux.batches.len() == r * bpc && aux.raw.len() == self.limbs(),
            "the auxiliary buffer does not match this key and r"
        );
        for v in aux.raw.iter_mut() {
            v.clear();
        }
        let limbs = self.limb_list();
        let mut st = cm::Scratch::new(&limbs);
        let mut raw = vec![[0u32; N]; limbs.len()];
        let mut data = Vec::with_capacity(4 * r);
        for c in 0..r {
            let chunk = &witness[c * self.len_f162..(c + 1) * self.len_f162];
            cm::commit_limbs_into(
                chunk,
                &limbs,
                Some(&mut aux.batches[c * bpc..(c + 1) * bpc]),
                &mut st,
                &mut raw,
            );
            data.extend(self.components(&raw));
            for (k, y) in raw.iter().enumerate() {
                aux.raw[k].push(*y);
            }
        }
        VerticallyAlignedMatrix::new(4, r, data)
    }

    fn check_witness(&self, witness: &[F162], r: usize) {
        assert!(r.is_power_of_two(), "r must be a power of two");
        assert_eq!(
            witness.len(),
            r * self.len_f162,
            "witness must be r * len_f162() elements ({} * {})",
            r,
            self.len_f162
        );
    }
}

/// `n` `Batch32`s whose 41472 bytes each are never read before the kernel writes them (zeroing
/// 85 MB would cost 4 ms of the very DRAM traffic the non-temporal stores are there to avoid).
fn uninit_batches(n: usize) -> Vec<Batch32> {
    let mut v: Vec<Batch32> = Vec::with_capacity(n);
    unsafe {
        let p = v.as_mut_ptr();
        for i in 0..n {
            (*p.add(i)).representation = Representation::Ntt;
        }
        v.set_len(n);
    }
    v
}

// =============================================================================================
// the auxiliary data
// =============================================================================================

/// Everything a commitment leaves behind that the folding step ([`crate::fold`]) needs, and
/// nothing a caller has to look inside: the witness's transform modulo [`BASE_PRIME`] in the
/// layout the kernel produced it, and the raw 648-row commitments of the `r` chunks for every
/// limb of the key.
///
/// Produced by [`CommitmentKey::commit_with_aux`] as a by-product of the commitment itself, so it
/// costs a memory stream rather than a second transform. For 2^16 ring elements it holds 2048
/// `Batch32` = 85 MB, whatever the limb list is: only the base limb's transform is kept.
pub struct AuxData {
    /// The transform modulo [`BASE_PRIME`], `batches[b].v[u][p]` = slot `u` of ring element
    /// `32 b + p`, lazily reduced (`|v| <= 7.5 q`, the binary kernel's declared output bound).
    pub(crate) batches: Vec<Batch32>,
    /// `raw[k][j]` = the commitment of chunk `j` for limb `k`, 648 rows in `[0, q)`.
    pub(crate) raw: Vec<Vec<[u32; N]>>,
    pub(crate) chunks: usize,
}

impl AuxData {
    /// An empty buffer for `r` chunks of `len_ring` ring elements each over `limbs` limbs, to be
    /// filled by [`CommitmentKey::commit_into_aux`]. The `Batch32`s are left uninitialised: the
    /// kernel writes every one of their 41472 bytes before anything reads them, and zeroing 85 MB
    /// would cost more than the commitment.
    pub fn new(len_ring: usize, r: usize, limbs: usize) -> AuxData {
        assert!(r > 0 && len_ring > 0 && len_ring % 32 == 0 && limbs > 0);
        AuxData {
            batches: uninit_batches(r * len_ring / 32),
            raw: (0..limbs).map(|_| Vec::with_capacity(r)).collect(),
            chunks: r,
        }
    }

    /// Number of chunks the witness was split into (`r`).
    pub fn chunks(&self) -> usize {
        self.chunks
    }
    /// Number of limbs the commitments were taken over.
    pub fn limbs(&self) -> usize {
        self.raw.len()
    }
    /// Bytes of witness transform held.
    pub fn bytes(&self) -> usize {
        self.batches.len() * core::mem::size_of::<Batch32>()
    }
    /// `Batch32`s per chunk: `len_ring / 32`, 8 for a 1024-`F162` key.
    pub fn batches_per_chunk(&self) -> usize {
        self.batches.len() / self.chunks
    }
    /// Batch `i` of the kept transform, `32 i` .. `32 i + 32` in witness order; chunk `j` is
    /// batches `j * batches_per_chunk()` onward. Read-only, for checking the fold against
    /// [`crate::scalar`].
    pub fn batch(&self, i: usize) -> &Batch32 {
        &self.batches[i]
    }
    /// The commitment of chunk `j` for limb `k`: 648 rows in `[0, q)`, before the four-way
    /// decomposition. This is the form the fold's consistency identity `A v = sum_j c_j C_j`
    /// lives in.
    pub fn commitment(&self, k: usize, j: usize) -> &[u32; N] {
        &self.raw[k][j]
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Wall time inside one [`CommitmentKey::commit_timed`] call.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Number of chunks (`r`).
    pub chunks: usize,
    /// Front end + transform + base multiplication, every limb, all chunks.
    pub commit_ms: f64,
    /// The four-way decomposition of the `r` output elements, every limb.
    pub decompose_ms: f64,
    /// Everything, including building the output matrix.
    pub total_ms: f64,
}

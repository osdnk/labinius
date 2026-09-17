//! The two rings and the decomposition between them: `R_648 = Z_q[X]/(X^648 - X^324 + 1)` and
//! the height-4 view of its elements over the 243-rd cyclotomic ring. This is the engine room of
//! [`crate::scheme`], not the surface a caller uses; the key that commits into it is
//! [`crate::key`].
//!
//! * [`PowerOfThreeRingElement`] (`RingElement162` in the public surface) is one element of
//!   `R_162 = Z_q[Z]/Phi_243(Z)` in its NTT domain: 162 slots, centered signed residues in
//!   `[-(q-1)/2, (q-1)/2]`, slot `s` holding the evaluation at the primitive 243-rd root of unity
//!   named by [`POW3_SLOT_EXP`]`[s]`.
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
use crate::limb::dispatch_limb;
use crate::params::{
    inv_mod, pow_mod, quadratic_slots, Params, ParamsQ, CONDUCTOR, CONDUCTOR_QUAD, N, QS, QS_LARGE,
    QS_QUAD, QUAD_CLASS_SLOT, QUAD_POW3_CLASS, SLOT_EXP,
};
use core::arch::x86_64::*;

/// The primes of the default limb list: the default base and [`Modulus::Q9721_FS_S`]. Both are
/// `1 mod 1944`, so `R_q` splits into 648 linear factors and the transform is complete.
pub const PRIMES: [u16; 2] = QS;

/// The default base limb, the one every existing configuration uses: the only prime below `2^13`
/// for which `R_648` splits completely. Any [`Modulus`] can take its place
/// ([`crate::Params::with_base`]); the base is the limb whose transform a commitment keeps and
/// whose domain the fold runs in.
pub const BASE_PRIME: u16 = QS[0];

/// A limb a [`CommitmentKey`] can carry, as its base or on top of it.
///
/// `Q3889_FS_S`, `Q9721_FS_S`, `Q17497_FS_L` and `Q19441_FS_L` are splitting primes (648 linear slots): the first two
/// run the hand-scheduled `ntt::bin_asm`, the two above `2^14` the reduce-at-every-level
/// `ntt::bin_large`. The other three are the *quadratic-slot* primes of
/// [`crate::params::QS_QUAD`] — `q = 1 mod 972` but not mod 1944, so `Phi_1944` factors into 324
/// irreducible quadratics and the transform ends at `Z_q[X]/(X^2 - psi'^u)` leaves
/// (`ntt::bin_quad`). All three kinds produce the same public output: four elements of
/// `R_162`, 162 slots each, in the [`POW3_SLOT_EXP`] order.
///
/// The suffix names the family, so a call site reads the trade-off off the name: `_FS_S` fully
/// splitting below `2^14`, `_Q_S` quadratic-slot, `_FS_L` fully splitting above `2^14`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Modulus {
    Q2917_Q_S,
    Q3889_FS_S,
    Q4861_Q_S,
    Q9721_FS_S,
    Q12637_Q_S,
    Q17497_FS_L,
    Q19441_FS_L,
}

impl Modulus {
    /// Every limb, in ascending order of modulus — which is the README's cost ranking except
    /// that the base 3889 sits one place later there, behind 4861.
    pub const ALL: [Modulus; 7] = [
        Modulus::Q2917_Q_S,
        Modulus::Q3889_FS_S,
        Modulus::Q4861_Q_S,
        Modulus::Q9721_FS_S,
        Modulus::Q12637_Q_S,
        Modulus::Q17497_FS_L,
        Modulus::Q19441_FS_L,
    ];

    /// The default base limb, [`BASE_PRIME`].
    pub const BASE: Modulus = Modulus::Q3889_FS_S;

    /// The prime.
    pub const fn prime(self) -> u16 {
        match self {
            Modulus::Q2917_Q_S => QS_QUAD[0],
            Modulus::Q3889_FS_S => QS[0],
            Modulus::Q4861_Q_S => QS_QUAD[1],
            Modulus::Q9721_FS_S => QS[1],
            Modulus::Q12637_Q_S => QS_QUAD[2],
            Modulus::Q17497_FS_L => QS_LARGE[0],
            Modulus::Q19441_FS_L => QS_LARGE[1],
        }
    }

    /// Does `R_648` end in 324 quadratic leaves for this prime (rather than 648 linear slots)?
    pub const fn is_quadratic(self) -> bool {
        quadratic_slots(self.prime())
    }

    /// The limb of a prime, if it is one.
    pub fn from_prime(q: u16) -> Option<Modulus> {
        Modulus::ALL.into_iter().find(|l| l.prime() == q)
    }
}

const _: () = assert!(Modulus::BASE.prime() == BASE_PRIME);

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
pub(crate) const SLOT_648: [[u16; N162]; 4] = POW3.1;

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
    /// `M = floor(2^43 / q)`, the Barrett magic of [`barrett31`].
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

/// `p mod q` for `0 <= p < 2^31`, 16 lanes at a time: `t = (p * M) >> 43` is `floor(p/q)` or one
/// less (the quotient defect is below `p / 2^43`), which leaves `p - t q` in `[0, 2q)` for one
/// conditional subtract. `M = floor(2^43/q)` has to fit the u32 halves `vpmuludq` reads and the
/// product has to stay inside a u64 lane; `barrett31_fits` asserts both for every limb.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn barrett31<const Q: u16>(p: __m512i) -> __m512i {
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
        let id = barrett31::<Q>(_mm512_mullo_epi32(d1, iq));
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
            let r = barrett31::<Q>(_mm512_mullo_epi32(m[kk], t));
            // centre: r in [0, q) -> r - q where r > (q-1)/2, i.e. (-(q-1)/2 ..= (q-1)/2)
            let hi = _mm512_cmpgt_epi32_mask(r, _mm512_set1_epi32((Q as i32 - 1) / 2));
            let r = _mm512_mask_sub_epi32(r, hi, r, q);
            _mm512_mask_cvtepi32_storeu_epi16(out[kk].v.as_mut_ptr().add(16 * b) as *mut _, k, r);
        }
    }
}

/// [`recombine`] feeds [`barrett31`] products of a lane below `4q` by a twist below `q`, so the
/// hypotheses are `4 q^2 < 2^31`, `M < 2^32` and `4 q^2 M < 2^64`. The `2^29` the magic was first
/// written for was the head-room of the two primes below `2^14`; 17497 and 19441 need the honest
/// bound, and clear it with a factor of 1.4.
const fn barrett31_fits(q: u16) -> bool {
    let p = 4 * q as u64 * q as u64;
    let m = (1u64 << 43) / q as u64;
    p < (1u64 << 31) && m < (1u64 << 32) && p <= u64::MAX / m
}
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(barrett31_fits(QS[i]) && barrett31_fits(QS_LARGE[i]));
        i += 1;
    }
};

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
pub fn components_of(q: u16, y: &[u32; N]) -> [PowerOfThreeRingElement; 4] {
    dispatch_limb!(
        q,
        split |Q| decompose_components::<Q>(y),
        quad |Q| decompose_components_quad::<Q>(y),
    )
}

pub mod element;
pub use element::*;

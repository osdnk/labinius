//! Short challenges over the 3^5-th cyclotomic ring `R_162 = Z[Z]/Phi_243(Z)`,
//! `Phi_243(Z) = Z^162 + Z^81 + 1`, and the blake3 transcript that samples them.
//!
//! A challenge is a weight-`w` *signed* element of `R_162`: `w` of the 162 coefficients are `+-1`
//! and the rest are zero, stored sparsely as the sorted positions and a sign per position.
//! Sampling is uniform over the weight-`w` position sets (a partial Fisher-Yates driven by the
//! transcript's XOF), each candidate is signed, and the pair is rejected until the challenge is
//! *short in the canonical embedding*:
//!
//! ```text
//!     max_u |c(zeta^u)|^2 <= bound^2,     zeta = exp(2 pi i / 243), gcd(u, 3) = 1,
//! ```
//!
//! the 162 primitive 243-rd roots of unity. That quantity — [`canonical_inf_norm_sq`] — is the
//! squared operator norm of multiplication by `c` on `R_162 (x) C` *in the canonical embedding*.
//! The power basis of a power-of-three conductor is not orthogonal there, so the expansion factor
//! on coefficient vectors that a security argument uses is `sqrt(3)` times the canonical bound —
//! `sqrt(3) * 12 = 20.78` at the default, not `12`. The default is `weight = 28`, `bound = 12`.
//!
//! The signs are not drawn from the transcript: they are a keyed blake3 hash of the candidate's
//! own position set ([`ShortChallenge::signed`]), a public map that leaves the sampler's
//! randomness, the number of squeezes and the wire untouched. It is applied to the candidate
//! *before* the bound is tested, because `Z^162 = -Z^81 - 1` mixes coefficients sign-dependently
//! and the accepted challenge is the signed one. Signs exist because the sum `C = sum_j c_j` over
//! a round's `r` challenges is the offset the fold carries: with `0/1` coefficients its
//! coefficients grow like `r`, with zero-mean ones like `sqrt(r)`.
//!
//! ```no_run
//! use bin_ntt::challenge::{sample_short_challenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
//! # use bin_ntt::ring::PowerOfThreeRingElementWithLimbs;
//! # let commitment = [PowerOfThreeRingElementWithLimbs::zero(2)];
//! let mut t = Transcript::new(b"bin-ntt/example");
//! t.absorb_elements(&commitment);
//! let (c, attempts) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
//! ```
use crate::ring::{PowerOfThreeRingElementWithLimbs, N162};
use crate::fields::scalar::F162;
use blake3::Hasher;
use core::arch::x86_64::*;
use std::f64::consts::PI;
use std::sync::LazyLock;

/// Conductor of the small ring: the challenge is evaluated at primitive 243-rd roots of unity.
pub const CONDUCTOR243: usize = 243;

/// Largest weight a [`ShortChallenge`] can hold.
pub const MAX_WEIGHT: usize = 32;

/// The weight the crate samples at unless told otherwise.
pub const DEFAULT_WEIGHT: usize = 28;

/// The default canonical-embedding bound: `max_u |c(zeta^u)|^2 <= 12^2 = 144`.
pub const DEFAULT_BOUND: f64 = 12.0;

// =============================================================================================
// transcript
// =============================================================================================

/// A blake3 Fiat-Shamir transcript: absorb, then derive.
///
/// Every derivation clones the absorbing state, appends a per-transcript sample counter and a
/// label, and reads from the resulting extendable output, so two samples from the same transcript
/// are independent, a sample is bound to everything absorbed before it, and the whole thing is a
/// deterministic function of the absorbed bytes.
#[derive(Clone)]
pub struct Transcript {
    state: Hasher,
    counter: u64,
}

impl Transcript {
    /// A fresh transcript whose state starts at the domain string `domain`.
    pub fn new(domain: &[u8]) -> Self {
        let mut state = Hasher::new();
        state.update(&(domain.len() as u64).to_le_bytes());
        state.update(domain);
        Transcript { state, counter: 0 }
    }

    /// Absorb raw bytes, length-prefixed so that concatenations cannot collide.
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.state.update(&(bytes.len() as u64).to_le_bytes());
        self.state.update(bytes);
    }

    /// Absorb an integer (little-endian).
    pub fn absorb_u64(&mut self, x: u64) {
        self.state.update(&x.to_le_bytes());
    }

    /// Absorb ring elements as their raw little-endian `i16` slots, limb after limb: 324 bytes
    /// per limb and element (648 for the default two-limb key).
    pub fn absorb_elements(&mut self, elements: &[PowerOfThreeRingElementWithLimbs]) {
        self.state.update(&(elements.len() as u64).to_le_bytes());
        let mut buf = vec![0u8; 2 * N162 * elements.first().map_or(0, |e| e.len())];
        for e in elements {
            for (k, limb) in e.limbs.iter().enumerate() {
                for (s, &x) in limb.v.iter().enumerate() {
                    buf[2 * (k * N162 + s)..2 * (k * N162 + s) + 2]
                        .copy_from_slice(&x.to_le_bytes());
                }
            }
            self.state.update(&buf);
        }
    }

    /// Fill `out` with the XOF output of the current state under `label`, then advance the sample
    /// counter, so the next derivation is independent.
    pub fn fill(&mut self, label: &[u8], out: &mut [u8]) {
        self.reader(label).fill(out);
    }

    /// The same derivation as [`fill`](Self::fill), as a stream — for samplers that consume a
    /// number of bytes they do not know in advance.
    fn reader(&mut self, label: &[u8]) -> blake3::OutputReader {
        let mut state = self.state.clone();
        state.update(&self.counter.to_le_bytes());
        state.update(&(label.len() as u64).to_le_bytes());
        state.update(label);
        self.counter += 1;
        state.finalize_xof()
    }
}

/// A buffered view of one XOF derivation.
struct Xof {
    reader: blake3::OutputReader,
    buf: [u8; 256],
    pos: usize,
}

impl Xof {
    fn new(reader: blake3::OutputReader) -> Self {
        Xof {
            reader,
            buf: [0u8; 256],
            pos: 256,
        }
    }
    #[inline]
    fn byte(&mut self) -> u8 {
        if self.pos == self.buf.len() {
            self.reader.fill(&mut self.buf);
            self.pos = 0;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }
    #[inline]
    fn u16(&mut self) -> u16 {
        if self.pos + 2 <= self.buf.len() {
            let x = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
            self.pos += 2;
            return x;
        }
        u16::from_le_bytes([self.byte(), self.byte()])
    }
    #[inline]
    fn below(&mut self, n: u16) -> u16 {
        let (limit, magic) = DIVIDE[n as usize];
        loop {
            let r = self.u16() as u32;
            if r < limit {
                let q = ((r as u64 * magic) >> 32) as u32;
                return (r - q * n as u32) as u16;
            }
        }
    }
}

const DIVIDE: [(u32, u64); N162 + 1] = {
    let mut t = [(0u32, 0u64); N162 + 1];
    let mut n = 1;
    while n <= N162 {
        t[n] = (
            ((u16::MAX as u32 + 1) / n as u32) * n as u32,
            (1u64 << 32) / n as u64 + 1,
        );
        n += 1;
    }
    t
};

// =============================================================================================
// the challenge
// =============================================================================================

/// A weight-`w` signed element of `R_162`: coefficient `positions[i]` is `-1` when bit `i` of
/// `signs` is set and `+1` otherwise, every other coefficient zero. Positions are sorted and
/// distinct; bits and entries beyond `weight` are unused and must be zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortChallenge {
    pub positions: [u8; MAX_WEIGHT],
    pub signs: u32,
    pub weight: usize,
}

static SIGN_KEY: LazyLock<Hasher> =
    LazyLock::new(|| Hasher::new_derive_key("bin-ntt 2026 challenge signs v1"));

impl ShortChallenge {
    /// The zero element (weight 0).
    pub fn zero() -> Self {
        ShortChallenge {
            positions: [0u8; MAX_WEIGHT],
            signs: 0,
            weight: 0,
        }
    }

    /// The dense coefficient vector: coefficient of `Z^p` at index `p`.
    pub fn coeffs(&self) -> [i8; N162] {
        let mut c = [0i8; N162];
        for i in 0..self.weight {
            c[self.positions[i] as usize] = 1 - 2 * ((self.signs >> i) & 1) as i8;
        }
        c
    }

    pub fn signed(&self) -> Self {
        let mut h = SIGN_KEY.clone();
        h.update(&(self.weight as u64).to_le_bytes());
        h.update(&self.positions[..self.weight]);
        let bits = u32::from_le_bytes(h.finalize().as_bytes()[..4].try_into().unwrap());
        let mask = ((1u64 << self.weight) - 1) as u32;
        ShortChallenge {
            signs: bits & mask,
            ..*self
        }
    }

    /// The challenge modulo 2, as an element of `F162 = GF(2)[x]/(x^162 + x^81 + 1)`: a bit at
    /// each of its positions. `R_162 mod 2` *is* that field under the crate's plain lift, and
    /// `+-1` reduce alike, so the signs do not reach it.
    pub fn to_f162(&self) -> F162 {
        let mut x = F162::ZERO;
        for i in 0..self.weight {
            let p = self.positions[i] as usize;
            x.0[p >> 6] |= 1u64 << (p & 63);
        }
        x
    }

    /// The sparse form of a signed coefficient vector. Panics unless every entry is in
    /// `{-1, 0, 1}` and at most [`MAX_WEIGHT`] of them are nonzero.
    pub fn from_coeffs(c: &[i8; N162]) -> Self {
        let mut out = Self::zero();
        for (p, &x) in c.iter().enumerate() {
            assert!(
                x == 0 || x == 1 || x == -1,
                "coefficient {p} is not signed binary"
            );
            if x != 0 {
                assert!(out.weight < MAX_WEIGHT, "weight exceeds MAX_WEIGHT");
                out.positions[out.weight] = p as u8;
                out.signs |= ((x < 0) as u32) << out.weight;
                out.weight += 1;
            }
        }
        out
    }

    /// `log2` of the number of weight-`w` challenges: `log2 C(162, w)`.
    pub fn log2_cardinality(weight: usize) -> f64 {
        assert!(weight <= N162);
        let mut bits = 0.0f64;
        for i in 0..weight {
            bits += ((N162 - i) as f64 / (i + 1) as f64).log2();
        }
        bits
    }
}

// =============================================================================================
// the canonical embedding
// =============================================================================================

/// Half of the 162 primitive 243-rd roots: `c` has integer coefficients, so `c(zeta^{243-u})` is
/// the conjugate of `c(zeta^u)` and the two have the same modulus.
const HALF: usize = N162 / 2;

/// Lanes of the phase table: [`HALF`] roots padded to a multiple of 8. The pad lanes hold zeros,
/// so they never win a maximum.
const LANES: usize = 88;

const VECS: usize = 6;
const VECS2: usize = LANES / 8 - VECS;
const BLOCK: usize = 8 * VECS;

/// The 162 units `u` mod 243 (`gcd(u, 3) = 1`) in increasing order: the primitive 243-rd roots of
/// unity are `zeta^u`, `zeta = exp(2 pi i / 243)`. The first [`HALF`] of them are the ones the
/// phase table carries.
pub static UNITS: LazyLock<[u16; N162]> = LazyLock::new(|| {
    let mut u = [0u16; N162];
    let mut n = 0;
    for x in 1..CONDUCTOR243 {
        if x % 3 != 0 {
            u[n] = x as u16;
            n += 1;
        }
    }
    assert_eq!(n, N162);
    u
});

/// `PHASE[p * LANES + k] = exp(2 pi i p u_k / 243)`, real and imaginary parts, for `p = 0..162`
/// and the first [`HALF`] units: 162 x 88 f64 each, ~228 KB together.
static PHASE: LazyLock<(Vec<f64>, Vec<f64>)> = LazyLock::new(|| {
    let units = &*UNITS;
    let mut re = vec![0.0f64; N162 * LANES];
    let mut im = vec![0.0f64; N162 * LANES];
    for p in 0..N162 {
        for k in 0..HALF {
            let m = (p * units[k] as usize) % CONDUCTOR243;
            let angle = 2.0 * PI * (m as f64) / (CONDUCTOR243 as f64);
            re[p * LANES + k] = angle.cos();
            im[p * LANES + k] = angle.sin();
        }
    }
    (re, im)
});

#[inline(always)]
unsafe fn accumulate<const V: usize>(
    c: &ShortChallenge,
    off: usize,
) -> ([__m512d; V], [__m512d; V]) {
    let (re, im) = &*PHASE;
    let (mut vr, mut vi) = ([_mm512_setzero_pd(); V], [_mm512_setzero_pd(); V]);
    for i in 0..c.weight {
        let b = c.positions[i] as usize * LANES + off;
        let s = _mm512_castsi512_pd(_mm512_set1_epi64(
            ((c.signs >> i) & 1) as i64 * i64::MIN,
        ));
        for k in 0..V {
            let pr = _mm512_loadu_pd(re.as_ptr().add(b + 8 * k));
            let pi = _mm512_loadu_pd(im.as_ptr().add(b + 8 * k));
            vr[k] = _mm512_add_pd(vr[k], _mm512_xor_pd(s, pr));
            vi[k] = _mm512_add_pd(vi[k], _mm512_xor_pd(s, pi));
        }
    }
    (vr, vi)
}

#[inline(always)]
unsafe fn block_sq<const V: usize>(vr: [__m512d; V], vi: [__m512d; V]) -> [__m512d; V] {
    core::array::from_fn(|k| {
        _mm512_add_pd(_mm512_mul_pd(vr[k], vr[k]), _mm512_mul_pd(vi[k], vi[k]))
    })
}

#[target_feature(enable = "avx512f")]
unsafe fn norm_sq_blocks(c: &ShortChallenge) -> f64 {
    let (vr, vi) = accumulate::<VECS>(c, 0);
    let mut m = _mm512_setzero_pd();
    for x in block_sq::<VECS>(vr, vi) {
        m = _mm512_max_pd(m, x);
    }
    let (wr, wi) = accumulate::<VECS2>(c, BLOCK);
    for x in block_sq::<VECS2>(wr, wi) {
        m = _mm512_max_pd(m, x);
    }
    _mm512_reduce_max_pd(m)
}

#[target_feature(enable = "avx512f")]
unsafe fn within_blocks(c: &ShortChallenge, bound_sq: f64) -> bool {
    let b = _mm512_set1_pd(bound_sq);
    let (vr, vi) = accumulate::<VECS>(c, 0);
    let mut over = 0u8;
    for x in block_sq::<VECS>(vr, vi) {
        over |= _mm512_cmp_pd_mask::<_CMP_GT_OQ>(x, b);
    }
    if over != 0 {
        return false;
    }
    let (wr, wi) = accumulate::<VECS2>(c, BLOCK);
    for x in block_sq::<VECS2>(wr, wi) {
        over |= _mm512_cmp_pd_mask::<_CMP_GT_OQ>(x, b);
    }
    over == 0
}

/// `max_u |c(zeta^u)|^2` over the 162 primitive 243-rd roots of unity — the squared sup norm of
/// the canonical embedding of `c`, i.e. the squared operator norm of multiplication by `c` on
/// `R_162 (x) C` *in that embedding*. On coefficient vectors the expansion factor is `sqrt(3)`
/// times its square root, the power basis of a power-of-three conductor not being orthogonal
/// under the canonical embedding.
///
/// Evaluated from the `w` nonzero terms only, at half the roots (conjugates give nothing new): the
/// accumulator is one f64 vector of real and one of imaginary parts, and each term adds one row of
/// the phase table to it — contiguous f64 loops, no gathers.
pub fn canonical_inf_norm_sq(c: &ShortChallenge) -> f64 {
    unsafe { norm_sq_blocks(c) }
}

/// `canonical_inf_norm_sq(c) <= bound_sq`, one block of roots at a time so that a rejection stops
/// as soon as some root exceeds the bound. This is what the rejection loop calls: at the default
/// bound the great majority of attempts die in the first block.
fn within(c: &ShortChallenge, bound_sq: f64) -> bool {
    unsafe { within_blocks(c, bound_sq) }
}

/// The same quantity by Horner's rule in complex f64 over all 162 roots and all 162 coefficients —
/// the independent reference [`canonical_inf_norm_sq`] is tested against.
pub fn canonical_inf_norm_sq_naive(c: &ShortChallenge) -> f64 {
    let coeffs = c.coeffs();
    let mut best = 0.0f64;
    for &u in UNITS.iter() {
        let angle = 2.0 * PI * (u as f64) / (CONDUCTOR243 as f64);
        let (zr, zi) = (angle.cos(), angle.sin());
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        for p in (0..N162).rev() {
            let (nr, ni) = (ar * zr - ai * zi, ar * zi + ai * zr);
            ar = nr + coeffs[p] as f64;
            ai = ni;
        }
        let m = ar * ar + ai * ai;
        if m > best {
            best = m;
        }
    }
    best
}

// =============================================================================================
// sampling
// =============================================================================================

/// One uniform weight-`w` position set, unsigned: a partial Fisher-Yates over the 162 positions
/// driven by the transcript's XOF, then the positions sorted.
pub fn sample_attempt(t: &mut Transcript, weight: usize) -> ShortChallenge {
    let mut perm = [0u8; N162];
    attempt(
        &mut Xof::new(t.reader(b"short-challenge")),
        weight,
        &mut perm,
    )
}

fn attempt(x: &mut Xof, weight: usize, perm: &mut [u8; N162]) -> ShortChallenge {
    assert!(
        weight <= MAX_WEIGHT && weight <= N162,
        "weight {weight} exceeds MAX_WEIGHT = {MAX_WEIGHT}"
    );
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i as u8;
    }
    let mut out = ShortChallenge::zero();
    out.weight = weight;
    for i in 0..weight {
        let j = i + x.below((N162 - i) as u16) as usize;
        perm.swap(i, j);
    }
    let out_positions = &perm[..weight];
    let mut seen = [0u64; N162.div_ceil(64)];
    for &p in out_positions {
        seen[p as usize >> 6] |= 1u64 << (p & 63);
    }
    let mut n = 0;
    for (w, mut b) in seen.into_iter().enumerate() {
        while b != 0 {
            out.positions[n] = (64 * w + b.trailing_zeros() as usize) as u8;
            b &= b - 1;
            n += 1;
        }
    }
    out
}

/// Rejection-sample a weight-`w` challenge with `canonical_inf_norm_sq <= bound^2`, returning it
/// together with the number of attempts it took. Each candidate position set is signed by
/// [`ShortChallenge::signed`] before the bound is tested, so the bound holds of the challenge the
/// round actually uses.
///
/// All attempts read one XOF derivation of the transcript, so the whole loop is one deterministic
/// function of what has been absorbed and takes one transcript finalisation however many attempts
/// the bound needs; the per-attempt work is then the Fisher-Yates, the sign hash of the candidate,
/// and the blocked evaluation, with no allocation.
pub fn sample_short_challenge(
    t: &mut Transcript,
    weight: usize,
    bound: f64,
) -> (ShortChallenge, u64) {
    let bound_sq = bound * bound + 1e-12;
    let mut x = Xof::new(t.reader(b"short-challenge"));
    let mut perm = [0u8; N162];
    let mut attempts = 0u64;
    loop {
        attempts += 1;
        let c = attempt(&mut x, weight, &mut perm).signed();
        if within(&c, bound_sq) {
            return (c, attempts);
        }
    }
}

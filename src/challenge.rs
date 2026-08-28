//! Short challenges over the 3^5-th cyclotomic ring `R_162 = Z[Z]/Phi_243(Z)`,
//! `Phi_243(Z) = Z^162 + Z^81 + 1`, and the blake3 transcript that samples them.
//!
//! A challenge is a weight-`w` ternary element of `R_162`: `w` of the 162 coefficients are `+-1`
//! and the rest are zero, stored sparsely as sorted positions plus signs. Sampling is uniform over
//! that set (a partial Fisher-Yates driven by the transcript's XOF) and then rejected until the
//! challenge is *short in the canonical embedding*:
//!
//! ```text
//!     max_u |c(zeta^u)|^2 <= bound^2,     zeta = exp(2 pi i / 243), gcd(u, 3) = 1,
//! ```
//!
//! the 162 primitive 243-rd roots of unity. That quantity — [`canonical_inf_norm_sq`] — bounds the
//! operator norm of multiplication by `c` on `R_162 (x) C`, so a bound on it is what a security
//! argument needs from a challenge set. The default is `weight = 21`, `bound = 9`.
//!
//! ```no_run
//! use bin_ntt::challenge::{sample_short_challenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
//! # use bin_ntt::PowerOfThreeRingElementWithLimbs;
//! # let commitment = [PowerOfThreeRingElementWithLimbs::zero(2)];
//! let mut t = Transcript::new(b"bin-ntt/example");
//! t.absorb_elements(&commitment);
//! let (c, attempts) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
//! ```
use crate::api::{PowerOfThreeRingElementWithLimbs, N162};
use bin_fields::scalar::F162;
use blake3::Hasher;
use std::f64::consts::PI;
use std::sync::LazyLock;

/// Conductor of the small ring: the challenge is evaluated at primitive 243-rd roots of unity.
pub const CONDUCTOR243: usize = 243;

/// Largest weight a [`ShortChallenge`] can hold.
pub const MAX_WEIGHT: usize = 32;

/// The weight the crate samples at unless told otherwise.
pub const DEFAULT_WEIGHT: usize = 21;

/// The default canonical-embedding bound: `max_u |c(zeta^u)|^2 <= 9^2 = 81`.
pub const DEFAULT_BOUND: f64 = 9.0;

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
        u16::from_le_bytes([self.byte(), self.byte()])
    }
    /// Uniform in `[0, n)` for `n <= 2^16`, by rejection on 16-bit draws.
    #[inline]
    fn below(&mut self, n: u16) -> u16 {
        let limit = (u16::MAX as u32 + 1) / n as u32 * n as u32;
        loop {
            let r = self.u16() as u32;
            if r < limit {
                return (r % n as u32) as u16;
            }
        }
    }
}

// =============================================================================================
// the challenge
// =============================================================================================

/// A weight-`w` ternary element of `R_162`: coefficient `positions[i]` is `signs[i]`, all other
/// coefficients zero. Positions are sorted and distinct; entries beyond `weight` are unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortChallenge {
    pub positions: [u8; MAX_WEIGHT],
    pub signs: [i8; MAX_WEIGHT],
    pub weight: usize,
}

impl ShortChallenge {
    /// The zero element (weight 0).
    pub fn zero() -> Self {
        ShortChallenge {
            positions: [0u8; MAX_WEIGHT],
            signs: [0i8; MAX_WEIGHT],
            weight: 0,
        }
    }

    /// The dense coefficient vector: coefficient of `Z^p` at index `p`.
    pub fn coeffs(&self) -> [i8; N162] {
        let mut c = [0i8; N162];
        for i in 0..self.weight {
            c[self.positions[i] as usize] = self.signs[i];
        }
        c
    }

    /// The challenge modulo 2, as an element of `F162 = GF(2)[x]/(x^162 + x^81 + 1)`: a bit at
    /// each of its positions. `R_162 mod 2` *is* that field under the crate's plain lift, and the
    /// signs vanish there.
    pub fn to_f162(&self) -> F162 {
        let mut x = F162::ZERO;
        for i in 0..self.weight {
            let p = self.positions[i] as usize;
            x.0[p >> 6] |= 1u64 << (p & 63);
        }
        x
    }

    /// The sparse form of a ternary coefficient vector. Panics unless every entry is in
    /// `{-1, 0, 1}` and at most [`MAX_WEIGHT`] of them are nonzero.
    pub fn from_coeffs(c: &[i8; N162]) -> Self {
        let mut out = Self::zero();
        for (p, &x) in c.iter().enumerate() {
            assert!(
                x == -1 || x == 0 || x == 1,
                "coefficient {p} is not ternary"
            );
            if x != 0 {
                assert!(out.weight < MAX_WEIGHT, "weight exceeds MAX_WEIGHT");
                out.positions[out.weight] = p as u8;
                out.signs[out.weight] = x;
                out.weight += 1;
            }
        }
        out
    }

    /// `log2` of the number of weight-`w` challenges: `log2 C(162, w) + w`.
    pub fn log2_cardinality(weight: usize) -> f64 {
        assert!(weight <= N162);
        let mut bits = weight as f64;
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

/// Lanes of one block of the blocked evaluation, tuned so that a rejected challenge usually stops
/// after the first block.
const BLOCK: usize = LANES / 2;

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

/// Accumulate `sum_i sign_i exp(2 pi i p_i u_k / 243)` over the challenge's nonzero terms, for the
/// [`BLOCK`] lanes starting at `off`.
#[inline]
fn accumulate(c: &ShortChallenge, off: usize, ar: &mut [f64; BLOCK], ai: &mut [f64; BLOCK]) {
    let (re, im) = &*PHASE;
    ar.fill(0.0);
    ai.fill(0.0);
    for i in 0..c.weight {
        let base = c.positions[i] as usize * LANES + off;
        let pr = &re[base..base + BLOCK];
        let pi = &im[base..base + BLOCK];
        if c.signs[i] > 0 {
            for k in 0..BLOCK {
                ar[k] += pr[k];
                ai[k] += pi[k];
            }
        } else {
            for k in 0..BLOCK {
                ar[k] -= pr[k];
                ai[k] -= pi[k];
            }
        }
    }
}

/// `max_u |c(zeta^u)|^2` over the 162 primitive 243-rd roots of unity — the squared sup norm of
/// the canonical embedding of `c`, i.e. the squared operator norm of multiplication by `c` on
/// `R_162 (x) C`.
///
/// Evaluated from the `w` nonzero terms only, at half the roots (conjugates give nothing new): the
/// accumulator is one f64 vector of real and one of imaginary parts, and each term adds `+-` one
/// row of the phase table to it — contiguous f64 loops, no gathers.
pub fn canonical_inf_norm_sq(c: &ShortChallenge) -> f64 {
    let mut ar = [0.0f64; BLOCK];
    let mut ai = [0.0f64; BLOCK];
    let mut best = 0.0f64;
    for off in (0..LANES).step_by(BLOCK) {
        accumulate(c, off, &mut ar, &mut ai);
        for k in 0..BLOCK {
            let m = ar[k] * ar[k] + ai[k] * ai[k];
            if m > best {
                best = m;
            }
        }
    }
    best
}

/// `canonical_inf_norm_sq(c) <= bound_sq`, one block of roots at a time so that a rejection stops
/// as soon as some root exceeds the bound. This is what the rejection loop calls: at the default
/// bound the great majority of attempts die in the first block.
fn within(c: &ShortChallenge, bound_sq: f64) -> bool {
    let mut ar = [0.0f64; BLOCK];
    let mut ai = [0.0f64; BLOCK];
    for off in (0..LANES).step_by(BLOCK) {
        accumulate(c, off, &mut ar, &mut ai);
        for k in 0..BLOCK {
            if ar[k] * ar[k] + ai[k] * ai[k] > bound_sq {
                return false;
            }
        }
    }
    true
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

/// One uniform weight-`w` ternary element: a partial Fisher-Yates over the 162 positions driven by
/// the transcript's XOF, uniform signs, then the positions sorted.
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
        out.positions[i] = perm[i];
        out.signs[i] = if x.byte() & 1 == 0 { 1 } else { -1 };
    }
    for i in 1..weight {
        let mut j = i;
        while j > 0 && out.positions[j - 1] > out.positions[j] {
            out.positions.swap(j - 1, j);
            out.signs.swap(j - 1, j);
            j -= 1;
        }
    }
    out
}

/// Rejection-sample a weight-`w` challenge with `canonical_inf_norm_sq <= bound^2`, returning it
/// together with the number of attempts it took.
///
/// All attempts read one XOF derivation of the transcript, so the whole loop is one deterministic
/// function of what has been absorbed and costs one blake3 finalisation however many attempts the
/// bound needs; the per-attempt work is then the Fisher-Yates and the blocked evaluation, with no
/// allocation.
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
        let c = attempt(&mut x, weight, &mut perm);
        if within(&c, bound_sq) {
            return (c, attempts);
        }
    }
}

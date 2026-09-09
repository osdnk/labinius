//! The Brakedown code of Golovnev, Lee, Setty, Thaler and Wahby, "Brakedown: Linear-time and
//! field-agnostic SNARKs for R1CS" (CRYPTO 2023, ePrint 2021/1043), over `B128`.
//!
//! # The recursion
//!
//! One level turns a message `x` of `k` symbols into `ceil(r k)` symbols with two sparse
//! multiplications and one recursive call:
//!
//! ```text
//!     y = x A,      A: k x m sparse,  m = ceil(alpha k),   c nonzeros per row,
//!     z = Enc(y),   the same code one level down,          |z| = ceil(r m),
//!     v = z B,      B: |z| x t sparse, t = ceil(r k) - k - |z|,  d nonzeros per row,
//!     Enc(x) = x || z || v.
//! ```
//!
//! The block length telescopes exactly: a level of message length `k` always emits `ceil(r k)`
//! symbols, because `t` is *defined* as the slack, so the whole code has rate `1/r` up to one
//! rounding and is systematic by construction.
//!
//! The recursion runs while the message is longer than [`N_0`]; below that the level's `y` is
//! encoded by a Reed–Solomon code of the same rate, evaluated at the `ceil(r m)` distinct field
//! elements `B128::new(0), .., B128::new(ceil(r m) - 1)` by Horner. `m <= N_0` there, so the base
//! case is `O(N_0^2)` field multiplications for the whole codeword and does not show up in any
//! measurement.
//!
//! Nothing about this is characteristic-dependent: the distance proof of GLSTW21 §4 only asks
//! that the nonzero entries of `A` and `B` be uniform in the field, and over `B128` they are
//! uniform `GF(2^128)`.
//!
//! # The flat layout
//!
//! [`BrakedownCode::encode_in_place`] never allocates and never moves a symbol twice. With levels
//! `0..L` of message lengths `k = n_0 > n_1 > .. > n_{L-1} > N_0` (`n_{i+1} = m_i`), the codeword
//! is laid out as
//!
//! ```text
//!     x | y_0 | y_1 | .. | y_{L-2} | RS(y_{L-1}) | v_{L-1} | v_{L-2} | .. | v_0,
//! ```
//!
//! so the forward pass walks left to right writing each `y_i` immediately after its own input,
//! and the backward pass walks the same buffer right to left as `B_i`'s input, appending `v_i`.
//! The sub-codeword `z_i` that `B_i` reads is the contiguous run `y_i | .. | v_{i+1}`, which is
//! exactly `ceil(r n_{i+1})` long by the telescoping above.
//!
//! # The parameters
//!
//! [`SPEC`] is Figure 2 of GLSTW21 verbatim, at `lambda = 128`:
//!
//! ```text
//!     spec   alpha    beta      r      delta = beta/r
//!       1   0.1195   0.0284   1.420        0.02
//!       2   0.1380   0.0444   1.470        0.03
//!       3   0.1780   0.0610   1.521        0.04
//!       4   0.2000   0.0820   1.640        0.05
//!       5   0.2110   0.0970   1.616        0.06
//!       6   0.2380   0.1205   1.720        0.07
//! ```
//!
//! `relative_distance` is `beta / r`, the provable bound of GLSTW21 Theorem 1. The row sparsities
//! `c` and `d` are the paper's [`Spec::c`] and [`Spec::d`], which depend on the level's message
//! length and (for `d`) on `log2 q = 128`; asymptotically spec 1 gives `c = 6`, `d = 33`.
//!
//! # What it costs
//!
//! One field multiplication per nonzero matrix entry, plus the base case: `sum_i n_i c_i` for the
//! `A`s and `sum_i |z_i| d_i` for the `B`s, a geometric series in `alpha` that converges to about
//! `13 k` from above for spec 1. Measured on the i7-11850H, one core, `BENCH_CPU`-pinned, median
//! of 9 after a warm-up, spec 1:
//!
//! ```text
//!     k       encode    muls/symbol   ns/symbol
//!     2^12    0.144 ms     15.4          35
//!     2^14    0.542 ms     14.8          33
//!     2^16    2.365 ms     14.6          36
//!     2^20   56.301 ms     13.3          54
//! ```
//!
//! The multiplication count falls with `k` and the time per symbol rises, because the encoding is
//! a *scatter*: every nonzero is a read-modify-write at a random column of its level's output.
//! At `k = 2^20` that is 14 M scattered accesses in 56 ms, 4 ns each, and the GHASH multiply
//! (`vpclmulqdq` plus the reduction) is not what the core is waiting on. The setup samples every
//! matrix once — 280 MB of `(u32, B128)` cells at `k = 2^20` — and is not in that column; a
//! tensor commitment encodes every row of the witness matrix with the same [`BrakedownCode`].
use crate::codes::LinearCode;
use bin_ntt::rng::Rng;
use binius_field::Field;
use binius_verifier::config::B128;

/// The recursion stops once a level's message is this short; the level's `y` then goes to
/// Reed–Solomon instead of one more sparse level.
pub const N_0: usize = 20;

const LOG2_Q: f64 = 128.0;

#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub name: &'static str,
    pub alpha: f64,
    pub beta: f64,
    pub r: f64,
}

/// Figure 2 of GLSTW21, `lambda = 128`.
pub const SPEC: [Spec; 6] = [
    Spec { name: "brakedown-1", alpha: 0.1195, beta: 0.0284, r: 1.420 },
    Spec { name: "brakedown-2", alpha: 0.1380, beta: 0.0444, r: 1.470 },
    Spec { name: "brakedown-3", alpha: 0.1780, beta: 0.0610, r: 1.521 },
    Spec { name: "brakedown-4", alpha: 0.2000, beta: 0.0820, r: 1.640 },
    Spec { name: "brakedown-5", alpha: 0.2110, beta: 0.0970, r: 1.616 },
    Spec { name: "brakedown-6", alpha: 0.2380, beta: 0.1205, r: 1.720 },
];

impl Spec {
    /// `beta / r`, the provable relative distance.
    pub fn relative_distance(&self) -> f64 {
        self.beta / self.r
    }

    fn mu(&self) -> f64 {
        self.r - 1.0 - self.r * self.alpha
    }

    fn nu(&self) -> f64 {
        self.beta + self.alpha * self.beta + 0.03
    }

    /// Nonzeros per row of the `A` of a level whose message is `n` long.
    pub fn c(&self, n: usize) -> usize {
        let (alpha, beta, n) = (self.alpha, self.beta, n as f64);
        let counting = ceil(1.28 * beta * n).max(ceil(beta * n) + 4);
        let entropy = ceil(
            (110.0 / n + h(beta) + alpha * h(1.28 * beta / alpha))
                / (beta * (alpha / (1.28 * beta)).log2()),
        );
        counting.min(entropy)
    }

    /// Nonzeros per row of the `B` of a level whose message is `n` long.
    pub fn d(&self, n: usize) -> usize {
        let (alpha, beta, r, n) = (self.alpha, self.beta, self.r, n as f64);
        let (mu, nu) = (self.mu(), self.nu());
        let counting = ceil((2.0 * beta + (r - 1.0 + 110.0 / n) / LOG2_Q) * n);
        let entropy = ceil(
            (r * alpha * h(beta / r) + mu * h(nu / mu) + 110.0 / n)
                / (alpha * beta * (mu / nu).log2()),
        );
        counting.min(entropy)
    }
}

struct Sparse {
    n: usize,
    m: usize,
    d: usize,
    cols: Vec<u32>,
    vals: Vec<B128>,
}

impl Sparse {
    fn sample(n: usize, m: usize, d: usize, pool: &mut Vec<u32>, rng: &mut Rng) -> Sparse {
        assert!(0 < d && d <= m && m <= u32::MAX as usize);
        pool.clear();
        pool.extend(0..m as u32);
        let mut cols = Vec::with_capacity(n * d);
        let mut vals = Vec::with_capacity(n * d);
        for _ in 0..n {
            for j in 0..d {
                pool.swap(j, j + rng.below((m - j) as u32) as usize);
                cols.push(pool[j]);
                vals.push(uniform(rng));
            }
        }
        Sparse { n, m, d, cols, vals }
    }

    fn dot_into(&self, x: &[B128], y: &mut [B128]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.m);
        y.fill(B128::ZERO);
        for (i, &xi) in x.iter().enumerate() {
            let row = i * self.d;
            for j in row..row + self.d {
                y[self.cols[j] as usize] += xi * self.vals[j];
            }
        }
    }

    fn nonzeros(&self) -> usize {
        self.n * self.d
    }
}

pub struct BrakedownCode {
    spec: Spec,
    k: usize,
    n: usize,
    a: Vec<Sparse>,
    b: Vec<Sparse>,
    muls: usize,
}

impl BrakedownCode {
    pub fn new(k: usize, spec: Spec, seed: u64) -> BrakedownCode {
        assert!(k > N_0);
        let mut rng = Rng::new(seed);
        let mut pool = Vec::new();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut len = k;
        while len > N_0 {
            let m = ceil(len as f64 * spec.alpha);
            let z = ceil(m as f64 * spec.r);
            let t = ceil(len as f64 * spec.r) - len - z;
            assert!(t > 0);
            a.push(Sparse::sample(len, m, spec.c(len).min(m), &mut pool, &mut rng));
            b.push(Sparse::sample(z, t, spec.d(len).min(t), &mut pool, &mut rng));
            len = m;
        }
        let base = a.last().unwrap().m;
        let n = k
            + a[..a.len() - 1].iter().map(|a| a.m).sum::<usize>()
            + b.last().unwrap().n
            + b.iter().map(|b| b.m).sum::<usize>();
        let muls = a.iter().chain(b.iter()).map(Sparse::nonzeros).sum::<usize>()
            + b.last().unwrap().n * (base - 1);
        BrakedownCode { spec, k, n, a, b, muls }
    }

    /// Field multiplications one [`LinearCode::encode`] performs, counted from the sampled
    /// matrices rather than estimated.
    pub fn field_muls(&self) -> usize {
        self.muls
    }

    /// The whole encoding, assuming `codeword[..k]` already holds the message. This is what
    /// [`crate::codes::lightning_code::LightningCode`] calls after writing its sketch straight
    /// into the systematic prefix of the base codeword, so that no row of a commitment ever
    /// allocates.
    pub fn encode_in_place(&self, codeword: &mut [B128]) {
        assert_eq!(codeword.len(), self.n);
        let last = self.a.len() - 1;
        let mut input = 0;
        for a in &self.a[..last] {
            let (x, y) = codeword[input..].split_at_mut(a.n);
            a.dot_into(x, &mut y[..a.m]);
            input += a.n;
        }

        let (a, b) = (&self.a[last], &self.b[last]);
        let mut base = [B128::ZERO; N_0];
        let (x, z) = codeword[input..].split_at_mut(a.n);
        a.dot_into(x, &mut base[..a.m]);
        reed_solomon_into(&base[..a.m], &mut z[..b.n]);

        let mut output = input + a.n + b.n;
        input += a.n + a.m;
        for (a, b) in self.a.iter().rev().zip(self.b.iter().rev()) {
            input -= a.m;
            let (z, v) = codeword.split_at_mut(output);
            b.dot_into(&z[input..input + b.n], &mut v[..b.m]);
            output += b.m;
        }
        debug_assert_eq!(output, self.n);
    }
}

impl LinearCode for BrakedownCode {
    fn name(&self) -> &'static str {
        self.spec.name
    }

    fn message_len(&self) -> usize {
        self.k
    }

    fn codeword_len(&self) -> usize {
        self.n
    }

    fn relative_distance(&self) -> f64 {
        self.spec.relative_distance()
    }

    fn encode(&self, message: &[B128], codeword: &mut [B128]) {
        assert_eq!(message.len(), self.k);
        codeword[..self.k].copy_from_slice(message);
        self.encode_in_place(codeword);
    }
}

fn reed_solomon_into(message: &[B128], codeword: &mut [B128]) {
    for (j, out) in codeword.iter_mut().enumerate() {
        let x = B128::new(j as u128);
        let mut acc = *message.last().unwrap();
        for &c in message[..message.len() - 1].iter().rev() {
            acc = acc * x + c;
        }
        *out = acc;
    }
}

fn uniform(rng: &mut Rng) -> B128 {
    B128::new(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
}

fn h(p: f64) -> f64 {
    assert!(0.0 < p && p < 1.0);
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

fn ceil(v: f64) -> usize {
    v.ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::test_util::{low_weight, random, weight};

    #[test]
    fn spec_sparsities_match_the_paper() {
        let n = 1 << 30;
        for (spec, (c, d, delta)) in SPEC.iter().zip([
            (6, 33, 0.02),
            (7, 26, 0.03),
            (7, 22, 0.04),
            (8, 19, 0.05),
            (9, 21, 0.06),
            (10, 20, 0.07),
        ]) {
            assert_eq!(spec.c(n), c);
            assert_eq!(spec.d(n), d);
            assert!((spec.relative_distance() - delta).abs() < 1e-3);
        }
    }

    #[test]
    fn block_length_is_the_rate() {
        for spec in SPEC {
            for k in [64usize, 1000, 4096] {
                let code = BrakedownCode::new(k, spec, 7);
                assert_eq!(code.codeword_len(), ceil(k as f64 * spec.r));
            }
        }
    }

    #[test]
    fn systematic_and_linear() {
        for spec in [SPEC[0], SPEC[5]] {
            let code = BrakedownCode::new(1000, spec, 11);
            let (k, n) = (code.message_len(), code.codeword_len());
            let (x, y) = (random(k, 1), random(k, 2));
            let sum: Vec<B128> = x.iter().zip(&y).map(|(a, b)| *a + *b).collect();
            let (mut cx, mut cy, mut cs) = (vec![B128::ZERO; n], vec![B128::ZERO; n], vec![B128::ZERO; n]);
            code.encode(&x, &mut cx);
            code.encode(&y, &mut cy);
            code.encode(&sum, &mut cs);
            assert_eq!(&cx[..k], &x[..]);
            for i in 0..n {
                assert_eq!(cx[i] + cy[i], cs[i]);
            }
        }
    }

    #[test]
    fn low_weight_messages_stay_above_the_distance() {
        let code = BrakedownCode::new(4096, SPEC[0], 13);
        let (k, n) = (code.message_len(), code.codeword_len());
        let d = (code.relative_distance() * n as f64).floor() as usize;
        let mut codeword = vec![B128::ZERO; n];
        for seed in 0..64 {
            for support in [1usize, 2, 3, 8, 40] {
                code.encode(&low_weight(k, support, seed), &mut codeword);
                assert!(weight(&codeword) >= d);
            }
        }
    }

    #[test]
    #[ignore]
    fn throughput() {
        bin_ntt_bench::pin(crate::CPU);
        for log_k in [12, 14, 16, 20] {
            let k = 1 << log_k;
            let code = BrakedownCode::new(k, SPEC[0], 3);
            let message = random(k, 5);
            let mut codeword = vec![B128::ZERO; code.codeword_len()];
            for _ in 0..(1 << 22) / k {
                code.encode(&message, &mut codeword);
            }
            let (ms, _) = bin_ntt_bench::median_of(9, || code.encode(&message, &mut codeword));
            println!(
                "brakedown-1 k = 2^{log_k}  {ms:>7.3} ms  {:.1} muls/symbol  {:.0} ns/symbol",
                code.field_muls() as f64 / k as f64,
                ms * 1e6 / k as f64,
            );
        }
    }
}

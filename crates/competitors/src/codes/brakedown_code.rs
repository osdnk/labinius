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
//! [`BrakedownCode::encode_in_place_interleaved`] is the same walk with every offset scaled by
//! `rows`, so a symbol of the codeword becomes `rows` contiguous ones and nothing about the
//! layout changes. It allocates nothing up to `rows = 16`; above that the base case's `y`, the
//! one run of at most `N_0` symbols the layout has no room for, goes on the heap.
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
//! # The two kernels
//!
//! Every level is one sparse multiplication `y = x A`, and there are two ways to walk it.
//!
//! `Sparse::scatter` is row-major: `x` streams, and each of the `d` nonzeros of row `i` is a
//! read-modify-write of `y` at a random column. `Sparse::sweep` is column-major: one output
//! column at a time, gathering its `x_i` at random and writing `y` sequentially. The sweep is the
//! cheaper *arithmetic*, because a column's whole dot product accumulates unreduced — GHASH
//! multiplication is four `vpclmulqdq` for the 256-bit product and two more for the reduction, so
//! reducing once per output symbol instead of once per nonzero drops the multiply from six
//! `vpclmulqdq` to four. On this core `vpclmulqdq zmm` is 2 port-5 uops and the two `vpslldq` of
//! the reduction are one each, so a nonzero costs 14 port-5 uops scattered and 8 swept, and port 5
//! is the only port either kernel contends for.
//!
//! Which one wins is a property of the level, not of the caller. Measured at `rows = 4`, spec 1,
//! forcing each kernel at every level (nanoseconds per symbol):
//!
//! ```text
//!     k         2^12   2^13   2^14   2^15   2^16   2^17
//!     scatter   12.4   12.6   12.4   12.5   13.2   15.0
//!     sweep      8.1    8.7    9.3   11.2   14.8   18.2
//! ```
//!
//! The sweep's gathers leave L2 as `n` grows and it loses; the crossing is at `n = 2^15` and moves
//! with `rows` only through the `rows * 16` bytes each gather pulls, which is why the rule is
//! `n * max(rows, 8) <= GATHER_LIMIT` rather than a plain footprint: at `k = 2^14, rows = 16`
//! the sweep wins 10.4 to 12.3 on the same 4 MB of input on which it loses at `k = 2^16,
//! rows = 4`.
//! The second arm is `n <= m`: a `B` level of spec 1 has `t = 0.250 k` outputs against `z = 0.170
//! k` inputs, so the sweep's random side is the *smaller* one and it wins however large the level
//! is. That arm alone is worth 22.3 -> 18.6 ns/symbol at `k = 2^20, rows = 4`; the levels it
//! catches are `B_0` and `B_1`, and forcing the `A` levels along with them gives 40.9.
//!
//! Both kernels prefetch their next-but-eight random line, which is worth 26.4 -> 22.9 ns/symbol
//! at `k = 2^20` (distance 4 gives 23.1, 16 gives 23.5) and nothing at `k = 2^16`. The sweep only
//! prefetches on the `n <= m` arm: on the resident arm the prefetch costs 8.2 -> 8.8 at
//! `k = 2^12`.
//!
//! # What it costs
//!
//! One field multiplication per nonzero matrix entry, plus the base case: `sum_i n_i c_i` for the
//! `A`s and `sum_i |z_i| d_i` for the `B`s, a geometric series in `alpha` that converges to about
//! `13 k` from above for spec 1. Measured on the i7-11850H, one core, `taskset -c 5`, median of 5
//! after a warm-up, spec 1, in nanoseconds per encoded symbol — that is, wall time divided by
//! `k * rows`, so the columns are directly comparable:
//!
//! ```text
//!                     this file                    the default of [`crate::codes`]
//!     rows =        1      4      8     16          1      4      8     16
//!     k = 2^12    15.3    8.6    8.4    8.8       25.5   25.6   27.2   27.6
//!     k = 2^16    21.9   12.2   12.3   12.2       26.6   31.0   40.0   47.9
//!     k = 2^20    34.4   18.2   24.0   35.7       49.0   52.0   60.1   63.2
//! ```
//!
//! The right-hand block is the default [`LinearCode::encode_interleaved`], which is what this
//! replaces: one encode per row, plus a gather and a scatter of the whole matrix to get in and
//! out of the symbol-major layout, which at `rows = 16, k = 2^16` is half of its time. Against
//! `rows` independent encodes of the old kernel with no transposition at all — 22.0, 23.4 and
//! 37.8 ns/symbol at the three sizes, flat in `rows`, since nothing is shared — this is 2.6x,
//! 1.9x and 2.1x at `rows = 4`.
//!
//! Where the time goes, at `rows = 4`, counted per nonzero with `perf stat` over two rep counts
//! and differenced:
//!
//! ```text
//!     k       cycles   port-5 uops   L2 misses      bound
//!     2^12      9.8        8.7         0.002        port 5, 89% occupied
//!     2^20     26.0       12.5         0.65         L2/L3 latency, port 5 48% occupied
//! ```
//!
//! Before the interleaving the same counters read 15.2 cycles and 14.2 port-5 uops per nonzero at
//! `k = 2^12` — port 5 90% occupied on a multiply that was 40% reduction. At `k = 2^20` the
//! encoder is memory-bound in either form: one encode streams 283 MB of matrix (20 bytes per
//! nonzero, and the interleaving is what amortises that over `rows`), 64 MB of message at
//! `rows = 4`, and read-modify-writes an 8 MB `y` for `A_0` and a 17 MB one for `B_0`.
//!
//! Growing `rows` past 4 buys nothing — 4 lanes are one zmm and the index stream is already
//! amortised — and at `k = 2^20` it costs, because the level's output grows with it: 32 MB for
//! `A_0` at `rows = 16` is past L3 and the encoder falls back to 35.7 ns/symbol, no better than
//! `rows = 1`. **`rows = 4` is the operating point.**
//!
//! # What was tried and rejected
//!
//! *Splitting `rows` into groups of 4 or 8 lanes*, so that a `rows = 16` level touches a quarter
//! of its output per pass and stays L3-resident, on the theory that the passes cost only extra
//! sequential reads. They cost more than that — each pass re-reads the whole 20-byte-per-nonzero
//! matrix stream — and it loses everywhere: at `k = 2^20, rows = 16`, 63.3 ns/symbol in groups of
//! 4, 50.9 in groups of 8, 43.5 in one pass of 16; at `k = 2^16, rows = 16`, 21.3 / 16.0 / 13.5.
//!
//! *Bucket-partitioning a level's nonzeros by target block* so that each pass writes into an
//! L2-resident window is the same trade one step further, and the column sweep is its limiting
//! case with a block of one column: it is exactly the pass structure that keeps the output in
//! registers, and it measures 40.9 against 22.3 ns/symbol at `k = 2^20, rows = 4` because a
//! uniformly random `A` with `c = 6` nonzeros per row makes every block touch nearly every input
//! row. Blocking `A_0` into `P` blocks re-reads the 64 MB message `P` times; even `P = 24`, the
//! smallest that touches under a quarter of the rows per pass, is 350 MB of extra reads against
//! the 8 MB of L2 misses it would save.
//!
//! *Splitting the `(index, value)` cells into two arrays* is what `Sparse` already does, and
//! it should stay that way: the index stream is 4 bytes per nonzero and the value stream 16, and
//! the scatter's prefetch has to read the index 8 nonzeros early.
//!
//! # The setup
//!
//! [`BrakedownCode::new`] samples every matrix once and is not in the table above:
//!
//! ```text
//!     k       setup      matrix   per nonzero
//!     2^12    1.1 ms      2.0 MB    31.0 B
//!     2^16   13.5 ms     20.5 MB    21.4 B
//!     2^20  250.3 ms    283.4 MB    20.3 B
//! ```
//!
//! 250 ms at `k = 2^20` is eleven encodes at `rows = 4`, so a benchmark that builds the code once
//! and encodes a witness matrix of 4 or more rows has already paid for it; one that rebuilds the
//! code per encode would be measuring the sampler. A level keeps only the layouts its
//! `Sparse::dot` can select — 20 bytes per nonzero either way — so the two kernels cost nothing
//! in memory except at the small levels that could go either way, which is the 31 B/nonzero at
//! `k = 2^12` and 2 MB of it.
use crate::codes::LinearCode;
use bin_ntt::rng::Rng;
use binius_field::{Field, PackedGhash1x128b, PackedGhash4x128b, WideMul};
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

type Packed = PackedGhash4x128b;
type Wide = <Packed as WideMul>::Output;
type Lane = PackedGhash1x128b;
type WideLane = <Lane as WideMul>::Output;

const LANES: usize = Packed::WIDTH;

/// The widest interleaving the stack base-case buffer covers; above it a level allocates.
const MAX_ROWS: usize = 16;

/// A level whose input is at most this many `B128` per lane-group sweeps it by column; see the
/// module doc for the measurements this comes from.
const GATHER_LIMIT: usize = 1 << 18;

/// Nonzeros ahead of the one being multiplied that the scatter's target line is prefetched.
const PREFETCH: usize = 8;

#[inline(always)]
unsafe fn load(x: *const B128) -> Packed {
    (x as *const Packed).read_unaligned()
}

#[inline(always)]
unsafe fn store(y: *mut B128, p: Packed) {
    (y as *mut Packed).write_unaligned(p)
}

#[inline(always)]
unsafe fn prefetch(y: *const B128) {
    core::arch::x86_64::_mm_prefetch(y as *const i8, core::arch::x86_64::_MM_HINT_T0)
}

/// `y[t] += x[t] * v` for `t < R`, [`LANES`] lanes to one `vpclmulqdq` and the rest scalar.
#[inline(always)]
unsafe fn axpy<const R: usize>(y: *mut B128, x: *const B128, v: B128) {
    let vp = Packed::broadcast(v);
    let mut t = 0;
    while t + LANES <= R {
        store(y.add(t), load(x.add(t)) * vp + load(y.add(t)));
        t += LANES;
    }
    while t < R {
        *y.add(t) += *x.add(t) * v;
        t += 1;
    }
}

struct Sparse {
    n: usize,
    m: usize,
    d: usize,
    cols: Vec<u32>,
    vals: Vec<B128>,
    ptr: Vec<u32>,
    src: Vec<u32>,
    tvals: Vec<B128>,
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
        let (mut ptr, mut src, mut tvals) = (Vec::new(), Vec::new(), Vec::new());
        if n <= m || n * 2 * LANES <= GATHER_LIMIT {
            ptr = vec![0u32; m + 2];
            for &c in &cols {
                ptr[c as usize + 2] += 1;
            }
            for c in 0..m {
                ptr[c + 2] += ptr[c + 1];
            }
            src = vec![0u32; n * d];
            tvals = vec![B128::ZERO; n * d];
            for i in 0..n {
                for j in i * d..(i + 1) * d {
                    let c = cols[j] as usize + 1;
                    let p = ptr[c] as usize;
                    ptr[c] += 1;
                    src[p] = i as u32;
                    tvals[p] = vals[j];
                }
            }
            ptr.pop();
        }
        if n <= m {
            (cols, vals) = (Vec::new(), Vec::new());
        }
        Sparse { n, m, d, cols, vals, ptr, src, tvals }
    }

    /// `y = x A` on `rows` interleaved messages, `x` and `y` symbol-major.
    ///
    /// Which of the two kernels runs is a property of the level, not of the caller: the column
    /// sweep pays for its random reads of `x` and wins only while they stay in cache or while
    /// `x` is no larger than the `y` the scatter would be writing at random.
    fn dot(&self, rows: usize, x: &[B128], y: &mut [B128]) {
        assert_eq!(x.len(), self.n * rows);
        assert_eq!(y.len(), self.m * rows);
        if self.n <= self.m {
            return self.gather(rows, PREFETCH, x, y);
        }
        if self.n * rows.max(2 * LANES) <= GATHER_LIMIT {
            return self.gather(rows, 0, x, y);
        }
        y.fill(B128::ZERO);
        let (x, y) = (x.as_ptr(), y.as_mut_ptr());
        unsafe {
            match rows {
                1 => self.scatter::<1>(x, y),
                2 => self.scatter::<2>(x, y),
                3 => self.scatter::<3>(x, y),
                4 => self.scatter::<4>(x, y),
                5 => self.scatter::<5>(x, y),
                6 => self.scatter::<6>(x, y),
                7 => self.scatter::<7>(x, y),
                8 => self.scatter::<8>(x, y),
                9 => self.scatter::<9>(x, y),
                10 => self.scatter::<10>(x, y),
                11 => self.scatter::<11>(x, y),
                12 => self.scatter::<12>(x, y),
                13 => self.scatter::<13>(x, y),
                14 => self.scatter::<14>(x, y),
                15 => self.scatter::<15>(x, y),
                16 => self.scatter::<16>(x, y),
                _ => self.scatter_dyn(rows, x, y),
            }
        }
    }

    /// Row-major: `x` streams, every nonzero is a read-modify-write of `R` contiguous symbols at
    /// a random column of `y`.
    unsafe fn scatter<const R: usize>(&self, x: *const B128, y: *mut B128) {
        let nnz = self.n * self.d;
        for i in 0..self.n {
            let xi = x.add(i * R);
            for j in i * self.d..(i + 1) * self.d {
                if j + PREFETCH < nnz {
                    prefetch(y.add(*self.cols.get_unchecked(j + PREFETCH) as usize * R));
                }
                let (c, v) = (*self.cols.get_unchecked(j), *self.vals.get_unchecked(j));
                axpy::<R>(y.add(c as usize * R), xi, v);
            }
        }
    }

    unsafe fn scatter_dyn(&self, rows: usize, x: *const B128, y: *mut B128) {
        for i in 0..self.n {
            let xi = x.add(i * rows);
            for j in i * self.d..(i + 1) * self.d {
                let (c, v) = (*self.cols.get_unchecked(j), *self.vals.get_unchecked(j));
                let y = y.add(c as usize * rows);
                let vp = Packed::broadcast(v);
                let mut t = 0;
                while t + LANES <= rows {
                    store(y.add(t), load(xi.add(t)) * vp + load(y.add(t)));
                    t += LANES;
                }
                while t < rows {
                    *y.add(t) += *xi.add(t) * v;
                    t += 1;
                }
            }
        }
    }

    fn gather(&self, rows: usize, pf: usize, x: &[B128], y: &mut [B128]) {
        match rows {
            1 => self.sweep::<1>(pf, x, y),
            2 => self.sweep::<2>(pf, x, y),
            3 => self.sweep::<3>(pf, x, y),
            4 => self.sweep::<4>(pf, x, y),
            5 => self.sweep::<5>(pf, x, y),
            6 => self.sweep::<6>(pf, x, y),
            7 => self.sweep::<7>(pf, x, y),
            8 => self.sweep::<8>(pf, x, y),
            9 => self.sweep::<9>(pf, x, y),
            10 => self.sweep::<10>(pf, x, y),
            11 => self.sweep::<11>(pf, x, y),
            12 => self.sweep::<12>(pf, x, y),
            13 => self.sweep::<13>(pf, x, y),
            14 => self.sweep::<14>(pf, x, y),
            15 => self.sweep::<15>(pf, x, y),
            16 => self.sweep::<16>(pf, x, y),
            _ => self.sweep_dyn(rows, pf, x, y),
        }
    }

    /// Column-major: one output column at a time, its whole dot product accumulated unreduced in
    /// registers, so the level pays one reduction per output symbol instead of one per nonzero.
    fn sweep<const R: usize>(&self, pf: usize, x: &[B128], y: &mut [B128]) {
        let (x, y) = (x.as_ptr(), y.as_mut_ptr());
        let nnz = self.n * self.d;
        for c in 0..self.m {
            let (s, e) = (self.ptr[c] as usize, self.ptr[c + 1] as usize);
            let mut acc = [Wide::default(); MAX_ROWS / LANES];
            let mut tail = [WideLane::default(); LANES];
            for j in s..e {
                unsafe {
                    if pf > 0 && j + pf < nnz {
                        prefetch(x.add(*self.src.get_unchecked(j + pf) as usize * R));
                    }
                    let (i, v) = (*self.src.get_unchecked(j), *self.tvals.get_unchecked(j));
                    let xi = x.add(i as usize * R);
                    let vp = Packed::broadcast(v);
                    let mut g = 0;
                    while (g + 1) * LANES <= R {
                        acc[g] += Packed::wide_mul(load(xi.add(g * LANES)), vp);
                        g += 1;
                    }
                    let vl = Lane::broadcast(v);
                    let mut t = g * LANES;
                    while t < R {
                        let a = (xi.add(t) as *const Lane).read_unaligned();
                        tail[t - g * LANES] += Lane::wide_mul(a, vl);
                        t += 1;
                    }
                }
            }
            let mut g = 0;
            while (g + 1) * LANES <= R {
                unsafe { store(y.add(c * R + g * LANES), Packed::reduce(acc[g])) };
                g += 1;
            }
            let mut t = g * LANES;
            while t < R {
                let out = unsafe { y.add(c * R + t) as *mut Lane };
                unsafe { out.write_unaligned(Lane::reduce(tail[t - g * LANES])) };
                t += 1;
            }
        }
    }

    fn sweep_dyn(&self, rows: usize, pf: usize, x: &[B128], y: &mut [B128]) {
        let (x, y) = (x.as_ptr(), y.as_mut_ptr());
        let nnz = self.n * self.d;
        for c in 0..self.m {
            let (s, e) = (self.ptr[c] as usize, self.ptr[c + 1] as usize);
            let mut base = 0;
            while base + LANES <= rows {
                let mut acc = Wide::default();
                for j in s..e {
                    unsafe {
                        if pf > 0 && j + pf < nnz {
                            prefetch(x.add(*self.src.get_unchecked(j + pf) as usize * rows));
                        }
                        let (i, v) = (*self.src.get_unchecked(j), *self.tvals.get_unchecked(j));
                        let xi = load(x.add(i as usize * rows + base));
                        acc += Packed::wide_mul(xi, Packed::broadcast(v));
                    }
                }
                unsafe { store(y.add(c * rows + base), Packed::reduce(acc)) };
                base += LANES;
            }
            for t in base..rows {
                let mut acc = WideLane::default();
                for j in s..e {
                    let (i, v) = unsafe {
                        (*self.src.get_unchecked(j), *self.tvals.get_unchecked(j))
                    };
                    let a = unsafe { (x.add(i as usize * rows + t) as *const Lane).read_unaligned() };
                    acc += Lane::wide_mul(a, Lane::broadcast(v));
                }
                let out = unsafe { y.add(c * rows + t) as *mut Lane };
                unsafe { out.write_unaligned(Lane::reduce(acc)) };
            }
        }
    }

    fn nonzeros(&self) -> usize {
        self.n * self.d
    }

    fn bytes(&self) -> usize {
        20 * (self.cols.len() + self.src.len()) + 4 * self.ptr.len()
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

    /// Bytes [`BrakedownCode::new`] holds: one index and one value per nonzero of every level,
    /// in whichever of the two layouts that level's kernel dispatch can select.
    pub fn matrix_bytes(&self) -> usize {
        self.a.iter().chain(self.b.iter()).map(Sparse::bytes).sum()
    }

    /// The whole encoding, assuming `codeword[..k]` already holds the message, so that no row of
    /// a commitment ever allocates.
    pub fn encode_in_place(&self, codeword: &mut [B128]) {
        self.encode_in_place_interleaved(1, codeword);
    }

    /// [`encode_in_place`](Self::encode_in_place) on `rows` messages held symbol-major, the
    /// codeword being `rows * n` long with symbol `i` of message `j` at `codeword[i * rows + j]`.
    pub fn encode_in_place_interleaved(&self, rows: usize, codeword: &mut [B128]) {
        assert_eq!(codeword.len(), self.n * rows);
        let last = self.a.len() - 1;
        let mut input = 0;
        for a in &self.a[..last] {
            let (x, y) = codeword[input * rows..].split_at_mut(a.n * rows);
            a.dot(rows, x, &mut y[..a.m * rows]);
            input += a.n;
        }

        let (a, b) = (&self.a[last], &self.b[last]);
        let mut stack = [B128::ZERO; N_0 * MAX_ROWS];
        let mut spill = if rows <= MAX_ROWS { Vec::new() } else { vec![B128::ZERO; a.m * rows] };
        let base: &mut [B128] =
            if rows <= MAX_ROWS { &mut stack[..a.m * rows] } else { &mut spill };
        let (x, z) = codeword[input * rows..].split_at_mut(a.n * rows);
        a.dot(rows, x, base);
        reed_solomon_into(rows, base, &mut z[..b.n * rows]);

        let mut output = input + a.n + b.n;
        input += a.n + a.m;
        for (a, b) in self.a.iter().rev().zip(self.b.iter().rev()) {
            input -= a.m;
            let (z, v) = codeword.split_at_mut(output * rows);
            b.dot(rows, &z[input * rows..(input + b.n) * rows], &mut v[..b.m * rows]);
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

    fn encode_interleaved(&self, rows: usize, messages: &[B128], codeword: &mut [B128]) {
        assert_eq!(messages.len(), self.k * rows);
        codeword[..self.k * rows].copy_from_slice(messages);
        self.encode_in_place_interleaved(rows, codeword);
    }
}

fn reed_solomon_into(rows: usize, message: &[B128], codeword: &mut [B128]) {
    let head = message.len() - rows;
    for (j, out) in codeword.chunks_exact_mut(rows).enumerate() {
        let x = B128::new(j as u128);
        out.copy_from_slice(&message[head..]);
        for c in message[..head].chunks_exact(rows).rev() {
            for (acc, &c) in out.iter_mut().zip(c) {
                *acc = *acc * x + c;
            }
        }
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
    use crate::codes::test_util::{
        encode_interleaved_naive, interleaved_agrees_with_encode, low_weight, random, weight,
    };

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
    fn interleaved_agrees_with_encode_at_every_width() {
        for spec in [SPEC[0], SPEC[5]] {
            let code = BrakedownCode::new(300, spec, 23);
            for rows in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 24] {
                interleaved_agrees_with_encode(&code, rows, 31 + rows as u64);
            }
        }
        let wide = BrakedownCode::new(40000, SPEC[0], 29);
        for rows in [1, 4, 5, 16] {
            interleaved_agrees_with_encode(&wide, rows, 41 + rows as u64);
        }
    }

    #[test]
    #[ignore]
    fn setup_cost() {
        for log_k in [12, 16, 20] {
            let k = 1 << log_k;
            let (ms, code) = bin_ntt_bench::once(|| BrakedownCode::new(k, SPEC[0], 3));
            println!(
                "brakedown-1 k = 2^{log_k}  setup {ms:>8.3} ms  {:>7.1} MB  {:.1} B/nonzero",
                code.matrix_bytes() as f64 / 1e6,
                code.matrix_bytes() as f64 / code.field_muls() as f64,
            );
        }
    }

    #[test]
    #[ignore]
    fn interleaved_throughput() {
        for log_k in [12, 16, 20] {
            let k = 1 << log_k;
            let code = BrakedownCode::new(k, SPEC[0], 3);
            let n = code.codeword_len();
            for rows in [1, 4, 8, 16] {
                let messages = random(k * rows, 5);
                let mut codeword = vec![B128::ZERO; n * rows];
                for _ in 0..(1 << 22) / k {
                    code.encode_interleaved(rows, &messages, &mut codeword);
                }
                let (ms, _) = bin_ntt_bench::median_of(5, || {
                    code.encode_interleaved(rows, &messages, &mut codeword)
                });
                let (naive, _) = bin_ntt_bench::median_of(3, || {
                    encode_interleaved_naive(&code, rows, &messages, &mut codeword)
                });
                let mut one = vec![B128::ZERO; n];
                let (single, _) = bin_ntt_bench::median_of(5, || code.encode(&messages[..k], &mut one));
                let per = |t: f64| t * 1e6 / (k * rows) as f64;
                println!(
                    "brakedown-1 k = 2^{log_k} rows {rows:>2}  {ms:>8.3} ms {:>5.1} ns/sym   naive {naive:>8.3} ms {:>5.1} ns/sym   rows*encode {:>8.3} ms {:>5.1} ns/sym",
                    per(ms),
                    per(naive),
                    single * rows as f64,
                    per(single * rows as f64),
                );
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

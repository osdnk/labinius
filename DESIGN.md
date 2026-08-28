# bin-ntt — design contract for the AVX-512 kernels

Target machine only: Intel i7-11850H (Tiger Lake), single thread. Goal: forward NTT of binary
polynomials over R_q = Z_q[X]/(X^648 - X^324 + 1), q in {3889, 9721}, as fast as possible; the
yardstick is Gregor Seiler's estimate of ~650 cycles per polynomial for a generic-input NTT on this
core. We report cycles AND instruction/uop counts per polynomial (see `src/perf.rs`).

## 1. Ring, tree, slot order (fixed — see `src/params.rs`)

* psi = smallest primitive 1944-th root of unity (7 for 3889, 17 for 9721);
  omega = psi^648 (omega^2 + omega + 1 = 0); zeta6 = psi^324 (zeta6^-1 = 1 - zeta6).
* Level 0 (Phi_6 split): a = a0 + X^324 a1 ->  block 0: a0 + zeta6*a1,  block 1: a0 + a1 - zeta6*a1.
* Level l (1..6), sub-ring k of degree n = DEGREE[l] at block offset k*n, radix p = RADIX[l],
  twiddle zeta = psi^twiddle_exp(l,k) (= `Params::<Q>::ZETA_L{l}[k]`), m = n/p, for i in 0..m:
  - p = 2:  a0 = v[i], a1 = v[m+i]:  v[i] = a0 + zeta*a1,  v[m+i] = a0 - zeta*a1.
  - p = 3:  t1 = zeta*a1, t2 = zeta^2*a2, u = omega*(t1 - t2):
            v[i] = a0 + t1 + t2,  v[m+i] = a0 - t2 + u,  v[2m+i] = a0 - t1 - u.
  (These are exactly what `scalar::ntt` does; SIMD outputs must match it modulo q, slot for slot.)
* Slot j holds a(psi^SLOT_EXP[j]).  Output type `Batch32 { v: [[i16;32]; 648], representation: Ntt }`
  (v[j][p] = slot j of polynomial p) for the vertical layouts.

## 2. Arithmetic (signed i16 lanes)

* Hard invariant: every lane value must satisfy |x| < 2^15. Budget: 8.42q for 3889, 3.37q for 9721.
* Twiddle multiplication = signed Montgomery with a precomputed companion (3 multiply uops on p0 +
  the final vpsubw on p05 = 4 uops total; count all 4 in uop budgets):
  w = to_mont(x) = x*2^16 mod q centered, w' = mont_pre(w) = w*qinv mod 2^16;
  mont(a, w, w') = mulhi(a, w) - mulhi(mullo(a, w'), q)  = a*x mod q, in (-q, q).
  Tighter: |mont| <= |a|*|w|/2^16 + q/2 <= |a|*q/2^17 + q/2  (|w| <= q/2), i.e. < 0.75q for any i16 a.
  Scalar model: `params::mont_mul_i16`. Vector model: `simd::pointwise::mont_mul_epi16` (4-uop
  general form) — the 3-uop twiddle form is mullo(a,w'), mulhi(a,w), mulhi(m,q), sub.
* Cheap partial reduction `barrett_i16` (vpmulhrsw + vpmullw + vpsubw = 2 multiply uops):
  |r| <= 0.899q (3889) / 0.809q (9721) for any i16 input. Use it only where the bound analysis
  needs it; never use vpminuw/vpmaxsw/shift-based tricks in hot loops (they occupy the multiply port).
* Products t1 - t2 in the radix-3 butterfly are < 1.5q < 2^15 for both primes, so u = mont(t1-t2, omega)
  is always safe when |t1|,|t2| < 0.75q.
* Every kernel must document its per-level bounds and its OUTPUT bound (as a multiple of q, per prime)
  and verify them in a test with an i32 shadow evaluation over many random + adversarial inputs
  (all-ones, alternating, single monomials, all-zero).

## 3. Measured port facts (tools/ubench, this CPU) — design around p0

| zmm instruction                                   | throughput | port |
|---------------------------------------------------|-----------:|------|
| vpmullw / vpmulhw / vpmulhrsw / vpmaddwd / vpdpwssd |  1 / cycle | p0 only (ymm: 2/cycle p01) |
| vpaddw / vpsubw / vpandd / vpternlogd / masked add / vpblendmw / vpmovm2w | 2 / cycle | p05 |
| vpminuw / vpmaxsw / vpsraw / vpsrlw / vpsllw / vgf2p8affineqb | 1 / cycle | p0 (competes with multiplies!) |
| vpermb / vpshufb / vpermd / vpermq / vpunpck* / vpalignr / vshufi64x2 | 1 / cycle | p5 |
| vpermw (zmm)                                        | 1 / cycle  | **2 uops: p0 + p5** (measured with port counters; use vpermb byte-split tables instead) |
| vpermi2w / vpermt2w / vpermi2b / vpmovwb            | 0.5 / cycle | 2 uops |
| vpbroadcastw m16 (p5 uop!)  vs  vpbroadcastd m32 (pure load, free) | 1 vs 0.5 | — |
| kmovd/kmovq k, m (1 uop on p5)                      | 1 / cycle  | p5 |
| zmm load 2/cycle, zmm store 1/cycle                  |            | — |

Consequences: cycles ~= max(p0 uops, (all ALU uops)/2); in practice ~15-20% of the adds are
dispatched to p0 even when it is saturated (measured on vertical_gen), so p0 = multiplies + ~0.17*adds. Twiddle broadcast trick: store each i16
constant duplicated as a u32 (w | w << 16) and use `_mm512_set1_epi32` from memory (vpbroadcastd) —
free. Latencies: vpmullw/vpmulhw 5, vpermw 3, vpaddw 1: keep >= 6 independent butterflies in flight.

## 4. Data types (`src/types.rs`)

* `BinaryPoly { bits: [u64; 11] }` — natural storage form, bit i = coefficient i.
* `BinaryBatch32 { idx: [[u8; 32]; 162] }` — nibble-sliced: idx[i][p] = b_i | b_{i+162}<<1 |
  b_{i+324}<<2 | b_{i+486}<<3 of polynomial p. This IS the index of the fused levels 0+1 lookup.
* `Batch32 { v: [[i16; 32]; 648], representation }` — vertical layout, 64-byte aligned.
* `RingElement { v: [i16; 648], representation }` — rokoko-style single element (Batch32::get/set).

## 5. Binary fast path: fused levels 0+1 as 16-entry lookups, with twiddle folding

After levels 0 and 1 the value at position i (0 <= i < 162) of block k = 2*s0 + s1 (offset k*162+i) is

    val = b_i + kappa_{s0} * b_{i+324} + sigma_{s1} * zeta1_{s0} * (b_{i+162} + kappa_{s0} * b_{i+486})
    kappa_0 = zeta6, kappa_1 = 1 - zeta6, sigma_0 = +1, sigma_1 = -1, zeta1_{s0} = ZETA_L1[s0],

a linear function of the 4-bit nibble n_i = idx[i], i.e. a table T_k[n] of 16 centered i16 values,
looked up with one `vpermb` on a byte-split table (low bytes at index n, high bytes at 16+n; index
row = (n, 16+n) pairs) — `vpermw` would cost an extra p0 uop per lookup.
Since each intermediate is consumed in exactly one role, later twiddles can be folded into tables:

* Level 2 (radix-2 on block k, pairs (i, i+81), zeta' = ZETA_L2[k]): positions i >= 81 use T_k*zeta'.
* Level 3 (radix-3 on the 81-blocks (k, s2), zeta'' = ZETA_L3[2k+s2], role r = (i mod 81)/27):
  the level-2 outputs are y0 = a_i + zeta' a_{i+81} (goes to s2=0) and y1 = a_i - zeta' a_{i+81}
  (s2=1); y0 needs fold f0 = ZETA_L3[2k]^r, y1 needs f1 = ZETA_L3[2k+1]^r. For i < 27 (r = 0):
  2 lookups + 2 adds per pair; for 27 <= i < 81: 4 lookups (T_k f0, T_k zeta' f0, T_k f1,
  T_k zeta' f1) + 2 adds. Level 3 then needs only the omega multiplication: per triple
  y0 = a0 + t1 + t2, u = mont(t1 - t2, omega), y1 = a0 - t2 + u, y2 = a0 - t1 - u.
  Table entries are centered (|T| <= q/2) so level-2 outputs are < q, level-3 outputs < 3q
  (< 2^15 for both primes), then levels 4-6 as in section 1 with Barretts where the bounds require
  (9721: the untwiddled a0 inputs of levels 4, 5, 6 need `barrett` — check; 3889: none needed).
  Number of tables: per block k, 10 (T, T zeta', and the 8 folded ones) x 16 entries; 40 tables,
  1280 bytes; use them as memory operands of vpermw.
* Expected cost per batch of 32: ~1080 vpermw + ~650 adds (levels 0-2) + 216 x (3 p0 + 8 adds)
  (level 3) + 3 x 216 x (9 p0 + 10 adds) (levels 4-6) ~= 16.5k uops, ~6.5k multiplies on p0
  => floor ~255 cycles/poly (total/2), realistically ~280-300 (measured vertical_gen: 423/467).

## 6. Variants and their APIs (each in its own file under `src/simd/`)

All kernels: `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]`
unsafe fns with `core::arch::x86_64` intrinsics; const generic `<const Q: u16>` (Q in QS). Tables
should be `const`-evaluated (see `Params`) or built once in a `LazyLock`. Depth-first over
sub-rings so the working set stays in L1 (a 162-block is 10 KB). Inline asm is allowed if the
compiler spills or mis-schedules.

* `vertical_bin.rs`  — `pub unsafe fn ntt_bin_batch32<const Q: u16>(input: &BinaryBatch32, out: &mut Batch32)`
  plus a driver `pub fn ntt_bin_polys<const Q: u16>(polys: &[BinaryPoly], out: &mut [Batch32])`
  (polys.len() == 32 * out.len(); transposes with `transpose::slice_polys` then runs the kernel;
  materialised output, consider non-temporal stores for the final level) and a streaming form
  `pub fn ntt_bin_stream<const Q: u16>(polys: &[BinaryPoly], f: impl FnMut(usize, &Batch32))`.
* `transpose.rs` — `pub unsafe fn slice_polys(polys: &[BinaryPoly; 32]) -> BinaryBatch32` (AVX-512;
  scalar reference is `BinaryBatch32::from_polys_scalar`). Report its cost separately.
* `vertical_gen.rs` — generic-input `pub unsafe fn ntt_gen_batch32<const Q: u16>(b: &mut Batch32)`
  in place, Coefficients -> Ntt, input lanes |x| <= q. Fuse levels 0+1 into one radix-4 pass over
  the 648 vectors (L2-resident), then depth-first per 162-block.
* `horizontal_gen.rs` — Gregor's layout: `pub struct HBatch4 { v: [[i16; 32]; 81] }` with
  v[r][8p + j] = coefficient r + 81*j of polynomial p (p in 0..4, j in 0..8), i.e. one polynomial per
  128-bit lane, 8 coefficients of stride 81 per lane, 81 registers per 4 polynomials. The three
  radix-2-type levels (0, 1, 2) are in-lane (vpshufd/vpshufb/vpermq-style duplication + per-lane
  twiddle vectors); the four radix-3 levels are register-to-register on r. Define and document the
  resulting slot order, and provide `HBatch4::get(p) -> RingElement` that returns TREE order so the
  common test applies. `pub unsafe fn ntt_gen_hbatch4<const Q: u16>(b: &mut HBatch4)`.

## 7. Tests and benches every variant must ship

* `tests/<variant>.rs`: for both primes, >= 64 batches of random binary polys plus adversarial
  ones: normalized SIMD output == `scalar::ntt::<Q>(&scalar::lift(&poly))` slot-for-slot; declared
  output bound holds (max |v|); and the multiplication test: `pointwise::mul_batch_batch` of
  NTT(a), NTT(b) normalized == `scalar::ntt(scalar::mul_mod_phi(a, b))`, and `mul_batch_element`
  against a single random element. Generic kernels also test random i16 inputs in [-q, q].
* `src/bin/bench_<variant>.rs`: pin to one core (sched_setaffinity via libc syscall or `taskset`
  externally), warm up, then measure with `perf::PerfGroup` (cycles, instructions, uops, p0, p1, p5)
  over (a) an L2-resident working set (e.g. 512 polys repeated) and (b) 2^18 polys materialised
  (340 MB output; DRAM floor measured: NT stores 37 GB/s = 9.1 ms, regular stores 14 GB/s = 24 ms)
  and (c) streaming with a trivial consumer. Print per-polynomial numbers. Build with
  `CARGO_TARGET_DIR=target/<variant>` to avoid fighting over the cargo lock with other agents.
* Do not edit files outside your own module/test/bench files except to add `pub mod` lines already
  present; `params.rs`, `types.rs`, `scalar.rs`, `pointwise.rs` are shared — propose changes in your
  report instead of editing them (unless they are bugs, then fix and say so).

# bin-ntt — design notes for the AVX-512 kernels

Target machine only: Intel i7-11850H (Tiger Lake), single thread. Goal: forward NTT of binary
polynomials over R_q = Z_q[X]/(X^648 - X^324 + 1), q in {3889, 9721}, as fast as possible. The
input is a stream of `bin_fields::scalar::F162`; four consecutive elements form one ring element
by plain interleaving (coefficient of X^{4m+k} = bit m of element k), see `src/f162.rs`. The
yardstick is Gregor Seiler's estimate of ~650 cycles per polynomial for a generic-input NTT on this
core. Cycles and instruction/uop counts per polynomial are measured with `src/perf.rs`. The
measured results and the per-kernel strategies are in README.md; this file holds the shared
definitions every kernel follows.

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
| merge-masked zmm load                               |            | ~1 extra p0/p5 uop on top of the load |
| zmm load 2/cycle, zmm store 1/cycle                  |            | — |

Consequences: cycles ~= max(p0 uops, (all ALU uops)/2); in practice 15-28% of the adds are
dispatched to p0 even when it is saturated (they depend on 5-cycle multiplies, see
`tools/ubench/ports.c`), so p0 = multiplies + ~0.2 * adds. Twiddle broadcast trick: store each
i16 constant duplicated as a u32 (w | w << 16) and use `_mm512_set1_epi32` from memory
(vpbroadcastd) — free. Latencies: vpmullw/vpmulhw 5, vpermb 3, vpaddw 1: keep >= 6 independent
butterflies in flight.

## 4. Data types (`src/types.rs`)

* `BinaryPoly { bits: [u64; 11] }` — a plain 648-coefficient 0/1 polynomial, bit i = coefficient
  i (test and comparison-bench input form).
* `BinaryBatch32 { idx: [[u8; 32]; 162] }` — nibble-sliced: idx[i][p] = b_i | b_{i+162}<<1 |
  b_{i+324}<<2 | b_{i+486}<<3 of polynomial p. This IS the index of the fused levels 0+1 lookup.
* `Batch32 { v: [[i16; 32]; 648], representation }` — vertical layout, 64-byte aligned.
* `RingElement { v: [i16; 648], representation }` — rokoko-style single element (Batch32::get/set).

## 5. Binary fast path: fused levels 0+1 as 16-entry lookups, with twiddle folding

After levels 0 and 1 the value at position i (0 <= i < 162) of block k = 2*s0 + s1 (offset k*162+i) is

    val = b_i + kappa_{s0} * b_{i+324} + sigma_{s1} * zeta1_{s0} * (b_{i+162} + kappa_{s0} * b_{i+486})
    kappa_0 = zeta6, kappa_1 = 1 - zeta6, sigma_0 = +1, sigma_1 = -1, zeta1_{s0} = ZETA_L1[s0],

a linear function of the 4-bit nibble n_i = (b_i, b_{i+162}, b_{i+324}, b_{i+486}), i.e. a table
T_k[n] of 16 centered i16 values, looked up with one `vpermb` on a byte-split table (low bytes at
index n, high bytes at 16+n; the input rows of `BinaryIndex32` are the (n, 16+n) index pairs) —
`vpermw` would cost an extra p0 uop per lookup.
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
  (9721: the untwiddled a0 inputs of levels 4, 5, 6 get a `barrett`; 3889: none needed).
  Number of tables: per block k, 10 (T, T zeta', and the 8 folded ones) x 16 entries; 40 tables,
  1280 bytes; they are used as memory operands of vpermb.
* Measured static cost per batch of 32 (q = 3889, from the disassembly): 23 363 instructions,
  6 480 multiply-port uops, 1 080 `vpermb`, ~8 900 add/sub; port floor 256 cycles per polynomial,
  measured 297 (330 for 9721, whose Barretts add 1 296 multiply-port uops).

## 6. Kernel APIs (one module per variant under `src/simd/`)

All kernels are `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]`
unsafe fns over `core::arch::x86_64` intrinsics, const generic in `<const Q: u16>` (Q in QS), with
tables `const`-evaluated through `Params` or built once. They work depth-first over sub-rings so
the working set stays in L1 (a 162-block is 10 KB). Inline `asm!` is used only where LLVM's
code generation for an intrinsic is poor (`vpmulhw`, twiddle broadcasts).

* `vertical_bin` — `ntt_bin_batch32::<Q>(&BinaryIndex32, &mut Batch32)`; drivers
  `ntt_bin_polys::<Q>(&[BinaryPoly], &mut [Batch32])` (materialised, non-temporal stores for the
  last level) and `ntt_bin_stream::<Q>(&[BinaryPoly], impl FnMut(usize, &Batch32))`.
* `transpose_f162` — `slice_f162(&[F162; 128]) -> BinaryIndex32`: the production front end
  (scalar reference `f162::index_rows_scalar`). `ntt_f162` — drivers `ntt_f162::<Q>(&[F162],
  &mut [Batch32])`, `ntt_f162_2q`, `ntt_f162_stream`.
* `vertical_bin_asm` — same API as `vertical_bin`, levels 4-6 in `asm!`, bit-identical output;
  the production kernel.
* `transpose` — `slice_polys(&[BinaryPoly; 32]) -> BinaryBatch32` (nibble form, equal to
  `BinaryBatch32::from_polys_scalar`) and `slice_polys_idx(..) -> BinaryIndex32` (the `vpermb`
  index rows the kernel consumes) for plain 648-bit polynomials.
* `vertical_gen` — `ntt_gen_batch32::<Q>(&mut Batch32)` in place, Coefficients -> Ntt, input lanes
  |x| <= q; `ntt_gen_batches` over a slice with prefetching.
* `horizontal_gen` — `HBatch4 { v: [[i16; 32]; 81] }` with v[r][8p + j] = coefficient r + 81 j of
  polynomial p (one polynomial per 128-bit lane, 8 coefficients of stride 81 per lane);
  `ntt_gen_hbatch4::<Q>(&mut HBatch4)`; output slot 81 j + r; `HBatch4::get(p)` returns tree order.
* `pointwise` — `mul_batch_batch`, `mul_batch_element` (Montgomery, exact results).

## 7. Tests and benchmarks

* `tests/<variant>.rs`: for both primes, >= 64 batches of random binary polynomials plus
  adversarial ones (all zero, all ones, alternating, monomials at block boundaries): normalized
  SIMD output == `scalar::ntt::<Q>(&scalar::lift(&poly))` slot for slot; the declared output
  bound holds; an i32 shadow model replays the kernel's operation sequence and asserts every
  intermediate is < 2^15; and the multiplication tests through `pointwise` against
  `scalar::ntt(scalar::mul_mod_phi(a, b))`. Generic kernels also test random i16 inputs in [-q, q].
* `src/bin/bench_f162.rs`: the headline (2^20 F162 -> 2^18 ring elements, both primes, single and
  two-prime drivers, streamed, accumulate product), pinned to one core, `perf::PerfGroup`
  counters (cycles,
  instructions, uops, ports 0/1/5). `src/bin/bench_<variant>.rs`: per-kernel breakdowns,
  cache-resident and out-of-cache cases, static instruction counts from the disassembly.
  DRAM floors measured with `tools/membw`: non-temporal stores 37 GB/s, regular stores 14 GB/s,
  reads 19.5 GB/s.

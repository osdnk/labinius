# bin-ntt

AVX-512 forward NTT over the 1944-th cyclotomic ring

    R_q = Z_q[X] / (X^648 - X^324 + 1),    1944 = 2^3 * 3^5,    q in {3889, 9721},

specialised for **binary** (0/1) input polynomials and vectorised across many polynomials.
Both primes are 1 mod 1944, so R_q splits into 648 linear factors and the transform is
complete. Single-threaded, tuned for one specific core (Intel i7-11850H, Tiger Lake), signed
16-bit lanes throughout. Rust stable, no dependencies.

## Headline: 2^18 polynomials, one core

Input: a plain `Vec<BinaryPoly>` (2^18 x `[u64; 11]`). Output: a preallocated `Vec<Batch32>`
(8192 batches of 32 polynomials, 340 MB, written with non-temporal stores). Best of 3, pinned
to one core, hardware counters via `perf_event_open`, machine otherwise idle:

| q    | kernel                                             | total ms | cycles / element | instructions / element | uops / element (p0 / p5) |
|------|----------------------------------------------------|---------:|-----------------:|-----------------------:|-------------------------:|
| 3889 | **vertical binary** (`vertical_bin`, the deliverable) | **21.4** | **364**       | 796                    | 805 (291 / 264)          |
| 9721 | **vertical binary**                                | **24.9** | **402**          | 857                    | 868 (330 / 286)          |
| 3889 | vertical generic, i16 input (`vertical_gen`)       | 31.1     | 513              | 1099                   | 1088 (417 / 338)         |
| 9721 | vertical generic                                   | 32.5     | 517              | 1175                   | 1160 (462 / 369)         |
| 3889 | horizontal generic, 4 polys / zmm (`horizontal_gen`) | 38.0   | 613              | 1314                   | 1298 (511 / 472)         |
| 9721 | horizontal generic                                 | 43.0     | 682              | 1443                   | 1470 (588 / 559)         |

For scale: 650 cycles per polynomial (the estimate we started from for a generic NTT of this
size on this core) is ~38 ms. The binary kernel alone, cache-resident, is 297 / 330 cycles per
polynomial (q = 3889 / 9721); the input transpose adds 31, and the remainder is the 340 MB
output stream (fully hidden by the non-temporal stores: the streamed variant that never writes
the output runs at 347 / 383 cycles). Cycles are the robust number; the clock under AVX-512
load varies between 4.2 and 4.6 GHz on this laptop, so milliseconds move by ~5 % run to run.

Multiplying in the NTT domain on the same 2^18 elements (not optimised, DRAM-bound on the
extra 340 MB of operands; the multiply itself is ~100 uops per element):

| q    | operation                                                   | total ms | cycles / element |
|------|-------------------------------------------------------------|---------:|-----------------:|
| 3889 | y = sum_i a_i * NTT(w_i), 2^18 distinct NTT-domain a_i      | 40.7     | 678              |
| 9721 | same                                                        | 43.0     | 713              |
| 3889 | c_i = NTT(w_i) * b for one fixed b, materialised            | 45.1     | 769              |
| 9721 | same                                                        | 48.0     | 794              |

Reproduce: `cargo run --release --bin bench_all -- <cpu>` (add `--quick` for 2^14). The
per-kernel binaries `bench_vertical_bin`, `bench_vertical_gen`, `bench_horizontal_gen` print
the component breakdowns, static instruction counts from the disassembly and the bound tables.

## Building and testing

`.cargo/config.toml` sets `-C target-cpu=native`; the kernels need AVX-512
F/BW/VL/VBMI/VBMI2/VNNI/GFNI and are only tuned for Tiger Lake.

    cargo test --release          # 27 tests: scalar reference, every kernel vs the reference
                                  # slot for slot, bounds via an i32 shadow model, products
    cargo run --release --bin bench_all -- 2

## What is computed

`psi` is the smallest primitive 1944-th root of unity mod q (7 for 3889, 17 for 9721),
`omega = psi^648`, `zeta6 = psi^324` (so `zeta6^-1 = 1 - zeta6`). The transform is a vanilla
mixed-radix Cooley-Tukey, no twisting, on the fixed tree

    level 0:     X^648 - X^324 + 1 = (X^324 - zeta6)(X^324 - zeta6^-1)       (Phi_6 split)
    level 1, 2:  radix 2   X^n - psi^e = (X^{n/2} - psi^{e/2})(X^{n/2} + psi^{e/2})
    level 3..6:  radix 3   X^n - psi^e = prod_s (X^{n/3} - psi^{e/3} omega^s)

with child s of a sub-ring stored at block offset s*n/p ("tree order"), so slot j of the output
holds `a(psi^SLOT_EXP[j])` (`params::SLOT_EXP`, a permutation of the units mod 1944). The
scalar reference `scalar::ntt` defines the order; every kernel is tested slot for slot against
it, and `scalar::ntt(a*b mod Phi) == NTT(a) o NTT(b)` is checked through the SIMD products.

Output values are lazily reduced signed residues stored as `i16`: |v| <= 7.5 q for q = 3889
and <= 2.31 q for q = 9721 in `vertical_bin` (each kernel documents its own bound);
`RingElement::normalized` maps to [0, q).

### Types (`src/types.rs`)

* `BinaryPoly { bits: [u64; 11] }` — bit i = coefficient i; the input storage form.
* `Batch32 { v: [[i16; 32]; 648], representation }` — 32 polynomials in the "vertical" layout:
  `v[j][p]` is slot (or coefficient) j of polynomial p, so one 512-bit vector is one slot of all
  32 polynomials. 64-byte aligned, 41 472 bytes.
* `RingElement { v: [i16; 648], representation }` — one element, rokoko-style flat array plus
  representation tag; `Batch32::get / set`.
* `BinaryBatch32` (nibble-sliced) and `BinaryIndex32` (`vpermb` index rows) — kernel-side forms
  of 32 binary polynomials, produced by `simd::transpose`.
* `HBatch4 { v: [[i16; 32]; 81] }` — the horizontal layout, 4 polynomials per batch.

### Arithmetic

Signed Montgomery multiplication by a constant with a precomputed companion
(`mullo(a, w')`, `mulhi(a, w)`, `mulhi(m, q)`, `sub`: three uops on the multiply port), the
radix-3 butterfly `t1 = zeta a1, t2 = zeta^2 a2, u = omega (t1 - t2); y0 = a0 + t1 + t2,
y1 = a0 - t2 + u, y2 = a0 - t1 - u` (3 multiplies, 10 add/sub), and lazy reduction under the
invariant |x| < 2^15 tracked per level (budget 8.42 q for 3889, 3.37 q for 9721; a Montgomery
product is bounded by |a| q / 2^17 + q/2 < 0.75 q). The only reduction used is a two-uop
`vpmulhrsw` Barrett (|r| < 0.9 q, exhaustively verified). All bound tables are `const`
recursions checked by an i32 shadow model in the tests.

## The binary trick

The NTT is linear, and a linear function of a few bits is a small table. After levels 0 and 1,
every value is a fixed combination of the four input bits (b_i, b_{i+162}, b_{i+324},
b_{i+486}); so both levels are replaced by one lookup per output vector in a 16-entry table,
and the input representation is chosen to be exactly the lookup index. Because every
intermediate value of the tree is consumed in exactly one role, the twiddles of level 2 and of
level 3 are pre-multiplied into the tables ("folding"), which removes all multiplications of
level 2 and two of the three of level 3 at the cost of extra lookups on the otherwise idle
shuffle port. Multiplications drop from 3564 per polynomial (generic) to 2160; the rest is the
same radix-3 machinery as the generic kernel. Details below.

## Strategies taken in each kernel

### `simd/vertical_bin.rs` — the binary kernel (297 / 330 cycles per polynomial resident)

* **Layout.** Vertical: one zmm = one slot of 32 polynomials, so no shuffle is ever needed for
  a butterfly and every twiddle is a broadcast constant. The batch (648 vectors, 41 KB) is
  processed depth-first per 162-block and 81-block, so the live working set is ~10 KB and stays
  in L1; the two 10 KB stack buffers are kept adjacent (padding them apart was measured neutral
  or worse).
* **Input.** `BinaryIndex32`: 162 rows of 64 bytes, `row[i][2p] = n, row[i][2p+1] = 16 + n`,
  where n is the nibble (b_i, b_{i+162}, b_{i+324}, b_{i+486}) of polynomial p. This is the
  index operand of `vpermb`, so the fused levels 0+1 are one `vpermb` per output vector with no
  index building at all. The tables are byte-split (low halves at index n, high halves at
  16 + n) because `vpermw` on this core is two uops (p0 + p5) while `vpermb` is one (p5),
  worth ~30 cycles per polynomial.
* **Folding.** 10 tables per 162-block (40 in all, 1280 bytes, used as memory operands): T,
  T zeta' (level-2 twiddle for positions >= 81), and the eight versions multiplied by the
  level-3 twiddle powers zeta''^r for both level-2 children. For pair positions i < 27 the
  level-2 butterfly is 2 lookups + 2 adds; for 27 <= i < 81 the two children need different
  folds, so 4 lookups + 2 adds. Level 3 keeps only the omega multiplication per triple.
* **Levels 4-6.** Radix-3 Montgomery butterflies; levels 5 and 6 are fused in one pass over each
  27-block (18 iterations), level 4 is its own pass (6 x 9). Fusing level 4 in as well needs
  34+ live registers and spills. Twiddles are duplicated-u32 constants broadcast by
  `vpbroadcastd`, a pure load.
* **Reduction.** Table entries are centred (|T| <= q/2), so level-2 outputs are < q and level-3
  outputs < 3 q. For 3889 nothing else is needed (output <= 7.5 q declared, 4.4 q observed). For
  9721 one Barrett on the un-twiddled a0 input of levels 4, 5 and 6 caps every output at
  0.81 q + 1.5 q = 2.31 q; skipping any one of them overflows.
* **Compiler workarounds.** stdarch's `_mm512_mulhi_epi16` is written as sext-mul-shift-trunc,
  and LLVM rematerialises the resulting `vpmovsxwd / vpmovdw / vinserti64x4` round trip inside
  the hot loops; `vpmulhw` and the twiddle broadcasts are therefore emitted with `asm!`
  (`pure`, `nomem/readonly`, so scheduling is unaffected). ~15 cycles per polynomial.
* **Drivers.** `ntt_bin_polys` (materialised, the last level stores with `vmovntdq`, which hides
  the 340 MB output entirely) and `ntt_bin_stream` (closure per batch); both reuse one
  `BinaryIndex32` scratch and call the transpose per batch.
* **Tried and rejected.** Folding omega into the tables too (kills level 3's multiply but adds
  864 lookups in a phase that is already shuffle-bound: worse floor); fusing level 4 into the
  table loop (register pressure); ymm arithmetic (floor 11160 vs 8370 uops per batch);
  non-temporal stores for cache-resident output (2 % slower).
* **Where it stands.** Static port floor 256 / 287 cycles per polynomial; measured 297 / 330
  with the multiply port 94-96 % busy. The gap is the hardware sending ~28 % of the flexible
  adds to port 0 where ~21 % would balance (see the port experiment below).

### `simd/transpose.rs` — `BinaryPoly` x 32 -> `BinaryIndex32` (31 cycles per polynomial)

Four phases, all shuffles and GFNI: (1) 8x8 qword transposes (`vshufi64x2` / `vpunpck*qdq`)
so that each qword holds the same 64-bit word of 8 polynomials; (2) one `vpermb` per word then
one `vgf2p8affineqb` with the identity matrix as the *vector* operand, which transposes each
8x8 bit block, giving byte t of qword B = bit 8B+t of the 8 polynomials, then a 4-way byte
interleave into a `[u32; 704]` coefficient-major mask array; (3) a 4x81 qword transpose of that
array so the four bit planes a row needs are 32 contiguous bytes; (4) per two output rows, one
merge-masked load of those bytes next to two constant bytes, one `vpermb` that lays them out as
the *matrix* operand of a second `vgf2p8affineqb` whose rows 4..7 are the four plane masks and
row 3 supplies the +16, producing `[n_0..n_31 | n_0+16..n_31+16]` in one instruction, and a
final `vpermb` interleave. No mask registers are used (`kmov k, m` is a p5 uop and the first
version spent 324 of them per batch). Port-5 bound, within 18 % of its static floor.

### `simd/vertical_gen.rs` — generic-input vertical kernel (423 / 467 resident)

* Same vertical layout and Montgomery arithmetic; the input is any `Batch32` of i16
  coefficients with |x| <= q.
* Pass A fuses levels 0 and 1 into one radix-4 pass over the 648 vectors (load 4, 4
  multiplies, store 4, 162 times) so the 41 KB batch is streamed once from L2; pass B fuses
  levels 2 and 3 over groups of 6 vectors; then one pass per 27-block for each of levels 4, 5
  and 6. Zero shuffle uops; twiddles are duplicated-u32 broadcasts.
* Reduction: for 3889 a single Barrett level (the a0 inputs of level 5), output <= 3.40 q; for
  9721 a Barrett on the `a0 + a1 - zeta6 a1` half inside pass A plus the a0 of levels 3-6,
  output <= 2.12 q.
* Radix-9 and radix-27 fusion of the tail were measured and rejected (439-468 vs 423 cycles):
  the kernel is multiply-port bound, not L1 bound, and fusing lengthens the dependency chain
  inside a group so fewer independent butterflies are in flight. The fused / unfused variants are
  kept behind `ntt_gen_batch32_plan` and tested for equality.
* For the out-of-cache 2^18 case a software prefetch of the next batch, one line per level-4/5/6
  butterfly, is worth 23 %; issuing 162 prefetches at once at each block head is 18 % slower
  than no prefetch (fill-buffer overrun).
* Where it stands: cycles are 97.5 % of the measured port-0 uops; static floor 363 / 404.

### `simd/horizontal_gen.rs` — Gregor's layout, generic input (540 / 587 resident)

* `HBatch4`: `v[r][8p + j]` = coefficient r + 81 j of polynomial p — one polynomial per 128-bit
  lane, the 8 coefficients of stride 81 inside a lane, 81 registers for 4 polynomials (5 KB).
* Levels 0-2 are in-lane and fused per register: `u[j] = a[j | s]`, `v[j] = a[j & !s]` built
  with `vpshufd` (0xEE/0x44 and 0xF5/0xA0) for levels 0-1 and `vpshufb` for level 2, then
  `out = v + c * u` with a per-lane twiddle vector (both halves get multiplied; inherent to the
  layout): 2 shuffles + 3 multiplies + 2 add/sub per register per level, 225 of the 540 cycles.
  After level 2 lane position `j = 4 s0 + 2 s1 + s2` carries the tree's 81-block k = j, so the
  slot order is `81 j + r` and `HBatch4::get` returns tree order.
* Levels 3-6 are register-to-register radix-3 passes on r (strides 27, 9, 3, 1), one pass per
  level with three independent triples per iteration; fusing levels into one pass measured
  slower (566 vs 542) for the same dependency-chain reason as above. The 40 (zeta, zeta^2)
  per-lane constant vectors are hoisted into registers by hand, because LLVM otherwise rebuilds
  them inside the loop with `vpmovsxwd / vpmovdw / vinserti64x4`.
* Reduction: none for 3889 (output <= 7.69 q); for 9721 a Barrett on the a0 of every radix-3
  level (output <= 2.12 q); bounds are `const`-derived with a compile-time assert.
* Out of cache: one `prefetcht0` per register of the first pass for the next batch, 772 -> 633
  cycles per polynomial.
* Tried and rejected: sharing the level-1 multiply across register pairs with a `vshufps` pack
  (fewer multiplies, but 548 vs 541); `black_box` on the table pointer (the loads stop folding
  into the multiplies).

### Why there is no horizontal binary kernel

The lookup trick needs, in every output lane, an index made of the four bits (b_i, b_{i+162},
b_{i+324}, b_{i+486}) of that lane's polynomial. In the vertical layout that index *is* the
input row (one nibble per polynomial per lane), so a lookup costs one `vpermb` and nothing
else, and folding a twiddle is merely choosing a different table. In the horizontal layout the
four bits sit at lane positions j, j+2, j+4, j+6 of the same 128-bit lane, so every register
first needs an in-lane bit gather (byte shuffles plus shift/or, ~4-6 uops) to build the index,
and the four sub-rings sharing a 128-bit lane need four different tables, i.e. a 64-entry
`vpermi2b` (2 uops) or two `vpermb` plus a blend per lookup. Levels 0-2 would drop from ~21 to
~14 uops per register — roughly 540 -> 450-470 cycles per polynomial — but the radix-3 levels
cost exactly what they cost in the vertical layout, the level-3 twiddle folding (which removes
two of that level's three multiplications) would need one table per (lane, role) and double
the gathers, and all of the extra work lands on p5, the port the in-lane levels already fill
with shuffles. That leaves it ~50 % above the vertical binary kernel, which wins precisely
because its input representation makes the index free and every fold a table choice.

### `simd/pointwise.rs` — products in the NTT domain (not optimised)

General Montgomery product of two batches (4 multiplies per slot, then a second multiplication
by 2^32 mod q to make the result exact), and batch x fixed element (3 multiplies per slot; the
element's Montgomery form and companion are stored as duplicated u32 so the broadcasts are pure
loads). `bench_all` also has the accumulating form `y += a_i o NTT(w_i)` with `vpmaddwd`
pairwise 16-to-32-bit accumulation, verified against the scalar reference.

## Machine facts that drove the design (`tools/ubench`, `tools/membw`)

On this Tiger Lake core every 512-bit 16-bit multiply (`vpmullw / vpmulhw / vpmulhrsw /
vpmaddwd / vpdpwssd`) issues only on port 0 at one per cycle (ymm: two per cycle on p0/p1);
`vpaddw / vpsubw / ternlog / masked ops` on p0 or p5; every shuffle (`vpermb / vpshufb /
vpermq / vpunpck* / vshufi64x2`) on p5 at one per cycle; `vpermw` zmm and `vpermi2*` are two
uops; `vpminuw`, shifts and `vgf2p8affineqb` are p0; `vpbroadcastw` from memory costs a p5 uop
but `vpbroadcastd` is a pure load; `kmov k, m` is a p5 uop. Loads two per cycle, stores one.
So for this instruction mix cycles ~= max(p0 uops, ALU uops / 2 + ~0.26 per zmm store), and
the whole game is the multiply count. `tools/ubench/ports.c` shows that the ~17-28 % of
flexible adds that land on the saturated port 0 is caused by the adds *depending* on 5-cycle
multiplies (queue-occupancy feedback in the allocator), that a software-pipelined
1-multiply : 1-add order recovers at most 2.4 %, and that ymm is 36 % slower. Single-core
DRAM: 37 GB/s non-temporal stores, 14 GB/s regular stores, 19.5 GB/s reads.

**Branches and front end are not a factor.** `perf stat` on the kernel-only loops
(`src/bin/kernel_loop.rs`, the tool for `perf stat` / `perf record`): 14-31 branches per
polynomial with 0.1-0.3 mispredictions (fixed trip counts; bad-speculation 1.2-1.6 % of slots),
99+ % of uops delivered from the decoded-uop cache, front-end bound 1.6 %, machine clears and
microcode switches in the noise; top-down puts ~48 % of issue slots at "retiring" and ~49 % at
"back-end bound" for all three kernels, i.e. the two-vector-ALU-port limit and nothing else.

## Remaining optimisation strategies

Measured or modelled on this core, roughly in order of value:

1. **Both primes from one transpose.** The index rows are q-independent, so computing 3889 and
   9721 together saves the 31-cycle transpose for the second prime (~4 % of the two-prime
   total). Easy; not done because the API is per prime.
2. **Incomplete NTT.** Stopping one level early (slots of degree 3, `Z_q[X]/(X^3 - c)`) removes
   1944 of the 6480 multiply-port uops per batch (~30 % of the kernel) at the price of a 3x
   more expensive pointwise product: a win for the transform alone, roughly break-even for
   NTT + one product. Depends on what the consumer needs.
3. **Hand-scheduled radix-3 levels in asm** (1 multiply : 1 add software pipeline): 2.4 %
   measured in isolation; LLVM does not preserve such orders from intrinsics.
4. **Fewer stores per butterfly** by keeping levels 4-6 in registers: ~3 %, needs 34+ live zmm,
   so only in hand-written asm.
5. **q = 9721's Barretts** cost ~40 cycles per polynomial (12 %); this is inherent to a 14-bit
   prime in 16-bit lanes (2^15 / q = 3.37). 3889 is the only fully splitting prime below 2^13.
6. **Multithreading** was excluded by design; the kernel is compute-bound with the output stream
   hidden, so it would scale across cores until DRAM (37 GB/s) saturates at ~2 cores.

## Layout of the crate

    src/params.rs               ring constants, twiddle tables, Montgomery/Barrett constants (const-evaluated)
    src/types.rs                BinaryPoly, BinaryBatch32, Batch32, RingElement
    src/scalar.rs               exact reference: lift, product mod Phi_1944, NTT, evaluation
    src/simd/vertical_bin.rs    the binary kernel (LUT + folding), drivers (materialised / streamed)
    src/simd/transpose.rs       BinaryPoly x 32 -> nibble / index rows (GFNI + VBMI)
    src/simd/vertical_gen.rs    generic-input vertical kernel
    src/simd/horizontal_gen.rs  generic-input horizontal kernel (HBatch4)
    src/simd/pointwise.rs       slot-wise Montgomery products
    src/perf.rs                 perf_event_open counters (cycles, instructions, uops, ports 0/1/5)
    src/bin/bench_all.rs        the headline benchmark;  src/bin/bench_*.rs  per-kernel benchmarks
    src/bin/kernel_loop.rs      one kernel in a tight loop, for perf stat / perf record
    tests/*.rs                  correctness;  tools/  C microbenchmarks (ports, instruction table, DRAM)
    DESIGN.md                   design notes: tree, arithmetic, bounds, port facts, kernel APIs

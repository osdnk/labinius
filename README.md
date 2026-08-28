# bin-ntt

AVX-512 forward NTT over the 1944-th cyclotomic ring

    R_q = Z_q[X] / (X^648 - X^324 + 1),    1944 = 2^3 * 3^5,    q in {3889, 9721},

for **binary inputs given as a stream of F162 elements**: four consecutive elements a, b, c, d
of `bin_fields::scalar::F162` (the type used by binius64-f162, `F162([u64; 3])`, 162 bits)
are read as one ring element by plain interleaving,

    coefficient of X^{4m+k}  =  bit m of element k        (k = 0..4 for a, b, c, d;  m = 0..162),

so 2^20 F162 elements are 2^18 ring elements with 0/1 coefficients. Both primes are 1 mod 1944,
so R_q splits into 648 linear factors and the transform is complete. Single-threaded, tuned for
one specific core (Intel i7-11850H, Tiger Lake), signed 16-bit lanes throughout, Rust stable.
All bit-shuffling from the F162 byte layout to the kernel's input is inside the measured time.

## Headline: 2^20 F162 = 2^18 ring elements, one core

Input: a plain `&[F162]` (25 MB). Output: a preallocated `Vec<Batch32>` (8192 batches of 32
ring elements, 340 MB per prime, written with non-temporal stores). Best of 3, pinned to one
core, hardware counters via `perf_event_open`, machine otherwise idle:

| q    | driver                                        | total ms | cycles / ring element | cycles / F162 | instructions / ring element | uops / ring element (p0 / p5) |
|------|-----------------------------------------------|---------:|----------------------:|--------------:|----------------------------:|------------------------------:|
| 3889 | `ntt_f162` (materialised)                     | **20.9** | **363**               | 91            | 749                         | 753 (280 / 271)               |
| 9721 | `ntt_f162` (materialised)                     | **23.4** | **382**               | 96            | 809                         | 808 (315 / 296)               |
| both | `ntt_f162_2q` (one front end, two primes)     | 42.5     | 355 per prime         | 89            | 753 per prime               | 756 per prime                 |
| 3889 | `ntt_f162_stream` (closure per batch, no output written) | 19.9 | 333            | 83            | 749                         | 748                           |

Cycles are the robust number: this laptop's clock under AVX-512 load moves between 4.0 and
4.6 GHz with temperature, so milliseconds vary by ~10 % between runs (the same binary measured
22.5 / 24.9 ms at 4.1 GHz). Components, cache-resident: F162 bit-slicing 32.4 cycles per ring
element (8.1 per F162, port-5 bound), kernel 280 / 318 (q = 3889 / 9721), together 311 / 349;
the rest of the headline is the DRAM streams, and the 340 MB output is fully hidden behind the
non-temporal stores (streamed 333 vs materialised 363).

For scale: 650 cycles per polynomial (an expert estimate for a generic-input NTT of this size on
this core) is ~38 ms. Multiplying in the NTT domain on the same 2^18 elements —
`y = sum_i a_i o NTT(w_i)` with 2^18 distinct NTT-domain `a_i` streamed from DRAM — takes
38.9 ms / 672 cycles per ring element (q = 3889); the product itself is ~100 uops per element,
the rest is the extra 340 MB of reads.

Reproduce: `cargo run --release --offline --bin bench_f162 -- <cpu>`.

Comparison kernels (generic i16 input, cache-resident cycles per polynomial): vertical generic
423 / 467, horizontal generic (Gregor Seiler's 4-polynomials-per-zmm layout) 540 / 587; both
under the 650 yardstick, both ~95 % multiply-port bound.

## Building and testing

`.cargo/config.toml` sets `-C target-cpu=native`; the kernels need AVX-512
F/BW/VL/VBMI/VBMI2/VNNI/GFNI and are tuned for Tiger Lake only. The `bin-fields` dependency is
pinned to the git revision binius64-f162 resolves to and is vendored in the cargo git cache, so
`--offline` works.

    cargo test --release --offline     # 49 tests: scalar reference, every kernel and driver vs
                                       # the reference slot for slot, bounds via i32 shadow
                                       # models, products, bit-exact driver equivalences
    cargo run --release --offline --bin bench_f162 -- 2

## What is computed

`psi` is the smallest primitive 1944-th root of unity mod q (7 for 3889, 17 for 9721),
`omega = psi^648`, `zeta6 = psi^324` (so `zeta6^-1 = 1 - zeta6`). The transform is a vanilla
mixed-radix Cooley-Tukey, no twisting, on the fixed tree

    level 0:     X^648 - X^324 + 1 = (X^324 - zeta6)(X^324 - zeta6^-1)       (Phi_6 split)
    level 1, 2:  radix 2   X^n - psi^e = (X^{n/2} - psi^{e/2})(X^{n/2} + psi^{e/2})
    level 3..6:  radix 3   X^n - psi^e = prod_s (X^{n/3} - psi^{e/3} omega^s)

with child s of a sub-ring stored at block offset s*n/p ("tree order"), so slot j of the output
holds `a(psi^SLOT_EXP[j])` (`params::SLOT_EXP`, a permutation of the units mod 1944). The
scalar reference `scalar::ntt` defines the order; `f162::lift4` defines the lift; every kernel
and driver is tested slot for slot against `scalar::ntt(lift4(..))`, and
`scalar::ntt(a*b mod Phi) == NTT(a) o NTT(b)` is checked through the SIMD products.

Output values are lazily reduced signed residues stored as `i16`: |v| <= 7.5 q for q = 3889
and <= 2.31 q for q = 9721 (`vertical_bin`, `vertical_bin_asm`); `RingElement::normalized`
maps to [0, q).

A remark on the lift: with Y = X^4 the ring element is a(Y) + X b(Y) + X^2 c(Y) + X^3 d(Y),
and psi^{4u} is a primitive 486-th root of unity, so NTT(e)[u] = sum_k psi^{uk} A_k(u mod 486)
where A_k is the length-162 transform of element k over Phi_486(Y) = Y^162 - Y^81 + 1. That
"four length-162 NTTs plus a radix-4 recombination" view costs exactly the same 6480
multiply-port uops per 32 ring elements as the kernel below (the multiply count of the tree is
layout-invariant, and the binary trick removes the same two levels either way), so the kernel
was kept and only the front end was rebuilt for the F162 layout. (The sign-alternating lift
Z -> -X^4 would make each component a ring homomorphism from Z[Z]/Phi_243; the plain interleave
is the specified lift.)

### Types

* `bin_fields::scalar::F162` — the input (`[u64; 3]`, bit 64k+j of limb k = coefficient of
  x^(64k+j), top 30 bits of limb 2 zero, 24-byte stride).
* `Batch32 { v: [[i16; 32]; 648], representation }` — 32 ring elements in the "vertical"
  layout: `v[j][p]` is slot (or coefficient) j of element p, so one 512-bit vector is one slot
  of all 32 elements. 64-byte aligned, 41 472 bytes. Lane p of batch b is ring element 32 b + p,
  i.e. F162 elements 4(32b + p) .. +3.
* `RingElement { v: [i16; 648], representation }` — one element, rokoko-style flat array plus
  representation tag; `Batch32::get / set`.
* `BinaryIndex32` — the kernel's input: 162 rows of 64 bytes, `row[i][2p] = n`,
  `row[i][2p+1] = 16 + n`, n = the 4-bit nibble (c_i, c_{i+162}, c_{i+324}, c_{i+486}) of ring
  element p, which is exactly the `vpermb` index the fused first two levels consume.
* `BinaryPoly { bits: [u64; 11] }` — a plain 648-coefficient 0/1 polynomial (bit i = coefficient
  i), the coefficient-domain form of a lifted ring element used by the tests and the comparison
  kernels' benches; `transpose.rs` slices it into `BinaryIndex32` with the same GFNI machinery.
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
every value is a fixed combination of the four input bits (c_i, c_{i+162}, c_{i+324},
c_{i+486}); so both levels are replaced by one lookup per output vector in a 16-entry table,
and the front end produces exactly the lookup index. Because every intermediate value of the
tree is consumed in exactly one role, the twiddles of level 2 and of level 3 are pre-multiplied
into the tables ("folding"), which removes all multiplications of level 2 and two of the three
of level 3 at the cost of extra lookups on the otherwise idle shuffle port. Multiplications
drop from 3564 per ring element (generic) to 2160; the rest is the same radix-3 machinery as
the generic kernel. It stops there: folding the level-4 twiddles too triplicates the lookups
(each level-3 output feeds a different level-4 twiddle) and, worse, leaves the folded inputs at
3q instead of 0.75q, which breaks the 2^15 budget for both primes — the Barretts that then
become necessary cost more than the multiplies saved.

## Strategies taken in each kernel and driver

### `simd/transpose_f162.rs` — F162 stream -> index rows (32.4 cycles per ring element)

Write `M[k][m]` for the 32-bit mask {bit m of F162 element 4p + k : p = 0..32}. Coefficient c of
ring element p is bit c/4 of element 4p + (c mod 4), so the four planes of index row i = 4t + s
are `M[s][t], M[s+2][t+40], M[s][t+81], M[s+2][t+121]` for s = 0, 1 and
`M[s][t], M[s-2][t+41], M[s][t+81], M[s-2][t+122]` for s = 2, 3. Four phases, all shuffles
and GFNI, no mask registers:

1. **qword transpose.** A phase-1 row is a pair of ring elements (j, j+8): 24 qwords, exactly
   three 8x8 transposes (`vshufi64x2` / `vpunpck*qdq`) with no wasted column.
2. **GFNI bit transpose + interleave.** One `vpermb` and one `vgf2p8affineqb` (identity matrix
   as the *vector* operand) per column give the 8-element mask of every bit; a byte/word/dword
   unpack tree over the four element groups writes the planes directly in the
   (k, k+1)-interleaved u64 order the rows need.
3. **4 x 8 qword transpose** so the four planes of a row pair are 32 contiguous bytes.
4. **mask -> index rows.** One `vpermb` lays that group out as the *matrix* operand of a second
   `vgf2p8affineqb` whose rows 0..3 are the four planes of row A in qwords 0..3 and of row B in
   qwords 4..7, so a single affine emits both rows' nibbles; per row one `vpermb` duplicates each
   byte and one `vpternlogd` masks the nibble and sets the +16 — three port-5 uops per two rows.

Port-5 bound (25.8 of 32.4 cycles). Measured and rejected: replacing phase 3 by merge-masked
loads (34.6 cycles: a masked `vmovdqu64` costs ~1 extra ALU uop on this core), software
prefetch of the input (noise).

### `simd/vertical_bin.rs` / `simd/vertical_bin_asm.rs` — the binary kernel (280 / 318 cycles)

* **Layout.** Vertical: one zmm = one slot of 32 ring elements, so no shuffle is ever needed
  for a butterfly and every twiddle is a broadcast constant. The batch (648 vectors, 41 KB) is
  processed depth-first per 162-block and 81-block, so the live working set is ~10 KB and stays
  in L1.
* **Levels 0-3.** One `vpermb` per output vector on byte-split 16-entry tables (low halves at
  index n, high halves at 16 + n; `vpermw` would cost an extra port-0 uop per lookup, ~30 cycles
  per element), with the level-2 and level-3 twiddles folded in: 10 tables per 162-block
  (40 in all, 1280 bytes, memory operands). For pair positions i < 27 the level-2 butterfly is
  2 lookups + 2 adds; for 27 <= i < 81 the two children need different folds, 4 lookups + 2 adds.
  Level 3 keeps only the omega multiplication per triple.
* **Levels 4-6** (`vertical_bin_asm`, the production kernel): one `asm!` block per 27-block keeps
  all 27 vectors in zmm0-26 across the three levels — loaded once, stored once instead of three
  times (648 loads and 648 stores per batch removed) — with a `[9 multiplies + 4 subs of
  butterfly k][6 adds of butterfly k-1]` block schedule that lowers the share of flexible adds
  the allocator sends to the saturated port 0 from ~27 % to ~23 %. Bit-identical to the
  intrinsics version in `vertical_bin.rs` (tested), 296 -> 280 (3889) and 330 -> 318 (9721).
  Measured alternatives: natural order 288 / 324, 1-mul:1-add pipeline 283 / 318, all loads up
  front or all stores deferred 286 / 322-324.
* **Reduction.** Table entries are centred (|T| <= q/2), so level-2 outputs are < q and level-3
  outputs < 3 q. For 3889 nothing else is needed (output <= 7.5 q declared, 4.4 q observed). For
  9721 one Barrett on the un-twiddled a0 input of levels 4, 5 and 6 caps every output at 2.31 q;
  skipping any one of them overflows.
* **Compiler workarounds.** stdarch's `_mm512_mulhi_epi16` lowers through a
  `vpmovsxwd / vpmovdw / vinserti64x4` round trip that LLVM rematerialises inside the hot loops;
  `vpmulhw` and the twiddle broadcasts are emitted with `asm!` (`pure`, `nomem/readonly`).
* **Tried and rejected.** Folding omega or the level-4 twiddles into the tables (see above);
  fusing level 4 into the lookup loop (34+ live registers); ymm arithmetic (36 % slower: the
  multiplies then also occupy p1 and the adds follow them); non-temporal stores for
  cache-resident output.
* **Where it stands.** Static port floor 256 / 287 cycles per element; measured 280 / 318 with
  the multiply port 95-96 % busy. `perf stat` top-down on the kernel loop: ~48 % of issue slots
  retiring, ~49 % back-end bound (the two vector ALU ports), 1.6 % front-end, 1.2-1.6 %
  bad speculation — no branch, decode or microcode issue to fix.

### `simd/ntt_f162.rs` — drivers

`ntt_f162` slices 128 F162 into one `BinaryIndex32` scratch, runs the kernel with
non-temporal stores on the last level, repeats per batch. `ntt_f162_2q` runs both primes off
one slicing (-21 cycles per transform). `ntt_f162_stream` hands each batch to a closure.
Driver variants measured and rejected: huge pages for the output (TLB walks 0.31 -> 0 per
element, time unchanged — they hide behind the NT stores), producer-supplied index rows (the
extra 58 MB of DRAM reads cost more than the slicing saved), `sfence` per batch, batch
grouping, cached stores (+200 cycles), input prefetching (noise to +45 with the NTA hint),
consumer-side prefetch of `a_i` in the accumulate loop (-15 at 2^15 elements, +30 at 2^18).
Software-pipelining the slicer's phases into the kernel's radix-3 loops of the previous batch
(p5-bound work against p0-bound work) gained 14 cycles in a prototype and is listed under the
remaining strategies.

### `simd/vertical_gen.rs` — generic-input vertical kernel (423 / 467)

Same layout and arithmetic, any i16 input with |x| <= q. Levels 0 and 1 fused into one radix-4
pass over the 648 vectors (41 KB streamed once from L2), levels 2 and 3 fused over groups of 6
vectors, then one pass per 27-block for each of levels 4, 5, 6; zero shuffles; Barretts: one
level for 3889 (a0 of level 5, output <= 3.40 q), for 9721 inside the first pass and on the a0
of levels 3-6 (output <= 2.12 q). Radix-9 / radix-27 fusion measured and rejected (439-468 vs
423: multiply-port bound, longer dependency chains); the variants stay behind
`ntt_gen_batch32_plan`. Out of cache: one prefetch line per level-4/5/6 butterfly for the next
batch is worth 23 %; 162 prefetches at once are 18 % slower than none. 2^18 in place: 527 / 542
cycles, ~22 GB/s read+write.

### `simd/horizontal_gen.rs` — Gregor's layout, generic input (540 / 587)

`HBatch4`: `v[r][8p + j]` = coefficient r + 81 j of polynomial p — one polynomial per 128-bit
lane, the 8 coefficients of stride 81 inside a lane, 81 registers for 4 polynomials (5 KB).
Levels 0-2 are in-lane and fused per register (`vpshufd` 0xEE/0x44 and 0xF5/0xA0, `vpshufb`
for level 2; `out = v + c * u` with a per-lane twiddle vector, both halves multiplied): 2
shuffles + 3 multiplies + 2 add/sub per register per level, 225 of the 540 cycles. After level
2 lane position `j = 4 s0 + 2 s1 + s2` carries the tree's 81-block k = j, so the slot order is
`81 j + r` and `HBatch4::get` returns tree order. Levels 3-6 are register-to-register radix-3
passes on r (strides 27, 9, 3, 1), one pass per level with three independent triples per
iteration (fusing levels: 566 vs 542); the 40 per-lane (zeta, zeta^2) constant vectors are
hoisted by hand. Reduction: none for 3889 (output <= 7.69 q), a Barrett on the a0 of every
radix-3 level for 9721 (2.12 q). Out of cache: `prefetcht0` per register of the first pass,
772 -> 633 cycles. Rejected: sharing the level-1 multiply across register pairs (548 vs 541),
`black_box` on the table pointer.

### Why there is no horizontal binary kernel

The lookup trick needs, in every output lane, an index made of four bits of that lane's
polynomial. In the vertical layout that index is the input row (one nibble per element per
lane), so a lookup costs one `vpermb` and nothing else, and folding a twiddle is merely choosing
a different table. In the horizontal layout the four bits sit at lane positions j, j+2, j+4, j+6
of the same 128-bit lane, so every register first needs an in-lane bit gather (~4-6 uops) to
build the index, the four sub-rings sharing a lane need four different tables (`vpermi2b`,
2 uops, or two `vpermb` plus a blend), the level-3 folding would need one table per (lane, role)
and double the gathers, and all of it lands on p5, the port the in-lane levels already fill
with shuffles. Levels 0-2 would drop from ~21 to ~14 uops per register — 540 -> ~450-470 cycles,
still ~60 % above the vertical binary kernel.

### `simd/pointwise.rs` — products in the NTT domain (not optimised)

General Montgomery product of two batches (4 multiplies per slot, then a second multiplication
by 2^32 mod q to make the result exact), and batch x fixed element (3 multiplies per slot; the
element's Montgomery form and companion are stored as duplicated u32 so the broadcasts are pure
loads). `bench_f162` also has the accumulating form `y += a_i o NTT(w_i)` with `vpmaddwd`
pairwise 16-to-32-bit accumulation, verified against the scalar reference. The second
multiplication can be avoided by scaling the tables by 2^16 so the NTT outputs come out in
Montgomery form, with 2^-16 (and 648^-1) absorbed by the inverse NTT's final scaling (see the
remaining strategies).

## Machine facts that drove the design (`tools/ubench`, `tools/membw`)

On this Tiger Lake core every 512-bit 16-bit multiply (`vpmullw / vpmulhw / vpmulhrsw /
vpmaddwd / vpdpwssd`) issues only on port 0 at one per cycle (ymm: two per cycle on p0/p1);
`vpaddw / vpsubw / ternlog / masked ops` on p0 or p5; every shuffle (`vpermb / vpshufb /
vpermq / vpunpck* / vshufi64x2`) on p5 at one per cycle; `vpermw` zmm and `vpermi2*` are two
uops; `vpminuw`, shifts and `vgf2p8affineqb` are p0; `vpbroadcastw` from memory costs a p5 uop
but `vpbroadcastd` is a pure load; `kmov k, m` is a p5 uop; a merge-masked load costs about one
extra ALU uop. Loads two per cycle, stores one. So for this instruction mix cycles ~= max(p0
uops, ALU uops / 2 + ~0.26 per zmm store), and the whole game is the multiply count.
`tools/ubench/ports.c` shows that the ~17-28 % of flexible adds that land on the saturated
port 0 is caused by the adds *depending* on 5-cycle multiplies (queue-occupancy feedback in the
allocator), that a software-pipelined multiply/add order recovers at most ~2.4 %, and that ymm
is 36 % slower. Single-core DRAM: 37 GB/s non-temporal stores, 14 GB/s regular stores,
19.5 GB/s reads. Branches and the front end are not a factor: `perf stat` on the kernel-only
loops (`src/bin/kernel_loop.rs`) shows 14-31 branches per polynomial with 0.1-0.3
mispredictions, 99+ % of uops from the decoded-uop cache, ~49 % of slots back-end bound and
~48 % retiring for all kernels.

## Remaining optimisation strategies

Measured or modelled on this core, roughly in order of value:

1. **Incomplete NTT.** Stopping one level early (slots of degree 3, `Z_q[X]/(X^3 - c)`) removes
   1944 of the 6480 multiply-port uops per batch (~30 % of the kernel), at the price of a 3x
   more expensive slot product: a win for the transform alone, a net loss for NTT + one product
   per element (2808 vs ~3670 multiplies). Depends on what the consumer needs.
2. **Lookup Barrett for q = 9721**: `vpmulhrsw` -> biased `vpaddw` -> `vpermb` on a 16-entry
   `-t*q` table -> `vpaddw` (1 p0 + 1 p5 + 2 flexible instead of 2 p0 + 1) on the 648 Barretts
   per batch; static p0 243 -> 223 per element on the prime that is genuinely p0-bound.
3. **Pipelined front end** (the slicer's phases interleaved with the previous batch's radix-3
   loops, p5-bound work against p0-bound work): -14 cycles per element in a prototype.
4. **Hand-scheduled level 3 and the lookup loop** in asm like levels 4-6: a few percent.
5. **Per-element bit permutation** as the slicer (VBMI2 funnel shifts + `vpmultishiftqb` +
   `vgf2p8affineqb` + `vpermb` on each F162 separately, then a byte transpose): ~10 uops per
   F162, perhaps 20 instead of 32 cycles per ring element.
6. **Montgomery-form outputs** (tables scaled by 2^16) so products need one multiplication and
   the factor is absorbed with 648^-1 in an inverse NTT.
7. For q = 9721 the mandatory Barretts cost ~40 cycles per element (12 %); inherent to a
   14-bit prime in 16-bit lanes (2^15 / q = 3.37). 3889 is the only fully splitting prime below
   2^13. Multithreading was excluded by design; the kernel is compute-bound with the output
   stream hidden, so it would scale across cores until DRAM saturates at ~2 cores.

## Layout of the crate

    src/params.rs               ring constants, twiddle tables, Montgomery/Barrett constants (const-evaluated)
    src/f162.rs                 the F162 lift (lift4, pack4, scalar index rows, random elements)
    src/types.rs                Batch32, RingElement, BinaryPoly (test/comparison input form)
    src/scalar.rs               exact reference: product mod Phi_1944, NTT, evaluation
    src/simd/transpose_f162.rs  F162 x 128 -> BinaryIndex32 (GFNI + VBMI)
    src/simd/transpose.rs       BinaryIndex32 and the GFNI bit-slicing of plain 648-bit polynomials
    src/simd/vertical_bin.rs    the binary kernel, intrinsics (LUT + folding), drivers
    src/simd/vertical_bin_asm.rs the binary kernel with the asm levels 4-6 (production kernel)
    src/simd/ntt_f162.rs        drivers for the F162 input (single prime, both primes, streamed)
    src/simd/vertical_gen.rs    generic-input vertical kernel
    src/simd/horizontal_gen.rs  generic-input horizontal kernel (HBatch4)
    src/simd/pointwise.rs       slot-wise Montgomery products
    src/perf.rs                 perf_event_open counters (cycles, instructions, uops, ports 0/1/5)
    src/bin/bench_f162.rs       the headline benchmark;  src/bin/bench_*.rs  per-kernel benchmarks
    src/bin/kernel_loop.rs      one kernel in a tight loop, for perf stat / perf record
    tests/*.rs                  correctness;  tools/  C microbenchmarks (ports, instruction table, DRAM)
    DESIGN.md                   design notes: tree, arithmetic, bounds, port facts, kernel APIs

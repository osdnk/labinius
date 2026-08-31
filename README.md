# bin-ntt

An Ajtai commitment, a folding step and their verifier over `R_648 = Z_q[X]/(X^648 - X^324 + 1)`,
the 1944-th cyclotomic ring, for a witness of binary ring elements carried as elements of
`F162 = GF(2)[x]/(x^162 + x^81 + 1)`. The witness is committed modulo any subset of
`2917, 3889, 4861, 9721, 12637, 17497, 19441`, one of which is the base — 3889 by default —
folded against short binary challenges of the subring `R_162 = Z_q[Z]/Phi_243(Z)`, 28 of whose
162 coefficients are 1, and the folded opening is checked against the multilinear extension of
the same witness over `F162` — one AVX-512 thread throughout. The opening is either sent in the
clear or recursed into a single LaBRADOR proof of 80 KB.

## The flow

```rust
use bin_ntt::{Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const WITNESS_SEED: [u8; 32] = [0xC7; 32];

let params = Params::basic();
let public_parameters = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
let witness = Witness::random(&params, WITNESS_SEED);
let mut prover = Prover::new(&public_parameters);
let verifier = Verifier::new(&public_parameters);

let (commitment, opening) = prover.commit(&witness);

let mut transcript = Transcript::new(b"bin-ntt/reference");
let evaluation_point = verifier.derive_evaluation_point(&mut transcript, &commitment);
let claimed_value = witness.mle_evaluate(&evaluation_point);
let row_evaluation = witness.row_evaluate(&evaluation_point);
let folding_challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);

let folded_witness = prover.fold(opening, &folding_challenges);
let folded_commitment = verifier.fold_commitment(&commitment, &folding_challenges);
let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &folding_challenges);

verifier
    .verify_evaluation(&evaluation_point, &claimed_value, &row_evaluation)
    .unwrap();
verifier
    .verify_folded_opening(
        &folded_commitment,
        &folded_witness,
        &evaluation_point,
        &folded_row_value,
    )
    .unwrap();
```

`verify_evaluation` checks `u . eq(p1) == t`; `verify_folded_opening` checks that the folded
witness `v` is short, that `A v` equals `sum_j c_j C_j` modulo every modulus, and that
`eq(p0) . (v mod 2)` equals `sum_j u_j (c_j mod 2)` over `F162`.

## The recursive opening

`Params::new(witness_log_len, column_log_len, extra_moduli, true)` — or `Params::with_base` with
a base other than 3889 — replaces the last three
messages by one LaBRADOR proof. The commitment the verifier receives becomes the Ajtai commitment
`T_Y` to the RNS residues of the matrix, the left expansion `u` becomes `T_u`, and `v` is never
sent: the prover commits to the rest of the witness as `T_R`, announces the exact squared norm of
every witness vector, takes the verifier's mask scalars, and proves everything at once.

```rust
let params = Params::new(18, 8, vec![Modulus::Q9721_FS_S], true).unwrap();
// public_parameters, witness, prover and verifier as above
let (commitment, opening) = prover.commit(&witness);            // commitment = T_Y

let mut transcript = Transcript::new(b"bin-ntt/reference");
let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
let claimed_value = witness.mle_evaluate(&point);
let row_evaluation = witness.row_evaluate(&point);
let left = prover.commit_left_expansion(&row_evaluation);       // left = T_u
let challenges = verifier.derive_folding_challenges(&mut transcript, &left);

let proof = prover
    .prove_opening(&mut transcript, opening, &challenges, &point, &left,
                   &row_evaluation, &claimed_value, &commitment)
    .unwrap();
verifier
    .verify_opening(&mut check, &commitment, &left, &point, &claimed_value, &challenges, &proof)
    .unwrap();
```

Each folding challenge is 28 distinct positions out of 162 drawn from the transcript's XOF and
rejected until its canonical infinity norm is at most 11: a set of size `2^104`, one attempt in
seven accepted, 0.73 ms for the whole 128. `prove_opening` returns
`Err(OpeningError::FoldTooLong { .. })` when the fold's squared norm exceeds its cap — about one
round in twenty — and the caller retries with fresh challenges.
`verify_opening` runs no field arithmetic of its own: `u . eq(p1) == t` and
`eq(p0) . (v mod 2) == sum_j u_j (c_j mod 2)` are two of the identities inside the proof.

## Runtime

`Params::basic()` — 2^18 `F162` = 2^16 ring elements of `R_648` in 128 columns, moduli 3889 and
9721 (the clear-text default; the wire trades commitment against folded witness at `columns`
versus `witness / columns`, and 2^7 columns sits at the optimum, while the recursion, whose cost
grows with the column length, runs at 2^8) — on one core of an i7-11850H, wall clock, median of 3 except the steps that consume what
they are given, which run once.

| step | ms |
|------|---:|
| **prover** | |
| `commit` | 13.86 |
| `row_evaluate` | 0.31 |
| `fold` | 6.01 |
| *total* | *20.19* |
| **statement** | |
| `derive_evaluation_point` | 0.28 |
| `mle_evaluate` | 0.34 |
| *total* | *0.62* |
| **verifier** | |
| `derive_folding_challenges` | 0.73 |
| `decode` | 2.00 |
| `fold_commitment` | 0.15 |
| `fold_row_evaluation` | 0.00 |
| `verify_evaluation` | 0.00 |
| `verify_folded_opening` | 0.65 |
| *total* | *3.53* |

The wire is 263.2 KB of commitment, 2.5 KB of row evaluation and 335.6 KB of folded witness —
601.3 KB in all, measured on the encodings themselves rather than assumed. `decode` is the
verifier reading all three back off the wire; everything after it in the table runs against the
decoded objects, so the round trip is on the binary's real path. The commitment is `ceil(log2 q)`
bits a slot and the row evaluation 162 bits an `F162`, both of which are the entropy of uniform
data; the fold is entropy-coded against its own histogram, which is where the 1.9x comes from
(see the last implementation note).

The same shape with the recursion on. `PublicParameters::from_seed` additionally inverts the key
rows, blocks them, converts the 221 184 key-time `phi` and the nine-bit pattern table to `polx`,
and picks the three commitment ranks (`kappa_Y = 11`, `kappa_u = 3`, `kappa_R = 8`) — 35 ms and
21 MB, paid once per key. The encoded witness is 18 LaBRADOR vectors, 10 752 polynomials.

| step | ms |
|------|---:|
| **prover** | |
| `commit`, including `T_Y` | 17.04 |
| `row_evaluate` | 0.30 |
| `commit_left_expansion` | 0.25 |
| `prove_opening` | 222.42 |
| — fold | 4.51 |
| — encoding | 21.25 |
| — `T_R` | 2.49 |
| — masks | 5.16 |
| — constraint `phi` | 2.37 |
| — statement build | 14.88 |
| — `labrador::prove` | 170.04 |
| *total* | *240.00* |
| **statement** | |
| `derive_evaluation_point` | 0.01 |
| `mle_evaluate` | 0.33 |
| *total* | *0.33* |
| **verifier** | |
| `derive_folding_challenges` | 1.61 |
| statement rebuild | 16.73 |
| — layout | 1.68 |
| — no-wrap bound | 2.24 |
| — constraint `phi` | 2.36 |
| — statement build | 7.88 |
| `labrador::verify` | 102.22 |
| *total* | *120.67* |

Both breakdowns are the wall clock of the run that produced the proof: `prove_opening` returns
its stage timings with the proof and `verify_opening` returns its own, so each side does its work
exactly once and the parts add up to the whole they were measured inside. The indented rows are
therefore cold — the prover's `masks` and `statement build` cost twice what a second, warm pass
at them would say.

The proof is 79.5 KB: `T_Y` 4.1 KB, `T_u` 1.1 KB, `T_R` 3.0 KB, the 18 announced norms 0.1 KB and
LaBRADOR's own 71.1 KB, against 601.3 KB in the clear. Per proof the constraint `phi` take 1 MB
on top of the key's 21 MB, plus 33 MB for the three mask rows; the peak resident set of one round
in each mode is 292 MB.

Against the first working version of the recursion, at the same shape and on the same core:
`commit` 49.6 -> 17.0, `prove_opening` 526.4 -> 222.4 and the verifier 287.8 -> 120.7 ms. Inside
LaBRADOR (300 -> 171 ms proving, 222 -> 103 ms verifying, transcript-neutral): the inner products
as a VNNI double schoolbook with two exact products per `vpdpwssd` and a per-call reduction
chunk (Lazer's random-walk chunk on the prover's honest data, the worst-case chunk wherever a
kernel reads the proof), the chain `phi`
handed over as nine int16 coefficients instead of 1 KB NTT images and collapsed in the
coefficient domain with `vpdpwssd` against the shifted quarternary challenge, one NTT per
destination run at the end (the aggregation 41 -> 13 ms, the key-time buffers 233 -> 20 MB,
the per-proof `phi` 81 -> 1 MB), the constant-term scaling deferred across the three lifts and
applied once, the JL collapse rewritten without register spills and with a prefetch ahead of
its strided reads, a batched refresh, every per-prime sweep blocked in 16-`polx` chunks, the pointwise,
scale and add passes of the `polx` kernels fused with their reductions, the constraint
aggregation walked source-major so that each shared `phi` buffer streams once for all the
constraints that alias it, the refresh after the constant-term collapse restricted to the ranges
it scaled, and the decomposition self-check made opt-in. On the crate's side: LaBRADOR's own stdout chatter, which a terminal charges at 48 ms of a proof and 27 ms of a
verification; `simple_verify`, which the honest prover has no reason to run (-60 ms); the residues
of `T_Y`, off the scalar inverse transform and onto the vectorised one (-32 ms); the diagonal
constraint order, which puts the key `phi` of a limb's four components in the last-level cache
together (-20 ms proving, -12 ms verifying); the encoding's transposes and gadget splits (-5 ms);
and the no-wraparound bound, summed once per public group instead of once per product (-2.5 ms).

`Prover::new` allocates and first-touches the 85 MB workspace and runs one commitment and one
fold to bring the kernels and the rejection tables up; `Prover::fold` hands the workspace back, so
a second `commit` allocates nothing.

## Keccak

`src/bin/keccak.rs` proves 482 keccak-f permutations — [binius64](https://github.com/IrreducibleOSS/binius64)'s
keccak example at `--message-len 65536`, pinned to the revision `32d8cd07` that
`binius64-f162` vendored — three ways: stock binius64, and this crate's commitment in place of
its BaseFold oracle with the recursion off and on.

binius64 builds the constraint system and the witness and packs the non-public trace as it
always does: two 64-bit words per `B128`, zero-padded to `2^18`. That vector lifts to `F162` by
zero-extension (`phi` carries the `beta` basis of `B128` onto `{1, X, ..., X^127}`) and is
committed here, in 128 columns for the clear-text opening and 256 for the recursion. binius64's reductions then run unchanged down to
the claim `w~(r) = s` on that trace, and the cross-field switch of `fields::crossfield` — the
128 partial evaluations, `r' ∈ F162^7` drawn after them, an 18-round sumcheck over `F162` —
turns it into `pi1~(r'') = opened`, which this crate's opening discharges.

One transcript throughout. binius64's `ProverTranscript` is the channel: the statement is
observed into it, the commitment is written into it before any challenge is drawn, and the
opening's own transcript is seeded from 32 bytes sampled off that channel after the switch, so
every challenge it draws is bound to everything before it. `r''` arrives most significant first
over the flat trace index, whose top 8 bits are the column and whose low 10 are the row, so the
leading 8 coordinates reversed are `p1` and the trailing 10 reversed are `p0`
(`EvaluationPoint::msb_first`).

The stock column runs binius64's own `Prover::prove` and `Verifier::verify` on the same circuit
and witness, at the example's defaults (`--log-inv-rate 1`, `--hash-suite sha256`), with the
per-stage numbers read off binius64's own `INFO` phase spans. Everything is one thread on core 3;
binius64's `rayon` feature is off by default, so its reductions are single-threaded too.

```
                                          stock    recursion    recursion
                                       binius64          off           on

SETUP (once, not per proof)
  circuit                               1059.31      1059.31      1059.31 ms
  commitment key and constraints         706.86       683.59       689.77 ms

PROVER
  witness                                  4.97         4.97         4.97 ms
  packing                                     —         0.65         0.72 ms
  commit                                  12.41        14.80        17.40 ms
  BitAnd check                            31.53        32.08        33.17 ms
  shift reduction                         73.68        73.90        73.47 ms
  ring-switch / cross-field switch         4.41        15.16        15.20 ms
  opening                                     —         8.83       179.12 ms
  rest                                     4.56         1.01         0.56 ms
  total                                  126.59       146.43       319.64 ms

VERIFIER
  read the commitment                         —         0.32         0.01 ms
  reductions                               0.37         0.44         0.40 ms
  ring-switch / cross-field switch         0.00         0.31         0.31 ms
  decode the opening                          —         1.66            — ms
  BaseFold / our opening                   0.46         1.69       103.69 ms
  wiring check (native)                   41.87        42.21        41.87 ms
  total                                   43.18        46.73       146.35 ms

SIZES
  binius64 LIOP                               —         5.84         5.84 KB
  cross-field switch                          —         3.12         3.12 KB
  commitment (wire form)                      —       263.25         4.12 KB
  opening                                     —       289.08        75.63 KB
  total                                  243.83       561.30        88.72 KB
```

`wiring check (native)` is `WiringEvalClaim::check_native`, which discharges the shift
reduction's last claim by evaluating the wiring multilinear over all 289 179 constraints from the
constraint system. It is per proof — it reads the challenges — and it dominates every verifier
column alike, because it is binius64's own step on binius64's own data; the reductions themselves
are 0.4 ms on both paths. It is written as a `par_iter`, so it is 42 ms only because
`binius-utils/rayon` is off and the process is pinned to one core.

The commitment sizes are the canonical wire form — `ceil(log2 q)` bits a slot, the bit-packing of
`wire`; `T_Y` travels on the tape as its `polx` image, which is 11.0 KB. The recursion-off
`opening` is `wire`'s too, measured on the bytes themselves: the claimed value at 21 bytes, the
row evaluation bit-packed and the folded witness entropy-coded, 289.08 KB against the 650.55 the
same three take at two bytes a coefficient over its 331 776. `decode the opening` is the verifier reading the last
two back off the wire, and the rows under it in that column check what it decoded. The stock
total is its whole proof tape.

## Running it

```
cargo run --release --offline          # taskset -c 3 to pin it, as the table above is measured
cargo run --release --offline --bin keccak
cargo test --release --offline
```

The host must have AVX-512 F/BW/VBMI/VBMI2/VNNI/GFNI, and the crate is built with
`-C target-cpu=native` (`.cargo/config.toml`).

## Changing the configuration

`src/main.rs` takes no arguments; the configuration is the block of constants at the top of the
file:

```rust
const WITNESS_LOG_LEN: u32 = 18;                      // 2^18 elements of F162
const COLUMN_LOG_LEN_CLEAR: u32 = 7;                  // 128 columns for the clear-text opening
const COLUMN_LOG_LEN_RECURSIVE: u32 = 8;              // 256 for the recursion
const EXTRA_MODULI: &[Modulus] = &[Modulus::Q9721_FS_S];   // plus the default base modulus 3889
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const WITNESS_SEED: [u8; 32] = [0xC7; 32];
const CPU: usize = 3;
```

`EXTRA_MODULI` is any subset of
`Modulus::{Q2917_Q_S, Q3889_FS_S, Q4861_Q_S, Q9721_FS_S, Q12637_Q_S, Q17497_FS_L, Q19441_FS_L}`
that leaves out the base (see the last implementation note for what each one costs); the suffix
names the family — `_FS_S` fully splitting below `2^14`, `_Q_S` quadratic-slot, `_FS_L` fully
splitting above `2^14`. `Params::basic()` is this same shape for library users, with the recursion
off. `Params::new(witness_log_len, column_log_len, extra_moduli, recursion)` builds it over the
default base 3889 and `Params::with_base(witness_log_len, column_log_len, base, extra_moduli,
recursion)` over any other; both refuse a column shorter than one 128-`F162` batch, fewer than two
columns, more columns than elements, a repeated modulus, or the base repeated among the extra
ones. `cargo run` prints both modes.

## Implementation notes

* **Vertical batch-of-32 layout.** A `Batch32` holds coefficient `j` of 32 polynomials in one
  512-bit vector, so every butterfly of the transform is a full-width i16 operation and no
  shuffle ever crosses a polynomial boundary.
* **The binary lookup trick.** The NTT is linear and the inputs are bits, so levels 0 and 1 of
  the tree collapse into one 16-entry table lookup per output vector, indexed by the nibble the
  bit-slicing front end already produces; the twiddles of the next levels are pre-multiplied into
  those tables. Montgomery products per batch of 32 drop from 3564 to 2160 — and to 1512 on the
  quadratic tree, where the level-3 twiddles fold in too if the level-2 butterfly is replaced by
  the sum it computes: three lookups and two adds per output row out of 108 tables, and levels 2
  and 3 between them are left with one multiplication instead of seven.
* **Per-modulus reduction schedules.** Each prime reduces only where its `2^15/q` head-room runs
  out: 2917 and 3889 need no reduction anywhere inside the binary kernel, 4861 and 9721 need a
  lookup Barrett on one level and 12637 on three, and each accumulator gets its own fold-back
  period from its own compile-time bound. The schedule is chosen by a `const` recursion over the
  exact bounds, and the folded-twiddle phase 1 above is used only where its wider intermediates
  still fit (which is why 12637 does not get it).
* **Above `2^14` the whole butterfly is reduced, not just its `a0`, and the reductions are
  port-balanced.** 17497 and 19441 are the same conductor-1944 tree as 3889 and 9721 — 648 linear
  slots, the same lookup tables, the same 27-row blocks — but a lane's budget is `2^15/q` = 1.87
  and 1.69. A Montgomery twiddle product never gets below `q/2`, since `mulhi(m, q)` is a full i16
  times `q`, so the three terms of `a0 + t1 + t2` already cost `1.5 q` before any slack: reducing
  only the untwiddled `a0`, which is all the kernels below `2^14` ever do, leaves `3 q` at level 3
  alone. `simd::vertical_bin_large` therefore reduces `a0`, `t1` and `t2` at every one of the four
  radix-3 levels, and 19441 also the `omega (t1 - t2)`: three and four reductions per butterfly,
  2592 and 3456 per batch, where the kernels below `2^14` place at most 432.

  Which reduction each of those sites gets is a port question, not a count. The shuffle-port
  lookup Barrett is 4 uops (`vpmultishiftqb`, `vpermb` and a `vpternlogd` index fix-up, two of
  them port-5 only) and leaves `0.53 q`; the two-multiply `vpmulhrsw` Barrett is 3 uops, two of
  them port-0 only, and leaves `0.60 q` at 17497 and `0.74 q` at 19441 because `round(2^15/q)` is
  2, an estimate with two significant bits. Nine of a butterfly's uops are already port-0 only —
  the three Montgomery products — and level 3's loop carries 12 `vpermb` per row on port 5, so
  neither port is the bottleneck by itself and the cheap-but-loose reduction is affordable exactly
  where port 0 has room. A `const` search walks all 3^12 placements against the exact bound
  recursion and scores the survivors on both ports: 17497 takes the `vpmulhrsw` form for both
  twiddle products at every level and for `a0` at level 3, 19441 has no head-room for `0.74 q`
  anywhere and stays on the lookup throughout. Every site 17497 gives the loose reduction has an
  input under 24576, where the two quotient estimates agree, so the kernel's output is
  bit-identical to the all-lookup schedule it replaces and 6 % cheaper; the last 4 % comes from
  software-pipelining levels 5 and 6, whose nine register-resident rows leave only three
  independent butterflies until one sub-ring's level 5 is run in the shadow of the previous one's
  level 6. Together, 459 to 415 cycles per ring element. The same argument, one level looser,
  gives the generic-input kernel and its inverse — which keep the uniform lookup schedule, being
  off the hot path. The other answer — unsigned lanes in `[0, q)`,
  where the head-room is `2^16/q` and a radix-3 sum fits — was written out and measured, and it
  is *slower*: 5.96 ns per butterfly against the signed form's 3.68 at 17497 and 6.08 against
  4.40 at 19441, 1.6x and 1.4x, because a Shoup product with its conditional subtract is five
  dependent uops where a Montgomery one is three. It would also need the transform centered
  before `vpdpwssd`, the fold and the decomposition could read it.
* **Block-fused base multiplication.** The kernel hands out 27 finished slot vectors at a time
  (18 on the quadratic tree) and a hook multiplies them against `A` and accumulates on the spot,
  so the transform output never reaches memory and only `A` streams. A quadratic-slot modulus
  carries three sums per two rows — the quadratic product has bilinear rank 3 — formed by
  Karatsuba where the lazily reduced operand still fits i16 and schoolbook where it does not.
  Prefetching the `A` a block will read one batch later is worth 0.5 ms per limb when `A` is a
  real stream and costs 0.2 ms when a key has enough columns for it to stay in cache, so the
  batch loop is compiled both ways and picks on the footprint.
* **The fold-down is vectorised.** A key with `r` columns folds its accumulators down `r` times,
  not once, so the eight lanes of a slot are summed in three `vpermt2d` stages and reduced sixteen
  at a time through the double unit rather than in scalar i64: 45 to 7 cycles per ring element
  for a splitting limb and 32 to 16 for a quadratic one.
* **The fold is one 85 MB read.** The commitment writes the base-modulus transform of the witness
  out as it goes, with non-temporal stores hidden behind the transform, so the fold never
  transforms anything again: it is one `vpmaddwd` per slot vector and pair of columns over a
  single pass of that buffer, and the result comes back to coefficients as a genuine small
  integer vector.
* **Any modulus can be the base.** The base limb is the one the commitment keeps the transform
  of, the fold accumulates over and the folded witness comes back from; `Params::with_base` picks
  it, and all three families cost the same one `vpmaddwd` per slot vector. A quadratic-slot base
  is *not* a degree-2 product: a challenge enters `R_648` as `c(-X^4)`, and `X^4` is a constant
  modulo a leaf `X^2 - psi'^u`, so its leaf image is the scalar `c(-theta^v)` and the fold is the
  same row-wise multiply once that scalar is written into both rows of the leaf. What does change
  is the fold-back period, which each prime's `|W| |c|` fixes at compile time: 64 batches for
  2917, 32 for 3889 and 4861, 16 for 9721, 8 for 12637 and 4 for the two above `2^14`, whose
  wider `|A|` is what makes them the only ones to pay for it. At the basic shape the fold is
  4.46 ms over 2917, 4.47 over 3889, 4.48 over 4861, 4.51 over 9721, 4.70 over 12637, 4.89 over
  17497 and 4.92 over 19441 — a 10 % spread, all of it the fold-backs. Coming back to coefficients uses
  that tree's inverse transform: `vertical_gen`'s below `2^14`, `vertical_gen_large`'s above it,
  and `vertical_gen_quad`'s for the quadratic tree, whose reduction placement is the same kind of
  `const` search as the forward kernel's (1080 lookup Barretts per batch of 32 for 2917 and 4861,
  1836 for 12637) and whose whole `1/324` normalisation, corrected by the Phi_6 determinant, sits
  in the three level-0 constants.
* **A reference kernel next to the generated one.** `simd::vertical_bin` is the pure-intrinsics
  implementation of the same split-tree binary kernel — one Rust expression per butterfly, and the
  two-multiply Barrett on the untwiddled `a0` of levels 4, 5 and 6 for q = 9721 — kept out of the
  production path but checked against `vertical_bin_asm` on every test run: bit-identical for
  q = 3889, equal modulo q for q = 9721, where the `asm!` kernel's lookup Barrett leaves smaller
  representatives.
* **Word-sliced `F162` arithmetic.** The binary side is one dot product over `F162` per step,
  computed with `fields::f162`'s word-sliced kernels — limb `k` of 8 consecutive elements in one
  `zmm`, 12 unreduced `clmul` products per block and a single reduction at the end of the whole
  product. The witness never leaves its own layout; it is transposed 8 elements at a time inside
  the loop.
* **The residues are a batched inverse transform.** `T_Y` opens the RNS residues of the
  commitment's columns, which are the four `R_162` components of each column read in the `Z`-basis.
  Recovering them means rebuilding the 648 slots of the `R_648` element from its four components
  and inverting the big transform; both used to be scalar, with a modular exponentiation per
  twiddle, and cost 36 ms of a 50 ms `commit`. The recombination `E_t = sum_k psi^{v k} i^{tk} Y_k`
  is now one table of `162 * 16` multipliers built with `162` exponentiations, and the inverse
  transform is the crate's own `intt_gen_batch32` over 32 columns of a `Batch32` at a time: 5 ms.
  A quadratic-slot limb goes the same way now that its tree has a batched inverse too — the
  recombination is then the class butterfly `y mod (X^2 -+ psi'^v) = Y_k -+ psi'^v Y_{k+2}` out of
  a table of 162 constants — which takes that limb from 5.5 ms to 1.0 ms at 256 columns.
* **The block equations are emitted diagonal by diagonal.** A key `phi` buffer is read by exactly
  two of the eighty limb constraints — the two output components whose `(k, twist)` it is, at the
  same diagonal — and LaBRADOR's `aggregate_sparsecnst` streams `phi` constraint by constraint,
  1.4 GB of it per pass. In chain order those two reads are eighteen constraints and 60 MB apart;
  interleaving a limb's four components puts them four constraints apart, where the limb's whole
  diagonal (12 MB) is still in the last-level cache. Worth 20 ms of proving and 12 ms of verifying,
  and nothing else changes: any order is sound, and the two parties run the same function.
* **The honest prover does not check its own witness.** `labrador::prove` used to call LaBRADOR's
  `simple_verify` first, which converts the whole witness to `polx` and evaluates all 183
  constraints against it — an aggregation pass' worth of work, 60 ms, to confirm what the encoding
  has already asserted coefficient by coefficient. `prove_verified` keeps it for the tests.
* **The library is quiet.** LaBRADOR prints a page of statement and proof-size chatter per
  recursion level; on a terminal that is 48 ms of a proof and 27 ms of a verification. The shim
  redirects fd 1 to `/dev/null` around every entry point and restores it after, so the crate's own
  timing table is all that reaches the console. `BIN_NTT_LABRADOR_VERBOSE=1` puts it back.
* **What the recursion precomputes.** A chain constraint's `phi` over a run of key rows is a
  function of the commitment key alone, so `recursion::setup` inverts the key rows once, blocks
  them, and converts all 432 buffers of `n` `polx` per limb — one per `(component, twist, chunk,
  diagonal)` — at key time; every proof then aliases them by pointer, and so do the quotients'
  constant multipliers, the carry weights and the three commitment keys. Two blocks that would
  hold the same `phi` are one buffer: the products of a chain are cut into the longest runs a
  single block covers, and a run is identified by its *group* — the key rows of one
  `(limb, component, twist)`, the folding challenges, or one set of binary lifts — so the four
  output components of a limb share the key's buffers and all ten chains share the challenges'.
  What is left per proof is 81 MB of `phi`: the challenges, converted in bulk, and the `eq` lifts,
  which are not transformed at all. Every shift the encoding applies is a multiple of the
  sub-chunk length, so one sub-chunk of a binary lift is a signed sum of at most three nine-bit
  windows of the lifted element, and `chunk::taps` reads that decomposition off the unit elements;
  the `polx` is then one lookup and at most two adds in a 512-entry table instead of eight NTTs.
* **Why `LOGQ = 48`.** The residues are witness vectors as they are, with no digit
  decomposition, so the witness's total squared norm is `2^40.4` — dominated by the four 9721
  vectors at `2^38.3` each. Dachshund's exact-norm proof wants that below `2^(LOGQ-3)`, which
  `LOGQ = 40` (`2^37`) does not clear and `LOGQ = 48` (`2^45`) clears with room. Digit-decomposing
  the residues to fit 40 costs more than the wider modulus does, and at 48 the no-wraparound
  bound of the plan's section 5 has margins of 1992x on 3889 and 552x on 9721 against `Q/2 = 2^47`.
  A 19441 limb is the widest the recursion takes: its four residue vectors are `2^40.3` each, the
  no-wrap bound is `2^38.9` and the margin 270x, and its carries reach `2^28.1` of the `2^31` that
  four base-256 digits give them — the same gadget 9721 uses. A residue coefficient is then 9720,
  which is what fixes `COEFF_LIMIT`; a public sub-chunk is twice a centred key row, so
  `BLOCK_LIMIT` is 19440, and the `i16` dot product of `recursion::chain` widens its `i32` lanes
  every four `vpmaddwd` instead of every eight (worth 2 % of the block sums, nothing of a proof).
* **The moduli, quantified.** Committing 2^18 `F162` in 256 columns modulo the base 3889 alone
  takes 7.5 ms; each extra modulus adds its own transform, base multiplication and fold-down on
  the shared front end (wall clock, one core, median of 15; the cycle columns are per ring
  element, cache-resident, and are a `perf` count of a fixed number of repetitions rather than a
  sample statistic). The name's suffix is the family: `_FS_S` fully splitting below `2^14`, `_Q_S`
  quadratic-slot, `_FS_L` fully splitting above `2^14`.

  | modulus | added to `commit` | transform | base multiplication | fold-down |
  |---------|------------------:|----------:|--------------------:|----------:|
  | `Q2917_Q_S`   | +5.87 ms | 247 cycles | 74 cycles | 17 cycles |
  | `Q4861_Q_S`   | +6.03 ms | 254 | 74 | 17 |
  | `Q3889_FS_S` (the default base) | 7.49 ms | 280 | 58 | 7 |
  | `Q9721_FS_S`  | +6.26 ms | 304 | 58 | 7 |
  | `Q12637_Q_S`  | +6.93 ms | 291 | 75 | 16 |
  | `Q17497_FS_L` | +8.48 ms | 415 | 58 | 9 |
  | `Q19441_FS_L` | +10.08 ms | 514 | 58 | 7 |

  The front end costs 31 more cycles per ring element and is paid once however many moduli
  follow. A quadratic-slot modulus runs a shorter tree — one radix-2 level fewer, and for 2917
  and 4861 two thirds of the Montgomery products of a splitting one — but pays for it in the base
  multiplication, where three sums per two rows cost 16 cycles more than one sum per row, and in
  the fold-down, which has three accumulators to reduce instead of one. 2917 is the cheapest
  modulus there is, 4861 and 9721 sit within 3 % of each other, and 12637 — three Barretts inside
  the kernel, and an accumulator it has to fold back every batch — is 20 % dearer than 2917.

  17497 and 19441 buy 4.2 more bits of modulus for 1.4x and 1.7x of 9721's transform: three
  Barretts per radix-3 butterfly and, for 19441, four, at every one of the four levels (see the
  implementation note above), which is 2592 and 3456 reductions per batch of 32 against 9721's
  216. Nothing else about them costs more — the base multiplication is the same `vpdpwssd`
  accumulation and measures the same to a tenth of a cycle, and the fold-down is the same, except
  that 17497 has to fold its accumulator back twice inside `hsum8` because
  `2^16 mod 17497 = 13045` is the one `R` for which eight lanes do not sum inside `i32` after one
  fold-back. 19441's transform improved by the same software pipelining as 17497's (532 to 514
  cycles) but its `commit` column did not move: what the wall clock of a limb this wide is waiting
  on is the `A` stream, not the last 3 % of the kernel. All six extra moduli together: 47.2 ms.
* **What the clear-text round puts on the wire.** `wire` codes the three messages of the
  non-recursive mode, and `src/main.rs` prints the lengths it measures rather than a padded width.
  The uniform objects are bit-packed: a commitment slot is a residue, so it takes `ceil(log2 q)`
  bits per limb — 12 at 3889, 14 at 9721 — and an `F162` is a uniform 162-bit element, so the row
  evaluation is 162 bits an element back to back, 5184 bytes for 256 rather than 6144. Those are
  entropy floors, not compression, and nothing can go under them. The folded witness is the one
  message with structure: `v = sum_j c_j W_j` is 256 binary columns against weight-28 binary
  challenges, so its coefficients sit inside `(q-1)/2 = 1944` but are a discrete Gaussian of a few
  tens, 8.1 bits of entropy against the 16 an `i16` spends. A 32-bit rANS with 16-bit
  renormalisation codes it against a histogram of the message itself, quantised to a total of
  2^12 by largest remainder with every occupied symbol floored at one slot; the table covers the
  occupied range `[offset, offset + length)` and travels in the header as Elias gamma codes of
  `count + 1`, one bit for each empty symbol in the tails. A symbol the quantisation cannot afford
  — which needs an adversarial fold spread over more than 4095 values of a large limb — is coded
  as an escape followed by its index in raw bits, so any `i16` message encodes and the coder is a
  bijection, not a heuristic. At the basic shape that is 163.5 KB for 165 888 coefficients, 0.7 %
  above the message's own zeroth-order entropy, in 0.8 ms of encoding and 0.7 ms of decoding; the
  verifier decodes all three objects in 2.0 ms and checks the ones it decoded. The fold's width
  varies from round to round more than a sum of 5376 signed bits suggests, because the 256
  challenges are shared by all 256 output ring elements: each coefficient position carries a
  common offset, the signed sum of the challenge coefficients that land on it — signed because the
  lift is `c(-X^4)`, so the parity of the exponent decides it — whose own spread is as wide as the
  fluctuation around it. Over the fourteen limb lists of the seven bases, with and without a
  second limb, the measured sigma runs from 53 to 120 and the coded fold from 158.3 to 177.7 KB —
  always within 0.72 % of that message's own entropy. `keccak::Session` sends its clear-text
  opening through the same code and decodes it on the verifier's side, which is the 289.08 KB of
  the three-column table above against 650.55 uncoded.

## BaseFold baseline

`src/bin/basefold.rs` runs binius64's own polynomial commitment on its own — the Merkle-committed
oracle, the `B128` ring switch and the BaseFold opening, driven through the same IOP channel
`Prover::prove` and `Verifier::verify` drive (`send_oracle`, `prove_oracle_relation`, `finish`),
with no circuit anywhere — over a uniformly random vector of `2^18` `B128` at the keccak example's
defaults: `--log-inv-rate 1`, `--hash-suite sha256`, one thread on core 3. So the row below is
that scheme alone rather than its share of a whole proof.

```
   Comm.   Prover  Verifier        C     |pi|
      ms       ms        ms       KB       KB
   12.37    10.04      0.44     0.03   237.95
```

`Comm.` and `Verifier` are medians of three, the prover a single run. The prover commits, draws
the 25-coordinate evaluation point off the transcript, evaluates the vector's bit multilinear
there — 2.2 ms, the statement rather than the proof — and opens that claim; the verifier replays
the same point off the same tape and closes the opening, and an opening of a vector the root does
not bind is rejected. `C` is the Merkle root as it travels, 32 bytes, since the FRI parameters the
verifier reads it with are public; `|pi|` is the rest of the tape, the ring switch and the batched
BaseFold opening. The stock column of the keccak table above commits the same `2^18` trace in
12.41 ms and verifies its BaseFold opening in 0.46 ms, which is the same scheme measured from
inside a proof.

```
cargo run --release --offline --bin basefold   # taskset -c 3, as above
```

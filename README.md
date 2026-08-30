# bin-ntt

An Ajtai commitment, a folding step and their verifier over `R_648 = Z_q[X]/(X^648 - X^324 + 1)`,
the 1944-th cyclotomic ring, for a witness of binary ring elements carried as elements of
`F162 = GF(2)[x]/(x^162 + x^81 + 1)`. The witness is committed modulo the base modulus 3889 and
any of `2917, 4861, 9721, 12637`, folded against short ternary challenges of the subring
`R_162 = Z_q[Z]/Phi_243(Z)`, and the folded opening is checked against the multilinear extension
of the same witness over `F162` — one AVX-512 thread throughout. The opening is either sent in
the clear or recursed into a single LaBRADOR proof of 80 KB.

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

`Params::new(witness_log_len, column_log_len, extra_moduli, true)` replaces the last three
messages by one LaBRADOR proof. The commitment the verifier receives becomes the Ajtai commitment
`T_Y` to the RNS residues of the matrix, the left expansion `u` becomes `T_u`, and `v` is never
sent: the prover commits to the rest of the witness as `T_R`, announces the exact squared norm of
every witness vector, takes the verifier's mask scalars, and proves everything at once.

```rust
let params = Params::new(18, 8, vec![Modulus::Q9721], true).unwrap();
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

`prove_opening` returns `Err(OpeningError::FoldTooLong { .. })` when the fold's squared norm
exceeds its cap — about one round in twenty — and the caller retries with fresh challenges.
`verify_opening` runs no field arithmetic of its own: `u . eq(p1) == t` and
`eq(p0) . (v mod 2) == sum_j u_j (c_j mod 2)` are two of the identities inside the proof.

## Runtime

`Params::basic()` — 2^18 `F162` = 2^16 ring elements of `R_648` in 256 columns, moduli 3889 and
9721 — on one core of an i7-11850H, wall clock, best of 3 except the steps that consume what they
are given, which run once.

| step | ms |
|------|---:|
| **prover** | |
| `commit` | 13.09 |
| `row_evaluate` | 0.28 |
| `fold` | 4.41 |
| *total* | *17.77* |
| **statement** | |
| `derive_evaluation_point` | 0.51 |
| `mle_evaluate` | 0.28 |
| *total* | *0.80* |
| **verifier** | |
| `derive_folding_challenges` | 1.12 |
| `fold_commitment` | 0.30 |
| `fold_row_evaluation` | 0.00 |
| `verify_evaluation` | 0.01 |
| `verify_folded_opening` | 0.31 |
| *total* | *1.74* |

The wire is 648 KB of commitment, 6 KB of row evaluation and 324 KB of folded witness.

The same shape with the recursion on. `PublicParameters::from_seed` additionally inverts the key
rows, blocks them, converts the 221 184 key-time `phi` and the nine-bit pattern table to `polx`,
and picks the three commitment ranks (`kappa_Y = 11`, `kappa_u = 3`, `kappa_R = 8`) — 34 ms and
20 MB, paid once per key. The encoded witness is 18 LaBRADOR vectors, 10 752 polynomials.

| step | ms |
|------|---:|
| **prover** | |
| `commit`, including `T_Y` | 18.06 |
| `row_evaluate` | 0.29 |
| `commit_left_expansion` | 0.25 |
| `prove_opening` | 232.95 |
| — fold | 4.56 |
| — encoding | 18.32 |
| — `T_R` | 2.23 |
| — masks | 1.92 |
| — constraint `phi` | 2.31 |
| — statement build | 7.69 |
| — `labrador::prove` | 171.33 |
| *total* | *251.54* |
| **statement** | |
| `derive_evaluation_point` | 0.00 |
| `mle_evaluate` | 0.30 |
| *total* | *0.30* |
| **verifier** | |
| `derive_folding_challenges` | 1.21 |
| statement rebuild | 16.77 |
| — layout | 1.39 |
| — no-wrap bound | 2.33 |
| — constraint `phi` | 2.31 |
| — statement build | 7.69 |
| `labrador::verify` | 102.78 |
| *total* | *120.76* |

The proof is 79.6 KB: `T_Y` 4.1 KB, `T_u` 1.1 KB, `T_R` 3.0 KB, the 18 announced norms 0.1 KB and
LaBRADOR's own 71.2 KB, against 978 KB in the clear. Per proof the constraint `phi` take 1 MB on
top of the key's 20 MB, plus 33 MB for the three mask rows; the peak resident set of one round in
each mode is 294 MB.

Against the first working version of the recursion, at the same shape and on the same core:
`commit` 49.6 -> 18.1, `prove_opening` 526.4 -> 233.0 and the verifier 287.8 -> 120.8 ms. Inside
LaBRADOR (300 -> 171 ms proving, 222 -> 103 ms verifying, transcript-neutral): the chain `phi`
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

## Running it

```
cargo run --release --offline          # taskset -c 3 to pin it, as the table above is measured
cargo test --release --offline
```

The host must have AVX-512 F/BW/VBMI/VBMI2/VNNI/GFNI, and the crate is built with
`-C target-cpu=native` (`.cargo/config.toml`).

## Changing the configuration

`src/main.rs` takes no arguments; the configuration is the block of constants at the top of the
file:

```rust
const WITNESS_LOG_LEN: u32 = 18;                      // 2^18 elements of F162
const COLUMN_LOG_LEN: u32 = 8;                        // 256 columns
const EXTRA_MODULI: &[Modulus] = &[Modulus::Q9721];   // plus the fixed base modulus 3889
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const WITNESS_SEED: [u8; 32] = [0xC7; 32];
const CPU: usize = 3;
```

`EXTRA_MODULI` is any subset of `Modulus::{Q2917, Q4861, Q9721, Q12637}` (see the last
implementation note for what each one costs); `Params::basic()` is this same shape for library
users, with the recursion off. `Params::new(witness_log_len, column_log_len, extra_moduli,
recursion)` refuses a column shorter than one 128-`F162` batch, fewer than two columns, more
columns than elements, or a repeated modulus. `cargo run` prints both modes.

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
* **A reference kernel next to the generated one.** `simd::vertical_bin` is the pure-intrinsics
  implementation of the same split-tree binary kernel — one Rust expression per butterfly, and the
  two-multiply Barrett on the untwiddled `a0` of levels 4, 5 and 6 for q = 9721 — kept out of the
  production path but checked against `vertical_bin_asm` on every test run: bit-identical for
  q = 3889, equal modulo q for q = 9721, where the `asm!` kernel's lookup Barrett leaves smaller
  representatives.
* **Word-sliced `F162` arithmetic.** The binary side is one dot product over `F162` per step,
  computed with `bin_fields`' word-sliced kernels — limb `k` of 8 consecutive elements in one
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
  A quadratic-slot limb has no batched inverse transform and keeps the scalar route.
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
* **The moduli, quantified.** Committing 2^18 `F162` in 256 columns modulo the base 3889 alone
  takes 7.3 ms; each extra modulus adds its own transform, base multiplication and fold-down on
  the shared front end (wall clock, one core, best of 15; the cycle columns are per ring element,
  cache-resident):

  | modulus | slots | added to `commit` | transform | base multiplication | fold-down |
  |---------|-------|------------------:|----------:|--------------------:|----------:|
  | 2917    | quadratic | +5.97 ms | 247 cycles | 74 cycles | 17 cycles |
  | 4861    | quadratic | +6.29 ms | 254 | 74 | 17 |
  | 3889 (base) | linear | 7.34 ms | 280 | 58 | 7 |
  | 9721    | linear | +6.50 ms | 304 | 58 | 7 |
  | 12637   | quadratic | +7.19 ms | 291 | 75 | 16 |

  The front end costs 31 more cycles per ring element and is paid once however many moduli
  follow. A quadratic-slot modulus runs a shorter tree — one radix-2 level fewer, and for 2917
  and 4861 two thirds of the Montgomery products of a splitting one — but pays for it in the base
  multiplication, where three sums per two rows cost 16 cycles more than one sum per row, and in
  the fold-down, which has three accumulators to reduce instead of one. 2917 is the cheapest
  modulus there is, 4861 and 9721 sit within 3 % of each other, and 12637 — three Barretts inside
  the kernel, and an accumulator it has to fold back every batch — is 20 % dearer than 2917. All four extra moduli together: 35.9 ms.

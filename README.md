# bin-ntt

An Ajtai commitment, a folding step and their verifier over `R_648 = Z_q[X]/(X^648 - X^324 + 1)`,
the 1944-th cyclotomic ring, for a witness of binary ring elements carried as elements of
`F162 = GF(2)[x]/(x^162 + x^81 + 1)`. The witness is committed modulo the base modulus 3889 and
any of `2917, 4861, 9721, 12637`, folded against short ternary challenges of the subring
`R_162 = Z_q[Z]/Phi_243(Z)`, and the folded opening is checked against the multilinear extension
of the same witness over `F162` — one AVX-512 thread throughout.

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

## Runtime

`Params::basic()` — 2^18 `F162` = 2^16 ring elements of `R_648` in 256 columns, moduli 3889 and
9721 — on one core of an i7-11850H, wall clock, best of 3 except `commit` and `fold`, which run
once each because the fold consumes the opening.

| step | ms |
|------|---:|
| **prover** | |
| `commit` | 13.17 |
| `row_evaluate` | 0.28 |
| `fold` | 4.52 |
| *total* | *17.96* |
| **statement** | |
| `derive_evaluation_point` | 0.53 |
| `mle_evaluate` | 0.29 |
| *total* | *0.82* |
| **verifier** | |
| `derive_folding_challenges` | 1.15 |
| `fold_commitment` | 0.30 |
| `fold_row_evaluation` | 0.00 |
| `verify_evaluation` | 0.01 |
| `verify_folded_opening` | 0.32 |
| *total* | *1.78* |

`Prover::new` allocates and first-touches the 85 MB workspace and runs one commitment and one
fold to bring the kernels and the rejection tables up; `Prover::fold` hands the workspace back, so
a second `commit` allocates nothing.

## Running it

```
cargo run --release --offline          # taskset -c 2 to pin it, as the table above is measured
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
const CPU: usize = 2;
```

`EXTRA_MODULI` is any subset of `Modulus::{Q2917, Q4861, Q9721, Q12637}` (see the last
implementation note for what each one costs); `Params::basic()` is this same shape for library
users. `Params::new(witness_log_len, column_log_len, extra_moduli)` refuses a column shorter than
one 128-`F162` batch, fewer than two columns, more columns than elements, or a repeated modulus.

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

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
| `commit` | 15.04 |
| `row_evaluate` | 0.29 |
| `fold` | 4.53 |
| *total* | *19.87* |
| **statement** | |
| `derive_evaluation_point` | 0.55 |
| `mle_evaluate` | 0.31 |
| *total* | *0.86* |
| **verifier** | |
| `derive_folding_challenges` | 1.18 |
| `fold_commitment` | 0.30 |
| `fold_row_evaluation` | 0.00 |
| `verify_evaluation` | 0.01 |
| `verify_folded_opening` | 0.33 |
| *total* | *1.81* |

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

`src/main.rs` takes no arguments: edit `MATRIX_SEED`, `WITNESS_SEED` and `CPU` at the top of the
file, and `Params::basic()` for the shape. Other shapes come from
`Params::new(witness_log_len, column_log_len, extra_moduli)`, which refuses a column shorter than
one 128-`F162` batch, fewer than two columns, more columns than elements, or a repeated modulus.

## Implementation notes

* **Vertical batch-of-32 layout.** A `Batch32` holds coefficient `j` of 32 polynomials in one
  512-bit vector, so every butterfly of the transform is a full-width i16 operation and no
  shuffle ever crosses a polynomial boundary.
* **The binary lookup trick.** The NTT is linear and the inputs are bits, so levels 0 and 1 of
  the tree collapse into one 16-entry table lookup per output vector, indexed by the nibble the
  bit-slicing front end already produces; the twiddles of levels 2 and 3 are pre-multiplied into
  those tables. Multiplications per ring element drop from 3564 to 2160.
* **Per-modulus reduction schedules.** Each prime reduces only where its `2^15/q` head-room runs
  out: 2917, 3889 and 4861 need no reduction anywhere inside the binary kernel, 9721 and 12637
  need a lookup Barrett on one or three levels, and each accumulator gets its own fold-back
  period from its own compile-time bound.
* **Block-fused base multiplication.** The kernel hands out 27 finished slot vectors at a time
  and a hook multiplies them against `A` and accumulates on the spot with `vpdpwssd`, so the
  transform output never reaches memory and only `A` streams. A quadratic-slot modulus carries
  three sums per two rows, formed by Karatsuba where the lazily reduced operand still fits i16
  and schoolbook where it does not.
* **The fold is one 85 MB read.** The commitment writes the base-modulus transform of the witness
  out as it goes, with non-temporal stores hidden behind the transform, so the fold never
  transforms anything again: it is one `vpmaddwd` per slot vector and pair of columns over a
  single pass of that buffer, and the result comes back to coefficients as a genuine small
  integer vector.
* **Word-sliced `F162` arithmetic.** The binary side is one dot product over `F162` per step,
  computed with `bin_fields`' word-sliced kernels — limb `k` of 8 consecutive elements in one
  `zmm`, 12 unreduced `clmul` products per block and a single reduction at the end of the whole
  product. The witness never leaves its own layout; it is transposed 8 elements at a time inside
  the loop.
* **The moduli, cheapest first.** 2917 ~ 3889 ~ 4861 < 9721 < 12637: a quadratic-slot modulus
  wins on the transform and gives most of it back on the base multiplication.

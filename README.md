# bin-ntt

An Ajtai commitment, a fold and their verifier over `R_648 = Z_q[X]/(X^648 - X^324 + 1)`, the
1944-th cyclotomic ring, for binary witnesses held as `F162 = GF(2)[x]/(x^162+x^81+1)`. The
witness is committed modulo a base prime and one or two extra primes, folded against short signed
challenges of the subring `R_162` (weight 28, canonical bound 12), and the folded opening is
checked against the multilinear extension of the same witness over `F162`. Three opening modes:
**clear**, the folded witness on the wire; **bit-dropped**, the commitment keeping only the top
bits and the verifier bounding the residual; **recursive**, the opening replaced by a LaBRADOR
proof. The mode is `Params::opening`, and one `Verifier::verify_opening` dispatches on it.
One AVX-512 thread throughout.

## Layout

- `crates/pcs` — `bin-ntt`: commitment, fold, verifier, AVX-512 kernels, the C LaBRADOR bridge.
- `crates/bench` — the reference round (`bin-ntt`), calibration (`calibrate`), `bench.sh`, `tables.py`.
- `crates/binius` — binius64's hash proofs over this commitment (`hashes-binius`).
- `crates/flock` — Flock as the second PIOP (`hashes-flock`).
- `crates/competitors` — BaseFold, WHIR and Ligerito on their own (`pcs-competitors`).

## Building and running

```
cargo build --release --workspace --bins
taskset -c 3 ./target/release/bin-ntt --suite m
./bench.sh                                  # every binary at every suite
cargo test --release --workspace
```

## One round

```rust
let pp = PublicParameters::from_seed(Params::basic(), MATRIX_SEED);   // Opening::Clear
let witness = Witness::random(pp.params(), WITNESS_SEED);
let (mut prover, verifier) = (Prover::new(&pp), Verifier::new(&pp));
let (commitment, opening) = prover.commit(&witness);
let mut t = Transcript::new(b"bin-ntt/reference");
let point = verifier.derive_evaluation_point(&mut t, &commitment);
let (claimed, row) = (witness.mle_evaluate(&point), witness.row_evaluate(&point));
let challenges = verifier.derive_folding_challenges(&mut t, &row);
let folded = prover.fold(opening, &challenges);
verifier.verify_evaluation(&point, &claimed, &row).unwrap();
verifier.verify_opening(&commitment, &challenges, &point, OpeningMessage::Clear {
    folded_commitment: &verifier.fold_commitment(&commitment, &challenges),
    folded_witness: &folded,
    folded_row_value: &verifier.fold_row_evaluation(&row, &challenges),
}).unwrap();
```

`crates/bench/src/bin/bin-ntt.rs` runs this in all three modes with the wall clock on every step.

# labinius

labinius is a lattice-based polynomial commitment scheme for binary witnesses. The commitment
is an Ajtai commitment over a fully splitting cyclotomic ring, and the evaluation claim lives in
the binary field `GF(2^162)`, so binary circuits are proved as they are. The opening is sent in
the clear, bit-dropped, or replaced by a LaBRADOR proof. The repository also plugs the scheme
under the Binius and Flock front ends to prove Keccak-256, SHA-256 and BLAKE3, and benchmarks it
against BaseFold, WHIR, Ligerito and Brakedown. Everything runs on one AVX-512 thread.

## Layout

- `crates/pcs` — `labinius`: commitment, fold, verifier, AVX-512 kernels, the C LaBRADOR bridge.
- `crates/bench` — the reference round (`labinius`), calibration (`calibrate`), `bench.sh`, `tables.py`.
- `crates/binius` — binius64's hash proofs over this commitment (`hashes-binius`).
- `crates/flock` — Flock as the second PIOP (`hashes-flock`).
- `crates/competitors` — BaseFold, WHIR and Ligerito on their own (`pcs-competitors`).

## Building and running

```
cargo build --release --workspace --bins
./target/release/labinius --suite m            # pins itself to core 3, or to $BENCH_CPU
./bench.sh                                  # every binary at every suite
cargo test --release --workspace
```

## One round

```rust
let pp = PublicParameters::from_seed(Params::basic(), MATRIX_SEED);   // Opening::Clear
let witness = Witness::random(pp.params(), WITNESS_SEED);
let (mut prover, verifier) = (Prover::new(&pp), Verifier::new(&pp));
let (commitment, opening) = prover.commit(&witness);
let mut t = Transcript::new(b"labinius/reference");
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

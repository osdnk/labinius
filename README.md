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
cargo test --release --workspace
```

Every binary takes `--suite s|m|l|xl` (witness of `2^18`, `2^20`, `2^22`, `2^24` bits; default `s`),
pins itself to core 3 or to `$BENCH_CPU`, and reports each timed step as the median of 10 runs.

| what you want to see | run |
| --- | --- |
| our scheme, opening in the clear and bit-dropped | `target/release/labinius --suite m` |
| our scheme, opening replaced by a LaBRADOR proof | `cargo run --release -p labinius-bench --features labrador -- --suite m` |
| Keccak-256, SHA-256, BLAKE3 under Binius64: stock vs. ours | `target/release/hashes-binius --suite m` |
| BLAKE3, SHA-256 under Flock: stock vs. ours | `target/release/hashes-flock --suite m` |
| BaseFold, WHIR, Ligerito, Brakedown on their own | `target/release/pcs-competitors --suite m` |
| the parameter calibrations behind the paper | `target/release/calibrate <bdstats\|boundcheck\|foldstats\|gadget\|moduli\|recursion\|wire_bench>` |

`--features labrador` swaps the `labinius` binary's body: without it the round is clear and
bit-dropped, with it the round is the recursive one. The C library under `crates/pcs/labrador`
is built either way.

### The whole benchmark

`./bench.sh` needs nothing run before it. It builds every binary itself (`cargo build --release
--offline`, so the only prerequisite is that the dependencies are in the cargo cache: `cargo
fetch` once), then runs `labinius`, `pcs-competitors`, `hashes-flock` and `hashes-binius` at
every suite, rebuilds with `--features labrador` and runs `labinius` again. It may take long time, so
run it detached:

```
setsid nohup ./bench.sh > bench.out 2>&1 < /dev/null & disown
```

It writes `bench-<timestamp>/`: `machine.txt`, `build.log`, one `<binary>-<suite>.log` per run
(the labrador runs as `labinius-<suite>-labrador.log`) and `summary.tsv` with status, seconds
and peak RSS per run; `bench.out` has the same summary at the end. Knobs, all optional:

| variable | default | meaning |
| --- | --- | --- |
| `SIZES` | `sizes sizem sizel sizexl` | which suites |
| `CPU` | `3` | the core every run is pinned to |
| `OUT` | `bench-<timestamp>` | the output directory |
| `SKIP_BUILD` | `0` | `1` reuses `target/release` as it is |
| `BINIUS_XL_GB` | `80` | `hashes-binius` at `xl` is skipped below this much RAM |

`crates/bench/tables.py <bench-dir> <bench-dir> <paper.tex>` patches the numbers from those logs
into the paper's tables.

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

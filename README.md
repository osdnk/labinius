# bin-ntt

AVX-512 NTT and Ajtai commitment (inner product in the NTT domain) over the 1944-th
cyclotomic ring

    R_q = Z_q[X] / (X^648 - X^324 + 1),    1944 = 2^3 * 3^5,    q in {3889, 9721},

for **binary inputs given as a stream of F162 elements**: four consecutive elements a, b, c, d
of `bin_fields::scalar::F162` (the type used by binius64-f162, `F162([u64; 3])`, 162 bits)
are read as one ring element by plain interleaving,

    coefficient of X^{4m+k}  =  bit m of element k        (k = 0..4 for a, b, c, d;  m = 0..162),

so 2^18 F162 elements are 2^16 ring elements with 0/1 coefficients. Both primes are 1 mod 1944,
so R_q splits into 648 linear factors and the transform is complete. Single-threaded, tuned for
one specific core (Intel i7-11850H, Tiger Lake), signed 16-bit lanes throughout, Rust stable.
All bit-shuffling from the F162 byte layout to the kernel's input is inside the measured time.

## Headline: the full inner product, 2^18 F162 = 2^16 ring elements, one core

The commitment `y[j] = sum_i A_i[j] * NTT_q(w_i)[j] mod q` (one row, j = 0..648), for both
primes, with A uniformly random in the NTT domain, centered, living in memory
(2^16 x 648 x 2 B = 85 MB per prime, cold), and the w_i lifted from a plain `&[F162]` (6 MB).
Output: one ring element per prime in the NTT domain, fully reduced. Best of 3, pinned to one
core, hardware counters via `perf_event_open`, machine otherwise idle:

| q    | `commit` (vertical, block-fused, prefetching) | total ms | cycles / ring element | of which: front end + transform | basemul | un-hidden A stream |
|------|-----------------------------------------------|---------:|----------------------:|--------------------------------:|--------:|-------------------:|
| 3889 |                                               | **7.6**  | **462**               | 311                             | 60      | 92                 |
| 9721 |                                               | **8.0**  | **485**               | 335                             | 60      | 90                 |
| both | `commit_2q` (one front end, two primes, 170 MB of A) | **15.7** | 477 per prime |                             |         |                    |

(milliseconds at the 4.0 GHz this run settled at; at 4.5 GHz the same cycle counts are
6.7 / 7.1 / 14 ms.) Floors per prime: streaming A from DRAM at the measured 19.5 GB/s takes
4.4 ms; the compute alone (front end + transform + basemul, cache-resident) is 371 / 395 cycles
per element = 6.1 / 6.5 ms at 4.0 GHz. The commitment therefore runs at ~80 % of
max(DRAM, compute); the remainder is A arriving at ~11 GB/s instead of 19.5 because its
prefetches can only be issued at the 24 block boundaries of each batch (see "Remaining
optimisation strategies").

The alternatives measured with the same inputs (cycles per ring element, q = 3889 / 9721):

| how the products are formed                                           | 3889 | 9721 |
|-----------------------------------------------------------------------|-----:|-----:|
| transform materialised to memory, multiply-accumulate in a second pass (`commit_unfused`) | 872 | 904 |
| multiply-accumulate right after each batch of 32, operands in L1/L2 (`commit_batch_fused`) | 623 | 654 |
| multiply-accumulate on each 27-slot block as the kernel finishes it, no prefetch | 600 | 624 |
| the same with A prefetched under the transform (`commit`) | **462** | **485** |
| horizontal layout, groups of 4 elements fully L1-resident (`commit_h`, Gregor Seiler's layout) | 842 | 898 |

The NTT alone (the same front end, transform materialised with non-temporal stores, 85 MB of
output per prime):

| q    | driver                                          | total ms | cycles / ring element | cycles / F162 | instructions / ring element |
|------|-------------------------------------------------|---------:|----------------------:|--------------:|----------------------------:|
| 3889 | `ntt_f162`                                      | **5.1**  | **353**               | 88            | 749                         |
| 9721 | `ntt_f162`                                      | **5.7**  | **376**               | 94            | 804                         |
| both | `ntt_f162_2q` (one front end, two primes)       | 10.2     | 353 per prime         | 88            | 750 per prime               |
| 3889 | `ntt_f162_stream` (closure per batch, nothing written) | 4.5 | 313               | 78            | 749                         |

Cycles are the robust number: this laptop's clock under AVX-512 load moves between 4.0 and
4.6 GHz with temperature, so milliseconds vary by ~10 % between runs (this table at 4.55 GHz,
the commitment tables above at 4.0 GHz). Components,
cache-resident: F162 bit-slicing 32 cycles per ring element (8 per F162, port-5 bound),
transform 280 / 305 (q = 3889 / 9721), together 310 / 337. For scale, 650 cycles per
polynomial (an expert estimate for a generic-input NTT of this size on this core) is ~9.4 ms
for 2^16 elements.

Reproduce: `cargo run --release --offline --bin bench_commit -- <cpu>` (the commitment, all
variants), `bench_commit_h` (the horizontal one), `bench_f162` (the NTT alone).

## API

Five types in `src/api.rs`, re-exported at the crate root, are the whole public surface.

* `PowerOfThreeRingElement { v: [i16; 162] }` — one element of the 3^5-th cyclotomic ring
  `R_162 = Z_q[Z] / Phi_243(Z)` for one prime, in its NTT domain: 162 slots, centered signed
  residues in `[-(q-1)/2, (q-1)/2]` (`normalized(q)` gives `[0, q)`).
* `PowerOfThreeRingElementWithTwoLimbs { limb: [PowerOfThreeRingElement; 2] }` — limb k is the
  residue modulo `PRIMES[k]`, `PRIMES = [3889, 9721]`.
* `CommitmentKey` — the matrix A for both primes, uniform in the NTT domain, centered, in the
  vertical layout `commit` streams. `CommitmentKey::random(len_f162, seed)` (length in F162
  elements, a multiple of 128 = 32 ring elements), `len_f162()`, `bytes()`.
* `VerticallyAlignedMatrix<T>` — `rows()` x `cols()`, stored column by column, with `get(row, col)`,
  `column(col)`, `columns()`. One column is one commitment.
* `AuxData` — what `commit_with_aux` leaves behind for the folding step below: the witness's
  transform modulo `PRIMES[0]` and the raw 648-slot commitments of the `r` chunks.

```rust
let ck = CommitmentKey::random(1 << 18, seed);
let c = ck.commit(&witness, 1);
let slots = &c.get(0, 0).limb[0].v[..];   // component 0 mod 3889: 162 centered slots
```

`ck.commit(&witness, r)` takes `r` a power of two and `witness.len() == r * ck.len_f162()`. It
splits the witness into r consecutive chunks, commits each under the same key, and returns a
**4 x r** matrix of `PowerOfThreeRingElementWithTwoLimbs`: column c is the commitment of chunk c.

**The four rows.** One commitment is a single element y of `R_648`; the four rows are its four
components in the basis 1, X, X^2, X^3 over `S = Z_q[Y]/(Y^162 - Y^81 + 1) = R_162`, Y = X^4,
Z = -Y — the height-4 view of "The lift is a ring extension of degree 4" below. Read that way the
commitment is a rank-4 module-SIS commitment over `R_162` whose 4 x 4 blocks are the Y-twisted
circulants of multiplication by the uniform A_i; any `R_162`-linear operation acts on the four rows
independently.

**The slot order.** With psi the primitive 1944-th root of unity that fixes `SLOT_EXP`,
theta = psi^4 is a primitive 486-th root of unity and the 648 units mod 1944 fall into 162 classes
of four modulo 486. `POW3_SLOT_EXP[s]` is the class v of the s-th slot, in the order in which the
classes first appear in `SLOT_EXP` (the tree order of `R_648`), and slot s of a component holds
y_k(theta^v) — equivalently the component read as a polynomial in Z evaluated at the primitive
243-rd root of unity `-theta^v`. The four slots u = v + 486 t of the big ring satisfy
`E_t = sum_k psi^{vk} i^{tk} Y_k(v)` with i = psi^486, inverted per class by
`Y_k(v) = 4^-1 psi^{-vk} sum_t i^{-tk} E_t`; `decompose_648_to_4x162::<Q>` is that map. It is a
radix-4 butterfly (i^2 = -1, so one product per class) over 16 slots at a time, 0.18 ms even when
it runs 512 times.

**Measured** (`cargo run --release --offline`, `taskset -c 2`, 2^18 F162 = 2^16 ring elements,
both primes, best of 3, ~4.2 GHz):

| r   | key, both primes | total ms | cycles / ring element and prime | of which decomposition |
|-----|-----------------:|---------:|--------------------------------:|-----------------------:|
| 1   |          170 MB  | **15.3** | **488**                         | 1 us                   |
| 4   |         42.5 MB  | 14.5     | 456                             | 5 us                   |
| 16  |         10.6 MB  | **13.5** | **417**                         | 13 us                  |
| 256 |         0.66 MB  | 13.9     | 433                             | 177 us                 |

The same 170 MB of A is read whatever r is; splitting the witness only changes where it is read
from. At r = 16 the key is 10.6 MB and L3-resident, which is worth 15 % over the cold r = 1 run; at
r = 256 it is 0.66 MB and L2-resident, and the gain is given back to the 256 output elements'
decomposition and to the per-chunk fixed cost of the kernel. Front end plus transform,
cache-resident, is 316 / 340 cycles per ring element (q = 3889 / 9721) and the base multiplication
59, so 375 / 399 of the r = 1 cost is compute and the rest is A that does not hide.

### Short challenges

`src/challenge.rs`, re-exported at the crate root, is the challenge side of a Fiat-Shamir protocol
over this commitment: short elements of the same `R_162` the commitment's four components live in.

* `Transcript` — a blake3 transcript. `Transcript::new(domain)`, `absorb_bytes`, `absorb_u64`,
  `absorb_elements(&[PowerOfThreeRingElementWithTwoLimbs])` (the raw little-endian `i16` slots,
  limb 0 then limb 1, 648 bytes per element). `fill(label, out)` clones the absorbing state,
  appends a per-transcript sample counter and the label, and reads the extendable output — so a
  derivation is bound to everything absorbed before it, successive derivations are independent,
  and everything is a deterministic function of the absorbed bytes.
* `ShortChallenge { positions: [u8; MAX_WEIGHT], signs: [i8; MAX_WEIGHT], weight }` — a weight-`w`
  ternary element of `R_162`, stored as sorted distinct positions plus signs (`MAX_WEIGHT = 32`).
  `coeffs() -> [i8; 162]`, `from_coeffs`, `log2_cardinality(w) = log2 C(162, w) + w`.
* `sample_attempt(&mut Transcript, w)` is uniform over that set: a partial Fisher-Yates over the
  162 positions with each index drawn uniformly from `[i, 162)` by rejection on 16-bit XOF draws,
  and uniform signs. `sample_short_challenge(t, w, bound)` rejects until the challenge is short and
  returns it with the number of attempts; all of its attempts read one XOF derivation and reuse one
  set of buffers, so an attempt costs no blake3 finalisation and no allocation. Defaults:
  `DEFAULT_WEIGHT = 21`, `DEFAULT_BOUND = 9.0`.

**The bound.** `canonical_inf_norm_sq(&c)` is `max_u |c(zeta^u)|^2` over the 162 primitive 243-rd
roots of unity `zeta^u` (`zeta = exp(2 pi i / 243)`, `gcd(u, 3) = 1`) — the squared sup norm of the
canonical embedding, i.e. the squared operator norm of multiplication by `c` on `R_162 (x) C`, which
is what a security argument needs from a challenge set. A challenge is accepted when that is at most
`bound^2`; the default `bound = 9` is an energy bound of `81` against a mean of `w = 21`. It is
evaluated from the `w` nonzero terms only, at 81 roots (`c` is real, so the conjugate half repeats):
`PHASE_RE[p][k] = cos(2 pi p u_k / 243)` and `PHASE_IM` are a 162 x 88 `f64` table (~228 KB, built
once), and a term adds `+-` one row of it to the accumulator — contiguous `f64` loops, no gathers.
The rejection loop runs that in two blocks of 44 lanes and stops at the first root over the bound,
which is where most rejected attempts die. `canonical_inf_norm_sq_naive` (Horner in complex `f64`
over all 162 roots) is the reference the whole thing is tested against.

```rust
let mut t = Transcript::new(b"bin-ntt/example");
t.absorb_elements(c.column(0));                                  // bind the commitment
let (chal, attempts) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
let coeffs = chal.coeffs();                   // the challenge as 162 coefficients in {-1, 0, 1}
```

As an element of `R_162` a challenge lifts into `R_648` as `c(-X^4)` (coefficient of `X^{4m}` is
`(-1)^m c_m`), which is what multiplying all four rows of a commitment by one challenge means.

**Measured** (`cargo test --release --offline --test challenge -- --nocapture`, weight 21, 2000
accepted challenges per row). The challenge set has `log2 C(162, 21) + 21 = 107.7` bits; an attempt
costs 0.7 us when the first block rejects it and 1.0 us when every root is evaluated.

| bound | attempts per accepted challenge | acceptance | us per accepted challenge | us per attempt |
|------:|--------------------------------:|-----------:|--------------------------:|---------------:|
| 7.5   | 1531                            | 0.065 %    | 1065                      | 0.70           |
| 8     | 103                             | 0.97 %     | 74                        | 0.71           |
| 9     | 5.70                            | 17.5 %     | 4.6                       | 0.80           |
| 10    | 1.81                            | 55.2 %     | 1.6                       | 0.91           |
| 13    | 1.01                            | 98.8 %     | 1.0                       | 1.02           |

Tightening the bound costs wall time and cardinality, both mildly: at the default 9 a challenge
takes 4.6 us and the accepted set holds 105.2 of the 107.7 bits; at 7.5 a challenge takes a
millisecond and the set still holds 97.1 bits.

### Folding

`src/fold.rs`, re-exported at the crate root, is the folding step of the paper's `Pi_fold` on top
of the commitment. A challenge `c_j` is a short ternary element of the subring `R_162`, embedded
into `R_648` as `c_j(-X^4)`; the folded (amortised) witness and its commitment are

    v = sum_j c_j W_j    (one chunk's worth of ring elements),        A v = sum_j c_j C_j,

the second identity by `R_648`-linearity of `A`. Multiplication by a subring element acts on the
four `R_162` components alike, so in the NTT domain the whole fold is one length-`r` inner product
of scalars per slot, `NTT(v)[u] = sum_j NTT(c_j)[u] NTT(W_j)[u]`, with no ring multiplication
anywhere.

`CommitmentKey::commit_with_aux(&witness, r)` returns the same `4 x r` matrix as `commit` together
with an opaque `AuxData`: the transform of every witness element modulo `q1 = 3889`, written
straight out of the commitment's block sink with non-temporal stores, and the raw 648-slot
commitments `C_j` of the `r` chunks for both primes. `fold(&key, &aux, &challenges)` then returns

```rust
let (c, aux) = ck.commit_with_aux(&witness, 256);          // 4 x 256, plus 85 MB of aux
let mut t = Transcript::new(b"bin-ntt/fold");
for j in 0..256 { t.absorb_elements(c.column(j)); }
let ch: Vec<_> = (0..256)
    .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
    .collect();
let out = fold(&ck, &aux, &ch);
let coeffs = &out.v[0].v[..];        // v as 648 centered integer coefficients of R_648
```

`FoldOutput` carries `v`, the amortised witness as `len_ring` `RingElement`s in **coefficient
form**, centered (`v_components(i)` splits one into its four `R_162` components, coefficients
`4m + k`); `v_ntt`, its transform for both primes; `y = A v` as four
`PowerOfThreeRingElementWithTwoLimbs`, the same shape as one column of a commitment; `y_raw`, the
same before the decomposition; and `max_abs_v`.

**The accumulation.** The witness transform keeps the binary kernel's own lazy reduction
(`|W| <= 7.5 q1`) and the challenge slots are fully reduced (`|c| <= (q1-1)/2`), so one chunk adds
at most `7.5 q1 (q1-1)/2 = 56 700 648` to an `i32` lane. Two chunks are processed together: a
`vpunpcklwd`/`vpunpckhwd` pair interleaves their witness rows and one `vpmaddwd` against the
broadcast dword `(c_j[u], c_{j+1}[u])` forms both products. Starting from the fold-back's own
bound `2^15 (1 + R) = 108 592 384`, 32 chunks fit (`1 923 013 120 < 2^31`) and 64 do not, so the
accumulator is folded back exactly — `x = l + (h + c) R mod q`, `simd::commit`'s reduction —
every 32 chunks; both bounds are `const` assertions. The accumulator is `len_ring/32 x 648`
groups of 32 lanes (663 KB at `len_ring = 256`, L2-resident) and the 85 MB of witness transform is
read once, which is the step's floor.

**Why `v` is a small-integer vector.** A coefficient of `v` is a sum of `r w = 5376` signed 0/1
terms, so it has mean zero and standard deviation `sqrt(r w / 2) = 52`; the largest of the
`648 x 256` coefficients measures **265**, against `q1 / 2 = 1944.5`. The centered lift of
`v mod q1` is therefore the true integer vector, which is what makes `v` usable as the witness of
the next round, and it is the only place in the fold where the integers matter. Modulo `q1` the
accumulator's own output is already `NTT(v)`, so only `q2` needs a forward transform — of 8
batches, not of the witness.

**Measured** (`cargo run --release --offline`, `taskset -c 2`, 2^18 `F162` = 2^16 ring elements,
`r = 256` chunks of 256 ring elements, weight-21 challenges, best of 3, ~4.2 GHz):

| stage                                        | ms       | note                                            |
|----------------------------------------------|---------:|-------------------------------------------------|
| `commit`                                     | 13.7     | the commitment alone                            |
| `commit_into_aux`                            | **14.8** | +8 %: 85 MB of transform kept, non-temporally   |
| 256 challenges from a transcript over `C`    | 2.3      | 5.8 attempts each (`bound = 9`)                 |
| `fold`                                       | **4.51** |                                                 |
| — challenge NTTs                             | 0.10     | 8 batches of `c(-X^4)`, `vertical_gen`          |
| — accumulation                               | 4.27     | 85 MB read at 20 GB/s: the DRAM floor           |
| — inverse NTT, `q1`                          | 0.07     | 8 batches, `vertical_gen::intt_gen_batch32`, plus reading `v` out of the vertical layout |
| — forward NTT, `q2`                          | 0.03     | 8 batches, `vertical_gen`                       |
| — `y = A v`                                  | 0.04     | 8 batches per prime on the commitment's `vpdpwssd` accumulator |
| `commit_into_aux` + `fold`                   | **19.3** | the whole prover is in "Left-expansion and the binary side" below |

The 85 MB an `AuxData` holds is one `mmap`: `commit_with_aux` measures 37 ms because 22 of them
are the kernel's first touch of 20 736 fresh pages. `commit_into_aux` writes into a buffer the
caller already owns (`AuxData::new`), which is what a prover that folds more than once does.

One number now sets the shape of the step: the accumulation is at the machine's read bandwidth —
the two witness rows and the accumulator are plain forward streams, and adding one `prefetcht1`
per slot a step ahead (what `commit` does for `A`) costs 4.96 ms against 4.27 with none, since
this loop has no compute to hide the extra fill-buffer pressure behind. Everything else together
is 0.24 ms. The inverse used to be the untuned `u64` reference — 256 x `scalar::intt`, 6.45 ms,
more than the 85 MB stream — and is now `vertical_gen::intt_gen_batch32` on the 8 batches at 541
cycles per polynomial: 0.07 ms including reading the 256 elements back out of the vertical
layout, which takes the fold from 11.0 to 4.51 ms and `commit_into_aux` + `fold` from 25.9 to 19.3.

### Left-expansion and the binary side

`src/eval.rs`, re-exported at the crate root, is the paper's `Pi_translate` and the field half of
one fold round. The witness the commitment takes *is* a `wdim x r` matrix `W` over
`F = GF(2)[x]/(x^162 + x^81 + 1)` — entry (i, j) is `witness[i + wdim j]`, so a column is one
chunk — and `F` is exactly `R_162 mod 2` under the crate's lift: the coefficients of an `R_162`
element reduced mod 2 are the bits of an `F162`, and the signs vanish. Every `R_162`-linear step
of the fold therefore has a shadow over `F`, and that shadow is what an evaluation claim about the
witness travels along. With nu = 18 variables split as r0 over the row index (10) and r1 over the
column index (8), and `eq(r, b) = prod_k (r_k if b_k = 1 else 1 + r_k)`:

    t   = sum_{i,j} eq(r1, j) eq(r0, i) W[i + wdim j]      the claim about the committed witness
    u   = B W,   B = eq(r0, .)                             the left expansion: r elements of F
    t   = u^T eq(r1)                                       the verifier's claim check
    v   = W c                                              the fold, over R_648
    B v = u^T c                                            the binary check, over F

The last line is the one that ties the two halves together: `B (W c) = (B W) c` mod 2. Its left
side is read straight off the folded witness the fold already produced — component k of packed
element m is the `F162` at stream index `4m + k`, and a coefficient's parity is that element's bit
(`components_mod_2`) — and its right side is an r-term inner product of the prover's message
against the challenges reduced mod 2 (`ShortChallenge::to_f162`: a bit at each of the challenge's
positions, the signs gone). The verifier never sees `W`: it holds the r commitments, the
challenges, `u`, `v` and the claim.

* `EvalPoint<LW, LR> { r0: [F162; LW], r1: [F162; LR] }` (defaults 10 and 8, the headline
  instance), `sample_point(&mut Transcript)`, and `eq_table(rs) -> Vec<F162>`, all `2^len` values
  by doubling, variable k being bit k of the index.
* `evaluate_mle(&witness, &point)` — the claim `t`, in the two-stage form `u` then `u^T eq(r1)`.
* `left_expand(&witness, &r0) -> LeftExpansion { u }` — the prover's message.
* `check_claim(&u, &r1, t)`, `fold_binary(&u, &challenges) -> F162`.
* `RawCommitments::from_aux(&aux)`, `verify_fold(&key, &raw, &challenges, &v)`,
  `verify_binary(&r0, &v, folded)`, and `Verifier { key, commitments, point, claim }` running all
  three checks.

```rust
let (c, aux) = ck.commit_with_aux(&witness, 256);
let point: EvalPoint = sample_point(&mut t);       // t bound to the commitment
let claim = evaluate_mle(&witness, &point);        // the statement
let u = left_expand(&witness, &point.r0).u;        // the prover's message, 256 field elements
let out = fold(&ck, &aux, &challenges);            // v = W c
let v = Verifier { key: &ck, commitments: &RawCommitments::from_aux(&aux), point: &point, claim };
assert!(v.verify(&u, &challenges, &out.v));        // u^T eq(r1) = t,  A v = Y c,  B v = u^T c
```

`verify_fold` recomputes `A v` from `v` alone — the centered range of `q1` first, then
`vertical_gen::ntt_gen_batch32` on the 8 batches per prime and the commitment's own accumulator —
and compares it slot by slot against `sum_j c_j C_j`, so nothing of the prover's is trusted.

**The kernels.** Every step is one dot product over `F`, and all of them run on `bin_fields`'
word-sliced AVX-512 kernels: limb k of 8 consecutive elements in one zmm, `mac_soa8` XOR-ing the
12 unreduced `clmul` products of a block into the accumulator and a *single* `reduce_soa8` at the
end of the whole product — deferred reduction, exactly what that crate's sumcheck round does. One
operand (an `eq` table, or the challenges) is word-sliced once up front; the other is a raw
`&[F162]` run, transposed 8 elements at a time inside the loop by three `vpermi2q`/`vpermq` pairs,
so the witness never leaves the layout the commitment reads it in and nothing is materialised.

**Measured** (`cargo run --release --offline`, `taskset -c 2`, 2^18 `F162` = a 1024 x 256 matrix
over `F`, weight-21 challenges, best of 3; one run, so the three commitment and fold rows repeat
the table above at the clock this one settled at):

| group     | step                                | ms        | note                                                     |
|-----------|-------------------------------------|----------:|----------------------------------------------------------|
| prover    | `commit_into_aux`                   | 14.94     |                                                          |
| prover    | 256 challenges                      | 2.44      |                                                          |
| prover    | `fold`                              | 4.53      |                                                          |
| prover    | `left_expand`                       | **0.28**  | 2^18 products, one per witness element                   |
| prover    | **total**                           | **22.19** |                                                          |
| statement | `sample_point`                      | 0.0004    | one XOF derivation, 18 elements                          |
| statement | `evaluate_mle`                      | **0.297** | the same 2^18 products, plus 256                         |
| verifier  | `check_claim`                       | 0.007     | 256 products                                             |
| verifier  | `fold_binary`                       | 0.005     | 256 products                                             |
| verifier  | `verify_fold`                       | 0.611     | 2 x (`NTT(v)` 0.04, `A v` 0.011, 256 challenge NTTs 0.10, 648-slot `sum_j c_j C_j` 0.08) |
| verifier  | `verify_binary`                     | 0.099     | `v mod 2` bit by bit 0.073, `eq(r0, .)` 0.025, the 1024-term product 0.002 |
| verifier  | **total**                           | **0.723** |                                                          |

The left expansion costs 0.28 ms against the fold's 4.5 and the commitment's 15, so it is 1.3 % of
the prover; the whole verifier is 0.72 ms, another 30x below that. The same 2^18 products through `F162`'s scalar
`Mul` — one `pclmul` chain and one reduction each — take 2.74 ms, so the word-sliced path with its
deferred reduction is **9.2x** faster; at 0.28 ms it is reading the 6.3 MB witness at 21 GB/s,
which is this machine's DRAM read bandwidth, so the transpose inside the loop is free and the step
is at its floor. `verify_fold` has no such floor to hit: its four pieces are all small, and the
largest of them is transforming the 256 challenges, which the verifier cannot avoid.

## Building and testing

`.cargo/config.toml` sets `-C target-cpu=native`; the kernels need AVX-512
F/BW/VL/VBMI/VBMI2/VNNI/GFNI and are tuned for Tiger Lake only. The `bin-fields` dependency is
pinned to the git revision binius64-f162 resolves to and is vendored in the cargo git cache, so
`--offline` works.

    cargo test --release --offline     # ~100 tests: scalar reference, every kernel, driver and
                                       # commitment vs the reference, bounds via i32/i64 shadow
                                       # models, overflow proofs, bit-exact equivalences
    cargo run --release --offline --bin bench_commit -- 2

## What is computed

`psi` is the smallest primitive 1944-th root of unity mod q (7 for 3889, 17 for 9721),
`omega = psi^648`, `zeta6 = psi^324` (so `zeta6^-1 = 1 - zeta6`). The transform is a vanilla
mixed-radix Cooley-Tukey, no twisting, on the fixed tree

    level 0:     X^648 - X^324 + 1 = (X^324 - zeta6)(X^324 - zeta6^-1)       (Phi_6 split)
    level 1, 2:  radix 2   X^n - psi^e = (X^{n/2} - psi^{e/2})(X^{n/2} + psi^{e/2})
    level 3..6:  radix 3   X^n - psi^e = prod_s (X^{n/3} - psi^{e/3} omega^s)

with child s of a sub-ring stored at block offset s*n/p ("tree order"), so slot j of the output
holds `a(psi^SLOT_EXP[j])` (`params::SLOT_EXP`, a permutation of the units mod 1944). The
scalar reference `scalar::ntt` defines the order; `f162::lift4` defines the lift; every kernel,
driver and commitment is tested against `scalar::ntt(lift4(..))`, and
`scalar::ntt(a*b mod Phi) == NTT(a) o NTT(b)` is checked through the SIMD products.

Transform values are lazily reduced signed residues stored as `i16`: |v| <= 7.5 q for q = 3889
and <= 2.30 q for q = 9721 (`vertical_bin_asm`); `RingElement::normalized` maps to [0, q). The
transforms can also be produced in Montgomery form (times 2^16 mod q, same cost), in which a
slot-wise product of two transforms is one Montgomery multiplication; the commitment does not
need that, since it accumulates raw products.

### The lift is a ring extension of degree 4

With Y = X^4 the ring element is a(Y) + X b(Y) + X^2 c(Y) + X^3 d(Y). In R_648, Y satisfies
Y^162 - Y^81 + 1 = 0, so Z_q[Y] is the 486-th cyclotomic ring S, isomorphic to the 243-rd
cyclotomic ring R_162 = Z_q[Z]/Phi_243(Z) via Z = -Y (mod 2 the sign disappears and S is F162
with Y = x, which is what makes the plain interleave a genuine lift). R_648 = S[X]/(X^4 - Y) is
a free S-module of rank 4 with basis 1, X, X^2, X^3. Consequently the commitment over R_648,
read in that basis, is a commitment to the 2^18 lifted F162 elements as elements of R_162 with a
matrix of height 4 and width 2^18 whose 4 x 4 blocks are the Y-twisted circulants of
multiplication by the uniform A_i:

    [ a0   Y a3  Y a2  Y a1 ]
    [ a1   a0    Y a3  Y a2 ]      A_i = a0 + X a1 + X^2 a2 + X^3 a3,  a_l in S uniform.
    [ a2   a1    a0    Y a3 ]
    [ a3   a2    a1    a0   ]

That is ring-SIS over R_648 presented as a structured rank-4 module-SIS over R_162; the packing
buys a factor 4 in randomness and multiplications over an unstructured rank-4 matrix. Any
R_162-linear operation on the witness acts on the four components independently, i.e. it is
R_648-linear too. The same decomposition gives NTT(e)[u] = sum_k psi^{uk} A_k(u mod 486): four
length-162 transforms of the four elements plus a radix-4 recombination per slot, which costs
exactly the same 6480 multiply-port uops per 32 ring elements as the tree above, so the tree
was kept and only the front end reads the interleaved layout.

### Types

* `bin_fields::scalar::F162` — the input (`[u64; 3]`, bit 64k+j of limb k = coefficient of
  x^(64k+j), top 30 bits of limb 2 zero, 24-byte stride).
* `Batch32 { v: [[i16; 32]; 648], representation }` — 32 ring elements in the "vertical"
  layout: `v[j][p]` is slot (or coefficient) j of element p, so one 512-bit vector is one slot
  of all 32 elements. 64-byte aligned, 41 472 bytes. Lane p of batch b is ring element 32 b + p,
  i.e. F162 elements 4(32b + p) .. +3. The commitment matrix A is stored in the same layout,
  centered (`a[b].v[j][p]` = slot j of A_{32b+p} in [-(q-1)/2, (q-1)/2]).
* `RingElement { v: [i16; 648], representation }` — one element, rokoko-style flat array plus
  representation tag; `Batch32::get / set`.
* `BinaryIndex32` — the kernel's input: 162 rows of 64 bytes, `row[i][2p] = n`,
  `row[i][2p+1] = 16 + n`, n = the 4-bit nibble (c_i, c_{i+162}, c_{i+324}, c_{i+486}) of ring
  element p, which is exactly the `vpermb` index the fused first two levels consume.
* `BinaryPoly { bits: [u64; 11] }` — a plain 648-coefficient 0/1 polynomial (bit i = coefficient
  i), the coefficient-domain form of a lifted ring element used by the tests and the comparison
  kernels' benches; `transpose.rs` slices it into `BinaryIndex32` with the same GFNI machinery.
* `HBatch4 { v: [[i16; 32]; 81] }` — the horizontal layout, 4 elements per batch, one per
  128-bit lane; A for `commit_h` is stored in this layout too.

### Arithmetic

Signed Montgomery multiplication by a constant with a precomputed companion
(`mullo(a, w')`, `mulhi(a, w)`, `mulhi(m, q)`, `sub`: three uops on the multiply port), the
radix-3 butterfly `t1 = zeta a1, t2 = zeta^2 a2, u = omega (t1 - t2); y0 = a0 + t1 + t2,
y1 = a0 - t2 + u, y2 = a0 - t1 - u` (3 multiplies, 10 add/sub), and lazy reduction under the
invariant |x| < 2^15 tracked per level (budget 8.42 q for 3889, 3.37 q for 9721; a Montgomery
product is bounded by |a| q / 2^17 + q/2 < 0.75 q). Reductions: a two-uop `vpmulhrsw` Barrett
(|r| < 0.9 q, exhaustively verified) and, for 9721, a lookup Barrett on the shuffle port
(|r| <= 0.579 q, see the kernel notes). All bound tables are `const` recursions checked by an
i32 shadow model in the tests.

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
(each level-3 output feeds a different level-4 twiddle) and leaves the folded inputs at 3q
instead of 0.75q, which breaks the 2^15 budget for both primes — the Barretts that then become
necessary cost more than the multiplies saved.

## The commitment (`simd/commit.rs`)

* **Raw products, rare reductions.** An inner product needs the sum of the products mod q, not
  each product mod q. `vpdpwssd` accumulates 16 x 16-bit products pairwise into 32-bit lanes —
  one multiply-port uop per slot vector and batch, against five for a Montgomery product plus
  add — and the i32 lanes absorb 16 batches (3889) or 8 batches (9721) of products before a
  fold-back, given |W| <= 7.5 q / 2.29 q (the transform's proved output bound) and
  |A| <= (q-1)/2. The fold-back is exact: write a lane as 2^16 h + u with u the unsigned low
  half; reading the low half as a signed word l gives u = l + 2^16 c with c its sign bit, so
  x = l + (h + c) 2^16 == l + (h + c) R (mod q) with R = 2^16 mod q — one `vpmaddwd` against
  the constant pair (1, R) and a masked add of R where the sign bit is set, three uops per
  vector, leaving |acc| <= 2^15 (1 + R). The bounds are asserted at compile time and replayed in
  i64 against the real kernel output on adversarial inputs in the tests.
* **Consume each block as it is finished.** The kernel produces 27 finished slot vectors per
  asm block; a hook multiplies them against the corresponding 27 vectors of A and accumulates
  right there, so the transform output never reaches memory and no output store or reload is
  paid. Only A streams: 1296 B per ring element, read once.
* **Packed accumulator.** 648 slots x 16 i32 lanes would be 41 KB and is swept twice per batch;
  folding each slot to 8 lanes and packing two slots per vector (21.5 KB, two `vshufi64x2`
  per two slots) keeps it in L1 and is worth 28 of the basemul's 86 cycles. The sweep cost is
  2 x accumulator / batch width per ring element, which is the reason a *wide* batch is what
  makes the accumulator affordable: at width 4-8 the same traffic is amortised over 4-8
  elements instead of 32.
* **A is prefetched under the transform.** The next batch's 648 A lines are prefetched into L2
  (`prefetcht1`, one batch ahead) spread over the 24 block boundaries of the current batch;
  the same prefetches issued as one burst are worth nothing, and `prefetchnta` is a disaster
  (the lines must survive in L2 until the accumulate reads them). Worth 600 -> 462 cycles per
  element; the remaining ~90 cycles are A arriving at ~11 GB/s.
* **Where the operands live, measured.** Basemul alone with the transform operand in L1 (a
  27-slot block) vs L2 (a 41 KB batch): 60 vs 85 cycles per element, against a 39-cycle uop
  floor — the L1 residency of the transform output is worth ~5 % of the commitment; the
  accumulator's residency and the prefetch schedule are what matter.
* `commit_2q` runs both primes off one front end (two accumulators, 170 MB of A).

### The horizontal commitment (`simd/commit_h.rs`)

Gregor Seiler's suggestion: transform only 4 elements at a time in the horizontal layout
(`HBatch4`, 5 KB per group) so that the transform output, its A rows and the accumulator are
all L1-resident and the basemul runs from L1 with no output store. Implemented in full: a
front end that expands the interleaved F162 bits into the stride-81 lanes (per polynomial two
`vpermi2b` + two `vpsrlvd` + two `vpermi2b` + two `vgf2p8affineqb` + two `vpermb`, then
`vpunpck` interleaves and one `kmovd` + zero-masked `vmovdqu16` per register; 44 cycles per
ring element), `ntt_gen_hbatch4` (546 / 593 cycles), and a basemul with `vpdpwssd` on lanes
permuted to `4j + p` so that adjacent lanes are the same slot of two elements (A is stored
pre-permuted, 46-56 cycles per element from L1), with an exact f32-rounded reduction every
18 / 10 groups. Measured 842 / 898 cycles per ring element (12.3 / 13.1 ms): the locality
claim holds — the basemul is ~12 cycles cheaper than the vertical one — but the transform
costs ~270 more because the lookup trick does not survive the horizontal layout, and the A
stream overlaps worse (35 % hidden vs 70 %) because A is consumed in an 81-line burst per
group with the transform an opaque call in between, so there is no loop in which to spread
prefetches. Groups of 8 or 16 elements per step change the total by at most 1 %; 2 MB pages
for A buy 2 %.

## Strategies taken in each kernel and driver

### `simd/transpose_f162.rs` — F162 stream -> index rows (32 cycles per ring element)

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

Port-5 bound (26 of 32 cycles). Measured and rejected: replacing phase 3 by merge-masked
loads (35 cycles: a masked `vmovdqu64` costs ~1 extra ALU uop on this core), software
prefetch of the input (noise), slicing the next batch inside the previous batch's kernel (the
shuffle chains at the head of the reorder buffer starve allocation; a wash).

### `simd/vertical_bin.rs` / `simd/vertical_bin_asm.rs` — the binary kernel (280 / 305 cycles)

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
  times — with a `[9 multiplies + 4 subs of butterfly k][6 adds of butterfly k-1]` block
  schedule that lowers the share of flexible adds the allocator sends to the saturated port 0
  from ~27 % to ~23 %. Measured alternatives: natural order +8, 1-mul:1-add pipeline +3, all
  loads up front or all stores deferred +6. The kernel exposes a per-block hook (`BlockSink`)
  that the commitment uses to consume each block's 27 output vectors while they are in L1; the
  plain entry points pass a no-op sink and compile to the same code. The block loop must not be
  specialised on the block index by the compiler (the 26 KB of code would become 100+ KB and
  i-cache misses cost 44 cycles per element).
* **Reduction.** Table entries are centred (|T| <= q/2), so level-2 outputs are < q and level-3
  outputs < 3 q. For 3889 nothing else is needed (output <= 7.5 q declared, 4.4 q observed). For
  9721 the un-twiddled a0 input of level 4 gets a **lookup Barrett**: `vpmultishiftqb` pulls the
  5-bit quotient window (bits 11..15) of the value into both bytes of the lane, `vpandd` /
  `vpord` turn it into the (n, 32 + n) index pair, one `vpermb` reads the nearest multiple of q
  from a 64-byte byte-split table, one `vpaddw` subtracts it — no multiply-port uop, and
  |r| <= q/2 + 2^10 = 0.579 q (exhaustive), tighter than the multiply Barrett's 0.809 q. That
  tightness is the real gain: level 5 then needs no reduction at all, and level 6 keeps the
  two-multiply Barrett on its a0 only to fit the output in i16. 432 reductions per batch instead
  of 648, output <= 2.294 q (worst case proved over all i16), 318 -> 305 cycles. Lookup
  Barretts at all three levels measure 316: the reduction itself is not what costs, dropping
  level 5's is.
* **Montgomery form.** `ntt_bin_batch32_mont` uses tables scaled by 2^16 mod q (re-centred):
  same schedule, same uops, same bounds, outputs times R. A product of two such transforms is
  one Montgomery multiplication and stays in the form; `scalar::intt_mont` absorbs R^-1 with
  648^-1.
* **Compiler workarounds.** stdarch's `_mm512_mulhi_epi16` lowers through a
  `vpmovsxwd / vpmovdw / vinserti64x4` round trip that LLVM rematerialises inside the hot loops;
  `vpmulhw` and the twiddle broadcasts are emitted with `asm!` (`pure`, `nomem/readonly`).
* **Tried and rejected.** Folding omega or the level-4 twiddles into the tables (see above);
  fusing level 4 into the lookup loop (34+ live registers); ymm arithmetic (36 % slower: the
  multiplies then also occupy p1 and the adds follow them); non-temporal stores for
  cache-resident output.
* **Where it stands.** Static port floor 256 / 281 cycles per element; measured 280 / 305 with
  the multiply port 95 % busy. `perf stat` top-down on the kernel loop: ~48 % of issue slots
  retiring, ~49 % back-end bound (the two vector ALU ports), 1.6 % front-end, 1.2-1.6 %
  bad speculation — no branch, decode or microcode issue to fix.

### `simd/ntt_f162.rs` — NTT drivers

`ntt_f162` slices 128 F162 into index rows (two uninitialised 10 KB buffers used alternately),
runs the kernel with non-temporal stores on the last level, repeats per batch. `ntt_f162_2q`
runs both primes off one slicing (-21 cycles per transform). `ntt_f162_stream` hands each
batch to a closure; `ntt_f162_mont` produces Montgomery-form outputs. Driver variants measured
and rejected: huge pages for the output (TLB walks 0.31 -> 0 per element, time unchanged — they
hide behind the NT stores), producer-supplied index rows (the extra 58 MB of DRAM reads cost
more than the slicing saved), `sfence` per batch, batch grouping, cached stores (+200 cycles),
input prefetching (noise to +45 with the NTA hint).

### `simd/vertical_gen.rs` — generic-input vertical kernel (423 / 467 forward, 541 / 601 inverse)

Same layout and arithmetic, any i16 input with |x| <= q. Levels 0 and 1 fused into one radix-4
pass over the 648 vectors (41 KB streamed once from L2), levels 2 and 3 fused over groups of 6
vectors, then one pass per 27-block for each of levels 4, 5, 6; zero shuffles; Barretts: one
level for 3889 (a0 of level 5, output <= 3.40 q), for 9721 inside the first pass and on the a0
of levels 3-6 (output <= 2.12 q). Radix-9 / radix-27 fusion measured and rejected (439-468 vs
423: multiply-port bound, longer dependency chains); the variants stay behind
`ntt_gen_batch32_plan`. Out of cache: one prefetch line per level-4/5/6 butterfly for the next
batch is worth 23 %; 162 prefetches at once are 18 % slower than none. `ntt_gen_batch32_mont`
gives Montgomery-form outputs for +1 % (one multiplication by R replaces the level-5 Barrett).

`intt_gen_batch32` is the inverse, the same five passes in the opposite order with
Gentleman-Sande butterflies — `u = omega (y2 - y1)`, then `(y0+y1+y2, (y0-y1+u) zeta^-1,
(y0-y2-u) zeta^-2)`, which costs exactly what the forward radix-3 costs. The per-level 1/3 and
1/2 are not applied, so every value reaching level 0 is 324 times the true one and the whole
1/648 corrected by the Phi_6 determinant (`scalar::intt`'s `inv2 / inv3 / det` chain, multiplied
out) sits in the three constants of the level-0 recombination `a1 = KA (Y0-Y1)`,
`a0 = KB (Y0+Y1) + KC (Y0-Y1)`; no scaling pass. Input: any lazily reduced transform this crate
produces (7.5 q / 2.3 q); output: coefficients, fully reduced and centered, which is what the
fold needs. The extra 118 / 134 cycles over the forward are three equal thirds — the Phi_6
inverse is a general 2x2 matrix (3 Montgomery products per level-0 butterfly against 1), the
untwiddled output `y0+y1+y2` triples what it is given instead of being a reduced Montgomery
product, so it needs 984 / 1836 Barretts against the forward's 216 / 1026, and the 648 outputs
are centered. Barrett placement is per position class, indexed by the loop variable that names
it, chosen by exhaustive search over that flag set and replayed by a `const` recursion that
proves every intermediate stays inside i16 (peak 8.09 q of a 8.43 q budget for 3889, 3.24 q of
3.37 q for 9721); the tests replay the same schedule in i32.

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

### `simd/pointwise.rs` — products in the NTT domain

General Montgomery product of two batches (4 multiplies per slot plus a second multiplication
by 2^32 mod q for plain-form inputs; one multiplication for Montgomery-form inputs), and batch x
fixed element (3 multiplies per slot; the element's Montgomery form and companion are stored as
duplicated u32 so the broadcasts are pure loads).

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

Measured or modelled on this core, roughly in order of value for the commitment:

1. **Hide the rest of the A stream.** The commitment is at ~80 % of the roofline with A at
   ~11 GB/s; the prefetches are issued only at the 24 block boundaries per batch. Interleaving
   them inside the asm radix-3 passes (one line every ~30 cycles) should recover most of the
   remaining ~90 cycles per element, i.e. ~20 %. Needs a modified asm block.
2. **Incomplete NTT.** Stopping one level early (slots of degree 3, `Z_q[X]/(X^3 - c)`) removes
   1944 of the 6480 multiply-port uops per batch (~30 % of the transform) at the price of a 3x
   more expensive slot product; for the commitment (one raw product per slot) the product side
   would rise from ~30 to ~90 uops per element, so the net is roughly -60 cycles per element.
   Changes the output's meaning; depends on what the consumer needs.
3. **Hand-scheduled level 3 and the lookup loop** in asm like levels 4-6: a few percent.
4. **Per-element bit permutation** as the slicer (VBMI2 funnel shifts + `vpmultishiftqb` +
   `vgf2p8affineqb` + `vpermb` on each F162 separately, then a byte transpose): ~10 uops per
   F162, perhaps 20 instead of 32 cycles per ring element.
5. For q = 9721 the remaining reductions cost ~25 cycles per element; inherent to a 14-bit
   prime in 16-bit lanes (2^15 / q = 3.37). 3889 is the only fully splitting prime below 2^13.
   Multithreading was excluded by design; the commitment is compute-bound with ~60 % of the A
   stream hidden, so two cores would already saturate DRAM.

## Layout of the crate

    src/api.rs                  the public API: CommitmentKey, PowerOfThreeRingElement(WithTwoLimbs), VerticallyAlignedMatrix
    src/challenge.rs            short fixed-weight ternary challenges over R_162, blake3 transcript
    src/fold.rs                 the folding step: v = sum_j c_j W_j in the NTT domain, and A v
    src/eval.rs                 the left-expansion over F162 (Pi_translate), the binary fold, the verifier
    src/main.rs                 the demo: commits 2^18 F162 for r = 1, 4, 16, 256, folds the r = 256 run and verifies it
    src/params.rs               ring constants, twiddle tables, Montgomery/Barrett constants (const-evaluated)
    src/f162.rs                 the F162 lift (lift4, pack4, scalar index rows, random elements)
    src/types.rs                Batch32, RingElement, BinaryPoly (test/comparison input form)
    src/scalar.rs               exact reference: product mod Phi_1944, NTT, inverse NTT, evaluation
    src/simd/transpose_f162.rs  F162 x 128 -> BinaryIndex32 (GFNI + VBMI)
    src/simd/transpose.rs       BinaryIndex32 and the GFNI bit-slicing of plain 648-bit polynomials
    src/simd/vertical_bin_asm.rs the binary kernel (LUT + folding, asm levels 4-6, block hook) — production
    src/simd/vertical_bin.rs    the same kernel in intrinsics (reference for the asm one)
    src/simd/ntt_f162.rs        NTT drivers for the F162 input (single prime, both primes, streamed, Montgomery form)
    src/simd/commit.rs          the Ajtai commitment on the vertical kernel (block-fused, VNNI accumulation)
    src/simd/commit_h.rs        the commitment in the horizontal layout (L1-resident groups of 4)
    src/simd/vertical_gen.rs    generic-input vertical kernel, forward and inverse
    src/simd/horizontal_gen.rs  generic-input horizontal kernel (HBatch4)
    src/simd/pointwise.rs       slot-wise Montgomery products
    src/perf.rs                 perf_event_open counters (cycles, instructions, uops, ports 0/1/5)
    src/bin/bench_commit.rs     the headline benchmark;  bench_commit_h.rs, bench_f162.rs, bench_*.rs
    src/bin/kernel_loop.rs      one kernel in a tight loop, for perf stat / perf record
    tests/*.rs                  correctness;  tools/  C microbenchmarks (ports, instruction table, DRAM)
    DESIGN.md                   design notes: tree, arithmetic, bounds, port facts, kernel APIs

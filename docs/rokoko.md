# The recursive opening over rokoko

This note fixes the design of `src/rokoko`, the second backend of the recursive opening. It proves
the relation of `src/recursion` with [rokoko](https://github.com/lattice-arguments/rokoko) (a git
dependency pinned to its `main`, behind the `rokoko` feature, nightly only) instead of LaBRADOR. Read `docs/snark.md` in that repository first: everything below is written against
its claim language.

## The relation

Unchanged from `src/recursion/mod.rs`. Over `S = Z[Z]/Phi_243(Z)`, degree 162, with `R_648` the
free rank-4 `S`-module of the folded witness:

- per RNS limb `q` and component `m`: `sum_{l,i} F^{(k,tw)}_i v_{i,l} - sum_j c_j C_{q,m,j} = q k_{q,m}`;
- the binary fold: `sum_{i,l} lift(eq(p0)_{4i+l}) v_{i,l} - sum_j c_j lift(u_j) = 2 w_b`;
- the binary evaluation: `sum_j lift(eq(p1)_j) lift(u_j) - lift(t) = 2 w_e`.

Every identity holds over `Z`. A quotient `k` (or `w`) is a witness `S`-element split into balanced
gadget digits, each digit entering as one more term with a constant public multiplier
`-divisor * base^d`. The public multipliers (`F`, the lifts, the challenges `c_j`, the constants)
are `S`-elements the verifier holds; the witness `S`-elements are `v`, the residue digits, the lift
`u` and the quotient digits.

## Encoding an `S`-identity as ring products

Rokoko's ring is `R = Z_q[X]/(X^128 + 1)`, `q = 2^50 - 2687`. An identity `sum_t g_t s_t = out`
in `S` is checked through chunks and diagonals exactly as `src/recursion/chain.rs` does for
LaBRADOR, with the geometry retuned to degree 128 (constants in `src/rokoko/mod.rs`):

| | LaBRADOR | rokoko |
|---|---|---|
| ring degree `DEG` | 64 | 128 |
| chunk `CHUNK` | 54 | 81 |
| chunks per `S`-element | 3 | 2 |
| block width `SUB` | 9 | 27 |
| diagonals `BLOCKS` | 18 | 6 |
| carry coefficients `CARRY` | 53 | 80 |

A witness `S`-element `s` is two committed elements, chunk `b` holding coefficients
`[81 b, 81 b + 81)` at positions `[0, 81)`. A public `S`-element `g` yields, for chunk `b` and
diagonal `a`, the block `g_{b,a}` = coefficients `[27 a, 27 a + 27)` of `Z^{81 b} g mod Phi_243`.
Then `g s mod Phi_243 = sum_a Z^{27 a} D_a` with `D_a = sum_b s_b g_{b,a}` a plain polynomial
product of degree at most `80 + 26 = 106 < 128`, so it is the ring product. The carries `e_a`
(`CARRY` coefficients each) turn the identity into `BLOCKS` block equations

```
D_a + e_{a-1} - Z^27 e_a - [a = 0] e_5 - [a = 3] e_5 = out_a        (a = 0..5)
```

`out_a` the 27 output coefficients of diagonal `a`, the last two terms the wrap
`Z^162 = -Z^81 - 1` of the last carry, `81 / 27 = 3`. Carries are gadget digits too: `e_a =
sum_d base^d e_{a,d}`, one committed element per `(a, d)`. Every term is a ring product of a
public weight of degree below 28 (a block, `+-Z^27 * base^d`, or a scalar) and one committed
element, so each block equation is a degree-1 sumcheck claim with a ring-element value.

Everything committed has its coefficients `[81, 128)` zero (`SUPPORT = 81`; honest carries use
only 80 and the pattern is shared so one support claim covers all). That is what makes the ring
equation a polynomial identity: no term reaches degree 128.

## Coefficient budget

Every committed coefficient is a balanced base-128 digit, `|x| <= 64`, the regime rokoko's `EN`
chains are calibrated for:

- `v_{i,l}` (folded witness components, up to a few thousand): two digits, `v = v^0 + 128 v^1`,
  each a witness `S`-element; the term `F v` becomes `(F, v^0)` and `(128 F, v^1)`.
- residues `C_{q,m,j}`, `|C| <= (q - 1) / 2 <= 9720`: two digits, the high one up to
  `(q - 1) / 256 + 1 <= 77`.
- `u_j`: bits, one digit.
- quotients and carries: base-128 gadgets, levels from `Gadget::covering` against the calibrated
  magnitudes (`src/recursion/limbs.rs`, `binary.rs`); the carry magnitudes must be re-measured for
  `SUB = 27`.

## The committed vector

One `rokoko` witness of `N` elements, `N` a power of two, built with `WitnessBuilder`: each
vector of the relation (in the sense of `recursion::Build`: `v^0`, `v^1`, `C[q][m]^0`, `C[q][m]^1`,
`u`, `k[q]`, `e[q]`, `w`, `e[w]`, ...) is one `push` of its elements padded to a power of two;
pad elements are zero and no claim reads them. `Layout` records the regions. Each vector has a
cap (`Cap::PerCoefficient(64)` for digits, `(77)` for residue high digits, `Betasq` for `v` in
terms of the fold cap) and one norm claim.

Sizes at the basic recursive shape (`n = 256` ring elements per column, `r = 256` columns,
limbs 3889 and 9721): `v` 16 n = 4096, residues 16 r per limb = 8192, `u` 2 r = 512, quotients
and carries a few hundred; total below 2^14, so `N = 2^14`. The medium shape (`n = r = 512`)
needs `2^15`; the large one of `main.rs` (`n = r = 1024`, three limbs) `2^17`.

## Claims

All built identically on both sides from the transcript after the commitment is absorbed.

1. **Block equations.** Per chain, one claim: the six diagonals combined with a transcript
   ring element `rho`, `sum_a rho^a (block equation a)` (a nonzero difference survives in some
   NTT slot with probability at most `5 / q^2` per slot). The weight of a committed element is the
   ring element `sum_a rho^a * weight_a` (its per-diagonal weights from `Relation`), so a chain is
   `sum over regions of table(weights).on(region.vars()) * witness_in(region)` with value
   `sum_a rho^a out_a` (public: zero except the evaluation chain's `lift(t)`). Ten claims at two
   limbs. Ring-element tables: the verifier's cost is linear in the key, as with LaBRADOR's `phi`.
2. **Outer commitments.** `T_Y` commits the residue digit vectors and `T_u` the lift, both under
   *tensor-structured* Ajtai keys over `R` (per row and region, `log2(region)` random ring
   elements expanded as `PreprocessedRow::from_layers`, the shape rokoko's own CRS uses), derived
   from the setup seed. Each key row is one claim, the sum over its regions of
   `eq(layers).on(region.vars()) * witness_in(region)`, equal to that row of `T`; verifier cost
   `O(log)`. `T_R` is gone: rokoko's commitment binds the rest. The binding of such a key is the
   structured Module-SIS assumption rokoko's chain already rests on; the ranks are set by
   rokoko's estimator as if the key were uniform (rank 2 for `T_Y` at 147 bits and rank 1 for
   `T_u` at 196 bits at the basic shape, the Sage lattice estimator agreeing on the former).
3. **Support.** Three claims `table(alpha_k) * witness()` with `alpha_k` transcript scalars in
   `Z_q` (zero on pad indices), each shipping a ring element `V_k = sum_i alpha_{k,i} w_i`; the
   verifier checks coefficients `[81, 128)` of every `V_k` are zero. Soundness `1/q` per claim,
   `q ~ 2^50`, three for 128 bits.
4. **Norms.** Per vector `r`: `witness_in(r) * witness_in(r).conjugate()` summing to a shipped
   value whose constant coefficient is `||w_r||^2`; the verifier checks it against the cap. The
   fold's own cap, D5 of the plan, is enforced on the triangle bound `(||v^0|| + 128 ||v^1||)^2`
   the verifier forms from the two shipped digit norms, at `round::FOLD_CAP` per ring element
   and challenge; the prover refuses a round over it, to be retried with fresh challenges.
5. **Binariness of `u`.** `witness_in(u) * witness_in(u).conjugate() - table(conj(J)).on(u.vars())
   * witness_in(u)`, `J = sum_{j < 81} X^j`, shipping a value whose constant coefficient must be
   zero: `sum x (x - 1) = 0` over integers.

Conjugates make it a two-opening statement, so the chain is compiled for `nof_openings = 2`.

## Soundness

- The block equations hold mod `q` in `R`; with the support claim every term has degree below
  128, so they hold mod `q` as polynomials.
- The no-wraparound bound of `src/recursion/bound.rs`, recomputed by the verifier from the
  layout's caps and the public weights, keeps every coefficient of every block equation's
  integer left side below `q / 2 = 2^49`, so they hold over `Z`; that is the whole chain
  argument of `chain.rs`, and the `S`-identities follow.
- The caps are exact statements about the extracted witness: rokoko's exact-norm chain certifies
  the aggregate `l2` norm, which keeps every shipped constant coefficient below `q / 2`. This
  needs the chain's norm bounds to be *enforced*: rokoko's `assert_norm_bounded` used to only
  report a violation; since lattice-arguments/rokoko#111 a norm over its bound rejects the proof
  unless the `soft-norms` feature (implied by `calibration`) is on, and the dependency is pinned
  past that commit.
- `T_Y` and `T_u` bind under Module-SIS at length twice the residue and lift caps; their ranks
  come from rokoko's estimator (`common::estimator::estimate_rsis_security`) at 128 bits.
- The chain's own commitments and openings are estimated per level by rokoko's `debug-hardness`
  feature (`rokoko-hardness` here); every level must clear 128 bits.

## Transcript

The rokoko `HashWrapper` starts from a 32-byte digest of the round's `Transcript` (which has
absorbed the parameters, `T_Y`, the evaluation point, `T_u`, the claim), then absorbs the rokoko
commitment, then the claims are built. Shipped values (`V_k`, norms, binariness) are absorbed by
`prove_claims`.

## Configs

`src/rokoko/config.rs`: exact-norm chains `N14` to `N17` for `N = 2^14` to `2^17`, hand-drafted with
rokoko's `AuxSumcheckConfig` builders (`generate_config`), two openings, `exact_projection_norm`
at the root. Every round projects *fine*: rokoko's coarse projection reads
`projection_height * projection_ratio` rows per block and is empty below a height of `2^13`,
which none of these roots reach. Each chain carries its norm table, the raw maxima of one
real-pipeline round (`cargo +nightly run --release --features rokoko-calibration,<size>`), with
rokoko's `NORM_MARGIN` on top, and the per-level security `rokoko-hardness` prints on the real
witness, all at or above 128 bits; the tables in the doc comments say which.

## Results, one core

| shape | committed vector | proof | prover | verifier | LaBRADOR proof |
|---|---|---|---|---|---|
| small (2^18, 256 columns, 3889 + 9721) | 2^14 | 100.4 KB | 0.53 s | 0.08 s | 79.5 KB |
| medium (2^20, 512 columns, 3889 + 9721) | 2^15 | 108.0 KB | 1.05 s | 0.16 s | |
| large (2^22, 1024 columns, 3889 + 2917 + 4861) | 2^17 | 110.8 KB | 3.18 s | 0.52 s | |

Everything that depends on the key alone is built at key time: the blocks of every twisted key
row in NTT form (`Setup::polys`, 49152 at the small shape, 25 MB per limb), the expanded rows
of the outer keys, the CRS. A round transforms only its own polynomials (challenge and lift
blocks) and forms each table entry as a few multiply-adds under the powers of `rho`. At the
small shape the verifier's 79 ms are the tables (34 ms, the linear read of the key), rokoko's
`verify_claims` (15 ms), the no-wraparound bound (10 ms), rokoko's chain (6 ms) and the layout
(3 ms). `ROKOKO_TIMINGS=1` prints the breakdown.

Security on the real witness (`rokoko-hardness`), the lowest level of each chain: 137 bits for
`N14`, 137 for `N15`, 134 for `N17` (all basic commitments of the root or a middle round; every
recursion level is at 157 or above). Peak resident set 0.5, 1.2 and 3.8 GB.

The proof is `T_Y` (1.6 KB at rank 2), `T_u` (0.8 KB), the shipped claim values, rokoko's claims
sumcheck and its chain, whose last round ships the folded vector in the clear and is the floor
of about 63 KB. Nine tenths of the prover is rokoko's fine projection.

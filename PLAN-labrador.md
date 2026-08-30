# Recursing the folded opening into LaBRADOR

Branch `labrador`. Status: plan for approval — nothing below is implemented. Reviewed once by Codex (xhigh);
its findings are folded in, see the end.

## 1. What changes, in one paragraph

Today the verifier receives the whole commitment (256 columns × 648 slots × 2 limbs ≈ 663 KB)
and the folded witness `v` in the clear (≈ 207 KB), and checks `A v = Σ_j c_j C_j` on every
limb, `‖v‖` small and `eq(p0)·(v mod 2) = Σ_j u_j (c_j mod 2)` itself. After this change the
commitment is a LaBRADOR-style Ajtai commitment `T_C` to the digits of the `C_j` (a few KB),
the left expansion `u` is still sent in the clear (5 KB), the prover sends the *lifted*
amortised left expansion `z = Σ_{i,l} lift(eq(p0)_{i,l}) · v_{i,l}` over the integers (0.7 KB),
and everything else — `‖v‖²` exactly, `A v − Σ_j c_j C_j ≡ 0 (mod q_i)` for both limbs with the
wraparound `k_i = (A v − Σ_j c_j C_j)/q_i` committed and short, the digits of `C` opening
`T_C`, and `z` being that exact integer combination of `v` — is one LaBRADOR proof. The
verifier's only field arithmetic left is `z mod 2 == Σ_j u_j (c_j mod 2)`.

## 2. The relation proven inside LaBRADOR

Notation. `R_648 = Z[X]/(X^648 − X^324 + 1)`, `Y := X^4`, `S := Z[Y]/(Y^162 − Y^81 + 1)`
(this is `R_162` in the basis `Y = −Z`), so `R_648 = S ⊕ X S ⊕ X² S ⊕ X³ S` with `X^4 = Y`. An
element `a ∈ R_648` has components `a_0..a_3 ∈ S`; the packing `X^{4m+l} ↔ bit m of F162
element l` means `v_{i,l} mod 2` *is* the `l`-th packed `F162` element of ring element `i`,
and `S/2S = F162` because `Y^162 + Y^81 + 1` is the `F162` modulus.

Public: the key rows `A^{(q)} ∈ R_648^256` in coefficient form, centered mod `q` (the inverse
NTT of the row the kernel uses — computed once per key, by both parties, from the seed), the
challenges `c_j ∈ S` (ternary, weight 21; `c_j(Z)` rewritten in `Y = −Z`), the lifts
`E_{i,l} := lift(eq(p0)_{i,l}) ∈ S` (0/1 coefficients), the commitment `T_C`, the value `z ∈ S`
sent by the prover, and the exact squared norms of every witness vector.

Witness (all vectors of small integers, coefficients ≤ 2^14 so LaBRADOR's `int16`/`vpdpwssd`
norm routine cannot overflow — the 23170 rule from rokoblador):

| vector | content | polys of `R_Q` | coefficient size (honest) |
|---|---|---|---|
| `v` | folded witness, `v_{i,l} ∈ S`, 1024 elements of 162 coefficients | 3072 | std ≈ 73, `‖v‖∞ ≤ 1944` |
| `Y_q` | the balanced residues of every `C_j^{(q)}` themselves (RNS, no digit decomposition), one vector per limb, split in two for 9721 | 6144 | ≤ 1944 / ≤ 4860 |
| `d_k` | 2 base-512 digits of `k_q ∈ R_648`, one per limb | 48 | ≤ 256 |
| `e` | the carries of §3, 3 (3889) or 4 (9721) digits each | 540 | ≤ 512 |

"Digits" are unrestricted signed vectors: no per-coefficient range is proven or needed
(`C_j^{(q)} := d_0 + B d_1 mod q` is a residue for any digits, `k` and the carries are free
integers), and every bound in §5 is taken over the `ℓ2` caps with the concentration a cheater
can achieve. Every digit level whose magnitude matters in §5 is its own LaBRADOR witness
vector, so `r ≈ 18` (`v`; two levels of `d_C`, `d_k` and three or four of `e` per limb; two
levels of the `z`-carries). Total ≈ 10k ring elements of `Z_Q[X]/(X^64+1)`, `Q = 2^40 − 195` (`LOGQ = 40`, see §5 for
why nothing smaller works).

Constraints (all over `Z_Q`, and each one is a *true integer identity* because the norm caps
of §5 keep every left-hand side below `Q/2`):

1. **Opening of `T_C`.** `T_C = ⟨key_C, Y⟩` over the residue vectors (their coefficients are
   below `q/2`, small enough for LaBRADOR directly; `‖Y_q‖² ≈ n·q²/12`, i.e. 2^37 and 2^40,
   which is why 9721's vector is split in two under the per-vector ceiling 2^39, and why
   `κ_C ≈ 12` rather than 9) in LaBRADOR's own commitment form (the
   truncated extension product `polxvec_mul_extension(·, key, s, n, κ, 1)` that `commit_raw`
   uses; `κ` need not be a power of two, LaBRADOR's own `κ` is not) — one constraint of
   extension degree `κ_C` whose `phi` *is* the key (aliased, no copy). `key_C` is expanded
   like `comkey` (`polxvec_almostuniform`) from its own domain-separated seed, so it is
   independent of LaBRADOR's internal key and of `key_R`. `κ_C` from `sis_secure` with
   LaBRADOR's own slack (`6·T·SLACK`): ≈ 9. `T_C` is computed at commit time, before any
   challenge, and *is* the commitment.
2. **Amortised commitment, per limb `q ∈ {3889, 9721}`.** In `R_648`:
   `Σ_i A_i^{(q)} v_i − Σ_j c_j · C_j^{(q)} − q · k_q = 0`, with `C_j^{(q)}` the committed residue
   and `k_q = d_{k,0} + 512 d_{k,1}`. Written per output component `m ∈ 0..4` it is four
   identities in `S`: `Σ_i Σ_{k+l≡m} Y^{[k+l≥4]} A_{i,k} v_{i,l} − Σ_j c_j C_{j,m} − q k_m = 0`.
   Encoded with the chunk machinery of §3 as 4 × 18 = 72 degree-1 constraints per limb.
3. **Lifted left expansion.** `Σ_{i,l} E_{i,l} · v_{i,l} = z` in `S` — 18 degree-1 constraints,
   inhomogeneous (`b` = the chunks of `z`).
4. **Zero parts.** Every chunk poly of `v`, `d_C`, `d_k`, `e` uses only its low coefficients
   (54, or 53 for carries); the rest must be zero. One constant-term constraint with random
   scalars, see §4.
5. **Norms.** The prover announces `betasq_i` for each witness vector — its exact `‖s_i‖²` —
   and Dachshund proves `‖s_i‖² + slack = betasq_i` with the slack a proven binary vector, so
   the verifier learns `‖s_i‖² ≤ betasq_i` (equality for the honest prover; a cheater may
   announce more, up to the caps). The verifier checks every `betasq_i` against the caps of
   §5, and all bounds are computed from the caps.

**Two commitments, one proof.** `T_C` and `T_R` are ordinary Ajtai commitments computed
outside LaBRADOR (`T_C` at commit time, before any challenge; `T_R` after the fold, before
the zero-mask randomness); inside the single LaBRADOR proof they are linear constraints on
the same witness vectors the chains use, so the proof shows that the digits it opens are the
ones committed before the challenges existed. LaBRADOR's own commitment to the whole
witness is made inside `prove`, once, when everything is known.

Outside LaBRADOR the verifier keeps `verify_evaluation` (`u · eq(p1) = t`) and checks
`z mod 2 == Σ_j u_j (c_j mod 2)` in `F162` (the existing `fold_row_evaluation`).

Why this is the same relation as today's `verify_folded_opening`: `z ≡ Σ E_{i,l} v_{i,l}`
over the integers and reduction mod 2 is a ring homomorphism `S → F162`, so
`z mod 2 = Σ eq(p0)_{i,l} · (v_{i,l} mod 2)` is exactly the binary check; (2) is exactly the
commitment check on each limb; (5) replaces the `‖v‖∞ ≤ 1944` test by an exact `ℓ2` norm.

### 2b. Any set of limbs

Nothing above is specific to 3889 and 9721. LaBRADOR sees a limb only through `A^{(q)}` and
`C_j^{(q)}` in coefficient form modulo `q` (for a quadratic-slot limb both come out of its own
inverse transform, `scalar::intt_quad` and the quadratic decomposition, once per key / per
commitment) and through three per-limb parameters derived from `q`: the digit base of `C`
(`⌈√q⌉`), the carry digits (from the honest product size `q·2^14` in std) and the
no-wraparound bound of §5. The code is generic over `Params.extra_moduli`; the base limb
3889 is always present; each limb adds `2·3·4·256 = 6144` digit polys, 72 chain constraints,
its carries and its `k`. The honest prover's abort uses the smallest limb present
(`‖v‖∞ < q_min/2`).

| limb | `C` digit base | carry digits | `A`-part bound | total bound (§5) | margin at `Q/2 = 2^39` |
|---|---|---|---|---|---|
| 2917 | 55 | 3 × base 1024 | 2^35.4 | 2^36.4 | 6× |
| 3889 | 63 | 3 × base 1024 | 2^35.8 | 2^36.8 | 4.6× |
| 4861 | 70 | 4 × base 256 | 2^36.2 | 2^37.2 | 3.5× |
| 9721 | 99 | 4 × base 256 | 2^37.2 | 2^38.0 | 2× |
| 12637 | 113 | 4 × base 256 | 2^37.6 | 2^38.4 | 1.5× |

All five limbs together: ≈ 35k witness polys, ≈ 550 MB of key-time `phi`, and `LOGQ = 40`
still clears every limb.

### 2c. Committing to the left expansion as well (D7, default on)

`u` need not be sent. Its lift `ũ_j ∈ S` (0/1 coefficients, 768 chunk polys) is committed as
`T_u = ⟨key_u, ũ⟩` (≈ 3 KB) in the transcript position `u` has now — before the folding
challenges are squeezed, which is the one ordering that matters — and LaBRADOR proves it
binary (`betasq = 0`). The two `F162` checks become `S`-identities over `Z` with 2 as one
more limb: `Σ_j lift(eq(p1)_j)·ũ_j − lift(t) = 2 w'` replaces `verify_evaluation`, and
`Σ_{i,l} E_{i,l} v_{i,l} − Σ_j c_j ũ_j = 2 w` replaces both the `z`-chain and the mod-2 check,
so `z` is never materialised. `w, w'` (≈ 2^12, 2^15) get two small digits each; magnitudes
≈ 2^28. Cost: +5 % witness, +18 constraints, one more key and degree-`κ` constraint, ≈ 25k
`phi` per proof (the `c_j` side sparse, the `eq(p1)` lifts dense); wire −5.2 KB − 0.7 KB
+ 3 KB, independent of `r`. The verifier is then: caps, the no-wrap bound, LaBRADOR verify.

## 3. Modelling `S` arithmetic in `Z_Q[X]/(X^64+1)`

LaBRADOR's linear constraint is `Σ_j phi_j · s_j = b` with negacyclic products in `X^64+1`
(64 `Z_Q` equations per constraint). A product of a witness chunk with support `[0, c)` and a
public chunk with support `[0, p)` is the *plain polynomial product* whenever `c + p ≤ 65`
— no wrap, so `Z[Y]`-arithmetic can be built from such products. I verified in `poly.c` that
the extension degrees (`deg > 1`) only give the ring `Z_Q[Y]/(Y^{64·deg}+1)`, which does not
help, so this is the only route; it is the "use half of the coefficients and prove the other
part is zero" idea, with the split chosen to make the `Y^162 = Y^81 − 1` wrap fall on chunk
boundaries.

**Chunking.** Witness `S`-elements are cut into 3 chunks of `c = 54` coefficients (positions
`0, 54, 108`); public `S`-elements into 18 sub-chunks of `p = 9`. Sub-chunk `a` of the public
element `G` times witness chunk `b` has degree ≤ 61 and belongs at offset `9a` of the product
(`G` here is already `A_{i,k} · Y^{54 b} [· Y] mod (Y^162 − Y^81 + 1)`, so the shift of the
witness chunk is folded into the public side and reduced once; `9 | 81` and `9 | 162` is what
makes the remaining wrap clean).

**Diagonals and carries.** For each output element (a fixed `(limb, m)` for (2), or `z` for
(3)) and each of the 18 diagonals `a = 0, …, 17`, let `D_a` be the sum over all contributing
pairs of the exact sub-products with offset `9a` (degree ≤ 61). Define the running remainder
by `T_a = D_a + e_{a−1} (+ wrap terms) − (output terms)`, `e_a := T_a div Y^9` (positions
9–61 of `T_a`, i.e. 53 coefficients), `e_{−1} = 0`. Constraint `a` is the 64-coefficient
identity

    D_a + e_{a−1} − Y^9 e_a − q k^{(a/6)}·[a ∈ {0,6,12}] + wrap_a = b_a

whose positions 0–8 say "the output window is right", positions 9–61 *define* `e_a`, and
positions 62–63 read `0 = 0`.
Here `q k` is the wraparound: `k_q = (A v − Σ_j c_j C_j)/q` over the integers, committed as
short digits, and `y_q` never appears explicitly — it is `Σ_j c_j C_j` with the digits of the
`C_j` as witness. The wrap: `e_17` sits at positions `≥ 162`, and `Y^162 = Y^81 − 1` puts `+e_17` into `T_9`
and `−e_17` into `T_0`. The witness chunks of `k` (and the public chunks of `z`) enter whole
at diagonals `0, 6, 12` with the scalar `phi = −q·512^d` (resp. `b_a`); their part above
position 9 is simply absorbed by the carry. Summing the 18 identities with weights `Y^{9a}`, the carries telescope to `−Y^162 e_17`
and the wrap terms add `(Y^81 − 1) e_17`, so the sum is the `S`-identity. Lemma (sufficient,
not "iff"): if every witness chunk — of `v`, of the digits, of the carries — has the support
it claims, every local product is the plain polynomial product and the 18 identities imply
the `S`-identity; without §4 a cheating prover could use the negacyclic aliases and would
only prove the identity modulo `X^64+1`. The public sub-chunk generators are computed by
both parties and asserted to have support `[0, 9)`. The whole recurrence — degrees, the
53-coefficient carries, the cyclic dependence of `e_17`, `k` entering whole at diagonals
0, 6, 12, and the output reproducing `A v mod q` — is checked numerically in
`chain_sim.py` (20 random instances) and becomes the Rust reference checker of step 1.

**Bounds.** Products have std ≈ 2^25 (3889) / 2^27 (9721), carries the same; the exact `k`
has std ≈ 2^13. Hence 3–4 digits for carries and 2 base-512 digits for `k`, each digit level
in its own witness vector (§5); the `z`-chain's carries are tiny — std ≈ 2^14 — and get 2
base-1024 digits. Every honest witness coefficient stays below
2^14 (`v` below 1944) and the whole witness has `‖·‖² ≈ 2^30.5` (`v` 2^29.7, `d_C` 2^28.5 per
limb, carries 2^30, `k` 2^22) — far below the JL cap 2^56.

**Public data.** For the `A`-part the `phi` arrays depend only on the key: 432 arrays of 256
`polx` per limb (one per `(k, twist, b, a)`, shared by the four output components through
pointer aliasing) = 110k `polx` = 110 MB per limb, generated and NTT'd once per key like
rokoblador's `warm_comkey`. Per proof: the `c_j`-part (27.6k `polx` per limb) and the `z`-part
(55k) — ≈ 110 MB, ≈ 100k `polx_frompoly` (≈ 50 ms each side). The constraint system is 165
constraints; evaluating/aggregating it is ≈ 0.9M `polx` multiply-adds (≈ 900 MB streamed).

## 4. Zero parts without touching LaBRADOR

Each chunk poly has 10–11 coefficients that must be zero (≈ 170k in total, over `v`, the
digits, the carries — every witness-side chunk). Separate constant-term constraints (one per
coefficient) would be 170k constraint objects and 170 MB of `phi` — not viable. Two ways out:

* **(recommended) A second pre-commitment and `LIFTS` random masks.** After the fold the
  prover commits to the rest of the witness the same way as `T_C`:
  `T_R = ⟨key_R, (v, d_k, e)⟩` (degree-`κ_R` constraint, `κ_R ≈ 9`, ≈ 3 KB, its own
  domain-separated key). The transcript then yields, for `t = 1..⌈128/LOGQ⌉ = 4`, uniform
  scalars `ρ^{(t)}_{j,c} ∈ Z_Q` for every (chunk `j`, zero position `c`), and four
  constant-term constraints `Σ_{j,c} ρ^{(t)}_{j,c} · s_{j,c} = 0` (each `phi_j = Σ_c
  ρ^{(t)}_{j,c} X^{−c}`, one `polx` per chunk) are added. One such equation catches a fixed
  nonzero vector with probability `1 − 1/Q`, so four independent ones give `Q^{−4} ≈ 2^{−160}`
  — the same repetition LaBRADOR itself uses (`LIFTS`) for its constant-term collapse, which
  then handles our four as usual. Soundness: `T_C, T_R` bind the witness before `ρ` is
  drawn. Cost: one commitment (≈ 25k `polx` MACs), ≈ 3 KB, four dense `polx` vectors.
* **A library patch.** A "zero-mask" constraint family in `dachshund.c/chihuahua.c` whose
  collapse draws the `ρ` from LaBRADOR's own hash state after its inner commitment, once per
  lift — the same soundness with no `T_R`. ~100 lines of C in the fork; deferred unless the
  3 KB matter.

## 5. The no-wraparound condition per limb, and the smallest `Q`

LaBRADOR's modulus is left exactly as in its table: `Q = 2^LOGQ − QOFF`, prime. For each
limb `q` the proof shows `A_q v − Σ_j c_j C_j^{(q)} = q · k_q` modulo `Q`; it certifies the
same identity over `Z` — hence `A_q v ≡ y_q (mod q)` — only if the integer left-hand side of
every chain constraint, at every coefficient, is below `Q/2` for every witness a cheater
could use under the caps. The caps are what the verifier enforces: `betasq_v ≤ 2^30.5`
(honest `2^29.7`, i.e. `‖v‖₂ ≤ 2^15.25`), and for every digit-level vector a cap at twice its
honest `‖·‖²`. The honest prover additionally aborts if `‖v‖∞ ≥ q_min/2`, so that its own
values sit where the honest sizes below assume them (a cheater gains nothing from skipping
this: the extracted `v` is one integer vector, the same in every limb, and only its `ℓ2`
cap enters the binding argument).

**The bound is computed, not estimated.** For a chain constraint, coefficient `t` of its
integer left-hand side is `Σ_vectors ⟨row_{t,i}, s_i⟩` where `row_{t,i}` is the vector of
`phi` coefficients (with sign and wrap) that feed position `t` from witness vector `i`. By
Cauchy–Schwarz `|·| ≤ Σ_i ‖row_{t,i}‖₂ · √betasq_i`, an exact worst case over the caps, tight
for a sign-aligned witness. The code computes `max_t Σ_i ‖row_{t,i}‖₂ √cap_i` per constraint
from the actual generated `phi` (the `A`-part rows once per key) and refuses a configuration
that does not clear `Q/2`; `z` is then an exact integer with coefficients below the same
bound and the verifier rejects a `z` outside it. Estimates, with the deterministic
`|A| ≤ q/2` in place of the key's actual `‖A‖₂` (≈ 0.8 bits more than a real key) and up to
three carry terms in a row (`e_{a−1}`, `e_a`, `e_17` at the wrap diagonals):

| term | 3889 | 9721 |
|---|---|---|
| `A`-part `2·√n_v·(q/2)·‖v‖₂` | 2^35.8 | 2^37.2 |
| three carries, each `≤ cap_top·B^{D−1} + lower levels` | 2^35.6 | 2^36.7 |
| `q·k`, two levels | 2^32.8 | 2^34.1 |
| `C`-part `‖c-row‖₂·‖d‖₂` | 2^26.4 | 2^27.4 |
| **sum** | **2^36.8** | **2^38.0** |

against `Q/2 = 2^39` at `LOGQ = 40`: margins 4.6× and 2×; honest values are ≈ 2^25–2^27.
Carries get 3 base-1024 digits for 3889 and 4 base-256 digits for 9721 (honest maxima
≈ 2^28.2 and 2^29.5); `k` 2 base-512 digits (honest maxima ≈ 2^15 and 2^16.3).

**Measured (2026-08-30): `LOGQ = 48`.** With the residues committed as they are (§11) the
witness's total squared norm is `2^40.4`, dominated by the four 9721 residue vectors at `2^38.3`
each; Dachshund's exact-norm proof wants that below `2^(LOGQ-3)`, which `LOGQ = 40` (`2^37`) misses
and `LOGQ = 48` (`2^45`) clears, and digit-decomposing the residues to fit 40 costs more than the
wider modulus. At 48 (`QOFF = 59`, `K = 8`, `L = 4`, `LIFTS = 3`, `sizeof(polx) = 1024`) the bound
below is `2^36.04` on 3889 and `2^37.89` on 9721 against `Q/2 = 2^47`, margins of 1992x and 552x;
`kappa_Y = 11`, `kappa_u = 3`, `kappa_R = 8` from `sis_secure` on the caps.

**Why `LOGQ = 40` is the smallest table entry that works.** At `LOGQ = 32`, `Q/2 = 2^31` is
below the 2^35 that the `A`-part alone reaches at the *honest* norm, so the proof would only
certify `A_q v ≡ y_q + Q·m (mod q)` for some small unknown `m` — a relaxed relation under
which two extracted openings no longer give a kernel vector of `A`, i.e. the binding argument
of the fold breaks. No norm statement on `v` changes that: Cauchy–Schwarz is attained by a
`v` proportional to a row of `A`, whose coefficients are only ≈ 100–200, so an `ℓ∞` bound
(even a bit decomposition) does not help. The only route to 32 is genuinely small-operand
arithmetic — `A_q` split into base-16 digits with a reduced, separately committed
intermediate per digit (3–4× the `A`-part public data and evaluation, one more wraparound
vector per digit) — and it is not planned. `LOGQ = 40` has 7 CRT primes (8 at 48/50), `L = 3`,
`LIFTS = 4`, `Q ≡ 5 (mod 8)` as LaBRADOR's analysis wants, per-vector `betasq ≤ 2^39`
against our 2^30, JL cap 2^56, `κ ≈ 9` from `sis_secure`, `κ, κ_1 ≤ 32`, every honest
witness coefficient ≤ 23170 (LaBRADOR's `int16`/`vpdpwssd` limit — a prover-side
implementation limit only). `r ≈ 18` witness vectors: LaBRADOR's quadratic garbage is
`(r²+r)/2 = 171` inner products over the witness, ≈ 90 ms.

## 6. Protocol and API on the branch

    let (commitment, opening) = prover.commit(&witness);            // Commitment = T_C
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let proof = prover.prove_opening(&mut transcript, opening, &challenges, &point);
    verifier.verify_evaluation(&point, &claimed_value, &row_evaluation)?;
    verifier.verify_opening(&mut transcript, &commitment, &point, &row_evaluation, &challenges, &proof)?;

`commit` additionally digit-decomposes the raw 648-slot commitments and computes `T_C` with
LaBRADOR's `comkey` (≈ 10 ms on top of today's 13 ms). `prove_opening` = today's `fold` +
exact integer products/carries/`k` (≈ 10^8 i64 MACs per limb, vectorised) + `z` + `T_R` +
statement/witness assembly + `composite_prove_simple`. `verify_opening` rebuilds the same
statement from public data (prover and verifier share one builder and the test asserts they
agree), checks `z mod 2`, the caps and the no-wrap bound, then `composite_verify_simple`.
`FoldedWitness`, `fold_commitment` and `verify_folded_opening` leave the public API.

Transcript (blake3, labelled): parameters and `LOGQ`, the key seed, `T_C`, `u` — then the
evaluation point and the folding challenges are squeezed — then `z`, `T_R`, every `betasq`;
then the four `ρ` vectors of §4 are squeezed under their own label; and LaBRADOR's statement
hash `st->h` is seeded from a final digest under yet another label. Every `phi` and `b` is a
deterministic function of what was absorbed, so the `phi` are *not* absorbed (rokoblador does
so redundantly, at ≈ 300 MB of hashing here); prover and verifier build the statement with
one function and the tests assert equality. LaBRADOR's internal Fiat–Shamir starts from
`st->h`.

`FoldedOpeningProof` holds `z`, the `r` values `betasq`, `T_R` and the LaBRADOR proof. The
LaBRADOR proof exists only as LaBRADOR's in-memory `composite` (no serialiser in the library;
`composite->size` is an analytic estimate, which is what rokoblador reports); a canonical
wire format (`u1, u2, z, p, lifts` per level) is a few hundred lines and is the only way the
size below becomes a measured number — decision D6.

**What is done when.** At key time: the `A`-part `phi` (442k `polx`, NTT'd once), their
row norms for the bound of §5, and the three keys (`comkey`, `key_C`, `key_R`). At commit
time: the digits of `C` and their `polx`, `T_C`. Per proof, and nothing else: the fold
(4.5 ms); the diagonal sums `D_a` over `Z` for the carries — 215M `i16×i16→i32` MACs with
`vpmaddwd`, ≈ 5 ms; `z` and its chain (≈ 2 ms); the `phi` that depend on `c_j` — sub-chunks
of a sparse ternary `c_j` are sums of ≤ 9 precomputed monomial `polx`, no NTTs, ≈ 1 ms; the
`phi` that depend on `p0` (the 1024 dense 0/1 lifts, 55k `polx`, ≈ 5–10 ms); `T_R`, the four
mask `phi` (scaled sums of 11 precomputed monomial `polx`) and the witness's `polx` form
(≈ 10 ms); then LaBRADOR itself: inner commitments `κ·n` (≈ 2 ms), the quadratic garbage —
cheap because the 18 vectors are short, 15 big pairs × 3072 polys (≈ 1 ms), the JL
projection (AES expansion of 33 MB plus 268M adds, ≈ 15 ms) and its four lifted collapses
(≈ 30 ms), aggregation of our sparse constraints (≈ 0.8M `polx` MACs, ≈ 10 ms), and the
recursion levels, which start from an amortised opening of 3072 polys (≈ 2× level one).

Measured at `Params::basic()` on one core of an i7-11850H, `LOGQ = 48`: proof 80.0 KB
(`T_Y` 4.1 + `T_u` 1.1 + `T_R` 3.0 + 29 norms 0.2 + LaBRADOR 71.5), against 978 KB in the clear.
Prover: `commit` 50.3 ms with `T_Y` (13.9 without), `commit_left_expansion` 0.2 ms,
`prove_opening` 489 ms = fold 4.5 + encoding 22.7 + `T_R` 2.7 + masks 2.4 + constraint `phi` 16.9
+ `labrador::prove` 404.7, so 49.2 ms of arithmetic of our own. Verifier: statement rebuild
26.7 ms + `labrador::verify` 235.7 ms. Key time (`PublicParameters::from_seed`): 124 ms and 233 MB
of key-time `phi` for two limbs; 81 MB of per-proof `phi`; peak resident set 592 MB.

## 7. Build and code layout

* `labrador/` — git submodule → `github.com/osdnk/labrador` (your fork, for its static-library
  target and `test_off`; built with `make LOGQ=40 liblabrador.a`, no LTO, stamped so a stale
  `LOGQ` build is cleaned), `csrc/shim.c`
  — opaque allocation, `init_sparsecnst_half`-based constraint setter that takes `deg`, block
  offsets and *pointers* to existing `polx` arrays (so `comkey` and the shared `A`-part arrays
  are aliased, not copied), `polx` conversion entry points, `composite->size`; `build.rs` as
  in rokoblador. All symbols `labrador40_*`.
* `src/recursion/`: `chunks.rs` (S-chunking, sub-chunk `phi` generation, shifts/reductions
  in `S`), `chain.rs` (diagonals, carries, digit bases, exact i64 evaluation of every
  constraint — the Rust reference checker), `statement.rs` (the 165 constraints, shared by
  prover and verifier), `witness.rs`, `ffi.rs`.
* `scheme.rs`: the API of §6. `fold.rs` unchanged. `main.rs`: the same three groups; the
  LaBRADOR proof size is `composite->size` (analytic KB, as rokoblador — LaBRADOR has no
  serialiser; writing one is out of scope here).
* Tests: constraint system satisfied over `Z` by the honest witness (reference checker,
  `Params::new(9, 2, ..)`); prover/verifier statements equal; `simple_verify` accepts; tamper
  tests (a coefficient of `v`, a digit of `C`, `z`, `k`, a zero position) each rejected; the
  existing api/fold/eval tests kept.

## 8. Steps

0. Submodule, `build.rs`, shim, FFI; LaBRADOR's own `test_off` passes from the crate build;
   a `test_off`-style C probe with the constraint mix we need (degree 0, 1 and `κ`; `r = 18`
   vectors with per-vector `betasq`) proves the front end takes it.
1. `recursion::chunks/chain` with the reference checker; a sizing probe: one honest instance
   through `simple_verify` → `composite_prove_simple` → `composite_verify_simple`, printing
   `κ`, proof KB and times. **Stop here and report if the numbers are far from §6.**
2. `commit` → `T_C`; `Commitment` becomes `T_C`.
3. `prove_opening` / `verify_opening`; tests; `main.rs`; README.

## 9. Decisions (settled 2026-08-29)

* **D1** `C` inside LaBRADOR; `T_C` is the commitment. — yes.
* **D2** zero parts via `T_R` and four transcript-derived masks; no LaBRADOR change. — yes.
* **D3** chunking 54/9. — yes.
* **D4** `LOGQ = 48` (was 40). At 40 the residue vectors alone put the total witness norm² at
  `2^40.4`, above the `2^(LOGQ-3) = 2^37` ceiling Dachshund's exact-norm proof wants; at 48 the
  ceiling is `2^45` and the no-wrap margins are 1992x (3889) and 552x (9721) against `Q/2 = 2^47`.
  Digit-decomposing the residues to fit 40 is dearer than the wider modulus.
* **D5** `betasq_v ≤ 2^30.9`: the 95th percentile of the honest `‖v‖²` for a random binary
  witness, so ≈ 5 % of honest folds are retried with fresh challenges. `‖v‖²` does not
  concentrate — it is `A·(1 + χ²₁)` with `A ≈ 2^28.65`, the second term being the square of
  the challenges' sign sum acting on the witness's mean-½ component — so this is the smallest
  cap with that failure rate (model from 40 simulated draws; a measured percentile can replace
  it). Margins at `LOGQ = 40` with it: 3889 ≈ 4×, 9721 ≈ 1.75×, 12637 ≈ 1.3×.
* **D6** LaBRADOR proof kept as an in-memory handle with the analytic size, as rokoblador;
  no serialiser. — yes.
* **D7** commit to the left expansion too (§2c). — yes.
* Arity stays `r = 256`; LaBRADOR untouched beyond the shim, modulus included.

## 10. Codex review, and what changed

Codex (gpt-5.6, xhigh) reviewed the previous revision. Adopted: four independent zero masks
instead of one (its `2^{-40}` soundness objection is right — LaBRADOR repeats its own
constant-term collapse `LIFTS` times for the same reason); the no-wrap bound computed from the
actual `phi` rows with Cauchy–Schwarz per witness vector, three carry terms per row, `q/2`
in the estimates; the `ℓ∞`-route-to-32 remark removed (a spread sign-aligned `v` has small
coefficients); explicit transcript binding with domain separation and separate seeds for
`key_C`, `key_R`; unrestricted digits stated as such; `r ≈ 18` made consistent; precise
index ranges in §3 and the lemma stated as sufficient; the mixed-degree probe in step 0; the
serialiser as D6. Not adopted: "`κ` must be a power of two" — LaBRADOR's `polxvec_mul_extension`
pads to the next power of two and truncates the output, and `init_proof` itself picks any
`κ ≤ 32`; and "the `ℓ2` cap weakens the relation" — the `‖v‖∞ ≤ 1944` test of today's
verifier is a representation check, not part of the binding argument, and the extracted `v` is
one integer vector for all limbs.

## 11. RNS, not digits (2026-08-29)

The residues `C_j^{(q)}` are LaBRADOR witness vectors as they are: no digit decomposition of
the commitment anywhere. Consequences: the `C` part of the witness halves (6144 polys), the
`C`-part `phi` halves (one block per limb and chunk instead of two), the `C`-part of the
no-wrap bound becomes `‖c-row‖₂·‖Y_q‖₂ ≈ 2^6.2·2^20 = 2^26`, `κ_C ≈ 12`, and LaBRADOR's total
witness norm is now dominated by the residues (`≈ 2^40`), which its parameter search absorbs
(JL cap 2^56, per-vector ceiling 2^39 met by splitting 9721's vector). Only the carries, the
wraparound quotients `k_q` and the two binary-side quotients are gadget-decomposed.

//! The left-expansion of the paper's `Pi_translate`, the binary side of the fold, and the
//! verifier — everything the commitment and the fold leave to the field `F = F162`.
//!
//! # The algebra
//!
//! The witness is the `wdim x r` matrix `W` over `F = GF(2)[x]/(x^162 + x^81 + 1)` held as the
//! flat `&[F162]` the commitment takes: entry `(i, j)` is `witness[i + wdim * j]`, so a column is
//! one chunk of the commitment. `F` is exactly `R_162 mod 2` under the crate's plain lift — the
//! coefficients of an `R_162` element reduced mod 2 are the bits of an `F162`, and the signs
//! vanish — so the whole fold has a shadow over `F`.
//!
//! Both sides of that shadow are multilinear extensions in `nu = log2(wdim) + log2(r)` variables,
//! split as `r0` (the row index `i`, low variables) and `r1` (the column index `j`, high ones),
//! with `eq(r, b) = prod_k (r_k if b_k = 1 else 1 + r_k)`. Writing `B = eq(r0, .)` for the row
//! vector of the left variables, the statement, the prover's message and the two checks are
//!
//! ```text
//!     t   = sum_{i,j} eq(r1, j) eq(r0, i) W[i + wdim j]    the claimed evaluation
//!     u   = B W          (r entries of F)                  the left-expansion, sent by the prover
//!     t   = u^T eq(r1)                                     the verifier's claim check
//!     v   = W c          (over R_648, the fold)            c_j the challenges
//!     B v = u^T c        (over F)                          the binary check
//! ```
//!
//! The last line is the only new identity: `B (W c) = (B W) c` mod 2, linearity of the left
//! expansion against the challenge vector. Its left side is read off the folded witness `v` the
//! fold already produced (component `k` of packed element `m` is the `F162` at index `4m + k`,
//! and a coefficient's parity is that element's bit), its right side is the `r`-term inner
//! product of the prover's `u` against the challenges reduced mod 2
//! ([`ShortChallenge::to_f162`](crate::ShortChallenge::to_f162)).
//!
//! # How it is computed
//!
//! Every step is one dot product over `F`. The kernels are `bin_fields`' word-sliced AVX-512
//! ones — limb `k` of 8 consecutive elements in one `zmm`, `mac_soa8` accumulating the 12
//! unreduced `clmul` products of a block and a single `reduce_soa8` at the end of the whole
//! product, exactly as that crate's sumcheck round does. One operand (the `eq` table, or the
//! challenges) is word-sliced once up front; the other is a raw `&[F162]` run — the witness never
//! leaves its own layout, it is transposed 8 elements at a time by three `vpermi2q`/`vpermq`
//! pairs inside the loop ([`load_soa8`]).
//!
//! ```no_run
//! # use bin_ntt::{eval, CommitmentKey, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
//! # use bin_fields::scalar::F162;
//! # let witness: Vec<F162> = Vec::new();
//! # let ck = CommitmentKey::random_default(1 << 10, 1);
//! let (c, aux) = ck.commit_with_aux(&witness, 256);
//! let mut t = Transcript::new(b"bin-ntt/eval");
//! for j in 0..256 { t.absorb_elements(c.column(j)); }
//! let point = eval::sample_point::<10, 8>(&mut t);
//! let claim = eval::evaluate_mle(&witness, &point);          // the statement
//! let lx = eval::left_expand(&witness, &point.r0);           // the prover's message
//! assert!(eval::check_claim(&lx.u, &point.r1, claim));
//! ```
use crate::api::{AuxData, CommitmentKey};
use crate::challenge::{ShortChallenge, Transcript};
use crate::fold::{a_times_v_limb, challenge_ntt_limb, combine_limb, forward_limb, Q1};
use crate::params::N;
use crate::types::{Batch32, Representation, RingElement};
use bin_fields::f162 as bf;
use bin_fields::scalar::F162;
use bin_fields::sumcheck::Poly;
use core::arch::x86_64::*;

// =============================================================================================
// the evaluation point
// =============================================================================================

/// A point of `F^nu` split the way the witness is: `r0` over the `LW` row variables (the index
/// inside a chunk), `r1` over the `LR` column variables (which chunk). The defaults are the
/// crate's headline instance, `wdim = 2^10` rows and `r = 2^8` chunks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EvalPoint<const LW: usize = 10, const LR: usize = 8> {
    pub r0: [F162; LW],
    pub r1: [F162; LR],
}

/// One uniform `F162` per variable, from one XOF derivation of the transcript.
pub fn sample_point<const LW: usize, const LR: usize>(t: &mut Transcript) -> EvalPoint<LW, LR> {
    let mut bytes = vec![0u8; 24 * (LW + LR)];
    t.fill(b"eval-point", &mut bytes);
    let elem = |n: usize| {
        let mut limb = [0u64; 3];
        for k in 0..3 {
            limb[k] = u64::from_le_bytes(bytes[24 * n + 8 * k..24 * n + 8 * k + 8].try_into().unwrap());
        }
        limb[2] &= (1u64 << 34) - 1;
        F162(limb)
    };
    EvalPoint {
        r0: core::array::from_fn(|k| elem(k)),
        r1: core::array::from_fn(|k| elem(LW + k)),
    }
}

/// `eq(rs, b) = prod_k (rs_k if bit k of b is 1 else 1 + rs_k)`, all `2^rs.len()` of them, by
/// doubling: variable `k` is bit `k` of the index.
pub fn eq_table(rs: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &r in rs {
        let n = t.len();
        t.resize(2 * n, F162::ZERO);
        for b in 0..n {
            let x = t[b];
            t[b + n] = x * r;
            t[b] = x * (F162::ONE + r);
        }
    }
    t
}

// =============================================================================================
// the dot product over F
// =============================================================================================

/// The `vpermi2q` / `vpermq` index vectors of [`load_soa8`]: 24 consecutive `u64` (8 `F162`) to
/// limb 0, 1, 2 of those 8 elements. Limb `k` of element `p` is `u64` number `3p + k`, which for
/// `k = 0` is `0, 3, 6, 9, 12, 15` inside the first two vectors and `2, 5` inside the third, and
/// so on.
struct Idx {
    a: [__m512i; 3],
    b: [__m512i; 3],
    m: [__mmask8; 3],
}

#[inline(always)]
unsafe fn idx() -> Idx {
    Idx {
        a: [
            _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0),
            _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0),
            _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0),
        ],
        b: [
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 2, 5),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 3, 6),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 4, 7),
        ],
        m: [0b1100_0000, 0b1110_0000, 0b1110_0000],
    }
}

/// 8 consecutive `F162` (192 contiguous bytes) read as the word-sliced `[limb0, limb1, limb2]`
/// the `bin_fields` kernels take.
///
/// # Safety
/// `p` addresses 24 readable `u64`.
#[inline(always)]
unsafe fn load_soa8(ix: &Idx, p: *const u64) -> [__m512i; 3] {
    let v0 = _mm512_loadu_si512(p as *const __m512i);
    let v1 = _mm512_loadu_si512(p.add(8) as *const __m512i);
    let v2 = _mm512_loadu_si512(p.add(16) as *const __m512i);
    core::array::from_fn(|k| {
        _mm512_mask_permutexvar_epi64(
            _mm512_permutex2var_epi64(v0, ix.a[k], v1),
            ix.m[k],
            ix.b[k],
            v2,
        )
    })
}

/// The 8 lanes of a finished accumulator summed into one element.
#[inline(always)]
unsafe fn horizontal(acc: [__m512i; 12]) -> F162 {
    let r = bf::reduce_soa8(acc);
    let mut w = [[0u64; 8]; 3];
    for k in 0..3 {
        _mm512_storeu_si512(w[k].as_mut_ptr() as *mut __m512i, r[k]);
    }
    let mut out = F162::ZERO;
    for i in 0..8 {
        out += F162([w[0][i], w[1][i], w[2][i]]);
    }
    out
}

/// `sum_k a_k b_k` over `F`, `a` word-sliced and `b` a raw run of `n` `F162`, with the reduction
/// deferred to the end: 12 unreduced `clmul` products per 8 elements, one [`bf::reduce_soa8`].
///
/// # Safety
/// `b` addresses `n` `F162` and `a` holds at least `n` elements.
unsafe fn dot(a: &Poly, b: *const F162, n: usize) -> F162 {
    debug_assert!(a.n >= n);
    let ix = idx();
    let (mut acc, blocks) = ([_mm512_setzero_si512(); 12], n / 8);
    for t in 0..blocks {
        let x = [
            _mm512_loadu_si512(a.w[0].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[1].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[2].as_ptr().add(8 * t) as *const __m512i),
        ];
        bf::mac_soa8(&mut acc, x, load_soa8(&ix, (b as *const u64).add(24 * t)));
    }
    let mut out = if blocks > 0 {
        horizontal(acc)
    } else {
        F162::ZERO
    };
    for k in 8 * blocks..n {
        out += a.get(k) * *b.add(k);
    }
    out
}

/// [`dot`] on two plain slices.
fn dot_slices(a: &[F162], b: &[F162]) -> F162 {
    assert_eq!(a.len(), b.len(), "the two operands have different lengths");
    unsafe { dot(&Poly::from_scalars(a), b.as_ptr(), b.len()) }
}

// =============================================================================================
// the statement
// =============================================================================================

/// `t = sum_{i,j} eq(r1, j) eq(r0, i) W[i + wdim j]`, the multilinear extension of the witness at
/// the point, in the two-stage form: the `2^nu` products of the left expansion and then the `r`
/// products against `eq(r1, .)`.
pub fn evaluate_mle<const LW: usize, const LR: usize>(
    witness: &[F162],
    r: &EvalPoint<LW, LR>,
) -> F162 {
    assert_eq!(
        witness.len(),
        1 << (LW + LR),
        "the witness is not 2^{} elements",
        LW + LR
    );
    claim(&left_expand(witness, &r.r0).u, &r.r1)
}

/// `u^T eq(r1)`, the claim a left-expansion `u` implies.
pub fn claim<const LR: usize>(u: &[F162], r1: &[F162; LR]) -> F162 {
    assert_eq!(u.len(), 1 << LR, "the left expansion is not 2^{LR} elements");
    dot_slices(&eq_table(r1), u)
}

/// The verifier's claim check: `sum_j u_j eq(r1, j) == t`.
pub fn check_claim<const LR: usize>(u: &[F162], r1: &[F162; LR], t: F162) -> bool {
    claim(u, r1) == t
}

// =============================================================================================
// the left expansion
// =============================================================================================

/// The prover's message of `Pi_translate`: `u = B W`, one field element per chunk.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LeftExpansion {
    /// `u_j = sum_i eq(r0, i) W[i + wdim j]`, one entry per column (chunk) of the witness.
    pub u: Vec<F162>,
}

/// `u_j = sum_i eq(r0, i) W[i + wdim j]`, `wdim = 2^LW`: one `wdim`-term dot product per column,
/// each a single deferred-reduction accumulation over a contiguous run of the witness.
pub fn left_expand<const LW: usize>(witness: &[F162], r0: &[F162; LW]) -> LeftExpansion {
    let wdim = 1usize << LW;
    assert_eq!(
        witness.len() % wdim,
        0,
        "the witness is not a whole number of columns of {wdim}"
    );
    let eq = Poly::from_scalars(&eq_table(r0));
    let u = (0..witness.len() / wdim)
        .map(|j| unsafe { dot(&eq, witness.as_ptr().add(j * wdim), wdim) })
        .collect();
    LeftExpansion { u }
}

/// `u^T c`, the binary side of the fold: `sum_j u_j (c_j mod 2)`.
pub fn fold_binary(u: &[F162], challenges: &[ShortChallenge]) -> F162 {
    assert_eq!(u.len(), challenges.len(), "one challenge per column");
    let c: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    dot_slices(&c, u)
}

// =============================================================================================
// the verifier
// =============================================================================================

/// The `r` commitments a verifier holds, in the form the fold's identity lives in: 648 rows in
/// `[0, q)` per chunk and limb, before the four-way `R_162` decomposition.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RawCommitments {
    c: Vec<Vec<[u32; N]>>,
}

impl RawCommitments {
    /// The commitments a [`CommitmentKey::commit_into_aux`] left behind.
    pub fn from_aux(aux: &AuxData) -> Self {
        let r = aux.chunks();
        RawCommitments {
            c: (0..aux.limbs())
                .map(|k| (0..r).map(|j| *aux.commitment(k, j)).collect())
                .collect(),
        }
    }
    /// Number of chunks.
    pub fn chunks(&self) -> usize {
        self.c[0].len()
    }
    /// Number of limbs.
    pub fn limbs(&self) -> usize {
        self.c.len()
    }
    /// The commitment of chunk `j` for limb `k`.
    pub fn get(&self, k: usize, j: usize) -> &[u32; N] {
        &self.c[k][j]
    }
    /// All `r` commitments of limb `k`.
    pub fn limb(&self, k: usize) -> &[[u32; N]] {
        &self.c[k]
    }
}

/// `A v == sum_j c_j C_j` for every limb, recomputed from `v` alone: the folded witness is
/// transformed forward modulo each limb's prime and multiplied into that limb's key, and the
/// right-hand side is the slot-wise inner product of the transformed challenges against the
/// commitments — a scalar product per slot for a splitting limb, the quadratic leaf product
/// (`scalar::mul_quad_slots`) for a quadratic one.
///
/// The centered range of `q1` is checked first — `v` reaching the verifier as anything larger is
/// not the small-integer vector the fold promises, and is rejected before it is transformed.
pub fn verify_fold(
    key: &CommitmentKey,
    c: &RawCommitments,
    challenges: &[ShortChallenge],
    v: &[RingElement],
) -> bool {
    assert_eq!(v.len(), key.len_ring(), "v is not one chunk of ring elements");
    assert_eq!(challenges.len(), c.chunks(), "one challenge per chunk");
    assert_eq!(key.limbs(), c.limbs(), "the key and the commitments disagree");
    let half = (Q1 as i16 - 1) / 2;
    if v.iter().any(|e| {
        e.representation != Representation::Coefficients || e.v.iter().any(|x| x.abs() > half)
    }) {
        return false;
    }

    let mut vb: Vec<Batch32> = (0..v.len() / 32)
        .map(|_| Batch32::zero(Representation::Coefficients))
        .collect();
    for (i, e) in v.iter().enumerate() {
        vb[i / 32].set(i % 32, e);
    }

    (0..key.limbs()).all(|k| {
        let (q, quad) = (key.prime(k), key.is_quadratic(k));
        let mut b = vb.clone();
        forward_limb(q, quad, &mut b);
        let y = a_times_v_limb(q, quad, key.row(k), &b);
        let ch = challenge_ntt_limb(q, quad, challenges);
        combine_limb(q, quad, &ch, c.limb(k)) == y
    })
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, and its bit `p` is the parity of
/// coefficient `4p + k` of that element.
pub fn components_mod_2(v: &[RingElement]) -> Vec<F162> {
    let mut out = vec![F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        for p in 0..crate::api::N162 {
            for k in 0..4 {
                out[4 * m + k].0[p >> 6] |= ((e.v[4 * p + k] & 1) as u64) << (p & 63);
            }
        }
    }
    out
}

/// The binary check: `B v == u^T c` over `F`, i.e. `sum_i eq(r0, i) (v_i mod 2) == u_folded`.
pub fn verify_binary<const LW: usize>(
    r0: &[F162; LW],
    v: &[RingElement],
    u_folded: F162,
) -> bool {
    assert_eq!(4 * v.len(), 1usize << LW, "v is not 2^{LW} field components");
    dot_slices(&eq_table(r0), &components_mod_2(v)) == u_folded
}

/// Everything a verifier of one round holds: the key, the `r` commitments and the claim at the
/// sampled point.
pub struct Verifier<'a, const LW: usize = 10, const LR: usize = 8> {
    pub key: &'a CommitmentKey,
    pub commitments: &'a RawCommitments,
    pub point: &'a EvalPoint<LW, LR>,
    pub claim: F162,
}

impl<const LW: usize, const LR: usize> Verifier<'_, LW, LR> {
    /// The three checks: `u^T eq(r1) = t`, `A v = sum_j c_j C_j`, and `B v = u^T c` over `F`.
    pub fn verify(&self, u: &[F162], challenges: &[ShortChallenge], v: &[RingElement]) -> bool {
        check_claim(u, &self.point.r1, self.claim)
            && verify_fold(self.key, self.commitments, challenges, v)
            && verify_binary(&self.point.r0, v, fold_binary(u, challenges))
    }
}

//! The binary shadow of the scheme: everything the commitment and the fold leave to the field
//! `F = F162 = GF(2)[x]/(x^162 + x^81 + 1)`.
//!
//! # The algebra
//!
//! The witness is the `wdim x r` matrix `W` over `F` held as the flat `&[F162]` the commitment
//! takes: entry `(i, j)` is `witness[i + wdim * j]`, so a column is one chunk of the commitment.
//! `F` is exactly `R_162 mod 2` under the crate's plain lift — the coefficients of an `R_162`
//! element reduced mod 2 are the bits of an `F162`, and a binary challenge is its own reduction —
//! so the whole fold has a shadow over `F`.
//!
//! Both sides of that shadow are multilinear extensions in `nu = log2(wdim) + log2(r)` variables,
//! split as `p0` (the row index `i`, low variables) and `p1` (the column index `j`, high ones),
//! with `eq(p, b) = prod_k (p_k if b_k = 1 else 1 + p_k)`. Writing `B = eq(p0, .)` for the row
//! vector of the left variables, the statement, the prover's message and the two checks are
//!
//! ```text
//!     t   = sum_{i,j} eq(p1, j) eq(p0, i) W[i + wdim j]    the claimed evaluation
//!     u   = B W          (r entries of F)                  the row evaluation, sent by the prover
//!     t   = u^T eq(p1)                                     the verifier's claim check
//!     v   = W c          (over R_648, the fold)            c_j the challenges
//!     B v = u^T c        (over F)                          the binary check
//! ```
//!
//! The last line is the only new identity: `B (W c) = (B W) c` mod 2, linearity of the row
//! evaluation against the challenge vector. Its left side is read off the folded witness `v` the
//! fold already produced (component `k` of packed element `m` is the `F162` at index `4m + k`,
//! and a coefficient's parity is that element's bit), its right side is the `r`-term inner
//! product of `u` against the challenges reduced mod 2
//! ([`ShortChallenge::to_f162`](crate::challenge::ShortChallenge::to_f162)).
//!
//! # How it is computed
//!
//! Every step is one dot product over `F`. The kernels are [`fields::f162`](crate::fields::f162)' word-sliced AVX-512
//! ones — limb `k` of 8 consecutive elements in one `zmm`, `mac_soa8` accumulating the 12
//! unreduced `clmul` products of a block and a single `reduce_soa8` at the end of the whole
//! product, exactly as that crate's sumcheck round does. One operand (the `eq` table, or the
//! challenges) is word-sliced once up front; the other is a raw `&[F162]` run — the witness never
//! leaves its own layout, it is transposed 8 elements at a time by three `vpermi2q`/`vpermq`
//! pairs inside the loop ([`load_soa8`]).
use crate::challenge::ShortChallenge;
use crate::fields::f162 as bf;
use crate::fields::scalar::F162;
use crate::fields::sumcheck::Poly;
use crate::types::RingElement;
use core::arch::x86_64::*;

/// `eq(ps, b) = prod_k (ps_k if bit k of b is 1 else 1 + ps_k)`, all `2^ps.len()` of them, by
/// doubling: variable `k` is bit `k` of the index.
pub fn eq_table(ps: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &p in ps {
        let n = t.len();
        t.resize(2 * n, F162::ZERO);
        for b in 0..n {
            let x = t[b];
            t[b + n] = x * p;
            t[b] = x * (F162::ONE + p);
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
/// the [`fields::f162`](crate::fields::f162) kernels take.
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
// the four steps
// =============================================================================================

/// `u_j = sum_i eq(p0, i) W[i + wdim j]`, `wdim = 2^p0.len()`: one `wdim`-term dot product per
/// column, each a single deferred-reduction accumulation over a contiguous run of the witness.
pub(crate) fn row_evaluate(witness: &[F162], p0: &[F162]) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        witness.len() % wdim,
        0,
        "the witness is not a whole number of columns of {wdim}"
    );
    let eq = Poly::from_scalars(&eq_table(p0));
    (0..witness.len() / wdim)
        .map(|j| unsafe { dot(&eq, witness.as_ptr().add(j * wdim), wdim) })
        .collect()
}

/// `u^T eq(p1)`, the claim a row evaluation `u` implies.
pub(crate) fn claim(u: &[F162], p1: &[F162]) -> F162 {
    assert_eq!(
        u.len(),
        1 << p1.len(),
        "the row evaluation is not 2^{} elements",
        p1.len()
    );
    dot_slices(&eq_table(p1), u)
}

/// `u^T c`, the binary side of the fold: `sum_j u_j (c_j mod 2)`.
pub(crate) fn fold_binary(u: &[F162], challenges: &[ShortChallenge]) -> F162 {
    assert_eq!(u.len(), challenges.len(), "one challenge per column");
    let c: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    dot_slices(&c, u)
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, and its bit `p` is the parity of
/// coefficient `4p + k` of that element.
pub(crate) fn components_mod_2(v: &[RingElement]) -> Vec<F162> {
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

/// The binary check `B v == u^T c` over `F`: `sum_i eq(p0, i) (v_i mod 2) == u_folded`.
pub(crate) fn binary_check(p0: &[F162], v: &[RingElement], u_folded: F162) -> bool {
    4 * v.len() == 1usize << p0.len() && dot_slices(&eq_table(p0), &components_mod_2(v)) == u_folded
}

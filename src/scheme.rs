//! The scheme: parameters, a prover, a verifier, and the five values that pass between them.
//!
//! One round is
//!
//! ```text
//!     (C, opening)   = prover.commit(w)
//!     p = (p0, p1)   = verifier.derive_evaluation_point(transcript, C)
//!     t              = w.mle_evaluate(p)                    the statement
//!     u              = w.row_evaluate(p)                    the prover's message
//!     c              = verifier.derive_folding_challenges(transcript, u)
//!     v              = prover.fold(opening, c)
//! ```
//!
//! and the verifier accepts when `u . eq(p1) = t`, `v` is short, `A v = sum_j c_j C_j` modulo
//! every modulus, and `eq(p0) . (v mod 2) = sum_j u_j (c_j mod 2)` over `F162`.
use crate::api::{
    components_of, AuxData, CommitmentKey, Modulus, PowerOfThreeRingElement,
    PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, BASE_PRIME, N162,
};
use crate::challenge::{
    sample_short_challenge, ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use crate::fold::{a_times_v_limb, challenge_slots162, fold_witness, forward_limb, Q1};
use crate::types::{Batch32, Representation};
use crate::{eval, RingElement162, RingElement648};
use bin_fields::scalar::F162;
use std::fmt;
use std::sync::Arc;

/// log2 of the shortest column: one `Batch32` is 32 ring elements of `R_648` = 128 `F162`, and
/// the kernels commit to a whole number of those.
const MIN_COLUMN_LOG_LEN: u32 = 7;

// =============================================================================================
// parameters
// =============================================================================================

/// Why a [`Params`] was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParamError {
    /// `column_log_len > witness_log_len`: more columns than the witness has elements.
    ColumnsExceedWitness,
    /// Fewer than two columns; the fold accumulates the chunks in pairs.
    TooFewColumns,
    /// A column shorter than the 128 `F162` of one `Batch32`, which the kernels commit to at once.
    ColumnTooShort,
    /// The same extra modulus twice.
    DuplicateModulus(Modulus),
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamError::ColumnsExceedWitness => write!(f, "column_log_len exceeds witness_log_len"),
            ParamError::TooFewColumns => write!(f, "column_log_len must be at least 1"),
            ParamError::ColumnTooShort => write!(
                f,
                "a column must hold at least 2^{MIN_COLUMN_LOG_LEN} F162 elements"
            ),
            ParamError::DuplicateModulus(m) => write!(f, "the modulus {m:?} is listed twice"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The shape of one round: a witness of `2^witness_log_len` `F162` read as `2^column_log_len`
/// columns, committed modulo the base modulus 3889 and every entry of `extra_moduli`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Params {
    pub witness_log_len: u32,
    pub column_log_len: u32,
    pub extra_moduli: Vec<Modulus>,
}

impl Params {
    /// The checked constructor.
    pub fn new(
        witness_log_len: u32,
        column_log_len: u32,
        extra_moduli: Vec<Modulus>,
    ) -> Result<Params, ParamError> {
        if column_log_len > witness_log_len {
            return Err(ParamError::ColumnsExceedWitness);
        }
        if column_log_len < 1 {
            return Err(ParamError::TooFewColumns);
        }
        if witness_log_len - column_log_len < MIN_COLUMN_LOG_LEN {
            return Err(ParamError::ColumnTooShort);
        }
        for (i, m) in extra_moduli.iter().enumerate() {
            if extra_moduli[..i].contains(m) {
                return Err(ParamError::DuplicateModulus(*m));
            }
        }
        Ok(Params {
            witness_log_len,
            column_log_len,
            extra_moduli,
        })
    }

    /// The configuration the crate is tuned for: 2^18 `F162` in 256 columns, moduli 3889 and 9721.
    pub fn basic() -> Params {
        Params::new(18, 8, vec![Modulus::Q9721]).expect("the basic parameters are valid")
    }

    /// Witness length in `F162` elements.
    pub fn witness_len(&self) -> usize {
        1usize << self.witness_log_len
    }

    /// Number of columns the witness is split into — also the number of folding challenges.
    pub fn columns(&self) -> usize {
        1usize << self.column_log_len
    }

    /// log2 of one column in `F162`: the number of row variables of the evaluation point.
    pub(crate) fn row_log_len(&self) -> u32 {
        self.witness_log_len - self.column_log_len
    }

    /// One column in `F162` elements.
    pub(crate) fn column_len(&self) -> usize {
        1usize << self.row_log_len()
    }

    /// The primes in index order: the base modulus first, then `extra_moduli`.
    pub(crate) fn primes(&self) -> Vec<u16> {
        core::iter::once(BASE_PRIME)
            .chain(self.extra_moduli.iter().map(|m| m.prime()))
            .collect()
    }
}

/// [`Params`] together with the public matrix `A` expanded from a seed.
pub struct PublicParameters {
    params: Params,
    key: Arc<CommitmentKey>,
}

impl PublicParameters {
    /// Expand `A` — one uniform row of `R_648` per modulus, in the NTT domain — from the seed.
    pub fn from_seed(params: Params, matrix_seed: [u8; 32]) -> PublicParameters {
        let digest = blake3::hash(&matrix_seed);
        let seed = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let key = CommitmentKey::random(params.column_len(), seed, &params.extra_moduli);
        PublicParameters {
            params,
            key: Arc::new(key),
        }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }
}

// =============================================================================================
// the witness
// =============================================================================================

/// Why a [`Witness`] was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WitnessError {
    /// Not [`Params::witness_len`] elements.
    WrongLength,
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the witness is not witness_len() elements")
    }
}

impl std::error::Error for WitnessError {}

/// The private input: `witness_len()` elements of `F162`, read as
/// `witness_len() / 4` binary ring elements of `R_648` four `F162` at a time.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Witness {
    params: Params,
    elements: Vec<F162>,
}

impl Witness {
    pub fn from_elements(params: &Params, elements: Vec<F162>) -> Result<Witness, WitnessError> {
        if elements.len() != params.witness_len() {
            return Err(WitnessError::WrongLength);
        }
        Ok(Witness {
            params: params.clone(),
            elements,
        })
    }

    /// A uniform witness from a blake3 XOF: three little-endian `u64` per element, the top 30 bits
    /// of the third cleared, so every element is a uniform 162-bit field element.
    pub fn random(params: &Params, seed: [u8; 32]) -> Witness {
        let n = params.witness_len();
        let mut bytes = vec![0u8; 24 * n];
        let mut hasher = blake3::Hasher::new_keyed(&seed);
        hasher.update(b"bin-ntt/witness");
        hasher.finalize_xof().fill(&mut bytes);
        let elements = (0..n)
            .map(|i| {
                let mut limb = [0u64; 3];
                for (k, l) in limb.iter_mut().enumerate() {
                    let o = 24 * i + 8 * k;
                    *l = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
                }
                limb[2] &= (1u64 << 34) - 1;
                F162(limb)
            })
            .collect();
        Witness {
            params: params.clone(),
            elements,
        }
    }

    pub fn elements(&self) -> &[F162] {
        &self.elements
    }

    /// The multilinear extension of the witness at the point — the statement being proved.
    pub fn mle_evaluate(&self, point: &EvaluationPoint) -> F162 {
        self.check(point);
        eval::claim(&eval::row_evaluate(&self.elements, &point.p0), &point.p1)
    }

    /// `u = B W`, one field element per column: the prover's message.
    pub fn row_evaluate(&self, point: &EvaluationPoint) -> RowEvaluation {
        self.check(point);
        RowEvaluation {
            values: eval::row_evaluate(&self.elements, &point.p0),
        }
    }

    fn check(&self, point: &EvaluationPoint) {
        assert_eq!(
            (point.p0.len(), point.p1.len()),
            (
                self.params.row_log_len() as usize,
                self.params.column_log_len as usize
            ),
            "the evaluation point does not match the witness"
        );
    }
}

// =============================================================================================
// the values that pass between the two parties
// =============================================================================================

/// The commitment: a `4 x columns()` matrix of `R_162` elements per modulus. Row `k` is the
/// coefficient of `X^k` in the basis `1, X, X^2, X^3` of `R_648` over `R_162`, column `j` is the
/// commitment of column `j` of the witness.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Commitment {
    primes: Vec<u16>,
    matrix: VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
}

impl Commitment {
    pub fn rows(&self) -> usize {
        self.matrix.rows()
    }

    pub fn columns(&self) -> usize {
        self.matrix.cols()
    }

    /// The primes, in the index order [`element`](Self::element) takes: the base modulus first.
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }

    pub fn element(&self, row: usize, column: usize, modulus_index: usize) -> &RingElement162 {
        &self.matrix.get(row, column).limbs[modulus_index]
    }
}

/// What the prover keeps from a commitment and the fold consumes: the witness's transform modulo
/// the base modulus, in the layout the kernel wrote it.
pub struct CommitmentOpening {
    aux: AuxData,
}

/// A point of `F162^nu` split the way the witness is: `p0` over the row variables (the index
/// inside a column), `p1` over the column variables.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EvaluationPoint {
    p0: Vec<F162>,
    p1: Vec<F162>,
}

impl EvaluationPoint {
    pub fn p0(&self) -> &[F162] {
        &self.p0
    }

    pub fn p1(&self) -> &[F162] {
        &self.p1
    }
}

/// `u_j = sum_i eq(p0, i) W[i, j]`, one field element per column.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowEvaluation {
    values: Vec<F162>,
}

impl RowEvaluation {
    pub fn values(&self) -> &[F162] {
        &self.values
    }

    /// Mutable access, so that a test can corrupt the message.
    pub fn values_mut(&mut self) -> &mut [F162] {
        &mut self.values
    }
}

/// The `columns()` short challenges of one round.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldingChallenges {
    challenges: Vec<ShortChallenge>,
}

impl FoldingChallenges {
    pub fn len(&self) -> usize {
        self.challenges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }
}

/// The amortised witness `v = sum_j c_j W_j`: one column's worth of `R_648` elements in
/// coefficient form, centered, and genuinely small — a coefficient is a sum of `r w` signed 0/1
/// terms, two orders of magnitude below `q1 / 2 = 1944.5`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedWitness {
    elements: Vec<RingElement648>,
}

impl FoldedWitness {
    pub fn elements(&self) -> &[RingElement648] {
        &self.elements
    }

    /// Mutable access, so that a test can corrupt the opening.
    pub fn elements_mut(&mut self) -> &mut [RingElement648] {
        &mut self.elements
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// `sum_j c_j C_j`: four `R_162` elements per modulus.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedCommitment {
    primes: Vec<u16>,
    rows: [PowerOfThreeRingElementWithLimbs; 4],
}

impl FoldedCommitment {
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }

    pub fn element(&self, row: usize, modulus_index: usize) -> &RingElement162 {
        &self.rows[row].limbs[modulus_index]
    }
}

/// The verifier rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VerificationError {
    Rejected,
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the opening was rejected")
    }
}

impl std::error::Error for VerificationError {}

// =============================================================================================
// the prover
// =============================================================================================

/// The prover: the public matrix and the workspace the commitment and the fold pass between them.
///
/// The 85 MB of witness transform a commitment leaves behind is one `mmap` and 20 736 first
/// touches, ~20 ms of page faults the kernel charges to whoever writes the pages first — more
/// than the commitment itself. [`Prover::new`] pays that once, and [`Prover::fold`] hands the
/// buffer back, so a second [`commit`](Prover::commit) allocates nothing.
pub struct Prover {
    params: Params,
    key: Arc<CommitmentKey>,
    workspace: Option<AuxData>,
}

impl Prover {
    /// Allocate the workspace and warm it up: one commitment and one fold over a zero witness,
    /// which touches every page of the workspace and runs every kernel and lazy table once.
    pub fn new(pp: &PublicParameters) -> Prover {
        let mut prover = Prover {
            params: pp.params.clone(),
            key: pp.key.clone(),
            workspace: None,
        };
        let verifier = Verifier {
            params: pp.params.clone(),
            key: pp.key.clone(),
        };
        let witness = Witness {
            params: pp.params.clone(),
            elements: vec![F162::ZERO; pp.params.witness_len()],
        };
        let (commitment, opening) = prover.commit(&witness);
        let mut transcript = Transcript::new(b"bin-ntt/warm-up");
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let row_evaluation = witness.row_evaluate(&point);
        let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
        std::hint::black_box(prover.fold(opening, &challenges));
        std::hint::black_box(verifier.fold_commitment(&commitment, &challenges));
        prover
    }

    pub fn commit(&mut self, witness: &Witness) -> (Commitment, CommitmentOpening) {
        assert_eq!(
            witness.params, self.params,
            "the witness was built for other parameters"
        );
        let mut aux = self.workspace.take().unwrap_or_else(|| {
            AuxData::new(self.key.len_ring(), self.params.columns(), self.key.limbs())
        });
        let matrix = self
            .key
            .commit_into_aux(&witness.elements, self.params.columns(), &mut aux);
        (
            Commitment {
                primes: self.params.primes(),
                matrix,
            },
            CommitmentOpening { aux },
        )
    }

    pub fn fold(
        &mut self,
        opening: CommitmentOpening,
        challenges: &FoldingChallenges,
    ) -> FoldedWitness {
        let elements = fold_witness(
            &opening.aux,
            &challenges.challenges,
            self.key.len_ring() / 32,
        );
        self.workspace = Some(opening.aux);
        FoldedWitness { elements }
    }
}

// =============================================================================================
// the verifier
// =============================================================================================

/// The verifier: the public parameters and nothing else.
pub struct Verifier {
    params: Params,
    key: Arc<CommitmentKey>,
}

impl Verifier {
    pub fn new(pp: &PublicParameters) -> Verifier {
        Verifier {
            params: pp.params.clone(),
            key: pp.key.clone(),
        }
    }

    /// Absorb the commitment, then derive `p = (p0, p1)`: one uniform `F162` per variable from a
    /// single extendable output of the transcript.
    pub fn derive_evaluation_point(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
    ) -> EvaluationPoint {
        transcript.absorb_bytes(b"bin-ntt/commitment");
        transcript.absorb_u64(commitment.columns() as u64);
        for j in 0..commitment.columns() {
            transcript.absorb_elements(commitment.matrix.column(j));
        }
        let (rows, cols) = (
            self.params.row_log_len() as usize,
            self.params.column_log_len as usize,
        );
        let mut bytes = vec![0u8; 24 * (rows + cols)];
        transcript.fill(b"bin-ntt/evaluation-point", &mut bytes);
        let element = |n: usize| {
            let mut limb = [0u64; 3];
            for (k, l) in limb.iter_mut().enumerate() {
                let o = 24 * n + 8 * k;
                *l = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
            }
            limb[2] &= (1u64 << 34) - 1;
            F162(limb)
        };
        EvaluationPoint {
            p0: (0..rows).map(element).collect(),
            p1: (0..cols).map(|k| element(rows + k)).collect(),
        }
    }

    /// Absorb `u`, then derive the `columns()` challenges — weight 21, canonical bound 9, one
    /// transcript derivation per challenge index.
    pub fn derive_folding_challenges(
        &self,
        transcript: &mut Transcript,
        row_evaluation: &RowEvaluation,
    ) -> FoldingChallenges {
        transcript.absorb_bytes(b"bin-ntt/row-evaluation");
        let mut bytes = Vec::with_capacity(24 * row_evaluation.values.len());
        for x in &row_evaluation.values {
            for limb in x.0 {
                bytes.extend_from_slice(&limb.to_le_bytes());
            }
        }
        transcript.absorb_bytes(&bytes);
        FoldingChallenges {
            challenges: (0..self.params.columns())
                .map(|_| sample_short_challenge(transcript, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
                .collect(),
        }
    }

    /// `sum_j c_j C_j` per modulus. Multiplication by a challenge — an element of the subring
    /// `R_162` — acts on the four components of a commitment alike and slot-wise in the `R_162`
    /// transform, so the whole step is four length-`r` inner products per slot and modulus.
    pub fn fold_commitment(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
    ) -> FoldedCommitment {
        assert_eq!(
            challenges.len(),
            commitment.columns(),
            "one challenge per column"
        );
        let limbs = self.key.limbs();
        let mut rows: [PowerOfThreeRingElementWithLimbs; 4] =
            core::array::from_fn(|_| PowerOfThreeRingElementWithLimbs::zero(limbs));
        for k in 0..limbs {
            let q = self.key.prime(k) as i64;
            let chi = challenge_slots162(self.key.prime(k), self.key.is_quadratic(k), &challenges.challenges);
            for (row, out) in rows.iter_mut().enumerate() {
                let mut acc = [0i64; N162];
                for (j, c) in chi.iter().enumerate() {
                    let e = &commitment.matrix.get(row, j).limbs[k].v;
                    for s in 0..N162 {
                        acc[s] += c[s] as i64 * e[s] as i64;
                    }
                }
                for s in 0..N162 {
                    let x = acc[s].rem_euclid(q);
                    out.limbs[k].v[s] = if x > (q - 1) / 2 { (x - q) as i16 } else { x as i16 };
                }
            }
        }
        FoldedCommitment {
            primes: self.params.primes(),
            rows,
        }
    }

    /// `u' = sum_j u_j (c_j mod 2)` over `F162`.
    pub fn fold_row_evaluation(
        &self,
        row_evaluation: &RowEvaluation,
        challenges: &FoldingChallenges,
    ) -> F162 {
        eval::fold_binary(&row_evaluation.values, &challenges.challenges)
    }

    /// The claim check: `u . eq(p1) == t`.
    pub fn verify_evaluation(
        &self,
        point: &EvaluationPoint,
        claimed_value: &F162,
        row_evaluation: &RowEvaluation,
    ) -> Result<(), VerificationError> {
        if row_evaluation.values.len() != self.params.columns()
            || point.p1.len() != self.params.column_log_len as usize
        {
            return Err(VerificationError::Rejected);
        }
        if eval::claim(&row_evaluation.values, &point.p1) == *claimed_value {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The opening check, recomputed from `v` alone: `v` is centered modulo the base modulus,
    /// `A v` equals the folded commitment on every modulus, and `eq(p0) . (v mod 2) == u'`.
    pub fn verify_folded_opening(
        &self,
        folded_commitment: &FoldedCommitment,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = (Q1 as i16 - 1) / 2;
        if v.len() != self.key.len_ring()
            || folded_commitment.primes != self.params.primes()
            || v.iter().any(|e| {
                e.representation != Representation::Coefficients
                    || e.v.iter().any(|x| x.abs() > half)
            })
        {
            return Err(VerificationError::Rejected);
        }

        let mut batches: Vec<Batch32> = (0..v.len() / 32)
            .map(|_| Batch32::zero(Representation::Coefficients))
            .collect();
        for (i, e) in v.iter().enumerate() {
            batches[i / 32].set(i % 32, e);
        }
        for k in 0..self.key.limbs() {
            let (q, quad) = (self.key.prime(k), self.key.is_quadratic(k));
            let mut b = batches.clone();
            forward_limb(q, quad, &mut b);
            let y = a_times_v_limb(q, quad, self.key.row(k), &b);
            let components: [PowerOfThreeRingElement; 4] = components_of(q, quad, &y);
            for (row, c) in components.iter().enumerate() {
                if *c != folded_commitment.rows[row].limbs[k] {
                    return Err(VerificationError::Rejected);
                }
            }
        }

        if eval::binary_check(&point.p0, v, *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }
}

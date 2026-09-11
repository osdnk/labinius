use crate::bd::Dropped;
use crate::challenge::{ShortChallenge, Transcript};
use crate::fields::scalar::F162;
use crate::key::AuxData;
use crate::labrador::{self, PolxBuf};
use crate::recursion;
use crate::ring::{PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix};
use crate::{RingElement162, RingElement648};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use super::*;

// =============================================================================================
// the values that pass between the two parties
// =============================================================================================

/// The commitment: a `4 x columns()` matrix of `R_162` elements per modulus. Row `k` is the
/// coefficient of `X^k` in the basis `1, X, X^2, X^3` of `R_648` over `R_162`, column `j` is the
/// commitment of column `j` of the witness.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Commitment {
    pub(crate) primes: Vec<u16>,
    pub(crate) columns: usize,
    pub(crate) value: CommitmentValue,
}

/// The two forms a commitment takes: the matrix itself, or the Ajtai commitment `T_Y` to the
/// residues of its columns, which is a few `polx` and is what the recursion sends.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CommitmentValue {
    Matrix(VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>),
    Recursive(Arc<PolxBuf>),
    Dropped(Arc<Dropped>),
}

impl Commitment {
    /// A commitment from its parts, for [`crate::wire`].
    pub fn of(primes: Vec<u16>, columns: usize, value: CommitmentValue) -> Commitment {
        Commitment {
            primes,
            columns,
            value,
        }
    }

    pub fn rows(&self) -> usize {
        4
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The primes, in the index order [`element`](Self::element) takes: the base modulus first.
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }

    pub fn value(&self) -> &CommitmentValue {
        &self.value
    }

    /// The matrix. Only a commitment made without recursion holds one.
    pub fn matrix(&self) -> &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs> {
        match &self.value {
            CommitmentValue::Matrix(m) => m,
            CommitmentValue::Recursive(_) => {
                panic!("a recursive commitment is T_Y, not the matrix")
            }
            CommitmentValue::Dropped(_) => {
                panic!("a bit-dropped commitment is its digits, not the matrix")
            }
        }
    }

    pub fn dropped(&self) -> &Arc<Dropped> {
        match &self.value {
            CommitmentValue::Dropped(d) => d,
            _ => panic!("this commitment is not bit-dropped"),
        }
    }

    /// `T_Y`. Only a commitment made with recursion holds one.
    pub fn t_y(&self) -> &Arc<PolxBuf> {
        match &self.value {
            CommitmentValue::Recursive(t) => t,
            _ => panic!("this commitment is the matrix, not T_Y"),
        }
    }

    pub fn element(&self, row: usize, column: usize, modulus_index: usize) -> &RingElement162 {
        &self.matrix().get(row, column).limbs[modulus_index]
    }

    /// The commitment as bytes, for a transcript that carries the proof itself, tagged by the
    /// form it takes: [`crate::wire::pack_tagged_commitment`].
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::wire::pack_tagged_commitment(self)
    }

    /// The inverse of [`to_bytes`](Self::to_bytes), against the parameters the verifier holds.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Commitment, VerificationError> {
        crate::wire::unpack_tagged_commitment(params, bytes).ok_or(VerificationError::Rejected)
    }

    /// Bytes on the wire: `LOGQ`-bit coefficients for `T_Y`, `ceil(log2 q)`-bit slots for the
    /// matrix, which is what [`crate::wire::pack_commitment`] writes.
    pub fn wire_bytes(&self) -> usize {
        match &self.value {
            CommitmentValue::Matrix(m) => crate::wire::commitment_bytes(&self.primes, m.cols()),
            CommitmentValue::Recursive(t) => t.len() * labrador::N * labrador::logq().div_ceil(8),
            CommitmentValue::Dropped(d) => d.wire_bytes(),
        }
    }
}

/// What the prover keeps from a commitment and the fold consumes: the witness's transform modulo
/// the base modulus, in the layout the kernel wrote it, and — with recursion on — the residues
/// `T_Y` opens, which the verifier no longer receives.
pub struct CommitmentOpening {
    pub(super) aux: AuxData,
    pub(super) residues: Option<recursion::limbs::Residues>,
}

impl CommitmentOpening {
    /// The residues `T_Y` opens; `None` unless [`Params::recursion`] is set.
    pub fn residues(&self) -> Option<&recursion::limbs::Residues> {
        self.residues.as_ref()
    }

    /// Mutable access, so that a test can corrupt an opening.
    pub fn residues_mut(&mut self) -> Option<&mut recursion::limbs::Residues> {
        self.residues.as_mut()
    }
}

/// A point of `F162^nu` split the way the witness is: `p0` over the row variables (the index
/// inside a column), `p1` over the column variables.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EvaluationPoint {
    pub(super) p0: Vec<F162>,
    pub(super) p1: Vec<F162>,
}

impl EvaluationPoint {
    /// A point from its two halves, for the key-time layout of [`recursion::setup`].
    pub fn of(p0: Vec<F162>, p1: Vec<F162>) -> EvaluationPoint {
        EvaluationPoint { p0, p1 }
    }

    /// A point given by its `witness_log_len` coordinates most significant first — the order a
    /// multilinear indexed by `i -> 2 i + b` produces, which is what a sumcheck over the flat
    /// witness hands back.
    ///
    /// The witness is `W[i + wdim j]`, so the flat index carries the row `i` in its low
    /// `row_log_len` bits and the column `j` above them, while `p0` and `p1` are read least
    /// significant first. The leading `column_log_len` coordinates are therefore `p1` reversed,
    /// and the trailing `row_log_len` are `p0` reversed.
    pub fn msb_first(params: &Params, coordinates: &[F162]) -> EvaluationPoint {
        assert_eq!(
            coordinates.len(),
            params.witness_log_len as usize,
            "one coordinate per variable of the witness"
        );
        let (high, low) = coordinates.split_at(params.column_log_len as usize);
        EvaluationPoint {
            p0: low.iter().rev().copied().collect(),
            p1: high.iter().rev().copied().collect(),
        }
    }

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
    pub(super) values: Vec<F162>,
}

impl RowEvaluation {
    /// A row evaluation from its values, for [`crate::wire`].
    pub fn of(values: Vec<F162>) -> RowEvaluation {
        RowEvaluation { values }
    }

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
    pub(super) challenges: Vec<ShortChallenge>,
}

impl FoldingChallenges {
    /// Challenges from a list, for the key-time layout of [`recursion::setup`].
    pub fn of(challenges: Vec<ShortChallenge>) -> FoldingChallenges {
        FoldingChallenges { challenges }
    }

    pub fn len(&self) -> usize {
        self.challenges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }

    /// The challenges themselves, for [`crate::recursion`], which needs them as `S`-elements.
    pub fn challenges(&self) -> &[ShortChallenge] {
        &self.challenges
    }
}

/// The amortised witness `v = sum_j c_j W_j`: one column's worth of `R_648` elements in
/// coefficient form, centered, and genuinely small — a coefficient is a sum of `r w` signed 0/1
/// terms with a standard deviation of a few dozen, far below `q1 / 2 = 1944.5`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedWitness {
    pub(super) elements: Vec<RingElement648>,
}

impl FoldedWitness {
    /// A fold from its elements, for [`crate::wire`].
    pub fn of(elements: Vec<RingElement648>) -> FoldedWitness {
        FoldedWitness { elements }
    }

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
    pub(super) primes: Vec<u16>,
    pub(super) rows: [PowerOfThreeRingElementWithLimbs; 4],
}

impl FoldedCommitment {
    pub fn moduli(&self) -> &[u16] {
        &self.primes
    }

    pub fn element(&self, row: usize, modulus_index: usize) -> &RingElement162 {
        &self.rows[row].limbs[modulus_index]
    }
}

/// `T_u = Com_{H_u}(lift(u))`: what the prover sends in place of the left expansion, and what
/// the folding challenges are derived from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LeftExpansionCommitment {
    pub(super) t_u: Arc<PolxBuf>,
}

impl LeftExpansionCommitment {
    pub fn t_u(&self) -> &Arc<PolxBuf> {
        &self.t_u
    }

    pub fn wire_bytes(&self) -> usize {
        self.t_u.len() * labrador::N * labrador::logq().div_ceil(8)
    }
}

/// What the folding challenges are derived from: the left expansion in the clear, or `T_u`.
pub trait FoldingSource {
    fn absorb(&self, transcript: &mut Transcript);
}

impl FoldingSource for RowEvaluation {
    fn absorb(&self, transcript: &mut Transcript) {
        transcript.absorb_bytes(b"labinius/row-evaluation");
        let mut bytes = Vec::with_capacity(24 * self.values.len());
        for x in &self.values {
            for limb in x.0 {
                bytes.extend_from_slice(&limb.to_le_bytes());
            }
        }
        transcript.absorb_bytes(&bytes);
    }
}

impl FoldingSource for LeftExpansionCommitment {
    fn absorb(&self, transcript: &mut Transcript) {
        transcript.absorb_bytes(b"labinius/left-expansion-commitment");
        transcript.absorb_bytes(self.t_u.as_bytes());
    }
}

/// The recursive opening: `T_R`, the exact squared norms of every witness vector, and the
/// LaBRADOR proof of [`crate::recursion`]'s relation.
#[derive(Debug)]
pub struct OpeningProof {
    pub(super) t_r: Arc<PolxBuf>,
    pub(super) norms: Vec<u64>,
    pub(super) proof: labrador::ProofHandle,
    pub(super) timings: OpeningTimings,
    pub(super) phi_bytes: usize,
}

/// Wall clock of each stage of [`Prover::prove_opening`], measured on the run that produced the
/// proof, so that a caller reporting a breakdown does not have to prove twice.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct OpeningTimings {
    pub fold: Duration,
    pub encoding: Duration,
    pub witness: Duration,
    pub t_r: Duration,
    pub masks: Duration,
    pub phi: Duration,
    pub statement: Duration,
    pub labrador: Duration,
}

/// The same for [`Verifier::verify_opening`]; `rebuild` is the whole statement rebuild, of which
/// `layout`, `bound`, `phi` and `statement` are the four measured parts.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct VerifyTimings {
    pub rebuild: Duration,
    pub layout: Duration,
    pub bound: Duration,
    pub phi: Duration,
    pub statement: Duration,
    pub labrador: Duration,
}

impl OpeningProof {
    pub fn t_r(&self) -> &Arc<PolxBuf> {
        &self.t_r
    }

    /// What each stage of the proving run cost.
    pub fn timings(&self) -> &OpeningTimings {
        &self.timings
    }

    /// Bytes the per-proof constraint `phi` took.
    pub fn phi_footprint(&self) -> usize {
        self.phi_bytes
    }

    /// The LaBRADOR proof itself, for a caller that verifies it against a statement it holds.
    pub fn handle(&self) -> &labrador::ProofHandle {
        &self.proof
    }

    pub fn norms(&self) -> &[u64] {
        &self.norms
    }

    /// Mutable access, so that a test can announce a wrong norm.
    pub fn norms_mut(&mut self) -> &mut [u64] {
        &mut self.norms
    }

    /// LaBRADOR's analytic proof size, in KB.
    pub fn labrador_kb(&self) -> f64 {
        self.proof.size_kb()
    }

    pub fn wire_bytes(&self) -> usize {
        self.t_r.len() * labrador::N * labrador::logq().div_ceil(8)
            + 8 * self.norms.len()
            + (self.proof.size_kb() * 1024.0) as usize
    }
}

/// Why the prover could not open.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum OpeningError {
    /// `‖v‖^2` exceeded its cap; the caller retries the round with fresh challenges.
    FoldTooLong { normsq: u64, cap: u64 },
    /// A chain's honest quotient or carry exceeded its gadget's reach; likewise retried.
    GadgetOverflow(recursion::Overflow),
    /// LaBRADOR refused the statement or the witness.
    Labrador(String),
}

impl fmt::Display for OpeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpeningError::FoldTooLong { normsq, cap } => {
                write!(f, "the fold has squared norm {normsq}, above the cap {cap}")
            }
            OpeningError::GadgetOverflow(o) => write!(
                f,
                "{}: {} does not fit {} base-{} digits",
                o.chain, o.magnitude, o.gadget.levels, o.gadget.base
            ),
            OpeningError::Labrador(e) => write!(f, "LaBRADOR refused the opening: {e}"),
        }
    }
}

impl std::error::Error for OpeningError {}

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

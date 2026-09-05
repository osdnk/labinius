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
//!
//! With [`Params::recursion`] the last three messages become one LaBRADOR proof of
//! [`crate::recursion`]'s relation:
//!
//! ```text
//!     (T_Y, opening) = prover.commit(w)
//!     p              = verifier.derive_evaluation_point(transcript, T_Y)
//!     T_u            = prover.commit_left_expansion(w.row_evaluate(p))
//!     c              = verifier.derive_folding_challenges(transcript, T_u)
//!     (T_R, eta, pi) = prover.prove_opening(transcript, opening, c, p, T_u, u, t, T_Y)
//! ```
//!
//! and the verifier accepts when every `eta_i` is under its cap, the no-wraparound bound of
//! [`crate::recursion::bound`] clears `Q / 2`, and LaBRADOR accepts `pi` against the statement it
//! rebuilds from those same public values. Both identities above are inside the proof.
use crate::api::{
    components_of, AuxData, CommitmentKey, Modulus, PowerOfThreeRingElement,
    PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162,
};
use crate::bd::Dropped;
use crate::challenge::{
    sample_short_challenge, ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use crate::fields::scalar::F162;
use crate::fold::{a_times_v_forward, fold_columns_slots, fold_witness};
use crate::simd::transpose32 as tr;
use crate::labrador::{self, PolxBuf};
use crate::recursion;
use crate::types::{Batch32, Representation};
use crate::{eval, RingElement162, RingElement648};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// The base modulus listed again among the extra ones.
    BaseIsAlsoExtra(Modulus),
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
            ParamError::BaseIsAlsoExtra(m) => {
                write!(f, "the base modulus {m:?} is listed again as an extra one")
            }
        }
    }
}

impl std::error::Error for ParamError {}

/// The shape of one round: a witness of `2^witness_log_len` `F162` read as `2^column_log_len`
/// columns, committed modulo `base` and every entry of `extra_moduli`.
///
/// `base` is the limb the round is anchored in: the commitment keeps the witness's transform
/// there, the fold runs in its NTT domain and comes back through its inverse transform, and the
/// folded witness is the centered integer vector modulo its prime. Any of the seven
/// [`Modulus`]es can take the part; [`Params::new`] gives the default one, 3889.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Params {
    pub witness_log_len: u32,
    pub column_log_len: u32,
    pub base: Modulus,
    pub extra_moduli: Vec<Modulus>,
    pub opening: Opening,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opening {
    Clear,
    BitDropped { bits: u32 },
    /// Recurse the folded opening into LaBRADOR: the commitment becomes `T_Y`, the left expansion
    /// `T_u`, and the fold a proof of [`crate::recursion`]'s relation instead of `v` itself.
    Recursive,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Suite {
    pub name: &'static str,
    pub witness_log_len: u32,
    pub column_log_len_clear: u32,
    pub column_log_len_recursive: u32,
    /// The base modulus first, then the extra ones.
    pub moduli: &'static [Modulus],
    pub moduli_bd: &'static [Modulus],
    pub dropped_bits: u32,
}

pub const SUITES: [Suite; 4] = [
    Suite {
        name: "sizes",
        witness_log_len: 18,
        column_log_len_clear: 7,
        column_log_len_recursive: 8,
        moduli: &[Modulus::Q9721_FS_S, Modulus::Q12637_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 10,
    },
    Suite {
        name: "sizem",
        witness_log_len: 20,
        column_log_len_clear: 8,
        column_log_len_recursive: 9,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 9,
    },
    Suite {
        name: "sizel",
        witness_log_len: 22,
        column_log_len_clear: 9,
        column_log_len_recursive: 10,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 8,
    },
    Suite {
        name: "sizexl",
        witness_log_len: 24,
        column_log_len_clear: 10,
        column_log_len_recursive: 11,
        moduli: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q9721_FS_S],
        moduli_bd: &[Modulus::Q3889_FS_S, Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        dropped_bits: 7,
    },
];

impl Suite {
    /// The suite `--suite s|m|l|xl` names, `None` for anything else.
    pub fn from_flag(flag: &str) -> Option<&'static Suite> {
        SUITES.iter().find(|r| r.name == format!("size{flag}"))
    }

    /// Position in [`SUITES`], for the tables a caller indexes by suite.
    pub fn index(&self) -> usize {
        ((self.witness_log_len - SUITES[0].witness_log_len) / 2) as usize
    }
}

/// The suite every binary takes `--suite` for, defaulting to the smallest.
pub fn suite_from_args() -> &'static Suite {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--suite" {
            let flag = args.next().expect("--suite takes s, m, l or xl");
            return Suite::from_flag(&flag).expect("--suite takes s, m, l or xl");
        }
    }
    &SUITES[0]
}

impl Params {
    /// The checked constructor over the default base modulus 3889.
    pub fn new(
        witness_log_len: u32,
        column_log_len: u32,
        extra_moduli: Vec<Modulus>,
        opening: Opening,
    ) -> Result<Params, ParamError> {
        Params::with_base(
            witness_log_len,
            column_log_len,
            Modulus::BASE,
            extra_moduli,
            opening,
        )
    }

    /// The checked constructor over a chosen base modulus.
    pub fn with_base(
        witness_log_len: u32,
        column_log_len: u32,
        base: Modulus,
        extra_moduli: Vec<Modulus>,
        opening: Opening,
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
            if *m == base {
                return Err(ParamError::BaseIsAlsoExtra(*m));
            }
            if extra_moduli[..i].contains(m) {
                return Err(ParamError::DuplicateModulus(*m));
            }
        }
        Ok(Params {
            witness_log_len,
            column_log_len,
            base,
            extra_moduli,
            opening,
        })
    }

    /// The configuration the crate is tuned for: 2^18 `F162` in 256 columns, moduli 3889 and 9721,
    /// the folded opening in the clear.
    pub fn basic() -> Params {
        Params::new(18, 7, vec![Modulus::Q9721_FS_S], Opening::Clear)
            .expect("the basic parameters are valid")
    }

    pub fn sized(suite: &Suite, opening: Opening) -> Params {
        let column_log_len = match opening {
            Opening::Recursive => suite.column_log_len_recursive,
            _ => suite.column_log_len_clear,
        };
        let list = match opening {
            Opening::BitDropped { .. } => suite.moduli_bd,
            _ => suite.moduli,
        };
        Params::with_base(
            suite.witness_log_len,
            column_log_len,
            list[0],
            list[1..].to_vec(),
            opening,
        )
        .expect("the sized parameters are valid")
    }

    pub fn dropping(mut self, dropped_bits: u32) -> Params {
        self.opening = match dropped_bits {
            0 => Opening::Clear,
            bits => Opening::BitDropped { bits },
        };
        self
    }

    pub fn recursion(&self) -> bool {
        self.opening == Opening::Recursive
    }

    pub fn dropped_bits(&self) -> u32 {
        match self.opening {
            Opening::BitDropped { bits } => bits,
            _ => 0,
        }
    }

    pub fn bd_cap(&self) -> u64 {
        crate::bd::cap(self.columns(), self.dropped_bits())
    }

    /// Witness length in `F162` elements.
    pub fn witness_len(&self) -> usize {
        1usize << self.witness_log_len
    }

    /// The cap on `‖v‖^2` at this shape. The fold has `witness_len / 4` ring elements' worth of
    /// coefficients however the columns are split, so the cap does not depend on `column_log_len`
    /// and the two modes share it.
    pub fn fold_cap(&self) -> u64 {
        (recursion::FOLD_CAP * (self.witness_len() / 4 * crate::params::N) as f64).ceil() as u64
    }

    /// Number of columns the witness is split into — also the number of folding challenges.
    pub fn columns(&self) -> usize {
        1usize << self.column_log_len
    }

    /// log2 of one column in `F162`: the number of row variables of the evaluation point.
    pub fn row_log_len(&self) -> u32 {
        self.witness_log_len - self.column_log_len
    }

    /// One column in `F162` elements.
    pub(crate) fn column_len(&self) -> usize {
        1usize << self.row_log_len()
    }

    /// The primes in index order: the base modulus first, then `extra_moduli`.
    pub fn primes(&self) -> Vec<u16> {
        core::iter::once(self.base.prime())
            .chain(self.extra_moduli.iter().map(|m| m.prime()))
            .collect()
    }
}

/// What the prover sends in place of the folded opening, one shape per [`Opening`] mode.
pub enum OpeningMessage<'a> {
    Clear {
        folded_commitment: &'a FoldedCommitment,
        folded_witness: &'a FoldedWitness,
        folded_row_value: &'a F162,
    },
    BitDropped {
        folded_witness: &'a FoldedWitness,
        folded_row_value: &'a F162,
    },
    Recursive {
        transcript: &'a mut Transcript,
        left: &'a LeftExpansionCommitment,
        claimed_value: &'a F162,
        proof: &'a OpeningProof,
    },
}

/// [`Params`] together with the public matrix `A` expanded from a seed.
pub struct PublicParameters {
    params: Params,
    matrix_seed: [u8; 32],
    key: Arc<CommitmentKey>,
    recursion: Option<Arc<recursion::setup::Setup>>,
}

impl PublicParameters {
    /// Expand `A` — one uniform row of `R_648` per modulus, in the NTT domain — from the seed, and,
    /// with recursion on, everything of [`recursion::setup`] that depends on it.
    pub fn from_seed(params: Params, matrix_seed: [u8; 32]) -> PublicParameters {
        let digest = blake3::hash(&matrix_seed);
        let seed = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let key =
            CommitmentKey::random(params.column_len(), seed, params.base, &params.extra_moduli);
        let mut pp = PublicParameters {
            params,
            matrix_seed,
            key: Arc::new(key),
            recursion: None,
        };
        if pp.params.recursion() {
            let setup = recursion::setup::Setup::new(&pp, &pp.params.clone(), matrix_seed);
            let rank: usize = setup.ranks.iter().sum();
            let warm = labrador::warm_comkey(labrador::comkey_len_for_rank(rank));
            pp.recursion = Some(Arc::new(setup));
            let _ = warm.join();
        }
        pp
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The expanded matrix, for [`crate::recursion`], which needs `A` in coefficient form.
    pub fn key(&self) -> &CommitmentKey {
        &self.key
    }

    /// The key-time data of the recursion; `None` unless [`Params::recursion`] is set.
    pub fn recursion(&self) -> Option<&Arc<recursion::setup::Setup>> {
        self.recursion.as_ref()
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
    columns: usize,
    value: CommitmentValue,
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

    /// The commitment as bytes, for a transcript that carries the proof itself: the `i16` slots
    /// of the matrix in column-major order, or the `polx` image of `T_Y`, after a one-byte tag
    /// and the element count. Wider than [`wire_bytes`](Self::wire_bytes) for `T_Y`, which is
    /// `LOGQ` bits per coefficient rather than a whole `polx`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.wire_bytes());
        match &self.value {
            CommitmentValue::Matrix(m) => {
                out.push(0);
                out.extend_from_slice(&(m.cols() as u32).to_le_bytes());
                for element in m.iter() {
                    for limb in &element.limbs {
                        for slot in limb.v {
                            out.extend_from_slice(&slot.to_le_bytes());
                        }
                    }
                }
            }
            CommitmentValue::Recursive(t) => {
                out.push(1);
                out.extend_from_slice(&(t.len() as u32).to_le_bytes());
                out.extend_from_slice(t.as_bytes());
            }
            CommitmentValue::Dropped(d) => {
                out.push(2);
                out.extend_from_slice(&(d.columns() as u32).to_le_bytes());
                out.extend_from_slice(&crate::wire::pack_dropped(d));
            }
        }
        out
    }

    /// The inverse of [`to_bytes`](Self::to_bytes), against the parameters the verifier holds.
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Commitment, VerificationError> {
        let primes = params.primes();
        let (&tag, rest) = bytes.split_first().ok_or(VerificationError::Rejected)?;
        if rest.len() < 4
            || (tag == 1) != params.recursion()
            || (tag == 2) != (params.dropped_bits() > 0)
        {
            return Err(VerificationError::Rejected);
        }
        let count = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        let body = &rest[4..];
        let value = if tag == 2 {
            if count != params.columns() {
                return Err(VerificationError::Rejected);
            }
            CommitmentValue::Dropped(Arc::new(
                crate::wire::unpack_dropped(params, body)
                    .map_err(|_| VerificationError::Rejected)?,
            ))
        } else if tag == 0 {
            let slots = 4 * count * primes.len() * N162;
            if count != params.columns() || body.len() != 2 * slots {
                return Err(VerificationError::Rejected);
            }
            let mut slot = body
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]));
            let data = (0..4 * count)
                .map(|_| PowerOfThreeRingElementWithLimbs {
                    limbs: (0..primes.len())
                        .map(|_| PowerOfThreeRingElement {
                            v: core::array::from_fn(|_| slot.next().unwrap()),
                        })
                        .collect(),
                })
                .collect();
            CommitmentValue::Matrix(VerticallyAlignedMatrix::new(4, count, data))
        } else {
            CommitmentValue::Recursive(Arc::new(
                PolxBuf::from_bytes(count, body).ok_or(VerificationError::Rejected)?,
            ))
        };
        Ok(Commitment {
            primes,
            columns: params.columns(),
            value,
        })
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
    aux: AuxData,
    residues: Option<recursion::limbs::Residues>,
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
    p0: Vec<F162>,
    p1: Vec<F162>,
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
    values: Vec<F162>,
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
    challenges: Vec<ShortChallenge>,
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
/// terms, two orders of magnitude below `q1 / 2 = 1944.5`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FoldedWitness {
    elements: Vec<RingElement648>,
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

/// `T_u = Com_{H_u}(lift(u))`: what the prover sends in place of the left expansion, and what
/// the folding challenges are derived from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LeftExpansionCommitment {
    t_u: Arc<PolxBuf>,
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
        transcript.absorb_bytes(b"bin-ntt/row-evaluation");
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
        transcript.absorb_bytes(b"bin-ntt/left-expansion-commitment");
        transcript.absorb_bytes(self.t_u.as_bytes());
    }
}

/// The recursive opening: `T_R`, the exact squared norms of every witness vector, and the
/// LaBRADOR proof of [`crate::recursion`]'s relation.
#[derive(Debug)]
pub struct OpeningProof {
    t_r: Arc<PolxBuf>,
    norms: Vec<u64>,
    proof: labrador::ProofHandle,
    timings: OpeningTimings,
    phi_bytes: usize,
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
    setup: Option<Arc<recursion::setup::Setup>>,
    workspace: Option<AuxData>,
}

impl Prover {
    /// Allocate the workspace and warm it up: one commitment and one fold over a zero witness,
    /// which touches every page of the workspace and runs every kernel and lazy table once.
    pub fn new(pp: &PublicParameters) -> Prover {
        let mut prover = Prover {
            params: pp.params.clone(),
            key: pp.key.clone(),
            setup: pp.recursion.clone(),
            workspace: None,
        };
        let verifier = Verifier::new(pp);
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
        if !prover.params.recursion() && prover.params.dropped_bits() == 0 {
            std::hint::black_box(verifier.fold_commitment(&commitment, &challenges));
        }
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
        let columns = matrix.cols();
        let primes = self.params.primes();
        let (value, residues) = match &self.setup {
            None if self.params.dropped_bits() > 0 => {
                let dropped = crate::bd::drop_bits(&matrix, &primes, self.params.dropped_bits());
                (CommitmentValue::Dropped(Arc::new(dropped)), None)
            }
            None => (CommitmentValue::Matrix(matrix), None),
            Some(setup) => {
                let residues = recursion::limbs::residues(&matrix, &primes);
                let parts: Vec<&[i16]> = (0..setup.residues.len())
                    .map(|k| residues.flat(k))
                    .collect();
                let t_y = Arc::new(setup.key_y.commit_blocks(&parts));
                (CommitmentValue::Recursive(t_y), Some(residues))
            }
        };
        (
            Commitment {
                primes,
                columns,
                value,
            },
            CommitmentOpening { aux, residues },
        )
    }

    /// `T_u = Com_{H_u}(lift(u))`, the message the folding challenges are derived from when the
    /// left expansion is not sent.
    pub fn commit_left_expansion(&self, row: &RowEvaluation) -> LeftExpansionCommitment {
        let setup = self.setup.as_ref().expect("recursion is off");
        let mut u = vec![0i16; setup.ranks[recursion::U] * labrador::N];
        let lifts: Vec<[recursion::Poly; recursion::CHUNKS]> = row
            .values
            .iter()
            .map(|x| recursion::chunk::chunks(&recursion::binary::lift(x)))
            .collect();
        for b in 0..recursion::CHUNKS {
            for (j, l) in lifts.iter().enumerate() {
                let at = (b * setup.r + j) * labrador::N;
                u[at..at + labrador::N].copy_from_slice(&l[b]);
            }
        }
        LeftExpansionCommitment {
            t_u: Arc::new(setup.key_u.commit_blocks(&[&u])),
        }
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
            self.key.prime(0),
        );
        self.workspace = Some(opening.aux);
        FoldedWitness { elements }
    }

    /// The fold, the encoding of [`crate::recursion`], `T_R`, the exact norms, the mask scalars,
    /// and one LaBRADOR proof of the whole relation.
    pub fn prove_opening(
        &mut self,
        transcript: &mut Transcript,
        opening: CommitmentOpening,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        left: &LeftExpansionCommitment,
        row: &RowEvaluation,
        claimed_value: &F162,
        commitment: &Commitment,
    ) -> Result<OpeningProof, OpeningError> {
        let setup = self.setup.clone().expect("recursion is off");
        let CommitmentOpening { aux, residues } = opening;
        let residues = residues.expect("the opening holds no residues");
        let mut timings = OpeningTimings::default();
        let clock = Instant::now();
        let folded = self.fold(
            CommitmentOpening {
                aux,
                residues: None,
            },
            challenges,
        );
        timings.fold = clock.elapsed();
        let normsq: u64 = folded
            .elements
            .iter()
            .flat_map(|e| e.v)
            .map(|x| (x as i64 * x as i64) as u64)
            .sum();
        let cap = setup.fold_cap as u64;
        if std::env::var_os("GADGET_STATS").is_some() {
            eprintln!(
                "gadget-stats fold normsq {normsq} cap {cap} fill {:.3}",
                normsq as f64 / cap as f64
            );
        }
        if normsq > cap {
            return Err(OpeningError::FoldTooLong { normsq, cap });
        }

        let clock = Instant::now();
        let instance = recursion::Instance::new(
            &setup,
            &residues,
            &folded,
            row,
            challenges,
            point,
            claimed_value,
        )
        .map_err(OpeningError::GadgetOverflow)?;
        timings.encoding = clock.elapsed();
        let clock = Instant::now();
        let witness = instance.witness();
        timings.witness = clock.elapsed();
        let norms: Vec<u64> = instance.vectors.iter().map(|v| v.betasq()).collect();
        let rest: Vec<&[i16]> = setup
            .rest
            .iter()
            .map(|&i| witness.vectors[i].as_slice())
            .collect();
        let clock = Instant::now();
        let t_r = Arc::new(setup.key_r.commit_blocks(&rest));
        timings.t_r = clock.elapsed();

        absorb_opening(transcript, claimed_value, &t_r, &norms);
        let clock = Instant::now();
        let masks = recursion::statement::Masks::squeeze(&setup, transcript);
        timings.masks = clock.elapsed();
        let digest = statement_digest(transcript);
        let clock = Instant::now();
        let phi = recursion::statement::ProofPhi::new(&setup, &instance);
        timings.phi = clock.elapsed();
        let phi_bytes = phi.footprint();
        let opening = recursion::statement::Opening {
            t_y: commitment.t_y(),
            t_u: &left.t_u,
            t_r: &t_r,
            norms: &norms,
        };
        let clock = Instant::now();
        let statement =
            recursion::statement::build(&setup, &instance, &phi, opening, masks, digest);
        timings.statement = clock.elapsed();
        let clock = Instant::now();
        let proof = labrador::prove(&statement, &labrador::Witness::new(witness.vectors))
            .map_err(OpeningError::Labrador)?;
        timings.labrador = clock.elapsed();
        Ok(OpeningProof {
            t_r,
            norms,
            proof,
            timings,
            phi_bytes,
        })
    }
}

/// The claimed value, `T_R` and the announced norms, in the transcript position `v` had.
fn absorb_opening(transcript: &mut Transcript, claim: &F162, t_r: &PolxBuf, norms: &[u64]) {
    transcript.absorb_bytes(b"bin-ntt/claim");
    for limb in claim.0 {
        transcript.absorb_u64(limb);
    }
    transcript.absorb_bytes(b"bin-ntt/rest-commitment");
    transcript.absorb_bytes(t_r.as_bytes());
    transcript.absorb_bytes(b"bin-ntt/norms");
    for &n in norms {
        transcript.absorb_u64(n);
    }
}

/// The LaBRADOR statement is a deterministic function of everything absorbed so far, so its
/// digest is one derivation of the transcript rather than a hash of the constraints.
fn statement_digest(transcript: &mut Transcript) -> [u8; 32] {
    let mut digest = [0u8; 32];
    transcript.fill(b"bin-ntt/recursion/statement", &mut digest);
    digest
}

// =============================================================================================
// the verifier
// =============================================================================================

/// The verifier: the public parameters and nothing else.
pub struct Verifier {
    params: Params,
    matrix_seed: [u8; 32],
    key: Arc<CommitmentKey>,
    setup: Option<Arc<recursion::setup::Setup>>,
}

impl Verifier {
    pub fn new(pp: &PublicParameters) -> Verifier {
        Verifier {
            params: pp.params.clone(),
            matrix_seed: pp.matrix_seed,
            key: pp.key.clone(),
            setup: pp.recursion.clone(),
        }
    }

    /// The shape, the moduli, the key seed and — with recursion on — LaBRADOR's modulus, absorbed
    /// before anything the prover chooses.
    fn absorb_parameters(&self, transcript: &mut Transcript) {
        transcript.absorb_bytes(b"bin-ntt/parameters");
        transcript.absorb_u64(self.params.witness_log_len as u64);
        transcript.absorb_u64(self.params.column_log_len as u64);
        for q in self.params.primes() {
            transcript.absorb_u64(q as u64);
        }
        transcript.absorb_u64(u64::from(self.params.recursion()));
        if self.params.dropped_bits() > 0 {
            transcript.absorb_bytes(b"bin-ntt/dropped-bits");
            transcript.absorb_u64(self.params.dropped_bits() as u64);
        }
        if self.params.recursion() {
            transcript.absorb_u64(labrador::logq() as u64);
        }
        transcript.absorb_bytes(&self.matrix_seed);
    }

    /// Absorb the commitment, then derive `p = (p0, p1)`: one uniform `F162` per variable from a
    /// single extendable output of the transcript.
    pub fn derive_evaluation_point(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
    ) -> EvaluationPoint {
        self.absorb_parameters(transcript);
        transcript.absorb_bytes(b"bin-ntt/commitment");
        transcript.absorb_u64(commitment.columns() as u64);
        match commitment.value() {
            CommitmentValue::Matrix(m) => {
                for j in 0..commitment.columns() {
                    transcript.absorb_elements(m.column(j));
                }
            }
            CommitmentValue::Recursive(t) => transcript.absorb_bytes(t.as_bytes()),
            CommitmentValue::Dropped(d) => {
                transcript.absorb_bytes(b"bin-ntt/dropped-commitment");
                transcript.absorb_bytes(&crate::wire::pack_dropped(d));
            }
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

    /// Absorb `u`, then derive the `columns()` challenges — weight 28, canonical bound 11, one
    /// transcript derivation per challenge index.
    pub fn derive_folding_challenges(
        &self,
        transcript: &mut Transcript,
        source: &impl FoldingSource,
    ) -> FoldingChallenges {
        source.absorb(transcript);
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
        let columns = commitment.columns();
        let mut rows: [PowerOfThreeRingElementWithLimbs; 4] =
            core::array::from_fn(|_| PowerOfThreeRingElementWithLimbs::zero(limbs));
        let mut cols: Vec<[*const i16; 4]> = vec![[core::ptr::null(); 4]; columns];
        for k in 0..limbs {
            let q = self.key.prime(k);
            for (j, c) in cols.iter_mut().enumerate() {
                for (t, p) in c.iter_mut().enumerate() {
                    *p = commitment.matrix().get(t, j).limbs[k].v.as_ptr();
                }
            }
            let mut out = [[0i16; N162]; 4];
            fold_columns_slots(q, &challenges.challenges, &cols, &mut out);
            for (t, o) in out.iter().enumerate() {
                rows[t].limbs[k].v = *o;
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
    fn verify_folded_opening(
        &self,
        folded_commitment: &FoldedCommitment,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || folded_commitment.primes != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            if e.representation != Representation::Coefficients {
                return Err(VerificationError::Rejected);
            }
            let (sq, top) = unsafe { crate::simd::norm::normsq_and_max(&e.v) }
                .ok_or(VerificationError::Rejected)?;
            normsq += sq;
            worst = worst.max(top);
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }

        let mut batches: Vec<Batch32> = (0..v.len() / 32)
            .map(|_| Batch32::zero(Representation::Coefficients))
            .collect();
        for (b, batch) in batches.iter_mut().enumerate() {
            let mut src = [tr::ZERO_ROW.as_ptr(); 32];
            for (p, s) in src.iter_mut().enumerate() {
                *s = v[32 * b + p].v.as_ptr();
            }
            unsafe { tr::transpose_into(&src, &tr::IDENTITY, batch) };
        }
        for k in 0..self.key.limbs() {
            let q = self.key.prime(k);
            let y = a_times_v_forward(q, self.key.row(k), &batches);
            let components: [PowerOfThreeRingElement; 4] = components_of(q, &y);
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

    fn verify_folded_opening_bd(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        folded_witness: &FoldedWitness,
        point: &EvaluationPoint,
        folded_row_value: &F162,
    ) -> Result<(), VerificationError> {
        let v = &folded_witness.elements;
        let half = ((self.key.prime(0) - 1) / 2) as i32;
        if v.len() != self.key.len_ring() || commitment.moduli() != self.params.primes() {
            return Err(VerificationError::Rejected);
        }
        let dropped = match commitment.value() {
            CommitmentValue::Dropped(d) => d,
            _ => return Err(VerificationError::Rejected),
        };
        if dropped.dropped_bits() != self.params.dropped_bits() {
            return Err(VerificationError::Rejected);
        }
        let mut normsq = 0u64;
        let mut worst = 0i32;
        for e in v {
            if e.representation != Representation::Coefficients {
                return Err(VerificationError::Rejected);
            }
            let (sq, top) = unsafe { crate::simd::norm::normsq_and_max(&e.v) }
                .ok_or(VerificationError::Rejected)?;
            normsq += sq;
            worst = worst.max(top);
        }
        if worst > half || normsq > self.params.fold_cap() {
            return Err(VerificationError::Rejected);
        }

        let residual = crate::bd::residual(&self.key, dropped, &challenges.challenges, v)
            .ok_or(VerificationError::Rejected)?;
        if residual > self.params.bd_cap() as u128 {
            return Err(VerificationError::Rejected);
        }

        if eval::binary_check(&point.p0, v, *folded_row_value) {
            Ok(())
        } else {
            Err(VerificationError::Rejected)
        }
    }

    /// The opening check in the mode [`Params::opening`] names, dispatching to the arm the
    /// message carries. What comes back on acceptance is the wall clock of each stage of the
    /// recursive arm, so that a caller reporting a breakdown does not have to verify twice; the
    /// other two arms leave it zero.
    pub fn verify_opening(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        message: OpeningMessage<'_>,
    ) -> Result<VerifyTimings, VerificationError> {
        match (self.params.opening, message) {
            (
                Opening::Clear,
                OpeningMessage::Clear {
                    folded_commitment,
                    folded_witness,
                    folded_row_value,
                },
            ) => self
                .verify_folded_opening(folded_commitment, folded_witness, point, folded_row_value)
                .map(|()| VerifyTimings::default()),
            (
                Opening::BitDropped { .. },
                OpeningMessage::BitDropped {
                    folded_witness,
                    folded_row_value,
                },
            ) => self
                .verify_folded_opening_bd(
                    commitment,
                    challenges,
                    folded_witness,
                    point,
                    folded_row_value,
                )
                .map(|()| VerifyTimings::default()),
            (
                Opening::Recursive,
                OpeningMessage::Recursive {
                    transcript,
                    left,
                    claimed_value,
                    proof,
                },
            ) => self.verify_recursive_opening(
                transcript,
                commitment,
                left,
                point,
                claimed_value,
                challenges,
                proof,
            ),
            _ => Err(VerificationError::Rejected),
        }
    }

    /// The recursive opening check: the announced norms against their caps, the no-wraparound
    /// bound of [`recursion::bound`] against `Q / 2`, and one LaBRADOR verification of the
    /// statement rebuilt from public data alone.
    #[allow(clippy::too_many_arguments)]
    fn verify_recursive_opening(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
    ) -> Result<VerifyTimings, VerificationError> {
        let mut timings = VerifyTimings::default();
        let clock = Instant::now();
        let statement = self.rebuild(
            transcript,
            commitment,
            left,
            point,
            claimed_value,
            challenges,
            proof,
            &mut timings,
        )?;
        timings.rebuild = clock.elapsed();
        let clock = Instant::now();
        labrador::verify(&statement, &proof.proof).map_err(|_| VerificationError::Rejected)?;
        timings.labrador = clock.elapsed();
        Ok(timings)
    }

    /// The statement the recursive arm of [`verify_opening`](Self::verify_opening) hands to
    /// LaBRADOR: the same function of public data that the prover ran, and what a test compares
    /// against.
    pub fn opening_statement(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
    ) -> Result<labrador::Statement, VerificationError> {
        self.rebuild(
            transcript,
            commitment,
            left,
            point,
            claimed_value,
            challenges,
            proof,
            &mut VerifyTimings::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild(
        &self,
        transcript: &mut Transcript,
        commitment: &Commitment,
        left: &LeftExpansionCommitment,
        point: &EvaluationPoint,
        claimed_value: &F162,
        challenges: &FoldingChallenges,
        proof: &OpeningProof,
        timings: &mut VerifyTimings,
    ) -> Result<labrador::Statement, VerificationError> {
        let setup = self.setup.as_ref().ok_or(VerificationError::Rejected)?;
        if proof.norms.len() != setup.caps.len()
            || proof.norms.iter().zip(&setup.caps).any(|(n, c)| n > c)
        {
            return Err(VerificationError::Rejected);
        }
        absorb_opening(transcript, claimed_value, &proof.t_r, &proof.norms);
        let masks = recursion::statement::Masks::squeeze(setup, transcript);
        let digest = statement_digest(transcript);
        let clock = Instant::now();
        let layout = recursion::Instance::layout(setup, challenges, point, claimed_value);
        timings.layout = clock.elapsed();
        let clock = Instant::now();
        let cleared = layout.clears();
        timings.bound = clock.elapsed();
        if !cleared {
            return Err(VerificationError::Rejected);
        }
        let clock = Instant::now();
        let phi = recursion::statement::ProofPhi::new(setup, &layout);
        timings.phi = clock.elapsed();
        let opening = recursion::statement::Opening {
            t_y: commitment.t_y(),
            t_u: &left.t_u,
            t_r: &proof.t_r,
            norms: &proof.norms,
        };
        let clock = Instant::now();
        let statement = recursion::statement::build(setup, &layout, &phi, opening, masks, digest);
        timings.statement = clock.elapsed();
        Ok(statement)
    }
}

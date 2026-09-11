use crate::challenge::Transcript;
use crate::fields::scalar::F162;
use crate::key::CommitmentKey;
use crate::labrador;
use crate::recursion;
use crate::ring::Modulus;
use std::fmt;
use std::sync::Arc;
use super::*;

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

    /// The default configuration: 2^18 `F162` in 128 columns, moduli 3889 and 9721, the folded
    /// opening in the clear. (The kernels' tuning shape is the 256-column `Params::new(18, 8, ..)`.)
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

    /// The cap on `‖v‖^2` at this shape, `FOLD_CAP * (witness_len / 4) * N`. The fold has
    /// `witness_len / (4 * columns)` ring elements and each coefficient is a sum of `28 * columns`
    /// signed terms, so the product is the same however the columns are split: the cap does not
    /// depend on `column_log_len` and the two modes share it.
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
    pub(super) params: Params,
    pub(super) matrix_seed: [u8; 32],
    pub(super) key: Arc<CommitmentKey>,
    pub(super) recursion: Option<Arc<recursion::setup::Setup>>,
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

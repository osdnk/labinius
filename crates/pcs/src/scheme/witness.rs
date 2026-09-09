use crate::eval;
use crate::fields::scalar::F162;
use std::fmt;
use super::*;

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
    pub(super) params: Params,
    pub(super) elements: Vec<F162>,
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
        hasher.update(b"labinius/witness");
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

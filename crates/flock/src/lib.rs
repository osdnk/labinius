pub mod circuit;
pub mod piop;
pub mod switch;

use labinius::scheme::{Opening as OpeningMode, OpeningMessage};
use labinius::Suite;
use labinius::fields::scalar::{B128 as SB, F162};
use labinius::scheme::{
    Commitment, EvaluationPoint, FoldedWitness, FoldingChallenges, LeftExpansionCommitment,
    OpeningProof, Params, Prover, PublicParameters, RowEvaluation, VerificationError, Verifier,
    Witness,
};
use labinius::wire;
use labinius::Transcript;
use flock_core::verifier::{verify_core_with_grinding, FlockVerifyError};
use flock_field::F128;
use flock_transcript::challenger::{Challenger, FsChallenger};
use std::time::Instant;

pub use circuit::{Hash, Instance};

pub const DOMAIN: &[u8] = b"labinius/flock";

#[derive(Debug)]
pub enum Error {
    Reduction(FlockVerifyError),
    Switch(&'static str),
    Opening(VerificationError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Reduction(e) => write!(f, "flock reduction: {e:?}"),
            Error::Switch(e) => write!(f, "cross-field switch: {e}"),
            Error::Opening(e) => write!(f, "evaluation proof: {e}"),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ProverTiming {
    pub pack: f64,
    pub commit: f64,
    pub bind: f64,
    pub zerocheck: f64,
    pub lincheck: f64,
    pub switch: f64,
    pub opening: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Default)]
pub struct VerifierTiming {
    pub reduce: f64,
    pub switch: f64,
    pub decode: f64,
    pub opening: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Default)]
pub struct Sizes {
    pub commitment: usize,
    pub zerocheck: usize,
    pub lincheck: usize,
    pub switch: usize,
    pub opening: usize,
}

impl Sizes {
    pub const fn total(&self) -> usize {
        self.commitment + self.zerocheck + self.lincheck + self.switch + self.opening
    }
}

pub enum Opening {
    Clear {
        row: Vec<u8>,
        folded: Vec<u8>,
    },
    Recursive {
        left: LeftExpansionCommitment,
        proof: OpeningProof,
    },
}

pub struct Proof {
    pub commitment: Vec<u8>,
    pub core: piop::Core,
    pub switch: switch::Proof,
    pub claimed_value: F162,
    pub opening: Opening,
}

pub struct Session {
    params: Params,
    prover: Prover,
    verifier: Verifier,
}

impl Session {
    pub fn new(suite: &Suite, recursion: bool, matrix_seed: [u8; 32]) -> Session {
        let opening = if recursion {
            OpeningMode::Recursive
        } else {
            OpeningMode::Clear
        };
        Session::with_params(Params::sized(suite, opening), matrix_seed)
    }

    pub fn bd(suite: &Suite, matrix_seed: [u8; 32]) -> Session {
        let opening = OpeningMode::BitDropped {
            bits: suite.dropped_bits,
        };
        Session::with_params(Params::sized(suite, opening), matrix_seed)
    }

    pub fn with_params(params: Params, matrix_seed: [u8; 32]) -> Session {
        let public_parameters = PublicParameters::from_seed(params.clone(), matrix_seed);
        Session {
            params,
            prover: Prover::new(&public_parameters),
            verifier: Verifier::new(&public_parameters),
        }
    }

    pub const fn params(&self) -> &Params {
        &self.params
    }

    pub fn prove(
        &mut self,
        instance: &Instance,
        witness: &(Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
    ) -> (Proof, ProverTiming, Sizes) {
        let (z_packed, a_packed, b_packed, z_lincheck) = witness;
        let whole = Instant::now();
        let mut timing = ProverTiming::default();
        let mut sizes = Sizes::default();

        let start = Instant::now();
        let trace: Vec<SB> = z_packed
            .iter()
            .map(|&x| SB((x.lo as u128) | ((x.hi as u128) << 64)))
            .collect();
        assert_eq!(
            trace.len(),
            self.params.witness_len(),
            "the packed trace is {} F128 but the commitment takes {}",
            trace.len(),
            self.params.witness_len()
        );
        let lifted = self.lift(&trace);
        timing.pack = milliseconds(start);

        let start = Instant::now();
        let (commitment, opening) = self.prover.commit(&lifted);
        timing.commit = milliseconds(start);

        let start = Instant::now();
        sizes.commitment = commitment.wire_bytes();
        let carrier = piop::carrier(&commitment, instance.pcs_params());
        timing.bind = milliseconds(start);

        let mut ch = FsChallenger::new(DOMAIN);
        let mut reductions = piop::Timing::default();
        let core = piop::prove(
            instance.r1cs(),
            instance.pcs_params(),
            &carrier,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            instance.lincheck_circuit(),
            &mut reductions,
            &mut ch,
        );
        timing.bind += reductions.bind;
        timing.zerocheck = reductions.zerocheck;
        timing.lincheck = reductions.lincheck;
        sizes.zerocheck = encoded(&core.zc_proof);
        sizes.lincheck = encoded(&core.lc_proof);

        let start = Instant::now();
        let (switch_proof, switched) = switch::prove(&trace, &core.claims, &mut ch);
        timing.switch = milliseconds(start);
        sizes.switch = 16 * switch_proof.v.iter().map(Vec::len).sum::<usize>()
            + wire::f162_bytes(2 * switch_proof.rounds.len());

        let mut opening_transcript = seeded(&mut ch);
        let point = EvaluationPoint::msb_first(&self.params, &switched.r_pp);

        let start = Instant::now();
        let row = lifted.row_evaluate(&point);
        let opened = if self.params.recursion() {
            let left = self.prover.commit_left_expansion(&row);
            let challenges = self
                .verifier
                .derive_folding_challenges(&mut opening_transcript, &left);
            let proof = self
                .prover
                .prove_opening(
                    &mut opening_transcript,
                    opening,
                    &challenges,
                    &point,
                    &left,
                    &row,
                    &switched.opened,
                    &commitment,
                )
                .expect("the honest fold is within its cap");
            sizes.opening = wire::f162_bytes(1) + left.wire_bytes() + proof.wire_bytes();
            Opening::Recursive { left, proof }
        } else {
            let challenges = self
                .verifier
                .derive_folding_challenges(&mut opening_transcript, &row);
            let folded = self.prover.fold(opening, &challenges);
            let row = wire::pack_row_evaluation(&row);
            let folded = wire::encode(&folded, self.params.base.prime());
            sizes.opening = wire::f162_bytes(1) + row.len() + folded.len();
            Opening::Clear { row, folded }
        };
        timing.opening = milliseconds(start);
        timing.total = milliseconds(whole);

        (
            Proof {
                commitment: commitment.to_bytes(),
                core,
                switch: switch_proof,
                claimed_value: switched.opened,
                opening: opened,
            },
            timing,
            sizes,
        )
    }

    pub fn verify(&self, instance: &Instance, proof: &Proof) -> Result<VerifierTiming, Error> {
        let mut timing = VerifierTiming::default();
        let whole = Instant::now();
        let commitment =
            Commitment::from_bytes(&self.params, &proof.commitment).map_err(Error::Opening)?;
        let carrier = piop::carrier(&commitment, instance.pcs_params());
        let mut ch = FsChallenger::new(DOMAIN);

        let start = Instant::now();
        let params = instance.pcs_params();
        let (ab, c) = verify_core_with_grinding(
            instance.r1cs(),
            &proof.core.zc_proof,
            &proof.core.lc_proof,
            &carrier,
            instance.lincheck_circuit(),
            params.zerocheck_grinding(),
            params.lincheck_grinding(),
            &mut ch,
        )
        .map_err(Error::Reduction)?;
        timing.reduce = milliseconds(start);

        let start = Instant::now();
        let check = switch::verify(&[ab, c], &proof.switch, &mut ch).map_err(Error::Switch)?;
        if check.d == F162::ZERO {
            return Err(Error::Switch("the switch left the opened value unbound"));
        }
        if check.s != check.d * proof.claimed_value {
            return Err(Error::Switch("the opened value does not close the switch"));
        }
        timing.switch = milliseconds(start);

        let mut opening_transcript = seeded(&mut ch);
        let point = EvaluationPoint::msb_first(&self.params, &check.r_pp);

        let mut start = Instant::now();
        match &proof.opening {
            Opening::Clear { row, folded } => {
                let reject = |_| Error::Opening(VerificationError::Rejected);
                let row =
                    wire::unpack_row_evaluation(row, self.params.columns()).map_err(reject)?;
                let folded = wire::decode(folded).map_err(reject)?;
                timing.decode = milliseconds(start);
                start = Instant::now();
                let challenges = self
                    .verifier
                    .derive_folding_challenges(&mut opening_transcript, &row);
                self.verifier
                    .verify_evaluation(&point, &proof.claimed_value, &row)
                    .map_err(Error::Opening)?;
                self.check_fold(&commitment, &challenges, &row, &folded, &point)
                    .map_err(Error::Opening)?;
            }
            Opening::Recursive {
                left,
                proof: opening,
            } => {
                let challenges = self
                    .verifier
                    .derive_folding_challenges(&mut opening_transcript, left);
                self.verifier
                    .verify_opening(
                        &commitment,
                        &challenges,
                        &point,
                        OpeningMessage::Recursive {
                            transcript: &mut opening_transcript,
                            left,
                            claimed_value: &proof.claimed_value,
                            proof: opening,
                        },
                    )
                    .map_err(Error::Opening)?;
            }
        }
        timing.opening = milliseconds(start);
        timing.total = milliseconds(whole);
        Ok(timing)
    }

    fn lift(&self, trace: &[SB]) -> Witness {
        Witness::from_elements(
            &self.params,
            trace.iter().map(|&x| F162::from_b128(x)).collect(),
        )
        .expect("the trace is the witness length")
    }

    fn check_fold(
        &self,
        commitment: &Commitment,
        challenges: &FoldingChallenges,
        row: &RowEvaluation,
        folded: &FoldedWitness,
        point: &EvaluationPoint,
    ) -> Result<(), VerificationError> {
        let folded_row = self.verifier.fold_row_evaluation(row, challenges);
        let folded_commitment = match self.params.opening {
            OpeningMode::Clear => Some(self.verifier.fold_commitment(commitment, challenges)),
            _ => None,
        };
        let message = match &folded_commitment {
            Some(folded_commitment) => OpeningMessage::Clear {
                folded_commitment,
                folded_witness: folded,
                folded_row_value: &folded_row,
            },
            None => OpeningMessage::BitDropped {
                folded_witness: folded,
                folded_row_value: &folded_row,
            },
        };
        self.verifier
            .verify_opening(commitment, challenges, point, message)
            .map(|_| ())
    }
}

fn seeded<Ch: Challenger>(ch: &mut Ch) -> Transcript {
    let mut transcript = Transcript::new(DOMAIN);
    for challenge in ch.sample_f128_vec(2) {
        transcript.absorb_bytes(&challenge.lo.to_le_bytes());
        transcript.absorb_bytes(&challenge.hi.to_le_bytes());
    }
    transcript
}

fn encoded<T: serde::Serialize>(x: &T) -> usize {
    bincode::serialized_size(x).expect("the proof serializes") as usize
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

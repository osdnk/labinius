//! binius64's hash proofs with this crate's commitment in place of its BaseFold oracle.
//!
//! One transcript throughout. binius64's own `ProverTranscript` is the channel: the statement is
//! observed into it, our commitment is written into it before any challenge is drawn, its
//! reductions run unchanged down to the claim `w~(r) = s` on the packed non-public trace, and the
//! cross-field switch of [`switch`] turns that into an `F162` evaluation claim
//! `pi1~(r'') = opened`. The opening's own transcript is then seeded from 32 bytes sampled off
//! that channel, so every challenge it draws is bound to everything before it.
//!
//! The opening's own messages do not go on that tape: `labrador::ProofHandle` is an opaque handle
//! over the C prover's proof, so the recursive opening has no byte form to append, and [`Opening`]
//! carries the two shapes side by side. The clear-text one does have a byte form and is held in
//! it — the row evaluation bit-packed and the folded witness entropy-coded by [`labinius::wire`], the
//! same code the reference binary sends — so its column of [`Sizes`] is measured rather than
//! assumed, and the verifier decodes what it is given before checking it.
pub mod channel;
pub mod circuit;
pub mod liop;
pub mod phases;
pub mod stock;
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
use binius_compute::GlobalAllocator;
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_core::word::Word;
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_prover::{pack_witness, OptimalPackedB128};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{StdChallenger, B128};
use channel::OracleFreeChannel;
use liop::Liop;
use std::time::Instant;

pub use circuit::{Circuit, Hash, MESSAGE_LEN};

/// Why a proof did not verify.
#[derive(Debug)]
pub enum Error {
    Reduction(binius_verifier::Error),
    Switch(&'static str),
    Opening(VerificationError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Reduction(e) => write!(f, "binius64 reduction: {e}"),
            Error::Switch(e) => write!(f, "cross-field switch: {e}"),
            Error::Opening(e) => write!(f, "evaluation proof: {e}"),
        }
    }
}

/// Milliseconds of wall clock per prover stage, named as binius64's phase spans name them.
#[derive(Clone, Copy, Default)]
pub struct ProverTiming {
    pub pack: f64,
    pub commit: f64,
    pub bitand: f64,
    pub shift: f64,
    pub switch: f64,
    pub opening: f64,
    pub total: f64,
}

/// Milliseconds of wall clock per verifier stage. `decode` is the clear-text opening coming off
/// the wire, and is zero when the recursion is on, whose opening never took a byte form.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifierTiming {
    pub commitment: f64,
    pub reduce: f64,
    pub wiring: f64,
    pub switch: f64,
    pub decode: f64,
    pub opening: f64,
    pub total: f64,
}

/// Bytes per part of the proof.
#[derive(Clone, Copy, Default)]
pub struct Sizes {
    /// Our commitment, as the tape carries it.
    pub commitment: usize,
    /// Our commitment in its canonical wire form, which for `T_Y` is narrower than the tape's.
    pub commitment_wire: usize,
    /// binius64's reduction messages.
    pub liop: usize,
    /// The cross-field switch: the 128 partial evaluations and the `l` round messages.
    pub switch: usize,
    /// Our evaluation proof: the claimed value at its packed width, and then either the coded
    /// row evaluation and folded witness or `T_u` and the recursive proof.
    pub opening: usize,
}

impl Sizes {
    pub const fn total(&self) -> usize {
        self.commitment + self.liop + self.switch + self.opening
    }
}

/// The evaluation proof, in the two shapes [`Params::recursion`] gives it.
pub enum Opening {
    /// The two messages as [`labinius::wire`] codes them: 162 bits an `F162` of row evaluation, and
    /// the folded witness against its own histogram.
    Clear { row: Vec<u8>, folded: Vec<u8> },
    Recursive {
        left: LeftExpansionCommitment,
        proof: OpeningProof,
    },
}

/// binius64's transcript tape, our commitment inside it, and the opening that follows.
pub struct Proof {
    pub tape: Vec<u8>,
    pub claimed_value: F162,
    pub opening: Opening,
}

/// The constraint system, the commitment key and both sides' state, set up once.
pub struct Session {
    liop: Liop,
    params: Params,
    prover: Prover,
    verifier: Verifier,
}

impl Session {
    /// `constraint_system` must pack to exactly [`Params::witness_len`] field elements.
    pub fn new(
        constraint_system: ConstraintSystem,
        suite: &Suite,
        recursion: bool,
        matrix_seed: [u8; 32],
    ) -> Session {
        let opening = if recursion {
            OpeningMode::Recursive
        } else {
            OpeningMode::Clear
        };
        Session::with_params(constraint_system, Params::sized(suite, opening), matrix_seed)
    }

    pub fn bd(constraint_system: ConstraintSystem, suite: &Suite, matrix_seed: [u8; 32]) -> Session {
        let opening = OpeningMode::BitDropped {
            bits: suite.dropped_bits,
        };
        Session::with_params(constraint_system, Params::sized(suite, opening), matrix_seed)
    }

    pub fn with_params(
        constraint_system: ConstraintSystem,
        params: Params,
        matrix_seed: [u8; 32],
    ) -> Session {
        let liop = Liop::new(constraint_system);
        assert_eq!(
            liop.log_witness_elems(),
            params.witness_log_len as usize,
            "the packed trace is 2^{} B128 but the commitment takes 2^{}",
            liop.log_witness_elems(),
            params.witness_log_len
        );
        let public_parameters = PublicParameters::from_seed(params.clone(), matrix_seed);
        Session {
            liop,
            params,
            prover: Prover::new(&public_parameters),
            verifier: Verifier::new(&public_parameters),
        }
    }

    pub const fn params(&self) -> &Params {
        &self.params
    }

    pub const fn constraint_system(&self) -> &ConstraintSystem {
        self.liop.constraint_system()
    }

    /// `tamper` flips the low bit of that element of the packed trace after the commitment is
    /// made, so the commitment binds the trace the switch and the opening no longer speak about.
    pub fn prove(
        &mut self,
        witness: &ValueVec,
        tamper: Option<usize>,
    ) -> (Proof, ProverTiming, Sizes) {
        let allocator = GlobalAllocator;
        let whole = Instant::now();
        let mut timing = ProverTiming::default();
        let mut sizes = Sizes::default();
        let mut transcript = ProverTranscript::<StdChallenger>::default();

        let start = Instant::now();
        let packed = pack_witness::<OptimalPackedB128, _>(
            &allocator,
            self.params.witness_log_len as usize,
            witness.non_public(),
        )
        .expect("the trace fits the constraint system's element count");
        let mut trace: Vec<SB> = packed.iter_scalars().map(|x| SB(u128::from(x))).collect();
        timing.pack = milliseconds(start);

        let start = Instant::now();
        let mut lifted = self.lift(&trace);
        let (commitment, opening) = self.prover.commit(&lifted);
        timing.commit = milliseconds(start);
        if let Some(i) = tamper {
            trace[i].0 ^= 1;
            lifted = self.lift(&trace);
        }

        WordIPProverChannel::<B128>::observe_words(&mut transcript, witness.inout());
        let commitment_bytes = commitment.to_bytes();
        transcript
            .message()
            .write_bytes(&(commitment_bytes.len() as u32).to_le_bytes());
        transcript.message().write_bytes(&commitment_bytes);
        sizes.commitment = tape_len(&transcript);
        sizes.commitment_wire = commitment.wire_bytes();

        let mut reductions = liop::ProveTiming::default();
        let eval_point = self.liop.prove::<_, OptimalPackedB128, _>(
            witness,
            &mut transcript,
            &allocator,
            &mut reductions,
        );
        timing.bitand = reductions.bitand;
        timing.shift = reductions.shift;
        sizes.liop = tape_len(&transcript) - sizes.commitment;

        let start = Instant::now();
        let switched = switch::prove(&trace, &eval_point, &mut transcript);
        timing.switch = milliseconds(start);
        sizes.switch = tape_len(&transcript) - sizes.commitment - sizes.liop;

        let mut opening_transcript = seeded(&mut IPProverChannel::<B128>::sample_array::<2>(
            &mut transcript,
        ));
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
                tape: transcript.finalize(),
                claimed_value: switched.opened,
                opening: opened,
            },
            timing,
            sizes,
        )
    }

    pub fn verify(&self, inout: &[Word], proof: &Proof) -> Result<VerifierTiming, Error> {
        let mut timing = VerifierTiming::default();
        let whole = Instant::now();
        let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.tape.clone());
        let mut channel = OracleFreeChannel {
            transcript: &mut transcript,
        };
        let inout = WordIPVerifierChannel::<B128>::observe_words(&mut channel, inout);

        let start = Instant::now();
        let mut length = [0u8; 4];
        channel
            .transcript
            .message()
            .read_bytes(&mut length)
            .map_err(|_| Error::Opening(VerificationError::Rejected))?;
        let mut bytes = vec![0u8; u32::from_le_bytes(length) as usize];
        channel
            .transcript
            .message()
            .read_bytes(&mut bytes)
            .map_err(|_| Error::Opening(VerificationError::Rejected))?;
        let commitment = Commitment::from_bytes(&self.params, &bytes).map_err(Error::Opening)?;
        timing.commitment = milliseconds(start);

        let (eval_point, claim, reductions) = self
            .liop
            .verify(&inout, &mut channel)
            .map_err(Error::Reduction)?;
        timing.reduce = reductions.reduce;
        timing.wiring = reductions.wiring;

        let start = Instant::now();
        let check = switch::verify(claim, &eval_point, &mut channel).map_err(Error::Switch)?;
        if check.d == F162::ZERO {
            return Err(Error::Switch("the switch left the opened value unbound"));
        }
        if check.s != check.d * proof.claimed_value {
            return Err(Error::Switch("the opened value does not close the switch"));
        }
        timing.switch = milliseconds(start);

        let mut opening_transcript = seeded(&mut IPVerifierChannel::<B128>::sample_array::<2>(
            &mut channel,
        ));
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

/// The opening's transcript, seeded with 32 bytes drawn off binius64's channel.
fn seeded(challenges: &mut [B128; 2]) -> Transcript {
    let mut transcript = Transcript::new(b"labinius/keccak");
    for challenge in challenges {
        transcript.absorb_bytes(&u128::from(*challenge).to_le_bytes());
    }
    transcript
}

fn tape_len(transcript: &ProverTranscript<StdChallenger>) -> usize {
    transcript.clone().finalize().len()
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

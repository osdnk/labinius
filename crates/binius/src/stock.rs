//! binius64's own pipeline on the same circuit and witness, for the baseline column: its Merkle
//! oracle commitment, its `B128` ring-switching and its BaseFold opening, driven through
//! `Prover::prove` and `Verifier::verify` with nothing replaced.
use super::phases::Phases;
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_core::word::Word;
use binius_hash::StdHashSuite;
use binius_ip::channel::WordIPVerifierChannel;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{StdChallenger, B128};
use binius_verifier::{Error, Verifier};
use std::time::Instant;

/// The example's default inverse rate (`--log-inv-rate 1`) and its default Merkle hash suite
/// (`--hash-suite sha256`).
pub const LOG_INV_RATE: usize = 1;

pub struct Stock {
    verifier: Verifier<StdHashSuite>,
    prover: Prover<OptimalPackedB128, StdHashSuite>,
}

impl Stock {
    pub fn new(constraint_system: ConstraintSystem) -> Stock {
        let verifier = Verifier::setup(constraint_system, LOG_INV_RATE)
            .expect("the circuit compiles to a valid system");
        let prover = Prover::setup(verifier.clone()).expect("the key collection builds");
        Stock { verifier, prover }
    }

    pub fn prove(&self, witness: &ValueVec) -> (Vec<u8>, Phases) {
        let phases = Phases::default();
        let mut transcript = ProverTranscript::<StdChallenger>::default();
        phases.record(|| {
            self.prover
                .prove(witness, &mut transcript)
                .expect("the witness is valid")
        });
        (transcript.finalize(), phases)
    }

    pub fn verify(&self, inout: &[Word], proof: &[u8]) -> (Result<(), Error>, Phases) {
        let phases = Phases::default();
        let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.to_vec());
        let verified = phases.record(|| self.verifier.verify(inout, &mut transcript));
        (verified, phases)
    }

    /// `Verifier::verify`'s three steps, timed apart: the reductions with the ring switch, the
    /// wiring claim's native discharge, and the BaseFold opening the Merkle channel defers to
    /// `finish`. None of the three carries a phase span of its own that covers only itself.
    pub fn verify_stages(&self, inout: &[Word], proof: &[u8]) -> Result<[f64; 3], Error> {
        let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.to_vec());
        let mut channel = self
            .verifier
            .iop_compiler()
            .create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut transcript);
        let inout = WordIPVerifierChannel::<B128>::observe_words(&mut channel, inout);

        let start = Instant::now();
        let wiring = self.verifier.iop_verifier().verify(&inout, &mut channel)?;
        let reductions = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        wiring.check_native()?;
        let native = start.elapsed().as_secs_f64() * 1e3;

        let start = Instant::now();
        channel.finish()?;
        Ok([reductions, native, start.elapsed().as_secs_f64() * 1e3])
    }
}

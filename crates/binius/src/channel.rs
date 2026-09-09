//! The verifier channel binius64's reductions run on.
//!
//! `reduce_constraints` is bounded on [`IOPVerifierChannel`], but the oracle it names is
//! binius64's Merkle/BaseFold commitment, which this pipeline does not use — our commitment is
//! written into the transcript directly and opened by our own PCS. So the channel is a
//! `VerifierTranscript` that forwards every interactive-protocol method and answers the oracle
//! ones with nothing: no oracle is ever received on this path.
use binius_core::word::Word;
use binius_iop::channel::{Error as IOPError, IOPVerifierChannel, OracleSpec, TransparentEvalFn};
use binius_ip::channel::{
    pack_words_concrete, select_word, subset_sum_word, Error as IPError, IPVerifierChannel,
    WordIPVerifierChannel,
};
use binius_transcript::{fiat_shamir::Challenger, VerifierTranscript};
use binius_verifier::config::B128;

pub struct OracleFreeChannel<'a, C> {
    pub transcript: &'a mut VerifierTranscript<C>,
}

impl<C: Challenger> IPVerifierChannel<B128> for OracleFreeChannel<'_, C> {
    type Elem = B128;

    fn recv_one(&mut self) -> Result<B128, IPError> {
        self.transcript.recv_one()
    }

    fn recv_many(&mut self, n: usize) -> Result<Vec<B128>, IPError> {
        self.transcript.recv_many(n)
    }

    fn recv_array<const N: usize>(&mut self) -> Result<[B128; N], IPError> {
        self.transcript.recv_array()
    }

    fn sample(&mut self) -> B128 {
        IPVerifierChannel::<B128>::sample(self.transcript)
    }

    fn observe_one(&mut self, val: B128) -> B128 {
        self.transcript.observe_one(val)
    }

    fn observe_many(&mut self, vals: &[B128]) -> Vec<B128> {
        self.transcript.observe_many(vals)
    }

    fn assert_zero(&mut self, val: B128) -> Result<(), IPError> {
        IPVerifierChannel::<B128>::assert_zero(self.transcript, val)
    }
}

impl<C: Challenger> WordIPVerifierChannel<B128> for OracleFreeChannel<'_, C> {
    type Word = Word;

    fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
        WordIPVerifierChannel::<B128>::observe_words(self.transcript, words)
    }

    fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
        subset_sum_word(elems, *word)
    }

    fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
        select_word(elems, *word)
    }

    fn sample_bits(&mut self, bits: usize) -> Word {
        WordIPVerifierChannel::<B128>::sample_bits(self.transcript, bits)
    }

    fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
        pack_words_concrete::<B128, B128>(words)
    }
}

impl<C: Challenger> IOPVerifierChannel<B128> for OracleFreeChannel<'_, C> {
    type Oracle = ();

    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &[]
    }

    fn recv_oracle(
        &mut self,
        _log_msg_len: usize,
        _witness_dependent: bool,
    ) -> Result<(), IOPError> {
        unreachable!("the hash constraint systems commit no binius64 oracle")
    }

    fn verify_oracle_relation(
        &mut self,
        _oracle: (),
        _transparent: TransparentEvalFn<B128>,
        _claim: B128,
    ) -> Result<(), IOPError> {
        unreachable!("the hash constraint systems commit no binius64 oracle")
    }
}

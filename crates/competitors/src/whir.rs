use super::basefold::evaluate_elements;
use super::{median_of, milliseconds, once, rate_label, Row, SECURITY_BITS};
use binius_compute::BufferPool;
use binius_core::word::Word;
use binius_hash::StdHashSuite;
use binius_iop::channel::{IOPVerifierChannel, OracleSpec};
use binius_iop::soundness::{Grinding, SoundnessRegime};
use binius_iop::whir::compiler::WHIRVerifierCompiler;
use binius_iop_prover::channel::IOPProverChannel;
use binius_iop_prover::whir::compiler::WHIRProverCompiler;
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::ntt::domain_context::GaoMateerPreExpanded;
use binius_math::ntt::NeighborsLastMultiThread;
use binius_math::multilinear::eq::{eq_ind, eq_ind_partial_eval_in};
use binius_math::FieldVec;
use binius_prover::{pack_witness, OptimalPackedB128};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{StdChallenger, B128};
use binius_verifier::merkle_tree::BinaryMerkleTreeScheme;
use std::time::Instant;

type Packed = OptimalPackedB128;
type Buffer<'a> = FieldVec<Packed, &'a BufferPool>;
type ProverNTT = NeighborsLastMultiThread<GaoMateerPreExpanded<B128>>;

pub struct Pcs {
    verifier: WHIRVerifierCompiler<B128>,
    prover: WHIRProverCompiler<Packed, ProverNTT>,
}

impl Pcs {
    pub fn new(log_len: usize, log_inv_rate: usize) -> Pcs {
        let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
        let verifier = WHIRVerifierCompiler::<B128>::optimal(
            &merkle_scheme,
            vec![OracleSpec::new(log_len)],
            log_inv_rate,
            SoundnessRegime::UniqueDecoding,
            SECURITY_BITS,
            Grinding::NONE,
        )
        .expect("a ladder over this message reaches the security target");
        let domain_context = GaoMateerPreExpanded::<B128>::generate(verifier.max_log_domain_size());
        let ntt = NeighborsLastMultiThread::new(domain_context, 0);
        let prover = WHIRProverCompiler::from_verifier_compiler(&verifier, ntt);
        Pcs { verifier, prover }
    }

    pub fn ladder(&self) -> String {
        let levels = self.verifier.params().levels();
        let rungs: Vec<String> = levels
            .iter()
            .map(|level| format!("1/{}x{}", 1usize << level.log_inv_rate, level.n_queries))
            .collect();
        rungs.join(" ")
    }

    pub fn achieved(&self) -> f64 {
        self.verifier.params().achieved_security_bits(128)
    }
}

fn pack<'a>(pool: &'a BufferPool, log_len: usize, words: &[Word]) -> Buffer<'a> {
    pack_witness::<Packed, _>(&pool, log_len, words).expect("the words fill the buffer exactly")
}

pub fn commit(pcs: &Pcs, pool: &BufferPool, message: &Buffer<'_>) -> Vec<u8> {
    let mut transcript = ProverTranscript::<StdChallenger>::default();
    let mut channel = pcs
        .prover
        .create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(&mut transcript, pool);
    channel.send_oracle(message.as_view());
    channel.finish();
    transcript.finalize()
}

pub fn prove(
    pcs: &Pcs,
    pool: &BufferPool,
    log_len: usize,
    words: &[Word],
    tamper: Option<usize>,
) -> (Vec<u8>, B128, f64) {
    let mut transcript = ProverTranscript::<StdChallenger>::default();
    let mut channel = pcs
        .prover
        .create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(&mut transcript, pool);

    let committed = pack(pool, log_len, words);
    let oracle = channel.send_oracle(committed.as_view());

    let message = match tamper {
        None => committed,
        Some(i) => {
            drop(committed);
            let mut tampered = words.to_vec();
            tampered[i].0 ^= 1;
            pack(pool, log_len, &tampered)
        }
    };

    let eval_point = channel.sample_many(log_len);
    let claim = evaluate_elements(&message, &eval_point);

    let start = Instant::now();
    let eq = eq_ind_partial_eval_in::<_, Packed>(&pool, &eval_point);
    channel.prove_oracle_relation(oracle, eq, claim);
    channel.finalize_oracle(oracle, message);
    channel.finish();
    let opening = milliseconds(start);

    (transcript.finalize(), claim, opening)
}

pub fn verify(
    pcs: &Pcs,
    log_len: usize,
    proof: &[u8],
    claim: B128,
) -> Result<(), binius_iop::channel::Error> {
    let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.to_vec());
    let mut channel = pcs
        .verifier
        .create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut transcript);

    let oracle = channel.recv_oracle(log_len, true)?;
    let eval_point: Vec<B128> = channel.sample_many(log_len);
    let transparent = Box::new(move |point: &[B128]| eq_ind(&eval_point, point));
    channel.verify_oracle_relation(oracle, transparent, claim)?;
    channel.finish()?;
    Ok(())
}

pub fn run(log_len: usize, log_inv_rate: usize, u64s: &[u64]) -> Row {
    let pool = BufferPool::new();
    let pcs = Pcs::new(log_len, log_inv_rate);
    let words: Vec<Word> = u64s.iter().map(|&w| Word(w)).collect();

    let (commit_ms, commitment) = {
        let message = pack(&pool, log_len, &words);
        median_of(3, || commit(&pcs, &pool, &message))
    };

    let (_, (proof, claim, opening)) = once(|| prove(&pcs, &pool, log_len, &words, None));
    let (verify_ms, verified) = median_of(3, || verify(&pcs, log_len, &proof, claim));
    verified.expect("the honest opening verifies");

    let (tampered, tampered_claim, _) = prove(&pcs, &pool, log_len, &words, Some(0));
    assert!(
        verify(&pcs, log_len, &tampered, tampered_claim).is_err(),
        "an opening of a vector the root does not bind must be rejected"
    );

    Row {
        scheme: "binius64 WHIR",
        rate: rate_label(log_inv_rate),
        target: format!("{SECURITY_BITS}"),
        security: format!(
            "unique decoding, {:.1} achieved, no grinding, ladder {}",
            pcs.achieved(),
            pcs.ladder()
        ),
        claim: "element-MLE",
        commit_ms,
        open_ms: opening,
        verify_ms,
        commitment: commitment.len(),
        proof: proof.len() - commitment.len(),
    }
}

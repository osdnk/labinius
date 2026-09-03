//! binius64's own polynomial commitment, standalone: the Merkle-committed oracle, the `B128` ring
//! switch and the BaseFold opening, on a uniformly random vector of 2^log_len `B128` and no
//! circuit at all.
//!
//! The seam is the IOP channel pair binius64's `Prover::prove` and `Verifier::verify` drive:
//! `send_oracle` / `recv_oracle` commit the trace, `binius_prover::ring_switch::prove` turns the
//! `B1` evaluation claim on its bits into the sumcheck claim BaseFold opens,
//! `prove_oracle_relation` / `verify_oracle_relation` queue that opening and `finish` runs it. The
//! compilers are built at the keccak example's hash suite, so the commitment and the opening are
//! the ones `src/hashes/stock.rs` measures inside a whole proof, read at the rate the caller asks
//! for rather than at that example's single one.
use super::{median_of, milliseconds, once, rate_label, Row, SECURITY_BITS};
use binius_compute::BufferPool;
use binius_core::word::Word;
use binius_field::ExtensionField;
use binius_hash::StdHashSuite;
use binius_iop::basefold::compiler::BaseFoldVerifierCompiler;
use binius_iop::channel::{IOPVerifierChannel, OracleSpec};
use binius_iop_prover::basefold::compiler::BaseFoldProverCompiler;
use binius_iop_prover::channel::IOPProverChannel;
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::multilinear::eq::eq_ind_partial_eval;
use binius_math::multilinear::evaluate::evaluate_inplace;
use binius_math::ntt::domain_context::GaoMateerPreExpanded;
use binius_math::ntt::NeighborsLastMultiThread;
use binius_math::FieldVec;
use binius_prover::ring_switch::{fold_1b_rows_for_b128_split, RingSwitchOutput, LOG_SPLIT_BLOCK};
use binius_prover::{pack_witness, OptimalPackedB128};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{StdChallenger, B1, B128};
use binius_verifier::fri::{calculate_n_test_queries, ConstantArityStrategy};
use binius_verifier::merkle_tree::BinaryMerkleTreeScheme;
use binius_verifier::ring_switch;
use std::time::Instant;

/// `log2` of the number of `B1` coordinates one `B128` packs, which is the number of leading
/// coordinates of the evaluation point the ring switch consumes.
pub const LOG_PACKING: usize = <B128 as ExtensionField<B1>>::LOG_DEGREE;

type Packed = OptimalPackedB128;
type Buffer<'a> = FieldVec<Packed, &'a BufferPool>;
type ProverNTT = NeighborsLastMultiThread<GaoMateerPreExpanded<B128>>;

/// The two compilers `Prover::setup` and `Verifier::setup` build, for one non-ZK oracle of
/// 2^log_len elements at `log_inv_rate`, the keccak example's `--hash-suite sha256`, and a single
/// NTT share because the process is pinned to one core.
pub struct Pcs {
    verifier: BaseFoldVerifierCompiler<B128>,
    prover: BaseFoldProverCompiler<Packed, ProverNTT>,
    log_inv_rate: usize,
}

impl Pcs {
    pub fn new(log_len: usize, log_inv_rate: usize) -> Pcs {
        let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
        let arity = ConstantArityStrategy::with_optimal_arity::<B128, _>(
            &merkle_scheme,
            log_len + log_inv_rate,
        );
        let verifier = BaseFoldVerifierCompiler::new(
            &merkle_scheme,
            vec![OracleSpec::new(log_len)],
            log_inv_rate,
            calculate_n_test_queries(SECURITY_BITS, log_inv_rate),
            &ConstantArityStrategy::new(arity.arity),
        );
        let domain_context = GaoMateerPreExpanded::<B128>::generate(verifier.max_log_domain_size());
        let ntt = NeighborsLastMultiThread::new(domain_context, 0);
        let prover = BaseFoldProverCompiler::from_verifier_compiler(&verifier, ntt);
        Pcs {
            verifier,
            prover,
            log_inv_rate,
        }
    }

    pub fn n_test_queries(&self) -> usize {
        calculate_n_test_queries(SECURITY_BITS, self.log_inv_rate)
    }
}

/// 2^(log_len + 1) uniform words, which pack into 2^log_len uniform `B128`.
pub fn random_words(log_len: usize, seed: u64) -> Vec<Word> {
    super::random_u64s(log_len, seed)
        .into_iter()
        .map(Word)
        .collect()
}

fn pack<'a>(pool: &'a BufferPool, log_len: usize, words: &[Word]) -> Buffer<'a> {
    pack_witness::<Packed, _>(&pool, log_len, words).expect("the words fill the buffer exactly")
}

/// The claim the opening proves: the multilinear extension of the vector's bits, at `eval_point`.
///
/// The point's low [`LOG_PACKING`] coordinates address the bit within an element and the high ones
/// the element, which is the split `ring_switch::prove` asserts. Folding the bit rows against the
/// high coordinates and then evaluating at the low ones is that extension, read out of the
/// vector's own memory rather than through a 2^(log_len + LOG_PACKING) tensor.
pub fn evaluate_bits(message: &Buffer<'_>, eval_point: &[B128]) -> B128 {
    let suffix = &eval_point[LOG_PACKING..];
    let (lo, hi) = suffix.split_at(suffix.len().min(LOG_SPLIT_BLOCK));
    let folded = fold_1b_rows_for_b128_split(
        message,
        &eq_ind_partial_eval::<B128>(lo),
        &eq_ind_partial_eval::<B128>(hi),
    );
    evaluate_inplace(folded, &eval_point[..LOG_PACKING])
}

/// Milliseconds of the prover's three parts.
#[derive(Clone, Copy, Default)]
pub struct Prove {
    pub commit: f64,
    pub statement: f64,
    pub opening: f64,
}

/// The commitment alone: the Reed-Solomon encoding, the Merkle tree and the root on the tape.
///
/// `finish` writes nothing when no relation was queued, so what the tape holds afterwards is the
/// commitment as it travels.
pub fn commit(pcs: &Pcs, pool: &BufferPool, message: &Buffer<'_>) -> Vec<u8> {
    let mut transcript = ProverTranscript::<StdChallenger>::default();
    let mut channel = pcs
        .prover
        .create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _, _>(
            &mut transcript,
            pool,
        );
    channel.send_oracle(message.as_view());
    channel.finish();
    transcript.finalize()
}

/// The whole prover: the commitment, the evaluation point drawn off the transcript, the claim, the
/// ring switch and the BaseFold opening.
///
/// `tamper` flips the low bit of that word after the commitment is made, so the claim, the ring
/// switch and the opening all speak about a vector the Merkle root does not bind.
pub fn prove(
    pcs: &Pcs,
    pool: &BufferPool,
    log_len: usize,
    words: &[Word],
    tamper: Option<usize>,
) -> (Vec<u8>, B128, Prove) {
    let mut timing = Prove::default();
    let mut transcript = ProverTranscript::<StdChallenger>::default();
    let mut channel = pcs
        .prover
        .create_channel_without_zk_from_transcript::<StdHashSuite, StdChallenger, _, _>(
            &mut transcript,
            pool,
        );

    let committed = pack(pool, log_len, words);
    let (commit_ms, oracle) = once(|| channel.send_oracle(committed.as_view()));
    timing.commit = commit_ms;

    let message = match tamper {
        None => committed,
        Some(i) => {
            drop(committed);
            let mut tampered = words.to_vec();
            tampered[i].0 ^= 1;
            pack(pool, log_len, &tampered)
        }
    };

    let eval_point = channel.sample_many(log_len + LOG_PACKING);
    let (statement_ms, claim) = once(|| evaluate_bits(&message, &eval_point));
    timing.statement = statement_ms;

    let start = Instant::now();
    let RingSwitchOutput {
        rs_eq_ind,
        sumcheck_claim,
    } = binius_prover::ring_switch::prove(&pool, message.as_view(), &eval_point, &mut channel);
    channel.prove_oracle_relation(oracle, rs_eq_ind, sumcheck_claim);
    channel.finalize_oracle(oracle, message);
    channel.finish();
    timing.opening = milliseconds(start);

    (transcript.finalize(), claim, timing)
}

/// The verifier's half: the same point off the same transcript, and the opening closed against it.
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
    let eval_point: Vec<B128> = channel.sample_many(log_len + LOG_PACKING);
    let ring_switch::RingSwitchVerifyOutput {
        eq_r_double_prime,
        sumcheck_claim,
    } = ring_switch::verify(claim, &eval_point, &mut channel).map_err(|e| match e {
        ring_switch::Error::Channel(e) => binius_iop::channel::Error::IPChannel(e),
    })?;

    let suffix = eval_point[LOG_PACKING..].to_vec();
    let transparent =
        Box::new(move |point: &[B128]| ring_switch::eval_rs_eq(&suffix, point, &eq_r_double_prime));
    channel.verify_oracle_relation(oracle, transparent, sumcheck_claim)?;
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

    let (_, (proof, claim, timing)) = once(|| prove(&pcs, &pool, log_len, &words, None));
    let (verify_ms, verified) = median_of(3, || verify(&pcs, log_len, &proof, claim));
    verified.expect("the honest opening verifies");

    let (tampered, tampered_claim, _) = prove(&pcs, &pool, log_len, &words, Some(0));
    assert!(
        verify(&pcs, log_len, &tampered, tampered_claim).is_err(),
        "an opening of a vector the root does not bind must be rejected"
    );

    Row {
        scheme: "binius64 BaseFold",
        rate: rate_label(log_inv_rate),
        security: format!("{SECURITY_BITS} bits, {} queries", pcs.n_test_queries()),
        claim: "bit-MLE",
        commit_ms,
        open_ms: timing.opening,
        verify_ms,
        commitment: commitment.len(),
        proof: proof.len() - commitment.len(),
    }
}

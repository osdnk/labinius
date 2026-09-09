//! The Ligero/Brakedown tensor commitment over an arbitrary systematic linear code, on a
//! multilinear given as 2^log_len `B128`.
//!
//! The evaluation vector is read as a `rows x k` matrix `M`, row-major, so the low `log2(k)`
//! coordinates of a point address a column and the high ones a row. `M` is then held transposed,
//! symbol `i` of every row contiguous, and all `rows` messages are encoded together under
//! [`LinearCode::encode_interleaved`], which leaves the `rows x n` codeword matrix in that same
//! symbol-major order. That order is already the Merkle leaf layout: leaf `j` is column `j`,
//! `rows` contiguous elements of `B128`. The commitment is that root as it travels on the tape,
//! 32 bytes, which is what [`commit`] measures.
//!
//! The point splits as `r = (r_lo, r_hi)` and the equality indicator factors with it,
//! `eq(i, r) = a[col] * b[row]` for `a = eq(r_lo)` and `b = eq(r_hi)`, so the claim is the bilinear
//! form `<b, M a>`. The opening is the two standard tests, answered by the same `t` columns:
//!
//! * proximity. The verifier samples `gamma` in `F^rows` after the root is fixed, the prover sends
//!   `w = gamma^T M`, and each opened column `j` must satisfy `<gamma, M[.][j]> = Enc(w)[j]`.
//! * consistency. The prover sends `u = b^T M` for the public `b`, each opened column must satisfy
//!   `<b, M[.][j]> = Enc(u)[j]`, and the claim must equal `<u, a>`.
//!
//! The proximity test is what makes the consistency test mean anything: it is the only reason the
//! committed matrix is close to a codeword matrix at all, and a codeword matrix is what makes
//! `t` random column agreements bind every row.
//!
//! # The query count
//!
//! Unique decoding, no proximity gap beyond it and no grinding. Write `delta = d/n` for the
//! code's provable relative distance and take the testing radius `e < d/3`, which is the regime
//! Ligero's analysis and Brakedown's Theorem 1 share: a matrix `e`-far from every codeword matrix
//! survives one column with probability at most `1 - e/n`, and the columns are drawn
//! independently, so `t` of them leave `(1 - delta/3)^t`. The two tests are answered by the same
//! columns but are two events, hence one bit of union bound:
//!
//! ```text
//! t = ceil((SECURITY_BITS + 1) / -log2(1 - delta/3))
//! ```
//!
//! The analysis also carries a `(e + 1)/|F|` term for the random combination itself, which over
//! GF(2^128) is below `2^-96` for any `n` under `2^31` and never binds. The columns are drawn
//! independently, repeats and all, because that is the experiment `(1 - delta/3)^t` describes;
//! drawing them distinct would be tighter and is not what is claimed here.
//!
//! Nothing reaches for the list-decoding radius or for a proximity gap above `d/3`, so `t` is far
//! larger than a Ligerito- or BaseFold-style count at the same rate; that is the price of a code
//! that has no more than a distance bound to its name. `delta = 1/2` gives 369 queries, a
//! Brakedown code's `delta = 0.02` about 10000, and a Lightning code's `delta = 0.004` about
//! 50000. Proofs in the tens of megabytes are the honest consequence.
//!
//! # The shape
//!
//! `t` is what fixes the matrix, not the other way round: the `t` columns are drawn from the `n`
//! the codeword has, so a code of these distances needs a matrix that is wide and short, never
//! square. [`balanced_log_k`] takes the code's distance and rate and the security target and
//! returns the message length `k` that minimises the proof,
//!
//! ```text
//! 2k + t * 2^log_len / k   field elements,   smallest at   k = sqrt(t * 2^log_len / 2),
//! ```
//!
//! subject to `2t <= n = k / rate`, which is the margin that keeps the rejection sampling of
//! column indices cheap and the query set from swallowing the codeword. The constraint binds at
//! the sizes these codes are used at and the balanced value binds above them; whichever is larger
//! wins, and a `2^log_len` too short to admit either is a panic rather than a silent reshape.
//! [`Tensor::new`] separately refuses a code with `t >= n` at all.
//!
//! # Costs
//!
//! The prover encodes `rows` messages of `k`, hashes `n` leaves of `rows`, and forms two `k`-long
//! row combinations at `2 * rows * k` multiplications. The verifier encodes twice at length `k`
//! and pays `2 * t * rows` multiplications for the column checks. The proof is
//! `2k` field elements for the two rows, `t * rows` for the opened columns, and the Merkle advice;
//! [`Tensor::proof_bytes`] is that accounting written out.
//!
//! The transpose is one pass over `2^log_len` elements and everything after it reads columns:
//! `gamma^T M` and `b^T M` become `k` dot products over contiguous runs of `rows` rather than
//! `rows` strided passes over a `k`-long accumulator, and the codeword matrix lands in leaf order
//! with no strided scatter. At 2^18 on one i7-11850H core, a single combination costs
//! 0.454 -> 0.415 ms for the Brakedown code at `rows = 8, k = 2^15` and 0.507 -> 0.425 ms for the
//! Lightning code at `rows = 2, k = 2^17`. The encoded buffer is unchanged element for element,
//! so the tree, the openings and the proof are byte for byte what the row-at-a-time layout wrote.
//! The encoding itself waits on the code: the default [`LinearCode::encode_interleaved`]
//! transposes in and out around the same per-row `encode`, 0.42 + 11.77 ms against 8.14 for
//! Brakedown and 0.35 + 5.97 against 5.53 for Lightning, so the commitment is the slower half of
//! the trade until a code encodes interleaved natively.
//!
//! `n` need not be a power of two, but a binary Merkle tree spans one, so the column-major buffer
//! is padded to `2^depth` columns of zeros. Query indices are drawn by rejection from `[0, n)`, so
//! the padding is committed and never opened.

use super::codes::LinearCode;
use super::milliseconds;
use binius_core::word::Word;
use binius_field::Field;
use binius_hash::StdHashSuite;
use binius_iop::merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel};
use binius_iop_prover::merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel};
use binius_ip::channel::{IPVerifierChannel, WordIPVerifierChannel};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::multilinear::eq::eq_ind_partial_eval;
use binius_math::multilinear::evaluate::evaluate;
use binius_math::FieldBuffer;
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::{StdChallenger, B128};
use binius_verifier::merkle_tree::{BinaryMerkleTreeScheme, MerkleTreeScheme};
use std::time::Instant;

type ProverChannel = ProverMerkleTranscriptChannel<
    ProverTranscript<StdChallenger>,
    StdChallenger,
    B128,
    StdHashSuite,
>;
type VerifierChannel = VerifierMerkleTranscriptChannel<
    VerifierTranscript<StdChallenger>,
    StdChallenger,
    B128,
    StdHashSuite,
>;
type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

/// Where the opening stopped, when it stopped.
#[derive(Debug)]
pub enum Error {
    /// A root, a row or a Merkle path did not read, or an opening did not close under the root.
    Channel(binius_iop::merkle_channel::Error),
    /// A column disagreed with the encoding of the random row combination.
    Proximity(usize),
    /// A column disagreed with the encoding of the tensor row combination.
    Consistency(usize),
    /// The consistency row does not evaluate to the claim.
    Evaluation,
    /// The tape held bytes the verifier never read.
    Transcript(binius_transcript::Error),
}

impl From<binius_iop::merkle_channel::Error> for Error {
    fn from(error: binius_iop::merkle_channel::Error) -> Error {
        Error::Channel(error)
    }
}

impl From<binius_ip::channel::Error> for Error {
    fn from(error: binius_ip::channel::Error) -> Error {
        Error::Channel(error.into())
    }
}

impl From<binius_transcript::Error> for Error {
    fn from(error: binius_transcript::Error) -> Error {
        Error::Transcript(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Channel(error) => write!(f, "merkle channel: {error}"),
            Error::Proximity(column) => write!(f, "column {column} fails the proximity test"),
            Error::Consistency(column) => write!(f, "column {column} fails the consistency test"),
            Error::Evaluation => write!(f, "the consistency row misses the claim"),
            Error::Transcript(error) => write!(f, "transcript: {error}"),
        }
    }
}

impl std::error::Error for Error {}

/// The number of columns the two tests open, from the code's provable relative distance.
///
/// `ceil((security_bits + 1) / -log2(1 - delta/3))`, the unique-decoding count the module doc
/// derives.
pub fn n_queries(relative_distance: f64, security_bits: usize) -> usize {
    assert!(relative_distance > 0.0 && relative_distance < 1.0);
    let survival = 1.0 - relative_distance / 3.0;
    (((security_bits + 1) as f64) / -survival.log2()).ceil() as usize
}

/// The message length the matrix is cut at: the balanced split where it fits, the narrowest shape
/// leaving `2t <= n` where it does not.
///
/// # Panics
///
/// When `2^log_len` is too short to hold a message long enough for the query count, which is the
/// only honest answer: no shape of that many elements admits this code at this security target.
pub fn balanced_log_k(
    log_len: usize,
    relative_distance: f64,
    rate: f64,
    security_bits: usize,
) -> usize {
    assert!(rate > 0.0 && rate < 1.0);
    let queries = n_queries(relative_distance, security_bits) as f64;
    let balanced = (queries * (1u64 << log_len) as f64 / 2.0)
        .sqrt()
        .log2()
        .round()
        .max(0.0);
    let feasible = (2.0 * queries * rate).log2().ceil().max(0.0);
    let log_k = balanced.max(feasible) as usize;
    assert!(
        log_k <= log_len,
        "{} queries at rate {rate:.3} need a message of 2^{log_k} to leave 2t <= n, \
         which does not fit in 2^{log_len} elements",
        queries as usize
    );
    log_k
}

/// The scheme at one code and one polynomial size.
pub struct Tensor<'a> {
    code: &'a dyn LinearCode,
    log_len: usize,
    log_k: usize,
    rows: usize,
    depth: usize,
    queries: usize,
    scheme: Scheme,
}

impl<'a> Tensor<'a> {
    /// # Preconditions
    ///
    /// * the code's message length is a power of two and at most `2^log_len`;
    /// * the query count the code's distance forces is below its block length.
    pub fn new(code: &'a dyn LinearCode, log_len: usize, security_bits: usize) -> Tensor<'a> {
        let k = code.message_len();
        assert!(k.is_power_of_two() && k <= 1 << log_len);
        let queries = n_queries(code.relative_distance(), security_bits);
        assert!(
            queries < code.codeword_len(),
            "{} opens {queries} of {} columns at {security_bits} bits: the matrix is too narrow \
             for this distance, cut it at a larger k",
            code.name(),
            code.codeword_len()
        );
        let log_k = k.trailing_zeros() as usize;
        Tensor {
            code,
            log_len,
            log_k,
            rows: 1 << (log_len - log_k),
            depth: code.codeword_len().next_power_of_two().trailing_zeros() as usize,
            queries,
            scheme: Scheme::new(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn n_queries(&self) -> usize {
        self.queries
    }

    /// The whole tape a full opening writes, the root included.
    pub fn proof_bytes(&self) -> usize {
        let layer_depth = self.scheme.optimal_verify_layer(self.queries, self.depth);
        32 + (2 << self.log_k) * 16
            + self.queries * self.rows * 16
            + self
                .scheme
                .proof_size(1 << self.depth, self.queries, layer_depth)
    }

    /// `witness[i * rows + j]` is symbol `i` of row `j`, the layout everything below reads.
    fn witness(&self, values: &[B128]) -> Vec<B128> {
        let k = 1 << self.log_k;
        let mut witness = Vec::with_capacity(values.len());
        for i in 0..k {
            witness.extend((0..self.rows).map(|j| values[j * k + i]));
        }
        witness
    }

    fn columns(&self, witness: &[B128]) -> FieldBuffer<B128> {
        let mut columns = vec![B128::ZERO; self.rows << self.depth];
        let encoded = self.rows * self.code.codeword_len();
        self.code
            .encode_interleaved(self.rows, witness, &mut columns[..encoded]);
        FieldBuffer::new(self.depth + self.log_len - self.log_k, columns)
    }

    fn combine(&self, witness: &[B128], weights: &[B128]) -> Vec<B128> {
        witness
            .chunks_exact(self.rows)
            .map(|column| {
                column
                    .iter()
                    .zip(weights)
                    .map(|(&value, &weight)| weight * value)
                    .sum()
            })
            .collect()
    }

    fn encoded(&self, row: &[B128]) -> Vec<B128> {
        let mut codeword = vec![B128::ZERO; self.code.codeword_len()];
        self.code.encode(row, &mut codeword);
        codeword
    }

    fn sample_columns(&self, mut sample: impl FnMut(usize) -> Word) -> Vec<Word> {
        let n = self.code.codeword_len() as u64;
        let mut columns = Vec::with_capacity(self.queries);
        while columns.len() < self.queries {
            let word = sample(self.depth);
            if word.as_u64() < n {
                columns.push(word);
            }
        }
        columns
    }
}

/// The multilinear extension of `values` at `point`, the claim the opening proves.
pub fn evaluate_elements(values: &[B128], point: &[B128]) -> B128 {
    evaluate(&FieldBuffer::<B128>::from_values(values), point)
}

/// Milliseconds of the prover's two parts.
#[derive(Clone, Copy, Default)]
pub struct Prove {
    pub commit: f64,
    pub opening: f64,
}

/// The commitment alone: the row encodings, the Merkle tree over the columns, and the root on the
/// tape.
pub fn commit(tensor: &Tensor<'_>, values: &[B128]) -> Vec<u8> {
    let mut channel = ProverChannel::new(ProverTranscript::default());
    let columns = tensor.columns(&tensor.witness(values));
    channel.send_merkle_commitment(columns.as_view(), tensor.rows);
    channel.into_transcript().finalize()
}

/// The whole prover: the commitment, then the two combined rows and the `t` opened columns.
///
/// `tamper` flips the low bit of one evaluation after the columns are committed, so the two rows
/// and the claim speak about a matrix the root does not bind.
pub fn prove(
    tensor: &Tensor<'_>,
    values: &[B128],
    point: &[B128],
    tamper: Option<usize>,
) -> (Vec<u8>, B128, Prove) {
    assert_eq!(values.len(), 1 << tensor.log_len);
    assert_eq!(point.len(), tensor.log_len);
    let mut timing = Prove::default();
    let mut channel = ProverChannel::new(ProverTranscript::default());

    let start = Instant::now();
    let witness = tensor.witness(values);
    let columns = tensor.columns(&witness);
    let commitment = channel.send_merkle_commitment(columns.as_view(), tensor.rows);
    timing.commit = milliseconds(start);

    let spoken = match tamper {
        None => witness,
        Some(i) => {
            let mut spoken = witness;
            let k = 1 << tensor.log_k;
            spoken[(i % k) * tensor.rows + i / k] += B128::ONE;
            spoken
        }
    };

    let start = Instant::now();
    let a = eq_ind_partial_eval::<B128>(&point[..tensor.log_k]);
    let b = eq_ind_partial_eval::<B128>(&point[tensor.log_k..]);
    let consistency = tensor.combine(&spoken, &b.iter_scalars().collect::<Vec<_>>());
    let claim = consistency
        .iter()
        .zip(a.iter_scalars())
        .map(|(&u, a)| u * a)
        .sum();

    channel.observe_many(point);
    channel.observe_one(claim);
    let gamma = channel.sample_many(tensor.rows);
    channel.send_many(&tensor.combine(&spoken, &gamma));
    channel.send_many(&consistency);

    let queried = tensor.sample_columns(|bits| channel.sample_bits(bits));
    channel.send_openings(&commitment, columns.as_view(), &queried);
    timing.opening = milliseconds(start);

    (channel.into_transcript().finalize(), claim, timing)
}

/// The verifier's half: the two rows re-encoded, every opened column checked against both, and the
/// claim read off the consistency row.
pub fn verify(
    tensor: &Tensor<'_>,
    proof: &[u8],
    point: &[B128],
    claim: B128,
) -> Result<(), Error> {
    assert_eq!(point.len(), tensor.log_len);
    let mut channel =
        VerifierChannel::new(VerifierTranscript::new(StdChallenger::default(), proof.to_vec()));

    let commitment = channel.recv_merkle_commitment(tensor.rows, tensor.depth)?;
    channel.observe_many(point);
    channel.observe_one(claim);
    let gamma = channel.sample_many(tensor.rows);
    let proximity = tensor.encoded(&channel.recv_many(1 << tensor.log_k)?);
    let consistency_row = channel.recv_many(1 << tensor.log_k)?;
    let consistency = tensor.encoded(&consistency_row);

    let queried = tensor.sample_columns(|bits| channel.sample_bits(bits));
    let opened = channel.recv_openings(&commitment, &queried)?;

    let b = eq_ind_partial_eval::<B128>(&point[tensor.log_k..]);
    for (q, word) in queried.iter().enumerate() {
        let j = word.as_u64() as usize;
        let column = &opened[q * tensor.rows..(q + 1) * tensor.rows];
        let mixed: B128 = gamma.iter().zip(column).map(|(&g, &c)| g * c).sum();
        if mixed != proximity[j] {
            return Err(Error::Proximity(j));
        }
        let tensored: B128 = b.iter_scalars().zip(column).map(|(b, &c)| b * c).sum();
        if tensored != consistency[j] {
            return Err(Error::Consistency(j));
        }
    }

    let a = eq_ind_partial_eval::<B128>(&point[..tensor.log_k]);
    let evaluation: B128 = consistency_row
        .iter()
        .zip(a.iter_scalars())
        .map(|(&u, a)| u * a)
        .sum();
    if evaluation != claim {
        return Err(Error::Evaluation);
    }

    channel.into_transcript().finalize()?;
    Ok(())
}

/// A uniform point of `log_len` coordinates, so a wrapper can open at one without a transcript.
pub fn eval_point(log_len: usize, seed: u64) -> Vec<B128> {
    let mut rng = labinius::rng::Rng::new(seed);
    (0..log_len)
        .map(|_| B128::new(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128))
        .collect()
}

/// The `2^log_len` `B128` a wrapper commits, packed from the driver's words.
pub fn elements(log_len: usize, u64s: &[u64]) -> Vec<B128> {
    assert_eq!(u64s.len(), 2 << log_len);
    u64s.chunks_exact(2)
        .map(|pair| B128::new(((pair[1] as u128) << 64) | pair[0] as u128))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::DenseRandom;
    use crate::SECURITY_BITS;

    fn fixture(log_len: usize, log_k: usize) -> (DenseRandom, Vec<B128>, Vec<B128>) {
        let k = 1 << log_k;
        let code = DenseRandom::new(k, 2 * k, 0x5E ^ log_k as u64);
        let values = elements(log_len, &crate::random_u64s(log_len, 0xA1));
        (code, values, eval_point(log_len, 0xB2))
    }

    #[test]
    fn the_tensor_split_is_the_multilinear() {
        let (code, values, point) = fixture(10, 8);
        let tensor = Tensor::new(&code, 10, SECURITY_BITS);
        let a = eq_ind_partial_eval::<B128>(&point[..tensor.log_k]);
        let b = eq_ind_partial_eval::<B128>(&point[tensor.log_k..]);
        let witness = tensor.witness(&values);
        let row = tensor.combine(&witness, &b.iter_scalars().collect::<Vec<_>>());
        let split: B128 = row
            .iter()
            .zip(a.iter_scalars())
            .map(|(&u, a)| u * a)
            .sum();
        assert_eq!(split, evaluate_elements(&values, &point));
    }

    #[test]
    fn an_honest_opening_verifies() {
        for (log_len, log_k) in [(10, 8), (12, 9), (14, 9)] {
            let (code, values, point) = fixture(log_len, log_k);
            let tensor = Tensor::new(&code, log_len, SECURITY_BITS);
            let (proof, claim, _) = prove(&tensor, &values, &point, None);
            assert_eq!(claim, evaluate_elements(&values, &point));
            verify(&tensor, &proof, &point, claim).expect("the honest opening verifies");
        }
    }

    #[test]
    fn a_tampered_row_is_rejected() {
        let (code, values, point) = fixture(10, 8);
        let tensor = Tensor::new(&code, 10, SECURITY_BITS);
        let (proof, claim, _) = prove(&tensor, &values, &point, Some(7));
        assert!(matches!(
            verify(&tensor, &proof, &point, claim),
            Err(Error::Proximity(_))
        ));
    }

    #[test]
    fn a_tampered_column_is_rejected() {
        let (code, values, point) = fixture(10, 8);
        let tensor = Tensor::new(&code, 10, SECURITY_BITS);
        let (mut proof, claim, _) = prove(&tensor, &values, &point, None);
        let layer_depth = tensor.scheme.optimal_verify_layer(tensor.queries, tensor.depth);
        proof[32 + (2 << tensor.log_k) * 16 + (32 << layer_depth)] ^= 1;
        assert!(matches!(
            verify(&tensor, &proof, &point, claim),
            Err(Error::Channel(_))
        ));
    }

    #[test]
    fn a_tampered_merkle_path_is_rejected() {
        let (code, values, point) = fixture(12, 9);
        let tensor = Tensor::new(&code, 12, SECURITY_BITS);
        let (mut proof, claim, _) = prove(&tensor, &values, &point, None);
        let last = proof.len() - 1;
        proof[last] ^= 1;
        assert!(matches!(
            verify(&tensor, &proof, &point, claim),
            Err(Error::Channel(_))
        ));
    }

    /// The claim is observed into the transcript, so a wrong one moves every later challenge and
    /// the proximity test is what usually catches it.
    #[test]
    fn a_wrong_evaluation_is_rejected() {
        let (code, values, point) = fixture(10, 8);
        let tensor = Tensor::new(&code, 10, SECURITY_BITS);
        let (proof, claim, _) = prove(&tensor, &values, &point, None);
        assert!(verify(&tensor, &proof, &point, claim + B128::ONE).is_err());
    }

    #[test]
    fn the_proof_size_is_the_accounting() {
        for (log_len, log_k) in [(10, 8), (12, 9)] {
            let (code, values, point) = fixture(log_len, log_k);
            let tensor = Tensor::new(&code, log_len, SECURITY_BITS);
            let (proof, _, _) = prove(&tensor, &values, &point, None);
            assert_eq!(proof.len(), tensor.proof_bytes());
            assert_eq!(commit(&tensor, &values).len(), 32);
        }
    }

    #[test]
    fn the_query_count_hits_the_target() {
        for &delta in &[0.5, 0.25, 0.07, 0.02, 0.004] {
            let t = n_queries(delta, SECURITY_BITS);
            assert!((1.0 - delta / 3.0).powi(t as i32) <= 2f64.powi(-(SECURITY_BITS as i32) - 1));
        }
    }

    /// The distances and rates the Brakedown and Lightning codes actually land at, where `t` runs
    /// to five figures and a square matrix has nowhere near enough columns.
    #[test]
    fn the_shape_leaves_the_queries_room() {
        for &(delta, rate) in &[(0.004, 0.720), (0.02, 0.704), (0.07, 0.581), (0.5, 0.5)] {
            let t = n_queries(delta, SECURITY_BITS);
            for log_len in [18, 20, 22, 24] {
                let log_k = balanced_log_k(log_len, delta, rate, SECURITY_BITS);
                let n = ((1u64 << log_k) as f64 / rate) as usize;
                assert!(log_k <= log_len);
                assert!(2 * t <= n, "delta {delta} at 2^{log_len}: {t} queries, {n} columns");
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn a_witness_too_short_for_the_query_count_is_a_panic() {
        balanced_log_k(12, 0.004, 0.720, SECURITY_BITS);
    }
}

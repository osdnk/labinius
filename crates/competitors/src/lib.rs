pub mod basefold;
pub mod brakedown;
pub mod codes;
pub mod ligerito_flock;
pub mod tensor;
pub mod whir;
use bin_ntt_bench::Rng;
use bin_ntt::Suite;
pub use bin_ntt_bench::{median_of, ms as milliseconds, once, pin, pinned};

pub fn log_len(suite: &Suite) -> usize {
    suite.witness_log_len as usize
}

pub const WITNESS_SEED: u64 = 0xC7;

pub const CPU: usize = 3;

/// The default target of the binius64 rows, and the one its own configuration uses.
pub const SECURITY_BITS: usize = binius_verifier::SECURITY_BITS;

/// The targets the table reports every scheme at: binius64's own, and the 100 bits two of the
/// Ligerito profiles are built for, so that the rows can be read against each other.
pub const TARGETS: [usize; 2] = [SECURITY_BITS, 100];

pub struct Row {
    pub scheme: &'static str,
    pub rate: String,
    pub target: String,
    pub security: String,
    pub claim: &'static str,
    pub commit_ms: f64,
    pub open_ms: f64,
    pub verify_ms: f64,
    pub commitment: usize,
    pub proof: usize,
}

pub fn random_u64s(log_len: usize, seed: u64) -> Vec<u64> {
    let mut rng = Rng::new(seed);
    (0..2 << log_len).map(|_| rng.next_u64()).collect()
}

pub fn rate_label(log_inv_rate: usize) -> String {
    format!("1/{}", 1usize << log_inv_rate)
}

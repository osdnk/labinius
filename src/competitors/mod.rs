pub mod basefold;
pub mod ligerito_flock;
pub mod whir;

use crate::rng::Rng;
use crate::scheme::{SIZE_STEP, WITNESS_LOG_LEN};
use std::time::Instant;

pub const LOG_LEN: usize = (WITNESS_LOG_LEN + SIZE_STEP) as usize;

pub const LOG_ELEM_BITS: usize = 7;

pub const LOG_BITS: usize = LOG_LEN + LOG_ELEM_BITS;

pub const WITNESS_SEED: u64 = 0xC7;

pub const CPU: usize = 3;

pub const SECURITY_BITS: usize = binius_verifier::SECURITY_BITS;

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

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

pub fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

pub fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

pub fn median_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut samples = Vec::with_capacity(reps);
    let mut out = None;
    for _ in 0..reps {
        let start = Instant::now();
        let value = std::hint::black_box(f());
        samples.push(milliseconds(start));
        out = Some(value);
    }
    samples.sort_by(f64::total_cmp);
    (samples[reps / 2], out.unwrap())
}

pub fn once<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let start = Instant::now();
    let value = std::hint::black_box(f());
    (milliseconds(start), value)
}

pub fn random_u64s(log_len: usize, seed: u64) -> Vec<u64> {
    let mut rng = Rng::new(seed);
    (0..2 << log_len).map(|_| rng.next_u64()).collect()
}

pub fn rate_label(log_inv_rate: usize) -> String {
    format!("1/{}", 1usize << log_inv_rate)
}

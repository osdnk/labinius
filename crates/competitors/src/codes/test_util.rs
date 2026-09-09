//! What the code tests in this directory share: random and low-weight messages, and the Hamming
//! weight a distance check reads off a codeword.
use bin_ntt::rng::Rng;
use binius_field::Field;
use binius_verifier::config::B128;

pub fn random(len: usize, seed: u64) -> Vec<B128> {
    let mut rng = Rng::new(seed);
    (0..len)
        .map(|_| B128::new(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128))
        .collect()
}

/// A message of Hamming weight exactly `support`, its nonzero positions and values uniform.
pub fn low_weight(len: usize, support: usize, seed: u64) -> Vec<B128> {
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let mut message = vec![B128::ZERO; len];
    let mut placed = 0;
    while placed < support {
        let i = rng.below(len as u32) as usize;
        if message[i] == B128::ZERO {
            let mut v = B128::new(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128);
            if v == B128::ZERO {
                v = B128::ONE;
            }
            message[i] = v;
            placed += 1;
        }
    }
    message
}

pub fn weight(codeword: &[B128]) -> usize {
    codeword.iter().filter(|c| **c != B128::ZERO).count()
}

//! What the code tests in this directory share: random and low-weight messages, and the Hamming
//! weight a distance check reads off a codeword.
use crate::codes::LinearCode;
use labinius::rng::Rng;
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

/// The default [`LinearCode::encode_interleaved`] of [`crate::codes`]: one `encode` per row, with
/// the gather and scatter that the symbol-major layout costs it. The baseline every interleaved
/// kernel here is measured against.
pub fn encode_interleaved_naive(code: &impl LinearCode, rows: usize, messages: &[B128], codeword: &mut [B128]) {
    let (k, n) = (code.message_len(), code.codeword_len());
    let mut message = vec![B128::ZERO; k];
    let mut out = vec![B128::ZERO; n];
    for j in 0..rows {
        for i in 0..k {
            message[i] = messages[i * rows + j];
        }
        code.encode(&message, &mut out);
        for i in 0..n {
            codeword[i * rows + j] = out[i];
        }
    }
}

/// `encode_interleaved` reproduces `encode` on every one of the `rows` messages.
pub fn interleaved_agrees_with_encode(code: &impl LinearCode, rows: usize, seed: u64) {
    let (k, n) = (code.message_len(), code.codeword_len());
    let messages = random(k * rows, seed);
    let mut codeword = vec![B128::ZERO; n * rows];
    code.encode_interleaved(rows, &messages, &mut codeword);
    let mut expected = vec![B128::ZERO; n];
    for j in 0..rows {
        let message: Vec<B128> = (0..k).map(|i| messages[i * rows + j]).collect();
        code.encode(&message, &mut expected);
        for i in 0..n {
            assert_eq!(codeword[i * rows + j], expected[i], "{} rows {rows}, row {j}, symbol {i}", code.name());
        }
    }
}

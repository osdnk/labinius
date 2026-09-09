//! The linear code the tensor commitment of [`crate::brakedown`] is built from, over `B128`.
//!
//! A code is `[n, k, d]` over `B128` and **systematic**: the first `k` symbols of a codeword are
//! the message, which is what lets the opening send a claimed message row and have the verifier
//! re-encode it. `relative_distance` is `d/n`, the *provable* bound the query count is derived
//! from, not a heuristic one.

pub mod brakedown_code;
#[cfg(test)]
mod test_util;

use binius_field::Field;
use binius_verifier::config::B128;

pub trait LinearCode: Sync {
    fn name(&self) -> &'static str;

    /// `k`.
    fn message_len(&self) -> usize;

    /// `n`.
    fn codeword_len(&self) -> usize;

    /// `d/n`, provable.
    fn relative_distance(&self) -> f64;

    /// `codeword[..k] = message`, `codeword.len() == n`.
    fn encode(&self, message: &[B128], codeword: &mut [B128]);

    /// `rows` messages at once, symbol-major: `messages[i * rows + j]` is symbol `i` of message
    /// `j`, and the same for `codeword`.
    ///
    /// A tensor commitment encodes every row of its matrix under one code, so the sparse index
    /// stream is walked once for all `rows` of them and every scattered write lands on
    /// `rows` contiguous elements rather than one. The layout is also the one the Merkle tree
    /// wants, since column `i` of the encoded matrix is `codeword[i * rows..(i + 1) * rows]`.
    fn encode_interleaved(&self, rows: usize, messages: &[B128], codeword: &mut [B128]) {
        let (k, n) = (self.message_len(), self.codeword_len());
        assert_eq!(messages.len(), rows * k);
        assert_eq!(codeword.len(), rows * n);
        let mut message = vec![B128::ZERO; k];
        let mut out = vec![B128::ZERO; n];
        for j in 0..rows {
            for i in 0..k {
                message[i] = messages[i * rows + j];
            }
            self.encode(&message, &mut out);
            for i in 0..n {
                codeword[i * rows + j] = out[i];
            }
        }
    }

    fn rate(&self) -> f64 {
        self.message_len() as f64 / self.codeword_len() as f64
    }
}

/// A systematic random linear code: the parity part is a dense uniform `k x (n - k)` matrix.
///
/// Only for testing the tensor commitment against a code whose distance is not in question; the
/// encoding is `O(k (n - k))` and far too slow for a benchmark.
pub struct DenseRandom {
    k: usize,
    n: usize,
    parity: Vec<B128>,
    distance: f64,
}

impl DenseRandom {
    pub fn new(k: usize, n: usize, seed: u64) -> DenseRandom {
        assert!(n > k && k > 0);
        let mut rng = bin_ntt::rng::Rng::new(seed);
        let parity = (0..k * (n - k))
            .map(|_| B128::new(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128))
            .collect();
        DenseRandom {
            k,
            n,
            parity,
            distance: 0.5,
        }
    }
}

impl LinearCode for DenseRandom {
    fn name(&self) -> &'static str {
        "dense-random"
    }

    fn message_len(&self) -> usize {
        self.k
    }

    fn codeword_len(&self) -> usize {
        self.n
    }

    fn relative_distance(&self) -> f64 {
        self.distance
    }

    fn encode(&self, message: &[B128], codeword: &mut [B128]) {
        assert_eq!(message.len(), self.k);
        assert_eq!(codeword.len(), self.n);
        codeword[..self.k].copy_from_slice(message);
        for j in 0..self.n - self.k {
            let mut acc = B128::ZERO;
            for i in 0..self.k {
                acc += message[i] * self.parity[i * (self.n - self.k) + j];
            }
            codeword[self.k + j] = acc;
        }
    }
}

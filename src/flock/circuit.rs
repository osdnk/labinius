use crate::flock::switch::LOG_PACKING;
use crate::scheme::{SIZE_STEP, WITNESS_LOG_LEN};
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProofMergedLigerito};
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier::FlockVerifyError;
use flock_field::F128;
use flock_prover::r1cs_hashes::blake3::{
    generate_witness_batch_major, Blake3Setup, Compression as Blake3Compression, K_LOG as BLAKE3_K,
    USEFUL_BITS as BLAKE3_USEFUL,
};
use flock_prover::r1cs_hashes::sha2::{
    generate_witness_with_ab_packed_and_lincheck, Compression as Sha2Compression,
    Sha256HybridSetup, K_LOG as SHA256_K, USEFUL_BITS as SHA256_USEFUL,
};
use flock_transcript::challenger::Challenger;

pub const LOG_INV_RATE: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Hash {
    Blake3,
    Sha256,
}

impl Hash {
    pub const ALL: [Hash; 2] = [Hash::Blake3, Hash::Sha256];

    pub const fn name(self) -> &'static str {
        match self {
            Hash::Blake3 => "BLAKE3",
            Hash::Sha256 => "SHA-256",
        }
    }

    pub const fn k_log(self) -> usize {
        match self {
            Hash::Blake3 => BLAKE3_K,
            Hash::Sha256 => SHA256_K,
        }
    }

    pub const fn useful_bits(self) -> usize {
        match self {
            Hash::Blake3 => BLAKE3_USEFUL,
            Hash::Sha256 => SHA256_USEFUL,
        }
    }

    pub const fn compressions_log(self) -> usize {
        (WITNESS_LOG_LEN + SIZE_STEP) as usize + LOG_PACKING - self.k_log()
    }

    pub const fn compressions(self) -> usize {
        1 << self.compressions_log()
    }
}

pub enum Instance {
    Blake3(Blake3Setup, Vec<Blake3Compression>),
    Sha256(Sha256HybridSetup, Vec<Sha2Compression>),
}

impl Instance {
    pub fn new(hash: Hash) -> Instance {
        let n = hash.compressions();
        let instance = match hash {
            Hash::Blake3 => Instance::Blake3(
                Blake3Setup::with_log_inv_rate(n, LOG_INV_RATE),
                (0..n).map(blake3_input).collect(),
            ),
            Hash::Sha256 => Instance::Sha256(
                Sha256HybridSetup::with_log_inv_rate(n, LOG_INV_RATE),
                (0..n).map(sha2_input).collect(),
            ),
        };
        instance.r1cs().statement_digest();
        instance
    }

    pub const fn r1cs(&self) -> &BlockR1cs {
        match self {
            Instance::Blake3(s, _) => &s.r1cs,
            Instance::Sha256(s, _) => &s.r1cs,
        }
    }

    pub const fn pcs_params(&self) -> &PcsParams {
        match self {
            Instance::Blake3(s, _) => &s.pcs_params,
            Instance::Sha256(s, _) => &s.pcs_params,
        }
    }

    pub fn lincheck_circuit(&self) -> &dyn LincheckCircuit {
        self.r1cs().csc_lincheck_circuit()
    }

    pub fn witness(&self) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        match self {
            Instance::Blake3(s, blocks) => generate_witness_batch_major(blocks, s.n_blocks_log()),
            Instance::Sha256(s, blocks) => {
                generate_witness_with_ab_packed_and_lincheck(blocks, s.n_blocks_log())
            }
        }
    }

    pub fn stock_prove<Ch: Challenger>(
        &self,
        ch: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        match self {
            Instance::Blake3(s, blocks) => s.prove_fast(blocks, ch),
            Instance::Sha256(s, blocks) => s.prove_fast(blocks, ch),
        }
    }

    pub fn stock_verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &R1csProofMergedLigerito,
        ch: &mut Ch,
    ) -> Result<R1csClaim, FlockVerifyError> {
        match self {
            Instance::Blake3(s, _) => s.verify(commitment, proof, ch),
            Instance::Sha256(s, _) => s.verify(commitment, proof, ch),
        }
    }
}

fn word(i: usize, j: usize) -> u32 {
    ((i as u64 * 0x9E37_79B9 + j as u64 * 0x85EB_CA6B) ^ 0xC2B2_AE35) as u32
}

fn blake3_input(i: usize) -> Blake3Compression {
    (
        std::array::from_fn(|j| word(i, j)),
        std::array::from_fn(|j| word(i, j + 8)),
        i as u64,
        64,
        0,
    )
}

fn sha2_input(i: usize) -> Sha2Compression {
    (
        std::array::from_fn(|j| word(i, j)),
        std::array::from_fn(|j| word(i, j + 8)),
    )
}

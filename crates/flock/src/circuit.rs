use crate::switch::LOG_PACKING;
use labinius::Suite;
use flock_core::lincheck::{LincheckCircuit, LincheckProof};
use flock_core::pcs::ligerito::embedded_initial_k_or_default;
use flock_core::pcs::{
    open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding, BatchOpeningProofLigerito,
    Commitment, PcsError, PcsParams,
};
use flock_core::proof::{R1csClaim, R1csProofMergedLigerito, ZClaim};
use flock_core::r1cs::BlockR1cs;
use flock_core::schedule::Registry;
use flock_core::verifier::{verify_claims_ligerito, verify_core_with_grinding, FlockVerifyError};
use flock_core::zerocheck::ZerocheckProof;
use flock_field::F128;
use flock_prover::prover::{prove_fast_core, ProveCore};
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

    pub fn compressions_log(self, suite: &Suite) -> usize {
        suite.witness_log_len as usize + LOG_PACKING - self.k_log()
    }

    pub fn compressions(self, suite: &Suite) -> usize {
        1 << self.compressions_log(suite)
    }
}

pub enum Instance {
    Blake3(Blake3Setup, Vec<Blake3Compression>),
    Sha256(Sha256HybridSetup, Vec<Sha2Compression>),
}

impl Instance {
    pub fn new(hash: Hash, suite: &Suite) -> Instance {
        let n = hash.compressions(suite);
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
        instance.registry().digest();
        instance
    }

    pub const fn registry(&self) -> &Registry {
        match self {
            Instance::Blake3(s, _) => &s.registry,
            Instance::Sha256(s, _) => &s.registry,
        }
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

    /// Flock's own end-to-end prover, `prove_fast` -> `prove_fast_ligerito_union`: it builds a
    /// union instance, compacts the witness to the used chunk-columns (46 of 64 lanes for BLAKE3,
    /// 50 for SHA-256 at `sizem`), commits lane-major and opens through the merged/jagged opening
    /// transport. That is a different protocol from the one our columns run, so it is printed as
    /// the "stock union" column and is not the phase-aligned baseline; see [`Self::core_params`].
    pub fn stock_prove<Ch: Challenger>(
        &self,
        ch: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        match self {
            Instance::Blake3(s, blocks) => s.prove_fast(blocks, ch),
            Instance::Sha256(s, blocks) => s.prove_fast(blocks, ch),
        }
    }

    /// Why the "stock core" column exists, and why it is not simply `stock_prove`.
    ///
    /// Our columns run flock's single-table reductions (zerocheck, lincheck) and open the two
    /// resulting claims with our own commitment. `prove_fast` runs the union machinery instead:
    /// a compacted, lane-major commitment and the merged opening. Comparing the two puts two
    /// different protocols in one totals row and says nothing about the commitment scheme. The
    /// core column therefore runs flock through its *core* path — `prove_fast_core` (flock's own
    /// commit, bind, zerocheck and lincheck, with the `s_hat_v` precomputation it hands out) and
    /// `open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding` over the two claims,
    /// verified by `verify_core_with_grinding` and `verify_claims_ligerito` — so that the
    /// committed buffer (the full padded `2^(m-7)` words) and every phase are the same as ours
    /// and the commitment is the only thing that differs.
    ///
    /// The union path could not serve here even if we wanted it to: the batch opening asserts
    /// `!lane_major || n_rs == 0`, i.e. a lane-major (compacted) commitment cannot carry
    /// ring-switched claims, so a phase-aligned baseline has to give up the compaction.
    ///
    /// Everything called is public upstream. The one `pub(crate)` helper on that path,
    /// `open_claims_with_precomputed_ligerito`, is two lines (the `x_inner_rest ‖ x_outer`
    /// concatenation) and is inlined in [`Self::core_open`] as `x_outer_full`. The `PcsParams`
    /// are built here rather than taken from the setup because the setup's are the union's:
    /// `m = dense_m` with the zero lanes dropped, whereas the core path needs `m = r1cs.m` and
    /// `num_lanes: None`, at the same profile and rate.
    ///
    /// What the choice costs against real flock: the union commits less and spends it back in
    /// the merged opening, so union/core prover totals were 1.31 at `sizes` and 1.00 at `sizel`
    /// when this column was added, while the core proof is 8–19% larger than the union's at
    /// every size. Both columns are printed so the reader can see both.
    pub fn core_params(&self) -> PcsParams {
        let m = self.r1cs().m;
        let profile = self.pcs_params().profile;
        PcsParams {
            m,
            log_inv_rate: LOG_INV_RATE,
            log_batch_size: embedded_initial_k_or_default(m, profile),
            profile,
            num_lanes: None,
            merkle_hash: Default::default(),
        }
    }

    pub fn core_reduce<Ch: Challenger>(
        &self,
        params: &PcsParams,
        witness: (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
        ch: &mut Ch,
    ) -> ProveCore {
        let (z_packed, a_packed, b_packed, z_lincheck) = witness;
        prove_fast_core(
            self.r1cs(),
            params,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.lincheck_circuit(),
            ch,
        )
    }

    pub fn core_open<Ch: Challenger>(
        &self,
        params: &PcsParams,
        core: ProveCore,
        ch: &mut Ch,
    ) -> CoreProof {
        let ProveCore {
            zc_proof,
            lc_proof,
            ab,
            c,
            commitment,
            prover_data,
            z_packed,
            s_hat_v_ab,
            s_hat_v_c,
        } = core;
        let config = params
            .ligerito_prover_config()
            .expect("the profile ships a config at this size");
        let x_fulls: Vec<Vec<F128>> = [&ab, &c].iter().map(|cl| x_outer_full(cl)).collect();
        let x_refs: Vec<&[F128]> = x_fulls.iter().map(Vec::as_slice).collect();
        let open = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
            z_packed,
            &prover_data,
            &commitment,
            &x_refs,
            &[s_hat_v_ab.as_deref(), Some(s_hat_v_c.as_slice())],
            &[],
            &self.r1cs().padding_spec(),
            &config,
            params.opening_grinding(),
            ch,
        );
        CoreProof {
            commitment,
            zerocheck: zc_proof,
            lincheck: lc_proof,
            open,
            claim: R1csClaim { ab, c },
        }
    }

    pub fn core_verify_reduce<Ch: Challenger>(
        &self,
        params: &PcsParams,
        proof: &CoreProof,
        ch: &mut Ch,
    ) -> Result<[ZClaim; 2], FlockVerifyError> {
        let (ab, c) = verify_core_with_grinding(
            self.r1cs(),
            &proof.zerocheck,
            &proof.lincheck,
            &proof.commitment,
            self.lincheck_circuit(),
            params.zerocheck_grinding(),
            params.lincheck_grinding(),
            ch,
        )?;
        Ok([ab, c])
    }

    pub fn core_verify_open<Ch: Challenger>(
        &self,
        params: &PcsParams,
        proof: &CoreProof,
        claims: &[ZClaim; 2],
        ch: &mut Ch,
    ) -> Result<(), PcsError> {
        verify_claims_ligerito(&proof.commitment, claims, &proof.open, params, ch)
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

pub struct CoreProof {
    pub commitment: Commitment,
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
    pub open: BatchOpeningProofLigerito,
    pub claim: R1csClaim,
}

fn x_outer_full(claim: &ZClaim) -> Vec<F128> {
    let mut v = claim.point.x_inner_rest.clone();
    v.extend_from_slice(&claim.point.x_outer);
    v
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

use flock_core::lincheck::{prove_padded_with_grinding, LincheckCircuit, LincheckProof, SkipPoint};
use flock_core::merkle::Hash;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::{bind_statement, ZClaim};
use flock_core::r1cs::BlockR1cs;
use flock_core::zerocheck::{prove_packed_padded_with_grinding, ZerocheckProof};
use flock_field::F128;
use flock_transcript::challenger::Challenger;
use std::time::Instant;

pub struct Core {
    pub zc_proof: ZerocheckProof,
    pub lc_proof: LincheckProof,
    pub claims: Vec<ZClaim>,
}

#[derive(Clone, Copy, Default)]
pub struct Timing {
    pub bind: f64,
    pub zerocheck: f64,
    pub lincheck: f64,
}

pub fn carrier(commitment: &crate::scheme::Commitment, params: &PcsParams) -> Commitment {
    let cap = commitment
        .to_bytes()
        .chunks(32)
        .map(|c| {
            let mut node = Hash::default();
            node[..c.len()].copy_from_slice(c);
            node
        })
        .collect();
    Commitment {
        cap,
        params: params.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prove<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    commitment: &Commitment,
    z_packed: &[F128],
    a_packed_f128: &[F128],
    b_packed_f128: &[F128],
    z_packed_lincheck: &[u8],
    lincheck_circuit: &dyn LincheckCircuit,
    timing: &mut Timing,
    ch: &mut Ch,
) -> Core {
    let start = Instant::now();
    bind_statement(ch, r1cs, commitment);
    timing.bind = milliseconds(start);

    let padding = r1cs.padding_spec();
    let start = Instant::now();
    let (zc_proof, zc_claim) = prove_packed_padded_with_grinding(
        raw(a_packed_f128),
        raw(b_packed_f128),
        raw(z_packed),
        r1cs.m,
        &padding,
        pcs_params.zerocheck_grinding(),
        ch,
    );
    timing.zerocheck = milliseconds(start);

    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);
    let start = Instant::now();
    let (lc_proof, lc_claim) = prove_padded_with_grinding(
        z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lincheck_circuit,
        &x_ab,
        pcs_params.lincheck_grinding(),
        ch,
    );
    timing.lincheck = milliseconds(start);

    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(SkipPoint::Phi8(zc_claim.z), &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };
    Core {
        zc_proof,
        lc_proof,
        claims: vec![ab, c],
    }
}

fn raw(x: &[F128]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) }
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

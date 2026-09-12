use super::{median_of, medians, once, rate_label, Row, REPS};
use labinius::rng::Rng;
use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::pcs::ligerito::{embedded_initial_k_or_default, LigeritoProfile};
use flock_core::pcs::{
    commit, open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding,
    verify_opening_batch_ligerito_mixed_with_grinding, DirectEqInd, PackedDirectClaim,
    PackedDirectClaimRef, PcsParams,
};
use flock_core::zerocheck::PaddingSpec;

const HASH_BYTES: usize = 32;

const LOG_PACKING: usize = 7;

const DOMAIN: &[u8] = b"labinius-competitors";

const POINT_SEED: u64 = 0x1D;

pub fn elements(u64s: &[u64]) -> Vec<F128> {
    u64s.chunks_exact(2)
        .map(|pair| F128::new(pair[0], pair[1]))
        .collect()
}

pub const PROFILES: [LigeritoProfile; 4] = [
    LigeritoProfile::Fast,
    LigeritoProfile::Slim,
    LigeritoProfile::Fast100,
    LigeritoProfile::Slim100,
];

pub fn target(profile: LigeritoProfile) -> &'static str {
    match profile {
        LigeritoProfile::Fast | LigeritoProfile::Slim => "~128",
        _ => "100",
    }
}

pub fn label(profile: LigeritoProfile) -> &'static str {
    match profile {
        LigeritoProfile::Fast => "flock-core Ligerito Fast",
        LigeritoProfile::Slim => "flock-core Ligerito Slim",
        LigeritoProfile::Fast100 => "flock-core Ligerito Fast100",
        LigeritoProfile::Slim100 => "flock-core Ligerito Slim100",
        LigeritoProfile::Secure => "flock-core Ligerito Secure",
    }
}

pub fn params(log_len: usize, profile: LigeritoProfile) -> PcsParams {
    let m = log_len + LOG_PACKING;
    PcsParams {
        m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: embedded_initial_k_or_default(m, profile),
        profile,
        num_lanes: None,
        merkle_hash: Default::default(),
    }
}

fn claim(poly: &[F128], eq: &[F128]) -> F128 {
    eq.iter()
        .zip(poly)
        .fold(F128::ZERO, |acc, (&e, &p)| acc + e * p)
}

pub fn run(log_len: usize, profile: LigeritoProfile, u64s: &[u64]) -> Row {
    let params = params(log_len, profile);
    let prover = params
        .ligerito_prover_config()
        .expect("the profile ships a config at this size");
    let verifier = params
        .ligerito_verifier_config()
        .expect("the profile ships a config at this size");
    let grinding = params.opening_grinding();
    let poly = elements(u64s);
    assert_eq!(poly.len(), params.msg_len_f128());

    let (commit_ms, (commitment, prover_data)) = median_of(REPS, || commit(&poly, &params));

    let mut rng = Rng::new(POINT_SEED);
    let point: Vec<F128> = (0..params.log_msg_len())
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let eq = build_eq_table(&point);
    let value = claim(&poly, &eq);

    // The opening takes the polynomial and the eq table by value; the copies are made off the clock.
    let (open_ms, proof) = medians(
        REPS,
        || (poly.clone(), eq.clone()),
        |(poly, eq)| {
            once(|| {
                let mut challenger = FsChallenger::new(DOMAIN);
                open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
                    poly,
                    &prover_data,
                    &commitment,
                    &[],
                    &[],
                    &[PackedDirectClaim {
                        point: point.clone(),
                        value,
                        eq_ind: DirectEqInd::Dense(eq),
                    }],
                    &PaddingSpec::dense(params.m),
                    &prover,
                    grinding,
                    &mut challenger,
                )
            })
        },
    );

    let (verify_ms, verified) = median_of(REPS, || {
        let mut challenger = FsChallenger::new(DOMAIN);
        verify_opening_batch_ligerito_mixed_with_grinding(
            &commitment,
            &[],
            &[],
            &[],
            &[PackedDirectClaimRef {
                point: &point,
                value,
            }],
            &proof,
            &verifier,
            grinding,
            &mut challenger,
        )
    });
    verified.expect("the honest opening verifies");

    let cap = commitment.cap.len() * HASH_BYTES;
    Row {
        scheme: label(profile),
        rate: rate_label(profile.log_inv_rate()),
        target: target(profile).to_string(),
        security: format!(
            "queries {:?}, rates {:?}, grinding {:?}",
            prover.queries, prover.log_inv_rates, prover.grinding_bits
        ),
        claim: "element-MLE, chosen point",
        commit_ms,
        open_ms,
        verify_ms,
        commitment: cap,
        proof: proof.ligerito.size_bytes() - proof.ligerito.initial_cap.len() * HASH_BYTES,
    }
}

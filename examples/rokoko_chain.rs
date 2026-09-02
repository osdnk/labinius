//! Runs one exact-norm chain end to end: `rokoko_chain <n14|n15|n16|n17>`. The witness has
//! the coefficient shape of the recursive opening's committed vector: balanced base-128 digits at
//! positions `[0, 81)`, zero above.
use std::time::Instant;

use bin_ntt::rokoko::config::{outer_rank, sumcheck, Shape};
use bin_ntt::rokoko::{CHUNK, DIGIT};
use rokoko::common::config::MOD_Q;
use rokoko::common::hash::HashWrapper;
use rokoko::common::init_common;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::protocol::config::{
    to_kb, NextRoundCommitment, RoundProof, SizeableProof, SumcheckConfig, SumcheckRoundProof,
};
use rokoko::protocol::crs::CRS;
use rokoko::protocol::parties::{commiter::commit, prover::prover_round, verifier::verifier_round};
use rokoko::protocol::snark::{
    challenge_point, eq, prove_claims, verify_claims, witness_in, Claim, Region,
};
use rokoko::protocol::sumcheck::init_sumcheck;
use rokoko::protocol::sumchecks::builder_verifier::init_verifier;

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn digit(&mut self) -> i64 {
        (self.next() % (DIGIT as u64 + 1)) as i64 - DIGIT / 2
    }
}

fn element(rng: &mut SplitMix) -> RingElement {
    let mut e = RingElement::new(Representation::Coefficients);
    for i in 0..CHUNK {
        e.v[i] = rng.digit().rem_euclid(MOD_Q as i64) as u64;
    }
    e.to_representation(Representation::IncompleteNTT);
    e
}

fn witness(config: &SumcheckConfig) -> VerticallyAlignedMatrix<RingElement> {
    let mut rng = SplitMix(config.witness_height as u64 * 31 + config.witness_width as u64);
    let n = config.witness_height * config.witness_width;
    VerticallyAlignedMatrix {
        height: config.witness_height,
        width: config.witness_width,
        used_cols: config.witness_width,
        data: (0..n).map(|_| element(&mut rng)).collect(),
    }
}

fn kb(elements: &[RingElement]) -> f64 {
    to_kb(elements.iter().map(RingElement::compact_size_in_bits).sum())
}

fn breakdown(proof: &SumcheckRoundProof, round: usize) {
    let polys = to_kb(
        proof
            .polys
            .iter()
            .flat_map(|p| p.coefficients[..p.num_coefficients].iter())
            .map(SizeableProof::size_in_bits)
            .sum(),
    );
    let claims = [
        &proof.claim_over_witness,
        &proof.claim_over_witness_conjugate,
        &proof.norm_claim,
        &proof.most_inner_norm_claim,
    ]
    .into_iter()
    .chain(proof.projection_norm_claim.iter())
    .map(RingElement::compact_size_in_bits)
    .sum::<usize>();
    println!(
        "round {round}: polys {polys:.2} KB, claims {:.2} KB",
        to_kb(claims)
    );
    println!("  rc opening inner {:.2} KB", kb(&proof.rc_opening_inner));
    if let Some(inner) = &proof.rc_coarse_projection_inner {
        println!("  rc coarse projection inner {:.2} KB", kb(inner));
    }
    if let Some((ct, batched)) = &proof.rc_fine_projection_inner {
        println!("  rc fine projection inner {:.2} KB", kb(ct) + kb(batched));
    }
    if let Some(ct) = &proof.constant_term_claims {
        println!("  constant term claims {:.2} KB", kb(ct));
    }
    match &proof.next_round_commitment {
        Some(NextRoundCommitment::Recursive(rc)) => {
            println!("  next round commitment {:.2} KB", kb(rc))
        }
        Some(NextRoundCommitment::Simple(m)) => {
            println!("  next round commitment {:.2} KB", kb(&m.data))
        }
        None => {}
    }
    match proof.next.as_deref() {
        Some(RoundProof::Sumcheck(next)) => breakdown(next, round + 1),
        Some(RoundProof::Simple(simple)) => {
            println!(
                "round {}: folded witness {:.2} KB, projection image ct {:.2} KB, batched projection image {:.2} KB, opening rhs {:.2} KB",
                round + 1,
                kb(&simple.folded_witness.data),
                kb(&simple.projection_image_ct.data),
                kb(&simple.batched_projection_image.data),
                kb(&simple.opening_rhs.data),
            );
        }
        Some(RoundProof::Intermediate(_)) => println!("round {}: intermediate", round + 1),
        None => {}
    }
}

fn main() {
    let shape = match std::env::args().nth(1).as_deref() {
        Some("n14") => Shape::N14,
        Some("n15") => Shape::N15,
        Some("n16") => Shape::N16,
        Some("n17") => Shape::N17,
        _ => panic!("usage: rokoko_chain <n14|n15|n16|n17>"),
    };
    let _guards = rokoko::tracing::setup();
    init_common();

    let config = sumcheck(shape);
    let n = shape.len();
    let crs = CRS::gen_prover_crs(config);
    let verifier_crs = CRS::gen_verifier_crs(config);
    let mut prover_context = init_sumcheck(&crs, config);
    let mut verifier_context = init_verifier(&verifier_crs, config);
    let witness = witness(config);

    let start = Instant::now();
    let (commitment_with_aux, rc_commitment) = commit(&crs, config, &witness);
    let commit_ms = start.elapsed().as_millis();

    let everything = Region::whole(n);
    let claims = |transcript: &mut HashWrapper, values: Option<(RingElement, RingElement)>| {
        transcript.update_with_ring_element_slice(&rc_commitment);
        let point = challenge_point(transcript, n.ilog2() as usize);
        let linear = eq(point) * witness_in(everything);
        let norm = witness_in(everything) * witness_in(everything).conjugate();
        let (t_linear, t_norm) =
            values.unwrap_or_else(|| (linear.sum(&witness), norm.sum(&witness)));
        (
            vec![
                Claim::sums_to(linear, t_linear.clone()),
                Claim::sums_to(norm, t_norm.clone()),
            ],
            (t_linear, t_norm),
        )
    };

    let start = Instant::now();
    let mut transcript = HashWrapper::new();
    let (prover_claims, values) = claims(&mut transcript, None);
    let values_ms = start.elapsed().as_millis();
    let (claims_proof, inputs) = prove_claims(&witness, &prover_claims, &mut transcript);
    let claims_ms = start.elapsed().as_millis() - values_ms;
    let (proof, _) = prover_round(
        &crs,
        config,
        &commitment_with_aux,
        &witness,
        &inputs.evaluation_points_inner,
        &inputs.evaluation_points_outer,
        &mut prover_context,
        false,
        Some(transcript),
        None,
    );
    let prover_ms = start.elapsed().as_millis() - values_ms;

    let start = Instant::now();
    let mut transcript = HashWrapper::new();
    let (verifier_claims, _) = claims(&mut transcript, Some(values));
    let inputs = verify_claims(
        (config.witness_height, config.witness_width),
        &verifier_claims,
        &claims_proof,
        &mut transcript,
    );
    verifier_round(
        &verifier_crs,
        config,
        &rc_commitment,
        &proof,
        &inputs.evaluation_points_inner,
        &inputs.evaluation_points_outer,
        &inputs.claims,
        &mut verifier_context,
        Some(transcript),
        None,
    );
    let verifier_ms = start.elapsed().as_millis();

    println!("\n=== {shape:?}: N = 2^{} ===", shape.log_len());
    println!("claims proof {:.2} KB", to_kb(claims_proof.size_in_bits()));
    breakdown(&proof, 0);
    println!(
        "proof {:.2} KB (chain {:.2} KB), commit {commit_ms} ms, prover {prover_ms} ms (claims {claims_ms} ms; claim values {values_ms} ms), verifier {verifier_ms} ms",
        to_kb(claims_proof.size_in_bits() + proof.size_in_bits()),
        to_kb(proof.size_in_bits()),
    );

    let residues = n / 2;
    let lift = n / 32;
    println!(
        "outer ranks: T_Y (m = {residues}) {}, T_u (m = {lift}) {}",
        outer_rank(residues, 77.0 * ((residues * CHUNK) as f64).sqrt()),
        outer_rank(lift, ((lift * CHUNK) as f64).sqrt()),
    );

    #[cfg(feature = "rokoko-calibration")]
    rokoko::common::norms::calibration::print_table();
}

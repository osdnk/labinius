use bin_ntt::scheme::suite_from_args;
use bin_ntt::{Opening, OpeningMessage};
use bin_ntt::bd;
use bin_ntt::{Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];

pub fn run() {
    let rounds: u64 = std::env::var("ROUNDS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(200);
    let suite = suite_from_args();
    let params = Params::sized(
        suite,
        Opening::BitDropped {
            bits: suite.dropped_bits,
        },
    );
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let expected = bd::expected_normsq(params.columns(), params.dropped_bits());
    println!(
        "size {} witness 2^{} columns {} moduli {:?} dropped {} expected normsq {:.4e} cap {}",
        suite.name,
        params.witness_log_len,
        params.columns(),
        params.primes(),
        params.dropped_bits(),
        expected,
        params.bd_cap()
    );
    let mut ratios = Vec::with_capacity(rounds as usize);
    let mut rejected = 0usize;
    for round in 0..rounds {
        let mut seed = [0xC7u8; 32];
        seed[0] = round as u8;
        seed[1] = (round >> 8) as u8;
        let witness = Witness::random(&params, seed);
        let (commitment, opening) = prover.commit(&witness);
        let mut transcript = Transcript::new(b"bin-ntt/bdstats");
        transcript.absorb_u64(round);
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let claim = witness.mle_evaluate(&point);
        let row = witness.row_evaluate(&point);
        let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
        let folded = prover.fold(opening, &challenges);
        let residual = bd::residual(
            pp.key(),
            commitment.dropped(),
            challenges.challenges(),
            folded.elements(),
        )
        .expect("the residual of an honest round");
        let value = verifier.fold_row_evaluation(&row, &challenges);
        let ok = verifier
            .verify_opening(
                &commitment,
                &challenges,
                &point,
                OpeningMessage::BitDropped {
                    folded_witness: &folded,
                    folded_row_value: &value,
                },
            )
            .is_ok()
            && verifier.verify_evaluation(&point, &claim, &row).is_ok();
        if !ok {
            rejected += 1;
        }
        ratios.push(residual as f64 / expected);
    }
    ratios.sort_by(f64::total_cmp);
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    println!(
        "rounds {} rejected {} ratio mean {:.4} min {:.4} median {:.4} max {:.4}",
        rounds,
        rejected,
        mean,
        ratios[0],
        ratios[ratios.len() / 2],
        ratios[ratios.len() - 1]
    );
    println!(
        "cap multiplier for 2x headroom over the observed max: {:.2}",
        2.0 * ratios[ratios.len() - 1]
    );
}

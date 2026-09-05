//! Wire sizes and the median of 11 runs of each code at the basic shape.
use bin_ntt::wire;
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::hint::black_box;
use std::time::Instant;

const MATRIX_SEED: [u8; 32] = [21u8; 32];
const WITNESS_SEED: [u8; 32] = [23u8; 32];
const RUNS: usize = 11;

fn median_ms(mut run: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[RUNS / 2]
}

fn line(name: &str, bytes: usize, ms: f64, items: usize, unit: &str) {
    println!(
        "  {name:<24} {:>8.1} KB ({bytes:>7} bytes)  {ms:>7.3} ms  {:>6.2} ns per {unit}",
        bytes as f64 / 1024.0,
        ms * 1e6 / items as f64
    );
}

pub fn run() {
    let params = Params::new(18, 7, vec![Modulus::Q9721_FS_S], false).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/example/wire_bench");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded = prover.fold(opening, &challenges);

    let commitment_wire = wire::pack_commitment(&commitment);
    let row_wire = wire::pack_row_evaluation(&row_evaluation);
    let fold_wire = wire::encode(&folded, params.base.prime());
    let residues = 4 * params.columns() * params.primes().len() * 162;
    let coefficients = folded.len() * 648;
    println!(
        "basic: base {}, {} columns, {} residues of commitment, {} coefficients of fold, entropy {:.1} KB",
        params.base.prime(),
        params.columns(),
        residues,
        coefficients,
        wire::entropy_bytes(&folded) / 1024.0
    );

    assert_eq!(
        wire::unpack_commitment(&params, &commitment_wire).unwrap(),
        commitment
    );
    assert_eq!(
        wire::unpack_row_evaluation(&row_wire, params.columns())
            .unwrap()
            .values(),
        row_evaluation.values()
    );
    assert_eq!(
        wire::decode(&fold_wire).unwrap().elements(),
        folded.elements()
    );

    let ms = median_ms(|| {
        black_box(wire::unpack_commitment(&params, black_box(&commitment_wire)).unwrap());
    });
    line(
        "unpack_commitment",
        commitment_wire.len(),
        ms,
        residues,
        "residue",
    );
    let ms = median_ms(|| {
        black_box(wire::unpack_row_evaluation(black_box(&row_wire), params.columns()).unwrap());
    });
    line(
        "unpack_row_evaluation",
        row_wire.len(),
        ms,
        params.columns(),
        "element",
    );
    let ms = median_ms(|| {
        black_box(wire::decode(black_box(&fold_wire)).unwrap());
    });
    line("decode", fold_wire.len(), ms, coefficients, "coefficient");
    let ms = median_ms(|| {
        black_box(wire::encode(black_box(&folded), params.base.prime()));
    });
    line("encode", fold_wire.len(), ms, coefficients, "coefficient");
}

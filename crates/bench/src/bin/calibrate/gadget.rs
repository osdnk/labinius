//! Calibration: run one recursive round at `witness_log column_log base extra...` and let
//! GADGET_STATS print the chain magnitudes.
use bin_ntt::Opening;
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};

fn modulus(s: &str) -> Modulus {
    match s {
        "2917" => Modulus::Q2917_Q_S,
        "3889" => Modulus::Q3889_FS_S,
        "4861" => Modulus::Q4861_Q_S,
        "9721" => Modulus::Q9721_FS_S,
        "12637" => Modulus::Q12637_Q_S,
        "17497" => Modulus::Q17497_FS_L,
        "19441" => Modulus::Q19441_FS_L,
        _ => panic!("unknown modulus {s}"),
    }
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let witness_log: u32 = args[0].parse().unwrap();
    let column_log: u32 = args[1].parse().unwrap();
    let base = modulus(&args[2]);
    let seed: u8 = args[3].parse().unwrap();
    let extra: Vec<Modulus> = args[4..].iter().map(|s| modulus(s)).collect();
    let params =
        Params::with_base(witness_log, column_log, base, extra, Opening::Recursive).unwrap();
    eprintln!(
        "shape: n {} r {} primes {:?}",
        params.witness_len() / params.columns() / 4,
        params.columns(),
        params.primes()
    );
    let pp = PublicParameters::from_seed(params.clone(), [seed; 32]);
    let witness = Witness::random(&params, [seed ^ 0x95; 32]);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let (commitment, opening) = prover.commit(&witness);
    let mut t = Transcript::new(&[b"bin-ntt/gadget".as_slice(), &[seed]].concat());
    let point = verifier.derive_evaluation_point(&mut t, &commitment);
    let claim = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let left = prover.commit_left_expansion(&row);
    let challenges = verifier.derive_folding_challenges(&mut t, &left);
    let proof = prover
        .prove_opening(
            &mut t,
            opening,
            &challenges,
            &point,
            &left,
            &row,
            &claim,
            &commitment,
        )
        .expect("within cap");
    eprintln!("proof {:.1} KB", proof.wire_bytes() as f64 / 1024.0);
}

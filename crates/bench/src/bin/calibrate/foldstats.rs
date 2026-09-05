use bin_ntt::api::Modulus;
use bin_ntt::{Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];

pub fn run() {
    let wl: u32 = std::env::var("WL").map(|s| s.parse().unwrap()).unwrap_or(24);
    let cl: u32 = std::env::var("CL").map(|s| s.parse().unwrap()).unwrap_or(11);
    let bp: u16 = std::env::var("BP").map(|s| s.parse().unwrap()).unwrap_or(19441);
    let rounds: u64 = std::env::var("ROUNDS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    let base = Modulus::from_prime(bp).unwrap();
    let params = Params::with_base(wl, cl, base, vec![], false).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let n = params.witness_len() / params.columns() / 4;
    let r = params.columns();
    println!(
        "wl {wl} cl {cl} base {bp} n {n} r {r} half {}",
        (bp as i64 - 1) / 2
    );
    println!("predicted sd {:.1}", (bin_ntt::recursion::FOLD_CAP * r as f64).sqrt());
    for round in 0..rounds {
        let mut seed = [0xC7u8; 32];
        seed[0] = round as u8;
        let witness = Witness::random(&params, seed);
        let mut prover = Prover::new(&pp);
        let verifier = Verifier::new(&pp);
        let t0 = std::time::Instant::now();
        let (commitment, opening) = prover.commit(&witness);
        let commit_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut transcript = Transcript::new(b"bin-ntt/foldstats");
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let row = witness.row_evaluate(&point);
        let ch = verifier.derive_folding_challenges(&mut transcript, &row);
        let t1 = std::time::Instant::now();
        let folded = prover.fold(opening, &ch);
        let fold_ms = t1.elapsed().as_secs_f64() * 1e3;
        println!("  commit {commit_ms:.0} ms fold {fold_ms:.0} ms");
        let mut max = 0i64;
        let mut sq = 0u64;
        let mut count = 0usize;
        let mut hist = [0usize; 20];
        for e in folded.elements() {
            for &x in e.v.iter() {
                let a = (x as i64).abs();
                if a > max {
                    max = a;
                }
                sq += (x as i64 * x as i64) as u64;
                count += 1;
                let b = (a as usize * 20) / ((bp as usize + 1) / 2);
                hist[b.min(19)] += 1;
            }
        }
        let sd = (sq as f64 / count as f64).sqrt();
        println!(
            "round {round}: coeffs {count} sd {sd:.2} max {max} ({:.2} sd) normsq {sq} fill {:.3}",
            max as f64 / sd,
            sq as f64 / (bin_ntt::recursion::FOLD_CAP * (n * 648 * r) as f64)
        );
        println!("  tail {:?}", &hist[12..]);
    }
}

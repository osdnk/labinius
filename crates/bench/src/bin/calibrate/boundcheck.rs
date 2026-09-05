use bin_ntt::Opening;
use bin_ntt::api::Modulus;
use bin_ntt::challenge::{sample_short_challenge, DEFAULT_BOUND, DEFAULT_WEIGHT};
use bin_ntt::fields::scalar::F162;
use bin_ntt::recursion::Instance;
use bin_ntt::scheme::{EvaluationPoint, FoldingChallenges};
use bin_ntt::{Params, PublicParameters, Transcript};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];

pub fn run() {
    let bp: u16 = std::env::var("BP")
        .map(|s| s.parse().unwrap())
        .unwrap_or(3889);
    let extras: Vec<Modulus> = std::env::var("EX")
        .unwrap_or_else(|_| "2917,4861".into())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| Modulus::from_prime(s.parse().unwrap()).unwrap())
        .collect();
    let wl: u32 = std::env::var("WL")
        .map(|s| s.parse().unwrap())
        .unwrap_or(24);
    let cl: u32 = std::env::var("CL")
        .map(|s| s.parse().unwrap())
        .unwrap_or(11);
    let params =
        Params::with_base(wl, cl, Modulus::from_prime(bp).unwrap(), extras, Opening::Recursive)
            .unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let setup = pp.recursion().expect("recursion is on").clone();
    let mut t = Transcript::new(b"bin-ntt/boundcheck");
    let challenges = FoldingChallenges::of(
        (0..params.columns())
            .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
            .collect(),
    );
    let point = EvaluationPoint::of(
        vec![F162::ZERO; params.row_log_len() as usize],
        vec![F162::ZERO; params.column_log_len as usize],
    );
    let layout = Instance::layout(&setup, &challenges, &point, &F162::ZERO);
    println!(
        "base {bp} wl {wl} cl {cl} n {} r {} clears {}",
        setup.n,
        setup.r,
        layout.clears()
    );
    let mut b = layout.bound();
    b.sort_by(|x, y| x.margin().total_cmp(&y.margin()));
    for c in b.iter().take(6) {
        println!(
            "  {:<26} margin {:>9.2} worst {} share {:.2}",
            c.name,
            c.margin(),
            c.worst,
            c.share
        );
    }
    println!("  ranks {:?}", setup.ranks);
}

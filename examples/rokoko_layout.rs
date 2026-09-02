use bin_ntt::rokoko::relation;
use bin_ntt::{Backend, Modulus, Params, PublicParameters};
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (w, c): (u32, u32) = (a[0].parse().unwrap(), a[1].parse().unwrap());
    let base = if a[2] == "L" {
        Modulus::Q17497_FS_L
    } else {
        Modulus::Q3889_FS_S
    };
    let extra = if a[2] == "L" {
        Modulus::Q19441_FS_L
    } else {
        Modulus::Q9721_FS_S
    };
    let params = Params::with_base(w, c, base, vec![extra], true)
        .unwrap()
        .with_backend(Backend::Labrador);
    let pp = PublicParameters::from_seed(
        Params {
            recursion: false,
            ..params.clone()
        },
        [1u8; 32],
    );
    let setup = relation::Setup::new(&pp);
    let quiet = bin_ntt::FoldingChallenges::of(vec![
        bin_ntt::challenge::ShortChallenge::from_coeffs(
            &[0i8; 162]
        );
        params.columns()
    ]);
    let origin = bin_ntt::EvaluationPoint::of(
        vec![bin_ntt::F162::ZERO; params.row_log_len() as usize],
        vec![bin_ntt::F162::ZERO; params.column_log_len as usize],
    );
    let layout = relation::layout(&setup, &quiet, &origin, &bin_ntt::F162::ZERO).layout;
    println!(
        "n {} r {} total {}",
        params.witness_len() / params.columns() / 4,
        params.columns(),
        layout.len
    );
    for (v, r) in layout.vectors.iter().zip(&layout.regions) {
        println!(
            "{:<12} used {:>6} len {:>6} start {:>7}",
            v.name, v.used, r.len, r.start
        );
    }
}

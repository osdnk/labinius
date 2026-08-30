//! `simd::vertical_gen_large` against `scalar::ntt` / `scalar::intt`, against its own declared
//! bounds and against the `const` recursions that prove them.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::vertical_gen_large as vgl;
use bin_ntt::types::{Batch32, Representation};

/// Adversarial coefficient batches at the declared input bound `|x| <= q`, then random ones.
fn inputs<const Q: u16>(count: usize, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let q = Q as i16;
    let mut out = Vec::new();
    for &f in &[q, -q] {
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = f;
            }
        }
        out.push(b);
    }
    let mut b = Batch32::zero(Representation::Coefficients);
    for j in 0..N {
        for p in 0..32 {
            b.v[j][p] = if (j + p) % 2 == 0 { q } else { -q };
        }
    }
    out.push(b);
    for _ in 0..count {
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = rng.below(2 * Q as u32 + 1) as i16 - q;
            }
        }
        out.push(b);
    }
    out
}

fn forward<const Q: u16>() {
    let bound = vgl::output_bound(Q);
    let mut worst = 0i32;
    for mut b in inputs::<Q>(8, 0x6EA1 ^ Q as u64) {
        let coeffs: Vec<[u32; N]> = (0..32).map(|p| b.get(p).normalized(Q)).collect();
        unsafe { vgl::ntt_gen_batch32::<Q>(&mut b) };
        assert_eq!(b.representation, Representation::Ntt);
        for p in 0..32 {
            let want = scalar::ntt::<Q>(&coeffs[p]);
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert_eq!(got.rem_euclid(Q as i32) as u32, want[j], "q={Q} slot {j} lane {p}");
                worst = worst.max(got.abs());
                assert!(got.abs() <= bound, "q={Q} |{got}| over the declared bound {bound}");
            }
        }
    }
    let (lm, peak) = vgl::fwd_model(Q);
    println!(
        "q={Q} forward: max |output| {worst} = {:.3} q (declared {:.3} q), \
         model {lm:?} peak {peak}",
        worst as f64 / Q as f64,
        bound as f64 / Q as f64
    );
}

#[test]
fn forward_matches_scalar() {
    forward::<17497>();
    forward::<19441>();
}

/// The inverse is fed exactly what `recursion::limbs::columns_split` feeds it: a fully reduced
/// centered transform.
fn inverse<const Q: u16>() {
    let mut rng = Rng::new(0x1177 ^ Q as u64);
    let half = ((Q - 1) / 2) as i32;
    for _ in 0..6 {
        let coeffs: Vec<[u32; N]> =
            (0..32).map(|_| core::array::from_fn(|_| rng.below(Q as u32))).collect();
        let mut b = Batch32::zero(Representation::Ntt);
        for p in 0..32 {
            let s = scalar::ntt::<Q>(&coeffs[p]);
            for j in 0..N {
                let x = s[j] as i32;
                b.v[j][p] = if x > half { (x - Q as i32) as i16 } else { x as i16 };
            }
        }
        unsafe { vgl::intt_gen_batch32::<Q>(&mut b) };
        assert_eq!(b.representation, Representation::Coefficients);
        for p in 0..32 {
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert!(got.abs() <= half, "q={Q} slot {j} lane {p}: |{got}| not centered");
                assert_eq!(
                    got.rem_euclid(Q as i32) as u32,
                    coeffs[p][j],
                    "q={Q} coefficient {j} lane {p}"
                );
            }
        }
    }
    let (lm, peak) = vgl::inv_model(Q);
    println!("q={Q} inverse: model {lm:?} peak {peak} of 32767");
}

#[test]
fn inverse_matches_scalar() {
    inverse::<17497>();
    inverse::<19441>();
}

#[test]
fn round_trip() {
    fn go<const Q: u16>() {
        let mut rng = Rng::new(0x2A2A ^ Q as u64);
        let half = ((Q - 1) / 2) as i16;
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = rng.below(Q as u32) as i16 - half;
            }
        }
        let want = b.clone();
        unsafe {
            vgl::ntt_gen_batch32::<Q>(&mut b);
            vgl::intt_gen_batch32::<Q>(&mut b);
        }
        assert_eq!(b.v, want.v, "q={Q}");
    }
    go::<17497>();
    go::<19441>();
}

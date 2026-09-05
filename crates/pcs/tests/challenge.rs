//! Short challenges: determinism of the transcript, the shape of a challenge, the sign map, the
//! canonical embedding against a naive reference, the rejection bound, and the acceptance
//! statistics.
use bin_ntt::ring::N162;
use bin_ntt::challenge::{
    canonical_inf_norm_sq, canonical_inf_norm_sq_naive, sample_attempt, sample_short_challenge,
    ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT, MAX_WEIGHT,
};
use std::time::Instant;

fn transcript(tag: u64) -> Transcript {
    let mut t = Transcript::new(b"bin-ntt/tests/challenge");
    t.absorb_u64(tag);
    t
}

#[test]
fn max_weight_and_default() {
    assert!(MAX_WEIGHT >= 32);
    assert!(DEFAULT_WEIGHT <= MAX_WEIGHT);
    assert_eq!(DEFAULT_WEIGHT, 28);
    assert_eq!(DEFAULT_BOUND, 12.0);
}

#[test]
fn deterministic() {
    let (c1, a1) = sample_short_challenge(&mut transcript(7), DEFAULT_WEIGHT, DEFAULT_BOUND);
    let (c2, a2) = sample_short_challenge(&mut transcript(7), DEFAULT_WEIGHT, DEFAULT_BOUND);
    assert_eq!(c1, c2);
    assert_eq!(a1, a2);

    let (c3, _) = sample_short_challenge(&mut transcript(8), DEFAULT_WEIGHT, DEFAULT_BOUND);
    assert_ne!(c1, c3);
}

#[test]
fn absorbed_data_separates() {
    let mut a = Transcript::new(b"dom");
    let mut b = Transcript::new(b"dom");
    a.absorb_bytes(b"hello");
    b.absorb_bytes(b"world");
    assert_ne!(sample_attempt(&mut a, 30), sample_attempt(&mut b, 30));

    let mut c = Transcript::new(b"dom");
    let mut d = Transcript::new(b"other");
    assert_ne!(sample_attempt(&mut c, 30), sample_attempt(&mut d, 30));

    // Successive samples from one transcript are independent (the sample counter).
    let mut e = Transcript::new(b"dom");
    assert_ne!(sample_attempt(&mut e, 30), sample_attempt(&mut e, 30));

    // Length prefixing: "ab" + "c" must not collide with "a" + "bc".
    let mut f = Transcript::new(b"dom");
    f.absorb_bytes(b"ab");
    f.absorb_bytes(b"c");
    let mut g = Transcript::new(b"dom");
    g.absorb_bytes(b"a");
    g.absorb_bytes(b"bc");
    assert_ne!(sample_attempt(&mut f, 30), sample_attempt(&mut g, 30));
}

#[test]
fn absorb_elements_binds() {
    use bin_ntt::ring::PowerOfThreeRingElementWithLimbs;
    let mut x = PowerOfThreeRingElementWithLimbs::zero(2);
    let mut y = PowerOfThreeRingElementWithLimbs::zero(2);
    y.limbs[1].v[161] = -3;
    let mut a = Transcript::new(b"dom");
    let mut b = Transcript::new(b"dom");
    a.absorb_elements(&[x.clone()]);
    b.absorb_elements(&[y.clone()]);
    assert_ne!(sample_attempt(&mut a, 30), sample_attempt(&mut b, 30));

    x.limbs[0].v[0] = 1;
    let mut c = Transcript::new(b"dom");
    let mut d = Transcript::new(b"dom");
    c.absorb_elements(&[x.clone(), y.clone()]);
    d.absorb_elements(&[y, x]);
    assert_ne!(sample_attempt(&mut c, 30), sample_attempt(&mut d, 30));
}

#[test]
fn shape() {
    let mut t = transcript(1);
    for weight in [1usize, 5, 28, 30, MAX_WEIGHT] {
        for _ in 0..200 {
            let c = sample_attempt(&mut t, weight);
            assert_eq!(c.weight, weight);
            for i in 0..weight {
                assert!((c.positions[i] as usize) < N162);
            }
            for i in 1..weight {
                assert!(c.positions[i - 1] < c.positions[i], "not sorted / distinct");
            }
            let d = c.coeffs();
            assert_eq!(d.iter().filter(|&&x| x != 0).count(), weight);
            assert!(
                d.iter().all(|&x| x == 0 || x == 1),
                "a challenge is not binary"
            );
            assert_eq!(ShortChallenge::from_coeffs(&d), c);
        }
    }
}

#[test]
fn positions_are_uniform() {
    // Every position must be reachable and roughly equally likely (weight/162 each).
    let mut t = transcript(2);
    let mut hits = [0u32; N162];
    let n = 20_000;
    for _ in 0..n {
        let c = sample_attempt(&mut t, DEFAULT_WEIGHT);
        for i in 0..c.weight {
            hits[c.positions[i] as usize] += 1;
        }
    }
    let expect = n as f64 * DEFAULT_WEIGHT as f64 / N162 as f64;
    let (lo, hi) = (
        *hits.iter().min().unwrap() as f64,
        *hits.iter().max().unwrap() as f64,
    );
    println!("position counts: expected {expect:.0}, min {lo:.0}, max {hi:.0}");
    assert!(lo > 0.85 * expect && hi < 1.15 * expect, "positions skewed");
}

#[test]
fn canonical_norm_matches_naive() {
    let mut t = transcript(3);
    let mut worst: f64 = 0.0;
    for weight in [0usize, 1, 2, 7, 21, 30, MAX_WEIGHT] {
        for _ in 0..300 {
            let c = sample_attempt(&mut t, weight).signed();
            let (a, b) = (canonical_inf_norm_sq(&c), canonical_inf_norm_sq_naive(&c));
            worst = worst.max((a - b).abs());
            assert!((a - b).abs() <= 1e-9, "{a} vs {b} at weight {weight}");
        }
    }
    println!("canonical_inf_norm_sq vs naive: worst |difference| {worst:.3e}");
}

#[test]
fn canonical_norm_of_monomials() {
    // A single Z^p has |c(zeta^u)| = 1 everywhere.
    for p in 0..N162 {
        let mut d = [0i8; N162];
        d[p] = 1;
        let c = ShortChallenge::from_coeffs(&d);
        assert!((canonical_inf_norm_sq(&c) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn sampled_challenges_meet_the_bound() {
    let mut t = transcript(4);
    for _ in 0..300 {
        let (c, _) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
        assert!(canonical_inf_norm_sq(&c) <= DEFAULT_BOUND * DEFAULT_BOUND + 1e-12);
        assert_eq!(c.weight, DEFAULT_WEIGHT);
    }
}

/// What the rest of the crate is entitled to assume of a folding challenge: exactly
/// `DEFAULT_WEIGHT` coefficients, all of them `+-1`, and a canonical norm inside `DEFAULT_BOUND`
/// once signed. The expansion factor a security argument uses is `sqrt(3)` times that bound,
/// because the power basis of a power-of-three conductor is not orthogonal under the canonical
/// embedding.
#[test]
fn the_default_challenge_is_signed_and_short() {
    let mut t = transcript(9);
    let mut worst = 0.0f64;
    let (mut plus, mut minus) = (0usize, 0usize);
    for _ in 0..500 {
        let (c, _) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
        let d = c.coeffs();
        assert!(
            d.iter().all(|&x| x == 0 || x == 1 || x == -1),
            "a challenge coefficient is not 0, 1 or -1"
        );
        assert_eq!(d.iter().filter(|&&x| x != 0).count(), DEFAULT_WEIGHT);
        assert_eq!(ShortChallenge::from_coeffs(&d), c);
        plus += d.iter().filter(|&&x| x == 1).count();
        minus += d.iter().filter(|&&x| x == -1).count();
        let norm = canonical_inf_norm_sq(&c);
        assert!(norm <= DEFAULT_BOUND * DEFAULT_BOUND + 1e-12);
        assert!((canonical_inf_norm_sq_naive(&c) - norm).abs() <= 1e-9);
        worst = worst.max(norm);
    }
    let n = (plus + minus) as f64;
    assert!((plus as f64 / n - 0.5).abs() < 0.05, "the signs are skewed");
    println!(
        "weight {DEFAULT_WEIGHT} signed ({:.3} of the coefficients are +1), worst canonical norm \
         {:.3} of {DEFAULT_BOUND} (expansion sqrt(3) * {DEFAULT_BOUND} = {:.2})",
        plus as f64 / n,
        worst.sqrt(),
        3f64.sqrt() * DEFAULT_BOUND
    );
}

#[test]
fn signs_are_a_function_of_the_positions() {
    let mut t = transcript(10);
    let mut seen = 0u32;
    for _ in 0..500 {
        let (c, _) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
        let bare = ShortChallenge { signs: 0, ..c };
        assert_eq!(bare.signed(), c);
        assert_eq!(c.signed(), c);
        assert_eq!(bare.positions, c.positions);
        seen |= c.signs;
    }
    assert_eq!(
        seen,
        (1u32 << DEFAULT_WEIGHT) - 1,
        "some sign bit is never set"
    );
}

#[test]
fn cardinality() {
    let bits = ShortChallenge::log2_cardinality(DEFAULT_WEIGHT);
    println!("log2 |challenge set| at weight {DEFAULT_WEIGHT}: {bits:.2}");
    assert!(bits >= 100.0);
    assert!((ShortChallenge::log2_cardinality(1) - 162f64.log2()).abs() < 1e-9);
    assert!((ShortChallenge::log2_cardinality(0)).abs() < 1e-12);
    for w in [10usize, 20, 28, 30, 40] {
        println!(
            "  weight {w:2}: {:.2} bits",
            ShortChallenge::log2_cardinality(w)
        );
    }
}

#[test]
fn acceptance_statistics() {
    let samples = 2000;
    // Cost of one attempt: sampling plus the canonical norm.
    let mut t = transcript(5);
    let mut sink = 0.0f64;
    let n_att = 20_000;
    let t0 = Instant::now();
    for _ in 0..n_att {
        let c = sample_attempt(&mut t, DEFAULT_WEIGHT);
        sink += canonical_inf_norm_sq(&c);
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n_att as f64;
    println!("isolated attempt (own XOF derivation + full canonical_inf_norm_sq): {us:.3} us  [{sink:.0}]");

    for bound in [9.0f64, 10.0, 11.0, 12.0, 15.0] {
        let mut t = transcript(6);
        let mut total = 0u64;
        let mut worst = 0u64;
        let t0 = Instant::now();
        for _ in 0..samples {
            let (_, a) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, bound);
            total += a;
            worst = worst.max(a);
        }
        let mean = total as f64 / samples as f64;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "weight {DEFAULT_WEIGHT}, bound {bound:.1}: {mean:.2} attempts per challenge \
             (acceptance {:.3} %, worst {worst}), {:.1} us per challenge, {:.3} us per attempt",
            100.0 / mean,
            ms * 1e3 / samples as f64,
            ms * 1e3 / total as f64
        );
        if bound == 15.0 {
            assert!(mean < 2.0, "bound 15 rejects too often: {mean}");
        }
    }
}

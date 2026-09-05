//! The ring constants and the exact scalar reference the SIMD kernels are tested against.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;

/// A random binary polynomial as its 648 coefficients (the form the kernels' inputs lift to).
fn random_bin(rng: &mut Rng) -> [u32; N] {
    let mut c = [0u32; N];
    for w in 0..N.div_ceil(64) {
        let x = rng.next_u64();
        for b in 0..64 {
            if 64 * w + b < N {
                c[64 * w + b] = ((x >> b) & 1) as u32;
            }
        }
    }
    c
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[test]
fn slot_exponents_are_the_units_mod_1944() {
    let mut seen = vec![false; 1944];
    for j in 0..N {
        let u = SLOT_EXP[j] as u32;
        assert_eq!(gcd(u, 1944), 1, "slot {j}: exponent {u} not a unit");
        assert!(!seen[u as usize], "duplicate exponent {u}");
        seen[u as usize] = true;
    }
}

fn check_prime<const Q: u16>() {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    assert_eq!(pow_mod(psi, 1944, q), 1);
    assert_ne!(pow_mod(psi, 972, q), 1);
    assert_ne!(pow_mod(psi, 648, q), 1);
    let w = Params::<Q>::OMEGA as u64;
    assert_eq!((w * w + w + 1) % q, 0, "omega^2 + omega + 1 = 0");
    let z6 = Params::<Q>::ZETA6 as u64;
    assert_eq!((z6 * z6 + q - z6 + 1) % q, 0, "zeta6^2 - zeta6 + 1 = 0");
    assert_eq!((Params::<Q>::QINV as u32 * Q as u32) & 0xffff, 1);
    // Montgomery helpers agree with exact arithmetic for all lanes values and a few constants.
    let mut rng = Rng::new(7);
    for _ in 0..2000 {
        let a = rng.next_u64() as u16 as i16;
        let x = rng.below(Q as u32) as u16;
        let wm = Params::<Q>::to_mont(x);
        let wp = Params::<Q>::mont_pre(wm);
        let r = mont_mul_i16(a, wm, wp, Q);
        assert!((r as i32).abs() < Q as i32);
        let want = ((a as i64).rem_euclid(q as i64) * x as i64) % q as i64;
        assert_eq!(
            (r as i64).rem_euclid(q as i64),
            want,
            "mont_mul a={a} x={x}"
        );
        let b = barrett_i16(a, Q);
        assert_eq!(
            (b as i64).rem_euclid(q as i64),
            (a as i64).rem_euclid(q as i64)
        );
        assert!((b as i32).abs() < Q as i32, "barrett |r|={} a={a}", b);
        let b = red16_i16(a, Q);
        assert_eq!(
            (b as i64).rem_euclid(q as i64),
            (a as i64).rem_euclid(q as i64)
        );
        assert!(b >= 0 && b <= Q as i16, "red16 r={} a={a}", b);
    }
    // NTT = direct evaluation at psi^SLOT_EXP[j], and it is a ring homomorphism.
    let a = random_bin(&mut rng);
    let b = random_bin(&mut rng);
    let na = scalar::ntt::<Q>(&a);
    for j in (0..N).step_by(37) {
        assert_eq!(
            na[j],
            scalar::eval_at::<Q>(&a, SLOT_EXP[j] as u32),
            "slot {j}"
        );
    }
    let nb = scalar::ntt::<Q>(&b);
    let ab = scalar::mul_mod_phi(&a, &b, Q);
    assert_eq!(scalar::ntt::<Q>(&ab), scalar::pointwise_mul(&na, &nb, Q));
    // x^648 = x^324 - 1 in the ring: NTT of X^648 must equal NTT of X^324 - 1.
    let mut x324 = [0u32; N];
    x324[324] = 1;
    x324[0] = Q as u32 - 1;
    let mut x = [0u32; N];
    x[1] = 1;
    let mut x648 = [0u32; N];
    x648[0] = 1;
    for _ in 0..648 {
        x648 = scalar::mul_mod_phi(&x648, &x, Q);
    }
    assert_eq!(x648, x324);
}

#[test]
fn prime_3889() {
    check_prime::<3889>();
}

#[test]
fn prime_9721() {
    check_prime::<9721>();
}

#[test]
fn prime_17497() {
    check_prime::<17497>();
}

#[test]
fn prime_19441() {
    check_prime::<19441>();
}

/// `scalar::intt` is the exact inverse of `scalar::ntt`.
fn check_intt<const Q: u16>() {
    let mut rng = Rng::new(0x1177 ^ Q as u64);
    let mut cases: Vec<[u32; N]> = Vec::new();
    cases.push([0u32; N]);
    cases.push([1u32; N]);
    let mut mono = [0u32; N];
    mono[324] = Q as u32 - 1;
    cases.push(mono);
    for _ in 0..8 {
        cases.push(std::array::from_fn(|_| rng.below(Q as u32)));
    }
    for _ in 0..4 {
        cases.push(random_bin(&mut rng));
    }
    for a in &cases {
        let n = scalar::ntt::<Q>(a);
        assert_eq!(&scalar::intt::<Q>(&n), a, "q={Q}: intt(ntt(a)) != a");
    }
    // the round trip through a product: intt of the slot product is a*b.
    let (a, b) = (random_bin(&mut rng), random_bin(&mut rng));
    let (na, nb) = (scalar::ntt::<Q>(&a), scalar::ntt::<Q>(&b));
    assert_eq!(
        scalar::intt::<Q>(&scalar::pointwise_mul(&na, &nb, Q)),
        scalar::mul_mod_phi(&a, &b, Q),
        "q={Q}: product"
    );
}

#[test]
fn intt_3889() {
    check_intt::<3889>();
}

#[test]
fn intt_9721() {
    check_intt::<9721>();
}

#[test]
fn intt_17497() {
    check_intt::<17497>();
}

#[test]
fn intt_19441() {
    check_intt::<19441>();
}

use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::types::*;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
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
        assert_eq!((r as i64).rem_euclid(q as i64), want, "mont_mul a={a} x={x}");
        let b = barrett_i16(a, Q);
        assert_eq!((b as i64).rem_euclid(q as i64), (a as i64).rem_euclid(q as i64));
        assert!((b as i32).abs() < Q as i32, "barrett |r|={} a={a}", b);
        let b = red16_i16(a, Q);
        assert_eq!((b as i64).rem_euclid(q as i64), (a as i64).rem_euclid(q as i64));
        assert!(b >= 0 && b <= Q as i16, "red16 r={} a={a}", b);
    }
    // NTT = direct evaluation at psi^SLOT_EXP[j], and it is a ring homomorphism.
    let pa = BinaryPoly::random(&mut rng);
    let pb = BinaryPoly::random(&mut rng);
    let a = scalar::lift(&pa);
    let b = scalar::lift(&pb);
    let na = scalar::ntt::<Q>(&a);
    for j in (0..N).step_by(37) {
        assert_eq!(na[j], scalar::eval_at::<Q>(&a, SLOT_EXP[j] as u32), "slot {j}");
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
fn binary_batch_roundtrip() {
    let mut rng = Rng::new(3);
    let polys: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let b = BinaryBatch32::from_polys_scalar(&polys);
    for p in 0..32 {
        assert_eq!(b.poly(p), polys[p]);
    }
    let batch = Batch32::from_binary(&polys);
    assert_eq!(batch.get(5), RingElement::from_binary(&polys[5]));
}

/// `scalar::intt` is the exact inverse of `scalar::ntt`, and `intt_mont` additionally undoes the
/// Montgomery factor the `*_mont` kernels leave on their output.
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
        let p = BinaryPoly::random(&mut rng);
        cases.push(scalar::lift(&p));
    }
    let r = Params::<Q>::R as u64;
    for a in &cases {
        let n = scalar::ntt::<Q>(a);
        assert_eq!(&scalar::intt::<Q>(&n), a, "q={Q}: intt(ntt(a)) != a");
        // Montgomery-form transform: R * ntt(a).
        let nm: [u32; N] =
            std::array::from_fn(|j| (n[j] as u64 * r % Q as u64) as u32);
        assert_eq!(&scalar::intt_mont::<Q>(&nm), a, "q={Q}: intt_mont(R ntt(a)) != a");
        assert_eq!(&scalar::intt_mont_pow::<Q>(&nm, 1), a, "q={Q}: intt_mont_pow k=1");
        let nm2: [u32; N] =
            std::array::from_fn(|j| (nm[j] as u64 * r % Q as u64) as u32);
        assert_eq!(&scalar::intt_mont_pow::<Q>(&nm2, 2), a, "q={Q}: intt_mont_pow k=2");
    }
    // the round trip through a product: intt_mont of the Montgomery-form slot product is a*b.
    let pa = BinaryPoly::random(&mut rng);
    let pb = BinaryPoly::random(&mut rng);
    let (a, b) = (scalar::lift(&pa), scalar::lift(&pb));
    let na: [u32; N] = std::array::from_fn(|j| (scalar::ntt::<Q>(&a)[j] as u64 * r % Q as u64) as u32);
    let nb: [u32; N] = std::array::from_fn(|j| (scalar::ntt::<Q>(&b)[j] as u64 * r % Q as u64) as u32);
    // mont(R a, R b) = R a b
    let rinv = Params::<Q>::RINV as u64;
    let prod: [u32; N] =
        std::array::from_fn(|j| (na[j] as u64 * nb[j] as u64 % Q as u64 * rinv % Q as u64) as u32);
    assert_eq!(scalar::intt_mont::<Q>(&prod), scalar::mul_mod_phi(&a, &b, Q), "q={Q}: product");
}

#[test]
fn intt_3889() {
    check_intt::<3889>();
}

#[test]
fn intt_9721() {
    check_intt::<9721>();
}

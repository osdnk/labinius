//! Correctness and bound tests for `simd::vertical_gen`.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::pointwise::{self, MontElement};
use bin_ntt::simd::vertical_gen::{
    ntt_gen_batch32, ntt_gen_batch32_plan, ntt_gen_batch32_r27, ntt_gen_batches, Tw,
};
use bin_ntt::types::*;

/// Exact i32 mirror of the kernel: same operation order, same Barrett placement, but every value
/// kept as i32 so that an i16 overflow is observable. Returns the output and the per-level maximum
/// absolute value (index l = after level l).
fn shadow<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 7]) -> [i32; N] {
    let q = Q as u64;
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        assert!(v[i].abs() <= Q as i32, "input bound");
    }
    let mont = |a: i32, x: u16| -> i32 {
        assert!(a.abs() < 32768, "i16 overflow feeding a multiplication: {a}");
        let w = Params::<Q>::to_mont(x);
        params_mont(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
    };
    // level 0: Phi_6 split, with the pass-A Barrett on the a0 half of the second child.
    let z6 = Params::<Q>::ZETA6;
    for i in 0..324 {
        let (a0, a1) = (v[i], v[i + 324]);
        let t = mont(a1, z6);
        v[i] = a0 + t;
        let mut x = a0 + a1 - t;
        if Tw::<Q>::BAR_A && i < 162 {
            assert!(x.abs() < 32768, "i16 overflow before barrett: {x}");
            x = barrett_i16(x as i16, Q) as i32;
        }
        v[i + 324] = x;
    }
    lmax[0] = v.iter().map(|x| x.abs()).max().unwrap();
    let w1 = Params::<Q>::OMEGA;
    for level in 1..=6 {
        let n = DEGREE[level];
        let p = RADIX[level];
        let m = n / p;
        for k in 0..SUBRINGS[level] {
            let base = k * n;
            let zeta = pow_mod(Params::<Q>::PSI as u64, twiddle_exp(level, k) as u64, q) as u16;
            let zeta2 = (zeta as u64 * zeta as u64 % q) as u16;
            for i in 0..m {
                let mut a0 = v[base + i];
                if Tw::<Q>::BAR_L[level] {
                    assert!(a0.abs() < 32768, "i16 overflow before barrett: {a0}");
                    a0 = barrett_i16(a0 as i16, Q) as i32;
                }
                if p == 2 {
                    let t = mont(v[base + m + i], zeta);
                    v[base + i] = a0 + t;
                    v[base + m + i] = a0 - t;
                } else {
                    let t1 = mont(v[base + m + i], zeta);
                    let t2 = mont(v[base + 2 * m + i], zeta2);
                    let u = mont(t1 - t2, w1);
                    v[base + i] = a0 + t1 + t2;
                    v[base + m + i] = a0 - t2 + u;
                    v[base + 2 * m + i] = a0 - t1 - u;
                }
            }
        }
        lmax[level] = v.iter().map(|x| x.abs()).max().unwrap();
    }
    v
}

fn params_mont(a: i16, w: i16, wp: i16, q: u16) -> i16 {
    mont_mul_i16(a, w, wp, q)
}

fn to_batch(cols: &[[i16; N]; 32]) -> Batch32 {
    let mut b = Batch32::zero(Representation::Coefficients);
    for p in 0..32 {
        for j in 0..N {
            b.v[j][p] = cols[p][j];
        }
    }
    b
}

fn check<const Q: u16>(cols: &[[i16; N]; 32], what: &str) {
    let mut b = to_batch(cols);
    unsafe { ntt_gen_batch32::<Q>(&mut b) };
    assert_eq!(b.representation, Representation::Ntt);
    let bound = Tw::<Q>::OUTPUT_BOUND;
    for p in 0..32 {
        let mut lmax = [0i32; 7];
        let want_shadow = shadow::<Q>(&cols[p], &mut lmax);
        let mut coeffs = [0u32; N];
        for j in 0..N {
            coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
        }
        let want = scalar::ntt::<Q>(&coeffs);
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(got, want_shadow[j], "{what} q={Q} poly {p} slot {j}: shadow mismatch");
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} slot {j}"
            );
            assert!(got.abs() <= bound, "{what} q={Q} poly {p} slot {j}: |{got}| > {bound}");
        }
    }
}

fn random_cols<const Q: u16>(rng: &mut Rng) -> [[i16; N]; 32] {
    std::array::from_fn(|_| {
        std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16)
    })
}

fn binary_cols(rng: &mut Rng) -> [[i16; N]; 32] {
    let polys: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(rng));
    std::array::from_fn(|p| std::array::from_fn(|j| polys[p].coeff(j) as i16))
}

fn adversarial<const Q: u16>() -> Vec<[[i16; N]; 32]> {
    let q = Q as i16;
    let mut out = Vec::new();
    out.push([[0i16; N]; 32]);
    out.push([[q; N]; 32]);
    out.push([[-q; N]; 32]);
    out.push(std::array::from_fn(|_| std::array::from_fn(|j| if j % 2 == 0 { q } else { -q })));
    out.push(std::array::from_fn(|p| {
        std::array::from_fn(|j| if (j + p) % 2 == 0 { q } else { -q })
    }));
    for &m in &[0usize, 161, 162, 323, 324, 647] {
        for &val in &[1i16, q, -q] {
            let mut c = [[0i16; N]; 32];
            for p in 0..32 {
                c[p][m] = val;
            }
            out.push(c);
        }
    }
    // one monomial per polynomial, all different positions
    let mut c = [[0i16; N]; 32];
    for p in 0..32 {
        c[p][(p * 21) % N] = q;
    }
    out.push(c);
    out
}

fn run<const Q: u16>() {
    let mut rng = Rng::new(0x5eed ^ Q as u64);
    for (i, c) in adversarial::<Q>().iter().enumerate() {
        check::<Q>(c, &format!("adversarial#{i}"));
    }
    for i in 0..32 {
        check::<Q>(&binary_cols(&mut rng), &format!("binary#{i}"));
    }
    for i in 0..32 {
        check::<Q>(&random_cols::<Q>(&mut rng), &format!("random#{i}"));
    }
}

#[test]
fn ntt_3889() {
    run::<3889>();
}

#[test]
fn ntt_9721() {
    run::<9721>();
}

/// Per-level bounds proven in the module comment of `vertical_gen`, as `ceil(bound * q)`.
fn level_bounds<const Q: u16>() -> [i32; 7] {
    if Q == 9721 {
        [25024, 21298, 27738, 21700, 20804, 20671, 20652]
    } else {
        [9838, 12075, 14378, 19120, 24143, 8819, 13231]
    }
}

fn bounds<const Q: u16>() {
    let mut rng = Rng::new(0xb0 ^ Q as u64);
    let mut worst = [0i32; 7];
    let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
    for _ in 0..8 {
        cases.push(random_cols::<Q>(&mut rng));
        cases.push(binary_cols(&mut rng));
    }
    for c in &cases {
        for p in 0..32 {
            let mut lmax = [0i32; 7];
            shadow::<Q>(&c[p], &mut lmax);
            for l in 0..7 {
                worst[l] = worst[l].max(lmax[l]);
            }
        }
    }
    let claim = level_bounds::<Q>();
    for l in 0..7 {
        let c = claim[l];
        assert!(worst[l] <= c, "q={Q} level {l}: observed {} > claimed {c}", worst[l]);
        assert!(c < 32768, "q={Q} level {l}: claimed bound {c} exceeds i16");
        println!(
            "q={Q} level {l}: observed {} ({:.3} q), claimed {c} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            c as f64 / Q as f64
        );
    }
    assert_eq!(Tw::<Q>::OUTPUT_BOUND, claim[6]);
}

#[test]
fn bounds_3889() {
    bounds::<3889>();
}

#[test]
fn bounds_9721() {
    bounds::<9721>();
}

fn mul<const Q: u16>() {
    let mut rng = Rng::new(0xf00d ^ Q as u64);
    let pa: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let pb: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let mut a = Batch32::from_binary(&pa);
    let mut b = Batch32::from_binary(&pb);
    let mut batches = [a.clone(), b.clone()];
    ntt_gen_batches::<Q>(&mut batches);
    a = batches[0].clone();
    b = batches[1].clone();
    let mut c = Batch32::zero(Representation::Ntt);
    unsafe { pointwise::mul_batch_batch::<Q>(&a, &b, &mut c) };
    for p in 0..32 {
        let ca = scalar::lift(&pa[p]);
        let cb = scalar::lift(&pb[p]);
        let want = scalar::ntt::<Q>(&scalar::mul_mod_phi(&ca, &cb, Q));
        for j in 0..N {
            assert_eq!((c.v[j][p] as i32).rem_euclid(Q as i32) as u32, want[j], "q={Q} p={p} j={j}");
        }
    }
    // mul_batch_element against a single random element.
    let pe = BinaryPoly::random(&mut rng);
    let ce = scalar::lift(&pe);
    let ne = scalar::ntt::<Q>(&ce);
    let mut e = RingElement::zero(Representation::Ntt);
    for j in 0..N {
        e.v[j] = ne[j] as i16;
    }
    let me = MontElement::new::<Q>(&e);
    let mut d = Batch32::zero(Representation::Ntt);
    unsafe { pointwise::mul_batch_element::<Q>(&a, &me, &mut d) };
    for p in 0..32 {
        let ca = scalar::lift(&pa[p]);
        let want = scalar::ntt::<Q>(&scalar::mul_mod_phi(&ca, &ce, Q));
        for j in 0..N {
            assert_eq!((d.v[j][p] as i32).rem_euclid(Q as i32) as u32, want[j], "q={Q} p={p} j={j}");
        }
    }
}

/// The structural variants measured by the bench must all compute the same thing.
fn variants<const Q: u16>() {
    let mut rng = Rng::new(0xa11 ^ Q as u64);
    let cols = random_cols::<Q>(&mut rng);
    let mut want = to_batch(&cols);
    unsafe { ntt_gen_batch32::<Q>(&mut want) };
    for v in 0..7u32 {
        let mut got = to_batch(&cols);
        unsafe {
            match v {
                0 => ntt_gen_batch32_plan::<Q, 0>(&mut got),
                1 => ntt_gen_batch32_plan::<Q, 1>(&mut got),
                2 => ntt_gen_batch32_plan::<Q, 2>(&mut got),
                3 => ntt_gen_batch32_plan::<Q, 3>(&mut got),
                4 => ntt_gen_batch32_plan::<Q, 4>(&mut got),
                5 => ntt_gen_batch32_plan::<Q, 5>(&mut got),
                _ => ntt_gen_batch32_r27::<Q>(&mut got),
            }
        }
        for j in 0..N {
            assert_eq!(got.v[j], want.v[j], "variant {v} q={Q} slot {j}");
        }
    }
}

#[test]
fn variants_3889() {
    variants::<3889>();
}

#[test]
fn variants_9721() {
    variants::<9721>();
}

#[test]
fn mul_3889() {
    mul::<3889>();
}

#[test]
fn mul_9721() {
    mul::<9721>();
}

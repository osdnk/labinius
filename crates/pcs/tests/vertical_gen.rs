//! Correctness and bound tests for `simd::ntt::gen_small`.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::ntt::gen_small::{intt_gen_batch32, ntt_gen_batch32, Tw, TwI};
use bin_ntt::ring::*;

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
        assert!(
            a.abs() < 32768,
            "i16 overflow feeding a multiplication: {a}"
        );
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
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} slot {j}: shadow mismatch"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} slot {j}"
            );
            assert!(
                got.abs() <= bound,
                "{what} q={Q} poly {p} slot {j}: |{got}| > {bound}"
            );
        }
    }
}

fn random_cols<const Q: u16>(rng: &mut Rng) -> [[i16; N]; 32] {
    std::array::from_fn(|_| {
        std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16)
    })
}

/// Binary columns: the input a commitment's generic-side kernel sees when it is fed a lift.
fn binary_cols(rng: &mut Rng) -> [[i16; N]; 32] {
    std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() & 1) as i16))
}

fn adversarial<const Q: u16>() -> Vec<[[i16; N]; 32]> {
    let q = Q as i16;
    let mut out = Vec::new();
    out.push([[0i16; N]; 32]);
    out.push([[q; N]; 32]);
    out.push([[-q; N]; 32]);
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 2 == 0 { q } else { -q })
    }));
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

/// Per-level bounds proven in the module comment of `ntt::gen_small`, as `ceil(bound * q)`.
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
        assert!(
            worst[l] <= c,
            "q={Q} level {l}: observed {} > claimed {c}",
            worst[l]
        );
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

// ------------------------------------------------------------------ the inverse transform

/// Exact i32 mirror of `intt_gen_batch32`: same butterfly order, same Barrett placement, every
/// value kept as i32 so an i16 overflow is observable. `lmax[l]` = max |value| after level `l`.
fn shadow_inv<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 7]) -> [i32; N] {
    let q = Q as u64;
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        assert!(v[i].abs() <= TwI::<Q>::IN_BOUND, "input bound: {}", v[i]);
    }
    let mont = |a: i32, x: u16| -> i32 {
        assert!(
            a.abs() < 32768,
            "i16 overflow feeding a multiplication: {a}"
        );
        let w = Params::<Q>::to_mont(x);
        mont_mul_i16(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
    };
    let bar = |a: i32| -> i32 {
        assert!(a.abs() < 32768, "i16 overflow before barrett: {a}");
        barrett_i16(a as i16, Q) as i32
    };
    let ck = |a: i32| -> i32 {
        assert!(a.abs() < 32768, "i16 overflow: {a}");
        a
    };
    let w1 = Params::<Q>::OMEGA;
    let zi = |level: usize, k: usize| -> u16 {
        inv_mod(
            pow_mod(Params::<Q>::PSI as u64, twiddle_exp(level, k) as u64, q),
            q,
        ) as u16
    };
    // one inverse radix-3 butterfly on the three positions, returning the untwiddled sum first
    let r3i = |v: &mut [i32; N], i0: usize, i1: usize, i2: usize, z: u16, bar_s: bool| {
        let (y0, y1, y2) = (v[i0], v[i1], v[i2]);
        let u = mont(ck(y2 - y1), w1);
        let s = ck(y0 + y1 + y2);
        let z2 = (z as u64 * z as u64 % q) as u16;
        let a1 = mont(ck(y0 - y1 + u), z);
        let a2 = mont(ck(y0 - y2 - u), z2);
        v[i0] = if bar_s { bar(s) } else { s };
        v[i1] = a1;
        v[i2] = a2;
    };
    for k4 in 0..24 {
        for g in 0..9 {
            let b = 27 * k4 + 3 * g;
            if TwI::<Q>::BAR_IN {
                for t in 0..3 {
                    v[b + t] = bar(v[b + t]);
                }
            }
            r3i(&mut v, b, b + 1, b + 2, zi(6, 9 * k4 + g), TwI::<Q>::BAR_S6);
        }
    }
    lmax[6] = v.iter().map(|x| x.abs()).max().unwrap();
    for k4 in 0..24 {
        for bb in 0..3 {
            for j in 0..3 {
                let b = 27 * k4 + 9 * bb + j;
                r3i(
                    &mut v,
                    b,
                    b + 3,
                    b + 6,
                    zi(5, 3 * k4 + bb),
                    TwI::<Q>::BAR_S5[j],
                );
            }
        }
    }
    lmax[5] = v.iter().map(|x| x.abs()).max().unwrap();
    for k4 in 0..24 {
        for i in 0..9 {
            let b = 27 * k4 + i;
            r3i(&mut v, b, b + 9, b + 18, zi(4, k4), TwI::<Q>::BAR_S4[i]);
        }
    }
    lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
    for k in 0..8 {
        for j in 0..27 {
            let b = 81 * k + j;
            r3i(&mut v, b, b + 27, b + 54, zi(3, k), TwI::<Q>::BAR_S3);
        }
    }
    lmax[3] = v.iter().map(|x| x.abs()).max().unwrap();
    for blk in 0..4 {
        for a in 0..3 {
            for j in 0..27 {
                let b = 162 * blk + 27 * a + j;
                let (y0, y1) = (v[b], v[b + 81]);
                let s = ck(y0 + y1);
                v[b] = if TwI::<Q>::BAR_S2[a] { bar(s) } else { s };
                v[b + 81] = mont(ck(y0 - y1), zi(2, blk));
            }
        }
    }
    lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
    for c in 0..2 {
        for i in 0..162 {
            let b = 324 * c + i;
            let (y0, y1) = (v[b], v[b + 162]);
            let s = ck(y0 + y1);
            v[b] = if TwI::<Q>::BAR_S1 { bar(s) } else { s };
            v[b + 162] = mont(ck(y0 - y1), zi(1, c));
        }
    }
    lmax[1] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 0: the Phi_6 recombination carries the whole 1/648 * det normalisation, then centering.
    let det = inv_mod((2 * Params::<Q>::ZETA6 as u64 + q - 1) % q, q);
    let ka = (det * inv_mod(324, q) % q) as u16;
    let kb = inv_mod(648, q) as u16;
    let kc = ((q - det * inv_mod(648, q) % q) % q) as u16;
    let half = (Q as i32 - 1) / 2;
    let mut raw = 0i32;
    let mut out = [0i32; N];
    for i in 0..324 {
        let (y0, y1) = (v[i], v[i + 324]);
        let d = ck(y0 - y1);
        let s = ck(y0 + y1);
        let a1 = mont(d, ka);
        let a0 = ck(mont(s, kb) + mont(d, kc));
        raw = raw.max(a0.abs()).max(a1.abs());
        let center = |mut x: i32| {
            if x > half {
                x -= Q as i32;
            }
            if x < -half {
                x += Q as i32;
            }
            x
        };
        out[i] = center(a0);
        out[i + 324] = center(a1);
    }
    lmax[0] = raw;
    out
}

fn check_inv<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut [i32; 7]) {
    let mut b = Batch32::zero(Representation::Ntt);
    for p in 0..32 {
        for j in 0..N {
            b.v[j][p] = cols[p][j];
        }
    }
    unsafe { intt_gen_batch32::<Q>(&mut b) };
    assert_eq!(b.representation, Representation::Coefficients);
    let half = TwI::<Q>::OUT_BOUND;
    for p in 0..32 {
        let mut lmax = [0i32; 7];
        let want_shadow = shadow_inv::<Q>(&cols[p], &mut lmax);
        for l in 0..7 {
            worst[l] = worst[l].max(lmax[l]);
        }
        let want = scalar::intt::<Q>(&scalar::normalize_i16(&cols[p], Q));
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} coeff {j}: shadow mismatch"
            );
            assert!(
                got.abs() <= half,
                "{what} q={Q} poly {p} coeff {j}: |{got}| > {half}"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} coeff {j}"
            );
        }
    }
}

/// NTT-domain columns with `|v| <= bound`.
fn random_ntt_cols<const Q: u16>(rng: &mut Rng, bound: i32) -> [[i16; N]; 32] {
    std::array::from_fn(|_| {
        std::array::from_fn(|_| (rng.below(2 * bound as u32 + 1) as i32 - bound) as i16)
    })
}

fn adversarial_ntt<const Q: u16>() -> Vec<[[i16; N]; 32]> {
    let m = TwI::<Q>::IN_BOUND as i16;
    let mut out = Vec::new();
    out.push([[0i16; N]; 32]);
    out.push([[m; N]; 32]);
    out.push([[-m; N]; 32]);
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 2 == 0 { m } else { -m })
    }));
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 3 == 0 { m } else { -m })
    }));
    out.push(std::array::from_fn(|p| {
        std::array::from_fn(|j| if (j / 27 + p) % 2 == 0 { m } else { -m })
    }));
    for &u in &[0usize, 1, 2, 26, 27, 80, 81, 323, 324, 647] {
        let mut c = [[0i16; N]; 32];
        for p in 0..32 {
            c[p][u] = if p % 2 == 0 { m } else { -m };
        }
        out.push(c);
    }
    out
}

fn run_inv<const Q: u16>() {
    let mut rng = Rng::new(0x1117 ^ Q as u64);
    let mut worst = [0i32; 7];
    for (i, c) in adversarial_ntt::<Q>().iter().enumerate() {
        check_inv::<Q>(c, &format!("adversarial#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, TwI::<Q>::IN_BOUND);
        check_inv::<Q>(&c, &format!("lazy#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, (Q as i32 - 1) / 2);
        check_inv::<Q>(&c, &format!("centered#{i}"), &mut worst);
    }
    // the real inputs: the forward kernel's own output
    for i in 0..8 {
        let cols = random_cols::<Q>(&mut rng);
        let mut b = to_batch(&cols);
        unsafe { ntt_gen_batch32::<Q>(&mut b) };
        let ntt: [[i16; N]; 32] = std::array::from_fn(|p| std::array::from_fn(|j| b.v[j][p]));
        check_inv::<Q>(&ntt, &format!("forward#{i}"), &mut worst);
    }
    let claim = TwI::<Q>::BOUND;
    for l in 0..7 {
        assert!(
            worst[l] <= claim[l],
            "q={Q} inverse level {l}: {} > claimed {}",
            worst[l],
            claim[l]
        );
        assert!(
            claim[l] < 32768,
            "q={Q} inverse level {l}: claimed {} exceeds i16",
            claim[l]
        );
        println!(
            "q={Q} inverse level {l}: observed {} ({:.3} q), claimed {} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            claim[l],
            claim[l] as f64 / Q as f64
        );
    }
    println!(
        "q={Q} inverse: peak intermediate {} ({:.4} q, budget {:.4} q), {} Barretts per batch",
        TwI::<Q>::PEAK,
        TwI::<Q>::PEAK as f64 / Q as f64,
        32767.0 / Q as f64,
        216 * (3 * TwI::<Q>::BAR_IN as u32 + TwI::<Q>::BAR_S6 as u32 + TwI::<Q>::BAR_S3 as u32)
            + 72 * (TwI::<Q>::BAR_S5[0] as u32
                + TwI::<Q>::BAR_S5[1] as u32
                + TwI::<Q>::BAR_S5[2] as u32)
            + 24 * TwI::<Q>::BAR_S4.iter().filter(|x| **x).count() as u32
            + 108
                * (TwI::<Q>::BAR_S2[0] as u32
                    + TwI::<Q>::BAR_S2[1] as u32
                    + TwI::<Q>::BAR_S2[2] as u32)
            + 324 * TwI::<Q>::BAR_S1 as u32
    );
}

#[test]
fn intt_3889() {
    run_inv::<3889>();
}

#[test]
fn intt_9721() {
    run_inv::<9721>();
}

/// `intt(ntt(x)) == x` for the centered representative of `x mod q`.
fn round_trip<const Q: u16>() {
    let mut rng = Rng::new(0x0d0d ^ Q as u64);
    let half = (Q as i32 - 1) / 2;
    let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
    for _ in 0..8 {
        cases.push(binary_cols(&mut rng));
        cases.push(random_cols::<Q>(&mut rng));
    }
    for (c, cols) in cases.iter().enumerate() {
        let mut b = to_batch(cols);
        unsafe {
            ntt_gen_batch32::<Q>(&mut b);
            intt_gen_batch32::<Q>(&mut b);
        }
        for p in 0..32 {
            for j in 0..N {
                let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                if want > half {
                    want -= Q as i32;
                }
                assert_eq!(
                    b.v[j][p] as i32, want,
                    "round trip #{c} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }
}

#[test]
fn round_trip_3889() {
    round_trip::<3889>();
}

#[test]
fn round_trip_9721() {
    round_trip::<9721>();
}

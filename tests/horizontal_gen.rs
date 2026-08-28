//! Correctness, bound and multiplication tests for the horizontal generic-input NTT.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::horizontal_gen::*;
use bin_ntt::simd::pointwise;
use bin_ntt::types::*;

// -------------------------------------------------------------------------------------------
// i32 shadow model: the exact sequence of operations the kernel performs, on one polynomial,
// with |x| < 2^15 checked after every operation and the per-level maximum recorded.
// -------------------------------------------------------------------------------------------

struct Shadow<const Q: u16> {
    /// s[r][j] = the lane of register r at in-lane position j.
    s: [[i16; 8]; 81],
    /// max |x| after each level (index l = after level l).
    max: [i32; 7],
}

fn chk(x: i32) -> i16 {
    assert!(x > -32768 && x < 32768, "i16 overflow in the shadow model: {x}");
    x as i16
}

impl<const Q: u16> Shadow<Q> {
    fn mul(a: i16, x: u16) -> i16 {
        let w = Params::<Q>::to_mont(x);
        mont_mul_i16(a, w, Params::<Q>::mont_pre(w), Q)
    }
    fn neg(x: u16) -> u16 {
        if x == 0 {
            0
        } else {
            Q - x
        }
    }
    fn record(&mut self, l: usize) {
        let mut m = 0i32;
        for r in 0..81 {
            for j in 0..8 {
                m = m.max((self.s[r][j] as i32).abs());
            }
        }
        self.max[l] = m;
    }

    fn run(input: &[i16; N]) -> Self {
        let mut sh = Shadow::<Q> { s: [[0i16; 8]; 81], max: [0i32; 7] };
        for r in 0..81 {
            for j in 0..8 {
                sh.s[r][j] = input[r + 81 * j];
            }
        }
        let z6 = Params::<Q>::ZETA6;
        let c0: [u16; 8] = std::array::from_fn(|j| if j < 4 { z6 } else { (1 + Q - z6) % Q });
        let c1: [u16; 8] = std::array::from_fn(|j| {
            let z = Params::<Q>::ZETA_L1[j / 4];
            if j % 4 < 2 {
                z
            } else {
                Self::neg(z)
            }
        });
        let c2: [u16; 8] = std::array::from_fn(|j| {
            let z = Params::<Q>::ZETA_L2[j / 2];
            if j % 2 == 0 {
                z
            } else {
                Self::neg(z)
            }
        });
        for (l, c) in [(0usize, c0), (1, c1), (2, c2)].iter() {
            let stride = 4 >> l;
            for r in 0..81 {
                let a = sh.s[r];
                for j in 0..8 {
                    let (uj, vj) = (j | stride, j & !stride);
                    sh.s[r][j] = chk(a[vj] as i32 + Self::mul(a[uj], c[j]) as i32);
                }
            }
            sh.record(*l);
        }
        let bar = uses_barrett::<Q>();
        let om = Params::<Q>::OMEGA;
        for level in 3..=6 {
            let m = DEGREE[level] / 3;
            let nb = SUBRINGS[level] / 8;
            for r in 0..81 {
                if (r / m) % 3 != 0 {
                    continue;
                }
                for j in 0..8 {
                    let k = j * nb + r / DEGREE[level];
                    let z = Params::<Q>::zeta(level, k);
                    let z2 = (z as u32 * z as u32 % Q as u32) as u16;
                    let (a0, a1, a2) = (sh.s[r][j], sh.s[r + m][j], sh.s[r + 2 * m][j]);
                    let t1 = Self::mul(a1, z);
                    let t2 = Self::mul(a2, z2);
                    let u = Self::mul(chk(t1 as i32 - t2 as i32), om);
                    let a0 = if bar { barrett_i16(a0, Q) } else { a0 };
                    sh.s[r][j] = chk(a0 as i32 + t1 as i32 + t2 as i32);
                    sh.s[r + m][j] = chk(a0 as i32 - t2 as i32 + u as i32);
                    sh.s[r + 2 * m][j] = chk(a0 as i32 - t1 as i32 - u as i32);
                }
            }
            sh.record(level);
        }
        sh
    }

    fn element(&self) -> RingElement {
        let mut e = RingElement::zero(Representation::Ntt);
        for r in 0..81 {
            for j in 0..8 {
                e.v[r + 81 * j] = self.s[r][j];
            }
        }
        e
    }
}

// -------------------------------------------------------------------------------------------
// Inputs
// -------------------------------------------------------------------------------------------

fn adversarial<const Q: u16>() -> Vec<[i16; N]> {
    let q = Q as i16;
    let mut out = Vec::new();
    out.push([0i16; N]);
    out.push([q; N]);
    out.push([-q; N]);
    out.push(std::array::from_fn(|i| if i % 2 == 0 { q } else { -q }));
    out.push(std::array::from_fn(|i| if i % 81 == 0 { q } else { -q }));
    for pos in [0usize, 80, 81, 161, 162, 323, 324, 647] {
        let mut v = [0i16; N];
        v[pos] = 1;
        out.push(v);
        let mut v = [0i16; N];
        v[pos] = q;
        out.push(v);
        let mut v = [0i16; N];
        v[pos] = -q;
        out.push(v);
    }
    out
}

fn random_i16<const Q: u16>(rng: &mut Rng) -> [i16; N] {
    std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16)
}

fn random_binary(rng: &mut Rng) -> [i16; N] {
    std::array::from_fn(|_| (rng.next_u64() & 1) as i16)
}

fn to_element(v: &[i16; N]) -> RingElement {
    RingElement { v: *v, representation: Representation::Coefficients }
}

fn run_batch<const Q: u16>(inputs: &[[i16; N]; 4]) -> HBatch4 {
    let es: [RingElement; 4] = std::array::from_fn(|p| to_element(&inputs[p]));
    let mut b = HBatch4::from_elements(&es);
    unsafe { ntt_gen_hbatch4::<Q>(&mut b) };
    b
}

// -------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------

fn layout_roundtrip<const Q: u16>() {
    let mut rng = Rng::new(11);
    let inputs: [[i16; N]; 4] = std::array::from_fn(|_| random_i16::<Q>(&mut rng));
    let es: [RingElement; 4] = std::array::from_fn(|p| to_element(&inputs[p]));
    let b = HBatch4::from_elements(&es);
    for p in 0..4 {
        assert_eq!(b.get_as(p, Representation::Coefficients).v, inputs[p]);
        for r in 0..81 {
            for j in 0..8 {
                assert_eq!(b.v[r][8 * p + j], inputs[p][r + 81 * j]);
            }
        }
    }
    let polys: [BinaryPoly; 4] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let bb = HBatch4::from_binary(&polys);
    for p in 0..4 {
        let e = bb.get_as(p, Representation::Coefficients);
        assert_eq!(e, RingElement::from_binary(&polys[p]));
    }
}

/// SIMD output == scalar::ntt of the normalized input, slot for slot, in tree order; declared
/// output bound holds; the i32 shadow model agrees and never overflows i16.
fn correctness<const Q: u16>() {
    let mut rng = Rng::new(0xC0FFEE + Q as u64);
    let mut cases: Vec<[i16; N]> = adversarial::<Q>();
    for _ in 0..96 {
        cases.push(random_binary(&mut rng));
        cases.push(random_i16::<Q>(&mut rng));
    }
    let bound = out_bound::<Q>();
    let mut worst = 0i32;
    let mut worst_level = [0i32; 7];
    for chunk in cases.chunks(4) {
        let mut inputs = [[0i16; N]; 4];
        for (p, c) in chunk.iter().enumerate() {
            inputs[p] = *c;
        }
        let b = run_batch::<Q>(&inputs);
        for p in 0..chunk.len() {
            let e = b.get(p);
            let want = scalar::ntt::<Q>(&scalar::normalize_i16(&inputs[p], Q));
            let got = scalar::normalize_i16(&e.v, Q);
            assert_eq!(got, want, "q={Q}, polynomial {p} of a batch");
            for j in 0..N {
                let a = (e.v[j] as i32).abs();
                assert!(a <= bound, "q={Q}: |out[{j}]| = {a} > declared bound {bound}");
                worst = worst.max(a);
            }
            let sh = Shadow::<Q>::run(&inputs[p]);
            assert_eq!(sh.element().v, e.v, "q={Q}: shadow model disagrees with the kernel");
            for l in 0..7 {
                worst_level[l] = worst_level[l].max(sh.max[l]);
            }
        }
    }
    let lb = level_bounds::<Q>();
    for l in 0..7 {
        let declared = (lb[l + 1] * Q as u64 / BSCALE) as i32;
        assert!(
            worst_level[l] <= declared,
            "q={Q}: measured max after level {l} = {} exceeds the declared {declared}",
            worst_level[l]
        );
    }
    println!(
        "q={Q}: barrett={} declared out bound {} ({:.3}q), measured max {} ({:.3}q)",
        uses_barrett::<Q>(),
        bound,
        bound as f64 / Q as f64,
        worst,
        worst as f64 / Q as f64
    );
    println!(
        "q={Q}: per-level declared {:?}",
        (0..7)
            .map(|l| format!("{:.3}q", lb[l + 1] as f64 / BSCALE as f64))
            .collect::<Vec<_>>()
    );
    println!(
        "q={Q}: per-level measured {:?}",
        (0..7).map(|l| format!("{:.3}q", worst_level[l] as f64 / Q as f64)).collect::<Vec<_>>()
    );
}

/// NTT is a ring homomorphism: pointwise product of the SIMD outputs == NTT of the product.
fn multiplication<const Q: u16>() {
    let mut rng = Rng::new(999 + Q as u64);
    let pa: [BinaryPoly; 4] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let pb: [BinaryPoly; 4] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let mut ha = HBatch4::from_binary(&pa);
    let mut hb = HBatch4::from_binary(&pb);
    unsafe {
        ntt_gen_hbatch4::<Q>(&mut ha);
        ntt_gen_hbatch4::<Q>(&mut hb);
    }
    let mut ba = Batch32::zero(Representation::Ntt);
    let mut bb = Batch32::zero(Representation::Ntt);
    for p in 0..4 {
        ba.set(p, &ha.get(p));
        bb.set(p, &hb.get(p));
    }
    let mut prod = Batch32::zero(Representation::Ntt);
    unsafe { pointwise::mul_batch_batch::<Q>(&ba, &bb, &mut prod) };
    for p in 0..4 {
        let a = scalar::lift(&pa[p]);
        let b = scalar::lift(&pb[p]);
        let want = scalar::ntt::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
        let got = scalar::normalize_i16(&prod.get(p).v, Q);
        assert_eq!(got, want, "q={Q}: mul_batch_batch, polynomial {p}");
        // and directly, without the vector pointwise multiply
        let na = scalar::normalize_i16(&ha.get(p).v, Q);
        let nb = scalar::normalize_i16(&hb.get(p).v, Q);
        assert_eq!(scalar::pointwise_mul(&na, &nb, Q), want);
    }
    // mul_batch_element against a single random element
    let e = pointwise::MontElement::new::<Q>(&hb.get(1));
    let mut out = Batch32::zero(Representation::Ntt);
    unsafe { pointwise::mul_batch_element::<Q>(&ba, &e, &mut out) };
    for p in 0..4 {
        let a = scalar::lift(&pa[p]);
        let b = scalar::lift(&pb[1]);
        let want = scalar::ntt::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
        assert_eq!(scalar::normalize_i16(&out.get(p).v, Q), want, "q={Q}: mul_batch_element {p}");
    }
}

/// The driver over a slice does the same thing as the per-batch kernel.
fn driver<const Q: u16>() {
    let mut rng = Rng::new(4242);
    let mut bs: Vec<HBatch4> = (0..7)
        .map(|_| {
            let es: [RingElement; 4] =
                std::array::from_fn(|_| to_element(&random_i16::<Q>(&mut rng)));
            HBatch4::from_elements(&es)
        })
        .collect();
    let copy = bs.clone();
    unsafe { ntt_gen_hbatch4_many::<Q>(&mut bs) };
    for (b, c) in bs.iter().zip(copy.iter()) {
        let mut one = c.clone();
        unsafe { ntt_gen_hbatch4::<Q>(&mut one) };
        assert_eq!(b.v, one.v);
    }
}

#[test]
fn layout_3889() {
    layout_roundtrip::<3889>();
}
#[test]
fn layout_9721() {
    layout_roundtrip::<9721>();
}
#[test]
fn correctness_3889() {
    correctness::<3889>();
}
#[test]
fn correctness_9721() {
    correctness::<9721>();
}
#[test]
fn multiplication_3889() {
    multiplication::<3889>();
}
#[test]
fn multiplication_9721() {
    multiplication::<9721>();
}
#[test]
fn driver_3889() {
    driver::<3889>();
}
#[test]
fn driver_9721() {
    driver::<9721>();
}

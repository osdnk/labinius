//! Correctness, bound and product tests for the quadratic-slot tree (q in `QS_QUAD`): the scalar
//! reference against the definition, all three SIMD kernels — binary, generic and the generic
//! inverse — against the scalar reference slot for slot, the declared bounds, an i32 shadow model
//! of the binary kernel's schedule, and the `R_162` decomposition of a transform.
use bin_ntt::F162;
use bin_ntt::f162;
use bin_ntt::params::*;
use bin_ntt::recursion::limbs;
use bin_ntt::rng::Rng;
use bin_ntt::scalar::{self, Coeffs};
use bin_ntt::simd::transpose_f162::{self as tf, BinaryIndex32};
use bin_ntt::simd::vertical_bin_quad as vq;
use bin_ntt::simd::vertical_gen_quad as vgq;
use bin_ntt::types::*;

// ------------------------------------------------------------------ inputs

/// The 648 binary coefficients of one ring element.
type Bin = [u32; N];

fn monomial(d: usize) -> Bin {
    let mut c = [0u32; N];
    c[d] = 1;
    c
}

fn random_bin(rng: &mut Rng) -> Bin {
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

/// A batch as the 128 `F162` a commitment reads it from, and the index rows the front end slices.
fn elems_of(polys: &[Bin; 32]) -> [F162; 128] {
    let mut e = [F162([0; 3]); 128];
    for p in 0..32 {
        e[4 * p..4 * p + 4].copy_from_slice(&f162::pack4(&polys[p]));
    }
    e
}

fn idx_of(polys: &[Bin; 32]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    unsafe { tf::slice_f162_into(&elems_of(polys), &mut out) };
    out
}

/// All-zero, all-ones, alternating patterns and single monomials at the block boundaries.
fn adversarial() -> Vec<Bin> {
    let mut v = Vec::new();
    v.push([0u32; N]);
    v.push([1u32; N]);
    for phase in 0..2 {
        let mut alt = [0u32; N];
        let mut alt3 = [0u32; N];
        for i in 0..N {
            alt[i] = (i % 2 == phase) as u32;
            alt3[i] = (i % 3 == phase) as u32;
        }
        v.push(alt);
        v.push(alt3);
    }
    for b in 0..4 {
        let mut p = [0u32; N];
        for i in 0..162 {
            p[i + 162 * b] = 1;
        }
        v.push(p);
    }
    for d in [0usize, 1, 53, 54, 107, 108, 161, 162, 323, 324, 485, 486, 646, 647] {
        v.push(monomial(d));
    }
    v
}

fn batches(count: usize, seed: u64) -> Vec<[Bin; 32]> {
    let mut rng = Rng::new(seed);
    let adv = adversarial();
    let mut out = Vec::new();
    let mut b0: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().take(32).enumerate() {
        b0[i] = *p;
    }
    out.push(b0);
    let mut b1: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().skip(32).enumerate() {
        b1[i] = *p;
    }
    for i in adv.len().saturating_sub(32)..32 {
        b1[i] = random_bin(&mut rng);
    }
    out.push(b1);
    for _ in 0..count {
        out.push(std::array::from_fn(|_| random_bin(&mut rng)));
    }
    out
}

fn random_coeffs(rng: &mut Rng, q: u16) -> Coeffs {
    std::array::from_fn(|_| (rng.next_u64() % q as u64) as u32)
}

// ------------------------------------------------------------------ the tree itself

#[test]
fn tree_shape() {
    // the leaves are the units mod 972, each exactly once
    let mut seen = [false; 972];
    for j in 0..QUAD_SLOTS {
        let u = QUAD_SLOT_EXP[j] as usize;
        assert!(u % 2 == 1 && u % 3 != 0, "leaf {j} exponent {u} is not a unit");
        assert!(!seen[u]);
        seen[u] = true;
    }
    // sub-ring invariant: X^n - psi'^e with (n/2) | e, and every split is integral
    for level in 1..=6 {
        for k in 0..SUBRINGS_Q[level] {
            let e = subring_exp_quad(level, k);
            assert_eq!(e as usize % (DEGREE_Q[level] / 2), 0, "level {level} sub-ring {k}");
            if level < 6 {
                assert_eq!(e % RADIX_Q[level] as u32, 0);
            }
        }
    }
    // the R_162 classes of the two trees agree, and each class owns a + and a - leaf
    for s in 0..162 {
        let v = QUAD_POW3_CLASS[s] as usize;
        assert_eq!(v, bin_ntt::api::POW3_SLOT_EXP[s] as usize);
        let (jp, jm) = (QUAD_CLASS_SLOT[0][s] as usize, QUAD_CLASS_SLOT[1][s] as usize);
        assert_eq!(QUAD_SLOT_EXP[jp] as usize, v);
        assert_eq!(QUAD_SLOT_EXP[jm] as usize, v + 486);
    }
}

fn roots<const Q: u16>() {
    let q = Q as u64;
    let psi = ParamsQ::<Q>::PSI972 as u64;
    assert_eq!(pow_mod(psi, 972, q), 1);
    assert_ne!(pow_mod(psi, 486, q), 1);
    assert_ne!(pow_mod(psi, 324, q), 1);
    assert_eq!(pow_mod(psi, 486, q), q - 1);
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    assert_eq!((z6 * z6 + q - z6 + 1) % q, 0);
    let om = ParamsQ::<Q>::OMEGA as u64;
    assert_eq!((om * om + om + 1) % q, 0);
    // Phi_1944 = prod_j (X^2 - psi'^u_j), checked at random points
    let mut rng = Rng::new(7);
    for _ in 0..4 {
        let x = rng.next_u64() % q;
        let mut p = 1u64;
        for j in 0..QUAD_SLOTS {
            p = p * ((x * x % q + q - ParamsQ::<Q>::LEAF_C[j] as u64) % q) % q;
        }
        let want = (pow_mod(x, 648, q) + q - pow_mod(x, 324, q) + 1) % q;
        assert_eq!(p, want, "factorisation at x = {x}");
    }
}

#[test]
fn roots_and_factorisation() {
    roots::<2917>();
    roots::<4861>();
    roots::<12637>();
}

// ------------------------------------------------------------------ scalar reference

fn scalar_matches_definition<const Q: u16>() {
    let mut rng = Rng::new(3 + Q as u64);
    for _ in 0..4 {
        let a = random_coeffs(&mut rng, Q);
        let out = scalar::ntt_quad::<Q>(&a);
        for j in 0..QUAD_SLOTS {
            let (r0, r1) = scalar::eval_quad_at::<Q>(&a, QUAD_SLOT_EXP[j] as u32);
            assert_eq!((out[2 * j], out[2 * j + 1]), (r0, r1), "leaf {j}");
        }
    }
}

#[test]
fn scalar_ntt_quad_is_the_leaf_reduction() {
    scalar_matches_definition::<2917>();
    scalar_matches_definition::<4861>();
    scalar_matches_definition::<12637>();
}

fn scalar_product<const Q: u16>() {
    let mut rng = Rng::new(11 + Q as u64);
    for _ in 0..3 {
        let a = random_coeffs(&mut rng, Q);
        let b = random_coeffs(&mut rng, Q);
        let want = scalar::ntt_quad::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
        let got = scalar::mul_quad_slots::<Q>(&scalar::ntt_quad::<Q>(&a), &scalar::ntt_quad::<Q>(&b));
        assert_eq!(want, got);
    }
}

#[test]
fn scalar_product_identity() {
    scalar_product::<2917>();
    scalar_product::<4861>();
    scalar_product::<12637>();
}

/// `y_k(theta^v)` computed directly: component k is `sum_m y[4m+k] Y^m`, evaluated at
/// `theta^v = psi'^{2v}` by Horner.
fn component_eval<const Q: u16>(y: &Coeffs, k: usize, v: u32) -> u32 {
    let q = Q as u64;
    let th = pow_mod(ParamsQ::<Q>::PSI972 as u64, 2 * v as u64 % 972, q);
    let mut acc = 0u64;
    for m in (0..162).rev() {
        acc = (acc * th + y[4 * m + k] as u64) % q;
    }
    acc as u32
}

fn decompose<const Q: u16>() {
    let mut rng = Rng::new(23 + Q as u64);
    for _ in 0..3 {
        let a = random_coeffs(&mut rng, Q);
        let got = scalar::decompose_quad_648_to_4x162::<Q>(&scalar::ntt_quad::<Q>(&a));
        for k in 0..4 {
            for s in 0..162 {
                assert_eq!(
                    got[k][s],
                    component_eval::<Q>(&a, k, QUAD_POW3_CLASS[s] as u32),
                    "component {k} slot {s}"
                );
            }
        }
    }
}

#[test]
fn decomposition_matches_the_definition() {
    decompose::<2917>();
    decompose::<4861>();
    decompose::<12637>();
}

// ------------------------------------------------------------------ i32 shadow model
//
// Replays the binary kernel's exact schedule lane-wise in i32, so every intermediate can be
// checked against the 2^15 invariant and the exact (not just mod q) output compared.

struct Shadow<const Q: u16> {
    /// max |value| after: [0] the fused lookups + level 2, [1..=3] levels 3, 4, 5.
    max: [i32; 4],
    max_any: i32,
}

impl<const Q: u16> Shadow<Q> {
    fn new() -> Self {
        Shadow { max: [0; 4], max_any: 0 }
    }
    fn see(&mut self, x: i32) -> i16 {
        let a = x.abs();
        if a > self.max_any {
            self.max_any = a;
        }
        assert!(a < 32768, "i16 overflow: {x}");
        x as i16
    }
    fn mont(&mut self, a: i16, x: u16) -> i16 {
        let w = Params::<Q>::to_mont(x);
        let r = mont_mul_i16(a, w, Params::<Q>::mont_pre(w), Q);
        self.see(r as i32)
    }
    fn bar(&mut self, a: i16) -> i16 {
        self.see(bin_ntt::simd::vertical_bin_asm::barrett_lut_i16(a, Q) as i32)
    }
    fn r3(&mut self, a0: i16, a1: i16, a2: i16, zeta: u16, bar: bool) -> (i16, i16, i16) {
        let q = Q as u64;
        let z2 = (zeta as u64 * zeta as u64 % q) as u16;
        let t1 = self.mont(a1, zeta);
        let t2 = self.mont(a2, z2);
        let d = self.see(t1 as i32 - t2 as i32);
        let u = self.mont(d, ParamsQ::<Q>::OMEGA);
        let a0 = if bar { self.bar(a0) } else { a0 };
        (
            self.see(a0 as i32 + t1 as i32 + t2 as i32),
            self.see(a0 as i32 - t2 as i32 + u as i32),
            self.see(a0 as i32 - t1 as i32 - u as i32),
        )
    }
    /// One polynomial through the binary kernel's schedule.
    fn run(&mut self, poly: &Bin) -> [i16; N] {
        let q = Q as u64;
        let z6 = ParamsQ::<Q>::ZETA6 as u64;
        let kappa = [z6, (1 + q - z6) % q];
        let bar = vq::bar_levels(Q);
        let mut v = [0i16; N];
        let om = ParamsQ::<Q>::OMEGA as u64;
        // levels 0 + 1 as the 16-entry table gives them, before any folded twiddle
        let base_of = |k: usize, i: usize| -> u64 {
            let (s0, s1) = (k / 2, k % 2);
            let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
            let n = [i, i + 162, i + 324, i + 486].map(|c| poly[c] as u64);
            let inner = z1 * ((n[1] + kappa[s0] * n[3]) % q) % q;
            let t = if s1 == 0 { inner } else { (q - inner) % q };
            ((n[0] + kappa[s0] * n[2]) % q + t) % q
        };
        if vq::fold3(Q) {
            // level 2 straight out of the tables: output s of position i is the sum of three
            // lookups carrying zeta2^r omega^{r s} zeta3^{i/18}
            for k in 0..4 {
                let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
                for s in 0..3 {
                    let z3 = ParamsQ::<Q>::ZETA_L3[3 * k + s] as u64;
                    for i in 0..54 {
                        let e = |r: usize| -> i32 {
                            let g = pow_mod(z2, r as u64, q) * pow_mod(om, (r * s) as u64, q) % q
                                * pow_mod(z3, (i / 18) as u64, q)
                                % q;
                            center(base_of(k, i + 54 * r) * g % q, q) as i32
                        };
                        let x = self.see(e(0) + e(1));
                        v[162 * k + 54 * s + i] = self.see(x as i32 + e(2));
                    }
                }
            }
        } else {
            for k in 0..4 {
                let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
                for i in 0..162 {
                    let f = pow_mod(z2, (i / 54) as u64, q);
                    v[162 * k + i] = self.see(center(base_of(k, i) * f % q, q) as i32);
                }
            }
            // level 2: omega-only radix-3 on (i, i+54, i+108)
            for k in 0..4 {
                for i in 0..54 {
                    let b = 162 * k + i;
                    let (t1, t2) = (v[b + 54], v[b + 108]);
                    let d = self.see(t1 as i32 - t2 as i32);
                    let u = self.mont(d, ParamsQ::<Q>::OMEGA);
                    let a0 = v[b];
                    v[b] = self.see(a0 as i32 + t1 as i32 + t2 as i32);
                    v[b + 54] = self.see(a0 as i32 - t2 as i32 + u as i32);
                    v[b + 108] = self.see(a0 as i32 - t1 as i32 - u as i32);
                }
            }
        }
        self.level_max(0, &v);
        if vq::fold3(Q) {
            // level 3: omega-only radix-3 on (i, i+18, i+36) of each 54-block
            for base in (0..N).step_by(54) {
                for i in 0..18 {
                    let (t1, t2) = (v[base + i + 18], v[base + i + 36]);
                    let d = self.see(t1 as i32 - t2 as i32);
                    let u = self.mont(d, ParamsQ::<Q>::OMEGA);
                    let a0 = v[base + i];
                    v[base + i] = self.see(a0 as i32 + t1 as i32 + t2 as i32);
                    v[base + i + 18] = self.see(a0 as i32 - t2 as i32 + u as i32);
                    v[base + i + 36] = self.see(a0 as i32 - t1 as i32 - u as i32);
                }
            }
            self.level_max(1, &v);
            for (l, (blk, level)) in [(18usize, 4usize), (6, 5)].iter().enumerate() {
                let m = blk / 3;
                for base in (0..N).step_by(*blk) {
                    let zeta = ParamsQ::<Q>::zeta(*level, base / blk);
                    for i in 0..m {
                        let (o0, o1, o2) = self.r3(
                            v[base + i],
                            v[base + i + m],
                            v[base + i + 2 * m],
                            zeta,
                            bar[l + 1],
                        );
                        v[base + i] = o0;
                        v[base + i + m] = o1;
                        v[base + i + 2 * m] = o2;
                    }
                }
                self.level_max(l + 2, &v);
            }
            return v;
        }
        for (l, (blk, level)) in [(54usize, 3usize), (18, 4), (6, 5)].iter().enumerate() {
            let m = blk / 3;
            for base in (0..N).step_by(*blk) {
                let zeta = ParamsQ::<Q>::zeta(*level, base / blk);
                for i in 0..m {
                    let (o0, o1, o2) =
                        self.r3(v[base + i], v[base + i + m], v[base + i + 2 * m], zeta, bar[l]);
                    v[base + i] = o0;
                    v[base + i + m] = o1;
                    v[base + i + 2 * m] = o2;
                }
            }
            self.level_max(l + 1, &v);
        }
        v
    }
    fn level_max(&mut self, l: usize, v: &[i16; N]) {
        for x in v.iter() {
            let a = (*x as i32).abs();
            if a > self.max[l] {
                self.max[l] = a;
            }
        }
    }
}

// ------------------------------------------------------------------ the binary kernel

fn bin_kernel<const Q: u16>() {
    let mut sh = Shadow::<Q>::new();
    let bound = vq::output_bound(Q);
    let mut out = Batch32::zero(Representation::Ntt);
    for polys in batches(24, 5 + Q as u64) {
        let idx = idx_of(&polys);
        unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
        for p in 0..32 {
            let want = scalar::ntt_quad::<Q>(&polys[p]);
            let shadow = sh.run(&polys[p]);
            for j in 0..N {
                let got = out.v[j][p];
                assert_eq!(
                    (got as i32).rem_euclid(Q as i32) as u32,
                    want[j],
                    "q={Q} row {j} lane {p}"
                );
                assert_eq!(got, shadow[j], "q={Q} row {j} lane {p}: shadow model");
                assert!(
                    (got as i32).abs() <= bound,
                    "q={Q} row {j} lane {p}: |{got}| over the declared bound {bound}"
                );
            }
        }
    }
    let model = vq::bin_model(Q, vq::bar_levels(Q));
    assert!(sh.max_any < 32768);
    assert!(sh.max_any <= model.1, "shadow peak {} > model {}", sh.max_any, model.1);
    for l in 0..4 {
        assert!(sh.max[l] <= model.0[l], "level {l}: {} > {}", sh.max[l], model.0[l]);
    }
    println!(
        "q={Q} binary: observed per level {:?} (model {:?}), output {:.3} q of the declared {:.3} q",
        sh.max,
        model.0,
        sh.max[3] as f64 / Q as f64,
        bound as f64 / Q as f64
    );
}

#[test]
fn binary_kernel_matches_scalar() {
    bin_kernel::<2917>();
    bin_kernel::<4861>();
    bin_kernel::<12637>();
}

/// The block sink the commitment consumes the transform through sees exactly the 648 rows the
/// plain entry point writes, 18 at a time, in block order.
fn bin_sink<const Q: u16>() {
    let mut rng = Rng::new(99 + Q as u64);
    let polys: [Bin; 32] = std::array::from_fn(|_| random_bin(&mut rng));
    let e = elems_of(&polys);
    let idx = idx_of(&polys);
    let mut out = Batch32::zero(Representation::Ntt);
    unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
    for p in 0..32 {
        let want = scalar::ntt_quad::<Q>(&f162::lift_elem(&e, p));
        for j in 0..N {
            assert_eq!(
                (out.v[j][p] as i32).rem_euclid(Q as i32) as u32,
                want[j],
                "q={Q} row {j} lane {p}"
            );
        }
    }

    #[repr(C, align(64))]
    struct Blk18([i16; 18 * 32]);
    struct Collect {
        buf: Vec<Blk18>,
        seen: Vec<usize>,
    }
    impl vq::BlockSink for Collect {
        unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
            self.buf[blk].0.as_mut_ptr()
        }
        unsafe fn block(&mut self, blk: usize, _dst: *const i16) {
            self.seen.push(blk);
        }
    }
    let mut c = Collect { buf: (0..36).map(|_| Blk18([0i16; 18 * 32])).collect(), seen: Vec::new() };
    unsafe { vq::ntt_quad_bin_batch32_sink::<Q, _>(&idx, &mut c) };
    assert_eq!(c.seen, (0..36).collect::<Vec<_>>());
    for blk in 0..36 {
        for r in 0..18 {
            for p in 0..32 {
                assert_eq!(c.buf[blk].0[32 * r + p], out.v[18 * blk + r][p], "block {blk} row {r}");
            }
        }
    }
}

#[test]
fn binary_sink_matches_the_plain_kernel() {
    bin_sink::<2917>();
    bin_sink::<4861>();
    bin_sink::<12637>();
}

// ------------------------------------------------------------------ the generic kernel

fn gen_inputs<const Q: u16>(count: usize, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let q = Q as i16;
    let mut out = Vec::new();
    // adversarial: every lane at +-q, alternating, the binary patterns, then random in [-q, q]
    let mut b = Batch32::zero(Representation::Coefficients);
    for j in 0..N {
        for p in 0..32 {
            b.v[j][p] = if (j + p) % 2 == 0 { q } else { -q };
        }
    }
    out.push(b);
    let mut b = Batch32::zero(Representation::Coefficients);
    for j in 0..N {
        for p in 0..32 {
            b.v[j][p] = if p % 3 == 0 { q } else if p % 3 == 1 { -q } else { 0 };
        }
    }
    out.push(b);
    for polys in batches(0, seed).into_iter().take(2) {
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = polys[p][j] as i16;
            }
        }
        out.push(b);
    }
    for _ in 0..count {
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = (rng.next_u64() % (2 * Q as u64 + 1)) as i16 - q;
            }
        }
        out.push(b);
    }
    out
}

fn gen_kernel<const Q: u16>() {
    let bound = vgq::output_bound(Q);
    let mut peak = 0i32;
    for b in gen_inputs::<Q>(12, 17 + Q as u64) {
        let mut got = b.clone();
        unsafe { vgq::ntt_quad_gen_batch32::<Q>(&mut got) };
        for p in 0..32 {
            let a: Coeffs =
                std::array::from_fn(|j| (b.v[j][p] as i32).rem_euclid(Q as i32) as u32);
            let want = scalar::ntt_quad::<Q>(&a);
            for j in 0..N {
                let g = got.v[j][p];
                assert_eq!(
                    (g as i32).rem_euclid(Q as i32) as u32,
                    want[j],
                    "generic q={Q} row {j} lane {p}"
                );
                assert!(
                    (g as i32).abs() <= bound,
                    "generic q={Q} row {j} lane {p}: |{g}| over the declared bound {bound}"
                );
                peak = peak.max((g as i32).abs());
            }
        }
    }
    println!(
        "q={Q} generic: output {:.3} q observed of the declared {:.3} q",
        peak as f64 / Q as f64,
        bound as f64 / Q as f64
    );
}

#[test]
fn generic_kernel_matches_scalar() {
    gen_kernel::<2917>();
    gen_kernel::<4861>();
    gen_kernel::<12637>();
}

// ------------------------------------------------------------------ products through the SIMD

/// `ntt_quad(a * b mod Phi) == mul_quad_slots(NTT(a), NTT(b))` with both transforms produced by
/// the SIMD kernels (the binary one for `a`, the generic one for `b`).
fn simd_product<const Q: u16>() {
    let mut rng = Rng::new(41 + Q as u64);
    let polys: [Bin; 32] = std::array::from_fn(|_| random_bin(&mut rng));
    let idx = idx_of(&polys);
    let mut ta = Batch32::zero(Representation::Ntt);
    unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut ta) };

    let mut bb = Batch32::zero(Representation::Coefficients);
    for j in 0..N {
        for p in 0..32 {
            bb.v[j][p] = (rng.next_u64() % Q as u64) as i16;
        }
    }
    let coeffs_b = bb.clone();
    let mut tb = bb;
    unsafe { vgq::ntt_quad_gen_batch32::<Q>(&mut tb) };

    for p in [0usize, 1, 17, 31] {
        let a: Coeffs = polys[p];
        let b: Coeffs = std::array::from_fn(|j| coeffs_b.v[j][p] as u32);
        let na: Coeffs =
            std::array::from_fn(|j| (ta.v[j][p] as i32).rem_euclid(Q as i32) as u32);
        let nb: Coeffs =
            std::array::from_fn(|j| (tb.v[j][p] as i32).rem_euclid(Q as i32) as u32);
        let want = scalar::ntt_quad::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
        assert_eq!(want, scalar::mul_quad_slots::<Q>(&na, &nb), "product q={Q} lane {p}");
    }
}

#[test]
fn product_through_the_simd_outputs() {
    simd_product::<2917>();
    simd_product::<4861>();
    simd_product::<12637>();
}

// ------------------------------------------------------------------ the vectorised inverse

/// A fully reduced transform written into a batch, centered — what the fold hands the inverse.
fn centered_slots<const Q: u16>(slots: &[Coeffs]) -> Batch32 {
    let half = ((Q - 1) / 2) as i32;
    let mut b = Batch32::zero(Representation::Ntt);
    for (p, s) in slots.iter().enumerate() {
        for j in 0..N {
            let x = s[j] as i32;
            b.v[j][p] = if x > half { (x - Q as i32) as i16 } else { x as i16 };
        }
    }
    b
}

/// `intt_quad_gen_batch32` against the scalar [`limbs::intt_quad`], slot for slot, on random
/// transforms fed in the centered form the fold produces.
fn inverse_kernel<const Q: u16>() {
    let mut rng = Rng::new(0x11D5 ^ Q as u64);
    let half = ((Q - 1) / 2) as i32;
    for _ in 0..6 {
        let coeffs: Vec<Coeffs> = (0..32).map(|_| random_coeffs(&mut rng, Q)).collect();
        let slots: Vec<Coeffs> = coeffs.iter().map(scalar::ntt_quad::<Q>).collect();
        let mut b = centered_slots::<Q>(&slots);
        unsafe { vgq::intt_quad_gen_batch32::<Q>(&mut b) };
        assert_eq!(b.representation, Representation::Coefficients);
        for p in 0..32 {
            let want = limbs::intt_quad::<Q>(&slots[p]);
            assert_eq!(want, coeffs[p], "q={Q} the scalar inverse is not the inverse");
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert!(got.abs() <= half, "q={Q} coefficient {j} lane {p}: |{got}| not centered");
                assert_eq!(
                    got.rem_euclid(Q as i32) as u32,
                    want[j],
                    "q={Q} coefficient {j} lane {p}"
                );
            }
        }
    }
    println!(
        "q={Q} inverse: reductions {:?}, per-level bound {:?}, input {:.3} q",
        vgq::inv_flags(Q),
        vgq::inv_bound(Q),
        vgq::in_bound(Q) as f64 / Q as f64
    );
}

#[test]
fn vectorised_inverse_matches_the_scalar_one() {
    inverse_kernel::<2917>();
    inverse_kernel::<4861>();
    inverse_kernel::<12637>();
}

/// The same on a lazily reduced transform: the binary kernel's output, inverted straight back to
/// the bits.
fn inverse_at_the_declared_bound<const Q: u16>() {
    let bound = vgq::in_bound(Q);
    let half = ((Q - 1) / 2) as i32;
    for polys in batches(3, 0x5EED ^ Q as u64) {
        let mut b = Batch32::zero(Representation::Ntt);
        unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx_of(&polys), &mut b) };
        for j in 0..N {
            for p in 0..32 {
                let x = b.v[j][p] as i32;
                assert!(x.abs() <= bound, "q={Q} row {j} lane {p}: |{x}| over the input bound");
            }
        }
        unsafe { vgq::intt_quad_gen_batch32::<Q>(&mut b) };
        for j in 0..N {
            for p in 0..32 {
                let got = b.v[j][p] as i32;
                assert!(got.abs() <= half, "q={Q} coefficient {j} lane {p} not centered");
                assert_eq!(got, polys[p][j] as i32, "q={Q} coefficient {j} lane {p}");
            }
        }
    }
}

#[test]
fn the_inverse_takes_a_lazily_reduced_transform() {
    inverse_at_the_declared_bound::<2917>();
    inverse_at_the_declared_bound::<4861>();
    inverse_at_the_declared_bound::<12637>();
}

#[test]
fn generic_round_trip() {
    fn go<const Q: u16>() {
        let mut rng = Rng::new(0x3C3C ^ Q as u64);
        let half = ((Q - 1) / 2) as i16;
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = rng.below(Q as u32) as i16 - half;
            }
        }
        let want = b.clone();
        unsafe {
            vgq::ntt_quad_gen_batch32::<Q>(&mut b);
            vgq::intt_quad_gen_batch32::<Q>(&mut b);
        }
        assert_eq!(b.v, want.v, "q={Q}");
    }
    go::<2917>();
    go::<4861>();
    go::<12637>();
}

/// A folding challenge enters `R_648` as `c(-X^4)`, a polynomial in `X^4`; modulo the leaf
/// `X^2 - psi'^u` that is `X^4 = psi'^{2u}`, a scalar, so the transform's odd rows vanish. This is
/// what lets the fold multiply a quadratic-slot base row by row instead of leaf by leaf.
fn subring_element_is_a_leaf_scalar<const Q: u16>() {
    let mut rng = Rng::new(0x4A4A ^ Q as u64);
    for _ in 0..4 {
        let mut a = [0u32; N];
        for m in 0..162 {
            a[4 * m] = rng.below(Q as u32);
        }
        let y = scalar::ntt_quad::<Q>(&a);
        for j in 0..QUAD_SLOTS {
            assert_eq!(y[2 * j + 1], 0, "q={Q} leaf {j} has an X coefficient");
        }
    }
}

#[test]
fn an_embedded_subring_element_has_no_x_coefficient() {
    subring_element_is_a_leaf_scalar::<2917>();
    subring_element_is_a_leaf_scalar::<4861>();
    subring_element_is_a_leaf_scalar::<12637>();
}

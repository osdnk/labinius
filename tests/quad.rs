//! Correctness, bound and product tests for the quadratic-slot tree (q in `QS_QUAD`): the scalar
//! reference against the definition, both SIMD kernels against the scalar reference slot for
//! slot, the declared output bounds, an i32 shadow model of the binary kernel's schedule, and the
//! `R_162` decomposition of a transform.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar::{self, Coeffs};
use bin_ntt::simd::transpose::{self, BinaryIndex32};
use bin_ntt::simd::vertical_bin_quad as vq;
use bin_ntt::simd::vertical_gen_quad as vgq;
use bin_ntt::types::*;

// ------------------------------------------------------------------ inputs

fn monomial(d: usize) -> BinaryPoly {
    let mut p = BinaryPoly::default();
    p.set(d, true);
    p
}

/// All-zero, all-ones, alternating patterns and single monomials at the block boundaries.
fn adversarial() -> Vec<BinaryPoly> {
    let mut v = Vec::new();
    v.push(BinaryPoly::default());
    let mut ones = BinaryPoly::default();
    for i in 0..N {
        ones.set(i, true);
    }
    v.push(ones);
    for phase in 0..2 {
        let mut alt = BinaryPoly::default();
        let mut alt3 = BinaryPoly::default();
        for i in 0..N {
            alt.set(i, i % 2 == phase);
            alt3.set(i, i % 3 == phase);
        }
        v.push(alt);
        v.push(alt3);
    }
    for b in 0..4 {
        let mut p = BinaryPoly::default();
        for i in 0..162 {
            p.set(i + 162 * b, true);
        }
        v.push(p);
    }
    for d in [0usize, 1, 53, 54, 107, 108, 161, 162, 323, 324, 485, 486, 646, 647] {
        v.push(monomial(d));
    }
    v
}

fn batches(count: usize, seed: u64) -> Vec<[BinaryPoly; 32]> {
    let mut rng = Rng::new(seed);
    let adv = adversarial();
    let mut out = Vec::new();
    let mut b0: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::default());
    for (i, p) in adv.iter().take(32).enumerate() {
        b0[i] = *p;
    }
    out.push(b0);
    let mut b1: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::default());
    for (i, p) in adv.iter().skip(32).enumerate() {
        b1[i] = *p;
    }
    for i in adv.len().saturating_sub(32)..32 {
        b1[i] = BinaryPoly::random(&mut rng);
    }
    out.push(b1);
    for _ in 0..count {
        out.push(std::array::from_fn(|_| BinaryPoly::random(&mut rng)));
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
    fn run(&mut self, poly: &BinaryPoly) -> [i16; N] {
        let q = Q as u64;
        let z6 = ParamsQ::<Q>::ZETA6 as u64;
        let kappa = [z6, (1 + q - z6) % q];
        let bar = vq::bar_levels(Q);
        let mut v = [0i16; N];
        // levels 0 + 1 + the folded level-2 twiddle, as the 16-entry table would give them
        for k in 0..4 {
            let (s0, s1) = (k / 2, k % 2);
            let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
            let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
            for i in 0..162 {
                let n = [i, i + 162, i + 324, i + 486].map(|c| poly.coeff(c) as u64);
                let inner = z1 * ((n[1] + kappa[s0] * n[3]) % q) % q;
                let t = if s1 == 0 { inner } else { (q - inner) % q };
                let base = ((n[0] + kappa[s0] * n[2]) % q + t) % q;
                let f = pow_mod(z2, (i / 54) as u64, q);
                v[162 * k + i] = self.see(center(base * f % q, q) as i32);
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
        self.level_max(0, &v);
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
        let idx = unsafe { transpose::slice_polys_idx(&polys) };
        unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
        for p in 0..32 {
            let want = scalar::ntt_quad::<Q>(&scalar::lift(&polys[p]));
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

/// The `_nt` entry point and the `F162` driver agree with the plain one.
fn bin_drivers<const Q: u16>() {
    use bin_fields::scalar::F162;
    let elems: Vec<F162> = bin_ntt::f162::random_elems(128 * 5, 99 + Q as u64);
    let mut out: Vec<Batch32> = (0..5).map(|_| Batch32::zero(Representation::Ntt)).collect();
    vq::ntt_quad_f162::<Q>(&elems, &mut out);
    for b in 0..5 {
        for p in 0..32 {
            let a = bin_ntt::f162::lift_elem(&elems, 32 * b + p);
            let want = scalar::ntt_quad::<Q>(&a);
            for j in 0..N {
                assert_eq!(
                    (out[b].v[j][p] as i32).rem_euclid(Q as i32) as u32,
                    want[j],
                    "f162 driver q={Q} batch {b} row {j} lane {p}"
                );
            }
        }
    }
    // the block hook sees the same 648 rows, 18 at a time
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
    let mut idx = BinaryIndex32::zero();
    unsafe {
        bin_ntt::simd::transpose_f162::slice_f162_into(
            &*(elems.as_ptr() as *const [F162; 128]),
            &mut idx,
        );
    }
    let mut c = Collect { buf: (0..36).map(|_| Blk18([0i16; 18 * 32])).collect(), seen: Vec::new() };
    unsafe { vq::ntt_quad_bin_batch32_sink::<Q, _>(&idx, &mut c) };
    assert_eq!(c.seen, (0..36).collect::<Vec<_>>());
    for blk in 0..36 {
        for r in 0..18 {
            for p in 0..32 {
                assert_eq!(c.buf[blk].0[32 * r + p], out[0].v[18 * blk + r][p], "block {blk} row {r}");
            }
        }
    }
}

#[test]
fn binary_drivers_and_hook() {
    bin_drivers::<2917>();
    bin_drivers::<4861>();
    bin_drivers::<12637>();
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
        out.push(Batch32::from_binary(&polys));
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

/// The fusion variants and the prefetching driver produce exactly what the shipped kernel does.
fn gen_variants<const Q: u16>() {
    let inputs = gen_inputs::<Q>(2, 71 + Q as u64);
    let mut want: Vec<Batch32> = inputs.clone();
    for b in want.iter_mut() {
        unsafe { vgq::ntt_quad_gen_batch32::<Q>(b) };
    }
    macro_rules! variant {
        ($plan:literal) => {{
            let mut got = inputs.clone();
            for b in got.iter_mut() {
                unsafe { vgq::ntt_quad_gen_batch32_plan::<Q, $plan>(b) };
            }
            for (g, w) in got.iter().zip(want.iter()) {
                for j in 0..N {
                    for p in 0..32 {
                        assert_eq!(
                            (g.v[j][p] as i32).rem_euclid(Q as i32),
                            (w.v[j][p] as i32).rem_euclid(Q as i32),
                            "plan {} q={Q} row {j} lane {p}",
                            $plan
                        );
                    }
                }
            }
        }};
    }
    variant!(0);
    variant!(1);
    variant!(2);
    variant!(3);
    let mut got = inputs.clone();
    vgq::ntt_quad_gen_batches::<Q>(&mut got);
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(g.v.iter().zip(w.v.iter()).all(|(a, b)| a == b), "prefetching driver q={Q}");
    }
}

#[test]
fn generic_variants_agree() {
    gen_variants::<2917>();
    gen_variants::<4861>();
    gen_variants::<12637>();
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
    let polys: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let idx = unsafe { transpose::slice_polys_idx(&polys) };
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
        let a: Coeffs = std::array::from_fn(|i| polys[p].coeff(i) as u32);
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

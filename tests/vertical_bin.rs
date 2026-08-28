//! Correctness, bound and multiplication tests for the binary vertical NTT.
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::pointwise::{self, MontElement};
use bin_ntt::simd::transpose::{self, BinaryIndex32};
use bin_ntt::simd::vertical_bin as vb;
use bin_ntt::types::*;

// ------------------------------------------------------------------ inputs

fn monomial(d: usize) -> BinaryPoly {
    let mut p = BinaryPoly::default();
    p.set(d, true);
    p
}

/// The adversarial set required by DESIGN.md section 7 plus a few more.
fn adversarial() -> Vec<BinaryPoly> {
    let mut v = Vec::new();
    v.push(BinaryPoly::default()); // all zero
    let mut ones = BinaryPoly::default();
    for i in 0..N {
        ones.set(i, true);
    }
    v.push(ones);
    for phase in 0..2 {
        let mut alt = BinaryPoly::default();
        for i in 0..N {
            alt.set(i, i % 2 == phase);
        }
        v.push(alt);
        let mut alt3 = BinaryPoly::default();
        for i in 0..N {
            alt3.set(i, i % 3 == phase);
        }
        v.push(alt3);
    }
    // block-structured: the four 162-blocks the nibble index is built from
    for b in 0..4 {
        let mut p = BinaryPoly::default();
        for i in 0..162 {
            p.set(i + 162 * b, true);
        }
        v.push(p);
    }
    for d in [0usize, 1, 80, 81, 161, 162, 163, 323, 324, 325, 485, 486, 646, 647] {
        v.push(monomial(d));
    }
    v
}

fn batches(count: usize, seed: u64) -> Vec<[BinaryPoly; 32]> {
    let mut rng = Rng::new(seed);
    let adv = adversarial();
    let mut out = Vec::new();
    // one batch made only of adversarial inputs (padded with zeros / repeats)
    let mut b0: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::default());
    for (i, p) in adv.iter().take(32).enumerate() {
        b0[i] = *p;
    }
    out.push(b0);
    let mut b1: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::default());
    for (i, p) in adv.iter().skip(32).enumerate() {
        b1[i] = *p;
    }
    // and one that mixes the extreme "all ones" with random
    for i in adv.len().saturating_sub(32)..32 {
        b1[i] = BinaryPoly::random(&mut rng);
    }
    out.push(b1);
    for _ in 0..count {
        out.push(std::array::from_fn(|_| BinaryPoly::random(&mut rng)));
    }
    out
}

// ------------------------------------------------------------------ transpose

#[test]
fn transpose_matches_scalar() {
    for polys in batches(64, 11) {
        let want = BinaryBatch32::from_polys_scalar(&polys);
        let got = unsafe { transpose::slice_polys(&polys) };
        assert!(want.idx == got.idx, "slice_polys != from_polys_scalar");
        for p in 0..32 {
            assert_eq!(got.poly(p), polys[p]);
        }
        // the kernel-side byte-index layout: rows[i][2p] = n, rows[i][2p+1] = 16 + n
        let widx = BinaryIndex32::from_nibbles(&want);
        let gidx = unsafe { transpose::slice_polys_idx(&polys) };
        for i in 0..162 {
            assert!(widx.rows[i] == gidx.rows[i], "slice_polys_idx row {i}");
        }
    }
}

// ------------------------------------------------------------------ i32 shadow model
//
// Mirrors the kernel's exact schedule lane-wise in i32 so that every intermediate can be checked
// against the 2^15 invariant, and the *exact* (not just mod q) output can be compared.

struct Shadow<const Q: u16> {
    /// max |value| seen after: [0] levels 0+1+2 (table combines), [1..=4] levels 3, 4, 5, 6.
    max: [i32; 5],
    /// max |value| of any intermediate whatsoever (must stay < 2^15).
    max_any: i32,
}

impl<const Q: u16> Shadow<Q> {
    fn new() -> Self {
        Shadow { max: [0; 5], max_any: 0 }
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
    /// The 16-entry table value used by the kernel for block k, level-3 child s2, role r, side ab.
    fn table(k: usize, s2: usize, r: usize, ab: usize, n: usize) -> i16 {
        let q = Q as u64;
        let (s0, s1) = (k / 2, k % 2);
        let z6 = Params::<Q>::ZETA6 as u64;
        let ka = if s0 == 0 { z6 } else { (1 + q - z6) % q };
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
        let f = pow_mod(z3, r as u64, q);
        let extra = if ab == 0 { 1 } else { zp };
        let (n0, n1, n2, n3) =
            ((n & 1) as u64, ((n >> 1) & 1) as u64, ((n >> 2) & 1) as u64, ((n >> 3) & 1) as u64);
        let inner = z1 * ((n1 + ka * n3) % q) % q;
        let t = if s1 == 0 { inner } else { (q - inner) % q };
        let base = ((n0 + ka * n2) % q + t) % q;
        center(base * f % q * extra % q, q)
    }
    /// a0 + t1 + t2, a0 - t2 + u, a0 - t1 - u with u = omega * (t1 - t2).
    fn r3(&mut self, a0: i16, t1: i16, t2: i16) -> (i16, i16, i16) {
        let d = self.see(t1 as i32 - t2 as i32);
        let u = self.mont(d, Params::<Q>::OMEGA);
        let o0 = self.see(a0 as i32 + t1 as i32 + t2 as i32);
        let o1 = self.see(a0 as i32 - t2 as i32 + u as i32);
        let o2 = self.see(a0 as i32 - t1 as i32 - u as i32);
        (o0, o1, o2)
    }
    fn r3_tw(&mut self, a0: i16, a1: i16, a2: i16, z: u16, bar: bool) -> (i16, i16, i16) {
        let z2 = (z as u64 * z as u64 % Q as u64) as u16;
        let t1 = self.mont(a1, z);
        let t2 = self.mont(a2, z2);
        let a0 = if bar { self.see(barrett_i16(a0, Q) as i32) } else { a0 };
        self.r3(a0, t1, t2)
    }

    fn run(&mut self, poly: &BinaryPoly) -> [i16; N] {
        let bar = vb::needs_barrett(Q);
        let mut v = [0i16; N];
        let nib: Vec<usize> = (0..162)
            .map(|i| {
                (poly.coeff(i)
                    | poly.coeff(i + 162) << 1
                    | poly.coeff(i + 324) << 2
                    | poly.coeff(i + 486) << 3) as usize
            })
            .collect();
        for k in 0..4 {
            let base = 162 * k;
            for ii in 0..27 {
                // levels 0+1+2 with the level-3 twiddles folded in
                let mut six = [[0i16; 3]; 2];
                for r in 0..3 {
                    let (lo, hi) = (nib[ii + 27 * r], nib[ii + 27 * r + 81]);
                    for h in 0..2 {
                        let s2 = if r == 0 { 0 } else { h };
                        let x = Self::table(k, s2, r, 0, lo) as i32;
                        let y = Self::table(k, s2, r, 1, hi) as i32;
                        let val = if h == 0 { x + y } else { x - y };
                        six[h][r] = self.see(val);
                        if six[h][r].unsigned_abs() as i32 > self.max[0] {
                            self.max[0] = six[h][r].unsigned_abs() as i32;
                        }
                    }
                }
                // level 3 (only the omega multiply is left)
                for h in 0..2 {
                    let (o0, o1, o2) = self.r3(six[h][0], six[h][1], six[h][2]);
                    v[base + 81 * h + ii] = o0;
                    v[base + 81 * h + ii + 27] = o1;
                    v[base + 81 * h + ii + 54] = o2;
                }
            }
        }
        for x in v.iter() {
            self.max[1] = self.max[1].max(x.unsigned_abs() as i32);
        }
        for (li, level) in [4usize, 5, 6].into_iter().enumerate() {
            let n = DEGREE[level];
            let m = n / 3;
            for k in 0..SUBRINGS[level] {
                let b = k * n;
                let z = Params::<Q>::zeta(level, k);
                for i in 0..m {
                    let (o0, o1, o2) =
                        self.r3_tw(v[b + i], v[b + m + i], v[b + 2 * m + i], z, bar);
                    v[b + i] = o0;
                    v[b + m + i] = o1;
                    v[b + 2 * m + i] = o2;
                }
            }
            for x in v.iter() {
                self.max[2 + li] = self.max[2 + li].max(x.unsigned_abs() as i32);
            }
        }
        v
    }
}

// ------------------------------------------------------------------ main kernel test

fn check_kernel<const Q: u16>() {
    let bound = (vb::output_bound_milli_q(Q) as i64 * Q as i64 / 1000) as i32;
    let mut sh = Shadow::<Q>::new();
    let mut worst_out = 0i32;
    for polys in batches(64, 0xABCD ^ Q as u64) {
        let bb = unsafe { transpose::slice_polys_idx(&polys) };
        let mut out = Batch32::zero(Representation::Coefficients);
        unsafe { vb::ntt_bin_batch32::<Q>(&bb, &mut out) };
        assert_eq!(out.representation, Representation::Ntt);
        for p in 0..32 {
            let want = scalar::ntt::<Q>(&scalar::lift(&polys[p]));
            let e = out.get(p);
            let got = e.normalized(Q);
            for j in 0..N {
                assert_eq!(got[j], want[j], "q={Q} poly={p} slot={j}");
            }
            // exact agreement with the i32 shadow schedule (not just modulo q)
            let shadow = sh.run(&polys[p]);
            for j in 0..N {
                assert_eq!(e.v[j], shadow[j], "shadow mismatch q={Q} poly={p} slot={j}");
            }
            for j in 0..N {
                let a = (e.v[j] as i32).abs();
                worst_out = worst_out.max(a);
                assert!(a <= bound, "q={Q} output {a} exceeds declared bound {bound}");
            }
        }
    }
    let q = Q as f64;
    println!(
        "q={Q}: bounds (multiples of q) levels 0-2 {:.3}, l3 {:.3}, l4 {:.3}, l5 {:.3}, l6 {:.3}; \
         max intermediate {:.3}q = {}; declared output bound {:.3}q",
        sh.max[0] as f64 / q,
        sh.max[1] as f64 / q,
        sh.max[2] as f64 / q,
        sh.max[3] as f64 / q,
        sh.max[4] as f64 / q,
        sh.max_any as f64 / q,
        sh.max_any,
        bound as f64 / q
    );
    assert!(sh.max_any < 32768);
    assert_eq!(worst_out, sh.max[4]);
}

#[test]
fn kernel_3889() {
    check_kernel::<3889>();
}
#[test]
fn kernel_9721() {
    check_kernel::<9721>();
}

// ------------------------------------------------------------------ multiplication

fn check_mul<const Q: u16>() {
    let mut rng = Rng::new(77 ^ Q as u64);
    let pa: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let pb: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let (mut na, mut nb, mut nc) = (
        Batch32::zero(Representation::Ntt),
        Batch32::zero(Representation::Ntt),
        Batch32::zero(Representation::Ntt),
    );
    unsafe {
        vb::ntt_bin_batch32::<Q>(&transpose::slice_polys_idx(&pa), &mut na);
        vb::ntt_bin_batch32::<Q>(&transpose::slice_polys_idx(&pb), &mut nb);
        pointwise::mul_batch_batch::<Q>(&na, &nb, &mut nc);
    }
    for p in 0..32 {
        let prod = scalar::mul_mod_phi(&scalar::lift(&pa[p]), &scalar::lift(&pb[p]), Q);
        assert_eq!(nc.get(p).normalized(Q), scalar::ntt::<Q>(&prod), "batch*batch q={Q} p={p}");
    }
    // batch * single element
    let pe = BinaryPoly::random(&mut rng);
    let ne = {
        let polys: [BinaryPoly; 32] = std::array::from_fn(|_| pe);
        let mut b = Batch32::zero(Representation::Ntt);
        unsafe { vb::ntt_bin_batch32::<Q>(&transpose::slice_polys_idx(&polys), &mut b) };
        b.get(0)
    };
    let me = MontElement::new::<Q>(&ne);
    unsafe { pointwise::mul_batch_element::<Q>(&na, &me, &mut nc) };
    for p in 0..32 {
        let prod = scalar::mul_mod_phi(&scalar::lift(&pa[p]), &scalar::lift(&pe), Q);
        assert_eq!(nc.get(p).normalized(Q), scalar::ntt::<Q>(&prod), "batch*elem q={Q} p={p}");
    }
}

#[test]
fn mul_3889() {
    check_mul::<3889>();
}
#[test]
fn mul_9721() {
    check_mul::<9721>();
}

// ------------------------------------------------------------------ drivers

fn check_drivers<const Q: u16>() {
    let mut rng = Rng::new(2024 ^ Q as u64);
    let nb = 5;
    let polys: Vec<BinaryPoly> = (0..32 * nb).map(|_| BinaryPoly::random(&mut rng)).collect();
    let mut want: Vec<Batch32> = Vec::new();
    for b in 0..nb {
        let chunk: [BinaryPoly; 32] = std::array::from_fn(|i| polys[32 * b + i]);
        let mut o = Batch32::zero(Representation::Ntt);
        // exercise the BinaryBatch32 -> BinaryIndex32 adapter on this path
        let idx = BinaryIndex32::from_nibbles(&unsafe { transpose::slice_polys(&chunk) });
        unsafe { vb::ntt_bin_batch32::<Q>(&idx, &mut o) };
        want.push(o);
    }
    let mut got: Vec<Batch32> = (0..nb).map(|_| Batch32::zero(Representation::Ntt)).collect();
    vb::ntt_bin_polys::<Q>(&polys, &mut got);
    for b in 0..nb {
        assert!(got[b].v == want[b].v, "ntt_bin_polys batch {b}");
        assert_eq!(got[b].representation, Representation::Ntt);
    }
    let mut seen = 0usize;
    vb::ntt_bin_stream::<Q>(&polys, |i, batch| {
        assert!(batch.v == want[i].v, "ntt_bin_stream batch {i}");
        seen += 1;
    });
    assert_eq!(seen, nb);
}

#[test]
fn drivers_3889() {
    check_drivers::<3889>();
}
#[test]
fn drivers_9721() {
    check_drivers::<9721>();
}

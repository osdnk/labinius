//! Correctness and bound tests for the binary vertical NTT.
//!
//! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
//! element (`f162::pack4`) and sliced by the production front end, so the kernel is fed exactly
//! what a commitment feeds it.
use bin_ntt::f162;
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::transpose_f162::{self as tf, BinaryIndex32};
use bin_ntt::simd::ntt::bin_asm as vb;
use bin_ntt::simd::ntt::bin_large as vl;
use bin_ntt::ring::*;
use bin_ntt::F162;

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

/// The 128 `F162` of a batch, and the index rows the front end slices out of them.
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

/// Adversarial inputs: all-zero, all-ones, alternating patterns and single monomials at the
/// block boundaries of the tree.
fn adversarial() -> Vec<Bin> {
    let mut v = Vec::new();
    v.push([0u32; N]);
    v.push([1u32; N]);
    for phase in 0..2 {
        let mut alt = [0u32; N];
        for i in 0..N {
            alt[i] = (i % 2 == phase) as u32;
        }
        v.push(alt);
        let mut alt3 = [0u32; N];
        for i in 0..N {
            alt3[i] = (i % 3 == phase) as u32;
        }
        v.push(alt3);
    }
    // block-structured: the four 162-blocks the nibble index is built from
    for b in 0..4 {
        let mut p = [0u32; N];
        for i in 0..162 {
            p[i + 162 * b] = 1;
        }
        v.push(p);
    }
    for d in [
        0usize, 1, 80, 81, 161, 162, 163, 323, 324, 325, 485, 486, 646, 647,
    ] {
        v.push(monomial(d));
    }
    v
}

fn batches(count: usize, seed: u64) -> Vec<[Bin; 32]> {
    let mut rng = Rng::new(seed);
    let adv = adversarial();
    let mut out = Vec::new();
    // one batch made only of adversarial inputs (padded with zeros / repeats)
    let mut b0: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().take(32).enumerate() {
        b0[i] = *p;
    }
    out.push(b0);
    let mut b1: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().skip(32).enumerate() {
        b1[i] = *p;
    }
    // and one that mixes the extreme "all ones" with random
    for i in adv.len().saturating_sub(32)..32 {
        b1[i] = random_bin(&mut rng);
    }
    out.push(b1);
    for _ in 0..count {
        out.push(std::array::from_fn(|_| random_bin(&mut rng)));
    }
    out
}

// ------------------------------------------------------------------ the front end

/// The AVX-512 slicer against the scalar definition of the index rows.
#[test]
fn transpose_matches_scalar() {
    for polys in batches(64, 11) {
        let e = elems_of(&polys);
        let want = f162::index_rows_scalar(&e);
        let mut got = BinaryIndex32::zero();
        unsafe { tf::slice_f162_into(&e, &mut got) };
        for i in 0..162 {
            assert!(want.rows[i] == got.rows[i], "slice_f162_into row {i}");
        }
        // and the lift really is the coefficient vector the test built
        for p in 0..32 {
            assert_eq!(f162::lift_elem(&e, p), polys[p], "lift_elem {p}");
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
        Shadow {
            max: [0; 5],
            max_any: 0,
        }
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
        let (n0, n1, n2, n3) = (
            (n & 1) as u64,
            ((n >> 1) & 1) as u64,
            ((n >> 2) & 1) as u64,
            ((n >> 3) & 1) as u64,
        );
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
    /// `red`: 0 = no reduction of the un-twiddled a0, 1 = the lookup Barrett, 2 = the
    /// two-multiply `vpmulhrsw` one.
    fn r3_tw(&mut self, a0: i16, a1: i16, a2: i16, z: u16, red: u8) -> (i16, i16, i16) {
        let z2 = (z as u64 * z as u64 % Q as u64) as u16;
        let t1 = self.mont(a1, z);
        let t2 = self.mont(a2, z2);
        let a0 = match red {
            1 => self.see(vb::barrett_lut_i16(a0, Q) as i32),
            2 => self.see(barrett_i16(a0, Q) as i32),
            _ => a0,
        };
        self.r3(a0, t1, t2)
    }

    fn run(&mut self, poly: &Bin) -> [i16; N] {
        let bar = vb::needs_barrett(Q);
        let mut v = [0i16; N];
        let nib: Vec<usize> = (0..162)
            .map(|i| {
                (poly[i] | poly[i + 162] << 1 | poly[i + 324] << 2 | poly[i + 486] << 3) as usize
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
            let red = match (bar, vb::LUT_BARRETT_LEVELS[li], vb::MUL_BARRETT_LEVELS[li]) {
                (true, true, _) => 1,
                (true, _, true) => 2,
                _ => 0,
            };
            let n = DEGREE[level];
            let m = n / 3;
            for k in 0..SUBRINGS[level] {
                let b = k * n;
                let z = Params::<Q>::zeta(level, k);
                for i in 0..m {
                    let (o0, o1, o2) = self.r3_tw(v[b + i], v[b + m + i], v[b + 2 * m + i], z, red);
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
        let bb = idx_of(&polys);
        let mut out = Batch32::zero(Representation::Coefficients);
        unsafe { vb::ntt_bin_batch32::<Q>(&bb, &mut out) };
        assert_eq!(out.representation, Representation::Ntt);
        for p in 0..32 {
            let want = scalar::ntt::<Q>(&polys[p]);
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
                assert!(
                    a <= bound,
                    "q={Q} output {a} exceeds declared bound {bound}"
                );
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

// ------------------------------------------------------------------ the lookup Barrett

/// The reduction the kernel's `vpmultishiftqb` + `vpermb` + `vpaddw` sequence performs, checked
/// exhaustively over every i16 input: it subtracts a multiple of q and leaves |r| <= 0.579 q,
/// where the two-multiply `barrett_i16` only guarantees 0.809 q.
#[test]
fn barrett_lut_exhaustive() {
    for &q in QS.iter().chain(QS_LARGE.iter()) {
        let (mut worst, mut worst_mul) = (0i32, 0i32);
        for a in i16::MIN..=i16::MAX {
            let r = vb::barrett_lut_i16(a, q);
            // same residue class
            assert_eq!(
                (a as i32).rem_euclid(q as i32),
                (r as i32).rem_euclid(q as i32),
                "q={q} a={a}"
            );
            // the correction really is a multiple of q, and the i16 addition is the one the
            // kernel does — above 2^14, `-k q` itself can leave i16 (k reaches 2 at 17497) and
            // only the result has to land back inside it, which the residue check above pins
            let corr = vb::barrett_lut_corr(((a >> 11) & 31) as usize, q);
            assert_eq!((a as i32 - r as i32) % q as i32, 0);
            assert_eq!(a.wrapping_add(corr), r, "q={q} a={a}");
            worst = worst.max((r as i32).abs());
            worst_mul = worst_mul.max((barrett_i16(a, q) as i32).abs());
        }
        println!(
            "q={q}: lookup Barrett |r| <= {worst} = {:.4} q (two-multiply {worst_mul} = {:.4} q)",
            worst as f64 / q as f64,
            worst_mul as f64 / q as f64
        );
        assert!(worst < q as i32);
        if q == 9721 {
            assert_eq!(worst, 5625);
        }
        // what `ntt::bin_large` reduces to, and what its `barrett_lut_max` sweep returns
        if vl::is_large(q) {
            assert_eq!(worst, vl::barrett_lut_max(q));
            assert!(worst as f64 / q as f64 <= 0.532);
        }
    }
}

/// The *proved* bound of the schedule (the shadow model only sees the inputs the test feeds it):
/// worst case over all i16 lane values, level by level, with
/// `|mont(a, w)| <= (|a| (q-1)/2 + 2^15 q) / 2^16` and `|barrett| <= 5625 / 7864`.
#[test]
fn proved_bounds_9721() {
    const Q: i64 = 9721;
    let mont = |b: i64| (b * (Q - 1) / 2 + 32768 * Q) / 65536;
    // one radix-3 butterfly: a0 bounded by `a`, a1 / a2 by `b`; returns (y0, y1 = y2, max
    // intermediate).
    let bf = |a: i64, b: i64| {
        let m = mont(b);
        let u = mont(2 * m);
        (a + 2 * m, a + m + u, (2 * m).max(a + 2 * m).max(a + m + u))
    };
    let lut = 5625; // barrett_lut_exhaustive
    let mul = 7864; // params::barrett_i16, exhaustive (see the same test)
                    // levels 0-2: two centred table entries, |T| <= (q-1)/2
    let l2 = Q - 1;
    // level 3: the twiddles are folded, so y0 = a0 + t1 + t2 with all three <= l2
    let l3 = 3 * l2;
    // the level-3 intermediates: t1 - t2 <= 2 l2, y0 = a0 + t1 + t2 <= 3 l2, y1 = y2 <= 2 l2 + u
    assert!(l3.max(2 * l2).max(2 * l2 + mont(2 * l2)) < 32768);
    // level 4: lookup Barrett on a0
    let (l4y0, l4y1, mi4) = bf(lut, l3);
    // level 5: unreduced; the three 9-blocks of a 27-block hold y0-, y1- and y2-values, so each
    // butterfly's three inputs are all of one kind
    let (l5y0a, l5y1a, mi5a) = bf(l4y0, l4y0);
    let (l5y0b, l5y1b, mi5b) = bf(l4y1, l4y1);
    let (l5y0, l5y1) = (l5y0a.max(l5y0b), l5y1a.max(l5y1b));
    // level 6: two-multiply Barrett on a0 (which is a level-5 y0), a1 / a2 are level-5 y1 / y2
    let (l6y0, l6y1, mi6) = bf(mul, l5y1);
    let out = l6y0.max(l6y1);
    let declared = vb::output_bound_milli_q(9721) as i64 * Q / 1000;
    println!(
        "q=9721 proved: l3 {:.3}q  l4 {:.3}q  l5 {:.3}q  out {:.4}q = {out}  (declared {:.3}q);\
         \n  max intermediate {} of 32767, i.e. {} to spare",
        l3 as f64 / Q as f64,
        l4y0 as f64 / Q as f64,
        l5y0 as f64 / Q as f64,
        out as f64 / Q as f64,
        declared as f64 / Q as f64,
        mi4.max(mi5a).max(mi5b).max(mi6),
        32767 - mi4.max(mi5a).max(mi5b).max(mi6)
    );
    for m in [mi4, mi5a, mi5b, mi6] {
        assert!(m < 32768, "i16 overflow in the proved bound: {m}");
    }
    assert!(
        out <= declared,
        "output {out} exceeds the declared bound {declared}"
    );
    // and the reason level 5 can be skipped at all: with the two-multiply Barrett at level 4 it
    // could not.
    let (w0, _, _) = bf(mul, l3);
    assert!(
        bf(w0, w0).2 >= 32768,
        "the level-5 reduction would not have been needed"
    );
}

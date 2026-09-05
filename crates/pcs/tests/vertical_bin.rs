//! The pure-intrinsics reference kernel (`simd::ntt::bin_small`) against the generated one
//! (`simd::ntt::bin_asm`) and against the scalar NTT.
//!
//! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
//! element (`f162::pack4`) and sliced by the production front end, so both kernels are fed
//! exactly what a commitment feeds them.
//!
//! For q = 3889 neither kernel reduces at all and they run the same operation order per
//! butterfly, so the two outputs are **bit-identical**. For q = 9721 they reduce at different
//! levels and with a different Barrett (reference: `params::barrett_i16` at levels 4, 5 and 6;
//! asm: the lookup Barrett at level 4 and `barrett_i16` at level 6), so they agree only modulo q.
use bin_ntt::f162;
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::transpose_f162::{self as tf, BinaryIndex32};
use bin_ntt::simd::ntt::bin_small as vb;
use bin_ntt::simd::ntt::bin_asm as vba;
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

fn idx_of(elems: &[F162; 128]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    unsafe { tf::slice_f162_into(elems, &mut out) };
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

// ------------------------------------------------------------------ the two kernels

/// `EXACT`: the two kernels must agree lane by lane in i16 (q = 3889, where neither reduces);
/// otherwise only modulo q.
fn check_against_asm<const Q: u16, const EXACT: bool>() {
    for polys in batches(64, 0x5EED ^ Q as u64) {
        let idx = idx_of(&elems_of(&polys));
        let mut got = Batch32::zero(Representation::Coefficients);
        let mut want = Batch32::zero(Representation::Coefficients);
        unsafe {
            vb::ntt_bin_batch32::<Q>(&idx, &mut got);
            vba::ntt_bin_batch32::<Q>(&idx, &mut want);
        }
        assert_eq!(got.representation, Representation::Ntt);
        for j in 0..N {
            for p in 0..32 {
                let (a, b) = (got.v[j][p] as i32, want.v[j][p] as i32);
                if EXACT {
                    assert_eq!(a, b, "q={Q} slot={j} poly={p}");
                } else {
                    assert_eq!((a - b) % Q as i32, 0, "q={Q} slot={j} poly={p}: {a} vs {b}");
                }
            }
        }
    }
}

#[test]
fn matches_asm_3889() {
    check_against_asm::<3889, true>();
}

#[test]
fn matches_asm_9721() {
    check_against_asm::<9721, false>();
}

// ------------------------------------------------------------------ the scalar reference

/// Output against `scalar::ntt` of the lift of the very `F162` elements the front end sliced,
/// plus the declared output bound.
fn check_against_scalar<const Q: u16>() {
    let bound = (vb::output_bound_milli_q(Q) as i64 * Q as i64 / 1000) as i32;
    let mut worst = 0i32;
    for polys in batches(64, 0xABCD ^ Q as u64) {
        let elems = elems_of(&polys);
        let idx = idx_of(&elems);
        let mut out = Batch32::zero(Representation::Coefficients);
        unsafe { vb::ntt_bin_batch32::<Q>(&idx, &mut out) };
        for p in 0..32 {
            let coeffs: [F162; 4] = elems[4 * p..4 * p + 4].try_into().unwrap();
            let coeffs = f162::lift4(&coeffs);
            assert_eq!(coeffs, polys[p], "lift4 {p}");
            let want = scalar::ntt::<Q>(&coeffs);
            let e = out.get(p);
            let got = e.normalized(Q);
            for j in 0..N {
                assert_eq!(got[j], want[j], "q={Q} poly={p} slot={j}");
                let a = (e.v[j] as i32).abs();
                worst = worst.max(a);
                assert!(
                    a <= bound,
                    "q={Q} output {a} exceeds declared bound {bound}"
                );
            }
        }
    }
    println!(
        "q={Q}: reference kernel max |output| {worst} = {:.3}q (declared {:.3}q)",
        worst as f64 / Q as f64,
        bound as f64 / Q as f64
    );
}

#[test]
fn matches_scalar_3889() {
    check_against_scalar::<3889>();
}

#[test]
fn matches_scalar_9721() {
    check_against_scalar::<9721>();
}

#![allow(dead_code)]
use bin_ntt::f162;
use bin_ntt::params::N;
use bin_ntt::rng::Rng;
use bin_ntt::simd::transpose_f162::{self as tf, BinaryIndex32};
use bin_ntt::F162;

/// The 648 binary coefficients of one ring element.
pub type Bin = [u32; N];

pub fn monomial(d: usize) -> Bin {
    let mut c = [0u32; N];
    c[d] = 1;
    c
}

pub fn random_bin(rng: &mut Rng) -> Bin {
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
pub fn elems_of(polys: &[Bin; 32]) -> [F162; 128] {
    let mut e = [F162([0; 3]); 128];
    for p in 0..32 {
        e[4 * p..4 * p + 4].copy_from_slice(&f162::pack4(&polys[p]));
    }
    e
}

pub fn idx_of(elems: &[F162; 128]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    unsafe { tf::slice_f162_into(elems, &mut out) };
    out
}

pub fn idx_of_polys(polys: &[Bin; 32]) -> BinaryIndex32 {
    idx_of(&elems_of(polys))
}

/// Adversarial inputs: all-zero, all-ones, alternating patterns and single monomials at the
/// block boundaries of the tree.
fn adversarial_at(boundaries: [usize; 14]) -> Vec<Bin> {
    let mut v = vec![[0u32; N], [1u32; N]];
    for phase in 0..2 {
        v.push(core::array::from_fn(|i| (i % 2 == phase) as u32));
        v.push(core::array::from_fn(|i| (i % 3 == phase) as u32));
    }
    // block-structured: the four 162-blocks the nibble index is built from
    for b in 0..4 {
        let mut p = [0u32; N];
        for i in 0..162 {
            p[i + 162 * b] = 1;
        }
        v.push(p);
    }
    for d in boundaries {
        v.push(monomial(d));
    }
    v
}

pub fn adversarial() -> Vec<Bin> {
    adversarial_at([
        0, 1, 80, 81, 161, 162, 163, 323, 324, 325, 485, 486, 646, 647,
    ])
}

pub fn adversarial_quad() -> Vec<Bin> {
    adversarial_at([
        0, 1, 53, 54, 107, 108, 161, 162, 323, 324, 485, 486, 646, 647,
    ])
}

fn batches_of(adv: Vec<Bin>, count: usize, seed: u64) -> Vec<[Bin; 32]> {
    let mut rng = Rng::new(seed);
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
        out.push(core::array::from_fn(|_| random_bin(&mut rng)));
    }
    out
}

pub fn batches(count: usize, seed: u64) -> Vec<[Bin; 32]> {
    batches_of(adversarial(), count, seed)
}

pub fn batches_quad(count: usize, seed: u64) -> Vec<[Bin; 32]> {
    batches_of(adversarial_quad(), count, seed)
}

pub fn small() -> bin_ntt::Params {
    bin_ntt::Params::new(
        11,
        3,
        vec![bin_ntt::Modulus::Q9721_FS_S],
        bin_ntt::Opening::Clear,
    )
    .unwrap()
}

/// A limb that is not `base`, so that a round has two of them.
pub fn second(base: bin_ntt::Modulus) -> bin_ntt::Modulus {
    if base == bin_ntt::Modulus::Q9721_FS_S {
        bin_ntt::Modulus::Q3889_FS_S
    } else {
        bin_ntt::Modulus::Q9721_FS_S
    }
}

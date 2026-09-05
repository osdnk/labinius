//! The per-modulus table of the README.
//!
//! `the_moduli_quantified` is the wall clock one extra modulus adds to `commit` at the
//! `Params::basic()` shape (2^18 `F162` in 256 columns), median of 15.
//!
//! `cycles_probe` is the cycle columns. It runs one phase of one modulus `REPS` times and does
//! nothing else, so `perf stat -e cycles` counts that phase plus a fixed startup that a second
//! run at `REPS=0` subtracts:
//!
//! ```text
//! PROBE=transform:19441 REPS=200000 taskset -c 3 perf stat -e cycles \
//!     ./target/release/calibrate moduli cycles_probe
//! ```
//!
//! One transform and one base multiplication are a batch of 32 ring elements; one fold-down is a
//! column, which at this shape is 256.
//!
//! `the_fold_per_base` is the wall clock of `Prover::fold` at the same shape with each modulus in
//! turn as the base limb.
use bin_ntt::Opening;
use bin_ntt::api::{AuxData, CommitmentKey, BASE_PRIME};
use bin_ntt::params::N;
use bin_ntt::simd::commit as cm;
use bin_ntt::simd::transpose_f162::{slice_f162_into, BinaryIndex32};
use bin_ntt::simd::{vertical_bin_asm as vb, vertical_bin_large as vl, vertical_bin_quad as vq};
use bin_ntt::types::{Batch32, Representation};
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness, F162};
use std::time::Instant;

const COLUMNS: usize = 256;
const F162_PER_COLUMN: usize = 1024;
const RING_PER_COLUMN: usize = F162_PER_COLUMN / 4;

fn witness(n: usize) -> Vec<F162> {
    bin_ntt::f162::random_elems(n, 0x243F_6A88)
}

fn index() -> BinaryIndex32 {
    let w = witness(128);
    let mut idx = BinaryIndex32::zero();
    unsafe { slice_f162_into(<&[F162; 128]>::try_from(&w[..]).unwrap(), &mut idx) };
    idx
}

fn transform_once(idx: &BinaryIndex32, out: &mut Batch32, q: u16, quad: bool) {
    unsafe {
        match (q, quad) {
            (3889, false) => vb::ntt_bin_batch32::<3889>(idx, out),
            (9721, false) => vb::ntt_bin_batch32::<9721>(idx, out),
            (2917, true) => vq::ntt_quad_bin_batch32::<2917>(idx, out),
            (4861, true) => vq::ntt_quad_bin_batch32::<4861>(idx, out),
            (12637, true) => vq::ntt_quad_bin_batch32::<12637>(idx, out),
            (17497, false) => vl::ntt_bin_batch32::<17497>(idx, out),
            _ => vl::ntt_bin_batch32::<19441>(idx, out),
        }
    }
}

/// The median of an odd-length sample of wall milliseconds.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn commit_ms(extra: &[Modulus]) -> f64 {
    let w = witness(COLUMNS * F162_PER_COLUMN);
    let key = CommitmentKey::random(F162_PER_COLUMN, 0xA11CE, Modulus::BASE, extra);
    let mut aux = AuxData::new(RING_PER_COLUMN, COLUMNS, key.limbs());
    let mut samples = Vec::with_capacity(15);
    for _ in 0..15 {
        let t = Instant::now();
        let m = key.commit_into_aux(&w, COLUMNS, &mut aux);
        samples.push(t.elapsed().as_secs_f64() * 1e3);
        core::hint::black_box(m.get(0, 0).limbs[0].v[0]);
    }
    median(samples)
}

/// Every modulus that is not the default base, i.e. every one it can be given as an extra limb.
fn extras() -> Vec<Modulus> {
    Modulus::ALL
        .into_iter()
        .filter(|l| *l != Modulus::BASE)
        .collect()
}

pub fn the_moduli_quantified() {
    let base = commit_ms(&[]);
    println!("\n  {BASE_PRIME} (base) alone: {base:.2} ms");
    for l in extras() {
        println!("  + {:>5}: {:+.2} ms", l.prime(), commit_ms(&[l]) - base);
    }
    println!(
        "  all six extra moduli together: {:.1} ms",
        commit_ms(&extras()) - base
    );
}

fn transform(reps: usize, q: u16, quad: bool) {
    let idx = index();
    let mut out = Batch32::zero(Representation::Ntt);
    for _ in 0..reps {
        transform_once(&idx, &mut out, q, quad);
    }
    core::hint::black_box(out.v[0][0]);
}

fn basemul(reps: usize, q: u16, quad: bool) {
    let a = Batch32::zero(Representation::Ntt);
    let w = Batch32::zero(Representation::Ntt);
    let wp = w.v.as_ptr() as *const i16;
    let ap = a.v.as_ptr() as *const i16;
    let apf = a.v.as_ptr() as *const i8;
    if quad {
        let mut acc = cm::QuadAcc::zero();
        let (p01, p2) = (
            acc.p01.as_mut_ptr() as *mut i32,
            acc.p2.as_mut_ptr() as *mut i32,
        );
        for _ in 0..reps {
            unsafe {
                match q {
                    2917 => cm::mac_quad_batch::<2917, false>(wp, ap, apf, p01, p2),
                    4861 => cm::mac_quad_batch::<4861, false>(wp, ap, apf, p01, p2),
                    _ => cm::mac_quad_batch::<12637, false>(wp, ap, apf, p01, p2),
                }
            }
        }
        core::hint::black_box(acc.p01[0][0]);
    } else {
        let mut acc = cm::Acc::zero();
        let p = acc.v.as_mut_ptr() as *mut i32;
        for _ in 0..reps {
            unsafe { cm::mac_batch::<false>(wp, ap, apf, p) };
        }
        core::hint::black_box(acc.v[0][0]);
    }
}

fn folddown(reps: usize, q: u16, quad: bool) {
    if quad {
        let acc = cm::QuadAcc::zero();
        for _ in 0..reps {
            let y = match q {
                2917 => cm::finish_quad::<2917>(&acc),
                4861 => cm::finish_quad::<4861>(&acc),
                _ => cm::finish_quad::<12637>(&acc),
            };
            core::hint::black_box(y[0]);
        }
    } else {
        let acc = cm::Acc::zero();
        for _ in 0..reps {
            let y = match q {
                3889 => cm::finish::<3889>(&acc),
                9721 => cm::finish::<9721>(&acc),
                17497 => cm::finish::<17497>(&acc),
                _ => cm::finish::<19441>(&acc),
            };
            core::hint::black_box(y[0]);
        }
    }
}

pub fn cycles_probe() {
    let Ok(probe) = std::env::var("PROBE") else {
        return;
    };
    let reps: usize = std::env::var("REPS").unwrap().parse().unwrap();
    let (phase, q) = probe.split_once(':').unwrap();
    let q: u16 = q.parse().unwrap();
    let quad = Modulus::from_prime(q).is_some_and(|l| l.is_quadratic());
    match phase {
        "transform" => transform(reps, q, quad),
        "basemul" => basemul(reps, q, quad),
        "folddown" => folddown(reps, q, quad),
        _ => panic!("no phase {phase}"),
    }
}

/// The transform's output for every prime, as a fingerprint of the kernels: any change to a
/// schedule that is not value-preserving shows up here.
pub fn kernel_fingerprints() {
    let idx = index();
    let mut out = Batch32::zero(Representation::Ntt);
    let mut primes: Vec<(u16, bool)> = Modulus::ALL
        .iter()
        .map(|l| (l.prime(), l.is_quadratic()))
        .collect();
    primes.sort();
    for (q, quad) in primes {
        transform_once(&idx, &mut out, q, quad);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for j in 0..N {
            for p in 0..32 {
                h = (h ^ (out.v[j][p] as u16 as u64)).wrapping_mul(0x100_0000_01b3);
            }
        }
        println!("  {q}: {h:016x}");
    }
}

/// One fold at the `Params::basic()` shape over `base`, the median of `reps`, in milliseconds.
fn fold_ms(base: Modulus, extra: Vec<Modulus>, reps: usize) -> f64 {
    let params = Params::with_base(18, 8, base, extra, Opening::Clear).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [0x5A; 32]);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let w = Witness::random(&params, [0xC7; 32]);
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (commitment, opening) = prover.commit(&w);
        let mut transcript = Transcript::new(b"bin-ntt/bench/fold");
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        let row = w.row_evaluate(&point);
        let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
        let start = Instant::now();
        let folded = prover.fold(opening, &challenges);
        samples.push(start.elapsed().as_secs_f64() * 1e3);
        core::hint::black_box(folded.elements()[0].v[0]);
    }
    median(samples)
}

/// The fold's wall clock at the basic shape with each modulus as the base limb. The 85 MB stream
/// and the one `vpmaddwd` per slot vector are the same for all seven; what separates them is the
/// fold-back period their `|W| |c|` allows.
pub fn the_fold_per_base() {
    println!();
    println!(
        "  3889 with the basic limb list: {:.2} ms",
        fold_ms(Modulus::Q3889_FS_S, vec![Modulus::Q9721_FS_S], 5)
    );
    for base in Modulus::ALL {
        println!(
            "  base {:>5} alone: {:.2} ms",
            base.prime(),
            fold_ms(base, vec![], 5)
        );
    }
}

pub fn run(which: Option<&str>) {
    match which {
        Some("the_moduli_quantified") => the_moduli_quantified(),
        Some("cycles_probe") => cycles_probe(),
        Some("kernel_fingerprints") => kernel_fingerprints(),
        Some("the_fold_per_base") => the_fold_per_base(),
        None => {
            the_moduli_quantified();
            cycles_probe();
            kernel_fingerprints();
            the_fold_per_base();
        }
        Some(other) => panic!("unknown case {other}"),
    }
}

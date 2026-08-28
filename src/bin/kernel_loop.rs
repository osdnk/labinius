//! Runs one kernel in a tight loop on a cache-resident batch so that whole-process profilers
//! (`perf stat`, `perf record`) see almost nothing but the kernel. Usage:
//! `kernel_loop <3889|9721|2917|4861|12637> <bin|gen|hgen|quad_bin|quad_gen> <iterations>`; pin
//! externally with `taskset`. The `quad_*` kinds are the quadratic-slot kernels and take one of
//! the three `QS_QUAD` primes; the others take one of the two splitting primes.
use bin_ntt::rng::Rng;
use bin_ntt::simd::horizontal_gen as hg;
use bin_ntt::simd::transpose;
use bin_ntt::simd::{vertical_bin as vb, vertical_bin_quad as vq, vertical_gen as vg};
use bin_ntt::simd::vertical_gen_quad as vgq;
use bin_ntt::types::*;

/// The quadratic-slot kernels (`params::QS_QUAD`), which have no 1944-th root of unity and so
/// cannot instantiate the splitting kernels at all.
fn run_quad<const Q: u16>(kind: &str, iters: usize) {
    let mut rng = Rng::new(1);
    let polys: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let mut sink = 0i16;
    match kind {
        "quad_bin" => {
            let idx = unsafe { transpose::slice_polys_idx(&polys) };
            let mut out = Batch32::zero(Representation::Ntt);
            for _ in 0..iters {
                unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
                sink ^= out.v[0][0];
            }
        }
        "quad_gen" => {
            let mut b = Batch32::from_binary(&polys);
            for _ in 0..iters {
                unsafe { vgq::ntt_quad_gen_batch32::<Q>(&mut b) };
                sink ^= b.v[0][0];
            }
        }
        _ => panic!("kind must be quad_bin | quad_gen for q in QS_QUAD"),
    }
    std::hint::black_box(sink);
}

fn run<const Q: u16>(kind: &str, iters: usize) {
    let mut rng = Rng::new(1);
    let polys: [BinaryPoly; 32] = std::array::from_fn(|_| BinaryPoly::random(&mut rng));
    let mut sink = 0i16;
    match kind {
        "bin" => {
            let idx = unsafe { transpose::slice_polys_idx(&polys) };
            let mut out = Batch32::zero(Representation::Ntt);
            for _ in 0..iters {
                unsafe { vb::ntt_bin_batch32::<Q>(&idx, &mut out) };
                sink ^= out.v[0][0];
            }
        }
        "gen" => {
            let pristine = Batch32::from_binary(&polys);
            let mut b = pristine.clone();
            for _ in 0..iters {
                // in place: the output is fed back in (data-independent instruction stream)
                unsafe { vg::ntt_gen_batch32::<Q>(&mut b) };
                sink ^= b.v[0][0];
            }
        }
        "hgen" => {
            let mut b = hg::HBatch4::from_binary(&polys[..4].try_into().unwrap());
            for _ in 0..iters * 8 {
                unsafe { hg::ntt_gen_hbatch4::<Q>(&mut b) };
                sink ^= b.v[0][0];
            }
        }
        _ => panic!("kind must be bin | gen | hgen"),
    }
    std::hint::black_box(sink);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q: u16 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3889);
    let kind = a.get(2).map(String::as_str).unwrap_or("bin");
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(20000);
    match q {
        3889 => run::<3889>(kind, iters),
        9721 => run::<9721>(kind, iters),
        2917 => run_quad::<2917>(kind, iters),
        4861 => run_quad::<4861>(kind, iters),
        12637 => run_quad::<12637>(kind, iters),
        _ => panic!("q must be 3889, 9721 (split) or 2917, 4861, 12637 (quadratic slots)"),
    }
}

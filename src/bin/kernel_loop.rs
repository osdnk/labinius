//! Runs one kernel in a tight loop on a cache-resident batch so that whole-process profilers
//! (`perf stat`, `perf record`) see almost nothing but the kernel. Usage:
//! `kernel_loop <3889|9721> <bin|gen|hgen> <iterations>`; pin externally with `taskset`.
use bin_ntt::rng::Rng;
use bin_ntt::simd::horizontal_gen as hg;
use bin_ntt::simd::transpose;
use bin_ntt::simd::{vertical_bin as vb, vertical_gen as vg};
use bin_ntt::types::*;

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
        _ => panic!("q must be 3889 or 9721"),
    }
}

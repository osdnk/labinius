//! Benchmarks for the hand-scheduled (asm tail) binary vertical NTT. Pin with `taskset -c 4`.
//!
//! Prints, per polynomial, cycles / instructions / uops / p0 / p5 for the cache-resident kernel of
//! both `vertical_bin` (the intrinsics baseline) and `vertical_bin_asm`, then the 2^18
//! materialised headline for the asm kernel.
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::transpose::{self, BinaryIndex32};
use bin_ntt::simd::vertical_bin as vb;
use bin_ntt::simd::vertical_bin_asm as va;
use bin_ntt::types::*;
use std::time::Instant;

fn report(name: &str, c: Counts, polys: u64, ms: Option<f64>) {
    let n = polys as f64;
    print!(
        "{name:<34} {:7.1} cyc {:7.1} ins {:7.1} uops  p0 {:6.1}  p1 {:5.1}  p5 {:6.1}",
        c.cycles as f64 / n,
        c.instructions as f64 / n,
        c.uops as f64 / n,
        c.port0 as f64 / n,
        c.port1 as f64 / n,
        c.port5 as f64 / n
    );
    if let Some(ms) = ms {
        print!("  {ms:8.2} ms");
    }
    println!();
}

const NB: usize = 8; // 256 polynomials: input 83 KB, output 331 KB -> L2 resident
const BIG: usize = 1 << 18;
const BIGB: usize = BIG / 32;

fn main() {
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");
    let mut rng = Rng::new(0xC0FFEE);

    let polys: Vec<BinaryPoly> = (0..32 * NB).map(|_| BinaryPoly::random(&mut rng)).collect();
    let idxs: Vec<BinaryIndex32> = (0..NB)
        .map(|b| unsafe {
            transpose::slice_polys_idx(&*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]))
        })
        .collect();
    let mut outs: Vec<Batch32> = (0..NB).map(|_| Batch32::zero(Representation::Ntt)).collect();

    macro_rules! kernel_bench {
        ($m:ident, $q:expr, $name:expr) => {{
            for _ in 0..20 {
                for b in 0..NB {
                    unsafe { $m::ntt_bin_batch32::<$q>(&idxs[b], &mut outs[b]) };
                }
            }
            const REP: usize = 200;
            let mut best: Option<Counts> = None;
            for _ in 0..5 {
                pg.start();
                for _ in 0..REP {
                    for b in 0..NB {
                        unsafe { $m::ntt_bin_batch32::<$q>(&idxs[b], &mut outs[b]) };
                    }
                }
                let c = pg.stop();
                if best.as_ref().map_or(true, |x| c.cycles < x.cycles) {
                    best = Some(c);
                }
            }
            report($name, best.unwrap(), (REP * NB * 32) as u64, None);
        }};
    }
    kernel_bench!(vb, 3889, "base vertical_bin q=3889");
    kernel_bench!(va, 3889, "asm  vertical_bin q=3889");
    kernel_bench!(vb, 9721, "base vertical_bin q=9721");
    kernel_bench!(va, 9721, "asm  vertical_bin q=9721");
    std::hint::black_box(&outs);

    // ---------------------------------------------------------------- 2^18 polynomials
    let big: Vec<BinaryPoly> = (0..BIG).map(|_| BinaryPoly::random(&mut rng)).collect();
    let mut bigout: Vec<Batch32> = (0..BIGB).map(|_| Batch32::zero(Representation::Ntt)).collect();
    println!();

    macro_rules! big_bench {
        ($q:expr) => {{
            va::ntt_bin_polys::<$q>(&big[..32 * 64], &mut bigout[..64]);
            let mut best: Option<(Counts, f64)> = None;
            for _ in 0..3 {
                let t0 = Instant::now();
                pg.start();
                va::ntt_bin_polys::<$q>(&big, &mut bigout);
                let c = pg.stop();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(&bigout);
                if best.as_ref().map_or(true, |x| c.cycles < x.0.cycles) {
                    best = Some((c, ms));
                }
            }
            let (c, ms) = best.unwrap();
            report(concat!("asm 2^18 materialised q=", $q), c, BIG as u64, Some(ms));
        }};
    }
    big_bench!(3889);
    big_bench!(9721);
    drop(bigout);
}

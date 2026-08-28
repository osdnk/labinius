//! Benchmarks for the binary vertical NTT. Pin with `taskset -c 2`.
//!
//! The static tables at the bottom were produced from
//! `objdump -d --no-show-raw-insn target/release/bench_vertical_bin`, by summing the instructions
//! of each loop body of `ntt_bin_batch32` / `slice_polys_idx_into` weighted by its trip count and
//! classifying by port with the table measured by `tools/ubench` and perf on this core, of which
//! three entries matter most here:
//!   * `vpermw` zmm is 2 uops (p0 + p5), hence the kernel uses `vpermb` (1 uop, p5) on
//!     byte-split tables;
//!   * `kmovq k, m64` is 1 uop on p5, so it competes with the shuffles, hence the transpose's last
//!     phase avoids mask registers entirely;
//!   * `vpbroadcastd zmm, m32` is a pure load (0 p0/p5 uops).
//! `cycles >= max(p0, p5, (p0 + p5 + p05)/2)`.
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::transpose::{self, BinaryIndex32};
use bin_ntt::simd::vertical_bin as vb;
use bin_ntt::types::*;
use std::time::Instant;

fn report(name: &str, c: Counts, polys: u64, ms: Option<f64>) {
    let n = polys as f64;
    print!(
        "{name:<32} {:7.1} cyc {:7.1} ins {:7.1} uops  p0 {:6.1}  p1 {:5.1}  p5 {:6.1}",
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

struct Static {
    name: &'static str,
    instrs: u64,
    p0: u64,
    p5: u64,
    p05: u64,
}

/// Static (objdump) counts per batch of 32 polynomials.
const STATIC: [Static; 3] = [
    Static { name: "kernel q=3889", instrs: 23363, p0: 6480, p5: 1080, p05: 8856 },
    Static { name: "kernel q=9721", instrs: 25317, p0: 7776, p5: 1080, p05: 9504 },
    Static { name: "slice_polys_idx", instrs: 1779, p0: 206, p5: 826, p05: 81 },
];

fn print_static() {
    println!("\nstatic count from objdump (per polynomial), port floor = max(p0, p5, sum/2):");
    for s in STATIC.iter() {
        let floor = (s.p0.max(s.p5)).max((s.p0 + s.p5 + s.p05).div_ceil(2));
        println!(
            "{:<32} {:7.1} ins  p0-only {:6.1}  p5-only {:6.1}  p0/p5 {:6.1}   floor {:6.1} cyc",
            s.name,
            s.instrs as f64 / 32.0,
            s.p0 as f64 / 32.0,
            s.p5 as f64 / 32.0,
            s.p05 as f64 / 32.0,
            floor as f64 / 32.0
        );
    }
}

const NB: usize = 8; // 256 polynomials: input 83 KB, output 331 KB -> L2 resident
const BIG: usize = 1 << 18;
const BIGB: usize = BIG / 32;

fn main() {
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");
    let mut rng = Rng::new(0xC0FFEE);

    // ---------------------------------------------------------------- (a) kernel, cache resident
    let polys: Vec<BinaryPoly> = (0..32 * NB).map(|_| BinaryPoly::random(&mut rng)).collect();
    let idxs: Vec<BinaryIndex32> = (0..NB)
        .map(|b| unsafe {
            transpose::slice_polys_idx(&*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]))
        })
        .collect();
    let mut outs: Vec<Batch32> = (0..NB).map(|_| Batch32::zero(Representation::Ntt)).collect();

    macro_rules! kernel_bench {
        ($q:expr, $name:expr) => {{
            for _ in 0..20 {
                for b in 0..NB {
                    unsafe { vb::ntt_bin_batch32::<$q>(&idxs[b], &mut outs[b]) };
                }
            }
            const REP: usize = 200;
            pg.start();
            for _ in 0..REP {
                for b in 0..NB {
                    unsafe { vb::ntt_bin_batch32::<$q>(&idxs[b], &mut outs[b]) };
                }
            }
            let c = pg.stop();
            report($name, c, (REP * NB * 32) as u64, None);
        }};
    }
    kernel_bench!(3889, "kernel q=3889 (256 polys, L2)");
    kernel_bench!(9721, "kernel q=9721 (256 polys, L2)");
    std::hint::black_box(&outs);

    // ---------------------------------------------------------------- (b) transpose
    let mut idx_out = BinaryIndex32::zero();
    for _ in 0..50 {
        for b in 0..NB {
            unsafe {
                transpose::slice_polys_idx_into(
                    &*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]),
                    &mut idx_out,
                )
            };
        }
    }
    const TREP: usize = 500;
    pg.start();
    for _ in 0..TREP {
        for b in 0..NB {
            unsafe {
                transpose::slice_polys_idx_into(
                    &*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]),
                    &mut idx_out,
                )
            };
        }
    }
    let c = pg.stop();
    std::hint::black_box(&idx_out);
    report("slice_polys_idx only", c, (TREP * NB * 32) as u64, None);

    // ---------------------------------------------------------------- (c)/(d) 2^18 polynomials
    let big: Vec<BinaryPoly> = (0..BIG).map(|_| BinaryPoly::random(&mut rng)).collect();
    let mut bigout: Vec<Batch32> = (0..BIGB).map(|_| Batch32::zero(Representation::Ntt)).collect();
    println!();

    macro_rules! big_bench {
        ($q:expr) => {{
            vb::ntt_bin_polys::<$q>(&big[..32 * 64], &mut bigout[..64]);
            let t0 = Instant::now();
            pg.start();
            vb::ntt_bin_polys::<$q>(&big, &mut bigout);
            let c = pg.stop();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&bigout);
            report(concat!("2^18 materialised q=", $q), c, BIG as u64, Some(ms));

            let mut acc = 0i64;
            vb::ntt_bin_stream::<$q>(&big[..32 * 64], |_, b| acc += b.v[0][0] as i64);
            let t0 = Instant::now();
            pg.start();
            vb::ntt_bin_stream::<$q>(&big, |i, b| {
                acc ^= b.v[i & 511][0] as i64 ^ b.v[647][31] as i64;
            });
            let c = pg.stop();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(acc);
            report(concat!("2^18 streamed     q=", $q), c, BIG as u64, Some(ms));
        }};
    }
    big_bench!(3889);
    big_bench!(9721);
    drop(bigout);
    print_static();
}

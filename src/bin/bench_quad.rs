//! Benchmark for the quadratic-slot kernels (q in `QS_QUAD` = 2917, 4861, 12637), printed next to
//! the splitting kernels of 3889 and 9721 so that a ranking of all five primes can be read off.
//! Pin with `taskset -c 2`.
//!
//! Three measurements per prime: the binary kernel on a cache-resident batch, the generic kernel
//! on a cache-resident batch, and the 2^18-`F162` materialised transform (85 MB of output per
//! prime, non-temporal stores).
use bin_ntt::f162;
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::transpose::{self, BinaryIndex32};
use bin_ntt::simd::{ntt_f162, vertical_bin_asm as va, vertical_bin_quad as vq};
use bin_ntt::simd::{vertical_gen as vg, vertical_gen_quad as vgq};
use bin_ntt::types::*;
use bin_fields::scalar::F162;
use std::time::Instant;

const NB: usize = 8; // 256 ring elements: input 83 KB, output 331 KB -> L2 resident
const BIG: usize = 1 << 18; // F162 elements = 2^16 ring elements
const BIGB: usize = BIG / 128;

struct Row {
    q: u16,
    tree: &'static str,
    bin: Counts,
    gen: Counts,
    big: (Counts, f64),
}

fn per(c: Counts, n: f64) -> (f64, f64, f64, f64, f64) {
    (
        c.cycles as f64 / n,
        c.instructions as f64 / n,
        c.uops as f64 / n,
        c.port0 as f64 / n,
        c.port5 as f64 / n,
    )
}

fn report(name: &str, c: Counts, elems: f64, ms: Option<f64>) {
    let (cy, ins, uops, p0, p5) = per(c, elems);
    print!("{name:<30} {cy:7.1} cyc {ins:7.1} ins {uops:7.1} uops  p0 {p0:6.1}  p5 {p5:6.1}");
    if let Some(ms) = ms {
        print!("   {ms:7.2} ms");
    }
    println!();
}

/// Best of `rounds` runs of `f`, by cycles.
fn best(pg: &PerfGroup, rounds: usize, mut f: impl FnMut()) -> Counts {
    let mut b: Option<Counts> = None;
    for _ in 0..rounds {
        pg.start();
        f();
        let c = pg.stop();
        if b.as_ref().map_or(true, |x| c.cycles < x.cycles) {
            b = Some(c);
        }
    }
    b.unwrap()
}

fn fill_gen<const Q: u16>(b: &mut Batch32, rng: &mut Rng) {
    for j in 0..bin_ntt::N {
        for p in 0..32 {
            b.v[j][p] = (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16;
        }
    }
    b.representation = Representation::Coefficients;
}

/// One prime's three measurements. The kernels arrive as closures: a `const QUAD: bool` branch
/// inside a generic function would monomorphise the splitting kernels for the quad primes too,
/// and `Params::<2917>::PSI` (a primitive 1944-th root) does not exist.
fn measure(
    pg: &PerfGroup,
    q: u16,
    quad: bool,
    outs: &mut [Batch32],
    gen: &mut [Batch32],
    big: &mut [Batch32],
    mut run_bin: impl FnMut(&mut [Batch32]),
    mut run_gen: impl FnMut(&mut [Batch32]),
    mut run_big: impl FnMut(&mut [Batch32]),
) -> Row {
    const REP: usize = 200;
    for _ in 0..20 {
        run_bin(outs);
    }
    let bin = best(pg, 5, || {
        for _ in 0..REP {
            run_bin(outs);
        }
    });
    std::hint::black_box(&outs);

    // the generic kernel is in place, so it is fed its own output: same instruction stream
    for _ in 0..20 {
        run_gen(gen);
    }
    let g = best(pg, 5, || {
        for _ in 0..REP {
            run_gen(gen);
        }
    });
    std::hint::black_box(&gen);

    let mut bb: Option<(Counts, f64)> = None;
    for _ in 0..3 {
        let t0 = Instant::now();
        pg.start();
        run_big(big);
        let c = pg.stop();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&big);
        if bb.as_ref().map_or(true, |x| c.cycles < x.0.cycles) {
            bb = Some((c, ms));
        }
    }
    Row {
        q,
        tree: if quad { "quadratic (324 x 2)" } else { "split (648)" },
        bin,
        gen: g,
        big: bb.unwrap(),
    }
}

fn main() {
    f162::assert_layout();
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");
    let mut rng = Rng::new(0xC0FFEE);

    let polys: Vec<BinaryPoly> = (0..32 * NB).map(|_| BinaryPoly::random(&mut rng)).collect();
    let idxs: Vec<BinaryIndex32> = (0..NB)
        .map(|b| unsafe {
            transpose::slice_polys_idx(&*(polys.as_ptr().add(32 * b) as *const [BinaryPoly; 32]))
        })
        .collect();
    let mut outs: Vec<Batch32> = (0..NB).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let mut gen: Vec<Batch32> = (0..NB)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Coefficients);
            fill_gen::<12637>(&mut b, &mut rng);
            b
        })
        .collect();
    let elems: Vec<F162> = f162::random_elems(BIG, 0x5EED);
    let mut big: Vec<Batch32> = (0..BIGB).map(|_| Batch32::zero(Representation::Ntt)).collect();

    println!("cache-resident kernels and the 2^18-F162 transform, per ring element\n");
    macro_rules! quad_row {
        ($q:literal) => {
            measure(
                &pg,
                $q,
                true,
                &mut outs,
                &mut gen,
                &mut big,
                |o: &mut [Batch32]| unsafe {
                    for b in 0..NB {
                        vq::ntt_quad_bin_batch32::<$q>(&idxs[b], &mut o[b]);
                    }
                },
                |g: &mut [Batch32]| unsafe {
                    for b in g.iter_mut() {
                        vgq::ntt_quad_gen_batch32::<$q>(b);
                    }
                },
                |b: &mut [Batch32]| vq::ntt_quad_f162::<$q>(&elems, b),
            )
        };
    }
    macro_rules! split_row {
        ($q:literal) => {
            measure(
                &pg,
                $q,
                false,
                &mut outs,
                &mut gen,
                &mut big,
                |o: &mut [Batch32]| unsafe {
                    for b in 0..NB {
                        va::ntt_bin_batch32::<$q>(&idxs[b], &mut o[b]);
                    }
                },
                |g: &mut [Batch32]| unsafe {
                    for b in g.iter_mut() {
                        vg::ntt_gen_batch32::<$q>(b);
                    }
                },
                |b: &mut [Batch32]| ntt_f162::ntt_f162::<$q>(&elems, b),
            )
        };
    }
    let rows = vec![
        quad_row!(2917),
        quad_row!(4861),
        quad_row!(12637),
        split_row!(3889),
        split_row!(9721),
    ];
    let n = (200 * NB * 32) as f64;
    for r in &rows {
        println!("q = {} ({})", r.q, r.tree);
        report("  binary kernel", r.bin, n, None);
        report("  generic kernel", r.gen, n, None);
        report("  2^18 F162 materialised", r.big.0, BIG as f64 / 4.0, Some(r.big.1));
    }

    // structural variants of the generic kernel (bit 0: levels 2, 3 unfused; bit 1: 4, 5 unfused)
    println!();
    macro_rules! plan_row {
        ($q:literal, $plan:literal) => {{
            for _ in 0..20 {
                for b in gen.iter_mut() {
                    unsafe { vgq::ntt_quad_gen_batch32_plan::<$q, $plan>(b) };
                }
            }
            let c = best(&pg, 5, || {
                for _ in 0..200 {
                    for b in gen.iter_mut() {
                        unsafe { vgq::ntt_quad_gen_batch32_plan::<$q, $plan>(b) };
                    }
                }
            });
            report(concat!("  gen q=", $q, " plan ", $plan), c, (200 * NB * 32) as f64, None);
        }};
    }
    plan_row!(2917, 0);
    plan_row!(2917, 1);
    plan_row!(2917, 2);
    plan_row!(2917, 3);
    plan_row!(12637, 0);
    plan_row!(12637, 1);
    plan_row!(12637, 2);
    plan_row!(12637, 3);

    println!("\n| q | tree | binary | generic | 2^18 F162 |");
    println!("|---|------|-------:|--------:|----------:|");
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| a.bin.cycles.cmp(&b.bin.cycles));
    for r in sorted {
        println!(
            "| {} | {} | {:.0} | {:.0} | {:.1} ms ({:.0} cyc) |",
            r.q,
            r.tree,
            r.bin.cycles as f64 / n,
            r.gen.cycles as f64 / n,
            r.big.1,
            r.big.0.cycles as f64 / (BIG as f64 / 4.0)
        );
    }
    drop(big);
}

//! Ajtai commitment benchmark: `y[j] = sum_i A_i[j] * NTT_q(w_i)[j] mod q` on 2^18 `F162`
//! (2^16 ring elements) against a cold 85 MB row of `A` per prime.
//!
//! Per prime it prints the API path ([`commit`](bin_ntt::simd::commit::commit)) with its
//! decomposition into front end + transform (cache-resident), base multiplication and the memory
//! that does not hide; then the alternatives it is built out of; then `commit_2q`, the
//! base-multiplication microbenchmark and the DRAM / compute floors.
//! Usage: `taskset -c 2 bench_commit [cpu] [--quick]`.
use bin_fields::scalar::F162;
use bin_ntt::f162::RandomF162;
use bin_ntt::params::N;
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::commit::{self as cm, Acc};
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::transpose_f162 as tf;
use bin_ntt::simd::vertical_bin_asm as va;
use bin_ntt::types::*;
use std::time::Instant;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

struct Run {
    ms: f64,
    c: Counts,
}

fn best(pg: &PerfGroup, reps: usize, mut f: impl FnMut()) -> Run {
    let mut best: Option<Run> = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        pg.start();
        f();
        let c = pg.stop();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if best.as_ref().map_or(true, |b| ms < b.ms) {
            best = Some(Run { ms, c });
        }
    }
    best.unwrap()
}

#[inline(always)]
fn bb<T>(p: T) -> T {
    std::hint::black_box(p)
}

fn cyc(r: &Run, e: f64) -> f64 {
    r.c.cycles as f64 / e
}

fn random_elems(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

fn random_a(nb: usize, q: u16, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i16;
    (0..nb)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(q as u32) as i16 - half;
                }
            }
            b
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let quick = args.iter().any(|a| a == "--quick");
    pin(cpu);
    bin_ntt::f162::assert_layout();
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");

    let log_e = if quick { 15 } else { 18 };
    let nf162 = 1usize << log_e;
    let nring = nf162 / 4;
    let nb = nring / 32;
    let e = nring as f64;
    let mb = nb as f64 * 41472.0 / 1e6;
    println!(
        "bin-ntt bench_commit: cpu {cpu}, 2^{log_e} F162 = {nring} ring elements = {nb} batches, \
         A = {mb:.0} MB per prime, best of 3, one thread"
    );

    let elems = random_elems(nf162, 42);
    let a3 = random_a(nb, 3889, 0xA3);
    let a9 = random_a(nb, 9721, 0xA9);
    let mut w: Vec<Batch32> = (0..nb).map(|_| Batch32::zero(Representation::Ntt)).collect();

    // ------------------------------------------------------- front end + transform, resident
    const NB8: usize = 8;
    let small = random_elems(128 * NB8, 0xF162);
    let mut idxs: Vec<BinaryIndex32> = (0..NB8).map(|_| BinaryIndex32::zero()).collect();
    let mut souts: Vec<Batch32> = (0..NB8).map(|_| Batch32::zero(Representation::Ntt)).collect();
    const REP: usize = 100;
    let mut resident = [0.0f64; 2];
    macro_rules! res {
        ($q:expr, $i:expr) => {{
            let f = |idxs: &mut Vec<BinaryIndex32>, souts: &mut Vec<Batch32>| unsafe {
                for b in 0..NB8 {
                    let c = &*(small.as_ptr().add(128 * b) as *const [F162; 128]);
                    tf::slice_f162_into(c, &mut idxs[b]);
                    va::ntt_bin_batch32::<$q>(&idxs[b], &mut souts[b]);
                }
            };
            f(&mut idxs, &mut souts);
            let r = best(&pg, 3, || {
                for _ in 0..REP {
                    f(&mut idxs, &mut souts);
                }
            });
            resident[$i] = cyc(&r, (REP * NB8 * 32) as f64);
            println!(
                "front end + transform, cache-resident, q = {:5}: {:6.1} cycles / ring element",
                $q, resident[$i]
            );
        }};
    }
    res!(3889, 0);
    res!(9721, 1);
    std::hint::black_box(&souts);

    // ------------------------------------------------- the base multiplication on its own
    let mut acc = Acc::zero();
    let accp = acc.v.as_mut_ptr() as *mut i32;
    let w_l1 = Batch32::zero(Representation::Ntt); // only its first 27 vectors are touched
    let w_l2 = Batch32::zero(Representation::Ntt);
    let a_l2 = &a3[0];
    let wp1 = w_l1.v.as_ptr() as *const i16;
    let wp2 = w_l2.v.as_ptr() as *const i16;
    let ap2 = a_l2.v.as_ptr() as *const i16;
    let dummy = ap2 as *const i8;
    let reps = nb; // same number of batches as a full run

    println!(
        "\nbase multiplication alone (packed accumulator, 21.5 KB), cycles per ring element:"
    );
    // W out of a 1728-byte block scratch, exactly as `commit` consumes the kernel's asm output.
    let basemul = {
        let r = best(&pg, 3, || unsafe {
            for _ in 0..reps {
                for k in 0..24 {
                    cm::mac27::<false>(bb(wp1), bb(ap2).add(32 * 27 * k), dummy, accp.add(16 * 14 * k));
                }
            }
        });
        let c = cyc(&r, (reps * 32) as f64);
        println!("  W in L1 (27-slot block), A in L2        : {c:6.2}");
        c
    };
    let r = best(&pg, 3, || unsafe {
        for b in 0..reps {
            let ab = a3[b].v.as_ptr() as *const i16;
            for k in 0..24 {
                cm::mac27::<false>(bb(wp1), ab.add(32 * 27 * k), dummy, accp.add(16 * 14 * k));
            }
        }
    });
    println!("  W in L1 (27-slot block), A in DRAM      : {:6.2}", cyc(&r, (reps * 32) as f64));

    let r = best(&pg, 3, || unsafe {
        for _ in 0..reps {
            for _ in 0..24 {
                cm::mac27::<false>(bb(wp1), bb(ap2), dummy, accp);
            }
        }
    });
    println!("  everything aliased into L1 (uop floor)  : {:6.2}", cyc(&r, (reps * 32) as f64));

    let r = best(&pg, 3, || unsafe {
        for _ in 0..reps {
            cm::mac_batch::<false>(bb(wp2), bb(ap2), dummy, accp);
        }
    });
    println!("  W in L2 (41 KB batch),   A in L2        : {:6.2}", cyc(&r, (reps * 32) as f64));

    let r = best(&pg, 3, || unsafe {
        for b in 0..reps {
            cm::mac_batch::<false>(bb(wp2), a3[b].v.as_ptr() as *const i16, dummy, accp);
        }
    });
    println!("  W in L2 (41 KB batch),   A in DRAM      : {:6.2}", cyc(&r, (reps * 32) as f64));

    let r = best(&pg, 3, || unsafe {
        for b in 0..reps {
            cm::mac_batch::<false>(
                w[b].v.as_ptr() as *const i16,
                a3[b].v.as_ptr() as *const i16,
                dummy,
                accp,
            );
        }
    });
    println!("  W in DRAM (materialised), A in DRAM     : {:6.2}", cyc(&r, (reps * 32) as f64));

    let r = best(&pg, 3, || unsafe {
        for b in 0..reps {
            let nx = a3[(b + 1).min(nb - 1)].v.as_ptr() as *const i8;
            cm::mac_batch::<true>(bb(wp2), a3[b].v.as_ptr() as *const i16, nx, accp);
        }
    });
    println!("  W in L2, A in DRAM, + prefetcht1        : {:6.2}", cyc(&r, (reps * 32) as f64));
    std::hint::black_box(&acc);

    // ---------------------------------------------------------------- the whole commitment
    println!(
        "\n{:<38} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "prime / path",
        "ms",
        "cyc/elt",
        "ins/elt",
        "uops",
        "p0",
        "front+K",
        "basemul",
        "memory"
    );
    let mut refy: [[u32; N]; 2] = [[0; N], [0; N]];
    macro_rules! paths {
        ($q:expr, $a:expr, $ix:expr) => {{
            let period = cm::red_period($q);
            let kern = resident[$ix];
            let mut y = [0u32; N];
            let mut rows: Vec<String> = Vec::new();
            // Cycles per element are only comparable at one clock, and this workload is partly
            // DRAM-bound: a turbo burst inflates them at unchanged wall time. Three untimed
            // passes put the core at its sustained AVX-512 frequency before anything is counted,
            // and the API path is measured last, in the same settled state as the alternatives.
            for _ in 0..3 {
                y = cm::commit::<$q>(&elems, $a);
            }
            std::hint::black_box(&y);
            macro_rules! one {
                ($label:expr, $body:expr) => {{
                    let r = best(&pg, 3, || y = $body);
                    if refy[$ix] == [0u32; N] {
                        refy[$ix] = y;
                    }
                    assert_eq!(y, refy[$ix], "path disagreement: {}", $label);
                    rows.push(format!(
                        "{:<38} {:8.2} {:8.1} {:8.1} {:8.1} {:8.1} | {:8.1} {:8.1} {:8.1}",
                        $label,
                        r.ms,
                        cyc(&r, e),
                        r.c.instructions as f64 / e,
                        r.c.uops as f64 / e,
                        r.c.port0 as f64 / e,
                        kern,
                        basemul,
                        cyc(&r, e) - kern - basemul
                    ));
                }};
            }
            one!(
                concat!("q=", stringify!($q), "   unfused"),
                cm::commit_unfused::<$q, false>(&elems, $a, &mut w, period)
            );
            one!(
                concat!("q=", stringify!($q), "   unfused          + prefetch"),
                cm::commit_unfused::<$q, true>(&elems, $a, &mut w, period)
            );
            one!(
                concat!("q=", stringify!($q), "   batch-fused"),
                cm::commit_batch_fused::<$q, false>(&elems, $a, period)
            );
            one!(
                concat!("q=", stringify!($q), "   batch-fused      + prefetch"),
                cm::commit_batch_fused::<$q, true>(&elems, $a, period)
            );
            one!(
                concat!("q=", stringify!($q), "   block-fused"),
                cm::commit_block_fused::<$q, false, 1>(&elems, $a, period)
            );
            one!(
                concat!("q=", stringify!($q), "   block-fused      + prefetch 2"),
                cm::commit_block_fused::<$q, true, 2>(&elems, $a, period)
            );
            one!(
                concat!("q=", stringify!($q), "   block-fused      + prefetch 3"),
                cm::commit_block_fused::<$q, true, 3>(&elems, $a, period)
            );
            one!(
                concat!("q=", stringify!($q), "   block-fused      + prefetch 6"),
                cm::commit_block_fused::<$q, true, 6>(&elems, $a, period)
            );
            one!(concat!("q=", stringify!($q), " commit"), cm::commit::<$q>(&elems, $a));
            rows.rotate_right(1);
            for r in &rows {
                println!("{r}");
            }
        }};
    }
    paths!(3889, &a3, 0);
    paths!(9721, &a9, 1);

    // --------------------------------------------------------------------------- both primes
    let mut y2 = ([0u32; N], [0u32; N]);
    let r = best(&pg, 3, || y2 = cm::commit_2q(&elems, &a3, &a9));
    assert_eq!(y2.0, refy[0]);
    assert_eq!(y2.1, refy[1]);
    println!(
        "\ncommit_2q (one slicing, {:.0} MB of A): {:.2} ms, {:.1} cycles per ring element and \
         prime, {:.2} GHz",
        2.0 * mb,
        r.ms,
        cyc(&r, 2.0 * e),
        r.c.cycles as f64 / (r.ms * 1e6)
    );

    // -------------------------------------------------------------------------------- floors
    let ghz = r.c.cycles as f64 / (r.ms * 1e6);
    let dram_ms = mb / 19500.0 * 1000.0;
    println!("\nfloors (at {ghz:.2} GHz):");
    for (i, q) in [3889u16, 9721].iter().enumerate() {
        let comp = resident[i] + basemul;
        let comp_ms = comp * e / (ghz * 1e6);
        println!(
            "  q = {q}: DRAM {:.2} ms ({:.0} MB of A at 19.5 GB/s), compute {:.2} ms \
             ({:.0} = {:.0} front end + transform + {:.0} basemul cycles/elt), floor = max = {:.2} ms",
            dram_ms,
            mb,
            comp_ms,
            comp,
            resident[i],
            basemul,
            dram_ms.max(comp_ms)
        );
    }
    println!(
        "  the unfused path additionally writes {:.0} MB (NT, 37 GB/s = {:.2} ms) and reads it \
         back ({:.2} ms): DRAM floor {:.2} ms",
        mb,
        mb / 37000.0 * 1000.0,
        dram_ms,
        2.0 * dram_ms + mb / 37000.0 * 1000.0
    );
}

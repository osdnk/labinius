//! The horizontal-layout Ajtai commitment: `y = sum_i A_i o NTT(w_i)` over 2^16 ring elements
//! (2^18 `F162`), four to sixteen polynomials at a time so the transform output never leaves L1.
//! `A` is 16384 `HBatch4` per prime (84.9 MB), cold.
//!
//! Usage: `taskset -c 4 bench_commit_h [cpu] [--quick]`.
use bin_fields::scalar::F162;
use bin_ntt::f162::RandomF162;
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::commit_h::*;
use bin_ntt::simd::horizontal_gen::{ntt_gen_hbatch4, HBatch4};
use std::time::Instant;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

const MADV_HUGEPAGE: i32 = 14;

/// `A` in 2 MB pages: the L2 stream prefetcher does not cross a 4 KB page boundary, and one group
/// of four ring elements consumes 5184 bytes, i.e. more than one 4 KB page.
fn alloc_a(n: usize, q: u16, seed: u64, huge: bool) -> &'static mut [HBatch4] {
    use std::alloc::{alloc_zeroed, Layout};
    let sz = n * std::mem::size_of::<HBatch4>();
    let lay = Layout::from_size_align(sz, if huge { 2 << 20 } else { 64 }).unwrap();
    let p = unsafe { alloc_zeroed(lay) };
    assert!(!p.is_null(), "allocation of {sz} bytes failed");
    if huge {
        unsafe { madvise(p, sz, MADV_HUGEPAGE) };
    }
    let s: &'static mut [HBatch4] = unsafe { std::slice::from_raw_parts_mut(p as *mut HBatch4, n) };
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i32;
    for b in s.iter_mut() {
        for r in 0..81 {
            for l in 0..32 {
                b.v[r][l] = (rng.below(q as u32) as i32 - half) as i16;
            }
        }
    }
    s
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

fn report(label: &str, r: &Run, elems: f64) -> f64 {
    let e = elems;
    println!(
        "{label:<42} {:8.2} ms | {:7.1} cyc {:8.1} ins {:8.1} uops (p0 {:7.1}, p5 {:7.1}) | {:.2} GHz",
        r.ms,
        r.c.cycles as f64 / e,
        r.c.instructions as f64 / e,
        r.c.uops as f64 / e,
        r.c.port0 as f64 / e,
        r.c.port5 as f64 / e,
        r.c.cycles as f64 / (r.ms * 1e6)
    );
    r.c.cycles as f64 / e
}

fn random_a(n: usize, q: u16, seed: u64) -> Vec<HBatch4> {
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i32;
    let mut v: Vec<HBatch4> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = HBatch4::zero();
        for r in 0..81 {
            for l in 0..32 {
                b.v[r][l] = (rng.below(q as u32) as i32 - half) as i16;
            }
        }
        v.push(b);
    }
    v
}

const NB: usize = 8; // resident working set: 8 groups = 32 polynomials

/// Cache-resident decomposition: front end, kernel, base multiplication (with its share of the
/// periodic reduction). Returns (front, kernel, basemul) cycles per ring element.
fn resident<const Q: u16>(pg: &PerfGroup, elems: &[F162], a: &[HBatch4]) -> (f64, f64, f64, f64) {
    const REP: usize = 300;
    let mut ws: Vec<HBatch4> = (0..NB).map(|_| HBatch4::zero()).collect();

    let front = |ws: &mut Vec<HBatch4>| unsafe {
        for g in 0..NB {
            hbatch4_from_f162_into(
                &*(elems.as_ptr().add(16 * g) as *const [F162; 16]),
                &mut ws[g],
            );
        }
    };
    front(&mut ws);
    let r = best(pg, 5, || {
        for _ in 0..REP {
            front(&mut ws);
        }
    });
    let f = report("  front end alone (L1)", &r, (REP * NB * 4) as f64);

    let r = best(pg, 5, || unsafe {
        for _ in 0..REP {
            for w in ws.iter_mut() {
                ntt_gen_hbatch4::<Q>(w);
            }
        }
    });
    let k = report("  kernel alone (L1)", &r, (REP * NB * 4) as f64);
    std::hint::black_box(&ws);

    // base multiplication with W, A and the accumulator all in L1 (2 groups: 26 KB), one
    // reduction every KRED groups exactly as in the driver.
    const NL1: usize = 2;
    let mut acc = Acc::zero();
    let kr = kred::<Q>();
    let r = best(pg, 5, || unsafe {
        let mut since = 0;
        for _ in 0..REP {
            for g in 0..NL1 {
                basemul_step::<false>(&ws[g], &a[g], &mut acc);
                since += 1;
                if since >= kr {
                    reduce_acc::<Q>(&mut acc);
                    since = 0;
                }
            }
        }
    });
    let b = report("  basemul + reduction alone (L1)", &r, (REP * NL1 * 4) as f64);
    let mut ap: Vec<HBatch4> = a[..NL1].to_vec();
    permute_a_slice(&mut ap);
    let r = best(pg, 5, || unsafe {
        let mut since = 0;
        for _ in 0..REP {
            for g in 0..NL1 {
                basemul_step::<true>(&ws[g], &ap[g], &mut acc);
                since += 1;
                if since >= kr {
                    reduce_acc::<Q>(&mut acc);
                    since = 0;
                }
            }
        }
    });
    let bp = report("  basemul, A pre-permuted (L1)", &r, (REP * NL1 * 4) as f64);
    std::hint::black_box(&acc);
    (f, k, b, bp)
}

fn headline<const Q: u16>(pg: &PerfGroup, elems: &[F162], ngroups: usize, nring: f64, huge: bool) {
    println!("\n===== q = {Q} ({} pages for A) =====", if huge { "2 MB" } else { "4 KB" });
    let small = random_a(NB, Q, 0xA1 ^ Q as u64);
    let (f, k, b, bp) = resident::<Q>(pg, elems, &small);
    let compute = f + k + b;
    let compute_p = f + k + bp;
    println!(
        "  compute sum: front {f:.1} + kernel {k:.1} + basemul {b:.1} = {compute:.1} cyc/elt \
         ({compute_p:.1} with A pre-permuted)"
    );
    drop(small);

    let a = alloc_a(ngroups, Q, 0xAA00 ^ Q as u64, huge);
    println!(
        "  A: {ngroups} HBatch4 = {:.1} MB, DRAM floor {:.2} ms at 19.5 GB/s",
        ngroups as f64 * 5184.0 / 1e6,
        ngroups as f64 * 5184.0 / 19.5e9 * 1e3
    );

    // correctness on the first groups, against the scalar reference
    let want = commit_h_scalar::<Q>(&elems[..16 * 3], &a[..3]);
    assert_eq!(commit_h::<Q>(&elems[..16 * 3], &a[..3]), want, "commit_h != scalar reference");
    println!("  (verified against sum_i A_i[j] NTT(w_i)[j] on 3 groups)");

    let stream;
    // How fast can this core pull the 85 MB of A alone, in exactly the commitment's pattern?
    {
        use std::arch::x86_64::*;
        let r = best(pg, 3, || unsafe {
            let mut acc = _mm512_setzero_si512();
            for b in a.iter() {
                let p = b.v.as_ptr() as *const __m512i;
                for r in 0..81 {
                    acc = _mm512_add_epi16(acc, _mm512_load_si512(p.add(r)));
                }
            }
            std::hint::black_box(acc);
        });
        stream = report("  A stream alone (no compute)", &r, nring);
        println!("    -> {:.1} GB/s", ngroups as f64 * 5184.0 / (r.ms * 1e-3) / 1e9);
    }

    let mut totals: Vec<(String, f64)> = Vec::new();
    macro_rules! run {
        ($g:expr, $pff:expr, $pa:expr, $pf:expr, $label:expr) => {{
            let r = best(pg, 3, || {
                std::hint::black_box(commit_h_var::<Q, $g, $pa, $pf, $pff>(elems, a));
            });
            let c = report($label, &r, nring);
            totals.push(($label.to_string(), c));
        }};
    }
    run!(1, 0, false, 0, "4 polys/step,  A as stored, no pf");
    run!(1, 0, false, 1, "4 polys/step,  A as stored, pf 1 blk");
    run!(1, 0, false, 2, "4 polys/step,  A as stored, pf 2 blk");
    run!(1, 2, false, 1, "4 polys/step,  A as stored, pf 1 + fe 2");
    run!(2, 0, false, 0, "8 polys/step,  A as stored, no pf");
    run!(2, 0, false, 1, "8 polys/step,  A as stored, pf 1 blk");
    run!(4, 0, false, 0, "16 polys/step, A as stored, no pf");
    run!(4, 0, false, 1, "16 polys/step, A as stored, pf 1 blk");

    permute_a_slice(a);
    let want2 = commit_h_var::<Q, 1, true, 1, 0>(&elems[..16 * 3], &a[..3]);
    assert_eq!(want2, want, "permuted-A commit != scalar reference");
    run!(1, 0, true, 0, "4 polys/step,  A permuted,  no pf");
    run!(1, 0, true, 1, "4 polys/step,  A permuted,  pf 1 blk");
    run!(1, 0, true, 2, "4 polys/step,  A permuted,  pf 2 blk");
    run!(2, 0, true, 0, "8 polys/step,  A permuted,  no pf");
    run!(2, 0, true, 1, "8 polys/step,  A permuted,  pf 1 blk");
    run!(4, 0, true, 0, "16 polys/step, A permuted,  no pf");
    run!(4, 0, true, 1, "16 polys/step, A permuted,  pf 1 blk");

    let bestc = totals.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
    let bestl = totals.iter().find(|t| t.1 == bestc).unwrap().0.clone();
    let ghz = 4.5;
    println!(
        "\n  --- q = {Q} summary (cycles per ring element) ---\n\
         \x20   front end                      {f:7.1}\n\
         \x20   kernel (L1 resident)           {k:7.1}\n\
         \x20   basemul + reduction (L1)       {b:7.1}   ({bp:.1} with A pre-permuted)\n\
         \x20   compute sum                    {compute:7.1}\n\
         \x20   A stream alone (85 MB)         {stream:7.1}   (DRAM floor {:.2} ms at 19.5 GB/s)\n\
         \x20   best measured total            {bestc:7.1}   ({bestl})\n\
         \x20   memory remainder               {:7.1}   ({:.0}% of the total, {:.0}% of the stream hidden)\n\
         \x20   total without the front end    {:7.1}\n\
         \x20   best total {:.2} ms; compute-only would be {:.2} ms, DRAM-only {:.2} ms",
        ngroups as f64 * 5184.0 / 19.5e9 * 1e3,
        bestc - compute,
        100.0 * (bestc - compute) / bestc,
        100.0 * (1.0 - (bestc - compute) / stream),
        bestc - f,
        bestc * nring / (ghz * 1e6),
        compute * nring / (ghz * 1e6),
        ngroups as f64 * 5184.0 / 19.5e9 * 1e3,
    );
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let quick = args.iter().any(|a| a == "--quick");
    pin(cpu);
    bin_ntt::f162::assert_layout();
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");
    let log_e = if quick { 14 } else { 18 };
    let nf162 = 1usize << log_e;
    let nring = nf162 / 4;
    let ngroups = nring / 4;
    println!(
        "bin-ntt bench_commit_h: cpu {cpu}, 2^{log_e} F162 = {nring} ring elements = {ngroups} \
         groups of 4, best of 3, one thread"
    );
    println!(
        "reduction interval: {} groups (q = 3889), {} groups (q = 9721)",
        kred::<3889>(),
        kred::<9721>()
    );
    // Spin the core up to its AVX-512 turbo before the first measurement (a cold core runs the
    // first few hundred microseconds at ~1 GHz, which distorts the cycle counts of small loops).
    {
        let mut b = HBatch4::zero();
        let t0 = Instant::now();
        while t0.elapsed().as_millis() < 300 {
            for _ in 0..200 {
                unsafe { ntt_gen_hbatch4::<3889>(&mut b) };
            }
        }
        std::hint::black_box(&b);
    }
    let mut rng = Rng::new(42);
    let elems: Vec<F162> = (0..nf162).map(|_| F162::random(&mut rng)).collect();

    let huge = args.iter().any(|a| a == "--huge");
    headline::<3889>(&pg, &elems, ngroups, nring as f64, huge);
    headline::<9721>(&pg, &elems, ngroups, nring as f64, huge);
}

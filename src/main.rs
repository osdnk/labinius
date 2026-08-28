//! The commitment API end to end: build a key, commit 2^18 `F162` under it, print the time.
//!
//! `cargo run --release --offline` (pin it with `taskset -c 2` or pass the core as the first
//! argument). The witness is always 2^18 `F162` = 2^16 ring elements of `R_648`; `r` says into how
//! many chunks it is split, each committed under the same key, so the key is 170 MB / r and at
//! r >= 8 it is small enough to be served by L3 instead of DRAM.
use bin_fields::scalar::F162;
use bin_ntt::f162::RandomF162;
use bin_ntt::perf::PerfGroup;
use bin_ntt::rng::Rng;
use bin_ntt::simd::commit::{self as cm, Acc};
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::transpose_f162 as tf;
use bin_ntt::simd::vertical_bin_asm as va;
use bin_ntt::types::{Batch32, Representation};
use bin_ntt::{
    fold, sample_short_challenge, CommitmentKey, Timings, Transcript, DEFAULT_BOUND,
    DEFAULT_WEIGHT, PRIMES,
};
use std::time::Instant;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

/// Best of `reps`: wall milliseconds and, if the counters are available, cycles.
fn best(pg: Option<&PerfGroup>, reps: usize, mut f: impl FnMut()) -> (f64, Option<u64>) {
    let mut out = (f64::MAX, None);
    for _ in 0..reps {
        let t0 = Instant::now();
        if let Some(p) = pg {
            p.start();
        }
        f();
        let c = pg.map(|p| p.stop().cycles);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < out.0 {
            out = (ms, c);
        }
    }
    out
}

/// Front end + transform on a cache-resident input, per prime; and the base multiplication with
/// the transform block in L1 and A in L2 — the two halves of the per-element cost inside a commit.
fn components(pg: Option<&PerfGroup>) -> ([f64; 2], f64) {
    const NB: usize = 8;
    const REP: usize = 100;
    let mut rng = Rng::new(0xF162);
    let small: Vec<F162> = (0..128 * NB).map(|_| F162::random(&mut rng)).collect();
    let mut idx: Vec<BinaryIndex32> = (0..NB).map(|_| BinaryIndex32::zero()).collect();
    let mut out: Vec<Batch32> = (0..NB)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();

    let mut kern = [0.0f64; 2];
    macro_rules! run {
        ($q:expr, $i:expr) => {{
            let (ms, cy) = best(pg, 3, || unsafe {
                for _ in 0..REP {
                    for b in 0..NB {
                        let c = &*(small.as_ptr().add(128 * b) as *const [F162; 128]);
                        tf::slice_f162_into(c, &mut idx[b]);
                        va::ntt_bin_batch32::<$q>(&idx[b], &mut out[b]);
                    }
                }
            });
            let _ = ms;
            kern[$i] = cy.map_or(f64::NAN, |c| c as f64 / (REP * NB * 32) as f64);
        }};
    }
    run!(3889, 0);
    run!(9721, 1);
    std::hint::black_box(&out);

    let mut acc = Acc::zero();
    let accp = acc.v.as_mut_ptr() as *mut i32;
    let w = Batch32::zero(Representation::Ntt);
    let a = Batch32::zero(Representation::Ntt);
    let (wp, ap) = (w.v.as_ptr() as *const i16, a.v.as_ptr() as *const i16);
    let reps = 2048;
    let (_, cy) = best(pg, 3, || unsafe {
        for _ in 0..reps {
            for k in 0..24 {
                cm::mac27::<false>(
                    std::hint::black_box(wp),
                    std::hint::black_box(ap).add(32 * 27 * k),
                    ap as *const i8,
                    accp.add(16 * cm::ACC_PER_BLK * k),
                );
            }
        }
    });
    std::hint::black_box(&acc);
    (kern, cy.map_or(f64::NAN, |c| c as f64 / (reps * 32) as f64))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let log_e: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(18);
    pin(cpu);
    bin_ntt::f162::assert_layout();
    let pg = PerfGroup::new().ok();

    let nf162 = 1usize << log_e;
    let nring = nf162 / 4;
    let mut rng = Rng::new(0x5EED_5EED);
    let witness: Vec<F162> = (0..nf162).map(|_| F162::random(&mut rng)).collect();

    println!(
        "bin-ntt commitment API, cpu {cpu}, one thread, primes {} and {}",
        PRIMES[0], PRIMES[1]
    );
    println!(
        "witness: 2^{log_e} F162 = {nring} ring elements of R_648 = {:.1} MB",
        (nf162 * 24) as f64 / 1e6
    );
    println!("output:  a 4 x r matrix of R_162 elements, 162 centered slots per prime\n");

    let (kern, basemul) = components(pg.as_ref());
    if pg.is_some() {
        println!(
            "front end + transform, cache-resident: {:.0} / {:.0} cycles per ring element \
             (q = {} / {})",
            kern[0], kern[1], PRIMES[0], PRIMES[1]
        );
        println!(
            "base multiplication, W in L1, A in L2: {basemul:.0} cycles per ring element and prime"
        );
    } else {
        println!("(no hardware counters: perf_event_paranoid, wall time only)");
    }

    println!(
        "\n{:>3}  {:>10}  {:>9}  {:>9}  {:>12}  {:>11}",
        "r", "key", "ms", "commit ms", "decompose us", "cyc/elt/q"
    );
    let mut base_ms = f64::MAX;
    for r in [1usize, 4, 16, 256] {
        let ck = CommitmentKey::random(nf162 / r, 0xA11CE ^ r as u64);
        for _ in 0..2 {
            std::hint::black_box(ck.commit(&witness, r));
        }
        let (mut ms, mut cy, mut t) = (f64::MAX, None, Timings::default());
        for _ in 0..3 {
            let t0 = Instant::now();
            if let Some(p) = pg.as_ref() {
                p.start();
            }
            let (c, ti) = ck.commit_timed(&witness, r);
            let cycles = pg.as_ref().map(|p| p.stop().cycles);
            let el = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(c);
            if el < ms {
                ms = el;
                cy = cycles;
                t = ti;
            }
        }
        let cyc = cy.map_or(f64::NAN, |c| c as f64 / (2 * nring) as f64);
        println!(
            "{r:>3}  {:>7.1} MB  {ms:>9.2}  {:>9.2}  {:>12.0}  {:>11.0}",
            ck.bytes() as f64 / 1e6,
            t.commit_ms,
            t.decompose_ms * 1e3,
            cyc
        );
        if r == 1 {
            base_ms = ms;
        } else {
            assert!(
                ms <= base_ms * 1.02,
                "r = {r} slower than r = 1: {ms} vs {base_ms} ms"
            );
        }
    }
    println!(
        "\ncolumn c of the 4 x r output is the commitment of chunk c, split into its four R_162 \
         components;\nslot s of a component evaluates at a primitive 243-rd root of unity indexed \
         by POW3_SLOT_EXP[s]."
    );

    folding(&witness, nf162);
}

/// The folding step on top of the r = 256 commitment: `v = sum_j c_j W_j` and `A v`.
fn folding(witness: &[F162], nf162: usize) {
    const R: usize = 256;
    let ck = CommitmentKey::random(nf162 / R, 0xF01D);
    println!(
        "\nfolding, r = {R} chunks of {} ring elements, weight-{DEFAULT_WEIGHT} ternary challenges",
        ck.len_ring()
    );

    let mut plain = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        std::hint::black_box(ck.commit(witness, R));
        plain = plain.min(t0.elapsed().as_secs_f64() * 1e3);
    }

    let t0 = Instant::now();
    let (_, mut aux) = ck.commit_with_aux(witness, R);
    let cold = t0.elapsed().as_secs_f64() * 1e3;

    let (mut keep, mut best) = (f64::MAX, None);
    for _ in 0..3 {
        let t0 = Instant::now();
        let c = ck.commit_into_aux(witness, R, &mut aux);
        let el = t0.elapsed().as_secs_f64() * 1e3;
        if el < keep {
            keep = el;
            best = Some(c);
        }
    }
    let c = best.unwrap();

    let t0 = Instant::now();
    let mut t = Transcript::new(b"bin-ntt/fold");
    for j in 0..R {
        t.absorb_elements(c.column(j));
    }
    let mut attempts = 0u64;
    let ch: Vec<_> = (0..R)
        .map(|_| {
            let (x, a) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
            attempts += a;
            x
        })
        .collect();
    let chal_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut out = fold(&ck, &aux, &ch);
    for _ in 0..2 {
        let o = fold(&ck, &aux, &ch);
        if o.timings.total_ms < out.timings.total_ms {
            out = o;
        }
    }
    let f = out.timings;

    println!("commit                     {plain:>8.2} ms");
    println!(
        "commit_into_aux            {keep:>8.2} ms   (+{:.1} %, {:.0} MB of witness transform kept)",
        100.0 * (keep / plain - 1.0),
        aux.bytes() as f64 / 1e6,
    );
    println!(
        "commit_with_aux            {cold:>8.2} ms   (the same, allocating the buffer: \
         {:.0} ms of first-touch page faults)",
        cold - keep
    );
    println!(
        "challenges ({R} sampled)   {chal_ms:>8.2} ms   ({:.1} attempts each)",
        attempts as f64 / R as f64
    );
    println!("fold                       {:>8.2} ms", f.total_ms);
    for (name, ms) in [
        ("challenge NTTs", f.challenge_ntt_ms),
        ("accumulation", f.accumulate_ms),
        ("inverse NTT (q1)", f.inverse_ntt_ms),
        ("forward NTT (q2)", f.forward_q2_ms),
        ("y = A v", f.y_ms),
    ] {
        println!("  {name:<24} {ms:>8.2} ms");
    }
    println!(
        "prover total               {:>8.2} ms   (commit_into_aux + fold)",
        keep + f.total_ms
    );
    println!(
        "v: {} ring elements of R_648 in coefficient form, max |coefficient| = {} (q1/2 = {:.1})",
        out.v.len(),
        out.max_abs_v,
        PRIMES[0] as f64 / 2.0
    );
}

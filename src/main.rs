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
use bin_ntt::eval::{
    check_claim, evaluate_mle, fold_binary, left_expand, sample_point, verify_binary, verify_fold,
    EvalPoint, RawCommitments,
};
use bin_ntt::{
    fold, sample_short_challenge, AdditionalLimb, CommitmentKey, Timings, Transcript,
    DEFAULT_BOUND, DEFAULT_LIMBS, DEFAULT_WEIGHT, BASE_PRIME,
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

/// `--limbs 2917,4861,9721,12637` (any subset, in any order) selects the additional limbs; the
/// rest of the command line is `<cpu> <log2 witness>`.
fn parse_limbs(args: &[String]) -> (Vec<String>, Vec<AdditionalLimb>) {
    let mut pos = Vec::new();
    let mut limbs: Option<Vec<AdditionalLimb>> = None;
    let mut i = 1;
    while i < args.len() {
        let list = if let Some(v) = args[i].strip_prefix("--limbs=") {
            i += 1;
            Some(v.to_string())
        } else if args[i] == "--limbs" {
            i += 2;
            Some(args.get(i - 1).expect("--limbs wants a list of primes").clone())
        } else {
            pos.push(args[i].clone());
            i += 1;
            None
        };
        if let Some(list) = list {
            limbs = Some(
                list.split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let q: u16 = s.trim().parse().expect("a prime");
                        AdditionalLimb::from_prime(q)
                            .unwrap_or_else(|| panic!("{q} is not an additional limb"))
                    })
                    .collect(),
            );
        }
    }
    (pos, limbs.unwrap_or_else(|| DEFAULT_LIMBS.to_vec()))
}

/// The commitment limb by limb: the base limb alone, then each additional limb as the difference
/// between a two-limb key and that base run, then the whole configured key. `r = 1`, so every
/// limb streams its own 85 MB of `A` from DRAM.
fn per_limb(witness: &[F162], nf162: usize, limbs: &[AdditionalLimb], pg: Option<&PerfGroup>) {
    let nring = nf162 / 4;
    let run = |add: &[AdditionalLimb]| -> (f64, f64) {
        let ck = CommitmentKey::random(nf162, 0xA11CE, add);
        for _ in 0..2 {
            std::hint::black_box(ck.commit(witness, 1));
        }
        let (ms, cy) = best(pg, 3, || {
            std::hint::black_box(ck.commit(witness, 1));
        });
        (ms, cy.map_or(f64::NAN, |c| c as f64 / nring as f64))
    };

    println!("\nper limb, r = 1 (85 MB of A per limb, cold); a limb alone is its key minus the base one\n");
    println!(
        "{:>6}  {:>10}  {:>9}  {:>13}  {:>12}",
        "q", "tree", "ms", "cyc/element", "with base ms"
    );
    let (base_ms, base_cy) = run(&[]);
    println!("{BASE_PRIME:>6}  {:>10}  {base_ms:>9.2}  {base_cy:>13.0}  {:>12}", "split", "-");
    for l in limbs {
        let (ms, cy) = run(std::slice::from_ref(l));
        println!(
            "{:>6}  {:>10}  {:>9.2}  {:>13.0}  {ms:>12.2}",
            l.prime(),
            if l.is_quadratic() { "quadratic" } else { "split" },
            ms - base_ms,
            cy - base_cy,
            ms = ms
        );
    }
    let (all_ms, all_cy) = run(limbs);
    println!(
        "{:>6}  {:>10}  {all_ms:>9.2}  {:>13.0}  {:>12}",
        "total",
        format!("{} limbs", 1 + limbs.len()),
        all_cy / (1 + limbs.len()) as f64,
        "-"
    );
    println!("(the total's cycles are per ring element and limb)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (pos, limbs) = parse_limbs(&args);
    let cpu = pos.first().and_then(|s| s.parse().ok()).unwrap_or(2);
    let log_e: u32 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(18);
    let default_cfg = limbs.as_slice() == DEFAULT_LIMBS.as_slice();
    pin(cpu);
    bin_ntt::f162::assert_layout();
    let pg = PerfGroup::new().ok();

    let nf162 = 1usize << log_e;
    let nring = nf162 / 4;
    let mut rng = Rng::new(0x5EED_5EED);
    let witness: Vec<F162> = (0..nf162).map(|_| F162::random(&mut rng)).collect();

    let names: Vec<String> = core::iter::once(BASE_PRIME.to_string())
        .chain(limbs.iter().map(|l| l.prime().to_string()))
        .collect();
    println!(
        "bin-ntt commitment API, cpu {cpu}, one thread, limbs {}",
        names.join(", ")
    );
    println!(
        "witness: 2^{log_e} F162 = {nring} ring elements of R_648 = {:.1} MB",
        (nf162 * 24) as f64 / 1e6
    );
    println!("output:  a 4 x r matrix of R_162 elements, 162 centered slots per limb\n");

    let (kern, basemul) = components(pg.as_ref());
    if pg.is_some() {
        println!(
            "front end + transform, cache-resident: {:.0} / {:.0} cycles per ring element \
             (q = 3889 / 9721)",
            kern[0], kern[1]
        );
        println!(
            "base multiplication, W in L1, A in L2: {basemul:.0} cycles per ring element and limb"
        );
    } else {
        println!("(no hardware counters: perf_event_paranoid, wall time only)");
    }

    if !default_cfg {
        per_limb(&witness, nf162, &limbs, pg.as_ref());
    }

    println!(
        "\n{:>3}  {:>10}  {:>9}  {:>9}  {:>12}  {:>11}",
        "r", "key", "ms", "commit ms", "decompose us", "cyc/elt/limb"
    );
    let mut base_ms = f64::MAX;
    for r in [1usize, 4, 16, 256] {
        let ck = CommitmentKey::random(nf162 / r, 0xA11CE ^ r as u64, &limbs);
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
        let cyc = cy.map_or(f64::NAN, |c| c as f64 / (ck.limbs() * nring) as f64);
        println!(
            "{r:>3}  {:>7.1} MB  {ms:>9.2}  {:>9.2}  {:>12.0}  {:>11.0}",
            ck.bytes() as f64 / 1e6,
            t.commit_ms,
            t.decompose_ms * 1e3,
            cyc
        );
        if r == 1 {
            base_ms = ms;
        } else if default_cfg {
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

    folding(&witness, nf162, &limbs, pg.as_ref());
}

/// The folding step on top of the r = 256 commitment: `v = sum_j c_j W_j` and `A v`.
fn folding(
    witness: &[F162],
    nf162: usize,
    limbs: &[AdditionalLimb],
    pg: Option<&PerfGroup>,
) {
    const R: usize = 256;
    let ck = CommitmentKey::random(nf162 / R, 0xF01D, limbs);
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

    let (mut keep, mut keep_cy, mut best) = (f64::MAX, None, None);
    for _ in 0..3 {
        let t0 = Instant::now();
        if let Some(p) = pg {
            p.start();
        }
        let c = ck.commit_into_aux(witness, R, &mut aux);
        let cy = pg.map(|p| p.stop().cycles);
        let el = t0.elapsed().as_secs_f64() * 1e3;
        if el < keep {
            keep = el;
            keep_cy = cy;
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
    let mut fold_cy = None;
    for _ in 0..3 {
        if let Some(p) = pg {
            p.start();
        }
        let o = fold(&ck, &aux, &ch);
        let cy = pg.map(|p| p.stop().cycles);
        if o.timings.total_ms < out.timings.total_ms {
            fold_cy = cy;
            out = o;
        }
    }
    let f = out.timings;

    let ev = evaluation(witness, &ck, &aux, &c, &ch, &out);

    println!("commit                     {plain:>8.2} ms");
    println!(
        "commit_into_aux            {keep:>8.2} ms   (+{:.1} %, {:.0} MB of witness transform kept{})",
        100.0 * (keep / plain - 1.0),
        aux.bytes() as f64 / 1e6,
        keep_cy.map_or(String::new(), |c| format!(", {:.1} Mcycles", c as f64 / 1e6)),
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
    println!(
        "fold                       {:>8.2} ms{}",
        f.total_ms,
        fold_cy.map_or(String::new(), |c| format!("   ({:.1} Mcycles)", c as f64 / 1e6))
    );
    for (name, ms) in [
        ("challenge NTTs", f.challenge_ntt_ms),
        ("accumulation", f.accumulate_ms),
        ("inverse NTT (q1)", f.inverse_ntt_ms),
        ("forward NTT (limbs)", f.forward_add_ms),
        ("y = A v", f.y_ms),
    ] {
        println!("  {name:<24} {ms:>8.2} ms");
    }
    println!("left_expand                {:>8.2} ms   (u = B W, 2^18 F162 products)", ev.left_expand);
    println!(
        "PROVER total               {:>8.2} ms   (commit_into_aux + left_expand + challenges + fold)",
        keep + ev.left_expand + chal_ms + f.total_ms
    );
    println!(
        "v: {} ring elements of R_648 in coefficient form, max |coefficient| = {} (q1/2 = {:.1})",
        out.v.len(),
        out.max_abs_v,
        BASE_PRIME as f64 / 2.0
    );

    println!("\nSTATEMENT (not prover time)");
    println!("  sample_point             {:>8.1} us", 1e3 * ev.point);
    println!(
        "  evaluate_mle             {:>8.1} us   (2^18 products, word-sliced; scalar F162: \
         {:.0} us, {:.1}x)",
        1e3 * ev.mle,
        1e3 * ev.scalar,
        ev.scalar / ev.mle
    );
    println!("STATEMENT total            {:>8.1} us", 1e3 * (ev.point + ev.mle));

    println!("\nVERIFIER");
    for (name, ms) in [
        ("check_claim", ev.check_claim),
        ("fold_binary", ev.fold_binary),
        ("verify_fold", ev.verify_fold),
        ("verify_binary", ev.verify_binary),
    ] {
        println!("  {name:<24} {:>8.1} us", 1e3 * ms);
    }
    println!(
        "VERIFIER total             {:>8.1} us",
        1e3 * (ev.check_claim + ev.fold_binary + ev.verify_fold + ev.verify_binary)
    );
}

/// Wall time of the left-expansion, the statement and the verifier, in milliseconds.
struct EvalTimings {
    point: f64,
    mle: f64,
    scalar: f64,
    left_expand: f64,
    check_claim: f64,
    fold_binary: f64,
    verify_fold: f64,
    verify_binary: f64,
}

/// Best of `reps` wall milliseconds, and the last value produced.
fn best_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let v = std::hint::black_box(f());
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        out = Some(v);
    }
    (best, out.unwrap())
}

/// The left-expansion `u = B W`, the claim it implies, and the three verifier checks — every
/// check recomputed from `v`, `u` and the commitments alone.
fn evaluation(
    witness: &[F162],
    ck: &CommitmentKey,
    aux: &bin_ntt::AuxData,
    c: &bin_ntt::VerticallyAlignedMatrix<bin_ntt::PowerOfThreeRingElementWithLimbs>,
    ch: &[bin_ntt::ShortChallenge],
    out: &bin_ntt::FoldOutput,
) -> EvalTimings {
    const LW: usize = 10;
    const LR: usize = 8;
    assert_eq!(ck.len_f162(), 1 << LW, "the eval stage wants 2^18 F162 in 256 chunks");
    let mut t = Transcript::new(b"bin-ntt/eval");
    for j in 0..1 << LR {
        t.absorb_elements(c.column(j));
    }
    let (point_ms, point) = best_of(3, || sample_point::<LW, LR>(&mut t.clone()));
    let (mle_ms, claim) = best_of(3, || evaluate_mle(witness, &point));
    let (le_ms, lx) = best_of(3, || left_expand(witness, &point.r0));
    let u = lx.u;

    let (scalar_ms, _) = best_of(3, || scalar_mle(witness, &point));

    let raw = RawCommitments::from_aux(aux);
    let (cc_ms, ok1) = best_of(3, || check_claim(&u, &point.r1, claim));
    let (fb_ms, folded) = best_of(3, || fold_binary(&u, ch));
    let (vf_ms, ok2) = best_of(3, || verify_fold(ck, &raw, ch, &out.v));
    let (vb_ms, ok3) = best_of(3, || verify_binary(&point.r0, &out.v, folded));
    assert!(ok1 && ok2 && ok3, "the verifier rejected an honest transcript");

    EvalTimings {
        point: point_ms,
        mle: mle_ms,
        scalar: scalar_ms,
        left_expand: le_ms,
        check_claim: cc_ms,
        fold_binary: fb_ms,
        verify_fold: vf_ms,
        verify_binary: vb_ms,
    }
}

/// The same evaluation through `F162`'s scalar `Mul` (one `pclmul` chain per product), the
/// baseline the word-sliced path is measured against.
fn scalar_mle<const LW: usize, const LR: usize>(w: &[F162], p: &EvalPoint<LW, LR>) -> F162 {
    let eq0 = bin_ntt::eq_table(&p.r0);
    let eq1 = bin_ntt::eq_table(&p.r1);
    let mut t = F162::ZERO;
    for j in 0..1 << LR {
        let mut s = F162::ZERO;
        for i in 0..1 << LW {
            s = s + eq0[i] * w[i + (j << LW)];
        }
        t = t + s * eq1[j];
    }
    t
}

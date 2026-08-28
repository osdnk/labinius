//! The scheme end to end, once, with the wall clock on every step.
//!
//! `cargo run --release --offline`, pinned with `taskset -c 2`. The configuration is the code
//! below: 2^18 `F162` of witness (2^16 ring elements of `R_648`) split into `R = 256` chunks,
//! committed under one key over the limbs of `DEFAULT_LIMBS`, then folded with weight-21 ternary
//! challenges and verified.
use bin_fields::scalar::F162;
use bin_ntt::eval::{
    check_claim, evaluate_mle, fold_binary, left_expand, sample_point, verify_binary, verify_fold,
    RawCommitments,
};
use bin_ntt::f162::RandomF162;
use bin_ntt::rng::Rng;
use bin_ntt::{
    fold, sample_short_challenge, AuxData, CommitmentKey, Transcript, BASE_PRIME, DEFAULT_BOUND,
    DEFAULT_LIMBS, DEFAULT_WEIGHT,
};
use std::time::Instant;

/// Chunks the witness is split into; also the number of challenges the fold consumes.
const R: usize = 256;
/// log2 of the witness in `F162` elements.
const LOG_E: u32 = 18;
/// The two halves of the evaluation point: `2^LR` chunks of `2^LW` `F162` each.
const LW: usize = 10;
const LR: usize = 8;
/// The core the process pins itself to.
const CPU: usize = 2;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Best of `reps` wall milliseconds, and the last value produced.
fn best_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let v = std::hint::black_box(f());
        best = best.min(ms(t0));
        out = Some(v);
    }
    (best, out.unwrap())
}

fn main() {
    pin(CPU);
    bin_ntt::f162::assert_layout();

    let nf162 = 1usize << LOG_E;
    let mut rng = Rng::new(0x5EED_5EED);
    let witness: Vec<F162> = (0..nf162).map(|_| F162::random(&mut rng)).collect();
    let ck = CommitmentKey::random(nf162 / R, 0xF01D, &DEFAULT_LIMBS);
    assert_eq!(ck.len_f162(), 1 << LW, "the eval stage wants 2^18 F162 in 256 chunks");

    let names: Vec<String> = core::iter::once(BASE_PRIME.to_string())
        .chain(DEFAULT_LIMBS.iter().map(|l| l.prime().to_string()))
        .collect();
    println!("bin-ntt, cpu {CPU}, one thread, limbs {}", names.join(", "));
    println!(
        "witness: 2^{LOG_E} F162 = {} ring elements of R_648 = {:.1} MB",
        nf162 / 4,
        (nf162 * 24) as f64 / 1e6
    );
    println!(
        "key: {:.1} MB, r = {R} chunks of {} ring elements, weight-{DEFAULT_WEIGHT} challenges\n",
        ck.bytes() as f64 / 1e6,
        ck.len_ring()
    );

    // The commitment, keeping the witness transform the fold consumes.
    let mut aux = AuxData::new(ck.len_ring(), R, ck.limbs());
    for _ in 0..4 {
        std::hint::black_box(ck.commit_into_aux(&witness, R, &mut aux));
    }
    let (commit_ms, c) = best_of(3, || ck.commit_into_aux(&witness, R, &mut aux));

    // The transcript and the challenges. The rejection tables (`challenge`'s `LazyLock`s) are
    // built once per process, so the step is timed the way every other one is: best of a few.
    let mut attempts = 0u64;
    let (chal_ms, ch) = best_of(3, || {
        let mut t = Transcript::new(b"bin-ntt/fold");
        for j in 0..R {
            t.absorb_elements(c.column(j));
        }
        attempts = 0;
        (0..R)
            .map(|_| {
                let (x, a) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
                attempts += a;
                x
            })
            .collect::<Vec<_>>()
    });

    // The fold.
    let (_, out) = best_of(4, || fold(&ck, &aux, &ch));
    let f = out.timings;

    // The statement and the left-expansion it is proved against.
    let mut te = Transcript::new(b"bin-ntt/eval");
    for j in 0..1 << LR {
        te.absorb_elements(c.column(j));
    }
    let (point_ms, point) = best_of(3, || sample_point::<LW, LR>(&mut te.clone()));
    let (mle_ms, claim) = best_of(3, || evaluate_mle(&witness, &point));
    let (le_ms, lx) = best_of(3, || left_expand(&witness, &point.r0));
    let u = lx.u;

    // The verifier: every check recomputed from v, u and the commitments alone.
    let raw = RawCommitments::from_aux(&aux);
    let (cc_ms, ok1) = best_of(3, || check_claim(&u, &point.r1, claim));
    let (fb_ms, folded) = best_of(3, || fold_binary(&u, &ch));
    let (vf_ms, ok2) = best_of(3, || verify_fold(&ck, &raw, &ch, &out.v));
    let (vb_ms, ok3) = best_of(3, || verify_binary(&point.r0, &out.v, folded));
    assert!(ok1 && ok2 && ok3, "the verifier rejected an honest transcript");

    println!("PROVER");
    println!(
        "  commit_into_aux          {commit_ms:>8.2} ms   ({:.0} MB of witness transform kept)",
        aux.bytes() as f64 / 1e6
    );
    println!("  left_expand              {le_ms:>8.2} ms   (u = B W, 2^18 F162 products)");
    println!(
        "  challenges               {chal_ms:>8.2} ms   ({R} sampled, {:.1} attempts each)",
        attempts as f64 / R as f64
    );
    println!("  fold                     {:>8.2} ms", f.total_ms);
    for (name, x) in [
        ("challenge NTTs", f.challenge_ntt_ms),
        ("accumulation", f.accumulate_ms),
        ("inverse NTT (q1)", f.inverse_ntt_ms),
        ("forward NTT (limbs)", f.forward_add_ms),
        ("y = A v", f.y_ms),
    ] {
        println!("    {name:<22} {x:>8.2} ms");
    }
    println!(
        "PROVER total               {:>8.2} ms",
        commit_ms + le_ms + chal_ms + f.total_ms
    );
    println!(
        "v: {} ring elements of R_648 in coefficient form, max |coefficient| = {} (q1/2 = {:.1})",
        out.v.len(),
        out.max_abs_v,
        BASE_PRIME as f64 / 2.0
    );

    println!("\nSTATEMENT (not prover time)");
    println!("  sample_point             {:>8.1} us", 1e3 * point_ms);
    println!("  evaluate_mle             {:>8.1} us   (2^18 products, word-sliced)", 1e3 * mle_ms);
    println!("STATEMENT total            {:>8.1} us", 1e3 * (point_ms + mle_ms));

    println!("\nVERIFIER");
    for (name, x) in [
        ("check_claim", cc_ms),
        ("fold_binary", fb_ms),
        ("verify_fold", vf_ms),
        ("verify_binary", vb_ms),
    ] {
        println!("  {name:<24} {:>8.1} us", 1e3 * x);
    }
    println!(
        "VERIFIER total             {:>8.1} us",
        1e3 * (cc_ms + fb_ms + vf_ms + vb_ms)
    );
}

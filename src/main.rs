//! The reference usage: one round end to end, with the wall clock on every step.
//!
//! `cargo run --release --offline`, pinned with `taskset -c 2`. The configuration is the three
//! constants below; edit them to change it (`Params::basic()` is the same shape as the defaults).
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::time::Instant;

/// The seed the public matrix `A` is expanded from.
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
/// The seed the witness is drawn from.
const WITNESS_SEED: [u8; 32] = [0xC7; 32];
/// The core the process pins itself to.
const CPU: usize = 2;

/// The shape of the instance: 2^WITNESS_LOG_LEN elements of F162 in 2^COLUMN_LOG_LEN columns,
/// committed modulo the base modulus 3889 and every modulus listed in EXTRA_MODULI
/// (any subset of Modulus::{Q2917, Q4861, Q9721, Q12637}).
const WITNESS_LOG_LEN: u32 = 18;
const COLUMN_LOG_LEN: u32 = 8;
const EXTRA_MODULI: &[Modulus] = &[Modulus::Q9721];

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
        let value = std::hint::black_box(f());
        best = best.min(ms(t0));
        out = Some(value);
    }
    (best, out.unwrap())
}

fn row(name: &str, milliseconds: f64) {
    println!("  {name:<26}{milliseconds:>9.2} ms");
}

fn main() {
    pin(CPU);

    let params = Params::new(WITNESS_LOG_LEN, COLUMN_LOG_LEN, EXTRA_MODULI.to_vec()).expect("valid parameters");
    let public_parameters = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    // Prover: the commitment, and the workspace it leaves for the fold.
    let commit_start = Instant::now();
    let (commitment, opening) = prover.commit(&witness);
    let commit_ms = ms(commit_start);

    // Statement: the evaluation point the commitment binds, and the value claimed at it.
    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = best_of(3, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = best_of(3, || witness.mle_evaluate(&evaluation_point));

    // Prover: the row evaluation, and the challenges it is folded against.
    let (row_evaluate_ms, row_evaluation) = best_of(3, || witness.row_evaluate(&evaluation_point));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = best_of(3, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &row_evaluation)
    });

    // Prover: the fold, which consumes the opening and hands the workspace back.
    let fold_start = Instant::now();
    let folded_witness = prover.fold(opening, &folding_challenges);
    let fold_ms = ms(fold_start);

    // Verifier: the folded statement, and the two checks.
    let (fold_commitment_ms, folded_commitment) =
        best_of(3, || verifier.fold_commitment(&commitment, &folding_challenges));
    let (fold_row_ms, folded_row_value) = best_of(3, || {
        verifier.fold_row_evaluation(&row_evaluation, &folding_challenges)
    });
    let (verify_evaluation_ms, evaluation_ok) = best_of(3, || {
        verifier.verify_evaluation(&evaluation_point, &claimed_value, &row_evaluation)
    });
    let (verify_opening_ms, opening_ok) = best_of(3, || {
        verifier.verify_folded_opening(
            &folded_commitment,
            &folded_witness,
            &evaluation_point,
            &folded_row_value,
        )
    });

    let moduli: Vec<String> = commitment.moduli().iter().map(|q| q.to_string()).collect();
    println!("bin-ntt, core {CPU}, one thread, moduli {}", moduli.join(", "));
    println!(
        "witness: 2^{} F162 = {} ring elements of R_648, {} columns of {} F162",
        params.witness_log_len,
        params.witness_len() / 4,
        params.columns(),
        params.witness_len() / params.columns()
    );
    println!(
        "folded witness: {} ring elements of R_648, {} F162 of row evaluation\n",
        folded_witness.len(),
        row_evaluation.values().len()
    );

    println!("PROVER");
    row("commit", commit_ms);
    row("row_evaluate", row_evaluate_ms);
    row("fold", fold_ms);
    row("total", commit_ms + row_evaluate_ms + fold_ms);

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    let verifier_ms = challenges_ms
        + fold_commitment_ms
        + fold_row_ms
        + verify_evaluation_ms
        + verify_opening_ms;
    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("fold_commitment", fold_commitment_ms);
    row("fold_row_evaluation", fold_row_ms);
    row("verify_evaluation", verify_evaluation_ms);
    row("verify_folded_opening", verify_opening_ms);
    row("total", verifier_ms);

    println!(
        "\nverification: {}",
        match (evaluation_ok, opening_ok) {
            (Ok(()), Ok(())) => "accepted".to_string(),
            (e, o) => format!("rejected ({e:?}, {o:?})"),
        }
    );
}

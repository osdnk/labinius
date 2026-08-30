//! The reference usage: one round end to end in each mode, with the wall clock on every step.
//!
//! `cargo run --release --offline`, pinned with `taskset -c 2`. The configuration is the three
//! constants below; edit them to change it (`Params::basic()` is the same shape as the defaults).
use bin_ntt::labrador;
use bin_ntt::recursion::statement::{build, Masks, Opening, ProofPhi};
use bin_ntt::recursion::Instance;
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::time::Instant;

/// The seed the public matrix `A` is expanded from.
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
/// The seed the witness is drawn from.
const WITNESS_SEED: [u8; 32] = [0xC7; 32];
/// The core the process pins itself to.
const CPU: usize = 3;

/// The shape of the instance: 2^WITNESS_LOG_LEN elements of F162 in 2^COLUMN_LOG_LEN columns,
/// committed modulo the base modulus 3889 and every modulus listed in EXTRA_MODULI
/// (any subset of Modulus::{Q2917_Q_S, Q4861_Q_S, Q9721_FS_S, Q12637_Q_S}).
const WITNESS_LOG_LEN: u32 = 18;
const COLUMN_LOG_LEN: u32 = 8;
const EXTRA_MODULI: &[Modulus] = &[Modulus::Q9721_FS_S];

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

/// One measurement of a step that may only be run once.
fn once<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t0 = Instant::now();
    let value = std::hint::black_box(f());
    (ms(t0), value)
}

fn row(name: &str, milliseconds: f64) {
    println!("  {name:<28}{milliseconds:>9.2} ms");
}

/// Peak resident set size in MB, from `/proc/self/status`.
fn peak_rss() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn main() {
    pin(CPU);
    plain();
    recursive();
    println!("\npeak resident set: {:.0} MB", peak_rss());
}

fn shape(recursion: bool) -> Params {
    Params::new(WITNESS_LOG_LEN, COLUMN_LOG_LEN, EXTRA_MODULI.to_vec(), recursion)
        .expect("valid parameters")
}

fn plain() {
    let params = shape(false);
    let (setup_ms, public_parameters) =
        once(|| PublicParameters::from_seed(params.clone(), MATRIX_SEED));
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    let (commit_ms, (commitment, opening)) = once(|| prover.commit(&witness));

    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = best_of(3, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = best_of(3, || witness.mle_evaluate(&evaluation_point));
    let (row_evaluate_ms, row_evaluation) = best_of(3, || witness.row_evaluate(&evaluation_point));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = best_of(3, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &row_evaluation)
    });

    let (fold_ms, folded_witness) = once(|| prover.fold(opening, &folding_challenges));
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
        "folded witness: {} ring elements of R_648, {} F162 of row evaluation",
        folded_witness.len(),
        row_evaluation.values().len()
    );
    println!("\n=== recursion off ===");
    row("public parameters", setup_ms);

    println!("\nPROVER");
    row("commit", commit_ms);
    row("row_evaluate", row_evaluate_ms);
    row("fold", fold_ms);
    row("total", commit_ms + row_evaluate_ms + fold_ms);

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("fold_commitment", fold_commitment_ms);
    row("fold_row_evaluation", fold_row_ms);
    row("verify_evaluation", verify_evaluation_ms);
    row("verify_folded_opening", verify_opening_ms);
    row(
        "total",
        challenges_ms + fold_commitment_ms + fold_row_ms + verify_evaluation_ms + verify_opening_ms,
    );
    println!(
        "\nwire: commitment {:.1} KB, row evaluation {:.1} KB, folded witness {:.1} KB",
        commitment.wire_bytes() as f64 / 1024.0,
        (row_evaluation.values().len() * 24) as f64 / 1024.0,
        (folded_witness.len() * 648 * 2) as f64 / 1024.0
    );
    println!(
        "verification: {}",
        match (evaluation_ok, opening_ok) {
            (Ok(()), Ok(())) => "accepted".to_string(),
            (e, o) => format!("rejected ({e:?}, {o:?})"),
        }
    );
}

fn recursive() {
    let params = shape(true);
    let (setup_ms, public_parameters) =
        once(|| PublicParameters::from_seed(params.clone(), MATRIX_SEED));
    let setup = public_parameters.recursion().expect("recursion is on").clone();
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    let (commit_ms, (commitment, opening)) = once(|| prover.commit(&witness));

    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = best_of(3, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = best_of(3, || witness.mle_evaluate(&evaluation_point));
    let (row_evaluate_ms, row_evaluation) = best_of(3, || witness.row_evaluate(&evaluation_point));
    let (left_ms, left) = best_of(3, || prover.commit_left_expansion(&row_evaluation));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = best_of(3, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &left)
    });

    let after_challenges = transcript.clone();
    let (prove_ms, proof) = once(|| {
        prover
            .prove_opening(
                &mut transcript,
                opening,
                &folding_challenges,
                &evaluation_point,
                &left,
                &row_evaluation,
                &claimed_value,
                &commitment,
            )
            .expect("the honest fold is within its cap")
    });

    let mut check = after_challenges.clone();
    let (rebuild_ms, statement) = once(|| {
        verifier
            .opening_statement(
                &mut check,
                &commitment,
                &left,
                &evaluation_point,
                &claimed_value,
                &folding_challenges,
                &proof,
            )
            .expect("the caps and the no-wrap bound hold")
    });
    let (labrador_verify_ms, verified) = once(|| labrador::verify(&statement, proof.handle()));
    drop(statement);

    // The same prover work again, step by step, for the breakdown of prove_opening.
    let (_, (_, second)) = once(|| prover.commit(&witness));
    let residues = second.residues().expect("recursion is on").clone();
    let (second_fold_ms, folded) = once(|| prover.fold(second, &folding_challenges));
    let (encode_ms, instance) = once(|| {
        Instance::new(
            &setup,
            &residues,
            &folded,
            &row_evaluation,
            &folding_challenges,
            &evaluation_point,
            &claimed_value,
        )
    });
    let (witness_ms, encoded) = once(|| instance.witness());
    let rest: Vec<&[i16]> = setup.rest.iter().map(|&i| encoded.vectors[i].as_slice()).collect();
    let (t_r_ms, t_r) = once(|| std::sync::Arc::new(setup.key_r.commit_blocks(&rest)));
    let mut masking = after_challenges.clone();
    let (mask_ms, masks) = once(|| Masks::squeeze(&setup, &mut masking));
    let (phi_ms, phi) = once(|| ProofPhi::new(&setup, &instance));
    let norms: Vec<u64> = instance.vectors.iter().map(|v| v.betasq()).collect();
    let (statement_ms, statement) = once(|| {
        build(
            &setup,
            &instance,
            &phi,
            Opening { t_y: commitment.t_y(), t_u: left.t_u(), t_r: &t_r, norms: &norms },
            masks,
            [0u8; 32],
        )
    });
    let (labrador_prove_ms, _) =
        once(|| labrador::prove(&statement, &labrador::Witness::new(encoded.vectors)).expect("prove"));
    let phi_bytes = phi.footprint();

    // The two halves of the verifier's rebuild that the prover does not run.
    let (layout_ms, layout) = once(|| {
        Instance::layout(&setup, &folding_challenges, &evaluation_point, &claimed_value)
    });
    let (bound_ms, cleared) = once(|| layout.clears());
    assert!(cleared, "the no-wrap bound holds");

    println!("\n=== recursion on ===");
    row("public parameters", setup_ms);
    println!(
        "  key-time buffers            {:>9.0} MB",
        setup.footprint() as f64 / (1 << 20) as f64
    );
    println!(
        "  witness: {} vectors, {} polynomials of Z_Q[X]/(X^64+1)",
        setup.ranks.len(),
        setup.ranks.iter().sum::<usize>()
    );
    println!(
        "  commitment ranks: kappa_Y {}, kappa_u {}, kappa_R {}",
        setup.key_y.rank(),
        setup.key_u.rank(),
        setup.key_r.rank()
    );

    println!("\nPROVER");
    row("commit (with T_Y)", commit_ms);
    row("row_evaluate", row_evaluate_ms);
    row("commit_left_expansion", left_ms);
    row("prove_opening", prove_ms);
    row("  fold", second_fold_ms);
    row("  encoding", encode_ms + witness_ms);
    row("  T_R", t_r_ms);
    row("  masks", mask_ms);
    row("  constraint phi", phi_ms);
    row("  statement build", statement_ms);
    row("  labrador::prove", labrador_prove_ms);
    row("total", commit_ms + row_evaluate_ms + left_ms + prove_ms);

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("statement rebuild", rebuild_ms);
    row("  layout", layout_ms);
    row("  no-wrap bound", bound_ms);
    row("  constraint phi", phi_ms);
    row("  statement build", statement_ms);
    row("labrador::verify", labrador_verify_ms);
    row("total", challenges_ms + rebuild_ms + labrador_verify_ms);

    println!(
        "\nwire: T_Y {:.1} KB, T_u {:.1} KB, T_R {:.1} KB, norms {:.1} KB, labrador {:.1} KB",
        commitment.wire_bytes() as f64 / 1024.0,
        left.wire_bytes() as f64 / 1024.0,
        (proof.t_r().len() * 64 * 6) as f64 / 1024.0,
        (proof.norms().len() * 8) as f64 / 1024.0,
        proof.labrador_kb()
    );
    println!(
        "proof: {:.1} KB total, per-proof phi {:.0} MB",
        (commitment.wire_bytes() + left.wire_bytes() + proof.wire_bytes()) as f64 / 1024.0,
        phi_bytes as f64 / (1 << 20) as f64
    );
    println!(
        "verification: {}",
        match verified {
            Ok(()) => "accepted".to_string(),
            Err(e) => format!("rejected ({e})"),
        }
    );
}

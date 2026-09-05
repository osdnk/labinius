//! The reference usage: one round end to end in each mode, with the wall clock on every step.
//!
//! `cargo run --release --offline`, pinned with `taskset -c 2`.
use bin_ntt::scheme::suite_from_args;
use bin_ntt::{Opening, OpeningMessage, Suite};
use bin_ntt_bench::{duration_ms, median_of, once, peak_rss, pin, row};
use bin_ntt::wire;
use bin_ntt::{Params, Prover, PublicParameters, Transcript, Verifier, Witness};

/// The seed the public matrix `A` is expanded from.
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
/// The seed the witness is drawn from.
const WITNESS_SEED: [u8; 32] = [0xC7; 32];
/// The core the process pins itself to.
const CPU: usize = 3;

/// Rejections per short challenge at the default weight and bound, over `SAMPLES` transcripts.
const SAMPLES: usize = 1000;

fn stats() {
    use bin_ntt::challenge::{sample_short_challenge, DEFAULT_BOUND, DEFAULT_WEIGHT};
    let mut attempts = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES as u64 {
        let mut t = Transcript::new(b"bin-ntt/stats");
        t.absorb_u64(i);
        attempts.push(sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).1);
    }
    attempts.sort_unstable();
    let total: u64 = attempts.iter().sum();
    let acceptance = SAMPLES as f64 / total as f64;
    println!(
        "=== short challenge: weight {DEFAULT_WEIGHT}, canonical bound {DEFAULT_BOUND}, {SAMPLES} samples ==="
    );
    println!(
        "  acceptance rate: {:.4} (1 in {:.2})",
        acceptance,
        1.0 / acceptance
    );
    println!();
}

fn main() {
    let suite = suite_from_args();
    println!("=== size {} ===", suite.name);

    pin(CPU);
    stats();
    #[cfg(feature = "labrador")]
    {
        recursive(suite);
    }
    #[cfg(not(feature = "labrador"))]
    {
        plain(suite);
        plain_bd(suite);
    }
    println!("\npeak resident set: {:.0} MB", peak_rss());
}

fn plain(suite: &Suite) {
    let params = Params::sized(suite, Opening::Clear);
    let (setup_ms, public_parameters) =
        once(|| PublicParameters::from_seed(params.clone(), MATRIX_SEED));
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    let (commit_ms, (commitment, opening)) = once(|| prover.commit(&witness));

    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = median_of(10, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = median_of(10, || witness.mle_evaluate(&evaluation_point));
    let (row_evaluate_ms, row_evaluation) =
        median_of(10, || witness.row_evaluate(&evaluation_point));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = median_of(10, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &row_evaluation)
    });

    let (fold_ms, folded_witness) = once(|| prover.fold(opening, &folding_challenges));

    // What the prover puts on the wire, and what the verifier takes off it: everything below
    // this point runs against the decoded objects, so the round trip is on the real path.
    let (pack_ms, (commitment_wire, row_wire)) = median_of(10, || {
        (
            wire::pack_commitment(&commitment),
            wire::pack_row_evaluation(&row_evaluation),
        )
    });
    let (encode_ms, fold_wire) =
        median_of(10, || wire::encode(&folded_witness, params.base.prime()));
    let (decode_ms, (commitment, row_evaluation, folded_witness)) = median_of(10, || {
        (
            wire::unpack_commitment(&params, &commitment_wire).expect("a commitment off the wire"),
            wire::unpack_row_evaluation(&row_wire, params.columns())
                .expect("a row evaluation off the wire"),
            wire::decode(&fold_wire).expect("a folded witness off the wire"),
        )
    });

    let (fold_commitment_ms, folded_commitment) = median_of(10, || {
        verifier.fold_commitment(&commitment, &folding_challenges)
    });
    let (fold_row_ms, folded_row_value) = median_of(10, || {
        verifier.fold_row_evaluation(&row_evaluation, &folding_challenges)
    });
    let (verify_evaluation_ms, evaluation_ok) = median_of(10, || {
        verifier.verify_evaluation(&evaluation_point, &claimed_value, &row_evaluation)
    });
    let (verify_opening_ms, opening_ok) = median_of(10, || {
        verifier.verify_opening(
            &commitment,
            &folding_challenges,
            &evaluation_point,
            OpeningMessage::Clear {
                folded_commitment: &folded_commitment,
                folded_witness: &folded_witness,
                folded_row_value: &folded_row_value,
            },
        )
    });

    let moduli: Vec<String> = commitment.moduli().iter().map(|q| q.to_string()).collect();
    println!(
        "bin-ntt, core {CPU}, one thread, moduli {}",
        moduli.join(", ")
    );
    println!("\n=== recursion off ===");
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
    row("public parameters", setup_ms);

    println!("\nPROVER");
    row("commit", commit_ms);
    row("row_evaluate", row_evaluate_ms);
    row("fold", fold_ms);
    row("pack commitment, row", pack_ms);
    row("encode folded witness", encode_ms);
    row(
        "total",
        commit_ms + row_evaluate_ms + fold_ms + pack_ms + encode_ms,
    );
    row(
        "total except encode",
        commit_ms + row_evaluate_ms + fold_ms + pack_ms,
    );

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("decode", decode_ms);
    row("fold_commitment", fold_commitment_ms);
    row("fold_row_evaluation", fold_row_ms);
    row("verify_evaluation", verify_evaluation_ms);
    row("verify_folded_opening", verify_opening_ms);
    row(
        "total",
        challenges_ms
            + decode_ms
            + fold_commitment_ms
            + fold_row_ms
            + verify_evaluation_ms
            + verify_opening_ms,
    );
    row(
        "total except decode",
        challenges_ms
            // + decode_ms
            + fold_commitment_ms
            + fold_row_ms
            + verify_evaluation_ms
            + verify_opening_ms,
    );
    println!(
        "\nwire: commitment {:.1} KB, row evaluation {:.1} KB, folded witness {:.1} KB, TOTAL {:.1} KB",
        commitment_wire.len() as f64 / 1024.0,
        row_wire.len() as f64 / 1024.0,
        fold_wire.len() as f64 / 1024.0,
        (commitment_wire.len() + row_wire.len() + fold_wire.len()) as f64 / 1024.0
    );
    println!(
        "verification: {}",
        match (evaluation_ok, opening_ok) {
            (Ok(()), Ok(_)) => "accepted".to_string(),
            (e, o) => format!("rejected ({e:?}, {o:?})"),
        }
    );
}

fn plain_bd(suite: &Suite) {
    let params = Params::sized(
        suite,
        Opening::BitDropped {
            bits: suite.dropped_bits,
        },
    );
    let (setup_ms, public_parameters) =
        once(|| PublicParameters::from_seed(params.clone(), MATRIX_SEED));
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    let (commit_ms, (commitment, opening)) = once(|| prover.commit(&witness));

    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = median_of(10, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = median_of(10, || witness.mle_evaluate(&evaluation_point));
    let (row_evaluate_ms, row_evaluation) =
        median_of(10, || witness.row_evaluate(&evaluation_point));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = median_of(10, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &row_evaluation)
    });

    let (fold_ms, folded_witness) = once(|| prover.fold(opening, &folding_challenges));

    let (pack_ms, (commitment_wire, row_wire)) = median_of(10, || {
        (
            wire::pack_commitment(&commitment),
            wire::pack_row_evaluation(&row_evaluation),
        )
    });
    let (encode_ms, fold_wire) =
        median_of(10, || wire::encode(&folded_witness, params.base.prime()));
    let (decode_ms, (commitment, row_evaluation, folded_witness)) = median_of(10, || {
        (
            wire::unpack_commitment(&params, &commitment_wire).expect("a commitment off the wire"),
            wire::unpack_row_evaluation(&row_wire, params.columns())
                .expect("a row evaluation off the wire"),
            wire::decode(&fold_wire).expect("a folded witness off the wire"),
        )
    });

    let (fold_row_ms, folded_row_value) = median_of(10, || {
        verifier.fold_row_evaluation(&row_evaluation, &folding_challenges)
    });
    let (verify_evaluation_ms, evaluation_ok) = median_of(10, || {
        verifier.verify_evaluation(&evaluation_point, &claimed_value, &row_evaluation)
    });
    let (verify_opening_ms, opening_ok) = median_of(10, || {
        verifier.verify_opening(
            &commitment,
            &folding_challenges,
            &evaluation_point,
            OpeningMessage::BitDropped {
                folded_witness: &folded_witness,
                folded_row_value: &folded_row_value,
            },
        )
    });

    let mut tampered = folded_witness.clone();
    tampered.elements_mut()[0].v[0] = tampered.elements_mut()[0].v[0].wrapping_add(1000);
    let tampered_ok = verifier.verify_opening(
        &commitment,
        &folding_challenges,
        &evaluation_point,
        OpeningMessage::BitDropped {
            folded_witness: &tampered,
            folded_row_value: &folded_row_value,
        },
    );

    println!("\n=== plain-bd ===");
    println!(
        "witness: 2^{} F162 = {} ring elements of R_648, {} columns of {} F162",
        params.witness_log_len,
        params.witness_len() / 4,
        params.columns(),
        params.witness_len() / params.columns()
    );
    println!(
        "dropped bits {}, residual cap {} over the expectation {:.3e}",
        params.dropped_bits(),
        params.bd_cap(),
        bin_ntt::bd::expected_normsq(params.columns(), params.dropped_bits())
    );
    row("public parameters", setup_ms);

    println!("\nPROVER");
    row("commit (with the digits)", commit_ms);
    row("row_evaluate", row_evaluate_ms);
    row("fold", fold_ms);
    row("pack commitment, row", pack_ms);
    row("encode folded witness", encode_ms);
    row(
        "total",
        commit_ms + row_evaluate_ms + fold_ms + pack_ms + encode_ms,
    );
    row(
        "total except encode",
        commit_ms + row_evaluate_ms + fold_ms + pack_ms,
    );

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("decode", decode_ms);
    row("fold_row_evaluation", fold_row_ms);
    row("verify_evaluation", verify_evaluation_ms);
    row("verify_folded_opening", verify_opening_ms);
    row(
        "total",
        challenges_ms + decode_ms + fold_row_ms + verify_evaluation_ms + verify_opening_ms,
    );
    row(
        "total except decode",
        challenges_ms + fold_row_ms + verify_evaluation_ms + verify_opening_ms,
    );
    println!(
        "\nwire: commitment {:.1} KB, row evaluation {:.1} KB, folded witness {:.1} KB, TOTAL {:.1} KB",
        commitment_wire.len() as f64 / 1024.0,
        row_wire.len() as f64 / 1024.0,
        fold_wire.len() as f64 / 1024.0,
        (commitment_wire.len() + row_wire.len() + fold_wire.len()) as f64 / 1024.0
    );
    println!(
        "verification: {}",
        match (evaluation_ok, opening_ok) {
            (Ok(()), Ok(_)) => "accepted".to_string(),
            (e, o) => format!("rejected ({e:?}, {o:?})"),
        }
    );
    println!(
        "tampered fold: {}",
        match tampered_ok {
            Ok(_) => "ACCEPTED".to_string(),
            Err(e) => format!("rejected ({e})"),
        }
    );
}

fn recursive(suite: &Suite) {
    let params = Params::sized(suite, Opening::Recursive);
    let (setup_ms, public_parameters) =
        once(|| PublicParameters::from_seed(params.clone(), MATRIX_SEED));
    let setup = public_parameters
        .recursion()
        .expect("recursion is on")
        .clone();
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&public_parameters);
    let verifier = Verifier::new(&public_parameters);

    let (commit_ms, (commitment, opening)) = once(|| prover.commit(&witness));

    let mut transcript = Transcript::new(b"bin-ntt/reference");
    let start = transcript.clone();
    let (evaluation_point_ms, evaluation_point) = median_of(10, || {
        transcript = start.clone();
        verifier.derive_evaluation_point(&mut transcript, &commitment)
    });
    let (mle_ms, claimed_value) = median_of(10, || witness.mle_evaluate(&evaluation_point));
    let (row_evaluate_ms, row_evaluation) =
        median_of(10, || witness.row_evaluate(&evaluation_point));
    let (left_ms, left) = median_of(10, || prover.commit_left_expansion(&row_evaluation));
    let after_point = transcript.clone();
    let (challenges_ms, folding_challenges) = median_of(10, || {
        transcript = after_point.clone();
        verifier.derive_folding_challenges(&mut transcript, &left)
    });

    let mut check = transcript.clone();
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
    let prove = *proof.timings();
    let (verify_ms, verified) = once(|| {
        verifier.verify_opening(
            &commitment,
            &folding_challenges,
            &evaluation_point,
            OpeningMessage::Recursive {
                transcript: &mut check,
                left: &left,
                claimed_value: &claimed_value,
                proof: &proof,
            },
        )
    });
    let verify = verified.unwrap_or_default();

    println!("\n=== recursion on ===");
    println!(
        "witness: 2^{} F162 = {} ring elements of R_648, {} columns of {} F162",
        params.witness_log_len,
        params.witness_len() / 4,
        params.columns(),
        params.witness_len() / params.columns()
    );

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
    row("  fold", duration_ms(prove.fold));
    row("  encoding", duration_ms(prove.encoding + prove.witness));
    row("  T_R", duration_ms(prove.t_r));
    row("  masks", duration_ms(prove.masks));
    row("  constraint phi", duration_ms(prove.phi));
    row("  statement build", duration_ms(prove.statement));
    row("  labrador::prove", duration_ms(prove.labrador));
    row("total", commit_ms + row_evaluate_ms + left_ms + prove_ms);

    println!("\nSTATEMENT");
    row("derive_evaluation_point", evaluation_point_ms);
    row("mle_evaluate", mle_ms);
    row("total", evaluation_point_ms + mle_ms);

    println!("\nVERIFIER");
    row("derive_folding_challenges", challenges_ms);
    row("statement rebuild", duration_ms(verify.rebuild));
    row("  layout", duration_ms(verify.layout));
    row("  no-wrap bound", duration_ms(verify.bound));
    row("  constraint phi", duration_ms(verify.phi));
    row("  statement build", duration_ms(verify.statement));
    row("labrador::verify", duration_ms(verify.labrador));
    row("total", challenges_ms + verify_ms);

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
        proof.phi_footprint() as f64 / (1 << 20) as f64
    );
    println!(
        "verification: {}",
        match verified {
            Ok(_) => "accepted".to_string(),
            Err(e) => format!("rejected ({e})"),
        }
    );
}

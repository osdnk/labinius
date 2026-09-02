//! The reference usage: one round end to end in each mode, with the wall clock on every step.
//!
//! `cargo run --release --offline`, pinned with `taskset -c 2`. The configuration is the three
//! constants below; edit them to change it (`Params::basic()` is the same shape as the defaults).
use bin_ntt::wire;
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::time::{Duration, Instant};

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
/// The clear-text opening ships the commitment and the folded witness, whose sizes trade at
/// `columns` against `witness / columns`; 2^7 columns sits at that optimum. The recursion's
/// cost grows with the column length instead, so it stays at 2^8.
const COLUMN_LOG_LEN_CLEAR: u32 = 7;
const COLUMN_LOG_LEN_RECURSIVE: u32 = 8;
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

/// Median of `reps` wall milliseconds, and the last value produced. `reps` is odd, so the median
/// is a measured sample rather than an average of two.
fn median_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut samples = Vec::with_capacity(reps);
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let value = std::hint::black_box(f());
        samples.push(ms(t0));
        out = Some(value);
    }
    samples.sort_by(f64::total_cmp);
    (samples[reps / 2], out.unwrap())
}

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
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
    let is_sizes: bool = {
        #[cfg(feature = "sizes")]
        {
            println!("=== sizes feature enabled ===");
            true
        }
        #[cfg(not(feature = "sizes"))]
        {
            false
        }
    };

    let is_sizem: bool = {
        #[cfg(feature = "sizem")]
        {
            println!("=== sizem feature enabled ===");
            true
        }
        #[cfg(not(feature = "sizem"))]
        {
            false
        }
    };

    let is_sizel: bool = {
        #[cfg(feature = "sizel")]
        {
            println!("=== sizel feature enabled ===");
            true
        }
        #[cfg(not(feature = "sizel"))]
        {
            false
        }
    };

    let sizes = (is_sizes as u8) + (is_sizem as u8) + (is_sizel as u8);
    if sizes != 1 {
        panic!("only one of the features `sizes`, `sizem`, or `sizel` must be enabled at once");
    }



    pin(CPU);
    stats();
    #[cfg(feature = "labrador")]
    {
        recursive();
    }
    #[cfg(not(feature = "labrador"))]
    {
        plain();
    }
    println!("\npeak resident set: {:.0} MB", peak_rss());
}

fn shape(recursion: bool) -> Params {
    if recursion {
        #[cfg(feature = "sizes")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN,
                COLUMN_LOG_LEN_RECURSIVE,
                Modulus::Q3889_FS_S,
                vec![Modulus::Q9721_FS_S],
                true,
            )
            .expect("valid parameters")
        }
        #[cfg(feature = "sizem")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN + 2,
                COLUMN_LOG_LEN_RECURSIVE + 1,
                Modulus::Q3889_FS_S,
                vec![Modulus::Q9721_FS_S],
                true,
            )
            .expect("valid parameters")
        }
        #[cfg(feature = "sizel")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN + 4,
                COLUMN_LOG_LEN_RECURSIVE + 2,
                Modulus::Q17497_FS_L,
                vec![Modulus::Q19441_FS_L], 
                true,
            )
            .expect("valid parameters");
        }
        panic!("you should never be here");
    } else {
        #[cfg(feature = "sizes")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN,
                COLUMN_LOG_LEN_CLEAR,
                Modulus::Q3889_FS_S,
                vec![Modulus::Q9721_FS_S],
                false,
            )
            .expect("valid parameters")
        }
        #[cfg(feature = "sizem")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN + 2,
                COLUMN_LOG_LEN_CLEAR + 1,
                Modulus::Q3889_FS_S,
                vec![Modulus::Q9721_FS_S],
                false,
            )
            .expect("valid parameters")
        }
        #[cfg(feature = "sizel")]
        {
            return Params::with_base(
                WITNESS_LOG_LEN + 4,
                COLUMN_LOG_LEN_CLEAR + 2,
                Modulus::Q17497_FS_L,
                vec![Modulus::Q19441_FS_L],
                false,
            )
            .expect("valid parameters");
        }
        panic!("you should never be here");
    }
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
        verifier.verify_folded_opening(
            &folded_commitment,
            &folded_witness,
            &evaluation_point,
            &folded_row_value,
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
    row("total", commit_ms + row_evaluate_ms + fold_ms + pack_ms + encode_ms);
    row("total except encode", commit_ms + row_evaluate_ms + fold_ms + pack_ms);

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
            (Ok(()), Ok(())) => "accepted".to_string(),
            (e, o) => format!("rejected ({e:?}, {o:?})"),
        }
    );
}

fn recursive() {
    let params = shape(true);
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
            &mut check,
            &commitment,
            &left,
            &evaluation_point,
            &claimed_value,
            &folding_challenges,
            &proof,
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

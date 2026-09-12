//! `cargo run --release --offline --bin hashes-binius`, pinned with `taskset -c 3`.
use labinius_binius::stock::{Stock, LOG_INV_RATE};
use labinius_binius::{Circuit, Hash, Session, Sizes};
use labinius::scheme::suite_from_args;
use labinius::Suite;
use labinius_bench::{median_of, medians, once, peak_rss, pin, pinned, table_row as row, REPS};

/// The seed the public matrix `A` is expanded from.
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
/// The core the process pins itself to.
const CPU: usize = 3;


fn main() {
    let suite = suite_from_args();
    pin(CPU);
    for hash in Hash::ALL {
        compare(hash, suite);
        println!();
    }
}

fn compare(hash: Hash, suite: &Suite) {
    let len = hash.message_len(suite);
    let message: Vec<u8> = (0..len)
        .map(|i| (i as u32).wrapping_mul(2654435761) as u8)
        .collect();
    let (circuit_ms, circuit) = median_of(REPS, || Circuit::new(hash, len));
    let (witness_ms, witness) = median_of(REPS, || circuit.witness(&message));
    let constraint_system = circuit.constraint_system();

    // Stock binius64, the same circuit and witness through its own prover and verifier.
    let (stock_setup, stock) = median_of(REPS, || Stock::new(constraint_system.clone()));
    let ([stock_prove, stock_commit, stock_bitand, stock_shift, stock_pcs], stock_proof) =
        medians(REPS, || (), |()| {
            let (ms, (proof, phases)) = once(|| stock.prove(&witness));
            (
                [
                    ms,
                    phases.milliseconds("Commit witness"),
                    phases.milliseconds("[phase] BitAnd check"),
                    phases.milliseconds("[phase] Shift Reduction"),
                    phases.milliseconds("[phase] PCS Opening"),
                ],
                proof,
            )
        });
    let ([stock_verify, stock_verify_pcs], stock_ok) = medians(REPS, || (), |()| {
        let (ms, (ok, phases)) = once(|| stock.verify(witness.inout(), &stock_proof));
        ([ms, phases.milliseconds("[phase] Verify PCS Opening")], ok)
    });
    stock_ok.expect("stock binius64 verifies its own proof");
    let ([stock_iop, stock_native, stock_basefold], ()) = medians(REPS, || (), |()| {
        let stages = stock
            .verify_stages(witness.inout(), &stock_proof)
            .expect("stock binius64 verifies its own proof");
        (stages, ())
    });
    let stock_verify_reduce = stock_iop - stock_verify_pcs;

    // The same instance with our commitment, without and with the recursion.
    let (off_setup, mut off) = median_of(REPS, || {
        Session::new(constraint_system.clone(), suite, false, MATRIX_SEED)
    });
    let (on_setup, mut on) = median_of(REPS, || {
        Session::new(constraint_system.clone(), suite, true, MATRIX_SEED)
    });
    let (bd_setup, mut bd) =
        median_of(REPS, || Session::bd(constraint_system.clone(), suite, MATRIX_SEED));
    let prove = |session: &mut Session| {
        medians(REPS, || (), |()| {
            let (proof, timing, sizes) = session.prove(&witness, None);
            (timing, (proof, sizes))
        })
    };
    let (off_prover, (off_proof, off_sizes)) = prove(&mut off);
    let (on_prover, (on_proof, on_sizes)) = prove(&mut on);
    let (bd_prover, (bd_proof, bd_sizes)) = prove(&mut bd);
    let verify = |session: &Session, proof| {
        medians(REPS, || (), |()| {
            let timing = session
                .verify(witness.inout(), proof)
                .expect("the honest proof verifies");
            (timing, ())
        })
        .0
    };
    let off_verifier = verify(&off, &off_proof);
    let on_verifier = verify(&on, &on_proof);
    let bd_verifier = verify(&bd, &bd_proof);

    println!(
        "labinius over binius64 {}, core {}, one thread, size {}",
        hash.name(),
        pinned(),
        suite.name
    );
    println!(
        "{} of {len} bytes: {} {}, {} AND constraints, {} non-public words",
        hash.name(),
        hash.compressions(len),
        hash.unit(),
        constraint_system.and_constraints.len(),
        witness.non_public().len()
    );
    let moduli = |session: &Session| {
        let params = session.params();
        let mut all = vec![params.base];
        all.extend(params.extra_moduli.iter().copied());
        all.iter()
            .map(|q| q.prime().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "stock binius64 at the example's defaults: --log-inv-rate {LOG_INV_RATE}, \
         --hash-suite sha256, rayon off"
    );
    println!(
        "recursion off: 2^{} F162 in {} columns, moduli {}",
        off.params().witness_log_len,
        off.params().columns(),
        moduli(&off)
    );
    println!(
        "recursion on:  2^{} F162 in {} columns, moduli {}",
        on.params().witness_log_len,
        on.params().columns(),
        moduli(&on)
    );
    println!(
        "bit-drop:      2^{} F162 in {} columns, moduli {}, {} bits dropped",
        bd.params().witness_log_len,
        bd.params().columns(),
        moduli(&bd),
        bd.params().dropped_bits()
    );
    println!(
        "\n  {:<32}{:>13}{:>13}{:>13}{:>13}",
        "", "stock", "recursion", "recursion", ""
    );
    println!(
        "  {:<32}{:>13}{:>13}{:>13}{:>13}",
        "", "binius64", "off", "on", "bit-drop"
    );

    println!("\nSETUP (once, not per proof)");
    row("circuit", [Some(circuit_ms); 4], "ms");
    row(
        "commitment key and constraints",
        [
            Some(stock_setup),
            Some(off_setup),
            Some(on_setup),
            Some(bd_setup),
        ],
        "ms",
    );

    println!("\nPROVER");
    row("witness", [Some(witness_ms); 4], "ms");
    row(
        "packing",
        [
            None,
            Some(off_prover.pack),
            Some(on_prover.pack),
            Some(bd_prover.pack),
        ],
        "ms",
    );
    row(
        "commit",
        [
            Some(stock_commit),
            Some(off_prover.commit),
            Some(on_prover.commit),
            Some(bd_prover.commit),
        ],
        "ms",
    );
    row(
        "BitAnd check",
        [
            Some(stock_bitand),
            Some(off_prover.bitand),
            Some(on_prover.bitand),
            Some(bd_prover.bitand),
        ],
        "ms",
    );
    row(
        "shift reduction",
        [
            Some(stock_shift),
            Some(off_prover.shift),
            Some(on_prover.shift),
            Some(bd_prover.shift),
        ],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [
            Some(stock_pcs),
            Some(off_prover.switch),
            Some(on_prover.switch),
            Some(bd_prover.switch),
        ],
        "ms",
    );
    row(
        "opening",
        [
            None,
            Some(off_prover.opening),
            Some(on_prover.opening),
            Some(bd_prover.opening),
        ],
        "ms",
    );
    let listed = |t: &labinius_binius::ProverTiming| {
        t.pack + t.commit + t.bitand + t.shift + t.switch + t.opening
    };
    row(
        "rest",
        [
            Some(stock_prove - stock_commit - stock_bitand - stock_shift - stock_pcs),
            Some(off_prover.total - listed(&off_prover)),
            Some(on_prover.total - listed(&on_prover)),
            Some(bd_prover.total - listed(&bd_prover)),
        ],
        "ms",
    );
    row(
        "total",
        [
            Some(stock_prove),
            Some(off_prover.total),
            Some(on_prover.total),
            Some(bd_prover.total),
        ],
        "ms",
    );

    println!("\nVERIFIER");
    row(
        "read the commitment",
        [
            None,
            Some(off_verifier.commitment),
            Some(on_verifier.commitment),
            Some(bd_verifier.commitment),
        ],
        "ms",
    );
    row(
        "reductions",
        [
            Some(stock_verify_reduce),
            Some(off_verifier.reduce),
            Some(on_verifier.reduce),
            Some(bd_verifier.reduce),
        ],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [
            Some(stock_verify_pcs),
            Some(off_verifier.switch),
            Some(on_verifier.switch),
            Some(bd_verifier.switch),
        ],
        "ms",
    );
    row(
        "decode the opening",
        [
            None,
            Some(off_verifier.decode),
            None,
            Some(bd_verifier.decode),
        ],
        "ms",
    );
    row(
        "BaseFold / our opening",
        [
            Some(stock_basefold),
            Some(off_verifier.opening),
            Some(on_verifier.opening),
            Some(bd_verifier.opening),
        ],
        "ms",
    );
    row(
        "wiring check (native)",
        [
            Some(stock_native),
            Some(off_verifier.wiring),
            Some(on_verifier.wiring),
            Some(bd_verifier.wiring),
        ],
        "ms",
    );
    row(
        "total",
        [
            Some(stock_verify),
            Some(off_verifier.total),
            Some(on_verifier.total),
            Some(bd_verifier.total),
        ],
        "ms",
    );

    println!("\nSIZES");
    let kb = |b: usize| Some(b as f64 / 1024.0);
    row(
        "binius64 LIOP",
        [
            None,
            kb(off_sizes.liop),
            kb(on_sizes.liop),
            kb(bd_sizes.liop),
        ],
        "KB",
    );
    row(
        "cross-field switch",
        [
            None,
            kb(off_sizes.switch),
            kb(on_sizes.switch),
            kb(bd_sizes.switch),
        ],
        "KB",
    );
    row(
        "commitment (wire form)",
        [
            None,
            kb(off_sizes.commitment_wire),
            kb(on_sizes.commitment_wire),
            kb(bd_sizes.commitment_wire),
        ],
        "KB",
    );
    row(
        "opening",
        [
            None,
            kb(off_sizes.opening),
            kb(on_sizes.opening),
            kb(bd_sizes.opening),
        ],
        "KB",
    );
    let total = |s: &Sizes| kb(s.liop + s.switch + s.commitment_wire + s.opening);
    row(
        "total",
        [
            kb(stock_proof.len()),
            total(&off_sizes),
            total(&on_sizes),
            total(&bd_sizes),
        ],
        "KB",
    );

    println!(
        "  the tape carries T_Y as its polx image, {:.1} KB, rather than the wire form above",
        on_sizes.commitment as f64 / 1024.0
    );

    println!("\npeak resident set: {:.0} MB", peak_rss());
}

use labinius_flock::circuit::LOG_INV_RATE;
use labinius_flock::{Hash, Instance, ProverTiming, Session, Sizes};
use labinius::scheme::suite_from_args;
use labinius::Suite;
use labinius_bench::{once, peak_rss, pin, pinned, table_row as row};
use flock_transcript::challenger::FsChallenger;

const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const CPU: usize = 3;


fn encoded<T: serde::Serialize>(x: &T) -> usize {
    bincode::serialized_size(x).expect("the proof serializes") as usize
}

fn main() {
    std::env::set_var("RAYON_NUM_THREADS", "1");
    let suite = suite_from_args();
    pin(CPU);
    for hash in Hash::ALL {
        compare(hash, suite);
        println!();
    }
}

fn compare(hash: Hash, suite: &Suite) {
    let (setup_ms, instance) = once(|| Instance::new(hash, suite));
    let (witness_ms, witness) = once(|| instance.witness());

    let (union_prove, (union_proof, union_commitment, _)) = once(|| {
        let mut ch = FsChallenger::new(labinius_flock::DOMAIN);
        instance.stock_prove(&mut ch)
    });
    let (union_verify, union_ok) = once(|| {
        let mut ch = FsChallenger::new(labinius_flock::DOMAIN);
        instance.stock_verify(&union_commitment, &union_proof, &mut ch)
    });
    union_ok.expect("stock flock verifies its own proof");
    let union_size = encoded(&union_proof) + encoded(&union_commitment);
    drop(union_proof);

    let core_params = instance.core_params();
    let mut ch = FsChallenger::new(labinius_flock::DOMAIN);
    let core_witness = instance.witness();
    let (core_reduce, core) = once(|| instance.core_reduce(&core_params, core_witness, &mut ch));
    let (core_open, core_proof) = once(|| instance.core_open(&core_params, core, &mut ch));
    let mut ch = FsChallenger::new(labinius_flock::DOMAIN);
    let (core_verify_reduce, core_claims) =
        once(|| instance.core_verify_reduce(&core_params, &core_proof, &mut ch));
    let core_claims = core_claims.expect("stock flock replays its own reductions");
    let (core_verify_open, core_ok) =
        once(|| instance.core_verify_open(&core_params, &core_proof, &core_claims, &mut ch));
    core_ok.expect("stock flock verifies its own opening");
    let core_sizes = Sizes {
        commitment: core_proof.commitment.cap.len() * 32,
        zerocheck: encoded(&core_proof.zerocheck),
        lincheck: encoded(&core_proof.lincheck),
        switch: encoded(&core_proof.open.ring_switches) + encoded(&core_proof.open.batching_nonces),
        opening: encoded(&core_proof.open.ligerito),
    };
    drop(core_proof);

    let (off_setup, mut off) = once(|| Session::new(suite, false, MATRIX_SEED));
    let (on_setup, mut on) = once(|| Session::new(suite, true, MATRIX_SEED));
    let (bd_setup, mut bd) = once(|| Session::bd(suite, MATRIX_SEED));
    let (_, (off_proof, off_prover, off_sizes)) = once(|| off.prove(&instance, &witness));
    let (_, (on_proof, on_prover, on_sizes)) = once(|| on.prove(&instance, &witness));
    let (_, (bd_proof, bd_prover, bd_sizes)) = once(|| bd.prove(&instance, &witness));
    let off_verifier = off
        .verify(&instance, &off_proof)
        .expect("the honest proof verifies");
    let on_verifier = on
        .verify(&instance, &on_proof)
        .expect("the honest proof verifies");
    let bd_verifier = bd
        .verify(&instance, &bd_proof)
        .expect("the honest proof verifies");

    let r1cs = instance.r1cs();
    println!(
        "labinius over flock {}, core {}, one thread, size {}",
        hash.name(),
        pinned(),
        suite.name
    );
    println!(
        "{} compressions of {}: m = {}, k_log = {}, {} useful bits per block, {} committed",
        hash.compressions(suite),
        hash.name(),
        r1cs.m,
        r1cs.k_log,
        r1cs.useful_bits,
        witness.0.len()
    );
    let params = instance.pcs_params();
    println!(
        "stock union: log_inv_rate {LOG_INV_RATE}, profile {:?}, Ligerito over the compacted \
         stack, dense m = {} in {} F128",
        params.profile,
        params.m,
        params.msg_len_f128()
    );
    println!(
        "stock core:  log_inv_rate {LOG_INV_RATE}, profile {:?}, Ligerito over the padded \
         buffer, m = {} in {} F128",
        core_params.profile,
        core_params.m,
        core_params.msg_len_f128()
    );
    let moduli = |session: &Session| {
        let p = session.params();
        let mut all = vec![p.base];
        all.extend(p.extra_moduli.iter().copied());
        all.iter()
            .map(|q| q.prime().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
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
        "\n  {:<32}{:>13}{:>13}{:>13}{:>13}{:>13}",
        "", "stock", "stock", "recursion", "recursion", ""
    );
    println!(
        "  {:<32}{:>13}{:>13}{:>13}{:>13}{:>13}",
        "", "union", "core", "off", "on", "bit-drop"
    );

    println!("\nSETUP (once, not per proof)");
    row("R1CS and PCS parameters", [Some(setup_ms); 5], "ms");
    row(
        "commitment key",
        [None, None, Some(off_setup), Some(on_setup), Some(bd_setup)],
        "ms",
    );

    println!("\nPROVER");
    row("witness", [Some(witness_ms); 5], "ms");
    row(
        "lift to F162",
        [
            None,
            None,
            Some(off_prover.pack),
            Some(on_prover.pack),
            Some(bd_prover.pack),
        ],
        "ms",
    );
    row(
        "commit, bind, zerocheck, lincheck",
        [None, Some(core_reduce), None, None, None],
        "ms",
    );
    row(
        "commit",
        [
            None,
            None,
            Some(off_prover.commit),
            Some(on_prover.commit),
            Some(bd_prover.commit),
        ],
        "ms",
    );
    row(
        "bind the commitment",
        [
            None,
            None,
            Some(off_prover.bind),
            Some(on_prover.bind),
            Some(bd_prover.bind),
        ],
        "ms",
    );
    row(
        "zerocheck",
        [
            None,
            None,
            Some(off_prover.zerocheck),
            Some(on_prover.zerocheck),
            Some(bd_prover.zerocheck),
        ],
        "ms",
    );
    row(
        "lincheck",
        [
            None,
            None,
            Some(off_prover.lincheck),
            Some(on_prover.lincheck),
            Some(bd_prover.lincheck),
        ],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [
            None,
            None,
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
            Some(core_open),
            Some(off_prover.opening),
            Some(on_prover.opening),
            Some(bd_prover.opening),
        ],
        "ms",
    );
    let listed = |t: &ProverTiming| {
        t.pack + t.commit + t.bind + t.zerocheck + t.lincheck + t.switch + t.opening
    };
    row(
        "rest",
        [
            None,
            None,
            Some(off_prover.total - listed(&off_prover)),
            Some(on_prover.total - listed(&on_prover)),
            Some(bd_prover.total - listed(&bd_prover)),
        ],
        "ms",
    );
    row(
        "proof after the witness",
        [
            None,
            Some(core_reduce + core_open),
            Some(off_prover.total),
            Some(on_prover.total),
            Some(bd_prover.total),
        ],
        "ms",
    );
    row(
        "total, witness included",
        [
            Some(union_prove),
            Some(witness_ms + core_reduce + core_open),
            Some(witness_ms + off_prover.total),
            Some(witness_ms + on_prover.total),
            Some(witness_ms + bd_prover.total),
        ],
        "ms",
    );

    println!("\nVERIFIER");
    row(
        "zerocheck and lincheck",
        [
            None,
            Some(core_verify_reduce),
            Some(off_verifier.reduce),
            Some(on_verifier.reduce),
            Some(bd_verifier.reduce),
        ],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [
            None,
            None,
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
            None,
            Some(off_verifier.decode),
            None,
            Some(bd_verifier.decode),
        ],
        "ms",
    );
    row(
        "Ligerito / our opening",
        [
            None,
            Some(core_verify_open),
            Some(off_verifier.opening),
            Some(on_verifier.opening),
            Some(bd_verifier.opening),
        ],
        "ms",
    );
    row(
        "total",
        [
            Some(union_verify),
            Some(core_verify_reduce + core_verify_open),
            Some(off_verifier.total),
            Some(on_verifier.total),
            Some(bd_verifier.total),
        ],
        "ms",
    );

    println!("\nSIZES");
    let kb = |b: usize| Some(b as f64 / 1024.0);
    row(
        "zerocheck",
        [
            None,
            kb(core_sizes.zerocheck),
            kb(off_sizes.zerocheck),
            kb(on_sizes.zerocheck),
            kb(bd_sizes.zerocheck),
        ],
        "KB",
    );
    row(
        "lincheck",
        [
            None,
            kb(core_sizes.lincheck),
            kb(off_sizes.lincheck),
            kb(on_sizes.lincheck),
            kb(bd_sizes.lincheck),
        ],
        "KB",
    );
    row(
        "cross-field switch",
        [
            None,
            kb(core_sizes.switch),
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
            kb(core_sizes.commitment),
            kb(off_sizes.commitment),
            kb(on_sizes.commitment),
            kb(bd_sizes.commitment),
        ],
        "KB",
    );
    row(
        "opening",
        [
            None,
            kb(core_sizes.opening),
            kb(off_sizes.opening),
            kb(on_sizes.opening),
            kb(bd_sizes.opening),
        ],
        "KB",
    );
    let total = |s: &Sizes| kb(s.total());
    row(
        "total",
        [
            kb(union_size),
            total(&core_sizes),
            total(&off_sizes),
            total(&on_sizes),
            total(&bd_sizes),
        ],
        "KB",
    );

    println!("\npeak resident set: {:.0} MB", peak_rss());
}

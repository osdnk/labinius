use bin_ntt::flock::circuit::LOG_INV_RATE;
use bin_ntt::flock::{Hash, Instance, ProverTiming, Session, Sizes};
use bin_ntt::scheme::SIZE;
use flock_transcript::challenger::FsChallenger;
use std::time::Instant;

const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const CPU: usize = 3;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

fn once<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let start = Instant::now();
    let value = std::hint::black_box(f());
    (start.elapsed().as_secs_f64() * 1e3, value)
}

fn row(name: &str, values: [Option<f64>; 4], unit: &str) {
    print!("  {name:<32}");
    for value in values {
        match value {
            Some(v) => print!("{v:>13.2}"),
            None => print!("{:>13}", "—"),
        }
    }
    println!(" {unit}");
}

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

fn encoded<T: serde::Serialize>(x: &T) -> usize {
    bincode::serialized_size(x).expect("the proof serializes") as usize
}

fn main() {
    std::env::set_var("RAYON_NUM_THREADS", "1");
    pin(CPU);
    for hash in Hash::ALL {
        compare(hash);
        println!();
    }
}

fn compare(hash: Hash) {
    let (setup_ms, instance) = once(|| Instance::new(hash));
    let (witness_ms, witness) = once(|| instance.witness());

    let (union_prove, (union_proof, union_commitment, _)) = once(|| {
        let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
        instance.stock_prove(&mut ch)
    });
    let (union_verify, union_ok) = once(|| {
        let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
        instance.stock_verify(&union_commitment, &union_proof, &mut ch)
    });
    union_ok.expect("stock flock verifies its own proof");
    let union_size = encoded(&union_proof) + encoded(&union_commitment);
    drop(union_proof);

    let core_params = instance.core_params();
    let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
    let core_witness = instance.witness();
    let (core_reduce, core) = once(|| instance.core_reduce(&core_params, core_witness, &mut ch));
    let (core_open, core_proof) = once(|| instance.core_open(&core_params, core, &mut ch));
    let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
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

    let (off_setup, mut off) = once(|| Session::new(false, MATRIX_SEED));
    let (on_setup, mut on) = once(|| Session::new(true, MATRIX_SEED));
    let (_, (off_proof, off_prover, off_sizes)) = once(|| off.prove(&instance, &witness));
    let (_, (on_proof, on_prover, on_sizes)) = once(|| on.prove(&instance, &witness));
    let off_verifier = off
        .verify(&instance, &off_proof)
        .expect("the honest proof verifies");
    let on_verifier = on
        .verify(&instance, &on_proof)
        .expect("the honest proof verifies");

    let r1cs = instance.r1cs();
    println!(
        "bin-ntt over flock {}, core {CPU}, one thread, size {SIZE}",
        hash.name()
    );
    println!(
        "{} compressions of {}: m = {}, k_log = {}, {} useful bits per block, {} committed",
        hash.compressions(),
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
        "\n  {:<32}{:>13}{:>13}{:>13}{:>13}",
        "", "stock", "stock", "recursion", "recursion"
    );
    println!(
        "  {:<32}{:>13}{:>13}{:>13}{:>13}",
        "", "union", "core", "off", "on"
    );

    println!("\nSETUP (once, not per proof)");
    row("R1CS and PCS parameters", [Some(setup_ms); 4], "ms");
    row(
        "commitment key",
        [None, None, Some(off_setup), Some(on_setup)],
        "ms",
    );

    println!("\nPROVER");
    row("witness", [Some(witness_ms); 4], "ms");
    row(
        "lift to F162",
        [None, None, Some(off_prover.pack), Some(on_prover.pack)],
        "ms",
    );
    row(
        "commit, bind, zerocheck, lincheck",
        [None, Some(core_reduce), None, None],
        "ms",
    );
    row(
        "commit",
        [None, None, Some(off_prover.commit), Some(on_prover.commit)],
        "ms",
    );
    row(
        "bind the commitment",
        [None, None, Some(off_prover.bind), Some(on_prover.bind)],
        "ms",
    );
    row(
        "zerocheck",
        [
            None,
            None,
            Some(off_prover.zerocheck),
            Some(on_prover.zerocheck),
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
        ],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [None, None, Some(off_prover.switch), Some(on_prover.switch)],
        "ms",
    );
    row(
        "opening",
        [
            None,
            Some(core_open),
            Some(off_prover.opening),
            Some(on_prover.opening),
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
        ],
        "ms",
    );
    row(
        "decode the opening",
        [None, None, Some(off_verifier.decode), None],
        "ms",
    );
    row(
        "Ligerito / our opening",
        [
            None,
            Some(core_verify_open),
            Some(off_verifier.opening),
            Some(on_verifier.opening),
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
        ],
        "KB",
    );

    println!("\npeak resident set: {:.0} MB", peak_rss());
}

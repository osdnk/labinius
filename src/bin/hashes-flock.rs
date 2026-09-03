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

fn row(name: &str, values: [Option<f64>; 3], unit: &str) {
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

    let (stock_prove, (stock_proof, stock_commitment, _)) = once(|| {
        let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
        instance.stock_prove(&mut ch)
    });
    let (stock_verify, stock_ok) = once(|| {
        let mut ch = FsChallenger::new(bin_ntt::flock::DOMAIN);
        instance.stock_verify(&stock_commitment, &stock_proof, &mut ch)
    });
    stock_ok.expect("stock flock verifies its own proof");
    let stock_size = encoded(&stock_proof) + encoded(&stock_commitment);

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
        "stock flock at the setup's defaults: log_inv_rate {LOG_INV_RATE}, profile {:?}, \
         Ligerito over the union commit of dense m = {} in {} F128",
        params.profile,
        params.m,
        params.msg_len_f128()
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
        "\n  {:<32}{:>13}{:>13}{:>13}",
        "", "stock", "recursion", "recursion"
    );
    println!("  {:<32}{:>13}{:>13}{:>13}", "", "flock", "off", "on");

    println!("\nSETUP (once, not per proof)");
    row("R1CS and PCS parameters", [Some(setup_ms); 3], "ms");
    row(
        "commitment key",
        [None, Some(off_setup), Some(on_setup)],
        "ms",
    );

    println!("\nPROVER");
    row("witness", [Some(witness_ms); 3], "ms");
    row(
        "lift to F162",
        [None, Some(off_prover.pack), Some(on_prover.pack)],
        "ms",
    );
    row(
        "commit",
        [None, Some(off_prover.commit), Some(on_prover.commit)],
        "ms",
    );
    row(
        "bind the commitment",
        [None, Some(off_prover.bind), Some(on_prover.bind)],
        "ms",
    );
    row(
        "zerocheck",
        [None, Some(off_prover.zerocheck), Some(on_prover.zerocheck)],
        "ms",
    );
    row(
        "lincheck",
        [None, Some(off_prover.lincheck), Some(on_prover.lincheck)],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [None, Some(off_prover.switch), Some(on_prover.switch)],
        "ms",
    );
    row(
        "opening",
        [None, Some(off_prover.opening), Some(on_prover.opening)],
        "ms",
    );
    let listed = |t: &ProverTiming| {
        t.pack + t.commit + t.bind + t.zerocheck + t.lincheck + t.switch + t.opening
    };
    row(
        "rest",
        [
            None,
            Some(off_prover.total - listed(&off_prover)),
            Some(on_prover.total - listed(&on_prover)),
        ],
        "ms",
    );
    row(
        "proof after the witness",
        [None, Some(off_prover.total), Some(on_prover.total)],
        "ms",
    );
    row(
        "total, witness included",
        [
            Some(stock_prove),
            Some(witness_ms + off_prover.total),
            Some(witness_ms + on_prover.total),
        ],
        "ms",
    );

    println!("\nVERIFIER");
    row(
        "zerocheck and lincheck",
        [None, Some(off_verifier.reduce), Some(on_verifier.reduce)],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [None, Some(off_verifier.switch), Some(on_verifier.switch)],
        "ms",
    );
    row(
        "decode the opening",
        [None, Some(off_verifier.decode), None],
        "ms",
    );
    row(
        "Ligerito / our opening",
        [None, Some(off_verifier.opening), Some(on_verifier.opening)],
        "ms",
    );
    row(
        "total",
        [
            Some(stock_verify),
            Some(off_verifier.total),
            Some(on_verifier.total),
        ],
        "ms",
    );

    println!("\nSIZES");
    let kb = |b: usize| Some(b as f64 / 1024.0);
    row(
        "zerocheck",
        [None, kb(off_sizes.zerocheck), kb(on_sizes.zerocheck)],
        "KB",
    );
    row(
        "lincheck",
        [None, kb(off_sizes.lincheck), kb(on_sizes.lincheck)],
        "KB",
    );
    row(
        "cross-field switch",
        [None, kb(off_sizes.switch), kb(on_sizes.switch)],
        "KB",
    );
    row(
        "commitment (wire form)",
        [None, kb(off_sizes.commitment), kb(on_sizes.commitment)],
        "KB",
    );
    row(
        "opening",
        [None, kb(off_sizes.opening), kb(on_sizes.opening)],
        "KB",
    );
    let total = |s: &Sizes| kb(s.total());
    row(
        "total",
        [kb(stock_size), total(&off_sizes), total(&on_sizes)],
        "KB",
    );

    println!("\npeak resident set: {:.0} MB", peak_rss());
}

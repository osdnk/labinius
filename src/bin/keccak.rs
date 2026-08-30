//! binius64's keccak circuit at `--message-len 65536` — 482 keccak-f permutations — proved three
//! ways: stock binius64, and this crate's commitment in place of its BaseFold oracle with the
//! recursion off and on.
//!
//! `cargo run --release --offline --bin keccak`, pinned with `taskset -c 3`.
use bin_ntt::keccak::stock::{LOG_INV_RATE, Stock};
use bin_ntt::keccak::{Circuit, MESSAGE_LEN, Session, Sizes};
use std::time::Instant;

/// The seed the public matrix `A` is expanded from.
const MATRIX_SEED: [u8; 32] = [0x5A; 32];
/// The core the process pins itself to.
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

/// One column per mode, `None` where that mode has no such stage.
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

fn main() {
    pin(CPU);
    let message: Vec<u8> =
        (0..MESSAGE_LEN).map(|i| (i as u32).wrapping_mul(2654435761) as u8).collect();
    let (circuit_ms, circuit) = once(|| Circuit::new(MESSAGE_LEN));
    let (witness_ms, witness) = once(|| circuit.witness(&message));
    let constraint_system = circuit.constraint_system();

    // Stock binius64, the same circuit and witness through its own prover and verifier.
    let (stock_setup, stock) = once(|| Stock::new(constraint_system.clone()));
    let (stock_prove, (stock_proof, stock_phases)) = once(|| stock.prove(&witness));
    let (stock_verify, (stock_ok, stock_verify_phases)) =
        once(|| stock.verify(witness.inout(), &stock_proof));
    stock_ok.expect("stock binius64 verifies its own proof");
    let stock_commit = stock_phases.milliseconds("Commit witness");
    let stock_bitand = stock_phases.milliseconds("[phase] BitAnd check");
    let stock_shift = stock_phases.milliseconds("[phase] Shift Reduction");
    let stock_pcs = stock_phases.milliseconds("[phase] PCS Opening");
    let stock_verify_pcs = stock_verify_phases.milliseconds("[phase] Verify PCS Opening");
    let [stock_iop, stock_native, stock_basefold] = stock
        .verify_stages(witness.inout(), &stock_proof)
        .expect("stock binius64 verifies its own proof");
    let stock_verify_reduce = stock_iop - stock_verify_pcs;

    // The same instance with our commitment, without and with the recursion.
    let (off_setup, mut off) = once(|| Session::new(constraint_system.clone(), false, MATRIX_SEED));
    let (on_setup, mut on) = once(|| Session::new(constraint_system.clone(), true, MATRIX_SEED));
    let (_, (off_proof, off_prover, off_sizes)) = once(|| off.prove(&witness, None));
    let (_, (on_proof, on_prover, on_sizes)) = once(|| on.prove(&witness, None));
    let off_verifier = off.verify(witness.inout(), &off_proof).expect("the honest proof verifies");
    let on_verifier = on.verify(witness.inout(), &on_proof).expect("the honest proof verifies");

    println!("bin-ntt over binius64 keccak, core {CPU}, one thread");
    println!(
        "keccak-256 of {MESSAGE_LEN} bytes: {} permutations, {} AND constraints, {} non-public words",
        (MESSAGE_LEN + 1).div_ceil(136),
        constraint_system.and_constraints.len(),
        witness.non_public().len()
    );
    println!(
        "stock binius64 at the example's defaults: --log-inv-rate {LOG_INV_RATE}, \
         --hash-suite sha256, rayon off; the other two commit 2^18 F162 in 256 columns"
    );
    println!("\n  {:<32}{:>13}{:>13}{:>13}", "", "stock", "recursion", "recursion");
    println!("  {:<32}{:>13}{:>13}{:>13}", "", "binius64", "off", "on");

    println!("\nSETUP (once, not per proof)");
    row("circuit", [Some(circuit_ms); 3], "ms");
    row(
        "commitment key and constraints",
        [Some(stock_setup), Some(off_setup), Some(on_setup)],
        "ms",
    );

    println!("\nPROVER");
    row("witness", [Some(witness_ms); 3], "ms");
    row("packing", [None, Some(off_prover.pack), Some(on_prover.pack)], "ms");
    row(
        "commit",
        [Some(stock_commit), Some(off_prover.commit), Some(on_prover.commit)],
        "ms",
    );
    row(
        "BitAnd check",
        [Some(stock_bitand), Some(off_prover.bitand), Some(on_prover.bitand)],
        "ms",
    );
    row(
        "shift reduction",
        [Some(stock_shift), Some(off_prover.shift), Some(on_prover.shift)],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [Some(stock_pcs), Some(off_prover.switch), Some(on_prover.switch)],
        "ms",
    );
    row("opening", [None, Some(off_prover.opening), Some(on_prover.opening)], "ms");
    let listed = |t: &bin_ntt::keccak::ProverTiming| {
        t.pack + t.commit + t.bitand + t.shift + t.switch + t.opening
    };
    row(
        "rest",
        [
            Some(stock_prove - stock_commit - stock_bitand - stock_shift - stock_pcs),
            Some(off_prover.total - listed(&off_prover)),
            Some(on_prover.total - listed(&on_prover)),
        ],
        "ms",
    );
    row(
        "total",
        [Some(stock_prove), Some(off_prover.total), Some(on_prover.total)],
        "ms",
    );

    println!("\nVERIFIER");
    row(
        "read the commitment",
        [None, Some(off_verifier.commitment), Some(on_verifier.commitment)],
        "ms",
    );
    row(
        "reductions",
        [Some(stock_verify_reduce), Some(off_verifier.reduce), Some(on_verifier.reduce)],
        "ms",
    );
    row(
        "ring-switch / cross-field switch",
        [Some(stock_verify_pcs), Some(off_verifier.switch), Some(on_verifier.switch)],
        "ms",
    );
    row(
        "BaseFold / our opening",
        [Some(stock_basefold), Some(off_verifier.opening), Some(on_verifier.opening)],
        "ms",
    );
    row(
        "wiring check (native)",
        [Some(stock_native), Some(off_verifier.wiring), Some(on_verifier.wiring)],
        "ms",
    );
    row(
        "total",
        [Some(stock_verify), Some(off_verifier.total), Some(on_verifier.total)],
        "ms",
    );

    println!("\nSIZES");
    let kb = |b: usize| Some(b as f64 / 1024.0);
    row("binius64 LIOP", [None, kb(off_sizes.liop), kb(on_sizes.liop)], "KB");
    row("cross-field switch", [None, kb(off_sizes.switch), kb(on_sizes.switch)], "KB");
    row(
        "commitment (wire form)",
        [None, kb(off_sizes.commitment_wire), kb(on_sizes.commitment_wire)],
        "KB",
    );
    row("opening", [None, kb(off_sizes.opening), kb(on_sizes.opening)], "KB");
    let total = |s: &Sizes| kb(s.liop + s.switch + s.commitment_wire + s.opening);
    row(
        "total",
        [kb(stock_proof.len()), total(&off_sizes), total(&on_sizes)],
        "KB",
    );

    println!(
        "  the tape carries T_Y as its polx image, {:.1} KB, rather than the wire form above",
        on_sizes.commitment as f64 / 1024.0
    );

    println!("\npeak resident set: {:.0} MB", peak_rss());
}

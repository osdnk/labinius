use labinius::scheme::suite_from_args;
use pcs_competitors::{
    basefold, brakedown, ligerito_flock, pin, pinned, random_u64s, whir, Row, CPU, TARGETS,
    WITNESS_SEED,
};

fn print(rows: &[Row]) {
    println!(
        "\n  {:<30}{:>8}{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "scheme", "rate", "target", "commit ms", "open ms", "verify ms", "C bytes", "|pi| bytes"
    );
    for row in rows {
        println!(
            "  {:<30}{:>8}{:>8}{:>12.2}{:>12.2}{:>12.2}{:>12}{:>12}",
            row.scheme,
            row.rate,
            row.target,
            row.commit_ms,
            row.open_ms,
            row.verify_ms,
            row.commitment,
            row.proof
        );
    }
    println!();
    for row in rows {
        println!(
            "  {} at {}, {} bits: {}, opening claim {}",
            row.scheme, row.rate, row.target, row.security, row.claim
        );
    }
}

fn main() {
    let suite = suite_from_args();
    let log_len = pcs_competitors::log_len(suite);
    println!("=== size {} ===", suite.name);
    pin(CPU);
    let u64s = random_u64s(log_len, WITNESS_SEED);

    let mut rows = Vec::new();
    for security_bits in TARGETS {
        for log_inv_rate in [1, 2] {
            rows.push(basefold::run(log_len, log_inv_rate, security_bits, &u64s));
        }
    }
    for security_bits in TARGETS {
        for log_inv_rate in [1, 2] {
            rows.push(whir::run(log_len, log_inv_rate, security_bits, &u64s));
        }
    }
    for profile in ligerito_flock::PROFILES {
        rows.push(ligerito_flock::run(log_len, profile, &u64s));
    }
    for spec in [0, 5] {
        rows.push(brakedown::run(log_len, spec, &u64s));
    }

    println!(
        "four hash-based PCSs on 2^{log_len} B128, opened at one point of {log_len} \
         coordinates, core {}, one thread",
        pinned()
    );
    print(&rows);
}

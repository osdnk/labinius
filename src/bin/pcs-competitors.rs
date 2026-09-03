use bin_ntt::competitors::{
    basefold, ligerito_bolt, ligerito_flock, pin, random_u64s, whir, Row, CPU, LOG_BITS, LOG_LEN,
    WITNESS_SEED,
};

fn print(rows: &[Row]) {
    println!(
        "\n  {:<22}{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}",
        "scheme", "rate", "commit ms", "open ms", "verify ms", "C bytes", "|pi| bytes"
    );
    for row in rows {
        println!(
            "  {:<22}{:>8}{:>12.2}{:>12.2}{:>12.2}{:>12}{:>12}",
            row.scheme,
            row.rate,
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
            "  {} at {}: {}, opening claim {}",
            row.scheme, row.rate, row.security, row.claim
        );
    }
}

fn main() {
    pin(CPU);
    let u64s = random_u64s(LOG_LEN, WITNESS_SEED);

    let mut rows = Vec::new();
    for log_inv_rate in [1, 2] {
        rows.push(basefold::run(LOG_LEN, log_inv_rate, &u64s));
    }
    for log_inv_rate in [1, 2] {
        rows.push(whir::run(LOG_LEN, log_inv_rate, &u64s));
    }
    for log_inv_rate in [1, 2] {
        rows.push(ligerito_bolt::run(LOG_LEN, log_inv_rate, &u64s));
    }
    for log_inv_rate in [1, 2] {
        rows.push(ligerito_flock::run(LOG_LEN, log_inv_rate, &u64s));
    }

    println!("four hash-based PCSs on 2^{LOG_LEN} B128 = 2^{LOG_BITS} bits, core {CPU}, one thread");
    print(&rows);
}

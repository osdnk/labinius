use pcs_competitors::{brakedown, pin, random_u64s, CPU, WITNESS_SEED};

fn main() {
    pin(CPU);
    let log_len: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let u64s = random_u64s(log_len, WITNESS_SEED);
    println!("{:<20}{:>8}{:>12}{:>12}{:>12}{:>14}  {}", "scheme", "rate", "commit", "open", "verify", "proof", "security");
    for spec in 0..6 {
        let r = brakedown::run(log_len, spec, &u64s);
        println!("{:<20}{:>8}{:>12.2}{:>12.2}{:>12.2}{:>14}  {}", format!("brakedown-{}", spec + 1), r.rate, r.commit_ms, r.open_ms, r.verify_ms, r.proof, r.security);
    }
}

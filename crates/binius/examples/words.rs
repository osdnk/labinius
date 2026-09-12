//! `words <keccak|sha|blake3> <len_bytes>...`: the private word count of each circuit.
use labinius_binius::{Circuit, Hash};

fn main() {
    let mut args = std::env::args().skip(1);
    let hash = match args.next().as_deref() {
        Some("keccak") => Hash::Keccak256,
        Some("sha") => Hash::Sha256,
        Some("blake3") => Hash::Blake3,
        _ => panic!("words <keccak|sha|blake3> <len_bytes>..."),
    };
    for len in args.map(|s| s.parse::<usize>().unwrap()) {
        let circuit = Circuit::new(hash, len);
        let cs = circuit.constraint_system();
        println!(
            "{} {len}: {} {}, n_private {}, n_inout {}, ands {}",
            hash.name(),
            hash.compressions(len),
            hash.unit(),
            cs.n_private,
            cs.n_inout,
            cs.and_constraints.len()
        );
    }
}

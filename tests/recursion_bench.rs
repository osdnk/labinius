//! Wall clock of the honest prover's recursion arithmetic at `Params::basic()`.
//! `cargo test --release --offline --test recursion_bench -- --nocapture`.
use bin_ntt::recursion::{limbs, Cap, Instance, BLOCKS, DEG, Q};
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::time::Instant;

const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const WITNESS_SEED: [u8; 32] = [0xC7; 32];

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn row(name: &str, milliseconds: f64) {
    println!("  {name:<34}{milliseconds:>9.2} ms");
}

#[test]
fn the_prover_side_at_the_basic_parameters() {
    let params = Params::new(18, 8, vec![Modulus::Q9721]).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/bench/recursion");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claim = witness.mle_evaluate(&point);
    let evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &evaluation);
    let folded = prover.fold(opening, &challenges);

    println!("\n{:?}, {} columns, limbs {:?}", params.witness_log_len, params.columns(), commitment.moduli());
    for limb in 0..commitment.moduli().len() {
        let q = commitment.moduli()[limb];
        let t = Instant::now();
        let rows = std::hint::black_box(limbs::key_rows(&pp, limb));
        row(&format!("key rows in coefficients [{q}]"), ms(t));
        let t = Instant::now();
        std::hint::black_box(limbs::residues(&commitment, limb));
        row(&format!("residues in coefficients [{q}]"), ms(t));
        assert_eq!(rows.rows.len(), folded.elements().len());
    }

    let t = Instant::now();
    let instance = Instance::new(&pp, &commitment, &folded, &evaluation, &challenges, &point, &claim);
    row("whole instance", ms(t));

    let t = Instant::now();
    let residuals = instance.residuals();
    row("block sums and residuals", ms(t));
    assert!(residuals.iter().flatten().flatten().all(|&x| x == 0));

    let t = Instant::now();
    let bound = instance.bound();
    row("no-wrap bound", ms(t));

    let t = Instant::now();
    let w = instance.witness();
    row("witness export", ms(t));

    println!("\n  limbs {:?}", instance.limbs);
    println!("\n  witness vectors");
    for (v, s) in instance.vectors.iter().zip(&w.vectors) {
        let max = s.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
        println!(
            "  {:<16}{:>6} polys  max {:>7}  betasq 2^{:>5.1}  cap 2^{:>5.1}",
            v.name,
            v.polys.len(),
            max,
            (v.betasq() as f64).log2(),
            2.0 * v.cap().log2()
        );
    }

    println!("\n  carries");
    for c in &instance.chains {
        let e = c.carry_values(&instance.vectors);
        let max = e.iter().flatten().map(|x| x.abs()).max().unwrap();
        println!("  {:<28}max |e| 2^{:.1}  reach 2^{:.1}", c.name, (max as f64).log2(), (c.carries.gadget.reach() as f64).log2());
    }

    println!("\n  no-wrap bound, Q/2 = 2^{:.2}", ((Q as f64) / 2.0).log2());
    for b in &bound {
        println!(
            "  {:<28}2^{:>6.2}  margin {:>5.2}x  worst {} ({:.0}%)",
            b.name,
            b.value.log2(),
            b.margin(),
            b.worst,
            100.0 * b.share
        );
        assert!(b.value < (Q as f64) / 2.0);
    }

    let polys: usize = instance.chains.iter().map(|c| c.products.len() + c.scaled.len() + c.carries.at.len() * BLOCKS).sum::<usize>() * BLOCKS;
    let mut tight = Instance::new(&pp, &commitment, &folded, &evaluation, &challenges, &point, &claim);
    for v in tight.vectors.iter_mut() {
        if let Cap::PerCoefficient(_) = v.cap {
            if v.name.starts_with('e') || v.name.starts_with('k') || v.name.starts_with('w') {
                v.cap = Cap::Betasq(2.0 * v.betasq() as f64);
            }
        }
    }
    println!("\n  the same with every digit level capped at twice its honest betasq");
    for b in tight.bound() {
        println!("  {:<28}2^{:>6.2}  margin {:>5.2}x  worst {}", b.name, b.value.log2(), b.margin(), b.worst);
    }

    println!(
        "\n  {} constraints, {} phi elements, {} MB of i64 phi",
        instance.chains.len() * BLOCKS,
        polys,
        polys * DEG * 8 / (1 << 20)
    );
    println!(
        "  {} witness polys, {} KB of i16 witness",
        w.vectors.iter().map(|v| v.len() / DEG).sum::<usize>(),
        w.vectors.iter().map(|v| v.len()).sum::<usize>() * 2 / 1024
    );
}

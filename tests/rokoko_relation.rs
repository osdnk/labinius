//! The rokoko relation: an honest round encodes and holds over `Z`, the verifier's layout is the
//! prover's and depends on the shape alone, the no-wraparound bound clears `q / 2`, and one
//! tampered coefficient breaks a block equation.
#![cfg(feature = "rokoko")]
use bin_ntt::rokoko::relation::{self, Setup};
use bin_ntt::rokoko::{Layout, Relation, Witness as Committed, BLOCKS, DEG, SUPPORT};
use bin_ntt::{
    Commitment, EvaluationPoint, FoldedWitness, FoldingChallenges, Modulus, Params, Prover,
    PublicParameters, RowEvaluation, Transcript, Verifier, Witness, F162,
};

const MATRIX_SEED: [u8; 32] = [17u8; 32];
const WITNESS_SEED: [u8; 32] = [29u8; 32];

struct Round {
    setup: Setup,
    commitment: Commitment,
    folded: FoldedWitness,
    row: RowEvaluation,
    challenges: FoldingChallenges,
    point: EvaluationPoint,
    claim: F162,
}

fn round(params: &Params, domain: &[u8]) -> Round {
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let setup = Setup::new(&pp);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(domain);
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claim = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
    let folded = prover.fold(opening, &challenges);
    Round {
        setup,
        commitment,
        folded,
        row,
        challenges,
        point,
        claim,
    }
}

impl Round {
    fn encode(&self) -> (Relation, Committed) {
        relation::encode(
            &self.setup,
            self.commitment.matrix(),
            &self.folded,
            &self.row,
            &self.challenges,
            &self.point,
            &self.claim,
        )
        .expect("the honest round is within its gadgets")
    }

    fn layout(&self) -> Relation {
        relation::layout(&self.setup, &self.challenges, &self.point, &self.claim)
    }
}

fn small() -> Params {
    Params::new(13, 3, vec![Modulus::Q9721_FS_S], false).unwrap()
}

fn same_layout(a: &Layout, b: &Layout) {
    assert_eq!(a.len, b.len);
    assert_eq!(a.regions, b.regions);
    assert_eq!(a.vectors.len(), b.vectors.len());
    for (x, y) in a.vectors.iter().zip(&b.vectors) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.cap, y.cap);
        assert_eq!(x.binary, y.binary);
        assert_eq!(x.used, y.used);
    }
}

fn same(a: &Relation, b: &Relation) {
    same_layout(&a.layout, &b.layout);
    assert_eq!(a.equations.len(), b.equations.len());
    for (x, y) in a.equations.iter().zip(&b.equations) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.diagonals.len(), y.diagonals.len());
        for (p, q) in x.diagonals.iter().zip(&y.diagonals) {
            assert_eq!(p.output, q.output, "{}", x.name);
            assert_eq!(p.entries, q.entries, "{}", x.name);
        }
    }
}

#[test]
fn the_honest_round_holds_over_z() {
    let r = round(&small(), b"bin-ntt/test/rokoko");
    let (relation, witness) = r.encode();
    relation::check(&relation, &witness).unwrap();
    let limbs = r.setup.primes.len();
    assert_eq!(relation.equations.len(), 4 * limbs + 2);
    assert_eq!(relation.layout.vectors.len(), 3 + 10 * limbs + 4);
    for e in &relation.equations {
        assert_eq!(e.diagonals.len(), BLOCKS);
    }
}

#[test]
fn the_layout_is_public() {
    let r = round(&small(), b"bin-ntt/test/rokoko");
    let (relation, _) = r.encode();
    same(&r.layout(), &relation);
}

#[test]
fn the_layout_depends_on_the_shape_alone() {
    let a = round(&small(), b"bin-ntt/test/rokoko/a");
    let b = round(&small(), b"bin-ntt/test/rokoko/b");
    assert_ne!(a.claim, b.claim);
    same_layout(&a.layout().layout, &b.layout().layout);
}

#[test]
fn regions_are_placed_as_the_witness_builder_places_them() {
    let layout = round(&small(), b"bin-ntt/test/rokoko").layout().layout;
    assert!(layout.len.is_power_of_two());
    let mut cursor = 0usize;
    for (v, region) in layout.vectors.iter().zip(&layout.regions) {
        assert!(region.len.is_power_of_two());
        assert_eq!(region.len, v.used.next_power_of_two());
        assert_eq!(
            region.start,
            cursor.next_multiple_of(region.len),
            "{}",
            v.name
        );
        cursor = region.start + region.len;
    }
    assert!(cursor <= layout.len);
}

#[test]
fn the_outer_vectors_are_the_encoded_ones() {
    let r = round(&small(), b"bin-ntt/test/rokoko");
    let (relation, witness) = r.encode();
    let residues = relation::residue_vectors(&r.setup, r.commitment.matrix());
    let at = r.setup.residue_vectors_at();
    assert_eq!(residues.len(), at.len());
    assert_eq!(at.len(), 8 * r.setup.primes.len());
    for (v, x) in at.iter().zip(&residues) {
        assert!(relation.layout.vectors[*v].name.starts_with("C["));
        assert_eq!(&witness[*v], x);
    }
    let u = r.setup.lift_vector_at();
    assert_eq!(relation.layout.vectors[u].name, "u");
    assert!(relation.layout.vectors[u].binary);
    assert_eq!(witness[u], relation::lift_vector(&r.setup, &r.row));
}

#[test]
fn every_element_keeps_to_the_support() {
    let (relation, mut witness) = round(&small(), b"bin-ntt/test/rokoko").encode();
    for (v, w) in witness.iter().enumerate() {
        assert_eq!(w.len(), relation.layout.vectors[v].used);
        for x in w {
            assert!(x[SUPPORT..].iter().all(|&c| c == 0));
        }
    }
    witness[0][0][SUPPORT] = 1;
    let err = relation::check(&relation, &witness).unwrap_err();
    assert!(err.contains("support"), "{err}");
}

#[test]
fn one_tampered_coefficient_breaks_an_equation() {
    let r = round(&small(), b"bin-ntt/test/rokoko");
    let (relation, honest) = r.encode();
    for v in 0..relation.layout.vectors.len() {
        let used = relation.layout.vectors[v].used;
        for (e, c) in [(0, 0), (used / 2, SUPPORT / 2), (used - 1, SUPPORT - 1)] {
            let mut witness = honest.clone();
            witness[v][e][c] += 1;
            assert!(
                relation::check(&relation, &witness).is_err(),
                "{} element {e} coefficient {c}",
                relation.layout.vectors[v].name
            );
        }
    }
    let mut witness = honest.clone();
    witness[0][0][DEG - 1] = 1;
    assert!(relation::check(&relation, &witness).is_err());
}

#[test]
fn the_small_shape_clears_the_no_wrap_bound() {
    let r = round(&small(), b"bin-ntt/test/rokoko");
    let bound = relation::no_wrap_bound(&r.layout());
    assert!(bound < (1u64 << 49) as f64, "2^{:.2}", bound.log2());
}

/// The basic recursive shape: the layout and the bound need no witness.
#[test]
fn the_basic_shape_clears_the_no_wrap_bound() {
    let params = Params::new(18, 8, vec![Modulus::Q9721_FS_S], true).unwrap();
    let r = round(&params, b"bin-ntt/test/rokoko");
    let relation = r.layout();
    let bound = relation::no_wrap_bound(&relation);
    eprintln!(
        "basic shape: N = {}, bound 2^{:.2}",
        relation.layout.len,
        bound.log2()
    );
    assert!(bound < (1u64 << 49) as f64, "2^{:.2}", bound.log2());
}

/// The basic shape in the clear, for `GADGET_STATS` calibration of the gadgets.
#[test]
fn the_basic_shape_encodes() {
    let params = Params::new(18, 8, vec![Modulus::Q9721_FS_S], false).unwrap();
    let r = round(&params, b"bin-ntt/test/rokoko");
    let (relation, witness) = r.encode();
    relation::check(&relation, &witness).unwrap();
}

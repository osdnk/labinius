#![cfg(feature = "rokoko")]
use bin_ntt::rng::Rng;
use bin_ntt::rokoko::claims::{
    coefficients, commit_lift, commit_residues, committed, prove, ring, verify, Crs, Keys, Proof,
};
use bin_ntt::rokoko::{
    BlockEquations, Cap, Diagonal, Element, Layout, Poly, Region, Relation, Vector, Witness,
    BLOCKS, DEG, SUB, SUPPORT,
};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::protocol::config::{Config, SimpleConfig, SumcheckConfig};
use rokoko::protocol::config_generator::{
    AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
};
use rokoko::protocol::snark::{eq, witness_in};
use std::sync::LazyLock;

static CHAIN: LazyLock<Config> = LazyLock::new(|| {
    let level = |base_log, chunks, next: Option<AuxRecursionConfig>| AuxRecursionConfig {
        decomposition_base_log: base_log,
        decomposition_chunks: chunks,
        rank: 1,
        next: next.map(Box::new),
    };
    AuxSumcheckConfig {
        exact_projection_norm: false,
        witness_height: 256,
        witness_width: 4,
        projection_ratio: 32,
        projection_height: 256,
        basic_commitment_rank: 3,
        nof_openings: 2,
        commitment_recursion: level(15, 4, Some(level(7, 8, None))),
        opening_recursion: level(15, 4, None),
        projection_recursion: AuxProjection::Fine {
            nof_batches: 2,
            recursion_constant_term: level(15, 2, None),
            recursion_batched_projection: level(15, 4, None),
        },
        witness_decomposition_chunks: 2,
        witness_decomposition_base_log: 15,
        next: Some(Box::new(AuxConfig::Simple(SimpleConfig {
            witness_height: 256,
            witness_width: 4,
            projection_ratio: 128,
            projection_height: 256,
            projection_nof_batches: 2,
            basic_commitment_rank: 2,
            witness_norm_bound: f64::INFINITY,
            projection_norm_bound: f64::INFINITY,
        }))),
    }
    .generate_config()
});

static CRS: LazyLock<Crs> = LazyLock::new(|| Crs::new(chain()));

fn chain() -> &'static SumcheckConfig {
    match &*CHAIN {
        Config::Sumcheck(c) => c,
        _ => unreachable!(),
    }
}

fn signed(rng: &mut Rng, bound: i64) -> i64 {
    rng.below((2 * bound + 1) as u32) as i64 - bound
}

fn element(rng: &mut Rng, bound: i64) -> Element {
    let mut e = [0i64; DEG];
    for x in &mut e[..SUPPORT] {
        *x = signed(rng, bound);
    }
    e
}

fn block(rng: &mut Rng) -> Poly {
    (0..SUB).map(|_| signed(rng, 3)).collect()
}

fn product(w: &Poly, e: &Element) -> Vec<i64> {
    let mut p = vec![0i64; DEG];
    for (i, &a) in w.iter().enumerate() {
        for (j, &b) in e.iter().enumerate() {
            if a != 0 && b != 0 {
                assert!(i + j < DEG, "the product wraps");
                p[i + j] += a * b;
            }
        }
    }
    p
}

const V: usize = 0;
const C: usize = 1;
const U: usize = 2;
const LOOSE: usize = 5;

struct Fixture {
    relation: Relation,
    witness: Witness,
    keys: Keys,
    t_y: Vec<RingElement>,
    t_u: Vec<RingElement>,
    digest: [u8; 32],
}

fn layout() -> Layout {
    let vector = |name: &str, cap, binary, used| Vector {
        name: name.into(),
        cap,
        binary,
        used,
    };
    Layout {
        vectors: vec![
            vector("v", Cap::PerCoefficient(64.0), false, 6),
            vector("c", Cap::PerCoefficient(77.0), false, 3),
            vector("u", Cap::PerCoefficient(1.0), true, 4),
        ],
        regions: vec![
            Region { start: 0, len: 8 },
            Region { start: 8, len: 4 },
            Region { start: 12, len: 4 },
        ],
        len: 16,
    }
}

fn witness(rng: &mut Rng) -> Witness {
    let bits = |rng: &mut Rng| {
        let mut e = element(rng, 0);
        for x in &mut e[..SUPPORT] {
            *x = rng.below(2) as i64;
        }
        e
    };
    vec![
        (0..6).map(|_| element(rng, 64)).collect(),
        (0..3).map(|_| element(rng, 77)).collect(),
        (0..4).map(|_| bits(rng)).collect(),
    ]
}

fn equations(rng: &mut Rng, layout: &Layout, witness: &Witness) -> Vec<BlockEquations> {
    let readable: Vec<usize> = (0..3)
        .flat_map(|v| (0..layout.vectors[v].used).map(move |e| (v, e)))
        .filter(|&(v, e)| !(v == V && e == LOOSE))
        .map(|(v, e)| layout.index(v, e))
        .collect();
    let owner = |i: usize| {
        (0..3)
            .find(|&v| {
                let r = layout.regions[v];
                (r.start..r.start + layout.vectors[v].used).contains(&i)
            })
            .unwrap()
    };
    (0..2)
        .map(|c| BlockEquations {
            name: format!("chain {c}"),
            diagonals: (0..BLOCKS)
                .map(|_| {
                    let entries: Vec<(usize, Poly)> = (0..3)
                        .map(|_| {
                            (
                                readable[rng.below(readable.len() as u32) as usize],
                                block(rng),
                            )
                        })
                        .collect();
                    let mut output = vec![0i64; DEG];
                    for (i, w) in &entries {
                        let v = owner(*i);
                        let e = &witness[v][*i - layout.regions[v].start];
                        for (o, p) in output.iter_mut().zip(product(w, e)) {
                            *o += p;
                        }
                    }
                    Diagonal { entries, output }
                })
                .collect(),
        })
        .collect()
}

fn fixture(seed: u64, tamper: impl FnOnce(&mut Witness)) -> Fixture {
    let mut rng = Rng::new(seed);
    let layout = layout();
    let mut witness = witness(&mut rng);
    tamper(&mut witness);
    let equations = equations(&mut rng, &layout, &witness);
    let keys = Keys::new(&layout, &[C], U, [7u8; 32], 2, 1);
    let t_y = commit_residues(&keys, &witness[C..=C]);
    let t_u = commit_lift(&keys, &witness[U]);
    Fixture {
        relation: Relation { layout, equations },
        witness,
        keys,
        t_y,
        t_u,
        digest: [3u8; 32],
    }
}

impl Fixture {
    fn prove(&self) -> Proof {
        prove(
            &self.relation,
            &self.witness,
            &self.keys,
            &self.t_y,
            &self.t_u,
            &CRS,
            chain(),
            self.digest,
        )
        .expect("an honest proof")
    }

    fn verify(&self, proof: &Proof) -> Result<(), String> {
        verify(
            &self.relation,
            &self.keys,
            &self.t_y,
            &self.t_u,
            &CRS,
            chain(),
            self.digest,
            proof,
        )
    }
}

#[test]
fn ring_round_trip() {
    let mut rng = Rng::new(1);
    let e = element(&mut rng, 1 << 20);
    assert_eq!(coefficients(&ring(&e)), e);
    let mut r = ring(&e);
    assert_eq!(r.representation, Representation::IncompleteNTT);
    r.to_representation(Representation::Coefficients);
    assert_eq!(coefficients(&r), e);
}

#[test]
fn keys_agree_with_their_claims() {
    let f = fixture(2, |_| {});
    let committed = committed(&f.relation.layout, &f.witness, chain()).unwrap();
    let n = committed.data.len();
    let over = |r: &rokoko::protocol::snark::Region| {
        rokoko::protocol::snark::Region::new(r.start(), r.len(), n)
    };
    for (row, t) in f.keys.rows_y.iter().zip(&f.t_y) {
        let mut acc = RingElement::zero(Representation::IncompleteNTT);
        for (layers, r) in row.iter().zip(&f.keys.residues) {
            let r = over(r);
            acc += &(eq(layers.clone()).on(r.vars()) * witness_in(r)).sum(&committed);
        }
        assert_eq!(&acc, t);
    }
    let lift = over(&f.keys.lift);
    for (layers, t) in f.keys.rows_u.iter().zip(&f.t_u) {
        assert_eq!(
            &(eq(layers.clone()).on(lift.vars()) * witness_in(lift)).sum(&committed),
            t
        );
    }
}

#[test]
fn honest_proof_verifies() {
    let f = fixture(3, |_| {});
    let proof = f.prove();
    println!("rokoko opening proof: {} bytes", proof.wire_bytes());
    f.verify(&proof).unwrap();
}

#[test]
fn norm_over_the_cap_is_rejected() {
    let f = fixture(4, |_| {});
    let mut proof = f.prove();
    proof.norms[V] = RingElement::constant(
        64 * 64 * 6 * SUPPORT as u64 + 1,
        Representation::IncompleteNTT,
    );
    let err = f.verify(&proof).unwrap_err();
    assert!(err.contains("norm"), "{err}");
}

#[test]
fn retagged_value_is_rejected() {
    let f = fixture(9, |_| {});
    let mut proof = f.prove();
    proof.norms[V].representation = Representation::Coefficients;
    let err = f.verify(&proof).unwrap_err();
    assert!(err.contains("representation"), "{err}");
}

#[test]
fn wrong_claim_value_is_rejected() {
    let f = fixture(5, |_| {});
    let mut proof = f.prove();
    proof.norms[V] += &RingElement::one(Representation::IncompleteNTT);
    assert!(f.verify(&proof).is_err());
}

#[test]
fn wrong_transcript_is_rejected() {
    let f = fixture(6, |_| {});
    let proof = f.prove();
    let mut other = fixture(6, |_| {});
    other.digest[0] ^= 1;
    assert!(other.verify(&proof).is_err());
}

#[test]
fn support_violation_is_rejected() {
    let f = fixture(7, |w| w[V][LOOSE][100] = 1);
    let proof = f.prove();
    let err = f.verify(&proof).unwrap_err();
    assert!(err.contains("support"), "{err}");
}

#[test]
fn non_bit_lift_is_rejected() {
    let f = fixture(8, |w| w[U][1][3] = 2);
    let proof = f.prove();
    let err = f.verify(&proof).unwrap_err();
    assert!(err.contains("binary"), "{err}");
}

//! The recursion encoding: the ring conventions, the chain on synthetic identities, and the whole
//! instance on the real pipeline.
use bin_ntt::api::N162;
use bin_ntt::params::N;
use bin_ntt::recursion::{
    binary, chunk, limbs, Gadget, Instance, SElem, BLOCKS, BLOCK_LIMIT, CHUNK, CHUNKS, DEG, Q, V,
};
use bin_ntt::rng::Rng;
use bin_ntt::{scalar, Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [17u8; 32];
const WITNESS_SEED: [u8; 32] = [29u8; 32];

/// One round of the real pipeline, encoded.
fn round(params: &Params) -> Instance {
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let setup = pp.recursion().expect("recursion is on").clone();
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let residues = opening.residues().expect("recursion is on").clone();
    let mut transcript = Transcript::new(b"bin-ntt/test/recursion");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claim = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let left = prover.commit_left_expansion(&row);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &left);
    let folded = prover.fold(opening, &challenges);
    Instance::new(&setup, &residues, &folded, &row, &challenges, &point, &claim)
}

fn small() -> Params {
    Params::new(9, 2, vec![Modulus::Q9721], true).unwrap()
}

// =============================================================================================
// the ring conventions
// =============================================================================================

/// `S`-arithmetic in the `Z`-basis reproduces the schoolbook product of `R_648`, twist included.
#[test]
fn the_twist_matches_the_scalar_reference() {
    let mut rng = Rng::new(7);
    let q = 3889i64;
    for _ in 0..8 {
        let a: [u32; N] = core::array::from_fn(|_| rng.below(q as u32));
        let b: [u32; N] = core::array::from_fn(|_| rng.below(q as u32));
        let c = scalar::mul_mod_phi(&a, &b, q as u16);
        let ac = limbs::split(&core::array::from_fn(|i| a[i] as i64));
        let bc = limbs::split(&core::array::from_fn(|i| b[i] as i64));
        for m in 0..4 {
            let mut acc = [0i64; N162];
            for l in 0..4 {
                let mut p = chunk::mul(&ac[(m + 4 - l) % 4], &bc[l]);
                if l > m {
                    p = chunk::shift(&p, 1);
                    p.iter_mut().for_each(|x| *x = -*x);
                }
                for (i, x) in p.iter().enumerate() {
                    acc[i] += x;
                }
            }
            let want = limbs::split(&core::array::from_fn(|i| c[i] as i64))[m];
            for t in 0..N162 {
                assert_eq!((acc[t] - want[t]).rem_euclid(q), 0, "component {m} coefficient {t}");
            }
        }
    }
}

/// The chunk encoding is exact, and a coefficient outside the support changes what it decodes to.
#[test]
fn chunks_round_trip() {
    let mut rng = Rng::new(11);
    for _ in 0..8 {
        let x: SElem = core::array::from_fn(|_| rng.below(4001) as i64 - 2000);
        let c = chunk::chunks(&x);
        assert!(chunk::supported(&c));
        assert_eq!(chunk::decode(&c), x);
        let mut bad = c;
        bad[1][CHUNK] = 1;
        assert!(!chunk::supported(&bad));
        assert_ne!(chunk::decode(&bad), x);
    }
}

/// The public blocks of `g` are the sub-chunks of `Z^{CHUNK b} g mod Phi_243`.
#[test]
fn blocks_are_the_shifted_reductions() {
    let mut rng = Rng::new(31);
    let g: SElem = core::array::from_fn(|_| rng.below(3889) as i64 - 1944);
    let b = chunk::blocks(&g);
    for k in 0..CHUNKS {
        let want = chunk::shift(&g, CHUNK * k);
        for a in 0..BLOCKS {
            for u in 0..9 {
                assert_eq!(b[k][a][u] as i64, want[9 * a + u]);
            }
        }
    }
}

/// The inverse transforms invert the crate's forward ones on every limb, splitting or quadratic.
#[test]
fn inverse_transforms_round_trip() {
    let mut rng = Rng::new(13);
    for (q, quad) in [(3889u16, false), (9721, false), (2917, true), (4861, true), (12637, true)] {
        for _ in 0..3 {
            let a: [u32; N] = core::array::from_fn(|_| rng.below(q as u32));
            let slots = limbs::transform(q, quad, &core::array::from_fn(|i| a[i] as i64));
            let back = limbs::coefficients(q, quad, &slots);
            for i in 0..N {
                assert_eq!(back[i].rem_euclid(q as i64) as u32, a[i], "q = {q}");
            }
        }
    }
}

/// The residues a recursive commitment opens re-transform to the columns of the matrix: the
/// batched recombination and vectorised inverse transform of `limbs::residues` against the
/// crate's own forward transform, over both a partial and a full batch of columns.
#[test]
fn residues_re_transform_to_the_commitment() {
    for params in [Params::new(9, 2, vec![Modulus::Q9721], false).unwrap(), Params::new(15, 6, vec![Modulus::Q2917, Modulus::Q9721], false).unwrap()] {
        let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
        let mut prover = Prover::new(&pp);
        let witness = Witness::random(&params, WITNESS_SEED);
        let (commitment, _) = prover.commit(&witness);
        let matrix = commitment.matrix();
        let primes = params.primes();
        let residues = limbs::residues(matrix, &primes);
        let r = params.columns();
        for (limb, &q) in primes.iter().enumerate() {
            let quad = limbs::Shape::of(q).quad;
            for j in 0..r {
                let mut a = [0i64; N];
                for m in 0..4 {
                    let c: [bin_ntt::recursion::Poly; CHUNKS] = core::array::from_fn(|b| {
                        residues.vectors[limb * 4 + m][b * r + j]
                    });
                    let e = chunk::decode(&c);
                    for t in 0..N162 {
                        a[4 * t + m] = if t % 2 == 0 { e[t] } else { -e[t] };
                    }
                }
                let back = bin_ntt::api::components_of(q, quad, &limbs::transform(q, quad, &a));
                for m in 0..4 {
                    assert_eq!(back[m], matrix.get(m, j).limbs[limb], "q = {q} column {j} component {m}");
                }
            }
        }
    }
}

/// The key rows really are the matrix the kernel commits with.
#[test]
fn key_rows_re_transform_to_the_key() {
    let params = small();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, _) = prover.commit(&witness);
    for limb in 0..commitment.moduli().len() {
        let rows = limbs::key_rows(&pp, limb);
        for i in 0..rows.rows.len() {
            let back = limbs::transform(rows.q, rows.quad, &rows.coefficients(i));
            assert_eq!(back, limbs::key_slots(&pp, limb, i), "limb {limb} row {i}");
        }
    }
}

// =============================================================================================
// the chain, on synthetic identities
// =============================================================================================

/// Random public multipliers against small witness elements: every block equation evaluates to
/// zero, and a chunk with a coefficient at position `CHUNK` or above breaks one.
#[test]
fn the_chain_encodes_a_random_identity() {
    let mut rng = Rng::new(19);
    let gadget = Gadget { base: 1024, levels: 3 };
    for terms in [1usize, 5, 40] {
        let g: Vec<SElem> = (0..terms)
            .map(|_| core::array::from_fn(|_| rng.below(3889) as i64 - 1944))
            .collect();
        let x: Vec<SElem> = (0..terms)
            .map(|_| core::array::from_fn(|_| rng.below(601) as i64 - 300))
            .collect();
        let instance = Instance::of_identity(&g, &x, gadget);
        assert!(instance.holds(), "{terms} terms: {:?}", instance.failure());
        let mut z = [0i64; N162];
        for (t, e) in x.iter().enumerate() {
            let c: [_; CHUNKS] = core::array::from_fn(|b| instance.vectors[0].polys[t * CHUNKS + b]);
            assert_eq!(chunk::decode(&c), *e);
            for (i, y) in chunk::mul(&g[t], &chunk::decode(&c)).iter().enumerate() {
                z[i] += y;
            }
        }
        assert_eq!(z, instance.chains[0].output);
        let mut broken = Instance::of_identity(&g, &x, gadget);
        broken.vectors[0].polys[0][CHUNK] = 3;
        assert!(!broken.holds(), "{terms} terms: an unsupported chunk went unnoticed");
    }
}

// =============================================================================================
// the whole instance on the pipeline
// =============================================================================================

#[test]
fn the_instance_holds_over_z() {
    let i = round(&small());
    assert!(i.holds(), "{:?}", i.failure());
    assert_eq!(i.chains.len(), 4 * 2 + 2);
    assert_eq!(i.vectors[V].support, CHUNK);
}

/// Every limb list the API offers, quadratic-slot primes included.
#[test]
fn the_instance_holds_on_every_limb() {
    for extra in [
        vec![],
        vec![Modulus::Q2917],
        vec![Modulus::Q2917, Modulus::Q4861, Modulus::Q12637],
        vec![Modulus::Q17497],
        vec![Modulus::Q19441],
        vec![Modulus::Q9721, Modulus::Q19441],
    ] {
        let n = extra.len() + 1;
        let i = round(&Params::new(9, 2, extra.clone(), true).unwrap());
        assert!(i.holds(), "{extra:?}: {:?}", i.failure());
        assert!(i.clears(), "{extra:?}: the no-wrap bound");
        assert_eq!(i.chains.len(), 4 * n + 2);
        assert_eq!(i.limbs.len(), n);
    }
}

#[test]
fn the_instance_clears_the_no_wrap_bound() {
    let i = round(&small());
    for b in i.bound() {
        assert!(b.value < (Q as f64) / 2.0, "{}: 2^{:.2}", b.name, b.value.log2());
    }
    assert!(i.clears());
}

/// One tampered coefficient anywhere in the witness breaks some block equation.
#[test]
fn one_tampered_coefficient_breaks_an_equation() {
    let base = round(&small());
    for v in 0..base.vectors.len() {
        let support = base.vectors[v].support;
        let polys = base.vectors[v].used;
        for (p, c) in [(0, 0), (polys / 2, support / 2), (polys - 1, support - 1)] {
            let mut i = round(&small());
            i.vectors[v].polys[p][c] += 1;
            assert!(!i.holds(), "{} poly {p} coefficient {c}", base.vectors[v].name);
        }
        let mut i = round(&small());
        i.vectors[v].polys[0][support] = 1;
        assert!(!i.holds(), "{} outside its support", base.vectors[v].name);
    }
}

/// The exported statement and witness are consistent with the instance.
#[test]
fn the_export_matches_the_instance() {
    let i = round(&small());
    let w = i.witness();
    let s = i.statement();
    assert_eq!(w.vectors.len(), i.vectors.len());
    assert_eq!(s.vectors.len(), i.vectors.len());
    for (v, (x, spec)) in i.vectors.iter().zip(w.vectors.iter().zip(&s.vectors)) {
        assert_eq!(x.len(), v.polys.len() * DEG);
        assert_eq!(spec.n, v.polys.len());
        assert!(spec.betasq <= spec.cap_betasq, "{} exceeds its cap", spec.name);
    }
    assert_eq!(s.constraints.len(), i.chains.len() * BLOCKS);
    for c in &s.constraints {
        assert_eq!(c.deg, 1);
        assert_eq!(c.phi.len(), c.blocks.iter().map(|b| b.len).sum::<usize>());
        for b in &c.blocks {
            assert!(b.off + b.len <= s.vectors[b.idx].n);
        }
    }
    assert!(s.constraints.iter().any(|c| c.blocks.iter().any(|b| b.key_time)));
    assert_eq!(s.residues.len(), 8);
    assert!(!s.rest.is_empty());
}

/// Every exported constraint, evaluated as LaBRADOR would, is satisfied.
#[test]
fn the_exported_constraints_are_satisfied() {
    let i = round(&small());
    let w = i.witness();
    let s = i.statement();
    for c in &s.constraints {
        let mut acc = [0i128; DEG];
        let mut t = 0;
        for b in &c.blocks {
            for p in 0..b.len {
                let phi = &c.phi[t];
                let s = &w.vectors[b.idx][(b.off + p) * DEG..(b.off + p + 1) * DEG];
                for x in 0..DEG {
                    for y in 0..DEG {
                        let (z, sign) = if x + y < DEG { (x + y, 1i128) } else { (x + y - DEG, -1) };
                        acc[z] += sign * phi[x] as i128 * s[y] as i128;
                    }
                }
                t += 1;
            }
        }
        let b = c.b.clone().unwrap_or_else(|| vec![[0i64; DEG]]);
        for x in 0..DEG {
            assert_eq!(acc[x], b[0][x] as i128, "{} coefficient {x}", c.name);
        }
    }
}

/// A lift and its reduction agree with `F162`.
#[test]
fn lifts_reduce_to_f162() {
    use bin_fields::scalar::F162;
    let mut rng = Rng::new(23);
    for _ in 0..16 {
        let x = F162([rng.next_u64(), rng.next_u64(), rng.next_u64() & ((1 << 34) - 1)]);
        assert_eq!(binary::reduce_mod_2(&binary::lift(&x)), x);
    }
}

/// The public blocks of every prime's key rows stay inside `BLOCK_LIMIT`, which is what makes the
/// `i16` dot product of `chain` exact: a block is `Z^{CHUNK b} g mod Phi_243` of a centered key
/// row, and the reduction's two foldings leave it at twice `|g|` at worst.
#[test]
fn public_blocks_stay_inside_their_limit() {
    for m in Modulus::ALL {
        let params = Params::new(9, 2, vec![m], false).unwrap();
        let pp = PublicParameters::from_seed(params, [3u8; 32]);
        for limb in 0..2 {
            let q = pp.key().prime(limb) as i64;
            let mut worst = 0i64;
            for r in &limbs::key_rows(&pp, limb).rows {
                for g in r {
                    for b in chunk::blocks(g) {
                        for a in b {
                            for x in a {
                                worst = worst.max((x as i64).abs());
                            }
                        }
                    }
                }
            }
            assert!(worst <= BLOCK_LIMIT, "q = {q}: block coefficient {worst}");
            assert!(worst <= q - 1, "q = {q}: block coefficient {worst} above 2 |g|");
        }
    }
}

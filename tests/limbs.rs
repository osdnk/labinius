//! The commitment key over a list of limbs: every limb's commitment against the scalar reference
//! in that limb's own domain (`scalar::ntt` and a slot product for a splitting limb,
//! `scalar::mul_quad_slots(scalar::ntt_quad(..), A)` for a quadratic one), the four-way `R_162`
//! decomposition per limb, the whole prover/verifier pipeline over four limb lists, corrupted
//! witnesses rejected per limb, and — the one thing that must not change — the default
//! configuration's commitment bit for bit against two single-limb kernel runs.
use bin_fields::scalar::F162;
use bin_ntt::api::{decompose_648_to_4x162, AdditionalLimb, N162};
use bin_ntt::challenge::{sample_short_challenge, ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
use bin_ntt::eval::{
    check_claim, evaluate_mle, fold_binary, left_expand, sample_point, verify_binary, verify_fold,
    EvalPoint, RawCommitments, Verifier,
};
use bin_ntt::f162::{self, RandomF162};
use bin_ntt::params::N;
use bin_ntt::rng::Rng;
use bin_ntt::simd::commit as cm;
use bin_ntt::types::Batch32;
use bin_ntt::{fold, scalar, CommitmentKey, PowerOfThreeRingElementWithLimbs};

/// The production commitment over one limb, straight through the kernel driver.
fn commit_one(q: u16, quad: bool, elems: &[F162], a: &[Batch32]) -> [u32; N] {
    cm::commit_limbs(elems, &[cm::Limb { q, quad, a }], None)
        .into_iter()
        .next()
        .unwrap()
}

use AdditionalLimb::*;

/// The challenge as an element of `R_648`: `c(-X^4)`, reduced modulo `q`.
fn embed_challenge(c: &ShortChallenge, q: u16) -> [u32; N] {
    let co = c.coeffs();
    let mut out = [0u32; N];
    for m in 0..N162 {
        let s = if m % 2 == 0 { co[m] as i32 } else { -(co[m] as i32) };
        out[4 * m] = s.rem_euclid(q as i32) as u32;
    }
    out
}

fn witness(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

/// One row of `A` as fully reduced residues.
fn a_elem<const Q: u16>(a: &[Batch32], i: usize) -> [u32; N] {
    let (b, p) = (i / 32, i % 32);
    core::array::from_fn(|j| (a[b].v[j][p] as i32).rem_euclid(Q as i32) as u32)
}

/// `y = sum_i A_i * NTT_q(w_i)` for a splitting limb: one scalar product per slot.
fn reference_split<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    let q = Q as u64;
    let mut y = [0u64; N];
    for i in 0..elems.len() / 4 {
        let w = scalar::ntt::<Q>(&f162::lift_elem(elems, i));
        let ai = a_elem::<Q>(a, i);
        for j in 0..N {
            y[j] = (y[j] + w[j] as u64 * ai[j] as u64) % q;
        }
    }
    core::array::from_fn(|j| y[j] as u32)
}

/// The same for a quadratic-slot limb: the leaf product `scalar::mul_quad_slots`.
fn reference_quad<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    let q = Q as u64;
    let mut y = [0u64; N];
    for i in 0..elems.len() / 4 {
        let w = scalar::ntt_quad::<Q>(&f162::lift_elem(elems, i));
        let p = scalar::mul_quad_slots::<Q>(&w, &a_elem::<Q>(a, i));
        for j in 0..N {
            y[j] = (y[j] + p[j] as u64) % q;
        }
    }
    core::array::from_fn(|j| y[j] as u32)
}

/// The commitment of one quadratic limb, straight through the kernel and through the key.
fn quad_limb<const Q: u16>(limb: AdditionalLimb, len_f162: usize, r: usize) {
    assert_eq!(limb.prime(), Q);
    let ck = CommitmentKey::random(len_f162, 0x11B ^ Q as u64, &[limb]);
    let w = witness(r * len_f162, 0x11C ^ Q as u64);
    let c = ck.commit(&w, r);
    for col in 0..r {
        let chunk = &w[col * len_f162..(col + 1) * len_f162];
        let want = reference_quad::<Q>(chunk, ck.row(1));

        // the kernel alone
        assert_eq!(commit_one(Q, true, chunk, ck.row(1)), want, "q = {Q}, chunk {col}");

        // the base limb of the same key is the splitting reference
        assert_eq!(
            commit_one(3889, false, chunk, ck.row(0)),
            reference_split::<3889>(chunk, ck.row(0))
        );

        // the four R_162 components of the quadratic limb
        let dec = scalar::decompose_quad_648_to_4x162::<Q>(&want);
        for k in 0..4 {
            for s in 0..N162 {
                let v = c.get(k, col).additional(0).v[s] as i32;
                assert!(2 * v.abs() <= Q as i32 - 1, "not centered: q = {Q}, slot {s}");
                assert_eq!(
                    v.rem_euclid(Q as i32) as u32,
                    dec[k][s],
                    "q = {Q}, chunk {col}, component {k}, slot {s}"
                );
            }
        }
        // and of the base limb
        let db = decompose_648_to_4x162::<3889>(&reference_split::<3889>(chunk, ck.row(0)));
        for k in 0..4 {
            for s in 0..N162 {
                assert_eq!(
                    (c.get(k, col).base().v[s] as i32).rem_euclid(3889) as u32,
                    db[k][s]
                );
            }
        }
    }
}

#[test]
fn commitment_2917() {
    quad_limb::<2917>(Q2917, 256, 2);
}

#[test]
fn commitment_4861() {
    quad_limb::<4861>(Q4861, 256, 2);
}

#[test]
fn commitment_12637() {
    quad_limb::<12637>(Q12637, 256, 2);
}

/// The Karatsuba `P_2` is only taken where the sum of two lazily reduced rows fits an i16 lane;
/// the other two primes use the schoolbook pair. Either way the fold-back periods have to hold.
#[test]
fn quad_bounds() {
    assert!(cm::karatsuba(2917));
    assert!(!cm::karatsuba(4861) && !cm::karatsuba(12637));
    for q in [2917u16, 4861, 12637] {
        for (per, period) in [
            (cm::acc_per_batch_quad01(q), cm::red_period_quad01(q)),
            (cm::acc_per_batch_quad2(q), cm::red_period_quad2(q)),
        ] {
            assert!(period.is_power_of_two());
            assert!(cm::acc_after_reduce(q) + period as i64 * per <= i32::MAX as i64);
            assert!(cm::acc_after_reduce(q) + 2 * period as i64 * per > i32::MAX as i64);
        }
        assert!(2 * cm::w_bound_quad(q) <= 2 * 32767);
    }
}

/// More batches than any fold-back period, with `A` at the extremes of the centered range and an
/// all-ones witness: the accumulators still agree with the scalar reference.
fn adversarial<const Q: u16>(limb: AdditionalLimb) {
    const FULL: F162 = F162([!0u64, !0u64, (1u64 << 34) - 1]);
    let nb = 2 * cm::red_period_quad01(Q) + 3;
    let ck = CommitmentKey::random(128 * nb, 0xAD ^ Q as u64, &[limb]);
    let elems = vec![FULL; 128 * nb];
    assert_eq!(
        commit_one(Q, true, &elems, ck.row(1)),
        reference_quad::<Q>(&elems, ck.row(1)),
        "q = {Q}: all-ones over {nb} batches"
    );
    let mixed = witness(128 * nb, 0xAE ^ Q as u64);
    assert_eq!(
        commit_one(Q, true, &mixed, ck.row(1)),
        reference_quad::<Q>(&mixed, ck.row(1))
    );
}

#[test]
fn adversarial_2917() {
    adversarial::<2917>(Q2917);
}

#[test]
fn adversarial_4861() {
    adversarial::<4861>(Q4861);
}

#[test]
fn adversarial_12637() {
    adversarial::<12637>(Q12637);
}

/// The default configuration is bit for bit what two single-limb kernel runs produce, key
/// included: the multi-limb driver must not have moved the existing commitment by one bit.
#[test]
fn default_configuration_unchanged() {
    for (len_f162, r) in [(256usize, 2usize), (512, 1)] {
        let ck = CommitmentKey::random_default(len_f162, 0xB17);
        assert_eq!(ck.limbs(), 2);
        assert_eq!(ck.additional(), &[Q9721]);
        let w = witness(r * len_f162, 0xB18);
        let c = ck.commit(&w, r);
        for col in 0..r {
            let chunk = &w[col * len_f162..(col + 1) * len_f162];
            let y3 = commit_one(3889, false, chunk, ck.row(0));
            let y9 = commit_one(9721, false, chunk, ck.row(1));
            let want = [
                decompose_648_to_4x162::<3889>(&y3),
                decompose_648_to_4x162::<9721>(&y9),
            ];
            for k in 0..4 {
                for (l, q) in [3889i32, 9721].into_iter().enumerate() {
                    for s in 0..N162 {
                        assert_eq!(
                            (c.get(k, col).limbs[l].v[s] as i32).rem_euclid(q) as u32,
                            want[l][k][s],
                            "the default configuration moved at component {k}, limb {l}, slot {s}"
                        );
                    }
                }
            }
        }
    }
}

/// A key over `additional` and its whole round: commit, claim, expand, fold, verify, and every
/// way of corrupting the prover's message.
fn pipeline<const LW: usize, const LR: usize>(len_f162: usize, additional: &[AdditionalLimb]) {
    let r = 1usize << LR;
    let ck = CommitmentKey::random(len_f162, 0x9E ^ additional.len() as u64, additional);
    let w = witness(r * len_f162, 0x9F ^ additional.len() as u64);

    let (c, aux) = ck.commit_with_aux(&w, r);
    assert_eq!(aux.limbs(), 1 + additional.len());
    assert_eq!(c.get(0, 0).len(), 1 + additional.len());

    // every limb of the commitment is the scalar reference in that limb's domain
    for col in 0..r {
        let chunk = &w[col * len_f162..(col + 1) * len_f162];
        for k in 0..ck.limbs() {
            let want = match (ck.prime(k), ck.is_quadratic(k)) {
                (3889, false) => reference_split::<3889>(chunk, ck.row(k)),
                (9721, false) => reference_split::<9721>(chunk, ck.row(k)),
                (2917, true) => reference_quad::<2917>(chunk, ck.row(k)),
                (4861, true) => reference_quad::<4861>(chunk, ck.row(k)),
                (12637, true) => reference_quad::<12637>(chunk, ck.row(k)),
                _ => unreachable!(),
            };
            assert_eq!(aux.commitment(k, col), &want, "limb {k} of chunk {col}");
        }
    }

    let mut t = Transcript::new(b"bin-ntt/test/limbs");
    for j in 0..r {
        t.absorb_elements(c.column(j));
    }
    let p: EvalPoint<LW, LR> = sample_point(&mut t);
    let claim = evaluate_mle(&w, &p);
    let u = left_expand(&w, &p.r0).u;
    assert!(check_claim(&u, &p.r1, claim));

    let ch: Vec<ShortChallenge> = (0..r)
        .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect();
    let out = fold::fold_checked(&ck, &aux, &ch);
    assert_eq!(out.v_ntt.len(), ck.limbs());
    assert_eq!(out.y_raw.len(), ck.limbs());
    assert_eq!(out.y[0].len(), ck.limbs());

    let raw = RawCommitments::from_aux(&aux);
    assert!(verify_fold(&ck, &raw, &ch, &out.v));
    let folded = fold_binary(&u, &ch);
    assert!(verify_binary(&p.r0, &out.v, folded));
    let v = Verifier {
        key: &ck,
        commitments: &raw,
        point: &p,
        claim,
    };
    assert!(v.verify(&u, &ch, &out.v));

    // sum_j c_j C_j == A v for every limb, recomputed in scalar arithmetic from the challenges
    // and the commitments alone — a scalar product per slot for a splitting limb, the leaf
    // product for a quadratic one.
    for k in 0..ck.limbs() {
        let (q, quad) = (ck.prime(k), ck.is_quadratic(k));
        let mut acc = [0u64; N];
        for (j, cj) in ch.iter().enumerate() {
            let e = embed_challenge(cj, q);
            let ct = match (q, quad) {
                (3889, false) => scalar::ntt::<3889>(&e),
                (9721, false) => scalar::ntt::<9721>(&e),
                (2917, true) => scalar::ntt_quad::<2917>(&e),
                (4861, true) => scalar::ntt_quad::<4861>(&e),
                (12637, true) => scalar::ntt_quad::<12637>(&e),
                _ => unreachable!(),
            };
            let cm_j = aux.commitment(k, j);
            let prod = match (q, quad) {
                (2917, true) => scalar::mul_quad_slots::<2917>(&ct, cm_j),
                (4861, true) => scalar::mul_quad_slots::<4861>(&ct, cm_j),
                (12637, true) => scalar::mul_quad_slots::<12637>(&ct, cm_j),
                _ => scalar::pointwise_mul(&ct, cm_j, q),
            };
            for u in 0..N {
                acc[u] = (acc[u] + prod[u] as u64) % q as u64;
            }
        }
        for u in 0..N {
            assert_eq!(
                acc[u] as u32, out.y_raw[k][u],
                "limb {k} (q = {q}): A v != sum_j c_j C_j at row {u}"
            );
        }
    }

    // one coefficient of v changed: rejected, whatever the limb list is
    let mut bad = out.v.clone();
    bad[0].v[0] += 1;
    assert!(!verify_fold(&ck, &raw, &ch, &bad));
    assert!(!v.verify(&u, &ch, &bad));

    // one commitment changed: the identity of the limb it belongs to fails
    for k in 0..ck.limbs() {
        let mut cs: Vec<[u32; N]> = (0..r).map(|j| *aux.commitment(k, j)).collect();
        cs[r / 2][3] = (cs[r / 2][3] + 1) % ck.prime(k) as u32;
        let mut acc = [0u64; N];
        for (j, cj) in ch.iter().enumerate() {
            let e = embed_challenge(cj, ck.prime(k));
            let ct = match (ck.prime(k), ck.is_quadratic(k)) {
                (3889, false) => scalar::ntt::<3889>(&e),
                (9721, false) => scalar::ntt::<9721>(&e),
                (2917, true) => scalar::ntt_quad::<2917>(&e),
                (4861, true) => scalar::ntt_quad::<4861>(&e),
                (12637, true) => scalar::ntt_quad::<12637>(&e),
                _ => unreachable!(),
            };
            let prod = match (ck.prime(k), ck.is_quadratic(k)) {
                (2917, true) => scalar::mul_quad_slots::<2917>(&ct, &cs[j]),
                (4861, true) => scalar::mul_quad_slots::<4861>(&ct, &cs[j]),
                (12637, true) => scalar::mul_quad_slots::<12637>(&ct, &cs[j]),
                _ => scalar::pointwise_mul(&ct, &cs[j], ck.prime(k)),
            };
            for u in 0..N {
                acc[u] = (acc[u] + prod[u] as u64) % ck.prime(k) as u64;
            }
        }
        assert!(
            (0..N).any(|u| acc[u] as u32 != out.y_raw[k][u]),
            "limb {k}: a changed commitment left the identity intact"
        );
    }

    // v outside the centered range of the base limb never reaches a transform
    let mut big = out.v.clone();
    big[0].v[1] = 3000;
    assert!(!verify_fold(&ck, &raw, &ch, &big));
}

#[test]
fn pipeline_default() {
    pipeline::<8, 2>(256, &[Q9721]);
}

#[test]
fn pipeline_2917() {
    pipeline::<8, 2>(256, &[Q2917]);
}

#[test]
fn pipeline_4861_12637() {
    pipeline::<8, 2>(256, &[Q4861, Q12637]);
}

#[test]
fn pipeline_all_four() {
    pipeline::<8, 2>(256, &[Q2917, Q4861, Q9721, Q12637]);
}

/// A key with no additional limb at all is the base commitment alone.
#[test]
fn base_only() {
    let ck = CommitmentKey::random(256, 0xBA5E, &[]);
    assert_eq!(ck.limbs(), 1);
    let w = witness(256, 0xBA5F);
    let c = ck.commit(&w, 1);
    assert_eq!(c.get(0, 0).len(), 1);
    let want = decompose_648_to_4x162::<3889>(&commit_one(3889, false, &w, ck.row(0)));
    for k in 0..4 {
        for s in 0..N162 {
            assert_eq!(
                (c.get(k, 0).base().v[s] as i32).rem_euclid(3889) as u32,
                want[k][s]
            );
        }
    }
    let zero = PowerOfThreeRingElementWithLimbs::zero(1);
    assert_eq!(zero.len(), 1);
}

/// The same limb twice is a mistake.
#[test]
#[should_panic]
fn duplicate_limb() {
    CommitmentKey::random(256, 1, &[Q2917, Q2917]);
}

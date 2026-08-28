//! The folding step: the commitment's auxiliary data against the scalar transform, the folded
//! witness against `sum_j c_j W_j` computed with `scalar::mul_mod_phi` (and over the integers),
//! the linearity identity `A v = sum_j c_j C_j` for both primes, and determinism.
use bin_fields::scalar::F162;
use bin_ntt::api::{AuxData, N162, PRIMES};
use bin_ntt::challenge::{sample_short_challenge, ShortChallenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
use bin_ntt::f162::{self, RandomF162};
use bin_ntt::params::N;
use bin_ntt::rng::Rng;
use bin_ntt::types::Representation;
use bin_ntt::{fold, scalar, CommitmentKey};

const Q1: u16 = PRIMES[0];
const Q2: u16 = PRIMES[1];

fn witness(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

/// `r` challenges bound to the commitment, as a verifier would draw them.
fn challenges(
    c: &bin_ntt::VerticallyAlignedMatrix<bin_ntt::PowerOfThreeRingElementWithTwoLimbs>,
    r: usize,
) -> Vec<ShortChallenge> {
    let mut t = Transcript::new(b"bin-ntt/test/fold");
    for j in 0..r {
        t.absorb_elements(c.column(j));
    }
    (0..r)
        .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect()
}

/// The challenge as an element of `R_648`: `c(-X^4)`, coefficient of `X^{4m}` is `(-1)^m c_m`.
fn embed(c: &ShortChallenge) -> [i64; N] {
    let co = c.coeffs();
    let mut out = [0i64; N];
    for m in 0..N162 {
        out[4 * m] = if m % 2 == 0 { co[m] as i64 } else { -(co[m] as i64) };
    }
    out
}

/// `a * b` in `Z[X]/(X^648 - X^324 + 1)` over the integers: `X^k = X^{k-324} - X^{k-648}`.
fn mul_mod_phi_int(a: &[i64; N], b: &[i64; N]) -> [i64; N] {
    let mut c = [0i64; 2 * N];
    for i in 0..N {
        if a[i] == 0 {
            continue;
        }
        for j in 0..N {
            c[i + j] += a[i] * b[j];
        }
    }
    for k in (N..2 * N).rev() {
        let v = c[k];
        c[k] = 0;
        c[k - 324] += v;
        c[k - N] -= v;
    }
    let mut out = [0i64; N];
    out.copy_from_slice(&c[..N]);
    out
}

/// The auxiliary transform is `scalar::ntt::<3889>` of the lifted ring element, slot for slot.
fn check_aux(aux: &AuxData, w: &[F162], len_ring: usize) {
    let bpc = aux.batches_per_chunk();
    for j in 0..aux.chunks() {
        for b in 0..bpc {
            for p in 0..32 {
                let i = j * len_ring + 32 * b + p;
                let want = scalar::ntt::<Q1>(&f162::lift_elem(w, i));
                let got = aux.batch(j * bpc + b).get(p);
                assert_eq!(got.representation, Representation::Ntt);
                for u in 0..N {
                    assert_eq!(
                        (got.v[u] as i32).rem_euclid(Q1 as i32) as u32,
                        want[u],
                        "element {i}, slot {u}"
                    );
                    assert!(
                        (got.v[u] as i32).unsigned_abs() <= 7 * Q1 as u32 + Q1 as u32 / 2,
                        "element {i}, slot {u}: |{}| over the kernel bound",
                        got.v[u]
                    );
                }
            }
        }
    }
}

/// `commit_with_aux` on `r` chunks of `len_f162` `F162`, checked against everything scalar.
fn run(len_f162: usize, r: usize, deep: bool) {
    let ck = CommitmentKey::random(len_f162, 0xF01D ^ r as u64);
    let len_ring = len_f162 / 4;
    let w = witness(r * len_f162, 0xC0FFEE ^ (r as u64) << 8);

    let plain = ck.commit(&w, r);
    let (c, aux) = ck.commit_with_aux(&w, r);
    assert_eq!(plain, c, "commit_with_aux changed the commitment");
    assert_eq!(aux.chunks(), r);
    assert_eq!(aux.batches_per_chunk(), len_ring / 32);
    if deep {
        check_aux(&aux, &w, len_ring);
    }

    let ch = challenges(&c, r);
    let out = fold::fold_checked(&ck, &aux, &ch);

    assert_eq!(out.v.len(), len_ring);
    assert_eq!(out.v_ntt[0].len(), len_ring / 32);
    assert_eq!(out.v_ntt[1].len(), len_ring / 32);

    // v == sum_j c_j W_j, over Z_3889 and (because the coefficients are small) over Z.
    let emb: Vec<[i64; N]> = ch.iter().map(embed).collect();
    let mut max_abs = 0i64;
    for i in 0..len_ring {
        let mut want = [0i64; N];
        for j in 0..r {
            let wi = f162::lift_elem(&w, j * len_ring + i);
            let mut b = [0i64; N];
            for k in 0..N {
                b[k] = wi[k] as i64;
            }
            let p = mul_mod_phi_int(&emb[j], &b);
            for k in 0..N {
                want[k] += p[k];
            }
        }
        for k in 0..N {
            max_abs = max_abs.max(want[k].abs());
            assert_eq!(
                out.v[i].v[k] as i64, want[k],
                "element {i}, coefficient {k}: the fold is not the integer sum"
            );
        }
        assert_eq!(out.v[i].representation, Representation::Coefficients);

        // the same, through the crate's own modular reference
        let mut ref_q = [0u32; N];
        for j in 0..r {
            let cj: [u32; N] = core::array::from_fn(|k| emb[j][k].rem_euclid(Q1 as i64) as u32);
            let p = scalar::mul_mod_phi(&cj, &f162::lift_elem(&w, j * len_ring + i), Q1);
            for k in 0..N {
                ref_q[k] = (ref_q[k] + p[k]) % Q1 as u32;
            }
        }
        for k in 0..N {
            assert_eq!((out.v[i].v[k] as i32).rem_euclid(Q1 as i32) as u32, ref_q[k]);
        }

        // the four R_162 components are the coefficients 4m + k
        let comp = out.v_components(i);
        for m in 0..N162 {
            for k in 0..4 {
                assert_eq!(comp[k][m], out.v[i].v[4 * m + k]);
            }
        }
    }
    assert_eq!(out.max_abs_v as i64, max_abs);
    assert!(2 * max_abs < Q1 as i64, "max |v| = {max_abs} does not fit q1/2");

    // v_ntt is the transform of v, both primes.
    for i in 0..len_ring {
        let coeffs: [u32; N] =
            core::array::from_fn(|k| (out.v[i].v[k] as i32).rem_euclid(Q1 as i32) as u32);
        let want1 = scalar::ntt::<Q1>(&coeffs);
        let coeffs2: [u32; N] =
            core::array::from_fn(|k| (out.v[i].v[k] as i32).rem_euclid(Q2 as i32) as u32);
        let want2 = scalar::ntt::<Q2>(&coeffs2);
        let g1 = out.v_ntt[0][i / 32].get(i % 32);
        let g2 = out.v_ntt[1][i / 32].get(i % 32);
        for u in 0..N {
            assert_eq!((g1.v[u] as i32).rem_euclid(Q1 as i32) as u32, want1[u]);
            assert_eq!((g2.v[u] as i32).rem_euclid(Q2 as i32) as u32, want2[u]);
            assert!(2 * (g1.v[u] as i32).abs() <= Q1 as i32 - 1);
            assert!(2 * (g2.v[u] as i32).abs() <= Q2 as i32 - 1);
        }
    }

    // A v = sum_j c_j C_j, slot by slot, both primes.
    for (k, &q) in PRIMES.iter().enumerate() {
        let chq: Vec<[u32; N]> = (0..r)
            .map(|j| {
                let cj: [u32; N] = core::array::from_fn(|t| emb[j][t].rem_euclid(q as i64) as u32);
                if q == Q1 {
                    scalar::ntt::<Q1>(&cj)
                } else {
                    scalar::ntt::<Q2>(&cj)
                }
            })
            .collect();
        for u in 0..N {
            let mut s = 0u64;
            for j in 0..r {
                s += chq[j][u] as u64 * aux.commitment(k, j)[u] as u64;
            }
            assert_eq!(
                (s % q as u64) as u32,
                out.y_raw[k][u],
                "q = {q}: A v != sum_j c_j C_j at slot {u}"
            );
            assert!(out.y_raw[k][u] < q as u32);
        }
        // y is the four-way decomposition of y_raw
        let want = bin_ntt::api::decompose_648_to_4x162::<Q1>(&out.y_raw[0]);
        let want2 = bin_ntt::api::decompose_648_to_4x162::<Q2>(&out.y_raw[1]);
        for t in 0..4 {
            for s in 0..N162 {
                assert_eq!(
                    (out.y[t].limb[0].v[s] as i32).rem_euclid(Q1 as i32) as u32,
                    want[t][s]
                );
                assert_eq!(
                    (out.y[t].limb[1].v[s] as i32).rem_euclid(Q2 as i32) as u32,
                    want2[t][s]
                );
            }
        }
    }
}

#[test]
fn small_instance() {
    run(128, 4, true);
}

#[test]
fn two_batches_per_chunk() {
    run(256, 8, true);
}

/// 64 chunks crosses the 32-chunk fold-back of the accumulator.
#[test]
fn crosses_the_fold_back() {
    run(128, 64, false);
}

/// Two folds of the same inputs agree bit for bit, and so do two commitments.
#[test]
fn deterministic() {
    let (len_f162, r) = (256usize, 8usize);
    let ck = CommitmentKey::random(len_f162, 0xD37);
    let w = witness(r * len_f162, 0xD37E);
    let (c1, a1) = ck.commit_with_aux(&w, r);
    let (c2, a2) = ck.commit_with_aux(&w, r);
    assert_eq!(c1, c2);
    let ch = challenges(&c1, r);
    let f1 = fold::fold(&ck, &a1, &ch);
    let f2 = fold::fold(&ck, &a2, &ch);
    assert_eq!(f1.v, f2.v);
    assert_eq!(f1.y_raw, f2.y_raw);
    assert_eq!(f1.y, f2.y);
    assert_eq!(f1.max_abs_v, f2.max_abs_v);
    for k in 0..2 {
        for b in 0..f1.v_ntt[k].len() {
            assert_eq!(f1.v_ntt[k][b].v, f2.v_ntt[k][b].v);
        }
    }
}

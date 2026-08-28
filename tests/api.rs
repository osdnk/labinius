//! The public API: the four-way decomposition of an `R_648` element into `R_162` components
//! against direct scalar evaluation, [`CommitmentKey::commit`] against the scalar reference
//! commitment for `r = 1, 2, 4`, and the matrix's indexing. Both primes throughout.
use bin_fields::scalar::F162;
use bin_ntt::api::{decompose_648_to_4x162, pow3_slot_exp, N162, POW3_SLOT_EXP, PRIMES};
use bin_ntt::f162::{self, RandomF162};
use bin_ntt::params::{pow_mod, Params, N, SLOT_EXP};
use bin_ntt::rng::Rng;
use bin_ntt::types::Batch32;
use bin_ntt::{
    scalar, CommitmentKey, PowerOfThreeRingElementWithTwoLimbs, VerticallyAlignedMatrix,
};

const CONDUCTOR: u64 = 1944;
const CONDUCTOR162: u64 = 486;

fn random_coeffs(q: u16, seed: u64) -> [u32; N] {
    let mut rng = Rng::new(seed);
    let mut c = [0u32; N];
    for x in c.iter_mut() {
        *x = rng.below(q as u32);
    }
    c
}

/// `y_k(Y) = sum_m y[4m + k] Y^m`, the four components in coefficient form.
fn split_mod4(y: &[u32; N]) -> [[u32; N162]; 4] {
    let mut out = [[0u32; N162]; 4];
    for m in 0..N162 {
        for k in 0..4 {
            out[k][m] = y[4 * m + k];
        }
    }
    out
}

fn eval<const Q: u16>(p: &[u32; N162], x: u64) -> u32 {
    let q = Q as u64;
    let mut acc = 0u64;
    for m in (0..N162).rev() {
        acc = (acc * x + p[m] as u64) % q;
    }
    acc as u32
}

/// The slot table is the first-appearance order of `SLOT_EXP[j] mod 486`, and every class has
/// exactly four members.
#[test]
fn slot_table() {
    assert_eq!(pow3_slot_exp(), &POW3_SLOT_EXP);
    let mut order: Vec<u16> = Vec::new();
    let mut count = std::collections::HashMap::new();
    for &u in SLOT_EXP.iter() {
        let v = u % CONDUCTOR162 as u16;
        if !order.contains(&v) {
            order.push(v);
        }
        *count.entry(v).or_insert(0) += 1;
    }
    assert_eq!(order.len(), N162);
    assert_eq!(order.as_slice(), &POW3_SLOT_EXP[..]);
    for v in &order {
        assert_eq!(count[v], 4, "class {v} does not have four lifts");
        assert_eq!(v % 2, 1, "class {v} is not a unit mod 1944");
        assert_ne!(v % 3, 0);
    }
}

/// `decompose_648_to_4x162(ntt(y))` is the four coefficient-split components, each evaluated at
/// `theta^{v_s}` — and the identity `E_t = sum_k psi^{vk} i^{tk} Y_k(v)` holds at every slot.
fn decomposition<const Q: u16>() {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    let theta = pow_mod(psi, 4, q);
    let i4 = pow_mod(psi, CONDUCTOR162, q);

    for seed in [1u64, 2, 3] {
        let y = random_coeffs(Q, seed ^ q);
        let parts = split_mod4(&y);
        let got = decompose_648_to_4x162::<Q>(&scalar::ntt::<Q>(&y));

        for s in 0..N162 {
            let v = POW3_SLOT_EXP[s] as u64;
            let x = pow_mod(theta, v, q);
            for k in 0..4 {
                assert_eq!(
                    got[k][s],
                    eval::<Q>(&parts[k], x),
                    "q = {Q}, seed {seed}: component {k}, slot {s} (v = {v})"
                );
                assert!(got[k][s] < Q as u32);
            }
        }

        // E_t = sum_k psi^{vk} i^{tk} Y_k(v), read off the big transform slot by slot.
        let e = scalar::ntt::<Q>(&y);
        for j in 0..N {
            let u = SLOT_EXP[j] as u64;
            let v = u % CONDUCTOR162;
            let t = (u - v) / CONDUCTOR162;
            let s = POW3_SLOT_EXP.iter().position(|&w| w as u64 == v).unwrap();
            let mut acc = 0u64;
            for k in 0..4 {
                acc += pow_mod(psi, v * k as u64 % CONDUCTOR, q) * pow_mod(i4, t * k as u64 % 4, q)
                    % q
                    * got[k][s] as u64;
            }
            assert_eq!(
                acc % q,
                e[j] as u64,
                "q = {Q}, seed {seed}: recombination at slot {j}"
            );
        }
    }
}

#[test]
fn decomposition_3889() {
    decomposition::<3889>();
}

#[test]
fn decomposition_9721() {
    decomposition::<9721>();
}

/// `y[j] = sum_i A_i[j] * NTT_q(w_i)[j] mod q` in scalar arithmetic.
fn reference<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0i64; N];
    for i in 0..elems.len() / 4 {
        let w = scalar::ntt::<Q>(&f162::lift_elem(elems, i));
        let (b, p) = (i / 32, i % 32);
        for j in 0..N {
            y[j] = (y[j] + w[j] as i64 * a[b].v[j][p] as i64) % q;
        }
    }
    let mut out = [0u32; N];
    for j in 0..N {
        out[j] = y[j].rem_euclid(q) as u32;
    }
    out
}

fn random_witness(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

fn check_commit(len_f162: usize, r: usize) {
    let ck = CommitmentKey::random(len_f162, 0x5EED ^ r as u64);
    assert_eq!(ck.len_f162(), len_f162);
    let witness = random_witness(r * len_f162, 0xBEEF_CAFEu64.wrapping_mul(r as u64 + 1));
    let c = ck.commit(&witness, r);
    assert_eq!(c.rows(), 4);
    assert_eq!(c.cols(), r);

    for col in 0..r {
        let chunk = &witness[col * len_f162..(col + 1) * len_f162];
        let want = [
            decompose_648_to_4x162::<{ PRIMES[0] }>(&reference::<{ PRIMES[0] }>(chunk, ck.row(0))),
            decompose_648_to_4x162::<{ PRIMES[1] }>(&reference::<{ PRIMES[1] }>(chunk, ck.row(1))),
        ];
        for k in 0..4 {
            for l in 0..2 {
                for s in 0..N162 {
                    assert_eq!(
                        c.get(k, col).limb[l].v[s] as u32,
                        want[l][k][s],
                        "r = {r}, column {col}, component {k}, limb {l}, slot {s}"
                    );
                }
            }
            assert_eq!(c.column(col)[k], *c.get(k, col));
        }
    }
}

#[test]
fn commit_r1() {
    check_commit(256, 1);
}

#[test]
fn commit_r2() {
    check_commit(256, 2);
}

#[test]
fn commit_r4() {
    check_commit(256, 4);
}

/// A key of a different length, and the zero witness.
#[test]
fn commit_zero_and_sizes() {
    let ck = CommitmentKey::random(512, 9);
    assert_eq!(ck.len_ring(), 128);
    assert_eq!(ck.bytes(), 2 * 4 * core::mem::size_of::<Batch32>());
    let zero = vec![F162([0; 3]); 2 * 512];
    let c = ck.commit(&zero, 2);
    for e in c.iter() {
        assert_eq!(*e, PowerOfThreeRingElementWithTwoLimbs::zero());
    }
}

#[test]
#[should_panic]
fn commit_wrong_length() {
    let ck = CommitmentKey::random(256, 1);
    ck.commit(&random_witness(256, 2), 2);
}

#[test]
#[should_panic]
fn commit_r_not_a_power_of_two() {
    let ck = CommitmentKey::random(256, 1);
    ck.commit(&random_witness(3 * 256, 2), 3);
}

#[test]
fn matrix_indexing() {
    let m = VerticallyAlignedMatrix::new(4, 3, (0..12).collect::<Vec<i32>>());
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 3);
    for col in 0..3 {
        assert_eq!(
            m.column(col),
            &[
                4 * col as i32,
                4 * col as i32 + 1,
                4 * col as i32 + 2,
                4 * col as i32 + 3
            ]
        );
        for row in 0..4 {
            assert_eq!(*m.get(row, col), (4 * col + row) as i32);
        }
    }
    assert_eq!(m.columns().count(), 3);
    assert_eq!(m.iter().sum::<i32>(), 66);
    assert_eq!(m.as_slice().len(), 12);
}

#[test]
#[should_panic]
fn matrix_out_of_range() {
    let m = VerticallyAlignedMatrix::new(4, 3, (0..12).collect::<Vec<i32>>());
    m.get(4, 0);
}

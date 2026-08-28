//! The left-expansion over `F162` and the verifier: the vectorised evaluation against a naive
//! scalar multilinear extension, the left expansion against naive column sums, the binary fold,
//! the linearity `B (W c) = (B W) c`, and the whole pipeline — commit, expand, challenge, fold —
//! through `verify_fold` and `verify_binary`, including two corruptions.
use bin_fields::scalar::F162;
use bin_ntt::eval::{
    self, check_claim, components_mod_2, eq_table, evaluate_mle, fold_binary, left_expand,
    sample_point, verify_binary, verify_fold, EvalPoint, RawCommitments, Verifier,
};
use bin_ntt::f162::RandomF162;
use bin_ntt::rng::Rng;
use bin_ntt::{
    fold, sample_short_challenge, CommitmentKey, ShortChallenge, Transcript, DEFAULT_BOUND,
    DEFAULT_WEIGHT,
};

fn witness(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

/// `eq(rs, b)` straight from the definition.
fn eq_naive(rs: &[F162], b: usize) -> F162 {
    let mut p = F162::ONE;
    for (k, &r) in rs.iter().enumerate() {
        p = p * if (b >> k) & 1 == 1 { r } else { F162::ONE + r };
    }
    p
}

/// The multilinear extension by the direct sum over all `2^nu` points, scalar `Mul` throughout.
fn mle_naive<const LW: usize, const LR: usize>(w: &[F162], p: &EvalPoint<LW, LR>) -> F162 {
    let mut t = F162::ZERO;
    for b in 0..w.len() {
        t += eq_naive(&p.r0, b & ((1 << LW) - 1)) * eq_naive(&p.r1, b >> LW) * w[b];
    }
    t
}

/// `u_j = sum_i eq(r0, i) W[i + wdim j]`, scalar.
fn columns_naive(w: &[F162], r0: &[F162], wdim: usize) -> Vec<F162> {
    (0..w.len() / wdim)
        .map(|j| {
            let mut s = F162::ZERO;
            for i in 0..wdim {
                s += eq_naive(r0, i) * w[i + wdim * j];
            }
            s
        })
        .collect()
}

fn challenges(
    c: &bin_ntt::VerticallyAlignedMatrix<bin_ntt::PowerOfThreeRingElementWithLimbs>,
    r: usize,
) -> Vec<ShortChallenge> {
    let mut t = Transcript::new(b"bin-ntt/test/eval");
    for j in 0..r {
        t.absorb_elements(c.column(j));
    }
    (0..r)
        .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect()
}

#[test]
fn eq_table_is_the_definition() {
    let rs = witness(9, 0xE9);
    for len in 0..=9 {
        let t = eq_table(&rs[..len]);
        assert_eq!(t.len(), 1 << len);
        let mut sum = F162::ZERO;
        for b in 0..1 << len {
            assert_eq!(t[b], eq_naive(&rs[..len], b), "len {len}, index {b}");
            sum += t[b];
        }
        assert_eq!(sum, F162::ONE, "sum_b eq(r, b) != 1");
    }
}

/// `wdim = 128` rows, `r = 4` columns: the vectorised path against the scalar one, both stages.
#[test]
fn small_instance_against_scalar() {
    const LW: usize = 7;
    const LR: usize = 2;
    let w = witness(1 << (LW + LR), 0xC0FFEE);
    let mut t = Transcript::new(b"bin-ntt/test/eval/point");
    let p: EvalPoint<LW, LR> = sample_point(&mut t);

    let want = mle_naive(&w, &p);
    assert_eq!(evaluate_mle(&w, &p), want);

    let u = left_expand(&w, &p.r0).u;
    assert_eq!(u, columns_naive(&w, &p.r0, 1 << LW));
    assert!(check_claim(&u, &p.r1, want));
    assert_eq!(eval::claim(&u, &p.r1), want);
    assert!(!check_claim(&u, &p.r1, want + F162::ONE));
}

/// The kernel's 8-element blocks and its scalar tail: every row count from 1 to 32 and every
/// column count from 1 to 20.
#[test]
fn tails_and_blocks() {
    let w = witness(32 * 20, 0x7A11);
    let rs = witness(5, 0x7A12);
    fn rows<const LW: usize>(w: &[F162], rs: &[F162]) {
        let r0: [F162; LW] = rs[..LW].try_into().unwrap();
        for cols in 1..=20 {
            let n = (1 << LW) * cols;
            assert_eq!(
                left_expand(&w[..n], &r0).u,
                columns_naive(&w[..n], &r0, 1 << LW),
                "2^{LW} rows, {cols} columns"
            );
        }
    }
    rows::<0>(&w, &rs);
    rows::<1>(&w, &rs);
    rows::<2>(&w, &rs);
    rows::<3>(&w, &rs);
    rows::<4>(&w, &rs);
    rows::<5>(&w, &rs);
}

#[test]
fn binary_fold_is_the_inner_product() {
    let u = witness(8, 0xB1);
    let mut t = Transcript::new(b"bin-ntt/test/eval/binary");
    let ch: Vec<ShortChallenge> = (0..8)
        .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect();
    let mut want = F162::ZERO;
    for j in 0..8 {
        let c = ch[j].to_f162();
        for (i, &s) in ch[j].coeffs().iter().enumerate() {
            assert_eq!((c.0[i >> 6] >> (i & 63)) & 1, (s != 0) as u64, "position {i}");
        }
        want += u[j] * c;
    }
    assert_eq!(fold_binary(&u, &ch), want);
}

/// The whole round: commit, sample a point, state the claim, expand, fold, verify.
fn pipeline<const LW: usize, const LR: usize>(len_f162: usize) {
    let r = 1usize << LR;
    let wdim = 1usize << LW;
    assert_eq!(len_f162, wdim);
    let ck = CommitmentKey::random_default(len_f162, 0xE7A1 ^ r as u64);
    let w = witness(r * len_f162, 0xE7A2 ^ (r as u64) << 8);

    let (c, aux) = ck.commit_with_aux(&w, r);
    let mut t = Transcript::new(b"bin-ntt/test/eval/pipeline");
    for j in 0..r {
        t.absorb_elements(c.column(j));
    }
    let p: EvalPoint<LW, LR> = sample_point(&mut t);
    let claim = evaluate_mle(&w, &p);
    assert_eq!(claim, mle_naive(&w, &p));

    let u = left_expand(&w, &p.r0).u;
    assert_eq!(u.len(), r);
    assert!(check_claim(&u, &p.r1, claim));

    let ch = challenges(&c, r);
    let out = fold::fold_checked(&ck, &aux, &ch);
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

    // B (W c) = (B W) c, both sides straight from the definition.
    let vm = components_mod_2(&out.v);
    assert_eq!(vm.len(), wdim);
    let mut left = F162::ZERO;
    for i in 0..wdim {
        left += eq_naive(&p.r0, i) * vm[i];
    }
    let mut right = F162::ZERO;
    for j in 0..r {
        let mut col = F162::ZERO;
        for i in 0..wdim {
            col += eq_naive(&p.r0, i) * w[i + wdim * j];
        }
        right += col * ch[j].to_f162();
    }
    assert_eq!(left, right, "B (W c) != (B W) c");
    assert_eq!(left, folded);

    // W c mod 2 is the fold read component by component.
    for i in 0..wdim {
        let mut s = F162::ZERO;
        for j in 0..r {
            s += w[i + wdim * j] * ch[j].to_f162();
        }
        assert_eq!(vm[i], s, "component {i} of v is not sum_j c_j W[i, j] mod 2");
    }

    // one coefficient of v changed: A v no longer matches, and neither does B v.
    let mut bad = out.v.clone();
    bad[0].v[0] += 1;
    assert!(!verify_fold(&ck, &raw, &ch, &bad));
    assert!(!verify_binary(&p.r0, &bad, folded));

    // one entry of u changed: the claim and the binary check both fail.
    let mut bu = u.clone();
    bu[r / 2] += F162::ONE;
    assert!(!check_claim(&bu, &p.r1, claim));
    assert!(!verify_binary(&p.r0, &out.v, fold_binary(&bu, &ch)));
    assert!(!v.verify(&bu, &ch, &out.v));

    // v out of the centered range of q1 is rejected before it is transformed.
    let mut big = out.v.clone();
    big[1].v[5] = 3000;
    assert!(!verify_fold(&ck, &raw, &ch, &big));
}

#[test]
fn pipeline_small() {
    pipeline::<7, 2>(128);
}

#[test]
fn pipeline_two_batches() {
    pipeline::<8, 3>(256);
}

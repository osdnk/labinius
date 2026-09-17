//! The `F162` side: the multilinear extension and the row evaluation against naive scalar
//! references, the claim check, and the two ways of getting it wrong.
use labinius::{
    EvaluationPoint, Params, Prover, PublicParameters, Transcript, VerificationError,
    Verifier, Witness, F162,
};

mod common;
use common::*;

const MATRIX_SEED: [u8; 32] = [31u8; 32];
const WITNESS_SEED: [u8; 32] = [37u8; 32];

/// `eq(ps, b)` straight from the definition.
fn eq_naive(ps: &[F162], b: usize) -> F162 {
    let mut p = F162::ONE;
    for (k, &x) in ps.iter().enumerate() {
        p = p * if (b >> k) & 1 == 1 { x } else { F162::ONE + x };
    }
    p
}

/// The multilinear extension by the direct sum over all `2^nu` points, scalar `Mul` throughout.
fn mle_naive(witness: &[F162], point: &EvaluationPoint) -> F162 {
    let wdim = 1usize << point.p0().len();
    let mut t = F162::ZERO;
    for b in 0..witness.len() {
        t += eq_naive(point.p0(), b % wdim) * eq_naive(point.p1(), b / wdim) * witness[b];
    }
    t
}

/// `u_j = sum_i eq(p0, i) W[i + wdim j]`, scalar.
fn row_naive(witness: &[F162], point: &EvaluationPoint) -> Vec<F162> {
    let wdim = 1usize << point.p0().len();
    witness
        .chunks_exact(wdim)
        .map(|column| {
            let mut s = F162::ZERO;
            for (i, &x) in column.iter().enumerate() {
                s += eq_naive(point.p0(), i) * x;
            }
            s
        })
        .collect()
}

struct Setup {
    verifier: Verifier,
    witness: Witness,
    point: EvaluationPoint,
    row_evaluation: labinius::RowEvaluation,
    challenges: labinius::FoldingChallenges,
}

fn setup(params: Params) -> Setup {
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, _) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"labinius/test/eval");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    Setup {
        verifier,
        witness,
        point,
        row_evaluation,
        challenges,
    }
}

#[test]
fn the_point_has_one_variable_per_index_bit() {
    let params = small();
    let s = setup(params.clone());
    assert_eq!(s.point.p1().len(), params.column_log_len as usize);
    assert_eq!(
        s.point.p0().len(),
        (params.witness_log_len - params.column_log_len) as usize
    );
    for x in s.point.p0().iter().chain(s.point.p1()) {
        assert_eq!(x.0[2] >> 34, 0, "a point coordinate left F162");
    }
}

#[test]
fn the_row_evaluation_matches_the_scalar_reference() {
    let s = setup(small());
    assert_eq!(
        s.row_evaluation.values(),
        &row_naive(s.witness.elements(), &s.point)[..]
    );
}

#[test]
fn the_claimed_value_matches_the_scalar_reference() {
    let s = setup(small());
    assert_eq!(
        s.witness.mle_evaluate(&s.point),
        mle_naive(s.witness.elements(), &s.point)
    );
}

#[test]
fn the_claim_check_accepts_the_honest_value() {
    let s = setup(small());
    let claimed_value = s.witness.mle_evaluate(&s.point);
    assert_eq!(
        s.verifier
            .verify_evaluation(&s.point, &claimed_value, &s.row_evaluation),
        Ok(())
    );
}

#[test]
fn a_wrong_claimed_value_is_rejected() {
    let s = setup(small());
    let claimed_value = s.witness.mle_evaluate(&s.point) + F162::ONE;
    assert_eq!(
        s.verifier
            .verify_evaluation(&s.point, &claimed_value, &s.row_evaluation),
        Err(VerificationError::Rejected)
    );
}

#[test]
fn a_corrupted_row_evaluation_is_rejected() {
    let s = setup(small());
    let claimed_value = s.witness.mle_evaluate(&s.point);
    for j in 0..s.row_evaluation.values().len() {
        let mut corrupted = s.row_evaluation.clone();
        corrupted.values_mut()[j] += F162::ONE;
        assert_eq!(
            s.verifier
                .verify_evaluation(&s.point, &claimed_value, &corrupted),
            Err(VerificationError::Rejected)
        );
    }
}

#[test]
fn the_binary_fold_is_linear_in_the_row_evaluation() {
    let s = setup(small());
    let mut shifted = s.row_evaluation.clone();
    for x in shifted.values_mut().iter_mut() {
        *x += F162::ONE;
    }
    let ones = {
        let mut r = s.row_evaluation.clone();
        for x in r.values_mut().iter_mut() {
            *x = F162::ONE;
        }
        r
    };
    assert_eq!(
        s.verifier.fold_row_evaluation(&shifted, &s.challenges),
        s.verifier
            .fold_row_evaluation(&s.row_evaluation, &s.challenges)
            + s.verifier.fold_row_evaluation(&ones, &s.challenges)
    );
}

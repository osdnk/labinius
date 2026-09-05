//! The fold: the shape and the size of the amortised witness, determinism, the opening check
//! `A v = sum_j c_j C_j` on every modulus, and what the verifier does with a corrupted opening.
use bin_ntt::{Opening, OpeningMessage};
use bin_ntt::{
    Modulus, Params, Prover, PublicParameters, Transcript, VerificationError, Verifier, Witness,
};

mod common;
use common::*;

use Modulus::*;

const MATRIX_SEED: [u8; 32] = [21u8; 32];
const WITNESS_SEED: [u8; 32] = [23u8; 32];

/// Everything one round produces, so that a test can corrupt any of it.
struct Round {
    verifier: Verifier,
    commitment: bin_ntt::Commitment,
    point: bin_ntt::EvaluationPoint,
    claimed_value: bin_ntt::F162,
    row_evaluation: bin_ntt::RowEvaluation,
    challenges: bin_ntt::FoldingChallenges,
    folded_witness: bin_ntt::FoldedWitness,
    folded_commitment: bin_ntt::FoldedCommitment,
    folded_row_value: bin_ntt::F162,
}

fn round(params: Params) -> Round {
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/fold");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claimed_value = witness.mle_evaluate(&point);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded_witness = prover.fold(opening, &challenges);
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &challenges);
    Round {
        verifier,
        commitment,
        point,
        claimed_value,
        row_evaluation,
        challenges,
        folded_witness,
        folded_commitment,
        folded_row_value,
    }
}

impl Round {
    fn verify(&self) -> Result<(), VerificationError> {
        self.verifier
            .verify_evaluation(&self.point, &self.claimed_value, &self.row_evaluation)?;
        self.verifier
            .verify_opening(
                &self.commitment,
                &self.challenges,
                &self.point,
                OpeningMessage::Clear {
                    folded_commitment: &self.folded_commitment,
                    folded_witness: &self.folded_witness,
                    folded_row_value: &self.folded_row_value,
                },
            )
            .map(|_| ())
    }
}

#[test]
fn an_honest_round_is_accepted() {
    assert_eq!(round(small()).verify(), Ok(()));
}

#[test]
fn the_folded_witness_is_one_short_column() {
    let params = small();
    let r = round(params.clone());
    assert_eq!(r.challenges.len(), params.columns());
    assert_eq!(
        r.folded_witness.len(),
        params.witness_len() / params.columns() / 4
    );
    // A coefficient is a sum of r * 21 signed 0/1 terms, so it stays far inside q1 / 2 = 1944.5.
    let max = r
        .folded_witness
        .elements()
        .iter()
        .flat_map(|e| e.v.iter())
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap();
    assert!(max <= 1944, "the folded witness left the centered range");
    assert!(max < 400, "the folded witness is unexpectedly large: {max}");
}

#[test]
fn the_fold_is_deterministic() {
    let a = round(small());
    let b = round(small());
    assert_eq!(a.folded_witness, b.folded_witness);
    assert_eq!(a.folded_commitment, b.folded_commitment);
    assert_eq!(a.folded_row_value, b.folded_row_value);
}

#[test]
fn a_corrupted_folded_witness_is_rejected() {
    for shift in [1i16, -1, 1000] {
        for index in [0usize, 5, 31] {
            let mut r = round(small());
            let target = index % r.folded_witness.len();
            r.folded_witness.elements_mut()[target].v[3] += shift;
            assert_eq!(r.verify(), Err(VerificationError::Rejected));
        }
    }
}

#[test]
fn a_folded_witness_outside_the_centered_range_is_rejected() {
    let mut r = round(small());
    r.folded_witness.elements_mut()[0].v[0] = 1945;
    assert_eq!(r.verify(), Err(VerificationError::Rejected));
}

#[test]
fn the_folded_commitment_is_bound_to_the_challenges() {
    let params = small();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);

    let mut transcript = Transcript::new(b"bin-ntt/test/fold");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded_witness = prover.fold(opening, &challenges);

    let mut other = Transcript::new(b"bin-ntt/test/fold-other");
    let other_challenges = verifier.derive_folding_challenges(&mut other, &row_evaluation);
    let folded_commitment = verifier.fold_commitment(&commitment, &other_challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &challenges);
    assert_eq!(
        verifier
            .verify_opening(
                &commitment,
                &challenges,
                &point,
                OpeningMessage::Clear {
                    folded_commitment: &folded_commitment,
                    folded_witness: &folded_witness,
                    folded_row_value: &folded_row_value,
                },
            )
            .map(|_| ()),
        Err(VerificationError::Rejected)
    );
}

// =============================================================================================
// the base limb
// =============================================================================================

/// Every modulus can be the base — the limb the witness transform is kept in, the fold
/// accumulates over and the folded witness comes back from. One honest round each, over a second
/// limb that is not the base, and one with no second limb at all.
#[test]
fn any_modulus_can_be_the_base() {
    for base in Modulus::ALL {
        let params = Params::with_base(11, 3, base, vec![second(base)], Opening::Clear).unwrap();
        assert_eq!(params.primes(), vec![base.prime(), second(base).prime()]);
        let r = round(params);
        assert_eq!(r.verify(), Ok(()), "base {base:?}");
        let half = ((base.prime() - 1) / 2) as i16;
        let max = r
            .folded_witness
            .elements()
            .iter()
            .flat_map(|e| e.v.iter())
            .map(|x| x.unsigned_abs())
            .max()
            .unwrap();
        assert!(
            max as i16 <= half,
            "base {base:?}: the fold left the centered range"
        );
        assert!(
            max < 400,
            "base {base:?}: the fold is unexpectedly large: {max}"
        );

        let alone = Params::with_base(11, 3, base, vec![], Opening::Clear).unwrap();
        assert_eq!(round(alone).verify(), Ok(()), "base {base:?} alone");
    }
}

/// The centring check reads the base's own `(q-1)/2`, not 3889's.
#[test]
fn the_centred_range_is_the_base_modulus() {
    for base in [Q2917_Q_S, Q17497_FS_L] {
        let half = ((base.prime() - 1) / 2) as i16;
        let params = Params::with_base(11, 3, base, vec![second(base)], Opening::Clear).unwrap();
        let mut r = round(params);
        r.folded_witness.elements_mut()[0].v[0] = half + 1;
        assert_eq!(
            r.verify(),
            Err(VerificationError::Rejected),
            "base {base:?}"
        );
    }
}

/// A tampered fold is caught over a quadratic-slot and a large base as well as over 3889.
#[test]
fn a_corrupted_fold_is_rejected_whatever_the_base() {
    for base in Modulus::ALL {
        let params = Params::with_base(11, 3, base, vec![second(base)], Opening::Clear).unwrap();
        let mut r = round(params);
        r.folded_witness.elements_mut()[2].v[7] += 1;
        assert_eq!(
            r.verify(),
            Err(VerificationError::Rejected),
            "base {base:?}"
        );
    }
}

/// 64 columns, so that every base's fold-back period fires at least once inside the
/// accumulation (64 for 2917 down to 4 for the two primes above `2^14`).
#[test]
fn the_fold_back_period_fires_for_every_base() {
    for base in Modulus::ALL {
        let params = Params::with_base(14, 6, base, vec![second(base)], Opening::Clear).unwrap();
        assert_eq!(round(params).verify(), Ok(()), "base {base:?}");
    }
}

//! The moduli lists: the whole round over the base modulus alone and over every combination of
//! extra moduli the API offers, splitting and quadratic-slot alike.
use bin_ntt::{
    Modulus, Params, Prover, PublicParameters, Transcript, VerificationError, Verifier, Witness,
};

use Modulus::*;

const MATRIX_SEED: [u8; 32] = [41u8; 32];
const WITNESS_SEED: [u8; 32] = [43u8; 32];

const LISTS: [&[Modulus]; 10] = [
    &[],
    &[Q9721_FS_S],
    &[Q2917_Q_S],
    &[Q4861_Q_S, Q12637_Q_S],
    &[Q2917_Q_S, Q4861_Q_S, Q9721_FS_S, Q12637_Q_S],
    &[Q12637_Q_S, Q2917_Q_S],
    &[Q17497_FS_L],
    &[Q19441_FS_L],
    &[Q9721_FS_S, Q19441_FS_L],
    &[Q2917_Q_S, Q9721_FS_S, Q17497_FS_L, Q19441_FS_L],
];

/// One round over `extra_moduli`, returning the commitment's modulus list and the two verdicts.
fn round(
    extra_moduli: &[Modulus],
    corrupt: Option<usize>,
) -> (
    Vec<u16>,
    Result<(), VerificationError>,
    Result<(), VerificationError>,
) {
    let params = Params::new(10, 2, extra_moduli.to_vec(), false).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);

    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/limbs");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claimed_value = witness.mle_evaluate(&point);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let mut folded_witness = prover.fold(opening, &challenges);
    if let Some(i) = corrupt {
        folded_witness.elements_mut()[i].v[7] += 2;
    }
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &challenges);

    assert_eq!(commitment.rows(), 4);
    assert_eq!(commitment.columns(), params.columns());
    assert_eq!(folded_commitment.moduli(), commitment.moduli());
    for (k, &q) in commitment.moduli().iter().enumerate() {
        let half = (q as i32 - 1) / 2;
        for row in 0..4 {
            assert!(folded_commitment
                .element(row, k)
                .v
                .iter()
                .all(|&x| (x as i32).abs() <= half));
        }
    }

    (
        commitment.moduli().to_vec(),
        verifier.verify_evaluation(&point, &claimed_value, &row_evaluation),
        verifier.verify_folded_opening(
            &folded_commitment,
            &folded_witness,
            &point,
            &folded_row_value,
        ),
    )
}

#[test]
fn every_moduli_list_verifies() {
    for list in LISTS {
        let (moduli, evaluation, opening) = round(list, None);
        let want: Vec<u16> = core::iter::once(3889u16)
            .chain(list.iter().map(|m| m.prime()))
            .collect();
        assert_eq!(moduli, want, "moduli {list:?}");
        assert_eq!(evaluation, Ok(()), "moduli {list:?}");
        assert_eq!(opening, Ok(()), "moduli {list:?}");
    }
}

/// A corrupted opening must fail on every list — a single modulus catches it, and so does a
/// mixture of splitting and quadratic-slot ones.
#[test]
fn every_moduli_list_rejects_a_corrupted_opening() {
    for list in LISTS {
        let (_, evaluation, opening) = round(list, Some(1));
        assert_eq!(evaluation, Ok(()), "moduli {list:?}");
        assert_eq!(
            opening,
            Err(VerificationError::Rejected),
            "moduli {list:?}"
        );
    }
}

#[test]
fn the_modulus_order_is_the_one_the_list_gives() {
    assert_eq!(round(&[Q12637_Q_S, Q2917_Q_S], None).0, vec![3889, 12637, 2917]);
    assert_eq!(round(&[Q2917_Q_S, Q12637_Q_S], None).0, vec![3889, 2917, 12637]);
    assert_eq!(round(&[Q19441_FS_L, Q17497_FS_L], None).0, vec![3889, 19441, 17497]);
}

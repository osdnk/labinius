//! The surface: parameter validation, witness construction, the commitment's accessors,
//! determinism, and the workspace a prover reuses between two rounds.
use bin_ntt::{
    F162, Modulus, ParamError, Params, Prover, PublicParameters, Transcript, Verifier, Witness,
    WitnessError,
};

use Modulus::*;

const MATRIX_SEED: [u8; 32] = [7u8; 32];
const WITNESS_SEED: [u8; 32] = [11u8; 32];

fn small() -> Params {
    Params::new(9, 2, vec![Q9721_FS_S], false).unwrap()
}

/// One round, returning whether both checks passed.
fn round(prover: &mut Prover, verifier: &Verifier, witness: &Witness) -> bool {
    let (commitment, opening) = prover.commit(witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/api");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claimed_value = witness.mle_evaluate(&point);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded_witness = prover.fold(opening, &challenges);
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &challenges);
    verifier
        .verify_evaluation(&point, &claimed_value, &row_evaluation)
        .is_ok()
        && verifier
            .verify_folded_opening(
                &folded_commitment,
                &folded_witness,
                &point,
                &folded_row_value,
            )
            .is_ok()
}

#[test]
fn params_shape() {
    let basic = Params::basic();
    assert_eq!(basic.witness_log_len, 18);
    assert_eq!(basic.base, Q3889_FS_S);
    assert_eq!(basic.column_log_len, 7);
    assert_eq!(basic.extra_moduli, vec![Q9721_FS_S]);
    assert_eq!(basic.witness_len(), 1 << 18);
    assert_eq!(basic.columns(), 128);
}

#[test]
fn params_rejects_more_columns_than_elements() {
    assert_eq!(
        Params::new(4, 6, vec![], false),
        Err(ParamError::ColumnsExceedWitness)
    );
}

#[test]
fn params_rejects_duplicate_moduli() {
    assert_eq!(
        Params::new(12, 2, vec![Q4861_Q_S, Q9721_FS_S, Q4861_Q_S], false),
        Err(ParamError::DuplicateModulus(Q4861_Q_S))
    );
    assert!(Params::new(12, 2, vec![Q4861_Q_S, Q9721_FS_S], false).is_ok());
}

#[test]
fn params_default_to_the_base_modulus_3889() {
    assert_eq!(Params::basic().primes(), vec![3889, 9721]);
    assert_eq!(Params::new(12, 2, vec![Q2917_Q_S], false).unwrap().base, Q3889_FS_S);
}

/// Any modulus can be the base, and `primes()` still lists it first.
#[test]
fn params_take_any_base() {
    for base in Modulus::ALL {
        let extra = if base == Q9721_FS_S { Q3889_FS_S } else { Q9721_FS_S };
        let params = Params::with_base(12, 2, base, vec![extra], false).unwrap();
        assert_eq!(params.base, base);
        assert_eq!(params.primes(), vec![base.prime(), extra.prime()]);
    }
}

#[test]
fn params_reject_the_base_among_the_extra_moduli() {
    assert_eq!(
        Params::new(12, 2, vec![Q9721_FS_S, Q3889_FS_S], false),
        Err(ParamError::BaseIsAlsoExtra(Q3889_FS_S))
    );
    assert_eq!(
        Params::with_base(12, 2, Q2917_Q_S, vec![Q9721_FS_S, Q2917_Q_S], false),
        Err(ParamError::BaseIsAlsoExtra(Q2917_Q_S))
    );
    assert!(Params::with_base(12, 2, Q2917_Q_S, vec![Q9721_FS_S, Q3889_FS_S], false).is_ok());
}

#[test]
fn params_rejects_a_column_below_one_batch() {
    assert_eq!(Params::new(9, 3, vec![], false), Err(ParamError::ColumnTooShort));
    assert_eq!(Params::new(9, 0, vec![], false), Err(ParamError::TooFewColumns));
    assert!(Params::new(9, 2, vec![], false).is_ok());
}

#[test]
fn witness_length_is_checked() {
    let params = small();
    assert_eq!(
        Witness::from_elements(&params, vec![F162::ZERO; params.witness_len() - 1]),
        Err(WitnessError::WrongLength)
    );
    let elements = vec![F162::ONE; params.witness_len()];
    let witness = Witness::from_elements(&params, elements.clone()).unwrap();
    assert_eq!(witness.elements(), &elements[..]);
}

#[test]
fn random_witness_is_a_field_element_and_deterministic() {
    let params = small();
    let witness = Witness::random(&params, WITNESS_SEED);
    assert_eq!(witness.elements().len(), params.witness_len());
    for x in witness.elements() {
        assert_eq!(x.0[2] >> 34, 0, "the top 30 bits of limb 2 must be zero");
    }
    assert_eq!(witness, Witness::random(&params, WITNESS_SEED));
    assert_ne!(witness, Witness::random(&params, [12u8; 32]));
}

#[test]
fn commitment_shape_and_accessors() {
    let params = small();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    assert_eq!(pp.params(), &params);
    let mut prover = Prover::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, _) = prover.commit(&witness);
    assert_eq!(commitment.rows(), 4);
    assert_eq!(commitment.columns(), params.columns());
    assert_eq!(commitment.moduli(), &[3889, 9721]);
    for row in 0..commitment.rows() {
        for column in 0..commitment.columns() {
            for (k, &q) in commitment.moduli().iter().enumerate() {
                let e = commitment.element(row, column, k);
                assert_eq!(e.v.len(), 162);
                assert!(e.v.iter().all(|&x| (x as i32).abs() <= (q as i32 - 1) / 2));
            }
        }
    }
}

#[test]
fn the_same_seeds_give_the_same_commitment() {
    let params = small();
    let witness = Witness::random(&params, WITNESS_SEED);
    let commit = || {
        let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
        Prover::new(&pp).commit(&witness).0
    };
    assert_eq!(commit(), commit());

    let other = PublicParameters::from_seed(params.clone(), [8u8; 32]);
    assert_ne!(commit(), Prover::new(&other).commit(&witness).0);
}

#[test]
fn one_prover_runs_two_rounds() {
    let params = small();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let first = Witness::random(&params, WITNESS_SEED);
    let second = Witness::random(&params, [13u8; 32]);
    assert!(round(&mut prover, &verifier, &first));
    assert!(round(&mut prover, &verifier, &second));
    assert!(round(&mut prover, &verifier, &first));
}

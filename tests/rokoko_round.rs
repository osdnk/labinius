//! The recursive opening over rokoko, end to end on the real pipeline.
#![cfg(feature = "rokoko")]
use bin_ntt::{Backend, Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [17u8; 32];
const WITNESS_SEED: [u8; 32] = [29u8; 32];

fn small() -> Params {
    Params::new(9, 2, vec![Modulus::Q9721_FS_S], true)
        .unwrap()
        .with_backend(Backend::Rokoko)
}

#[test]
fn an_honest_round_verifies() {
    let params = small();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/rokoko");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claim = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let left = prover.commit_left_expansion(&row);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &left);
    let mut check = transcript.clone();
    let proof = prover
        .prove_opening_rokoko(
            &mut transcript,
            opening,
            &challenges,
            &point,
            &left,
            &row,
            &claim,
            &commitment,
        )
        .expect("the honest round is within its gadgets");
    verifier
        .verify_opening_rokoko(
            &mut check,
            &commitment,
            &left,
            &point,
            &claim,
            &challenges,
            &proof,
        )
        .expect("the honest opening is accepted");
    assert!(proof.wire_bytes() > 0);

    let wrong = bin_ntt::F162([1, 0, 0]) + claim;
    let mut other = transcript.clone();
    assert!(verifier
        .verify_opening_rokoko(
            &mut other,
            &commitment,
            &left,
            &point,
            &wrong,
            &challenges,
            &proof,
        )
        .is_err());
}

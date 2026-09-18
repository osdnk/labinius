//! The commitment at size `s` on the two kinds of witness it takes: `2^18` elements of `F162`
//! opened at a point of `F162^18`, and `2^18` words of binius64's `B128` opened at a point of
//! `B128^18`, which the switch turns into an `F162` opening. The points are fixed up front; a
//! protocol draws them from its transcript after the commitment.
//!
//! `cargo run --release -p labinius --example two_fields`
use labinius::switch::{self, SwitchProof};
use labinius::{
    Commitment, EvaluationPoint, FoldedWitness, FoldingChallenges, Opening, OpeningMessage, Params,
    Prover, PublicParameters, RowEvaluation, Transcript, VerificationError, Verifier, Witness,
    B128, F162, SUITES,
};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];

fn main() {
    let suite = SUITES.iter().find(|s| s.name == "sizes").expect("size s");
    let params = Params::sized(suite, Opening::Clear);
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let l = params.witness_log_len as usize;
    let mut rng = Transcript::new(b"example");

    // Native: a witness over F162, opened at a point of F162^18.
    let witness = Witness::random(&params, [0xC7; 32]);
    let point = EvaluationPoint::msb_first(&params, &rng.sample_f162(b"point", l));
    let (value, proof) = native_prove(&params, &pp, &witness, &point);
    println!(
        "native F162: {:?}",
        native_verify(&params, &pp, &point, value, &proof)
    );

    // GHASH: a trace of B128 words, evaluated at a point of B128^18.
    let trace = rng.sample_b128(b"trace", 1 << l);
    let r = rng.sample_b128(b"r", l);
    let claim = switch::mle(&trace, &r);
    let proof = ghash_prove(&params, &pp, &trace, &r, claim);
    println!(
        "GHASH B128:  {:?}",
        ghash_verify(&params, &pp, &r, claim, &proof)
    );
}

struct NativeProof {
    commitment: Commitment,
    row: RowEvaluation,
    folded: FoldedWitness,
}

fn native_prove(
    params: &Params,
    pp: &PublicParameters,
    witness: &Witness,
    point: &EvaluationPoint,
) -> (F162, NativeProof) {
    let mut prover = Prover::new(pp);
    let mut transcript = Transcript::new(b"example/native");

    let (commitment, opening) = prover.commit(witness);
    let value = witness.mle_evaluate(point);
    let row = witness.row_evaluate(point);
    transcript.absorb_bytes(&commitment.to_bytes());
    let challenges = FoldingChallenges::derive(params, &mut transcript, &row);
    let folded = prover.fold(opening, &challenges);
    (
        value,
        NativeProof {
            commitment,
            row,
            folded,
        },
    )
}

fn native_verify(
    params: &Params,
    pp: &PublicParameters,
    point: &EvaluationPoint,
    value: F162,
    proof: &NativeProof,
) -> Result<(), VerificationError> {
    let verifier = Verifier::new(pp);
    let mut transcript = Transcript::new(b"example/native");

    transcript.absorb_bytes(&proof.commitment.to_bytes());
    let challenges = FoldingChallenges::derive(params, &mut transcript, &proof.row);
    verifier.verify_evaluation(point, &value, &proof.row)?;
    verifier.verify_opening(
        &proof.commitment,
        &challenges,
        point,
        OpeningMessage::Clear {
            folded_commitment: &verifier.fold_commitment(&proof.commitment, &challenges),
            folded_witness: &proof.folded,
            folded_row_value: &verifier.fold_row_evaluation(&proof.row, &challenges),
        },
    )?;
    Ok(())
}

struct GhashProof {
    commitment: Commitment,
    switch: SwitchProof,
    row: RowEvaluation,
    folded: FoldedWitness,
}

fn ghash_prove(
    params: &Params,
    pp: &PublicParameters,
    trace: &[B128],
    r: &[B128],
    claim: B128,
) -> GhashProof {
    let mut prover = Prover::new(pp);
    let mut transcript = Transcript::new(b"example/ghash");

    let witness = Witness::lifted(params, trace).expect("the trace is the witness length");
    let (commitment, opening) = prover.commit(&witness);
    transcript.absorb_bytes(&commitment.to_bytes());
    let (switch, point) = switch::prove(params, trace, r, claim, &mut transcript);
    let row = witness.row_evaluate(&point);
    let challenges = FoldingChallenges::derive(params, &mut transcript, &row);
    let folded = prover.fold(opening, &challenges);
    GhashProof {
        commitment,
        switch,
        row,
        folded,
    }
}

fn ghash_verify(
    params: &Params,
    pp: &PublicParameters,
    r: &[B128],
    claim: B128,
    proof: &GhashProof,
) -> Result<(), Box<dyn std::error::Error>> {
    let verifier = Verifier::new(pp);
    let mut transcript = Transcript::new(b"example/ghash");

    transcript.absorb_bytes(&proof.commitment.to_bytes());
    let (point, value) = switch::verify(params, r, claim, &proof.switch, &mut transcript)?;
    let challenges = FoldingChallenges::derive(params, &mut transcript, &proof.row);
    verifier.verify_evaluation(&point, &value, &proof.row)?;
    verifier.verify_opening(
        &proof.commitment,
        &challenges,
        &point,
        OpeningMessage::Clear {
            folded_commitment: &verifier.fold_commitment(&proof.commitment, &challenges),
            folded_witness: &proof.folded,
            folded_row_value: &verifier.fold_row_evaluation(&proof.row, &challenges),
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The other two opening modes, as the native flow above changes for them. Either takes the
// GHASH flow the same way: `switch::prove` still hands back the point to open at.
// ---------------------------------------------------------------------------------------------
//
// Bit-dropped: the commitment on the wire keeps only the top bits of each coefficient, so the
// verifier cannot fold it and checks the folded witness against the dropped commitment instead.
//
//     let params = Params::sized(suite, Opening::BitDropped { bits: suite.dropped_bits });
//     let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
//
//     // prover: exactly as `native_prove`
//     let (commitment, opening) = prover.commit(witness);
//     let value = witness.mle_evaluate(point);
//     let row = witness.row_evaluate(point);
//     transcript.absorb_bytes(&commitment.to_bytes());
//     let challenges = FoldingChallenges::derive(&params, &mut transcript, &row);
//     let folded = prover.fold(opening, &challenges);
//
//     // verifier: no `fold_commitment`, and the message names the mode
//     transcript.absorb_bytes(&commitment.to_bytes());
//     let challenges = FoldingChallenges::derive(&params, &mut transcript, &row);
//     verifier.verify_evaluation(point, &value, &row)?;
//     verifier.verify_opening(
//         &commitment,
//         &challenges,
//         point,
//         OpeningMessage::BitDropped {
//             folded_witness: &folded,
//             folded_row_value: &verifier.fold_row_evaluation(&row, &challenges),
//         },
//     )?;
//
// Recursive (LaBRADOR): the fold is not sent. The prover commits to the row evaluation as the
// left expansion `T_u`, the folding challenges come from that commitment rather than from the
// row, and `prove_opening` runs the fold and proves the whole relation — the fold, its norm, and
// the row evaluation — in one LaBRADOR proof, so the verifier never sees `row` or `folded` and
// `verify_evaluation` is subsumed. `PublicParameters::from_seed` builds the LaBRADOR setup too,
// which takes longer.
//
//     let params = Params::sized(suite, Opening::Recursive);
//     let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
//
//     // prover
//     let (commitment, opening) = prover.commit(witness);
//     let value = witness.mle_evaluate(point);
//     let row = witness.row_evaluate(point);
//     let left = prover.commit_left_expansion(&row);
//     transcript.absorb_bytes(&commitment.to_bytes());
//     let challenges = FoldingChallenges::derive(&params, &mut transcript, &left);
//     let proof = prover.prove_opening(
//         &mut transcript,
//         opening,
//         &challenges,
//         point,
//         &left,
//         &row,
//         &value,
//         &commitment,
//     )?;
//     // on the wire: commitment, left, value, proof
//
//     // verifier
//     transcript.absorb_bytes(&commitment.to_bytes());
//     let challenges = FoldingChallenges::derive(&params, &mut transcript, &left);
//     verifier.verify_opening(
//         &commitment,
//         &challenges,
//         point,
//         OpeningMessage::Recursive {
//             transcript: &mut transcript,
//             left: &left,
//             claimed_value: &value,
//             proof: &proof,
//         },
//     )?;

//! The commitment at size `s` on the two kinds of witness it takes: `2^18` elements of `F162`
//! opened at a point of `F162^18`, in each of the three opening modes, and `2^18` words of
//! binius64's `B128` opened at a point of `B128^18`, which the switch turns into an `F162`
//! opening. The points are inputs, as they would come from the surrounding protocol.
//!
//! `cargo run --release -p labinius --example two_fields`
use labinius::switch::{self, SwitchProof};
use labinius::{
    Commitment, EvaluationPoint, FoldedWitness, FoldingChallenges, LeftExpansionCommitment,
    Opening, OpeningMessage, OpeningProof, Params, Prover, PublicParameters, RowEvaluation,
    Transcript, VerificationError, Verifier, Witness, B128, F162, SUITES,
};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];

fn main() {
    let suite = SUITES.iter().find(|s| s.name == "sizes").expect("size s");
    let mut rng = Transcript::new(b"example");
    let l = suite.witness_log_len as usize;

    // Native: a witness over F162, opened at a point of F162^18, in each opening mode.
    let coordinates = rng.sample_f162(b"point", l);
    let modes = [
        Opening::Clear,
        Opening::BitDropped {
            bits: suite.dropped_bits,
        },
        Opening::Recursive,
    ];
    for mode in modes {
        let pp = PublicParameters::from_seed(Params::sized(suite, mode), MATRIX_SEED);
        let witness = Witness::random(pp.params(), [0xC7; 32]);
        let point = EvaluationPoint::msb_first(pp.params(), &coordinates);
        let (value, proof) = native_prove(&pp, &witness, &point);
        println!(
            "native F162, {mode:?}: {:?}",
            native_verify(&pp, &point, value, &proof)
        );
    }

    // GHASH: a trace of B128 words, evaluated at a point of B128^18.
    let pp = PublicParameters::from_seed(Params::sized(suite, Opening::Clear), MATRIX_SEED);
    let trace = rng.sample_b128(b"trace", 1 << l);
    let r = rng.sample_b128(b"r", l);
    let claim = switch::mle(&trace, &r);
    let proof = ghash_prove(&pp, &trace, &r, claim);
    println!(
        "GHASH B128, Clear: {:?}",
        ghash_verify(&pp, &r, claim, &proof)
    );
}

struct NativeProof {
    commitment: Commitment,
    opening: NativeOpening,
}

enum NativeOpening {
    /// Clear and bit-dropped: the row evaluation and the folded witness.
    Folded {
        row: RowEvaluation,
        folded: FoldedWitness,
    },
    /// Recursive: the left expansion and one LaBRADOR proof of the fold and the row evaluation.
    Recursive {
        left: LeftExpansionCommitment,
        proof: OpeningProof,
    },
}

/// The value of the witness at `point`, and the proof of it.
fn native_prove(
    pp: &PublicParameters,
    witness: &Witness,
    point: &EvaluationPoint,
) -> (F162, NativeProof) {
    let params = pp.params();
    let mut prover = Prover::new(pp);
    let mut transcript = Transcript::new(b"example/native");

    let (commitment, opening) = prover.commit(witness);
    let value = witness.mle_evaluate(point);
    let row = witness.row_evaluate(point);
    transcript.absorb_bytes(&commitment.to_bytes());

    let opening = match params.opening {
        Opening::Clear | Opening::BitDropped { .. } => {
            let challenges = FoldingChallenges::derive(params, &mut transcript, &row);
            let folded = prover.fold(opening, &challenges);
            NativeOpening::Folded { row, folded }
        }
        Opening::Recursive => {
            let left = prover.commit_left_expansion(&row);
            let challenges = FoldingChallenges::derive(params, &mut transcript, &left);
            let proof = prover
                .prove_opening(
                    &mut transcript,
                    opening,
                    &challenges,
                    point,
                    &left,
                    &row,
                    &value,
                    &commitment,
                )
                .expect("the honest fold is within its cap");
            NativeOpening::Recursive { left, proof }
        }
    };
    (
        value,
        NativeProof {
            commitment,
            opening,
        },
    )
}

fn native_verify(
    pp: &PublicParameters,
    point: &EvaluationPoint,
    value: F162,
    proof: &NativeProof,
) -> Result<(), VerificationError> {
    let params = pp.params();
    let verifier = Verifier::new(pp);
    let mut transcript = Transcript::new(b"example/native");
    transcript.absorb_bytes(&proof.commitment.to_bytes());

    match &proof.opening {
        NativeOpening::Folded { row, folded } => {
            let challenges = FoldingChallenges::derive(params, &mut transcript, row);
            verifier.verify_evaluation(point, &value, row)?;
            let folded_row_value = &verifier.fold_row_evaluation(row, &challenges);
            let message = match params.opening {
                Opening::Clear => OpeningMessage::Clear {
                    folded_commitment: &verifier.fold_commitment(&proof.commitment, &challenges),
                    folded_witness: folded,
                    folded_row_value,
                },
                _ => OpeningMessage::BitDropped {
                    folded_witness: folded,
                    folded_row_value,
                },
            };
            verifier.verify_opening(&proof.commitment, &challenges, point, message)?;
        }
        NativeOpening::Recursive {
            left,
            proof: opening,
        } => {
            let challenges = FoldingChallenges::derive(params, &mut transcript, left);
            verifier.verify_opening(
                &proof.commitment,
                &challenges,
                point,
                OpeningMessage::Recursive {
                    transcript: &mut transcript,
                    left,
                    claimed_value: &value,
                    proof: opening,
                },
            )?;
        }
    }
    Ok(())
}

struct GhashProof {
    commitment: Commitment,
    switch: SwitchProof,
    row: RowEvaluation,
    folded: FoldedWitness,
}

fn ghash_prove(pp: &PublicParameters, trace: &[B128], r: &[B128], claim: B128) -> GhashProof {
    let params = pp.params();
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
    pp: &PublicParameters,
    r: &[B128],
    claim: B128,
    proof: &GhashProof,
) -> Result<(), Box<dyn std::error::Error>> {
    let params = pp.params();
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

//! The wire forms: exact round trips for the bit-packed uniform objects and for the entropy
//! coder, on honest rounds at several shapes and bases and on adversarial folds, and the size
//! and wall clock the README quotes (`--nocapture`, under `taskset -c 3`).
use bin_ntt::types::{Representation, RingElement};
use bin_ntt::wire::{self, WireError};
use bin_ntt::{Modulus, Params, Prover, PublicParameters, Transcript, Verifier, Witness};
use std::time::Instant;

use Modulus::*;

const MATRIX_SEED: [u8; 32] = [21u8; 32];
const WITNESS_SEED: [u8; 32] = [23u8; 32];

struct Round {
    params: Params,
    commitment: bin_ntt::Commitment,
    row_evaluation: bin_ntt::RowEvaluation,
    folded_witness: bin_ntt::FoldedWitness,
}

fn round(params: Params) -> Round {
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/wire");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded_witness = prover.fold(opening, &challenges);
    Round {
        params,
        commitment,
        row_evaluation,
        folded_witness,
    }
}

/// A modulus that is not `base`, to be the second limb.
fn second(base: Modulus) -> Modulus {
    if base == Q9721_FS_S {
        Q3889_FS_S
    } else {
        Q9721_FS_S
    }
}

fn fold_of(values: Vec<Vec<i16>>) -> bin_ntt::FoldedWitness {
    bin_ntt::FoldedWitness::of(
        values
            .into_iter()
            .map(|v| {
                let mut e = RingElement::zero(Representation::Coefficients);
                e.v.copy_from_slice(&v);
                e
            })
            .collect(),
    )
}

// =============================================================================================
// bit-packed uniform objects
// =============================================================================================

/// 162 bits per element with nothing between them, and back.
#[test]
fn the_row_evaluation_packs_at_162_bits() {
    for count in [0usize, 1, 2, 7, 256, 1024] {
        let values = bin_ntt::f162::random_elems(count, 0xF162 + count as u64);
        let bytes = wire::pack_f162(&values);
        assert_eq!(bytes.len(), (count * 162).div_ceil(8), "{count} elements");
        assert_eq!(wire::unpack_f162(&bytes, count).unwrap(), values);
    }
    let bytes = wire::pack_f162(&bin_ntt::f162::random_elems(256, 7));
    assert_eq!(bytes.len(), 5184);
    assert_eq!(
        wire::unpack_f162(&bytes[..5183], 256),
        Err(WireError::Malformed)
    );
}

/// An honest commitment at `ceil(log2 q)` bits per slot per limb, and back to the same matrix.
#[test]
fn the_commitment_packs_at_the_residue_width() {
    for base in Modulus::ALL {
        let params = Params::with_base(11, 3, base, vec![second(base)], false).unwrap();
        let r = round(params.clone());
        let bytes = wire::pack_commitment(&r.commitment);
        let width: u32 = params.primes().iter().map(|&q| wire::residue_bits(q)).sum();
        assert_eq!(
            bytes.len(),
            (4 * params.columns() * 162 * width as usize).div_ceil(8)
        );
        assert_eq!(bytes.len(), r.commitment.wire_bytes());
        let back = wire::unpack_commitment(&params, &bytes).unwrap();
        assert_eq!(back, r.commitment, "base {base:?}");
    }
}

/// Every slot at its two extremes, which is where a centred residue meets its packed range.
#[test]
fn the_commitment_packer_takes_the_extremes() {
    let params = Params::with_base(11, 3, Q19441_FS_L, vec![Q2917_Q_S], false).unwrap();
    let primes = params.primes();
    for pick in [0usize, 1, 2] {
        let data = (0..4 * params.columns())
            .map(|i| bin_ntt::api::PowerOfThreeRingElementWithLimbs {
                limbs: primes
                    .iter()
                    .map(|&q| {
                        let half = ((q - 1) / 2) as i16;
                        let value = [0, half, -half][(pick + i) % 3];
                        bin_ntt::api::PowerOfThreeRingElement { v: [value; 162] }
                    })
                    .collect(),
            })
            .collect();
        let commitment = bin_ntt::Commitment::of(
            primes.clone(),
            params.columns(),
            bin_ntt::CommitmentValue::Matrix(bin_ntt::api::VerticallyAlignedMatrix::new(
                4,
                params.columns(),
                data,
            )),
        );
        let bytes = wire::pack_commitment(&commitment);
        assert_eq!(
            wire::unpack_commitment(&params, &bytes).unwrap(),
            commitment
        );
    }
}

// =============================================================================================
// the entropy coder
// =============================================================================================

/// Honest folds at four shapes and every base modulus, byte for byte.
#[test]
fn an_honest_fold_round_trips() {
    for base in Modulus::ALL {
        for (witness_log_len, column_log_len) in [(11u32, 3u32), (12, 2), (14, 6)] {
            let params = Params::with_base(
                witness_log_len,
                column_log_len,
                base,
                vec![second(base)],
                false,
            )
            .unwrap();
            let r = round(params.clone());
            let bytes = wire::encode(&r.folded_witness, base.prime());
            let back = wire::decode(&bytes).unwrap();
            assert_eq!(
                back.elements(),
                r.folded_witness.elements(),
                "base {base:?}"
            );
            assert!(
                bytes.len() < r.folded_witness.len() * 648 * 2,
                "base {base:?}: the coder did not beat two bytes a coefficient"
            );
        }
    }
}

/// The degenerate and the adversarial messages: nothing, one element, a constant fold, the
/// coefficient bound of every base in both signs, a fold that alternates between the bounds, and
/// one uniform over the whole centred range — which is what overflows the 4096-slot table and
/// puts the escape symbol to work.
#[test]
fn an_adversarial_fold_round_trips() {
    let empty = bin_ntt::FoldedWitness::of(Vec::new());
    let bytes = wire::encode(&empty, 3889);
    assert_eq!(bytes.len(), 16);
    assert!(wire::decode(&bytes).unwrap().is_empty());

    for base in Modulus::ALL {
        let q = base.prime();
        let half = ((q - 1) / 2) as i16;
        let mut cases: Vec<Vec<Vec<i16>>> = vec![
            vec![vec![0i16; 648]],
            vec![vec![half; 648]],
            vec![vec![-half; 648]],
            vec![(0..648)
                .map(|i| if i % 2 == 0 { half } else { -half })
                .collect()],
            vec![
                (0..648)
                    .map(|i| (i as i16 % (2 * half + 1)) - half)
                    .collect(),
                vec![half; 648],
                vec![-half; 648],
            ],
        ];
        // Uniform over the whole centred range: 3889 symbols at 3889, 19441 at 19441.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        cases.push(
            (0..16)
                .map(|_| {
                    (0..648)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            (state % (2 * half as u64 + 1)) as i16 - half
                        })
                        .collect()
                })
                .collect(),
        );
        for (case, values) in cases.into_iter().enumerate() {
            let folded = fold_of(values);
            let bytes = wire::encode(&folded, q);
            let back = wire::decode(&bytes).unwrap();
            assert_eq!(
                back.elements(),
                folded.elements(),
                "base {base:?}, case {case}"
            );
        }
    }
}

/// The whole `i16` range, both extremes included: the coder is a bijection on any message, not
/// only on a centred one.
#[test]
fn the_coder_takes_any_i16() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let folded = fold_of(
        (0..8)
            .map(|_| {
                (0..648)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (state >> 32) as i16
                    })
                    .collect()
            })
            .collect(),
    );
    let mut extremes = folded.elements().to_vec();
    extremes[0].v[0] = i16::MIN;
    extremes[0].v[1] = i16::MAX;
    let folded = bin_ntt::FoldedWitness::of(extremes);
    let bytes = wire::encode(&folded, 3889);
    assert_eq!(wire::decode(&bytes).unwrap().elements(), folded.elements());
}

/// A truncated or corrupted stream is refused, never mis-decoded into a shorter object.
#[test]
fn a_broken_stream_is_refused() {
    let r = round(Params::with_base(11, 3, Q3889_FS_S, vec![Q9721_FS_S], false).unwrap());
    let bytes = wire::encode(&r.folded_witness, 3889);
    for cut in [0usize, 1, 15, 16, 20, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            wire::decode(&bytes[..cut]).is_err(),
            "a prefix of {cut} bytes decoded"
        );
    }
    let mut long = bytes.clone();
    long.extend_from_slice(&[0, 0]);
    assert_eq!(wire::decode(&long), Err(WireError::Malformed));
    let mut wrong = bytes.clone();
    wrong[0] = 2;
    assert_eq!(wire::decode(&wrong), Err(WireError::Malformed));
}

/// The verifier accepts what came off the wire, and rejects it once a byte of the stream moves.
#[test]
fn the_verifier_takes_the_decoded_objects() {
    let params = Params::basic();
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let witness = Witness::random(&params, WITNESS_SEED);
    let (commitment, opening) = prover.commit(&witness);
    let mut transcript = Transcript::new(b"bin-ntt/test/wire");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let claimed_value = witness.mle_evaluate(&point);
    let row_evaluation = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row_evaluation);
    let folded_witness = prover.fold(opening, &challenges);

    let commitment_wire = wire::pack_commitment(&commitment);
    let row_wire = wire::pack_row_evaluation(&row_evaluation);
    let fold_wire = wire::encode(&folded_witness, params.base.prime());

    let commitment = wire::unpack_commitment(&params, &commitment_wire).unwrap();
    let row_evaluation = wire::unpack_row_evaluation(&row_wire, params.columns()).unwrap();
    let folded_witness = wire::decode(&fold_wire).unwrap();
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row_evaluation, &challenges);
    assert_eq!(
        verifier.verify_evaluation(&point, &claimed_value, &row_evaluation),
        Ok(())
    );
    assert_eq!(
        verifier.verify_folded_opening(
            &folded_commitment,
            &folded_witness,
            &point,
            &folded_row_value
        ),
        Ok(())
    );

    let mut tampered = fold_wire.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    if let Ok(other) = wire::decode(&tampered) {
        assert_ne!(other.elements(), folded_witness.elements());
    }
}

// =============================================================================================
// the numbers
// =============================================================================================

fn report(name: &str, params: Params) {
    let r = round(params.clone());
    let commitment_wire = wire::pack_commitment(&r.commitment);
    let row_wire = wire::pack_row_evaluation(&r.row_evaluation);
    let mut encode_ms = f64::INFINITY;
    let mut fold_wire = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        fold_wire = wire::encode(&r.folded_witness, r.params.base.prime());
        encode_ms = encode_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut decode_ms = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let back = wire::decode(&fold_wire).unwrap();
        decode_ms = decode_ms.min(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(back.elements(), r.folded_witness.elements());
    }
    let coefficients = r.folded_witness.len() * 648;
    let entropy = wire::entropy_bytes(&r.folded_witness);
    let square: f64 = r
        .folded_witness
        .elements()
        .iter()
        .flat_map(|e| e.v)
        .map(|x| (x as f64) * (x as f64))
        .sum();
    let peak = r
        .folded_witness
        .elements()
        .iter()
        .flat_map(|e| e.v)
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap();
    let plain = coefficients * 2;
    println!(
        "\n{name}: base {}, {} columns, {} coefficients of fold",
        r.params.base.prime(),
        r.params.columns(),
        coefficients
    );
    println!(
        "  commitment       {:>9.1} KB  ({:>9.1} KB at i16)",
        commitment_wire.len() as f64 / 1024.0,
        (4 * r.params.columns() * r.params.primes().len() * 162 * 2) as f64 / 1024.0
    );
    println!(
        "  row evaluation   {:>9.1} KB  ({:>9.1} KB at 24 bytes)",
        row_wire.len() as f64 / 1024.0,
        (r.row_evaluation.values().len() * 24) as f64 / 1024.0
    );
    println!(
        "  folded witness   {:>9.1} KB  ({:>9.1} KB at i16), entropy {:.1} KB, {:+.2} %",
        fold_wire.len() as f64 / 1024.0,
        plain as f64 / 1024.0,
        entropy / 1024.0,
        100.0 * (fold_wire.len() as f64 - entropy) / entropy
    );
    println!(
        "  fold sigma {:.1}, peak {peak}",
        (square / coefficients as f64).sqrt()
    );
    println!(
        "  {:.3} bits per coefficient against {:.3} of entropy; encode {:.2} ms, decode {:.2} ms",
        8.0 * fold_wire.len() as f64 / coefficients as f64,
        8.0 * entropy / coefficients as f64,
        encode_ms,
        decode_ms
    );
}

/// The wire sizes and the coder's wall clock at the basic shape and at a large base.
#[test]
fn the_wire_quantified() {
    report("basic", Params::basic());
    report(
        "large base 19441",
        Params::with_base(18, 8, Q19441_FS_L, vec![Q17497_FS_L], false).unwrap(),
    );
}

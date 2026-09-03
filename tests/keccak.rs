//! The keccak pipeline: the coordinate mapping the switch and the commitment agree on, and the
//! real 482-permutation instance end to end in both PCS modes, honest and tampered.
use bin_ntt::fields::crossfield::eval_pi1;
use bin_ntt::fields::scalar::B128;
use bin_ntt::keccak::{Circuit, Error, Hash, Session};
use bin_ntt::rng::Rng;
use bin_ntt::{EvaluationPoint, Modulus, Params, Witness, F162};

const MATRIX_SEED: [u8; 32] = [0x5A; 32];
const HASH: Hash = Hash::Keccak256;
const MESSAGE_LEN: usize = HASH.message_len();

fn message() -> Vec<u8> {
    (0..MESSAGE_LEN)
        .map(|i| (i as u32).wrapping_mul(2654435761) as u8)
        .collect()
}

fn f162(rng: &mut Rng) -> F162 {
    F162([
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64() & ((1 << 34) - 1),
    ])
}

/// The switch hands back `r''` most significant first over the flat trace index, and the
/// commitment splits the same index into a column half and a row half. Both readings of the same
/// multilinear must agree.
#[test]
fn the_switch_point_is_the_evaluation_point() {
    for params in [
        Params::new(11, 3, vec![Modulus::Q9721_FS_S], false).unwrap(),
        Params::basic(),
    ] {
        let mut rng = Rng::new(0x5EED_1234);
        let trace: Vec<B128> = (0..params.witness_len())
            .map(|_| B128(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)))
            .collect();
        let witness =
            Witness::from_elements(&params, trace.iter().map(|&x| F162::from_b128(x)).collect())
                .unwrap();
        let r_pp: Vec<F162> = (0..params.witness_log_len)
            .map(|_| f162(&mut rng))
            .collect();
        let point = EvaluationPoint::msb_first(&params, &r_pp);
        assert_eq!(witness.mle_evaluate(&point), eval_pi1(&trace, &r_pp));
    }
}

#[test]
fn the_honest_proof_verifies_in_both_modes() {
    let circuit = Circuit::new(HASH, MESSAGE_LEN);
    let witness = circuit.witness(&message());
    for recursion in [false, true] {
        let mut session = Session::new(circuit.constraint_system().clone(), recursion, MATRIX_SEED);
        let (proof, _, sizes) = session.prove(&witness, None);
        assert!(sizes.total() > 0);
        session
            .verify(witness.inout(), &proof)
            .unwrap_or_else(|e| panic!("recursion {recursion}: {e}"));
    }
}

/// One bit of the packed trace flipped after the commitment: the switch's partial evaluations no
/// longer open binius64's claim, so the proof dies before the opening is even read.
#[test]
fn a_flipped_trace_bit_is_rejected() {
    let circuit = Circuit::new(HASH, MESSAGE_LEN);
    let witness = circuit.witness(&message());
    let mut session = Session::new(circuit.constraint_system().clone(), false, MATRIX_SEED);
    let (proof, _, _) = session.prove(&witness, Some(12345));
    match session.verify(witness.inout(), &proof) {
        Err(Error::Switch(_)) => {}
        other => panic!("expected the switch to reject, got {other:?}"),
    }
}

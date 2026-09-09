use labinius::{Opening, OpeningMessage};
use labinius::SUITES;
use labinius::bd::{self, Dropped};
use labinius::params::N;
use labinius::wire;
use labinius::Modulus;
use labinius::{Params, Prover, PublicParameters, Transcript, Verifier, Witness};

const MATRIX_SEED: [u8; 32] = [0x31; 32];
const WITNESS_SEED: [u8; 32] = [0x77; 32];

fn small(base: Modulus, extra: Vec<Modulus>, dropped_bits: u32) -> Params {
    Params::with_base(9, 2, base, extra, Opening::Clear)
        .unwrap()
        .dropping(dropped_bits)
}

fn plain_matrix_and_dropped(params: &Params) -> (Vec<Vec<i16>>, Dropped) {
    let plain = params.clone().dropping(0);
    let pp = PublicParameters::from_seed(plain.clone(), MATRIX_SEED);
    let witness = Witness::random(&plain, WITNESS_SEED);
    let (commitment, _) = Prover::new(&pp).commit(&witness);
    let primes = params.primes();
    let residues = bd::column_coefficients(commitment.matrix(), &primes);
    let dropped = bd::drop_bits(commitment.matrix(), &primes, params.dropped_bits());
    (residues, dropped)
}

fn crt(residues: &[u64], primes: &[u64]) -> u128 {
    let modulus: u128 = primes.iter().map(|&q| q as u128).product();
    let mut t = 0u128;
    for (&r, &q) in residues.iter().zip(primes) {
        let m = modulus / q as u128;
        let inv = labinius::params::inv_mod((m % q as u128) as u64, q) as u128;
        t = (t + r as u128 * m % modulus * inv) % modulus;
    }
    t
}

fn check_digits(params: &Params) {
    let (residues, dropped) = plain_matrix_and_dropped(params);
    let primes: Vec<u64> = params.primes().iter().map(|&q| q as u64).collect();
    let d = params.dropped_bits();
    let modulus: u128 = primes.iter().map(|&q| q as u128).product();
    let bound = bd::top_bound(primes[0] as u16, d);
    let count = params.columns() * N;
    assert_eq!(dropped.top().len(), count);
    assert_eq!(dropped.digits().len(), primes.len() - 1);
    let mut limb_out = vec![0i16; count];
    let mut limbs: Vec<Vec<i16>> = Vec::new();
    for k in 0..primes.len() {
        bd::limb_residues(&dropped, k, &mut limb_out);
        limbs.push(limb_out.clone());
    }
    for i in 0..count {
        let r: Vec<u64> = (0..primes.len())
            .map(|k| (residues[k][i] as i64).rem_euclid(primes[k] as i64) as u64)
            .collect();
        let t = crt(&r, &primes);
        let top = dropped.top()[i] as u128;
        assert!(top <= bound as u128);
        let mut big = 0u128;
        for k in (1..primes.len()).rev() {
            let digit = dropped.digits()[k - 1][i] as u128;
            assert!(digit < primes[k] as u128);
            big = big * primes[k] as u128 + digit;
        }
        let big_t = ((top << d) + primes[0] as u128 * big) % modulus;
        let diff = (big_t + modulus - t) % modulus;
        let lo = if diff > modulus / 2 {
            -((modulus - diff) as i128)
        } else {
            diff as i128
        };
        assert!(
            lo >= -(1i128 << (d - 1)) && lo <= (1i128 << (d - 1)),
            "coefficient {i}: t_lo = {lo}"
        );
        for k in 0..primes.len() {
            let want = (big_t % primes[k] as u128) as i64;
            let got = (limbs[k][i] as i64).rem_euclid(primes[k] as i64);
            assert_eq!(got, want, "limb {k}, coefficient {i}");
        }
    }
}

#[test]
fn two_limb_digits_recover_the_integer_up_to_the_dropped_bits() {
    check_digits(&small(Modulus::Q9721_FS_S, vec![Modulus::Q12637_Q_S], 10));
}

#[test]
fn three_limb_digits_recover_the_integer_up_to_the_dropped_bits() {
    check_digits(&small(
        Modulus::Q3889_FS_S,
        vec![Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        9,
    ));
}

#[test]
fn dropped_commitment_round_trips_the_wire_and_rejects_an_oversized_digit() {
    let params = small(
        Modulus::Q3889_FS_S,
        vec![Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        9,
    );
    let (_, dropped) = plain_matrix_and_dropped(&params);
    let bytes = wire::pack_dropped(&dropped);
    assert_eq!(bytes.len(), dropped.wire_bytes());
    assert_eq!(
        bytes.len(),
        bd::bytes(&params.primes(), params.columns(), params.dropped_bits())
    );
    let back = wire::unpack_dropped(&params, &bytes).unwrap();
    assert_eq!(back, dropped);
    assert!(wire::unpack_dropped(&params, &bytes[..bytes.len() - 1]).is_err());

    let mut top = dropped.top().to_vec();
    top[0] = (bd::top_bound(params.primes()[0], params.dropped_bits()) + 1) as u16;
    let oversized = Dropped::of(
        params.primes(),
        params.columns(),
        params.dropped_bits(),
        top,
        dropped.digits().to_vec(),
    );
    assert!(wire::unpack_dropped(&params, &wire::pack_dropped(&oversized)).is_err());

    let mut digits = dropped.digits().to_vec();
    digits[1][0] = params.primes()[2];
    let oversized = Dropped::of(
        params.primes(),
        params.columns(),
        params.dropped_bits(),
        dropped.top().to_vec(),
        digits,
    );
    assert!(wire::unpack_dropped(&params, &wire::pack_dropped(&oversized)).is_err());
}

#[test]
fn a_bd_opening_verifies_and_a_wrapped_fold_or_a_swapped_column_is_refused() {
    let suite = &SUITES[0];
    let params = Params::sized(
        suite,
        Opening::BitDropped {
            bits: suite.dropped_bits,
        },
    );
    let pp = PublicParameters::from_seed(params.clone(), MATRIX_SEED);
    let witness = Witness::random(&params, WITNESS_SEED);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let (commitment, opening) = prover.commit(&witness);

    let mut transcript = Transcript::new(b"labinius/test-bd");
    let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
    let row = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
    let folded = prover.fold(opening, &challenges);
    let folded_row = verifier.fold_row_evaluation(&row, &challenges);

    let commitment = wire::unpack_commitment(&params, &wire::pack_commitment(&commitment)).unwrap();
    let dropped = commitment.dropped();
    let residual = bd::residual(
        pp.key(),
        dropped,
        challenges.challenges(),
        folded.elements(),
    )
    .unwrap();
    assert!(residual <= params.bd_cap() as u128);
    assert!(residual as f64 >= 0.25 * bd::expected_normsq(params.columns(), params.dropped_bits()));

    assert!(verifier
        .verify_opening(
            &commitment,
            &challenges,
            &point,
            OpeningMessage::BitDropped {
                folded_witness: &folded,
                folded_row_value: &folded_row,
            },
        )
        .is_ok());

    let mut tampered = folded.clone();
    tampered.elements_mut()[0].v[0] = tampered.elements_mut()[0].v[0].wrapping_add(1000);
    assert!(verifier
        .verify_opening(
            &commitment,
            &challenges,
            &point,
            OpeningMessage::BitDropped {
                folded_witness: &tampered,
                folded_row_value: &folded_row,
            },
        )
        .is_err());

    let mut top = dropped.top().to_vec();
    let mut digits = dropped.digits().to_vec();
    for i in 0..N {
        top.swap(i, N + i);
        for digit in digits.iter_mut() {
            digit.swap(i, N + i);
        }
    }
    let swapped = labinius::scheme::Commitment::of(
        params.primes(),
        params.columns(),
        labinius::scheme::CommitmentValue::Dropped(std::sync::Arc::new(Dropped::of(
            params.primes(),
            params.columns(),
            params.dropped_bits(),
            top,
            digits,
        ))),
    );
    assert!(verifier
        .verify_opening(
            &swapped,
            &challenges,
            &point,
            OpeningMessage::BitDropped {
                folded_witness: &folded,
                folded_row_value: &folded_row,
            },
        )
        .is_err());
}

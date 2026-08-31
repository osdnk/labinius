//! The tests the inlined `bin-fields` code came with: the `F162` and `B128` scalar products and
//! the word-sliced AVX-512 kernels against a bit-by-bit reference, and the cross-field switch end
//! to end, honest and tampered.
//!
//! The reference lives here rather than in `src/fields`: nothing in the crate uses it. Its
//! randomness is this crate's [`Rng`] in place of the library's `rand_chacha`, so the draws differ
//! from upstream's but each test covers the same ground.
use bin_ntt::fields::crossfield::*;
use bin_ntt::fields::f162;
use bin_ntt::fields::scalar::{B128, F162};
use bin_ntt::rng::Rng;
use std::arch::x86_64::*;

// ------------------------------------------------------- the bit-by-bit reference

fn bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}

fn flip(v: &mut [u64], i: usize) {
    v[i / 64] ^= 1u64 << (i % 64);
}

fn xor_shifted(acc: &mut [u64], src: &[u64], sh: usize) {
    for i in 0..src.len() * 64 {
        if bit(src, i) {
            flip(acc, i + sh);
        }
    }
}

/// `a b mod x^162 + x^81 + 1`.
fn mul162(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    let mut p = [0u64; 6];
    for i in 0..162 {
        if bit(&a, i) {
            xor_shifted(&mut p, &b, i);
        }
    }
    for j in (162..324).rev() {
        if bit(&p, j) {
            flip(&mut p, j);
            flip(&mut p, j - 162);
            flip(&mut p, j - 81);
        }
    }
    [p[0], p[1], p[2]]
}

/// `a b mod x^128 + x^7 + x^2 + x + 1`, the GHASH basis.
fn mul_ghash(a: u128, b: u128) -> u128 {
    let av = [a as u64, (a >> 64) as u64];
    let bv = [b as u64, (b >> 64) as u64];
    let mut p = [0u64; 4];
    for i in 0..128 {
        if bit(&av, i) {
            xor_shifted(&mut p, &bv, i);
        }
    }
    for j in (128..256).rev() {
        if bit(&p, j) {
            flip(&mut p, j);
            flip(&mut p, j - 128);
            flip(&mut p, j - 121);
            flip(&mut p, j - 126);
            flip(&mut p, j - 127);
        }
    }
    (p[0] as u128) | ((p[1] as u128) << 64)
}

// ------------------------------------------------------- the scalars and the kernels

fn to_arr(x: __m512i) -> [u64; 8] {
    unsafe { std::mem::transmute(x) }
}

fn from_arr(x: [u64; 8]) -> __m512i {
    unsafe { std::mem::transmute(x) }
}

fn rand162(r: &mut Rng) -> [u64; 3] {
    [r.next_u64(), r.next_u64(), r.next_u64() & ((1u64 << 34) - 1)]
}

fn rand128(r: &mut Rng) -> u128 {
    (r.next_u64() as u128) | ((r.next_u64() as u128) << 64)
}

/// The two funnel shifts [`f162::reduce_soa8`] folds the 324-bit product with.
#[test]
fn funnel_shift_semantics() {
    unsafe {
        let a = 0x0123_4567_89ab_cdefu64;
        let b = 0xfedc_ba98_7654_3210u64;
        let va = _mm512_set1_epi64(a as i64);
        let vb = _mm512_set1_epi64(b as i64);
        assert_eq!(to_arr(_mm512_shrdi_epi64::<17>(va, vb))[0], (a >> 17) | (b << 47));
        assert_eq!(to_arr(_mm512_shldi_epi64::<17>(va, vb))[0], (a << 17) | (b >> 47));
    }
}

/// `x` has order 243 in `F162^*`, so `F162` carries the 243-rd roots of unity `R_162` folds over.
#[test]
fn cyclotomic_order_243() {
    let x: [u64; 3] = [2, 0, 0];
    let mut acc: [u64; 3] = [1, 0, 0];
    let mut hit_one = Vec::new();
    for k in 1..=243 {
        acc = mul162(acc, x);
        if acc == [1, 0, 0] {
            hit_one.push(k);
        }
    }
    assert_eq!(hit_one, vec![243]);
}

#[test]
fn scalar_f162_matches_reference() {
    let mut r = Rng::new(22);
    for _ in 0..2000 {
        let a = rand162(&mut r);
        let b = rand162(&mut r);
        assert_eq!((F162(a) * F162(b)).0, mul162(a, b));
    }
}

#[test]
fn scalar_f162_order_243() {
    let x = F162([2, 0, 0]);
    let mut acc = F162::ONE;
    for k in 1..=243 {
        acc = acc * x;
        if k < 243 {
            assert_ne!(acc, F162::ONE, "order divides {k}");
        }
    }
    assert_eq!(acc, F162::ONE);
}

#[test]
fn scalar_b128_matches_reference() {
    let mut r = Rng::new(21);
    for _ in 0..2000 {
        let a = rand128(&mut r);
        let b = rand128(&mut r);
        assert_eq!((B128(a) * B128(b)).0, mul_ghash(a, b));
    }
}

/// The 8 lanes of [`f162::mul_soa8`] against the reference, element by element.
#[test]
fn f162_soa8() {
    let mut r = Rng::new(3);
    for _ in 0..256 {
        let a: Vec<[u64; 3]> = (0..8).map(|_| rand162(&mut r)).collect();
        let b: Vec<[u64; 3]> = (0..8).map(|_| rand162(&mut r)).collect();
        let mut aw = [[0u64; 8]; 3];
        let mut bw = [[0u64; 8]; 3];
        for i in 0..8 {
            for w in 0..3 {
                aw[w][i] = a[i][w];
                bw[w][i] = b[i][w];
            }
        }
        let va = [from_arr(aw[0]), from_arr(aw[1]), from_arr(aw[2])];
        let vb = [from_arr(bw[0]), from_arr(bw[1]), from_arr(bw[2])];
        let out = unsafe { f162::mul_soa8(va, vb) };
        let o = [to_arr(out[0]), to_arr(out[1]), to_arr(out[2])];
        for i in 0..8 {
            assert_eq!([o[0][i], o[1][i], o[2][i]], mul162(a[i], b[i]), "elem {i}");
        }
    }
}

/// Five [`f162::mac_soa8`] accumulated unreduced and reduced once agree with the reference sum.
#[test]
fn f162_mac_deferred() {
    let mut r = Rng::new(11);
    for _ in 0..64 {
        let k = 5;
        let mut acc = unsafe { [_mm512_setzero_si512(); 12] };
        let mut want = vec![[0u64; 3]; 8];
        for _ in 0..k {
            let a: Vec<[u64; 3]> = (0..8).map(|_| rand162(&mut r)).collect();
            let b: Vec<[u64; 3]> = (0..8).map(|_| rand162(&mut r)).collect();
            let mut aw = [[0u64; 8]; 3];
            let mut bw = [[0u64; 8]; 3];
            for i in 0..8 {
                for w in 0..3 {
                    aw[w][i] = a[i][w];
                    bw[w][i] = b[i][w];
                }
            }
            unsafe {
                f162::mac_soa8(
                    &mut acc,
                    [from_arr(aw[0]), from_arr(aw[1]), from_arr(aw[2])],
                    [from_arr(bw[0]), from_arr(bw[1]), from_arr(bw[2])],
                )
            };
            for i in 0..8 {
                let p = mul162(a[i], b[i]);
                for w in 0..3 {
                    want[i][w] ^= p[w];
                }
            }
        }
        let out = unsafe { f162::reduce_soa8(acc) };
        let o = [to_arr(out[0]), to_arr(out[1]), to_arr(out[2])];
        for i in 0..8 {
            assert_eq!([o[0][i], o[1][i], o[2][i]], want[i], "elem {i}");
        }
    }
}

// ------------------------------------------------------- the cross-field switch

fn setup(l: usize, seed: u64) -> (Vec<B128>, Vec<B128>, Vec<B128>, Transcript) {
    let mut r = Rng::new(seed);
    let pi0: Vec<B128> = (0..1 << l).map(|_| B128(rand128(&mut r))).collect();
    let r_lo: Vec<B128> = (0..LOG_PACK).map(|_| B128(rand128(&mut r))).collect();
    let r_hi: Vec<B128> = (0..l).map(|_| B128(rand128(&mut r))).collect();
    let ch = Transcript {
        r_prime: (0..LOG_PACK).map(|_| F162(rand162(&mut r))).collect(),
        r_pp: (0..l).map(|_| F162(rand162(&mut r))).collect(),
    };
    (pi0, r_lo, r_hi, ch)
}

fn true_claim(pi0: &[B128], r_lo: &[B128], r_hi: &[B128]) -> B128 {
    let eq_lo = eq_expand_b128(r_lo);
    let eq_hi = eq_expand_b128(r_hi);
    let mut s = B128::ZERO;
    for (y, &p) in pi0.iter().enumerate() {
        for i in 0..PACK {
            if p.bit(i) {
                s = s + eq_lo[i] * eq_hi[y];
            }
        }
    }
    s
}

fn true_pi1_eval(pi0: &[B128], r_pp: &[F162]) -> F162 {
    let eq = eq_expand_f162(r_pp);
    pi0.iter().zip(&eq).fold(F162::ZERO, |a, (&p, &e)| a + F162::from_b128(p) * e)
}

#[test]
fn switch_end_to_end() {
    for l in [1usize, 2, 5, 8] {
        let (pi0, r_lo, r_hi, ch) = setup(l, 100 + l as u64);
        let claim = true_claim(&pi0, &r_lo, &r_hi);
        let proof = prove(&pi0, &r_lo, &r_hi, &ch);
        let z = verify(&proof, claim, &r_lo, &r_hi, &ch).expect("verify");
        assert_eq!(z, true_pi1_eval(&pi0, &ch.r_pp), "l={l}: opened wrong value");
    }
}

#[test]
fn switch_rejects_wrong_claim() {
    let l = 6;
    let (pi0, r_lo, r_hi, ch) = setup(l, 7);
    let claim = true_claim(&pi0, &r_lo, &r_hi);
    let proof = prove(&pi0, &r_lo, &r_hi, &ch);
    assert!(verify(&proof, claim + B128::ONE, &r_lo, &r_hi, &ch).is_err());
}

#[test]
fn switch_rejects_tampered_v() {
    let l = 6;
    let (pi0, r_lo, r_hi, ch) = setup(l, 9);
    let claim = true_claim(&pi0, &r_lo, &r_hi);
    let mut proof = prove(&pi0, &r_lo, &r_hi, &ch);
    proof.v[3] = proof.v[3] + B128(0x1234);
    assert!(verify(&proof, claim, &r_lo, &r_hi, &ch).is_err());
}

#[test]
fn switch_rejects_tampered_round() {
    let l = 6;
    let (pi0, r_lo, r_hi, ch) = setup(l, 11);
    let claim = true_claim(&pi0, &r_lo, &r_hi);
    let mut proof = prove(&pi0, &r_lo, &r_hi, &ch);
    proof.rounds[2][0] += F162::ONE;
    assert!(verify(&proof, claim, &r_lo, &r_hi, &ch).is_err());
}

#[test]
fn transparent_coeff_matches_definition() {
    let l = 5;
    let (_, _, r_hi, ch) = setup(l, 42);
    let batch = eq_expand_f162(&ch.r_prime);
    let tab = psi_table(&batch);
    let eq_hi = eq_expand_b128(&r_hi);
    let eq_pp = eq_expand_f162(&ch.r_pp);
    let want = eq_hi.iter().zip(&eq_pp).fold(F162::ZERO, |a, (&h, &v)| a + v * psi(&tab, h));
    assert_eq!(transparent_coeff(&r_hi, &ch.r_pp, &batch), want);
}

/// The stepwise prover and verifier the keccak path drives agree with the batch `prove`/`verify`.
#[test]
fn stepwise_matches_batch() {
    let l = 7;
    let (pi0, r_lo, r_hi, ch) = setup(l, 314);
    let claim = true_claim(&pi0, &r_lo, &r_hi);
    let batch = eq_expand_f162(&ch.r_prime);

    let (v, eq_hi) = SwitchProver::partial_evals_and_eq(&pi0, &r_hi);
    let mut pv = SwitchProver::new(&pi0, &eq_hi, &batch);
    let mut vf = SwitchVerifier::start(&v, claim, &r_lo, &batch).expect("start");
    for round in 0..l {
        let m = pv.msg();
        let r = ch.r_pp[round];
        vf.round(m, r);
        pv.fold(r);
    }
    let opened = eval_pi1(&pi0, &ch.r_pp);
    assert_eq!(pv.final_eval(), opened);
    vf.finish(&r_hi, &ch.r_pp, &batch, opened).expect("finish");
}

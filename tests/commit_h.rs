//! Correctness, bound and overflow tests for the horizontal-layout commitment.
use bin_fields::scalar::F162;
use bin_ntt::f162::{lift_elem, random_elems, RandomF162};
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::commit_h::*;
use bin_ntt::simd::horizontal_gen::{ntt_gen_hbatch4, out_bound, HBatch4};
use bin_ntt::types::*;

fn random_a(n: usize, q: u16, seed: u64) -> Vec<HBatch4> {
    let mut rng = Rng::new(seed);
    let half = (q - 1) / 2;
    (0..n)
        .map(|_| {
            let mut b = HBatch4::zero();
            for r in 0..81 {
                for l in 0..32 {
                    b.v[r][l] = rng.below(q as u32) as i16 - half as i16;
                }
            }
            b
        })
        .collect()
}

// ------------------------------------------------------------------------------- the front end

#[test]
fn front_end_matches_scalar_and_lift() {
    bin_ntt::f162::assert_layout();
    let mut rng = Rng::new(7);
    for trial in 0..64 {
        let elems: [F162; 16] = std::array::from_fn(|i| {
            if trial == 1 {
                F162([0, 0, 0])
            } else if trial == 2 {
                F162([!0, !0, (1u64 << 34) - 1])
            } else if trial == 3 {
                F162([0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555, i as u64 & 1])
            } else {
                F162::random(&mut rng)
            }
        });
        let got = unsafe { hbatch4_from_f162(&elems) };
        let want = hbatch4_from_f162_scalar(&elems);
        assert_eq!(got.v, want.v, "SIMD front end != scalar reference (trial {trial})");

        // ... and the scalar reference is the crate's lift, laid out by HBatch4.
        let polys: [BinaryPoly; 4] = std::array::from_fn(|p| {
            let c = bin_ntt::f162::lift4(elems[4 * p..4 * p + 4].try_into().unwrap());
            let mut bp = BinaryPoly::default();
            for i in 0..N {
                bp.set(i, c[i] == 1);
            }
            bp
        });
        assert_eq!(got.v, HBatch4::from_binary(&polys).v, "front end != lift4 (trial {trial})");
    }
}

#[test]
fn front_end_then_kernel_matches_scalar_ntt() {
    let elems = random_elems(16 * 5, 0x1234);
    for g in 0..5 {
        let chunk: &[F162; 16] = elems[16 * g..16 * g + 16].try_into().unwrap();
        for q in QS {
            let mut b = unsafe { hbatch4_from_f162(chunk) };
            if q == 3889 {
                unsafe { ntt_gen_hbatch4::<3889>(&mut b) };
            } else {
                unsafe { ntt_gen_hbatch4::<9721>(&mut b) };
            }
            for p in 0..4 {
                let i = 4 * g + p;
                let want = if q == 3889 {
                    scalar::ntt::<3889>(&lift_elem(&elems, i))
                } else {
                    scalar::ntt::<9721>(&lift_elem(&elems, i))
                };
                assert_eq!(scalar::normalize_i16(&b.get(p).v, q), want, "q {q} elem {i}");
            }
        }
    }
}

// ------------------------------------------------------------------------------ the commitment

fn check_commit<const Q: u16>(ngroups: usize, seed: u64) {
    let elems = random_elems(16 * ngroups, seed);
    let a = random_a(ngroups, Q, seed ^ 0x5EED);
    let want = commit_h_scalar::<Q>(&elems, &a);

    assert_eq!(commit_h::<Q>(&elems, &a), want, "commit_h, q = {Q}, {ngroups} groups");
    assert_eq!(commit_h_var::<Q, 1, false, 0, 0>(&elems, &a), want, "G=1 nopf");
    assert_eq!(commit_h_var::<Q, 2, false, 1, 2>(&elems, &a), want, "G=2");
    assert_eq!(commit_h_var::<Q, 4, false, 3, 1>(&elems, &a), want, "G=4");

    let mut ap = a.clone();
    permute_a_slice(&mut ap);
    assert_eq!(commit_h_var::<Q, 1, true, 2, 0>(&elems, &ap), want, "G=1 permuted A");
    assert_eq!(commit_h_var::<Q, 2, true, 4, 3>(&elems, &ap), want, "G=2 permuted A");
    assert_eq!(commit_h_var::<Q, 4, true, 0, 2>(&elems, &ap), want, "G=4 permuted A nopf");
}

#[test]
fn commitment_matches_scalar_reference() {
    for n in [1usize, 4, 5, 6] {
        check_commit::<3889>(n, 0xC0FFEE + n as u64);
        check_commit::<9721>(n, 0xBEEF + n as u64);
    }
}

/// More groups than one reduction interval, so the periodic reduction runs several times
/// (KRED = 18 / 10, and G = 4 rounds it down to 16 / 8).
#[test]
fn commitment_over_several_reduction_intervals() {
    check_commit::<3889>(40, 11);
    check_commit::<9721>(25, 12);
}

// ----------------------------------------------------------------------------------- the bounds

/// The declared reduction interval really keeps the raw i32 accumulator inside i32: feed the
/// worst case (every W lane at the kernel's declared output bound, every A lane at +-(q-1)/2,
/// signs chosen so both products of a pair have the same sign) and shadow the accumulator in i64.
fn check_overflow_bound<const Q: u16>() {
    let wmax = out_bound::<Q>() as i16;
    let amax = ((Q - 1) / 2) as i16;
    let k = kred::<Q>();
    let mut w = HBatch4::zero();
    let mut a = HBatch4::zero();
    for r in 0..81 {
        for l in 0..32 {
            let s = if (r + l / 2) % 2 == 0 { 1i16 } else { -1 };
            w.v[r][l] = s * wmax;
            a.v[r][l] = s * amax;
        }
    }
    let wp = permute_a(&w);
    let ap = permute_a(&a);

    let mut acc = Acc::zero();
    let mut shadow = [[0i64; 16]; 81];
    for _ in 0..k {
        unsafe { basemul_step::<true>(&w, &ap, &mut acc) };
        for r in 0..81 {
            for t in 0..16 {
                shadow[r][t] += wp.v[r][2 * t] as i64 * ap.v[r][2 * t] as i64
                    + wp.v[r][2 * t + 1] as i64 * ap.v[r][2 * t + 1] as i64;
                assert!(shadow[r][t].abs() < i32::MAX as i64, "q {Q}: i32 accumulator overflows");
                assert!(
                    shadow[r][t].abs() <= acc_cap::<Q>(),
                    "q {Q}: {} exceeds the declared cap {}",
                    shadow[r][t],
                    acc_cap::<Q>()
                );
                assert_eq!(acc.v[r][t] as i64, shadow[r][t], "q {Q}: accumulator != i64 shadow");
            }
        }
    }
    // The worst case really is attained (the cap is not slack by more than the initial |acc|),
    // and one further group would break i32.
    let hit = (0..81).flat_map(|r| (0..16).map(move |t| (r, t))).map(|(r, t)| shadow[r][t].abs()).max().unwrap();
    assert_eq!(hit, k as i64 * acc_step::<Q>(), "q {Q}: worst case not attained");
    assert!(
        acc_cap::<Q>() + acc_step::<Q>() > i32::MAX as i64 - 2 * Q as i64,
        "q {Q}: KRED = {k} is not maximal"
    );
}

#[test]
fn overflow_bound_is_tight_and_safe() {
    check_overflow_bound::<3889>();
    check_overflow_bound::<9721>();
}

/// `reduce_acc` is exact (congruent mod q) and lands inside 1.5 q, over the whole i32 range the
/// accumulator can reach.
fn check_reduce<const Q: u16>() {
    let cap = acc_cap::<Q>();
    let mut rng = Rng::new(Q as u64 ^ 0xABCD);
    let mut acc = Acc::zero();
    let mut want = [[0i64; 16]; 81];
    for r in 0..81 {
        for t in 0..16 {
            let x = match (r * 16 + t) % 6 {
                0 => cap,
                1 => -cap,
                2 => 0,
                3 => Q as i64,
                4 => -(Q as i64),
                _ => (rng.next_u64() % (2 * cap as u64 + 1)) as i64 - cap,
            };
            acc.v[r][t] = x as i32;
            want[r][t] = x;
        }
    }
    unsafe { reduce_acc::<Q>(&mut acc) };
    for r in 0..81 {
        for t in 0..16 {
            let got = acc.v[r][t] as i64;
            assert_eq!(
                got.rem_euclid(Q as i64),
                want[r][t].rem_euclid(Q as i64),
                "q {Q}: reduction is not congruent"
            );
            assert!(
                got.abs() <= 3 * Q as i64 / 2,
                "q {Q}: reduction left |{got}| > 1.5 q (input {})",
                want[r][t]
            );
        }
    }
}

#[test]
fn reduction_is_exact_and_bounded() {
    check_reduce::<3889>();
    check_reduce::<9721>();
}

#[test]
fn declared_reduction_intervals() {
    assert_eq!(kred::<3889>(), 18);
    assert_eq!(kred::<9721>(), 10);
    assert!(acc_cap::<3889>() < i32::MAX as i64);
    assert!(acc_cap::<9721>() < i32::MAX as i64);
}

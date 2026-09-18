//! Correctness of the NTT-domain commitment `y[j] = sum_i A_i[j] * NTT_q(w_i)[j] mod q`: the
//! commitment against the scalar reference for both primes, the raw accumulator's exact fold-back
//! and its overflow bound (replayed in i64 against the real kernel output), and the two-limb
//! commitment against two single-limb ones.
use labinius::f162::{self, RandomF162};
use labinius::params::N;
use labinius::rng::Rng;
use labinius::scalar;
use labinius::simd::commit::{self as cm, Acc};
use labinius::simd::transpose_f162 as tf;
use labinius::simd::ntt::bin_asm as vb;
use labinius::simd::ntt::bin_large as vl;
use labinius::ring::*;
use labinius::F162;

/// The binary kernel of a splitting prime: `ntt::bin_asm` below `2^14`, `ntt::bin_large`
/// above it.
///
/// # Safety
/// See the two kernels; `out` is 64-byte aligned.
unsafe fn transform<const Q: u16>(idx: &tf::BinaryIndex32, out: &mut Batch32) {
    if vl::is_large(Q) {
        vl::ntt_bin_batch32::<Q>(idx, out);
    } else {
        vb::ntt_bin_batch32::<Q>(idx, out);
    }
}

/// The production commitment over a single splitting limb.
fn commit<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    cm::commit_limbs(
        elems,
        &[cm::Limb {
            q: Q,
            quad: false,
            a,
        }],
        None,
    )
    .into_iter()
    .next()
    .unwrap()
}

const FULL: F162 = F162([!0u64, !0u64, (1u64 << 34) - 1]);

fn random_elems(nb: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..128 * nb).map(|_| F162::random(&mut rng)).collect()
}

/// `nb` batches of A, uniform in the centered range [-(q-1)/2, (q-1)/2].
fn random_a(nb: usize, q: u16, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i16;
    (0..nb)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(q as u32) as i16 - half;
                }
            }
            b
        })
        .collect()
}

/// A at the two extremes of the centered range only.
fn extreme_a(nb: usize, q: u16, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i16;
    (0..nb)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = if rng.next_u64() & 1 == 0 { half } else { -half };
                }
            }
            b
        })
        .collect()
}

fn reference<const Q: u16>(elems: &[F162], a: &[Batch32]) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0i64; N];
    for i in 0..elems.len() / 4 {
        let w = scalar::ntt::<Q>(&f162::lift_elem(elems, i));
        let (b, p) = (i / 32, i % 32);
        for j in 0..N {
            y[j] = (y[j] + w[j] as i64 * a[b].v[j][p] as i64) % q;
        }
    }
    let mut out = [0u32; N];
    for j in 0..N {
        out[j] = y[j].rem_euclid(q) as u32;
    }
    out
}

fn check<const Q: u16>(label: &str, elems: &[F162], a: &[Batch32]) {
    let want = reference::<Q>(elems, a);
    let got = commit::<Q>(elems, a);
    for j in 0..N {
        assert_eq!(
            got[j],
            want[j],
            "{label}: q = {Q}, slot {j} ({} batches)",
            a.len()
        );
    }
}

fn random_case<const Q: u16>() {
    // batch counts on both sides of the fold-back period, and one that is a multiple of it.
    let p = cm::red_period(Q);
    for (nb, seed) in [(1usize, 1u64), (4, 2), (p, 3), (2 * p + 3, 4)] {
        let elems = random_elems(nb, seed);
        let a = random_a(nb, Q, seed ^ 0x5a5a);
        check::<Q>("random", &elems, &a);
    }
}

#[test]
fn random_3889() {
    random_case::<3889>();
}

#[test]
fn random_9721() {
    random_case::<9721>();
}

#[test]
fn random_17497() {
    random_case::<17497>();
}

#[test]
fn random_19441() {
    random_case::<19441>();
}

/// The fold-back is exact modulo q and lands inside the declared bound, over the whole i32 range
/// the accumulator can reach.
#[test]
fn reduce_scheme_exact() {
    for q in [3889u16, 9721, 17497, 19441] {
        let lim = cm::acc_after_reduce(q) + cm::red_period(q) as i64 * cm::acc_per_batch(q);
        assert!(lim <= i32::MAX as i64);
        assert!(8 * cm::acc_after_hsum(q) <= i32::MAX as i64);
        let mut rng = Rng::new(0xC0FFEE ^ q as u64);
        let mut xs: Vec<i32> = vec![
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN + 1,
            32767,
            32768,
            -32768,
            -32769,
        ];
        xs.push(lim as i32);
        xs.push(-(lim as i32));
        for _ in 0..200_000 {
            let x = rng.next_u64() as i32;
            xs.push(x);
            xs.push((x as i64 % (lim + 1)) as i32);
        }
        for &x in &xs {
            let r = cm::reduce_acc_i32(x, q);
            assert_eq!(
                (r as i64).rem_euclid(q as i64),
                (x as i64).rem_euclid(q as i64),
                "fold-back not exact at x = {x}, q = {q}"
            );
            assert!(
                (r as i64).abs() <= cm::acc_after_reduce(q),
                "fold-back bound broken at x = {x}, q = {q}: {r}"
            );
        }
    }
}

/// All-ones inputs and A at +-(q-1)/2, over more than two reduction periods: every entry point
/// still matches the reference, the kernel output stays inside the bound the period was derived
/// from, and an i64 replay of the exact accumulation order shows no lane ever leaves i32.
fn bound_adversarial<const Q: u16>() {
    let period = cm::red_period(Q);
    let nb = 2 * period + 3;
    let elems = vec![FULL; 128 * nb];
    let a = extreme_a(nb, Q, 0xBEEF);

    let mut w: Vec<Batch32> = (0..nb)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    let mut idx = labinius::simd::transpose_f162::BinaryIndex32::zero();
    for (b, o) in w.iter_mut().enumerate() {
        unsafe {
            tf::slice_f162_into(
                &*(elems.as_ptr().add(128 * b) as *const [F162; 128]),
                &mut idx,
            );
            transform::<Q>(&idx, o);
        }
    }

    let mut maxw = 0i64;
    for b in &w {
        for j in 0..N {
            for p in 0..32 {
                maxw = maxw.max((b.v[j][p] as i64).abs());
            }
        }
    }
    assert!(
        maxw <= cm::w_bound(Q),
        "kernel output {maxw} above the declared bound for q = {Q}"
    );

    // i64 shadow of the 8 i32 lanes of every slot: lane l carries the ring elements
    // 2l, 2l+1, 2l+16, 2l+17 of every batch, in the order `mac27` accumulates them.
    let mut lanes = vec![[0i64; 8]; N];
    for b in 0..nb {
        for j in 0..N {
            for l in 0..8 {
                let mut v = lanes[j][l];
                for p in [2 * l, 2 * l + 1, 2 * l + 16, 2 * l + 17] {
                    v += w[b].v[j][p] as i64 * a[b].v[j][p] as i64;
                }
                assert!(
                    v.abs() <= i32::MAX as i64,
                    "accumulator overflow at batch {b}, slot {j}, lane {l}: {v} (q = {Q})"
                );
                lanes[j][l] = v;
            }
        }
        if (b + 1) % period == 0 {
            for j in 0..N {
                for l in 0..8 {
                    lanes[j][l] = cm::reduce_acc_i32(lanes[j][l] as i32, Q) as i64;
                }
            }
        }
    }
    let q = Q as i64;
    let want = reference::<Q>(&elems, &a);
    for j in 0..N {
        let s: i64 = lanes[j].iter().sum();
        assert_eq!(
            s.rem_euclid(q) as u32,
            want[j],
            "shadow model disagrees at slot {j}"
        );
    }

    check::<Q>("adversarial", &elems, &a);

    let elems = random_elems(nb, 7);
    check::<Q>("random x extreme A", &elems, &a);
}

#[test]
fn bound_adversarial_3889() {
    bound_adversarial::<3889>();
}

#[test]
fn bound_adversarial_9721() {
    bound_adversarial::<9721>();
}

#[test]
fn bound_adversarial_17497() {
    bound_adversarial::<17497>();
}

#[test]
fn bound_adversarial_19441() {
    bound_adversarial::<19441>();
}

/// Two limbs off one slicing pass are the two single-limb commitments, and the kept transform is
/// the base limb's kernel output.
#[test]
fn two_limbs() {
    let nb = 5;
    let elems = random_elems(nb, 11);
    let a3 = random_a(nb, 3889, 12);
    let a9 = random_a(nb, 9721, 13);
    let a19 = random_a(nb, 19441, 14);
    let y3 = commit::<3889>(&elems, &a3);
    let y9 = commit::<9721>(&elems, &a9);
    let y19 = commit::<19441>(&elems, &a19);
    assert_eq!(y3, reference::<3889>(&elems, &a3));
    assert_eq!(y9, reference::<9721>(&elems, &a9));
    assert_eq!(y19, reference::<19441>(&elems, &a19));

    let mut w: Vec<Batch32> = (0..nb)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    let limbs = [
        cm::Limb {
            q: 3889,
            quad: false,
            a: &a3,
        },
        cm::Limb {
            q: 9721,
            quad: false,
            a: &a9,
        },
        cm::Limb {
            q: 19441,
            quad: false,
            a: &a19,
        },
    ];
    let z = cm::commit_limbs(&elems, &limbs, Some(&mut w));
    assert_eq!(z[0], y3);
    assert_eq!(z[1], y9);
    assert_eq!(z[2], y19);

    let mut idx = labinius::simd::transpose_f162::BinaryIndex32::zero();
    let mut want = Batch32::zero(Representation::Ntt);
    for (b, got) in w.iter().enumerate() {
        unsafe {
            tf::slice_f162_into(
                &*(elems.as_ptr().add(128 * b) as *const [F162; 128]),
                &mut idx,
            );
            vb::ntt_bin_batch32::<3889>(&idx, &mut want);
        }
        assert!(got.v == want.v, "kept transform, batch {b}");
        assert_eq!(got.representation, Representation::Ntt);
    }
}

/// The all-zero input and the all-zero matrix.
#[test]
fn degenerate() {
    let nb = 4;
    let zero = vec![F162([0; 3]); 128 * nb];
    let a = random_a(nb, 3889, 21);
    assert_eq!(commit::<3889>(&zero, &a), [0u32; N]);
    let elems = random_elems(nb, 22);
    let za: Vec<Batch32> = (0..nb)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    assert_eq!(commit::<9721>(&elems, &za), [0u32; N]);
}

/// The accumulator is what the kernels assume, and the slot map is a bijection onto the
/// (vector, lane group) pairs it claims.
#[test]
fn acc_layout() {
    assert_eq!(core::mem::size_of::<Acc>(), cm::ACC_VECS * 64);
    assert_eq!(core::mem::align_of::<Acc>(), 64);
    let a = Acc::zero();
    assert_eq!(a.v[0][0], 0);
    assert_eq!(a.v[cm::ACC_VECS - 1][15], 0);

    let mut seen = std::collections::HashSet::new();
    for s in 0..N {
        let (v, g) = cm::slot_lane(s);
        assert!(v < cm::ACC_VECS && g < 2);
        assert!(seen.insert((v, g)), "slot {s} collides at ({v}, {g})");
    }
    assert_eq!(seen.len(), N);
}

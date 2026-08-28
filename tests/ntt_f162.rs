//! Correctness of the `F162` front end: the slicer against a scalar construction of the index
//! rows, the full pipeline against `scalar::ntt(lift4(..))` slot for slot, the 2q and streamed
//! drivers against the materialised one, and a product in the NTT domain.
use bin_fields::scalar::F162;
use bin_ntt::f162::{self, RandomF162};
use bin_ntt::params::N;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::ntt_f162 as nf;
use bin_ntt::simd::pointwise;
use bin_ntt::simd::transpose_f162 as tf;
use bin_ntt::types::*;

fn one(m: usize) -> F162 {
    let mut x = F162([0; 3]);
    x.0[m >> 6] |= 1u64 << (m & 63);
    x
}

const FULL: F162 = F162([!0u64, !0u64, (1u64 << 34) - 1]);

/// Adversarial ring elements (four `F162` each): all-zero, all bits set, single bits at the limb
/// boundaries of each of the four elements separately, alternating patterns.
fn adversarial() -> Vec<[F162; 4]> {
    let mut v = vec![[F162([0; 3]); 4], [FULL; 4]];
    for k in 0..4 {
        for m in [0usize, 63, 64, 127, 128, 161] {
            let mut e = [F162([0; 3]); 4];
            e[k] = one(m);
            v.push(e);
        }
        let mut e = [F162([0; 3]); 4];
        e[k] = FULL;
        v.push(e);
    }
    for phase in 0..2 {
        let mut e = [F162([0; 3]); 4];
        for k in 0..4 {
            for m in 0..162 {
                if m % 2 == phase {
                    e[k].0[m >> 6] |= 1u64 << (m & 63);
                }
            }
        }
        v.push(e);
        let mut f = [F162([0; 3]); 4];
        for k in 0..4 {
            for m in 0..162 {
                if (m + k) % 3 == phase {
                    f[k].0[m >> 6] |= 1u64 << (m & 63);
                }
            }
        }
        v.push(f);
    }
    v
}

/// `nb` batches of 128 elements: the adversarial ring elements first, then random ones.
fn inputs(nb: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    let mut ring = adversarial();
    while ring.len() < 32 * nb {
        ring.push(std::array::from_fn(|_| F162::random(&mut rng)));
    }
    ring.truncate(32 * nb);
    ring.into_iter().flatten().collect()
}

fn run<const Q: u16>(elems: &[F162]) -> Vec<Batch32> {
    let mut out: Vec<Batch32> =
        (0..elems.len() / 128).map(|_| Batch32::zero(Representation::Ntt)).collect();
    nf::ntt_f162::<Q>(elems, &mut out);
    out
}

#[test]
fn layout() {
    f162::assert_layout();
}

#[test]
fn lift4_semantics() {
    let mut rng = Rng::new(7);
    for _ in 0..16 {
        let q: [F162; 4] = std::array::from_fn(|_| F162::random(&mut rng));
        let c = f162::lift4(&q);
        for i in 0..N {
            assert_eq!(c[i], f162::bit(&q[i & 3], i >> 2));
            assert!(c[i] < 2);
        }
        assert_eq!(f162::pack4(&c), q);
    }
}

#[test]
fn slicer_matches_scalar() {
    let elems = inputs(8, 11);
    for b in 0..8 {
        let chunk: &[F162; 128] = elems[128 * b..128 * b + 128].try_into().unwrap();
        let got = unsafe { tf::slice_f162(chunk) };
        let want = f162::index_rows_scalar(chunk);
        for i in 0..162 {
            assert_eq!(got.rows[i], want.rows[i], "batch {b}, row {i}");
        }
    }
}

fn ntt_matches<const Q: u16>() {
    let nb = 64;
    let elems = inputs(nb, 0x1234 + Q as u64);
    let out = run::<Q>(&elems);
    for r in 0..32 * nb {
        let want = scalar::ntt::<Q>(&f162::lift_elem(&elems, r));
        let got = scalar::normalize_i16(&out[r / 32].get(r % 32).v, Q);
        assert_eq!(got, want, "q = {Q}, ring element {r}");
    }
}

#[test]
fn ntt_3889() {
    ntt_matches::<3889>();
}

#[test]
fn ntt_9721() {
    ntt_matches::<9721>();
}

#[test]
fn two_primes_and_stream() {
    let nb = 12;
    let elems = inputs(nb, 99);
    let a = run::<3889>(&elems);
    let b = run::<9721>(&elems);
    let mut a2: Vec<Batch32> = (0..nb).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let mut b2 = a2.clone();
    nf::ntt_f162_2q::<3889, 9721>(&elems, &mut a2, &mut b2);
    for i in 0..nb {
        assert_eq!(a[i].v, a2[i].v, "2q q=3889 batch {i}");
        assert_eq!(b[i].v, b2[i].v, "2q q=9721 batch {i}");
    }
    let mut seen = 0;
    nf::ntt_f162_stream::<3889>(&elems, |i, x| {
        assert_eq!(a[i].v, x.v, "streamed batch {i}");
        seen += 1;
    });
    assert_eq!(seen, nb);
    let mut pf: Vec<Batch32> = (0..nb).map(|_| Batch32::zero(Representation::Ntt)).collect();
    nf::ntt_f162_pf::<9721, 4>(&elems, &mut pf);
    for i in 0..nb {
        assert_eq!(b[i].v, pf[i].v, "prefetched batch {i}");
    }
}

fn product<const Q: u16>() {
    let elems = inputs(2, 5150);
    let out = run::<Q>(&elems);
    let mut prod = Batch32::zero(Representation::Ntt);
    unsafe { pointwise::mul_batch_batch::<Q>(&out[0], &out[1], &mut prod) };
    for p in [0usize, 1, 7, 31] {
        let a = f162::lift_elem(&elems, p);
        let b = f162::lift_elem(&elems, 32 + p);
        let want = scalar::ntt::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
        let got = scalar::normalize_i16(&prod.get(p).v, Q);
        assert_eq!(got, want, "q = {Q}, product at lane {p}");
    }
}

#[test]
fn product_3889() {
    product::<3889>();
}

#[test]
fn product_9721() {
    product::<9721>();
}

fn mont_driver<const Q: u16>() {
    use bin_ntt::params::Params;
    let elems = bin_ntt::f162::random_elems(128 * 3, 0xA11CE);
    let mut out = vec![Batch32::zero(Representation::Ntt); 3];
    bin_ntt::simd::ntt_f162::ntt_f162_mont::<Q>(&elems, &mut out);
    for r in 0..96 {
        let want = scalar::ntt::<Q>(&bin_ntt::f162::lift_elem(&elems, r));
        let got = scalar::normalize_i16(&out[r / 32].get(r % 32).v, Q);
        for j in 0..bin_ntt::params::N {
            assert_eq!(got[j], (want[j] as u64 * Params::<Q>::R as u64 % Q as u64) as u32, "r={r} j={j}");
        }
    }
}

#[test]
fn ntt_f162_mont_is_r_times_reference() {
    mont_driver::<3889>();
    mont_driver::<9721>();
}

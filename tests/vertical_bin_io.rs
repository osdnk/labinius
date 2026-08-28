//! Every entry point of `vertical_bin_io` must reproduce `vertical_bin::ntt_bin_polys` bit for
//! bit, for both primes, on random and adversarial inputs; the accumulating consumer is checked
//! against the scalar reference.
use bin_ntt::params::N;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::vertical_bin as vb;
use bin_ntt::simd::vertical_bin_io as io;
use bin_ntt::types::*;

const NB: usize = 37; // batches; not a multiple of any group size used below

fn inputs() -> Vec<BinaryPoly> {
    let mut rng = Rng::new(0x1234_5678);
    let mut v: Vec<BinaryPoly> = (0..32 * NB).map(|_| BinaryPoly::random(&mut rng)).collect();
    // adversarial: all zero, all ones, alternating, monomials at block boundaries
    v[0] = BinaryPoly::default();
    let mut ones = BinaryPoly::default();
    for i in 0..N {
        ones.set(i, true);
    }
    v[1] = ones;
    for phase in 0..2 {
        let mut alt = BinaryPoly::default();
        for i in 0..N {
            alt.set(i, i % 2 == phase);
        }
        v[2 + phase] = alt;
    }
    for (k, d) in [0usize, 1, 80, 81, 161, 162, 323, 324, 485, 486, 647].iter().enumerate() {
        let mut p = BinaryPoly::default();
        p.set(*d, true);
        v[4 + k] = p;
    }
    v
}

fn zeros() -> Vec<Batch32> {
    (0..NB).map(|_| Batch32::zero(Representation::Ntt)).collect()
}

fn reference<const Q: u16>(polys: &[BinaryPoly]) -> Vec<Batch32> {
    let mut out = zeros();
    vb::ntt_bin_polys::<Q>(polys, &mut out);
    out
}

fn same(a: &[Batch32], b: &[Batch32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.representation, y.representation, "{what}: batch {i} representation");
        for j in 0..N {
            assert_eq!(x.v[j], y.v[j], "{what}: batch {i} slot {j}");
        }
    }
}

fn check<const Q: u16>() {
    let polys = inputs();
    let want = reference::<Q>(&polys);

    // --- transposed forms
    let mut idx: Vec<BinaryIndex32> = (0..NB).map(|_| BinaryIndex32::zero()).collect();
    io::transpose_polys(&polys, &mut idx);
    let mut nib: Vec<BinaryBatch32> = (0..NB).map(|_| BinaryBatch32::zero()).collect();
    io::nibble_polys(&polys, &mut nib);
    // the nibble expansion must reproduce the index rows exactly
    let mut e = BinaryIndex32::zero();
    for b in 0..NB {
        unsafe { io::expand_nibbles(&nib[b], &mut e) };
        assert_eq!(e.rows, idx[b].rows, "expand_nibbles batch {b}");
        let want_idx = BinaryIndex32::from_nibbles(&nib[b]);
        assert_eq!(idx[b].rows, want_idx.rows, "transpose vs scalar nibbles, batch {b}");
    }

    let mut got = zeros();
    io::ntt_bin_idx::<Q>(&idx, &mut got);
    same(&want, &got, "ntt_bin_idx");

    let mut got = zeros();
    io::ntt_bin_idx_pf::<Q, 2, 162>(&idx, &mut got);
    same(&want, &got, "ntt_bin_idx_pf");

    let mut got = zeros();
    io::ntt_bin_nib::<Q>(&nib, &mut got);
    same(&want, &got, "ntt_bin_nib");

    let mut got = zeros();
    io::ntt_bin_polys_pf::<Q, 3>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_pf");

    let mut got = zeros();
    io::ntt_bin_polys_copy::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_copy");

    let mut got = zeros();
    io::ntt_bin_polys_fence::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_fence");

    let mut got = zeros();
    io::ntt_bin_polys_cached::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_cached");

    let mut got = zeros();
    io::ntt_bin_polys_grouped::<Q, 8>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_grouped");

    let mut got = zeros();
    io::ntt_bin_polys_pipelined_off::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_pipelined_off");

    let mut got = zeros();
    io::ntt_bin_polys_pipelined::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_pipelined");

    let mut got = zeros();
    io::ntt_bin_polys_pipelined_p34::<Q>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_pipelined_p34");

    let mut got = zeros();
    io::ntt_bin_polys_pfmid::<Q, 2, 3>(&polys, &mut got);
    same(&want, &got, "ntt_bin_polys_pfmid");

    // --- huge-page output buffer
    for huge in [false, true] {
        let mut hb = io::Batches::new(NB, huge);
        io::ntt_bin_polys_pf::<Q, 2>(&polys, &mut hb);
        same(&want, &hb, if huge { "Batches(huge)" } else { "Batches(4K)" });
    }

    // --- also equal to the scalar reference, slot for slot
    let mut rng = Rng::new(7);
    for _ in 0..4 {
        let i = rng.below((32 * NB) as u32) as usize;
        let w = scalar::ntt::<Q>(&scalar::lift(&polys[i]));
        assert_eq!(scalar::normalize_i16(&want[i / 32].get(i % 32).v, Q), w);
    }
}

fn check_2q() {
    let polys = inputs();
    let want_a = reference::<3889>(&polys);
    let want_b = reference::<9721>(&polys);

    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_polys_2q::<3889, 9721>(&polys, &mut a, &mut b);
    same(&want_a, &a, "2q q=3889");
    same(&want_b, &b, "2q q=9721");

    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_polys_2q_pf::<3889, 9721, 2>(&polys, &mut a, &mut b);
    same(&want_a, &a, "2q_pf q=3889");
    same(&want_b, &b, "2q_pf q=9721");

    let mut idx: Vec<BinaryIndex32> = (0..NB).map(|_| BinaryIndex32::zero()).collect();
    io::transpose_polys(&polys, &mut idx);
    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_idx_2q::<3889, 9721>(&idx, &mut a, &mut b);
    same(&want_a, &a, "idx_2q q=3889");
    same(&want_b, &b, "idx_2q q=9721");

    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_polys_2q_pipelined::<3889, 9721>(&polys, &mut a, &mut b);
    same(&want_a, &a, "2q_pipelined q=3889");
    same(&want_b, &b, "2q_pipelined q=9721");

    let mut nib: Vec<BinaryBatch32> = (0..NB).map(|_| BinaryBatch32::zero()).collect();
    io::nibble_polys(&polys, &mut nib);
    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_polys_2q_pipelined::<3889, 9721>(&polys, &mut a, &mut b);
    same(&want_a, &a, "2q_pipelined q=3889");
    same(&want_b, &b, "2q_pipelined q=9721");

    let (mut a, mut b) = (zeros(), zeros());
    io::ntt_bin_nib_2q::<3889, 9721>(&nib, &mut a, &mut b);
    same(&want_a, &a, "nib_2q q=3889");
    same(&want_b, &b, "nib_2q q=9721");
}

/// The streamed accumulating consumer, against the scalar reference.
fn check_accumulate<const Q: u16>() {
    let polys = inputs();
    let mut rng = Rng::new(0xABCD);
    let a: Vec<Batch32> = (0..NB)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(Q as u32) as i16;
                }
            }
            b
        })
        .collect();

    let mut want = [0u64; N];
    for (bi, chunk) in polys.chunks(32).enumerate() {
        for p in 0..32 {
            let w = scalar::ntt::<Q>(&scalar::lift(&chunk[p]));
            for j in 0..N {
                want[j] = (want[j] + w[j] as u64 * a[bi].v[j][p] as u64) % Q as u64;
            }
        }
    }
    for pfd in 0..2 {
        let mut acc = [[0i32; 16]; N];
        if pfd == 0 {
            io::ntt_bin_accumulate::<Q, 0>(&polys, &a, &mut acc);
        } else {
            io::ntt_bin_accumulate::<Q, 2>(&polys, &a, &mut acc);
        }
        let got = io::finish_accumulator::<Q>(&acc);
        for j in 0..N {
            assert_eq!(got[j] as u64, want[j], "accumulate pfd={pfd} slot {j}");
        }
    }
}

#[test]
fn io_entry_points_3889() {
    check::<3889>();
}
#[test]
fn io_entry_points_9721() {
    check::<9721>();
}
#[test]
fn io_two_primes_one_transpose() {
    check_2q();
}
#[test]
fn io_accumulate_3889() {
    check_accumulate::<3889>();
}
#[test]
fn io_accumulate_9721() {
    check_accumulate::<9721>();
}

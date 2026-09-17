use crate::challenge::ShortChallenge;
use crate::fold::{a_times_v_limb, challenge_batches, forward_limb};
use crate::key::CommitmentKey;
use crate::limb::dispatch_limb;
use crate::params::N;
use crate::ring::{Batch32, Representation, RingElement};
use crate::simd::bd as kernel;
use crate::simd::ntt::gen_small::intt_gen_batch32;
use crate::simd::ntt::gen_large as vgl;
use crate::simd::ntt::gen_quad::intt_quad_gen_batch32;
use super::*;

fn inverse_limb(q: u16, batch: &mut Batch32) {
    batch.representation = Representation::Ntt;
    unsafe {
        dispatch_limb!(
            q,
            small |Q| intt_gen_batch32::<Q>(batch),
            large |Q| vgl::intt_gen_batch32::<Q>(batch),
            quad |Q| intt_quad_gen_batch32::<Q>(batch),
        )
    }
}

fn quad_columns(dropped: &Dropped, limb: usize, at: [usize; 4], out: &mut [u64; N]) {
    let digits: Vec<&[u16]> = dropped.digits.iter().map(|d| d.as_slice()).collect();
    let (t, d, p) = (&dropped.top, dropped.dropped_bits, &dropped.primes);
    unsafe {
        dispatch_limb!(dropped.primes[limb], |Q| kernel::residue_quad::<Q>(
            t, &digits, d, p, at, out
        ))
    }
}

fn fold_columns(
    q: u16,
    dropped: &Dropped,
    limb: usize,
    challenges: &[Batch32],
    scratch: &mut [Batch32],
) -> [u32; N] {
    let columns = dropped.columns;
    let mut buf = [0u64; N];
    for (b, batch) in scratch.iter_mut().enumerate() {
        let first = 32 * b;
        let cols = (columns - first).min(32);
        for p in (0..cols).step_by(4) {
            let at = core::array::from_fn(|c| (first + p + c) * N);
            quad_columns(dropped, limb, at, &mut buf);
            for i in 0..N {
                unsafe {
                    (batch.v[i].as_mut_ptr().add(p) as *mut u64).write_unaligned(buf[i]);
                }
            }
        }
        for p in cols..32 {
            for i in 0..N {
                batch.v[i][p] = 0;
            }
        }
    }
    forward_limb(q, scratch);
    a_times_v_limb(q, challenges, scratch)
}

pub fn residual(
    key: &CommitmentKey,
    dropped: &Dropped,
    challenges: &[ShortChallenge],
    folded: &[RingElement],
) -> Option<u128> {
    let limbs = key.limbs();
    let matched = dropped.primes.len() == limbs
        && (0..limbs).all(|k| dropped.primes[k] == key.prime(k))
        && dropped.top.len() == dropped.columns * N
        && dropped
            .digits
            .iter()
            .all(|d| d.len() == dropped.columns * N);
    if !matched || dropped.columns != challenges.len() || folded.len() % 32 != 0 {
        return None;
    }
    let mut batches: Vec<Batch32> = (0..folded.len() / 32)
        .map(|_| Batch32::zero(Representation::Coefficients))
        .collect();
    for (i, e) in folded.iter().enumerate() {
        batches[i / 32].set(i % 32, e);
    }
    let mut columns: Vec<Batch32> = (0..dropped.columns.div_ceil(32))
        .map(|_| Batch32::zero(Representation::Coefficients))
        .collect();
    let mut work: Vec<Batch32> = if limbs > 1 {
        batches.clone()
    } else {
        Vec::new()
    };
    let mut centred = vec![[0i16; N]; limbs];
    for k in 0..limbs {
        let q = key.prime(k);
        let ch = challenge_batches(q, challenges);
        let folded_commitment = fold_columns(q, dropped, k, &ch, &mut columns);
        let v = if k + 1 == limbs {
            &mut batches
        } else {
            for (w, b) in work.iter_mut().zip(&batches) {
                w.v.copy_from_slice(&b.v);
            }
            &mut work
        };
        forward_limb(q, v);
        let y = a_times_v_limb(q, key.row(k), v);
        let half = ((q - 1) / 2) as u32;
        let mut z = Batch32::zero(Representation::Ntt);
        for u in 0..N {
            let x = (y[u] + q as u32 - folded_commitment[u]) % q as u32;
            z.v[u][0] = if x > half {
                (x as i32 - q as i32) as i16
            } else {
                x as i16
            };
        }
        inverse_limb(q, &mut z);
        for i in 0..N {
            centred[k][i] = z.v[i][0];
        }
    }
    let garner = Garner::of(&(0..limbs).map(|k| key.prime(k)).collect::<Vec<u16>>());
    let normsq = garner.normsq(&centred);
    Some(normsq)
}

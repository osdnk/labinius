use crate::limb::dispatch_limb;
use crate::params::{inv_mod, ParamsQ, N, QUAD_CLASS_SLOT, QUAD_POW3_CLASS};
use crate::recursion::limbs::recombination;
use crate::ring::{
    Batch32, PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, Representation,
    VerticallyAlignedMatrix, N162, SLOT_648,
};
use crate::simd::bd as kernel;
use crate::simd::ntt::bin_large as vl;
use crate::simd::ntt::gen_small::intt_gen_batch32;
use crate::simd::ntt::gen_large as vgl;
use crate::simd::ntt::gen_quad::intt_quad_gen_batch32;
use super::*;

struct Inv<const Q: u16>;

impl<const Q: u16> Inv<Q> {
    const LARGE: bool = vl::is_large(Q);
}

fn drain(batch: &Batch32, first: usize, cols: usize, out: &mut [i16]) {
    for p in 0..cols {
        let at = (first + p) * N;
        for i in 0..N {
            out[at + i] = batch.v[i][p];
        }
    }
}

fn columns_split<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    out: &mut [i16],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let e = recombination::<Q>();
    let r = matrix.cols();
    let mut batch = Batch32::zero(Representation::Ntt);
    for first in (0..r).step_by(32) {
        let cols = (r - first).min(32);
        batch.v.iter_mut().for_each(|row| *row = [0i16; 32]);
        batch.representation = Representation::Ntt;
        for p in 0..cols {
            let c: [&PowerOfThreeRingElement; 4] =
                core::array::from_fn(|k| &matrix.get(k, first + p).limbs[limb]);
            for s in 0..N162 {
                let y: [i32; 4] = core::array::from_fn(|k| c[k].v[s] as i32);
                for t in 0..4 {
                    let g = &e[(s * 4 + t) * 4..(s * 4 + t) * 4 + 4];
                    let acc = (0..4)
                        .map(|k| g[k] as i32 * y[k])
                        .sum::<i32>()
                        .rem_euclid(q);
                    batch.v[SLOT_648[t][s] as usize][p] = if acc > half {
                        (acc - q) as i16
                    } else {
                        acc as i16
                    };
                }
            }
        }
        unsafe {
            if Inv::<Q>::LARGE {
                vgl::intt_gen_batch32::<Q>(&mut batch);
            } else {
                intt_gen_batch32::<Q>(&mut batch);
            }
        }
        drain(&batch, first, cols, out);
    }
}

fn columns_quad<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    out: &mut [i16],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let pv: [i32; N162] =
        core::array::from_fn(|s| ParamsQ::<Q>::psi_pow(QUAD_POW3_CLASS[s] as u32) as i32);
    let r = matrix.cols();
    let mut batch = Batch32::zero(Representation::Ntt);
    for first in (0..r).step_by(32) {
        let cols = (r - first).min(32);
        batch.v.iter_mut().for_each(|row| *row = [0i16; 32]);
        batch.representation = Representation::Ntt;
        for p in 0..cols {
            let c: [&PowerOfThreeRingElement; 4] =
                core::array::from_fn(|k| &matrix.get(k, first + p).limbs[limb]);
            for s in 0..N162 {
                let (jp, jm) = (
                    QUAD_CLASS_SLOT[0][s] as usize,
                    QUAD_CLASS_SLOT[1][s] as usize,
                );
                for k in 0..2 {
                    let y0 = (c[k].v[s] as i32).rem_euclid(q);
                    let y2 = (pv[s] as i64 * (c[k + 2].v[s] as i32).rem_euclid(q) as i64 % q as i64)
                        as i32;
                    let plus = (y0 + y2) % q;
                    let minus = (y0 + q - y2) % q;
                    batch.v[2 * jp + k][p] = if plus > half {
                        (plus - q) as i16
                    } else {
                        plus as i16
                    };
                    batch.v[2 * jm + k][p] = if minus > half {
                        (minus - q) as i16
                    } else {
                        minus as i16
                    };
                }
            }
        }
        unsafe { intt_quad_gen_batch32::<Q>(&mut batch) };
        drain(&batch, first, cols, out);
    }
}

pub fn column_coefficients(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    primes: &[u16],
) -> Vec<Vec<i16>> {
    let r = matrix.cols();
    primes
        .iter()
        .enumerate()
        .map(|(limb, &q)| {
            let mut out = vec![0i16; r * N];
            dispatch_limb!(
                q,
                split |Q| columns_split::<Q>(matrix, limb, &mut out),
                quad |Q| columns_quad::<Q>(matrix, limb, &mut out),
            );
            out
        })
        .collect()
}

pub fn drop_bits(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    primes: &[u16],
    dropped_bits: u32,
) -> Dropped {
    let residues = column_coefficients(matrix, primes);
    let columns = matrix.cols();
    let count = columns * N;
    let limbs = primes.len();
    let inverses: Vec<Vec<u32>> = (0..limbs)
        .map(|k| {
            (0..k)
                .map(|m| inv_mod(primes[m] as u64, primes[k] as u64) as u32)
                .collect()
        })
        .collect();
    let mut top = vec![0u16; count];
    let mut digits: Vec<Vec<u16>> = (1..limbs).map(|_| vec![0u16; count]).collect();
    let round = 1u32 << (dropped_bits - 1);
    let mut mixed = vec![0u32; limbs];
    for i in 0..count {
        let base = (residues[0][i] as i32).rem_euclid(primes[0] as i32) as u32;
        top[i] = ((base + round) >> dropped_bits) as u16;
        mixed[0] = base;
        for k in 1..limbs {
            let q = primes[k] as u64;
            let mut x = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
            for m in 0..k {
                x = (x + q - mixed[m] as u64 % q) % q * inverses[k][m] as u64 % q;
            }
            mixed[k] = x as u32;
            digits[k - 1][i] = x as u16;
        }
    }
    Dropped {
        primes: primes.to_vec(),
        columns,
        dropped_bits,
        top,
        digits,
    }
}

pub fn limb_residues(dropped: &Dropped, limb: usize, out: &mut [i16]) {
    let digits: Vec<&[u16]> = dropped.digits.iter().map(|d| d.as_slice()).collect();
    let q = dropped.primes[limb];
    unsafe {
        dispatch_limb!(q, |Q| kernel::residues::<Q>(
            &dropped.top,
            &digits,
            dropped.dropped_bits,
            &dropped.primes,
            out,
        ))
    }
}

pub struct Garner {
    primes: Vec<u64>,
    magic: Vec<u64>,
    inverses: Vec<Vec<u64>>,
    modulus: u64,
}

impl Garner {
    pub fn of(primes: &[u16]) -> Garner {
        let primes: Vec<u64> = primes.iter().map(|&q| q as u64).collect();
        Garner {
            magic: primes.iter().map(|&q| (1u64 << 43) / q).collect(),
            inverses: (0..primes.len())
                .map(|k| (0..k).map(|m| inv_mod(primes[m], primes[k])).collect())
                .collect(),
            modulus: primes.iter().product(),
            primes,
        }
    }

    #[inline(always)]
    fn reduce(&self, k: usize, x: u64) -> u64 {
        let q = self.primes[k];
        let r = x - (x * self.magic[k] >> 43) * q;
        if r >= q {
            r - q
        } else {
            r
        }
    }

    pub(super) fn normsq(&self, centred: &[[i16; N]]) -> u128 {
        let limbs = self.primes.len();
        let mut mixed = vec![0u64; limbs];
        let mut normsq = 0u128;
        for i in 0..N {
            for k in 0..limbs {
                let q = self.primes[k];
                let x = centred[k][i] as i64;
                let mut x = if x < 0 {
                    (x + q as i64) as u64
                } else {
                    x as u64
                };
                for m in 0..k {
                    x = self.reduce(k, (x + q - self.reduce(k, mixed[m])) * self.inverses[k][m]);
                }
                mixed[k] = x;
            }
            let mut value = 0u64;
            for k in (0..limbs).rev() {
                value = value * self.primes[k] + mixed[k];
            }
            let magnitude = if value > self.modulus / 2 {
                self.modulus - value
            } else {
                value
            } as u128;
            normsq += magnitude * magnitude;
        }
        normsq
    }
}

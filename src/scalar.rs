//! Exact scalar reference: lift, schoolbook multiplication modulo X^648 - X^324 + 1, the NTT in
//! the canonical tree order, and pointwise products. Everything is fully reduced u32 in [0, q).
//! Not optimised; it exists to define correctness for the SIMD kernels.
use crate::params::*;

pub type Coeffs = [u32; N];

pub fn lift(p: &crate::types::BinaryPoly) -> Coeffs {
    p.to_coeffs()
}

/// a * b in Z_q[X]/(X^648 - X^324 + 1), using X^648 = X^324 - 1.
pub fn mul_mod_phi(a: &Coeffs, b: &Coeffs, q: u16) -> Coeffs {
    let q = q as u64;
    let mut c = [0u64; 2 * N];
    for i in 0..N {
        if a[i] == 0 {
            continue;
        }
        let ai = a[i] as u64;
        for j in 0..N {
            c[i + j] += ai * b[j] as u64;
        }
    }
    for k in (N..2 * N).rev() {
        let v = c[k] % q;
        c[k] = 0;
        c[k - 324] += v;
        c[k - 648] += q * q - v; // q*q is a multiple of q, keeps the sum non-negative
    }
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = (c[i] % q) as u32;
    }
    out
}

/// Forward NTT in tree order (see `params`): out[j] = a(psi^SLOT_EXP[j]).
pub fn ntt<const Q: u16>(a: &Coeffs) -> Coeffs {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    let w = Params::<Q>::OMEGA as u64;
    let w2 = w * w % q;
    let mut v = [0u64; N];
    for i in 0..N {
        v[i] = a[i] as u64 % q;
    }
    // level 0: Phi_6 split, children X^324 - zeta6 and X^324 - zeta6^-1 with zeta6^-1 = 1 - zeta6.
    let z6 = Params::<Q>::ZETA6 as u64;
    for i in 0..324 {
        let (a0, a1) = (v[i], v[i + 324]);
        let t = a1 * z6 % q;
        v[i] = (a0 + t) % q;
        v[i + 324] = (a0 + a1 + q - t) % q;
    }
    for level in 1..=6 {
        let n = DEGREE[level];
        let p = RADIX[level];
        let m = n / p;
        for k in 0..SUBRINGS[level] {
            let base = k * n;
            let zeta = pow_mod(psi, twiddle_exp(level, k) as u64, q);
            if p == 2 {
                for i in 0..m {
                    let (a0, a1) = (v[base + i], v[base + m + i]);
                    let t = a1 * zeta % q;
                    v[base + i] = (a0 + t) % q;
                    v[base + m + i] = (a0 + q - t) % q;
                }
            } else {
                let zeta2 = zeta * zeta % q;
                for i in 0..m {
                    let a0 = v[base + i];
                    let t1 = v[base + m + i] * zeta % q;
                    let t2 = v[base + 2 * m + i] * zeta2 % q;
                    v[base + i] = (a0 + t1 + t2) % q;
                    v[base + m + i] = (a0 + w * t1 + w2 * t2) % q;
                    v[base + 2 * m + i] = (a0 + w2 * t1 + w * t2) % q;
                }
            }
        }
    }
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = v[i] as u32;
    }
    out
}

/// Direct evaluation a(psi^u) by Horner's rule (independent check of `ntt`).
pub fn eval_at<const Q: u16>(a: &Coeffs, u: u32) -> u32 {
    let q = Q as u64;
    let x = pow_mod(Params::<Q>::PSI as u64, u as u64, q);
    let mut acc = 0u64;
    for i in (0..N).rev() {
        acc = (acc * x + a[i] as u64) % q;
    }
    acc as u32
}

pub fn pointwise_mul(a: &Coeffs, b: &Coeffs, q: u16) -> Coeffs {
    let mut c = [0u32; N];
    for j in 0..N {
        c[j] = (a[j] as u64 * b[j] as u64 % q as u64) as u32;
    }
    c
}

pub fn normalize_i16(v: &[i16; N], q: u16) -> Coeffs {
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = (v[i] as i32).rem_euclid(q as i32) as u32;
    }
    out
}

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

/// Exact inverse of [`ntt`] in the same tree order, with `scale` applied to every coefficient of
/// the result: `intt_scaled(&ntt(a), s)[i] = s * a[i] mod q`.
///
/// Every butterfly is inverted together with its own normalisation — 1/2 per radix-2 level
/// (1, 2), 1/3 per radix-3 level (3..6) and 1/(2*zeta6 - 1) for the determinant of the Phi_6
/// split of level 0 — so the constant folded into the whole pass is
/// `1 / (4 * 81 * (2*zeta6 - 1)) = 2 / (648 * (2*zeta6 - 1))`: the familiar 1/648 corrected by
/// the Phi_6 determinant, which satisfies `(2*zeta6 - 1)^2 = -3`.
///
/// This is the only inverse transform in the crate; it exists so that the round trip of the SIMD
/// forward kernels is testable, and so that the `R^-1` a Montgomery-form transform leaves behind
/// has somewhere to go ([`intt_mont`]).
pub fn intt_scaled<const Q: u16>(v: &Coeffs, scale: u32) -> Coeffs {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    let w = Params::<Q>::OMEGA as u64;
    let w2 = w * w % q;
    let inv2 = inv_mod(2, q);
    let inv3 = inv_mod(3, q);
    let mut u = [0u64; N];
    for i in 0..N {
        u[i] = v[i] as u64 % q;
    }
    for level in (1..=6).rev() {
        let n = DEGREE[level];
        let p = RADIX[level];
        let m = n / p;
        for k in 0..SUBRINGS[level] {
            let base = k * n;
            let zi = inv_mod(pow_mod(psi, twiddle_exp(level, k) as u64, q), q);
            if p == 2 {
                for i in 0..m {
                    let (y0, y1) = (u[base + i], u[base + m + i]);
                    u[base + i] = (y0 + y1) % q * inv2 % q;
                    u[base + m + i] = (y0 + q - y1) % q * inv2 % q * zi % q;
                }
            } else {
                let zi2 = zi * zi % q;
                for i in 0..m {
                    let (y0, y1, y2) = (u[base + i], u[base + m + i], u[base + 2 * m + i]);
                    let a0 = (y0 + y1 + y2) % q * inv3 % q;
                    let t1 = (y0 + w2 * y1 + w * y2) % q * inv3 % q;
                    let t2 = (y0 + w * y1 + w2 * y2) % q * inv3 % q;
                    u[base + i] = a0;
                    u[base + m + i] = t1 * zi % q;
                    u[base + 2 * m + i] = t2 * zi2 % q;
                }
            }
        }
    }
    // level 0: y0 = a0 + zeta6 a1, y1 = a0 + zeta6^-1 a1 (zeta6^-1 = 1 - zeta6), so
    // a1 = (y0 - y1) / (2 zeta6 - 1) and a0 = y0 - zeta6 a1.
    let z6 = Params::<Q>::ZETA6 as u64;
    let det = inv_mod((2 * z6 + q - 1) % q, q);
    let s = scale as u64 % q;
    for i in 0..324 {
        let (y0, y1) = (u[i], u[i + 324]);
        let a1 = (y0 + q - y1) % q * det % q;
        let a0 = (y0 + q - z6 * a1 % q) % q;
        u[i] = a0 * s % q;
        u[i + 324] = a1 * s % q;
    }
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = u[i] as u32;
    }
    out
}

/// Exact inverse of [`ntt`]: `intt(&ntt(a)) == a` for any fully reduced `a`.
pub fn intt<const Q: u16>(v: &Coeffs) -> Coeffs {
    intt_scaled::<Q>(v, 1)
}

/// Inverse of a Montgomery-form transform (`v[j] = R * a(psi^SLOT_EXP[j])`, R = 2^16 mod q): the
/// `R^-1` is folded into the same final scaling as the 1/648, so it is free for the caller.
/// A slot-wise product of two such transforms is again Montgomery form (one factor R), so this is
/// also the right inverse for `pointwise::mul_batch_batch_mont` output.
pub fn intt_mont<const Q: u16>(v: &Coeffs) -> Coeffs {
    intt_scaled::<Q>(v, Params::<Q>::RINV as u32)
}

/// Inverse of a transform carrying `k` Montgomery factors (`v[j] = R^k * a(psi^u_j)`).
pub fn intt_mont_pow<const Q: u16>(v: &Coeffs, k: u32) -> Coeffs {
    intt_scaled::<Q>(v, pow_mod(Params::<Q>::RINV as u64, k as u64, Q as u64) as u32)
}

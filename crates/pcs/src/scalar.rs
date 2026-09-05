//! Exact scalar reference: lift, schoolbook multiplication modulo X^648 - X^324 + 1, the NTT in
//! the canonical tree order, and pointwise products. Everything is fully reduced u32 in [0, q).
//! Not optimised; it exists to define correctness for the SIMD kernels.
use crate::params::*;

pub type Coeffs = [u32; N];

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

/// Exact inverse of [`ntt`], in the same tree order.
///
/// Every butterfly is inverted together with its own normalisation — 1/2 per radix-2 level
/// (1, 2), 1/3 per radix-3 level (3..6) and 1/(2*zeta6 - 1) for the determinant of the Phi_6
/// split of level 0 — so the constant folded into the whole pass is
/// `1 / (4 * 81 * (2*zeta6 - 1)) = 2 / (648 * (2*zeta6 - 1))`: the familiar 1/648 corrected by
/// the Phi_6 determinant, which satisfies `(2*zeta6 - 1)^2 = -3`.
///
/// This is the only inverse transform in the crate; it exists so that the round trip of the SIMD
/// forward kernels is testable. `intt(&ntt(a)) == a` for any fully reduced `a`.
pub fn intt<const Q: u16>(v: &Coeffs) -> Coeffs {
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
    for i in 0..324 {
        let (y0, y1) = (u[i], u[i + 324]);
        let a1 = (y0 + q - y1) % q * det % q;
        let a0 = (y0 + q - z6 * a1 % q) % q;
        u[i] = a0;
        u[i + 324] = a1;
    }
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = u[i] as u32;
    }
    out
}

/// Forward NTT on the quadratic-slot tree (see `params`): the 324 leaves
/// `Z_q[X]/(X^2 - psi'^QUAD_SLOT_EXP[j])` in tree order, leaf j in rows `2j` (constant term) and
/// `2j+1` (X coefficient) — `out[2j] + out[2j+1] X = a mod (X^2 - psi'^u_j)`.
///
/// Levels: the Phi_6 split, one radix-2 level, then four radix-3 levels 162 -> 54 -> 18 -> 6 -> 2.
/// The butterflies are exactly [`ntt`]'s; only the tree and the root of unity differ.
pub fn ntt_quad<const Q: u16>(a: &Coeffs) -> Coeffs {
    let q = Q as u64;
    let psi = ParamsQ::<Q>::PSI972 as u64;
    let w = ParamsQ::<Q>::OMEGA as u64;
    let w2 = w * w % q;
    let mut v = [0u64; N];
    for i in 0..N {
        v[i] = a[i] as u64 % q;
    }
    // level 0: Phi_6 split, children X^324 - zeta6 and X^324 - zeta6^-1 with zeta6^-1 = 1 - zeta6.
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    for i in 0..324 {
        let (a0, a1) = (v[i], v[i + 324]);
        let t = a1 * z6 % q;
        v[i] = (a0 + t) % q;
        v[i + 324] = (a0 + a1 + q - t) % q;
    }
    for level in 1..=5 {
        let n = DEGREE_Q[level];
        let p = RADIX_Q[level];
        let m = n / p;
        for k in 0..SUBRINGS_Q[level] {
            let base = k * n;
            let zeta = pow_mod(psi, twiddle_exp_quad(level, k) as u64, q);
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

/// `a mod (X^2 - psi'^u) = (r0, r1)` by Horner's rule in X, an independent check of [`ntt_quad`].
pub fn eval_quad_at<const Q: u16>(a: &Coeffs, u: u32) -> (u32, u32) {
    let q = Q as u64;
    let c = pow_mod(ParamsQ::<Q>::PSI972 as u64, u as u64, q);
    let (mut r0, mut r1) = (0u64, 0u64);
    for i in (0..N).rev() {
        // r <- r * X + a[i]  in Z_q[X]/(X^2 - c)
        let (n0, n1) = ((r1 * c + a[i] as u64) % q, r0);
        r0 = n0;
        r1 = n1;
    }
    (r0 as u32, r1 as u32)
}

/// Slot-wise product of two quadratic-slot transforms: for every leaf j,
/// `(a_0 + a_1 X)(b_0 + b_1 X) mod (X^2 - c_j) = (a_0 b_0 + c_j a_1 b_1) + (a_0 b_1 + a_1 b_0) X`.
///
/// `ntt_quad(a * b mod Phi_1944) == mul_quad_slots(ntt_quad(a), ntt_quad(b))`.
pub fn mul_quad_slots<const Q: u16>(a: &Coeffs, b: &Coeffs) -> Coeffs {
    let q = Q as u64;
    let mut c = [0u32; N];
    for j in 0..QUAD_SLOTS {
        let cj = ParamsQ::<Q>::LEAF_C[j] as u64;
        let (a0, a1) = (a[2 * j] as u64 % q, a[2 * j + 1] as u64 % q);
        let (b0, b1) = (b[2 * j] as u64 % q, b[2 * j + 1] as u64 % q);
        c[2 * j] = ((a0 * b0 + cj * (a1 * b1 % q)) % q) as u32;
        c[2 * j + 1] = ((a0 * b1 + a1 * b0) % q) as u32;
    }
    c
}

/// The quadratic-slot analogue of `api::decompose_648_to_4x162`: the four `R_162` components of a
/// ring element, read off its quadratic-slot transform.
///
/// With `Y = X^4` an element is `y = y_0(Y) + X y_1(Y) + X^2 y_2(Y) + X^3 y_3(Y)`, the `y_k` in
/// `R_162 = Z_q[Y]/(Y^162 - Y^81 + 1)`. A leaf `X^2 = c` has `Y = c^2`, so
///
/// ```text
///     y mod (X^2 - c) = [Y_0 + c Y_2] + X [Y_1 + c Y_3],       Y_k = y_k(c^2).
/// ```
///
/// The two leaves of one class `v` are `c = +psi'^v` and `c = -psi'^v` ([`QUAD_CLASS_SLOT`]) and
/// both give `c^2 = psi'^{2v} = theta^v`, so the class is inverted by a 2-point butterfly
///
/// ```text
///     Y_0 = (E^+_0 + E^-_0)/2,   Y_2 = (E^+_0 - E^-_0)/(2 psi'^v),
///     Y_1 = (E^+_1 + E^-_1)/2,   Y_3 = (E^+_1 - E^-_1)/(2 psi'^v),
/// ```
///
/// `E^+` the plus leaf's two rows, `E^-` the minus leaf's. The output is in the `R_162` slot order
/// of [`crate::ring::POW3_SLOT_EXP`]: `out[k][s] = y_k(theta^{v_s})`, fully reduced in `[0, q)`.
pub fn decompose_quad_648_to_4x162<const Q: u16>(y: &Coeffs) -> [[u32; 162]; 4] {
    let q = Q as u64;
    let inv2 = inv_mod(2, q);
    let mut out = [[0u32; 162]; 4];
    for s in 0..162 {
        let v = QUAD_POW3_CLASS[s] as u64;
        let jp = QUAD_CLASS_SLOT[0][s] as usize;
        let jm = QUAD_CLASS_SLOT[1][s] as usize;
        // psi'^{-v} = psi'^{972 - v}
        let ipv = pow_mod(
            ParamsQ::<Q>::PSI972 as u64,
            (CONDUCTOR_QUAD as u64 - v) % CONDUCTOR_QUAD as u64,
            q,
        );
        for k in 0..2 {
            let ep = y[2 * jp + k] as u64 % q;
            let em = y[2 * jm + k] as u64 % q;
            out[k][s] = ((ep + em) % q * inv2 % q) as u32;
            out[k + 2][s] = ((ep + q - em) % q * inv2 % q * ipv % q) as u32;
        }
    }
    out
}

//! Ring parameters for R_q = Z_q[X] / Phi_1944(X), Phi_1944(X) = X^648 - X^324 + 1.
//!
//! Everything here is `const`-evaluated. The NTT tree is fixed once and for all so that every
//! implementation (scalar reference and all SIMD variants) produces the same slot order:
//!
//! level 0: the Phi_6 split   X^648 - X^324 + 1 = (X^324 - psi^324)(X^324 - psi^1620)
//! level 1: radix 2           X^324 - psi^e = (X^162 - psi^{e/2})(X^162 - psi^{(e+1944)/2})
//! level 2: radix 2           X^162 -> X^81
//! level 3..6: radix 3        X^81 -> X^27 -> X^9 -> X^3 -> X^1
//!
//! A sub-ring is Z_q[X]/(X^n - psi^e) with n | e; radix-p split into children
//! X^{n/p} - psi^{(e + 1944 s)/p}, s = 0..p-1, child s stored at block offset s*n/p. The leaf in
//! slot j is therefore Z_q[X]/(X - psi^SLOT_EXP[j]) and NTT(a)[j] = a(psi^SLOT_EXP[j]).
//!
//! psi is the smallest primitive 1944-th root of unity mod q; omega = psi^648 (cube root of 1),
//! zeta6 = psi^324 (sixth root of 1, zeta6^-1 = 1 - zeta6).

/// Degree of the ring.
pub const N: usize = 648;
/// Conductor.
pub const CONDUCTOR: u32 = 1944;
/// The supported primes.
pub const QS: [u16; 2] = [3889, 9721];
/// Radix of the split that turns level `l` into level `l+1` (level 0 is the whole ring).
pub const RADIX: [usize; 7] = [2, 2, 2, 3, 3, 3, 3];
/// Number of sub-rings at level `l` (level 7 = the 648 leaves).
pub const SUBRINGS: [usize; 8] = [1, 2, 4, 8, 24, 72, 216, 648];
/// Degree of one sub-ring at level `l`.
pub const DEGREE: [usize; 8] = [648, 324, 162, 81, 27, 9, 3, 1];

pub const fn pow_mod(mut b: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % q;
        }
        b = b * b % q;
        e >>= 1;
    }
    r
}

pub const fn inv_mod(a: u64, q: u64) -> u64 {
    pow_mod(a, q - 2, q)
}

/// Smallest x in [2, q) whose multiplicative order is exactly 1944.
pub const fn find_psi(q: u64) -> u64 {
    let mut x = 2u64;
    loop {
        if pow_mod(x, 1944, q) == 1 && pow_mod(x, 972, q) != 1 && pow_mod(x, 648, q) != 1 {
            return x;
        }
        x += 1;
    }
}

/// Exponent e of the sub-ring k at level `level` (1..=7): that sub-ring is Z_q[X]/(X^n - psi^e).
pub const fn subring_exp(level: usize, k: usize) -> u32 {
    if level == 1 {
        return if k == 0 { 324 } else { 1620 };
    }
    let p = RADIX[level - 1] as u32;
    (subring_exp(level - 1, k / p as usize) + CONDUCTOR * (k as u32 % p)) / p
}

/// psi-exponent of the twiddle used when splitting sub-ring k of level `level` (1..=6):
/// zeta = psi^(e/p); the p children are X^{n/p} - zeta * rho_p^s.
pub const fn twiddle_exp(level: usize, k: usize) -> u32 {
    subring_exp(level, k) / RADIX[level] as u32
}

const fn slot_exp_table() -> [u16; N] {
    let mut t = [0u16; N];
    let mut j = 0;
    while j < N {
        t[j] = subring_exp(7, j) as u16;
        j += 1;
    }
    t
}

/// `SLOT_EXP[j]` = u such that slot j of the NTT holds a(psi^u).
pub const SLOT_EXP: [u16; N] = slot_exp_table();

/// q^-1 mod 2^16 (Newton iteration; q odd).
pub const fn qinv16(q: u16) -> u16 {
    let mut x = q;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u16.wrapping_sub(q.wrapping_mul(x)));
        i += 1;
    }
    x
}

/// Centered representative in (-q/2, q/2].
pub const fn center(x: u64, q: u64) -> i16 {
    let x = x % q;
    if x > q / 2 {
        (x as i64 - q as i64) as i16
    } else {
        x as i16
    }
}

/// Per-prime constants. `Params::<3889>::PSI` etc.
pub struct Params<const Q: u16>;

impl<const Q: u16> Params<Q> {
    pub const Q: u16 = Q;
    pub const Q64: u64 = Q as u64;
    /// q^-1 mod 2^16.
    pub const QINV: u16 = qinv16(Q);
    /// Smallest primitive 1944-th root of unity.
    pub const PSI: u16 = find_psi(Q as u64) as u16;
    /// Primitive cube root of unity, omega = psi^648.
    pub const OMEGA: u16 = pow_mod(Self::PSI as u64, 648, Q as u64) as u16;
    /// Primitive sixth root of unity, zeta6 = psi^324 (level-0 twiddle); zeta6^-1 = 1 - zeta6.
    pub const ZETA6: u16 = pow_mod(Self::PSI as u64, 324, Q as u64) as u16;
    /// 2^16 mod q and 2^32 mod q (Montgomery constants).
    pub const R: u16 = (65536u64 % Q as u64) as u16;
    pub const R2: u16 = (65536u64 * 65536u64 % Q as u64) as u16;
    /// 2^-16 mod q: the factor that has to be undone once per Montgomery power carried by a
    /// value (`R * RINV = 1 mod q`). A transform whose outputs are in Montgomery form leaves it
    /// to whoever leaves the NTT domain (`scalar::intt_mont`).
    pub const RINV: u16 = inv_mod(Self::R as u64, Q as u64) as u16;
    /// 648^-1 mod q, the degree part of an inverse NTT's normalisation.
    pub const N_INV: u16 = inv_mod(N as u64, Q as u64) as u16;

    /// psi^e mod q.
    pub const fn psi_pow(e: u32) -> u16 {
        pow_mod(Self::PSI as u64, e as u64, Q as u64) as u16
    }
    /// Plain twiddle zeta for sub-ring k at level `level` (1..=6).
    pub const fn zeta(level: usize, k: usize) -> u16 {
        Self::psi_pow(twiddle_exp(level, k))
    }
    /// x * 2^16 mod q, centered: the Montgomery form used by the SIMD kernels.
    pub const fn to_mont(x: u16) -> i16 {
        center(x as u64 * 65536u64, Q as u64)
    }
    /// x * R mod q (plain, not centered): the scaling applied to a kernel's tables or twiddles to
    /// make its outputs come out in Montgomery form.
    pub const fn scale_r(x: u16) -> u16 {
        (x as u64 * Self::R as u64 % Q as u64) as u16
    }
    /// For a Montgomery-form constant w, the precomputed w * q^-1 mod 2^16 (signed), so that
    /// mont_mul(a, w, w') = a * x mod q needs only mullo/mulhi/mulhi.
    pub const fn mont_pre(w: i16) -> i16 {
        w.wrapping_mul(Self::QINV as i16)
    }
    /// Table of plain twiddles for a whole level (K = SUBRINGS[level]).
    pub const fn zetas<const K: usize>(level: usize) -> [u16; K] {
        let mut t = [0u16; K];
        let mut k = 0;
        while k < K {
            t[k] = Self::zeta(level, k);
            k += 1;
        }
        t
    }
    pub const ZETA_L1: [u16; 2] = Self::zetas::<2>(1);
    pub const ZETA_L2: [u16; 4] = Self::zetas::<4>(2);
    pub const ZETA_L3: [u16; 8] = Self::zetas::<8>(3);
    pub const ZETA_L4: [u16; 24] = Self::zetas::<24>(4);
    pub const ZETA_L5: [u16; 72] = Self::zetas::<72>(5);
    pub const ZETA_L6: [u16; 216] = Self::zetas::<216>(6);
}

/// Signed Montgomery multiplication on 16-bit values, exactly what the SIMD kernels do lane-wise:
/// returns a * x mod q in (-q, q) where w = to_mont(x), w_pre = mont_pre(w). Requires only that
/// `a` is any i16.
#[inline]
pub fn mont_mul_i16(a: i16, w: i16, w_pre: i16, q: u16) -> i16 {
    let m = a.wrapping_mul(w_pre);
    let hi = ((a as i32 * w as i32) >> 16) as i16;
    let t = ((m as i32 * q as i16 as i32) >> 16) as i16;
    hi.wrapping_sub(t)
}

/// Cheap partial reduction with `vpmulhrsw` semantics (2 multiply uops):
/// t = round(a * BARRETT_V / 2^15), r = a - t*q. Because BARRETT_V = round(2^15/q) has only a few
/// significant bits the quotient estimate is off by a few percent, so the guarantee is only
/// |r| < q: exhaustively over all i16 inputs, max |r| = 3497 = 0.899q for q = 3889 and
/// 7864 = 0.809q for q = 9721.
#[inline]
pub fn barrett_i16(a: i16, q: u16) -> i16 {
    let v = barrett_v(q) as i32;
    let t = (((a as i32) * v * 2 + (1 << 15)) >> 16) as i16;
    a.wrapping_sub(t.wrapping_mul(q as i16))
}

/// round(2^15 / q), the `vpmulhrsw` constant of `barrett_i16`.
pub const fn barrett_v(q: u16) -> i16 {
    (((1u32 << 15) + (q as u32) / 2) / q as u32) as i16
}

/// Shift s used by `red16_i16`: the largest s with round(2^(16+s)/q) < 2^15.
pub const fn red16_shift(q: u16) -> u32 {
    let mut s = 0;
    while ((1u64 << (17 + s)) + (q as u64) / 2) / (q as u64) < (1 << 15) {
        s += 1;
    }
    s
}

/// round(2^(16+s) / q), the `vpmulhw` constant of `red16_i16`.
pub const fn red16_v(q: u16) -> i16 {
    (((1u64 << (16 + red16_shift(q))) + (q as u64) / 2) / (q as u64)) as i16
}

/// Kyber-style floor reduction (vpmulhw, vpsraw, vpmullw, vpsubw: 3 multiply-port uops):
/// t = floor(a * V / 2^16) >> s = floor(a / q) (error at most 1 downwards), r = a - t*q in [0, q].
/// Costs one more p0 uop than `barrett_i16` for the same |r| <= q guarantee, so the kernels
/// prefer `barrett_i16`; kept for completeness / non-negative outputs.
#[inline]
pub fn red16_i16(a: i16, q: u16) -> i16 {
    let t = ((((a as i32) * red16_v(q) as i32) >> 16) >> red16_shift(q)) as i16;
    a.wrapping_sub(t.wrapping_mul(q as i16))
}

// =============================================================================================
// The quadratic-slot tree: q = 1 mod 972 but not mod 1944
// =============================================================================================
//
// For q in [`QS_QUAD`] the group Z_q^* has order 2^2 * 3^5 * k, so a primitive 972-nd root of
// unity psi' exists but a 1944-th one does not: Phi_1944 splits into 324 irreducible quadratics
// instead of 648 linear factors. A quadratic factor is X^2 - psi'^u for a unit u mod 972 — its
// two roots would be psi^u and psi^{u+972} = -psi^u if psi = sqrt(psi') existed — and every unit
// appears exactly once.
//
// The tree keeps the shape of the splitting one with the *second* radix-2 level removed:
//
//     level 0: the Phi_6 split   X^648 - X^324 + 1 = (X^324 - psi'^162)(X^324 - psi'^810)
//     level 1: radix 2           X^324 - psi'^e = (X^162 - psi'^{e/2})(X^162 + psi'^{e/2})
//     level 2..5: radix 3        X^n - psi'^e = prod_s (X^{n/3} - psi'^{(e + 972 s)/3})
//                                162 -> 54 -> 18 -> 6 -> 2
//
// A sub-ring is Z_q[X]/(X^n - psi'^e) with (n/2) | e (the splitting tree's n | e, halved because
// the leaves are quadratic); child s of a radix-p split sits at block offset s*n/p, exactly as in
// the splitting tree. Leaf j is therefore Z_q[X]/(X^2 - psi'^`QUAD_SLOT_EXP[j]`) and holds
// `a mod (X^2 - c_j) = a_0 + a_1 X` in rows `2j` and `2j+1` of the 648-row output.
//
// Multiply count: 216 radix-3 butterflies per level and 4 radix-3 levels below the Phi_6 split,
// against the splitting tree's 4 radix-3 levels below one more radix-2 level — the same 6480
// multiply-port uops per batch of 32 for the binary kernel, 3240 Montgomery products against
// 3564 for the generic one.

/// Conductor of the root of unity the quadratic-slot tree uses: psi' is a primitive 972-nd root.
pub const CONDUCTOR_QUAD: u32 = 972;
/// The primes for which `R_648` does *not* split completely: q = 1 mod 972, q = 973 mod 1944.
pub const QS_QUAD: [u16; 3] = [2917, 4861, 12637];
/// Number of quadratic leaves.
pub const QUAD_SLOTS: usize = 324;
/// Radix of the split that turns level `l` into level `l+1` (level 0 is the whole ring).
pub const RADIX_Q: [usize; 6] = [2, 2, 3, 3, 3, 3];
/// Number of sub-rings at level `l` (level 6 = the 324 quadratic leaves).
pub const SUBRINGS_Q: [usize; 7] = [1, 2, 4, 12, 36, 108, 324];
/// Degree of one sub-ring at level `l`.
pub const DEGREE_Q: [usize; 7] = [648, 324, 162, 54, 18, 6, 2];

/// Smallest x in [2, q) whose multiplicative order is exactly 972 (972 = 2^2 * 3^5, so the
/// maximal proper divisors are 486 and 324).
pub const fn find_psi972(q: u64) -> u64 {
    let mut x = 2u64;
    loop {
        if pow_mod(x, 972, q) == 1 && pow_mod(x, 486, q) != 1 && pow_mod(x, 324, q) != 1 {
            return x;
        }
        x += 1;
    }
}

/// Exponent e of sub-ring k at level `level` (1..=6) of the quadratic-slot tree: that sub-ring is
/// `Z_q[X]/(X^n - psi'^e)`, n = `DEGREE_Q[level]`. The recursion is `subring_exp`'s with the
/// conductor 972 and the tree `RADIX_Q`; level 1 is the Phi_6 split, whose children are
/// `X^324 - zeta6` and `X^324 - zeta6^-1` with zeta6 = psi'^162.
pub const fn subring_exp_quad(level: usize, k: usize) -> u32 {
    if level == 1 {
        return if k == 0 { 162 } else { 810 };
    }
    let p = RADIX_Q[level - 1] as u32;
    (subring_exp_quad(level - 1, k / p as usize) + CONDUCTOR_QUAD * (k as u32 % p)) / p
}

/// psi'-exponent of the twiddle used when splitting sub-ring k of level `level` (1..=5):
/// zeta = psi'^(e/p); the p children are `X^{n/p} - zeta * rho_p^s` with rho_2 = -1 = psi'^486
/// and rho_3 = omega = psi'^324.
pub const fn twiddle_exp_quad(level: usize, k: usize) -> u32 {
    subring_exp_quad(level, k) / RADIX_Q[level] as u32
}

const fn quad_slot_exp_table() -> [u16; QUAD_SLOTS] {
    let mut t = [0u16; QUAD_SLOTS];
    let mut j = 0;
    while j < QUAD_SLOTS {
        t[j] = subring_exp_quad(6, j) as u16;
        j += 1;
    }
    t
}

/// `QUAD_SLOT_EXP[j]` = u such that leaf j is `Z_q[X]/(X^2 - psi'^u)`; a permutation of the 324
/// units mod 972. Rows `2j` and `2j+1` of a transform hold `a mod (X^2 - psi'^u) = a_0 + a_1 X`.
pub const QUAD_SLOT_EXP: [u16; QUAD_SLOTS] = quad_slot_exp_table();

const _: () = {
    // every leaf exponent is a unit mod 972 and each of the 324 units occurs exactly once
    let mut seen = [false; 972];
    let mut j = 0;
    while j < QUAD_SLOTS {
        let u = QUAD_SLOT_EXP[j] as usize;
        assert!(u % 2 == 1 && u % 3 != 0);
        assert!(!seen[u]);
        seen[u] = true;
        j += 1;
    }
};

/// The `R_162` class of leaf j and which of the class's two leaves it is.
///
/// The two roots of `X^2 - psi'^u` are `psi^u` and `psi^{u + 972}` (in the quadratic extension
/// where psi = sqrt(psi') lives), and both have the same class `v = u mod 486` — the exponent
/// that names an `R_162` slot ([`crate::api::POW3_SLOT_EXP`]), since `theta = psi^4 = psi'^2` and
/// `theta^u` depends only on `u mod 486`. Each class therefore owns exactly two leaves,
/// `u = v` (`c = +psi'^v`) and `u = v + 486` (`c = -psi'^v`, because `psi'^486 = -1`).
///
/// `QUAD_CLASS_SLOT[0][s]` is the leaf of class `POW3_SLOT_EXP[s]` with `c = +psi'^v`,
/// `QUAD_CLASS_SLOT[1][s]` the one with `c = -psi'^v`.
const fn quad_class_tables() -> ([u16; 162], [[u16; 162]; 2]) {
    let mut class = [0u16; QUAD_SLOTS];
    let mut slot = [[u16::MAX; 162]; 2];
    let mut j = 0;
    while j < QUAD_SLOTS {
        let u = QUAD_SLOT_EXP[j] as usize;
        class[j] = (u % 486) as u16;
        j += 1;
    }
    // POW3_SLOT_EXP is the R_162 slot order (the classes in the order of first appearance in the
    // splitting tree's SLOT_EXP); the class sets of the two trees agree — both are the units
    // mod 486 — which the assertion below checks.
    let mut s = 0;
    while s < 162 {
        let v = crate::api::POW3_SLOT_EXP[s] as usize;
        let mut j = 0;
        while j < QUAD_SLOTS {
            let u = QUAD_SLOT_EXP[j] as usize;
            if u == v {
                slot[0][s] = j as u16;
            } else if u == v + 486 {
                slot[1][s] = j as u16;
            }
            j += 1;
        }
        assert!(slot[0][s] != u16::MAX && slot[1][s] != u16::MAX);
        s += 1;
    }
    let mut cl = [0u16; 162];
    let mut s = 0;
    while s < 162 {
        cl[s] = crate::api::POW3_SLOT_EXP[s];
        s += 1;
    }
    (cl, slot)
}

const QUAD_CLASS: ([u16; 162], [[u16; 162]; 2]) = quad_class_tables();

/// The 162 `R_162` classes in [`crate::api::POW3_SLOT_EXP`] order (a copy of it, kept here so the
/// quad tables read from one place).
pub const QUAD_POW3_CLASS: [u16; 162] = QUAD_CLASS.0;
/// `QUAD_CLASS_SLOT[sign][s]`: the leaf of class `QUAD_POW3_CLASS[s]` whose constant is
/// `+psi'^v` (sign = 0) or `-psi'^v` (sign = 1).
pub const QUAD_CLASS_SLOT: [[u16; 162]; 2] = QUAD_CLASS.1;

/// Per-prime constants of the quadratic-slot tree. `ParamsQ::<2917>::PSI972` etc.
///
/// Everything that does not involve a root of unity (`QINV`, `R`, `to_mont`, `mont_pre`, ...) is
/// taken from [`Params`], which never touches its own `PSI` — that one is the primitive 1944-th
/// root and does not exist for these primes.
pub struct ParamsQ<const Q: u16>;

impl<const Q: u16> ParamsQ<Q> {
    /// Smallest primitive 972-nd root of unity mod q.
    pub const PSI972: u16 = find_psi972(Q as u64) as u16;
    /// Primitive cube root of unity, omega = psi'^324 (omega^2 + omega + 1 = 0).
    pub const OMEGA: u16 = pow_mod(Self::PSI972 as u64, 324, Q as u64) as u16;
    /// Primitive sixth root of unity, zeta6 = psi'^162 (zeta6^-1 = 1 - zeta6).
    pub const ZETA6: u16 = pow_mod(Self::PSI972 as u64, 162, Q as u64) as u16;
    /// theta = psi'^2, the primitive 486-th root of unity an `R_162` slot evaluates at.
    pub const THETA: u16 = pow_mod(Self::PSI972 as u64, 2, Q as u64) as u16;

    /// psi'^e mod q.
    pub const fn psi_pow(e: u32) -> u16 {
        pow_mod(Self::PSI972 as u64, e as u64, Q as u64) as u16
    }
    /// Plain twiddle zeta for sub-ring k at level `level` (1..=5).
    pub const fn zeta(level: usize, k: usize) -> u16 {
        Self::psi_pow(twiddle_exp_quad(level, k))
    }
    /// Table of plain twiddles for a whole level (K = `SUBRINGS_Q[level]`).
    pub const fn zetas<const K: usize>(level: usize) -> [u16; K] {
        let mut t = [0u16; K];
        let mut k = 0;
        while k < K {
            t[k] = Self::zeta(level, k);
            k += 1;
        }
        t
    }
    pub const ZETA_L1: [u16; 2] = Self::zetas::<2>(1);
    pub const ZETA_L2: [u16; 4] = Self::zetas::<4>(2);
    pub const ZETA_L3: [u16; 12] = Self::zetas::<12>(3);
    pub const ZETA_L4: [u16; 36] = Self::zetas::<36>(4);
    pub const ZETA_L5: [u16; 108] = Self::zetas::<108>(5);

    /// `LEAF_C[j] = psi'^QUAD_SLOT_EXP[j]`, the constant of leaf j: slot j is
    /// `Z_q[X]/(X^2 - LEAF_C[j])`.
    pub const LEAF_C: [u16; QUAD_SLOTS] = {
        let mut t = [0u16; QUAD_SLOTS];
        let mut j = 0;
        while j < QUAD_SLOTS {
            t[j] = Self::psi_pow(QUAD_SLOT_EXP[j] as u32);
            j += 1;
        }
        t
    };
}

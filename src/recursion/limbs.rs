//! The per-limb identity `sum_i F_i v_i - sum_j c_j C_j - q k = 0` in `S`-component form.
//!
//! The key rows and the RNS residues reach this module in the NTT domain, so both are inverted
//! once — [`crate::scalar::intt`] for a splitting limb, [`intt_quad`] for a quadratic-slot one —
//! and read in the `Z`-basis of `S`. For output component `m` the multiplier of `v_{i,l}` is
//! `F_{i,(m-l) mod 4}`, twisted by `-Z` when `l > m`.
use super::chain::{At, Carries, Chain, Product};
use super::setup::Setup;
use super::{centre, chunk, Build, Cap, Gadget, Kind, Poly, SElem, CHUNK, CHUNKS};
use crate::api::{
    PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162,
    POW3_SLOT_EXP, SLOT_648,
};
use crate::params::{
    inv_mod, pow_mod, Params, ParamsQ, CONDUCTOR, DEGREE_Q, N, QUAD_CLASS_SLOT,
    QUAD_POW3_CLASS, RADIX_Q, SUBRINGS_Q,
};
use crate::scalar;
use crate::scheme::PublicParameters;
use crate::types::RingElement;

/// What one limb costs beyond its prime: the carry gadget of the plan's section 2b and the two
/// base-512 digits of the wraparound quotient.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape {
    pub q: u16,
    pub quad: bool,
    pub carry: Gadget,
    pub quotient: Gadget,
}

impl Shape {
    pub fn of(q: u16) -> Shape {
        let (base, levels) = match q {
            2917 | 3889 => (1024, 3),
            4861 | 9721 | 12637 => (256, 4),
            _ => unreachable!("no limb with q = {q}"),
        };
        Shape {
            q,
            quad: !matches!(q, 3889 | 9721),
            carry: Gadget { base, levels },
            quotient: Gadget { base: 512, levels: 2 },
        }
    }
}

// =============================================================================================
// the two inverse transforms
// =============================================================================================

/// The inverse of [`crate::scalar::ntt_quad`], butterfly for butterfly the inverse of
/// [`crate::scalar::intt`] on the quadratic-slot tree.
pub fn intt_quad<const Q: u16>(v: &[u32; N]) -> [u32; N] {
    let q = Q as u64;
    let psi = ParamsQ::<Q>::PSI972 as u64;
    let w = ParamsQ::<Q>::OMEGA as u64;
    let w2 = w * w % q;
    let inv2 = inv_mod(2, q);
    let inv3 = inv_mod(3, q);
    let mut u = [0u64; N];
    for i in 0..N {
        u[i] = v[i] as u64 % q;
    }
    for level in (1..=5).rev() {
        let n = DEGREE_Q[level];
        let p = RADIX_Q[level];
        let m = n / p;
        for k in 0..SUBRINGS_Q[level] {
            let base = k * n;
            let zi = inv_mod(
                pow_mod(psi, crate::params::twiddle_exp_quad(level, k) as u64, q),
                q,
            );
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
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    let det = inv_mod((2 * z6 + q - 1) % q, q);
    for i in 0..324 {
        let (y0, y1) = (u[i], u[i + 324]);
        let a1 = (y0 + q - y1) % q * det % q;
        let a0 = (y0 + q - z6 * a1 % q) % q;
        u[i] = a0;
        u[i + 324] = a1;
    }
    core::array::from_fn(|i| u[i] as u32)
}

/// The coefficients of the `R_648` element a limb's transform holds, centred.
pub fn coefficients(q: u16, quad: bool, slots: &[u32; N]) -> [i64; N] {
    let c = match (q, quad) {
        (3889, false) => scalar::intt::<3889>(slots),
        (9721, false) => scalar::intt::<9721>(slots),
        (2917, true) => intt_quad::<2917>(slots),
        (4861, true) => intt_quad::<4861>(slots),
        (12637, true) => intt_quad::<12637>(slots),
        _ => unreachable!("no limb with q = {q}"),
    };
    core::array::from_fn(|i| centre(c[i] as i64, q as i64))
}

/// The forward transform of the same limb, for checking [`coefficients`].
pub fn transform(q: u16, quad: bool, coefficients: &[i64; N]) -> [u32; N] {
    let a: [u32; N] = core::array::from_fn(|i| coefficients[i].rem_euclid(q as i64) as u32);
    match (q, quad) {
        (3889, false) => scalar::ntt::<3889>(&a),
        (9721, false) => scalar::ntt::<9721>(&a),
        (2917, true) => scalar::ntt_quad::<2917>(&a),
        (4861, true) => scalar::ntt_quad::<4861>(&a),
        (12637, true) => scalar::ntt_quad::<12637>(&a),
        _ => unreachable!("no limb with q = {q}"),
    }
}

/// The 648 slots of the `R_648` element whose four `R_162` components are `c`: the inverse of
/// [`crate::api::decompose_648_to_4x162`], `E_t = sum_k psi^{v k} i^{t k} Y_k(v)`.
fn slots_split<const Q: u16>(c: &[PowerOfThreeRingElement; 4]) -> [u32; N] {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    let i4 = pow_mod(psi, (CONDUCTOR / 4) as u64, q);
    let mut y = [0u32; N];
    for s in 0..N162 {
        let v = POW3_SLOT_EXP[s] as u64;
        for t in 0..4u64 {
            let mut acc = 0u64;
            for k in 0..4u64 {
                let yk = (c[k as usize].v[s] as i64).rem_euclid(q as i64) as u64;
                let e = pow_mod(psi, v * k % CONDUCTOR as u64, q) * pow_mod(i4, t * k % 4, q) % q;
                acc = (acc + e * yk) % q;
            }
            y[SLOT_648[t as usize][s] as usize] = acc as u32;
        }
    }
    y
}

/// The same for a quadratic-slot limb: `y mod (X^2 -+ psi'^v) = [Y_k -+ psi'^v Y_{k+2}]`.
fn slots_quad<const Q: u16>(c: &[PowerOfThreeRingElement; 4]) -> [u32; N] {
    let q = Q as u64;
    let mut y = [0u32; N];
    for s in 0..N162 {
        let pv = ParamsQ::<Q>::psi_pow(QUAD_POW3_CLASS[s] as u32) as u64;
        let jp = QUAD_CLASS_SLOT[0][s] as usize;
        let jm = QUAD_CLASS_SLOT[1][s] as usize;
        for k in 0..2 {
            let y0 = (c[k].v[s] as i64).rem_euclid(q as i64) as u64;
            let y2 = pv * (c[k + 2].v[s] as i64).rem_euclid(q as i64) as u64 % q;
            y[2 * jp + k] = ((y0 + y2) % q) as u32;
            y[2 * jm + k] = ((y0 + q - y2) % q) as u32;
        }
    }
    y
}

fn slots_of(q: u16, quad: bool, c: &[PowerOfThreeRingElement; 4]) -> [u32; N] {
    match (q, quad) {
        (3889, false) => slots_split::<3889>(c),
        (9721, false) => slots_split::<9721>(c),
        (2917, true) => slots_quad::<2917>(c),
        (4861, true) => slots_quad::<4861>(c),
        (12637, true) => slots_quad::<12637>(c),
        _ => unreachable!("no limb with q = {q}"),
    }
}

// =============================================================================================
// the coefficient form of the public and committed data
// =============================================================================================

/// The four `S`-components of an `R_648` element in the `Z`-basis: `a_l = sum_m (-1)^m a_{4m+l}`.
pub fn split(a: &[i64; N]) -> [SElem; 4] {
    core::array::from_fn(|l| {
        core::array::from_fn(|m| if m % 2 == 0 { a[4 * m + l] } else { -a[4 * m + l] })
    })
}

/// The four `S`-components of a folded-witness element.
pub fn components(e: &RingElement) -> [SElem; 4] {
    let a: [i64; N] = core::array::from_fn(|i| e.v[i] as i64);
    split(&a)
}

/// The key rows in coefficient form, `F[i]` the four `S`-components of `A_i` modulo the limb's
/// prime, centred. The FFI layer aliases these; the blocks of the plan's section 3 are
/// [`chunk::blocks`] of each of them.
pub struct KeyRows {
    pub q: u16,
    pub quad: bool,
    pub rows: Vec<[SElem; 4]>,
}

impl KeyRows {
    /// The `R_648` element of row `i` in coefficient form, centred.
    pub fn coefficients(&self, i: usize) -> [i64; N] {
        let mut a = [0i64; N];
        for l in 0..4 {
            for m in 0..N162 {
                a[4 * m + l] = if m % 2 == 0 { self.rows[i][l][m] } else { -self.rows[i][l][m] };
            }
        }
        a
    }
}

/// The key rows of one limb, inverted once.
pub fn key_rows(pp: &PublicParameters, limb: usize) -> KeyRows {
    let key = pp.key();
    let (q, quad) = (key.prime(limb), key.is_quadratic(limb));
    let rows = (0..key.len_ring())
        .map(|i| split(&coefficients(q, quad, &key_slots(pp, limb, i))))
        .collect();
    KeyRows { q, quad, rows }
}

/// The 648 slots of `A_i` as the key holds them, for checking the inverse transform.
pub fn key_slots(pp: &PublicParameters, limb: usize, i: usize) -> [u32; N] {
    let key = pp.key();
    let q = key.prime(limb);
    let row = key.row(limb);
    core::array::from_fn(|j| (row[i / 32].v[j][i % 32] as i32).rem_euclid(q as i32) as u32)
}

/// The RNS residues the recursion commits to, chunked in the order the encoding stores them:
/// `vectors[limb * 4 + m][b * columns + j]` is chunk `b` of component `m` of column `j`.
#[derive(Clone)]
pub struct Residues {
    pub vectors: Vec<Vec<Poly>>,
}

impl Residues {
    /// The coefficients of one vector, as the commitment and the witness export want them.
    pub fn flat(&self, vector: usize) -> &[i16] {
        self.vectors[vector].as_flattened()
    }
}

/// The residues of every limb of one commitment, in coefficient form.
pub fn residues(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    primes: &[u16],
) -> Residues {
    let r = matrix.cols();
    let mut vectors = Vec::with_capacity(4 * primes.len());
    for (limb, &prime) in primes.iter().enumerate() {
        let Shape { q, quad, .. } = Shape::of(prime);
        let columns: Vec<[SElem; 4]> = (0..r)
            .map(|j| {
                let c: [PowerOfThreeRingElement; 4] =
                    core::array::from_fn(|m| matrix.get(m, j).limbs[limb]);
                split(&coefficients(q, quad, &slots_of(q, quad, &c)))
            })
            .collect();
        for m in 0..4 {
            let chunks: Vec<[Poly; CHUNKS]> =
                columns.iter().map(|c| chunk::chunks(&c[m])).collect();
            let mut v = Vec::with_capacity(CHUNKS * r);
            for b in 0..CHUNKS {
                for c in chunks.iter() {
                    v.push(c[b]);
                }
            }
            v.resize((CHUNKS * r).next_multiple_of(super::PAD), [0i16; super::DEG]);
            vectors.push(v);
        }
    }
    Residues { vectors }
}

// =============================================================================================
// the four chains of one limb
// =============================================================================================

/// Append the four component identities of one limb to `build`.
pub fn encode(build: &mut Build, setup: &Setup, residues: Option<&Residues>, limb: usize) {
    let shape = setup.limbs[limb];
    let q = shape.q as i64;
    let (n, r) = (build.n, build.r);
    build.limbs.push(shape);

    let base_key = build.public.len();
    for part in 0..8 {
        build.group_blocks(Kind::Key { limb, part }, setup.key_blocks(limb, part));
    }

    let residues_vector: Vec<usize> = (0..4)
        .map(|m| {
            let v = build.vector(
                format!("C[{q}][{m}]"),
                Cap::PerCoefficient((q - 1) as f64 / 2.0),
                CHUNK,
                false,
            );
            match residues {
                Some(res) => {
                    for &p in res.vectors[limb * 4 + m].iter().take(CHUNKS * r) {
                        build.vectors[v].push(p);
                    }
                }
                None => {
                    build.vectors[v].zeros(CHUNKS * r);
                }
            }
            build.residues.push(v);
            v
        })
        .collect();
    let quotient = build.digit_vectors(&format!("k[{q}]"), shape.quotient);
    let carry = build.carry_vectors(&format!("{q}"), shape.carry);

    for m in 0..4 {
        let mut products = Vec::with_capacity(4 * CHUNKS * n + CHUNKS * r);
        for l in 0..4 {
            let k = (m + 4 - l) % 4;
            let twist = usize::from(l > m);
            for b in 0..CHUNKS {
                for i in 0..n {
                    products.push(Product {
                        blocks: base_key + (k * 2 + twist) * n + i,
                        chunk: b,
                        at: build.v_at(l, b, i),
                    });
                }
            }
        }
        for b in 0..CHUNKS {
            for j in 0..r {
                products.push(Product {
                    blocks: build.challenges + j,
                    chunk: b,
                    at: At { vector: residues_vector[m], off: b * r + j },
                });
            }
        }
        build.seal(
            Chain {
                name: format!("limb {q} component {m}"),
                products,
                scaled: Vec::new(),
                output: [0i64; N162],
                carries: Carries { gadget: shape.carry, at: Vec::new() },
            },
            q,
            (shape.quotient, &quotient),
            &carry,
        );
    }
}

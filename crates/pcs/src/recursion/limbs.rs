//! The per-limb identity `sum_i F_i v_i - sum_j c_j C_j - q k = 0` in `S`-component form.
//!
//! The key rows and the RNS residues reach this module in the NTT domain and are inverted into
//! the `Z`-basis of `S`. The key rows go once per key through the scalar [`crate::scalar::intt`]
//! or [`intt_quad`]; the residues are per commitment and go 32 columns at a time through their
//! limb's recombination — the table of [`recombination`] for a splitting limb, the class
//! butterfly `y mod (X^2 -+ psi'^v) = Y_k -+ psi'^v Y_{k+2}` for a quadratic-slot one — and that
//! tree's batched inverse transform.
//! For output component `m` the multiplier of `v_{i,l}` is `F_{i,(m-l) mod 4}`, twisted by `-Z`
//! when `l > m`.
use super::chain::{padded, witness_table, At, Carries, Chain, Product, Run};
use super::setup::Setup;
use super::{centre, Build, Cap, Gadget, Kind, Overflow, Poly, SElem, CHUNK, CHUNKS, DEG, PAD};
use crate::limb::dispatch_limb;
use crate::params::{
    inv_mod, pow_mod, Params, ParamsQ, CONDUCTOR, DEGREE_Q, N, QUAD_CLASS_SLOT, QUAD_POW3_CLASS,
    RADIX_Q, SUBRINGS_Q,
};
use crate::scalar;
use crate::ring::{
    Batch32, PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, Representation,
    RingElement, VerticallyAlignedMatrix, N162, POW3_SLOT_EXP, SLOT_648,
};
use crate::scheme::PublicParameters;
use crate::simd::ntt::bin_large as vl;
use crate::simd::ntt::gen_small::intt_gen_batch32;
use crate::simd::ntt::gen_large as vgl;
use crate::simd::ntt::gen_quad::intt_quad_gen_batch32;

/// What one limb costs beyond its prime: the carry gadget of the plan's section 2b and the
/// base-512 digits of the wraparound quotient, both sized from the shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape {
    pub q: u16,
    pub quad: bool,
    pub carry: Gadget,
    pub quotient: Gadget,
}

/// The limb's quotient and carries are linear in the fold, whose length the cap holds under
/// `sqrt(FOLD_CAP n N r)`; at the cap the largest quotient coefficient measured `226 sqrt(n r)`
/// and the largest carry `207 q sqrt(n r)`, each over 33 calibration rounds (`examples/gadget.rs`
/// under `GADGET_STATS`) across every limb and shapes from `(n, r) = (256, 64)` to
/// `(1024, 256)`, with a two-fold spread between rounds once normalised. The constants carry a
/// 1.4 margin over those maxima.
pub const QUOTIENT_PER_ROOT: f64 = 320.0;
pub const CARRY_PER_ROOT: f64 = 300.0;

/// Whether `q` splits `X^648 + 1` into quadratic slots rather than linear ones.
pub fn quad(q: u16) -> bool {
    crate::limb::is_quad(q)
}

impl Shape {
    /// The shape of limb `q` at `n` ring elements per column and `r` columns.
    pub fn of(q: u16, n: usize, r: usize) -> Shape {
        // the carry base: 1024 below `2^12`, 256 above it, so that a carry level stays a
        // per-coefficient cap of the size the keys' SIS rule is set for.
        let base = dispatch_limb!(q, |Q| if Q < 1 << 12 { 1024 } else { 256 });
        let root = ((n * r) as f64).sqrt();
        Shape {
            q,
            quad: quad(q),
            carry: Gadget::covering(base, CARRY_PER_ROOT * q as f64 * root),
            quotient: Gadget::covering(512, QUOTIENT_PER_ROOT * root),
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
pub fn coefficients(q: u16, slots: &[u32; N]) -> [i64; N] {
    let c = dispatch_limb!(
        q,
        split |Q| scalar::intt::<Q>(slots),
        quad |Q| intt_quad::<Q>(slots),
    );
    core::array::from_fn(|i| centre(c[i] as i64, q as i64))
}

/// The forward transform of the same limb, for checking [`coefficients`].
pub fn transform(q: u16, coefficients: &[i64; N]) -> [u32; N] {
    let a: [u32; N] = core::array::from_fn(|i| coefficients[i].rem_euclid(q as i64) as u32);
    dispatch_limb!(
        q,
        split |Q| scalar::ntt::<Q>(&a),
        quad |Q| scalar::ntt_quad::<Q>(&a),
    )
}

// =============================================================================================
// the coefficient form of the public and committed data
// =============================================================================================

/// The four `S`-components of an `R_648` element in the `Z`-basis: `a_l = sum_m (-1)^m a_{4m+l}`.
pub fn split(a: &[i64; N]) -> [SElem; 4] {
    core::array::from_fn(|l| {
        core::array::from_fn(|m| {
            if m % 2 == 0 {
                a[4 * m + l]
            } else {
                -a[4 * m + l]
            }
        })
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
                a[4 * m + l] = if m % 2 == 0 {
                    self.rows[i][l][m]
                } else {
                    -self.rows[i][l][m]
                };
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
        .map(|i| split(&coefficients(q, &key_slots(pp, limb, i))))
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

/// The recombination `E_t = sum_k psi^{v_s k} i^{tk} Y_k(v_s)` of the plan's section 3, as a
/// table: `e[(s * 4 + t) * 4 + k]` is the multiplier of component `k` in slot `SLOT_648[t][s]`.
///
/// `psi^{v_s k}` for `k = 0..4` is three multiplications by `psi^{v_s}` and `i^{tk}` is one of
/// four constants, so the whole table costs `N162` exponentiations rather than `16 N162`.
pub(crate) fn recombination<const Q: u16>() -> Vec<u16> {
    let q = Q as u64;
    let psi = Params::<Q>::PSI as u64;
    let i4 = pow_mod(psi, (CONDUCTOR / 4) as u64, q);
    let mut e = vec![0u16; N162 * 16];
    for s in 0..N162 {
        let base = pow_mod(psi, POW3_SLOT_EXP[s] as u64, q);
        let mut pk = 1u64;
        for k in 0..4 {
            let step = pow_mod(i4, k as u64, q);
            let mut it = 1u64;
            for t in 0..4 {
                e[(s * 4 + t) * 4 + k] = (pk * it % q) as u16;
                it = it * step % q;
            }
            pk = pk * base % q;
        }
    }
    e
}

/// Which vectorised inverse transform a splitting prime runs, as an associated const so the
/// choice is made before the branches are emitted.
struct Inv<const Q: u16>;

impl<const Q: u16> Inv<Q> {
    const LARGE: bool = vl::is_large(Q);
}

/// The residues of one splitting limb, 32 columns at a time: the recombination above out of a
/// table, then the crate's vectorised inverse transform on the whole batch — [`intt_gen_batch32`]
/// below `2^14`, `ntt::gen_large`'s above it.
fn columns_split<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    r: usize,
    out: &mut [Vec<Poly>],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let e = recombination::<Q>();
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
        drain_batch(&batch, first, cols, r, out);
    }
}

/// The four `S`-components of one column read straight out of an inverted batch, in chunk order.
fn drain_batch(batch: &Batch32, first: usize, cols: usize, r: usize, out: &mut [Vec<Poly>]) {
    for p in 0..cols {
        for (l, vector) in out.iter_mut().enumerate() {
            for b in 0..CHUNKS {
                let poly = &mut vector[b * r + first + p];
                for (u, x) in poly.iter_mut().take(CHUNK).enumerate() {
                    let m = CHUNK * b + u;
                    let c = batch.v[4 * m + l][p];
                    *x = if m % 2 == 0 { c } else { -c };
                }
            }
        }
    }
}

/// The residues of one quadratic-slot limb, 32 columns at a time: the class butterfly
/// `y mod (X^2 -+ psi'^v) = Y_k -+ psi'^v Y_{k+2}` — the inverse of
/// [`crate::scalar::decompose_quad_648_to_4x162`] — out of a table of the 162 `psi'^v`, then the
/// tree's own vectorised inverse transform on the whole batch.
fn columns_quad<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    r: usize,
    out: &mut [Vec<Poly>],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let pv: [i32; N162] =
        core::array::from_fn(|s| ParamsQ::<Q>::psi_pow(QUAD_POW3_CLASS[s] as u32) as i32);
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
        drain_batch(&batch, first, cols, r, out);
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
        let q = prime;
        let mut out: Vec<Vec<Poly>> = (0..4)
            .map(|_| vec![[0i16; DEG]; (CHUNKS * r).next_multiple_of(PAD)])
            .collect();
        dispatch_limb!(
            q,
            split |Q| columns_split::<Q>(matrix, limb, r, &mut out),
            quad |Q| columns_quad::<Q>(matrix, limb, r, &mut out),
        );
        vectors.append(&mut out);
    }
    Residues { vectors }
}

// =============================================================================================
// the four chains of one limb
// =============================================================================================

/// Append the four component identities of one limb to `build`.
pub fn encode(
    build: &mut Build,
    setup: &Setup,
    residues: Option<&Residues>,
    limb: usize,
) -> Result<(), Overflow> {
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
        let mut runs = Vec::with_capacity(4 * CHUNKS + CHUNKS);
        for l in 0..4 {
            let part = ((m + 4 - l) % 4) * 2 + usize::from(l > m);
            for b in 0..CHUNKS {
                for i in 0..n {
                    products.push(Product {
                        blocks: base_key + part * n + i,
                        chunk: b,
                        at: build.v_at(l, b, i),
                    });
                }
                if build.witness {
                    runs.push(Run {
                        g: setup.key_table(limb, part, b).clone(),
                        x: build.v_tables[l * CHUNKS + b].clone(),
                        terms: padded(n),
                    });
                }
            }
        }
        for b in 0..CHUNKS {
            for j in 0..r {
                products.push(Product {
                    blocks: build.challenges + j,
                    chunk: b,
                    at: At {
                        vector: residues_vector[m],
                        off: b * r + j,
                    },
                });
            }
            if build.witness {
                runs.push(Run {
                    g: build.challenge_tables[b].clone(),
                    x: witness_table(&build.vectors[residues_vector[m]].polys, b * r, r, 1),
                    terms: padded(r),
                });
            }
        }
        build.seal(
            Chain {
                name: format!("limb {q} component {m}"),
                products,
                scaled: Vec::new(),
                output: [0i64; N162],
                carries: Carries {
                    gadget: shape.carry,
                    at: Vec::new(),
                },
            },
            runs,
            q,
            (shape.quotient, &quotient),
            &carry,
        )?;
    }
    Ok(())
}

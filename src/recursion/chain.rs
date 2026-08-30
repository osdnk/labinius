//! One `S`-identity `sum_nu g_nu x_nu = z` as `BLOCKS` degree-1 constraints over `Z_Q[X]/(X^64+1)`.
//!
//! Diagonal `a` carries the sub-products of every public sub-chunk `a` against every witness chunk;
//! its positions `0 .. SUB` are the `a`-th window of the output, its positions `SUB .. CHUNK+SUB-1`
//! define the carry `e_a`, and positions above that are zero because `CHUNK + SUB - 1 <= DEG`. With
//! `e_{-1} = 0` the equation at diagonal `a` is
//!
//! ```text
//!     D_a + e_{a-1} - Z^SUB e_a - [a = 0] e_{BLOCKS-1} - [a = BLOCKS/2] e_{BLOCKS-1} - b_a = 0,
//! ```
//!
//! the two wrap terms being `-e_{BLOCKS-1}` and `-Z^81 e_{BLOCKS-1}`, i.e. `Z^162 e_{BLOCKS-1}`
//! moved back by `Z^162 = -Z^81 - 1`. Summed against `Z^{SUB a}` the carries telescope to
//! `-Z^162 e_{BLOCKS-1}` and the identity is `sum_a Z^{SUB a} D_a = z + Phi_243 e_{BLOCKS-1}`.
use super::{
    Blocks, Gadget, SElem, Vector, BLOCKS, CARRY, CHUNK, CHUNKS, DEG, SPAN, SUB,
};
use crate::api::N162;
use core::arch::x86_64::*;

/// The `SUB`-tap sums of one diagonal, negacyclically reduced in `X^DEG + 1`.
pub type Sums = [[i64; DEG]; BLOCKS];

/// A witness polynomial: entry `off` of vector `vector`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct At {
    pub vector: usize,
    pub off: usize,
}

/// `public[blocks][chunk][a]` against the witness poly at `at`.
pub struct Product {
    pub blocks: usize,
    pub chunk: usize,
    pub at: At,
}

/// A constant public multiplier: `factor` against the witness poly at `at`, which is chunk `chunk`
/// of its element and so lands whole at diagonal `SPAN * chunk`.
pub struct Scaled {
    pub factor: i64,
    pub chunk: usize,
    pub at: At,
}

/// The carries of one chain: `levels` gadget digits per diagonal, `at[d]` the first of the
/// `BLOCKS` polys of level `d`.
pub struct Carries {
    pub gadget: Gadget,
    pub at: Vec<At>,
}

/// One `S`-identity.
pub struct Chain {
    pub name: String,
    pub products: Vec<Product>,
    pub scaled: Vec<Scaled>,
    pub output: SElem,
    pub carries: Carries,
}

/// The public sub-chunks of a chain in term-major order, `g[(a * SUB + u) * terms + t]`.
pub struct Prepared {
    pub terms: usize,
    g: Vec<i16>,
}

fn poly<'a>(w: &'a [Vector], at: At) -> &'a [i16; DEG] {
    &w[at.vector].polys[at.off]
}

impl Chain {
    pub fn prepare(&self, public: &[Blocks]) -> Prepared {
        let terms = self.products.len().next_multiple_of(32).max(32);
        let mut g = vec![0i16; BLOCKS * SUB * terms];
        for (tile, ps) in self.products.chunks(32).enumerate() {
            for a in 0..BLOCKS {
                for u in 0..SUB {
                    let row = &mut g[(a * SUB + u) * terms + 32 * tile..];
                    for (t, p) in ps.iter().enumerate() {
                        row[t] = public[p.blocks][p.chunk][a][u];
                    }
                }
            }
        }
        Prepared { terms, g }
    }

    /// `D_a` for every diagonal: `sum_t public_t[a] * s_t` over the products alone, reduced
    /// negacyclically in `X^DEG + 1`.
    pub fn sums(&self, prep: &Prepared, w: &[Vector]) -> Sums {
        let terms = prep.terms;
        let mut x = vec![0i16; DEG * terms];
        for (tile, ps) in self.products.chunks(32).enumerate() {
            for j in 0..DEG {
                let row = &mut x[j * terms + 32 * tile..];
                for (t, p) in ps.iter().enumerate() {
                    row[t] = poly(w, p.at)[j];
                }
            }
        }
        let mut out = [[0i64; DEG]; BLOCKS];
        let mut acc = [0i64; DEG + SUB];
        for a in 0..BLOCKS {
            acc.fill(0);
            for j in 0..DEG {
                let d = unsafe {
                    dots(
                        prep.g.as_ptr().add(a * SUB * terms),
                        x.as_ptr().add(j * terms),
                        terms,
                    )
                };
                for (u, &x) in d.iter().enumerate() {
                    acc[j + u] += x;
                }
            }
            out[a][..DEG].copy_from_slice(&acc[..DEG]);
            for j in DEG..DEG + SUB {
                out[a][j - DEG] -= acc[j];
            }
        }
        out
    }

    /// The constant multipliers, which enter whole at diagonal `SPAN * chunk`.
    pub fn scaled_into(&self, sums: &mut Sums, w: &[Vector]) {
        for s in &self.scaled {
            let q = poly(w, s.at);
            for j in 0..DEG {
                sums[SPAN * s.chunk][j] += s.factor * q[j] as i64;
            }
        }
    }

    /// `sum_a Z^{SUB a} D_a mod Phi_243`: what the identity's left side is worth over `Z`, before
    /// the output and the quotient are subtracted. Panics unless every diagonal fits its degree
    /// bound, which is the support claim of every witness chunk.
    pub fn value(sums: &Sums) -> SElem {
        let mut p = vec![0i64; SUB * (BLOCKS - 1) + CHUNK + SUB - 1];
        for (a, d) in sums.iter().enumerate() {
            assert!(
                d[CHUNK + SUB - 1..].iter().all(|&x| x == 0),
                "diagonal {a} exceeds its degree bound"
            );
            for j in 0..CHUNK + SUB - 1 {
                p[SUB * a + j] += d[j];
            }
        }
        super::chunk::reduce(&p)
    }

    /// The honest carries: `e_{BLOCKS-1}` is the part of `sum_a Z^{SUB a} D_a` at and above
    /// `Z^162`, the rest follow the recurrence.
    pub fn honest_carries(sums: &Sums) -> [[i64; CARRY]; BLOCKS] {
        let mut p = vec![0i64; SUB * (BLOCKS - 1) + CHUNK + SUB - 1];
        for (a, d) in sums.iter().enumerate() {
            for j in 0..CHUNK + SUB - 1 {
                p[SUB * a + j] += d[j];
            }
        }
        for (a, d) in sums.iter().enumerate() {
            assert!(
                d[CHUNK + SUB - 1..].iter().all(|&x| x == 0),
                "diagonal {a} exceeds its degree bound"
            );
        }
        let mut last = [0i64; CARRY];
        last.copy_from_slice(&p[N162..]);
        let mut e = [[0i64; CARRY]; BLOCKS];
        let mut prev = [0i64; CARRY];
        for a in 0..BLOCKS {
            let mut t = [0i64; CHUNK + SUB - 1];
            t[..CHUNK + SUB - 1].copy_from_slice(&sums[a][..CHUNK + SUB - 1]);
            for j in 0..CARRY {
                t[j] += prev[j];
            }
            if a == 0 || a == BLOCKS / 2 {
                for j in 0..CARRY {
                    t[j] -= last[j];
                }
            }
            prev.copy_from_slice(&t[SUB..SUB + CARRY]);
            e[a] = prev;
        }
        assert_eq!(e[BLOCKS - 1], last, "the cyclic carry is not a fixed point");
        e
    }

    /// The carries as the digits store them — every coefficient of every digit poly, so that a
    /// digit outside the claimed support is seen exactly where LaBRADOR would see it.
    pub fn carry_values(&self, w: &[Vector]) -> [[i64; DEG]; BLOCKS] {
        let mut e = [[0i64; DEG]; BLOCKS];
        for (a, out) in e.iter_mut().enumerate() {
            for (d, at) in self.carries.at.iter().enumerate() {
                let q = poly(w, At { vector: at.vector, off: at.off + a });
                let s = self.carries.gadget.base.pow(d as u32);
                for j in 0..DEG {
                    out[j] += s * q[j] as i64;
                }
            }
        }
        e
    }

    /// The exact left-hand side of every block equation.
    pub fn residuals(&self, prep: &Prepared, w: &[Vector]) -> [[i128; DEG]; BLOCKS] {
        let mut sums = self.sums(prep, w);
        self.scaled_into(&mut sums, w);
        let e = self.carry_values(w);
        let mut out = [[0i128; DEG]; BLOCKS];
        for a in 0..BLOCKS {
            let r = &mut out[a];
            for j in 0..DEG {
                r[j] = sums[a][j] as i128;
            }
            if a > 0 {
                for j in 0..DEG {
                    r[j] += e[a - 1][j] as i128;
                }
            }
            for j in 0..DEG {
                let (t, sign) = if j + SUB < DEG { (j + SUB, 1) } else { (j + SUB - DEG, -1) };
                r[t] -= sign * e[a][j] as i128;
            }
            if a == 0 || a == BLOCKS / 2 {
                for j in 0..DEG {
                    r[j] -= e[BLOCKS - 1][j] as i128;
                }
            }
            for j in 0..SUB {
                r[j] -= self.output[SUB * a + j] as i128;
            }
        }
        out
    }
}

/// `sum_k a_{u,k} b_k` over `n` `i16` pairs for the [`SUB`] consecutive rows of `a`, `n` a multiple
/// of 32: `vpmaddwd` into `i32` lanes, widened every eight accumulations so that
/// `8 * 2 * BLOCK_LIMIT * COEFF_LIMIT` cannot overflow. One load of `b` feeds all [`SUB`] rows,
/// which is what the diagonal loop wants and what keeps the kernel off the load ports.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dots(a: *const i16, b: *const i16, n: usize) -> [i64; SUB] {
    let mut wide = [_mm512_setzero_si512(); SUB];
    let mut acc = [_mm512_setzero_si512(); SUB];
    for k in 0..n / 32 {
        let y = _mm512_loadu_si512(b.add(32 * k) as *const __m512i);
        for u in 0..SUB {
            let x = _mm512_loadu_si512(a.add(u * n + 32 * k) as *const __m512i);
            acc[u] = _mm512_add_epi32(acc[u], _mm512_madd_epi16(x, y));
        }
        if k % 8 == 7 {
            for u in 0..SUB {
                wide[u] = _mm512_add_epi64(wide[u], widen(acc[u]));
                acc[u] = _mm512_setzero_si512();
            }
        }
    }
    core::array::from_fn(|u| _mm512_reduce_add_epi64(_mm512_add_epi64(wide[u], widen(acc[u]))))
}

#[inline(always)]
unsafe fn widen(a: __m512i) -> __m512i {
    _mm512_add_epi64(
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a)),
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(a)),
    )
}

const _: () = assert!(CHUNKS * CHUNK == N162);

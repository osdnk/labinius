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
    Blocks, Gadget, Poly, SElem, Vector, BLOCKS, BLOCK_LIMIT, CARRY, CHUNK, CHUNKS, COEFF_LIMIT,
    DEG, SPAN, SUB,
};
use crate::ring::N162;
use core::arch::x86_64::*;
use std::sync::Arc;

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

/// A term-major table and the largest magnitude in it, which sets how many products the kernel
/// may accumulate in `i32` before widening.
#[derive(Clone)]
pub struct Table {
    pub data: Arc<[i16]>,
    pub bound: i64,
}

/// One run of a chain's terms in term-major form: `g[(a * SUB + u) * terms + t]` is coefficient
/// `u` of sub-chunk `a` of the public multiplier of term `t`, `x[j * terms + t]` coefficient `j` of
/// its witness poly. `terms` is a multiple of 32 and both tables are zero past the run.
#[derive(Clone)]
pub struct Run {
    pub g: Table,
    pub x: Table,
    pub terms: usize,
}

/// A chain's products as runs, the form [`Chain::sums`] consumes.
pub type Prepared = Vec<Run>;

pub fn padded(len: usize) -> usize {
    len.next_multiple_of(32).max(32)
}

/// The term-major sub-chunks of `len` consecutive public entries from `first`, at chunk `b`.
pub fn public_table(public: &[Blocks], first: usize, len: usize, b: usize) -> Table {
    assert!(first + len <= public.len());
    let terms = padded(len);
    let mut g = vec![0i16; BLOCKS * SUB * terms];
    unsafe {
        transpose(
            public
                .as_ptr()
                .add(first)
                .cast::<i16>()
                .add(b * BLOCKS * SUB),
            CHUNKS * BLOCKS * SUB,
            len,
            BLOCKS * SUB,
            g.as_mut_ptr(),
            terms,
        );
    }
    Table::of(g)
}

/// The term-major coefficients of `len` polys from `polys[first]`, every `step`-th.
pub fn witness_table(polys: &[Poly], first: usize, len: usize, step: usize) -> Table {
    assert!(len == 0 || first + (len - 1) * step < polys.len());
    let terms = padded(len);
    let mut x = vec![0i16; DEG * terms];
    unsafe {
        transpose(
            polys.as_ptr().add(first).cast::<i16>(),
            step * DEG,
            len,
            DEG,
            x.as_mut_ptr(),
            terms,
        );
    }
    Table::of(x)
}

impl Table {
    fn of(data: Vec<i16>) -> Table {
        let bound = data.iter().map(|&x| (x as i64).abs()).max().unwrap_or(0);
        Table {
            data: data.into(),
            bound,
        }
    }
}

/// `dst[c * dst_stride + r] = src[r * src_stride + c]` for `r < rows`, `c < cols`, in 32 x 32
/// tiles; `dst` outside that rectangle is left alone.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn transpose(
    src: *const i16,
    src_stride: usize,
    rows: usize,
    cols: usize,
    dst: *mut i16,
    dst_stride: usize,
) {
    for r in (0..rows).step_by(32) {
        for c in (0..cols).step_by(32) {
            tile(
                src.add(r * src_stride + c),
                src_stride,
                (rows - r).min(32),
                (cols - c).min(32),
                dst.add(c * dst_stride + r),
                dst_stride,
            );
        }
    }
}

/// The `vpermi2w` indexes that swap bit `s` of the row index with bit `s` of the lane index
/// across a pair of rows: `lo` builds the row whose bit is clear, `hi` the one whose bit is set.
const fn swap(s: usize, hi: bool) -> [i16; 32] {
    let m = 1 << s;
    let mut idx = [0i16; 32];
    let mut c = 0;
    while c < 32 {
        idx[c] = match (c & m == 0, hi) {
            (true, false) => c,
            (false, false) => 32 + (c ^ m),
            (true, true) => c ^ m,
            (false, true) => 32 + c,
        } as i16;
        c += 1;
    }
    idx
}
static SWAP_LO: [[i16; 32]; 4] = [
    swap(0, false),
    swap(1, false),
    swap(2, false),
    swap(3, false),
];
static SWAP_HI: [[i16; 32]; 4] = [swap(0, true), swap(1, true), swap(2, true), swap(3, true)];

/// One tile, as two halves of 16 rows: four butterflies of `vpermi2w` swap the row bits with the
/// low four lane bits, after which register `c` holds output row `c` in its low lanes and output
/// row `c + 16` in its high lanes.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn tile(
    src: *const i16,
    src_stride: usize,
    rows: usize,
    cols: usize,
    dst: *mut i16,
    dst_stride: usize,
) {
    let load = if cols == 32 {
        u32::MAX
    } else {
        (1u32 << cols) - 1
    };
    let lo: [__m512i; 4] =
        core::array::from_fn(|s| _mm512_loadu_si512(SWAP_LO[s].as_ptr() as *const __m512i));
    let hi: [__m512i; 4] =
        core::array::from_fn(|s| _mm512_loadu_si512(SWAP_HI[s].as_ptr() as *const __m512i));
    for half in 0..2 {
        let first = 16 * half;
        if first >= rows {
            break;
        }
        let live = (rows - first).min(16);
        let mut v = [_mm512_setzero_si512(); 16];
        macro_rules! rows {
            ($($i:literal),*) => {
                $(
                    if $i < live {
                        v[$i] = _mm512_maskz_loadu_epi16(load, src.add((first + $i) * src_stride));
                    }
                )*
            };
        }
        rows!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        macro_rules! stage {
            ($s:literal, $(($a:literal, $b:literal)),*) => {
                $(
                    let (p, q) = (v[$a], v[$b]);
                    v[$a] = _mm512_permutex2var_epi16(p, lo[$s], q);
                    v[$b] = _mm512_permutex2var_epi16(p, hi[$s], q);
                )*
            };
        }
        stage!(
            0,
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (12, 13),
            (14, 15)
        );
        stage!(
            1,
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (8, 10),
            (9, 11),
            (12, 14),
            (13, 15)
        );
        stage!(
            2,
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (8, 12),
            (9, 13),
            (10, 14),
            (11, 15)
        );
        stage!(
            3,
            (0, 8),
            (1, 9),
            (2, 10),
            (3, 11),
            (4, 12),
            (5, 13),
            (6, 14),
            (7, 15)
        );
        let store = ((1u32 << live) - 1) as u16;
        macro_rules! cols {
            ($($c:literal),*) => {
                $(
                    if $c < cols {
                        _mm256_mask_storeu_epi16(
                            dst.add($c * dst_stride + first),
                            store,
                            _mm512_castsi512_si256(v[$c]),
                        );
                    }
                    if $c + 16 < cols {
                        _mm256_mask_storeu_epi16(
                            dst.add(($c + 16) * dst_stride + first),
                            store,
                            _mm512_extracti64x4_epi64::<1>(v[$c]),
                        );
                    }
                )*
            };
        }
        cols!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
}

fn poly<'a>(w: &'a [Vector], at: At) -> &'a [i16; DEG] {
    &w[at.vector].polys[at.off]
}

impl Chain {
    /// `D_a` for every diagonal: `sum_t public_t[a] * s_t` over the products alone, reduced
    /// negacyclically in `X^DEG + 1`.
    pub fn sums(prep: &Prepared) -> Sums {
        let mut out = [[0i64; DEG]; BLOCKS];
        let mut acc = [0i64; SPREAD];
        for (a, d) in out.iter_mut().enumerate() {
            let rows: Vec<Rows> = prep
                .iter()
                .map(|r| Rows {
                    g: unsafe { r.g.data.as_ptr().add(a * SUB * r.terms) },
                    x: r.x.data.as_ptr(),
                    steps: r.terms / 32,
                    stride: (i32::MAX as i64
                        / (2 * PARITY as i64 * r.g.bound.max(1) * r.x.bound.max(1)))
                    .max(1) as usize,
                })
                .collect();
            acc.fill(0);
            unsafe {
                diagonal::<0>(&rows, &mut acc);
                diagonal::<8>(&rows, &mut acc);
                diagonal::<16>(&rows, &mut acc);
                diagonal::<24>(&rows, &mut acc);
                diagonal::<32>(&rows, &mut acc);
                diagonal::<40>(&rows, &mut acc);
                diagonal::<48>(&rows, &mut acc);
                diagonal::<56>(&rows, &mut acc);
                diagonal::<64>(&rows, &mut acc);
            }
            d.copy_from_slice(&acc[..DEG]);
            for j in DEG..SPREAD {
                d[j - DEG] -= acc[j];
            }
        }
        out
    }

    /// The same straight from the products, one sub-product at a time: the reference
    /// [`sums`](Self::sums) is checked against.
    pub fn reference_sums(&self, public: &[Blocks], w: &[Vector]) -> Sums {
        let mut out = [[0i64; DEG]; BLOCKS];
        for p in &self.products {
            let x = poly(w, p.at);
            for (a, d) in out.iter_mut().enumerate() {
                for (u, &g) in public[p.blocks][p.chunk][a].iter().enumerate() {
                    for j in 0..DEG - u {
                        d[j + u] += g as i64 * x[j] as i64;
                    }
                    for j in DEG - u..DEG {
                        d[j + u - DEG] -= g as i64 * x[j] as i64;
                    }
                }
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
                let q = poly(
                    w,
                    At {
                        vector: at.vector,
                        off: at.off + a,
                    },
                );
                let s = self.carries.gadget.base.pow(d as u32);
                for j in 0..DEG {
                    out[j] += s * q[j] as i64;
                }
            }
        }
        e
    }

    /// The exact left-hand side of every block equation.
    pub fn residuals(&self, public: &[Blocks], w: &[Vector]) -> [[i128; DEG]; BLOCKS] {
        let mut sums = self.reference_sums(public, w);
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
                let (t, sign) = if j + SUB < DEG {
                    (j + SUB, 1)
                } else {
                    (j + SUB - DEG, -1)
                };
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

/// Positions of one diagonal before the negacyclic wrap, `j + u` for `j < DEG` and `u < SUB`.
const SPREAD: usize = DEG + SUB - 1;
/// Taps of one parity, the most products one `i32` accumulator takes per step of 32 terms.
const PARITY: usize = SUB.div_ceil(2);
const _: () = assert!(SPREAD == 9 * 8);
const _: () = assert!(2 * (PARITY as i64) * BLOCK_LIMIT * COEFF_LIMIT <= i32::MAX as i64);

/// One run seen from one diagonal: the [`SUB`] rows of `g`, the `DEG` rows of `x`, both `steps`
/// vectors of 32 terms long, and the number of steps the `i32` accumulators may take between
/// widenings.
struct Rows {
    g: *const i16,
    x: *const i16,
    steps: usize,
    stride: usize,
}

/// Positions `T0 .. T0 + 8` of one diagonal over every run: position `t` collects
/// `g_u . x_{t - u}` for the taps `u`, the even and the odd taps into separate `i32` accumulators
/// (`vpdpwssd`) so that a step adds at most [`PARITY`] products to either, widened to `i64` every
/// `stride` steps. Sixteen accumulators and the nine `g` vectors of a step stay in registers, and
/// the `x` rows are read once per step.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn diagonal<const T0: usize>(rows: &[Rows], acc: &mut [i64; SPREAD]) {
    let mut wide = [_mm512_setzero_si512(); 8];
    for r in rows {
        let mut even = [_mm512_setzero_si512(); 8];
        let mut odd = [_mm512_setzero_si512(); 8];
        let mut since = 0;
        for k in 0..r.steps {
            let g: [__m512i; SUB] = core::array::from_fn(|u| {
                _mm512_loadu_si512(r.g.add(u * 32 * r.steps + 32 * k) as *const __m512i)
            });
            macro_rules! taps {
                ($($i:literal),*) => {
                    $(
                        tap::<T0, $i, 0>(&mut even, &g, r, k);
                        tap::<T0, $i, 1>(&mut odd, &g, r, k);
                        tap::<T0, $i, 2>(&mut even, &g, r, k);
                        tap::<T0, $i, 3>(&mut odd, &g, r, k);
                        tap::<T0, $i, 4>(&mut even, &g, r, k);
                        tap::<T0, $i, 5>(&mut odd, &g, r, k);
                        tap::<T0, $i, 6>(&mut even, &g, r, k);
                        tap::<T0, $i, 7>(&mut odd, &g, r, k);
                        tap::<T0, $i, 8>(&mut even, &g, r, k);
                    )*
                };
            }
            taps!(0, 1, 2, 3, 4, 5, 6, 7);
            since += 1;
            if since == r.stride {
                for i in 0..8 {
                    wide[i] = _mm512_add_epi64(wide[i], widen(_mm512_add_epi32(even[i], odd[i])));
                    even[i] = _mm512_setzero_si512();
                    odd[i] = _mm512_setzero_si512();
                }
                since = 0;
            }
        }
        for i in 0..8 {
            wide[i] = _mm512_add_epi64(wide[i], widen(even[i]));
            wide[i] = _mm512_add_epi64(wide[i], widen(odd[i]));
        }
    }
    for i in 0..8 {
        acc[T0 + i] += _mm512_reduce_add_epi64(wide[i]);
    }
}

/// `acc[I] += g_U . x_{T0 + I - U}` at step `k`, when that row of `x` exists.
#[inline(always)]
unsafe fn tap<const T0: usize, const I: usize, const U: usize>(
    acc: &mut [__m512i; 8],
    g: &[__m512i; SUB],
    r: &Rows,
    k: usize,
) {
    if U <= T0 + I && T0 + I - U < DEG {
        let x = _mm512_loadu_si512(r.x.add((T0 + I - U) * 32 * r.steps + 32 * k) as *const __m512i);
        acc[I] = _mm512_dpwssd_epi32(acc[I], g[U], x);
    }
}

#[inline(always)]
unsafe fn widen(a: __m512i) -> __m512i {
    _mm512_add_epi64(
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a)),
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(a)),
    )
}

const _: () = assert!(CHUNKS * CHUNK == N162);

//! Everything the recursion can compute from the commitment key alone.
//!
//! The `phi` of a chain constraint over a run of key rows is a function of the key and of nothing
//! else, so it is converted to `polx` once here — 432 buffers of `n` per limb, one per
//! `(component, twist, chunk, diagonal)` — and every proof aliases them by pointer. The same holds
//! for the quotients' constant multipliers and the carry weights, which come from the limb shapes,
//! for the nine-bit patterns that turn a binary lift into a table lookup instead of a transform,
//! and for the three commitment keys, whose ranks follow LaBRADOR's own SIS rule on the caps.
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{chunk, limbs, Blocks, Instance, BLOCKS, CHUNKS, DEG, FOLD_CAP, Q, SUB};
use crate::api::N162;
use crate::challenge::ShortChallenge;
use crate::fields::scalar::F162;
use crate::labrador::{sis_rank, sizeof_polx, CommitmentKey, PolxBuf, ShortPhi};
use crate::params::N;
use crate::scheme::{EvaluationPoint, FoldingChallenges, Params, PublicParameters};

/// LaBRADOR's own slack in front of a commitment's norm when it picks a rank, `6 T SLACK`.
const SIS_SLACK: f64 = 6.0 * 14.0 * 2.0;

pub struct Setup {
    /// Ring elements of one column, and columns.
    pub n: usize,
    pub r: usize,
    pub fold_cap: f64,
    pub limbs: Vec<limbs::Shape>,
    /// `blocks[(limb * 8 + part) * n + i]`, `part = component * 2 + twist`.
    blocks: Vec<Blocks>,
    /// `key_phi[((limb * 8 + part) * CHUNKS + b) * BLOCKS + a]`, a buffer of `n` sub-chunks
    /// of [`SUB`] `i16` -- the coefficient form LaBRADOR aggregates directly.
    key_phi: Vec<Arc<ShortPhi>>,
    scalars: BTreeMap<i64, Arc<PolxBuf>>,
    /// `carry_phi[(base, level)][a]`, a buffer of `BLOCKS` `polx`.
    carry_phi: BTreeMap<(i64, usize), Vec<Arc<PolxBuf>>>,
    pub ranks: Vec<usize>,
    pub caps: Vec<u64>,
    pub binary: Vec<bool>,
    pub supports: Vec<usize>,
    /// Polynomials of each vector that a constraint reads; the rest are padding and must vanish
    /// at every position.
    pub used: Vec<usize>,
    pub residues: Vec<usize>,
    pub left_expansion: usize,
    pub rest: Vec<usize>,
    pub key_y: CommitmentKey,
    pub key_u: CommitmentKey,
    pub key_r: CommitmentKey,
}

impl Setup {
    pub fn new(pp: &PublicParameters, params: &Params, seed: [u8; 32]) -> Setup {
        let key = pp.key();
        let (n, r) = (key.len_ring(), params.columns());
        let limbs: Vec<limbs::Shape> = params
            .primes()
            .iter()
            .map(|&q| limbs::Shape::of(q))
            .collect();
        let mut blocks = Vec::with_capacity(limbs.len() * 8 * n);
        for limb in 0..limbs.len() {
            let rows = limbs::key_rows(pp, limb);
            for k in 0..4 {
                for twist in 0..2 {
                    for i in 0..n {
                        let mut g = rows.rows[i][k];
                        if twist == 1 {
                            g = chunk::shift(&g, 1);
                            g.iter_mut().for_each(|x| *x = -*x);
                        }
                        blocks.push(chunk::blocks(&g));
                    }
                }
            }
        }
        let empty = || CommitmentKey::expand(1, 1, &[0u8; 16], 0);
        let mut setup = Setup {
            n,
            r,
            fold_cap: FOLD_CAP * (n * N * r) as f64,
            limbs,
            blocks,
            key_phi: Vec::new(),
            scalars: BTreeMap::new(),
            carry_phi: BTreeMap::new(),
            ranks: Vec::new(),
            caps: Vec::new(),
            binary: Vec::new(),
            supports: Vec::new(),
            used: Vec::new(),
            residues: Vec::new(),
            left_expansion: 0,
            rest: Vec::new(),
            key_y: empty(),
            key_u: empty(),
            key_r: empty(),
        };

        let quiet = FoldingChallenges::of(vec![ShortChallenge::from_coeffs(&[0i8; N162]); r]);
        let origin = EvaluationPoint::of(
            vec![F162::ZERO; params.row_log_len() as usize],
            vec![F162::ZERO; params.column_log_len as usize],
        );
        let layout = Instance::layout(&setup, &quiet, &origin, &F162::ZERO);
        setup.ranks = layout.vectors.iter().map(|v| v.polys.len()).collect();
        setup.caps = layout
            .vectors
            .iter()
            .map(|v| (v.cap() * v.cap()).ceil() as u64)
            .collect();
        setup.binary = layout.vectors.iter().map(|v| v.binary).collect();
        setup.supports = layout.vectors.iter().map(|v| v.support).collect();
        setup.used = layout.vectors.iter().map(|v| v.used).collect();
        setup.residues = layout.hooks.residues.clone();
        setup.left_expansion = layout.hooks.left_expansion;
        setup.rest = layout.hooks.rest.clone();

        let key_of = |group: &[usize], label: &[u8]| {
            let norm: f64 = group
                .iter()
                .map(|&i| setup.caps[i] as f64)
                .sum::<f64>()
                .sqrt();
            let length: usize = group.iter().map(|&i| setup.ranks[i]).sum();
            let mut h = blake3::Hasher::new();
            h.update(b"bin-ntt/recursion/key");
            h.update(&seed);
            h.update(label);
            let mut sub = [0u8; 16];
            sub.copy_from_slice(&h.finalize().as_bytes()[..16]);
            CommitmentKey::expand(length, sis_rank(SIS_SLACK * norm), &sub, 0)
        };
        let key_y = key_of(&setup.residues, b"Y");
        let key_u = key_of(&[setup.left_expansion], b"u");
        let key_r = key_of(&setup.rest, b"R");
        setup.key_y = key_y;
        setup.key_u = key_u;
        setup.key_r = key_r;

        let key_phi = (0..setup.blocks.len() / n)
            .flat_map(|part| (0..CHUNKS).flat_map(move |b| (0..BLOCKS).map(move |a| (part, b, a))))
            .map(|(part, b, a)| {
                let mut c = vec![0i16; n * SUB];
                for i in 0..n {
                    c[i * SUB..(i + 1) * SUB].copy_from_slice(&setup.blocks[part * n + i][b][a]);
                }
                Arc::new(ShortPhi::new(0, SUB, c))
            })
            .collect();
        setup.key_phi = key_phi;

        chunk::taps();
        setup.scalar(0);
        for c in &layout.chains {
            for s in &c.scaled {
                setup.scalar(s.factor);
            }
            for d in 0..c.carries.at.len() {
                setup.carry(c.carries.gadget.base, d);
            }
        }
        setup
    }

    pub fn key_blocks(&self, limb: usize, part: usize) -> &[Blocks] {
        let at = (limb * 8 + part) * self.n;
        &self.blocks[at..at + self.n]
    }

    pub fn key_phi(&self, limb: usize, part: usize, b: usize, a: usize) -> &Arc<ShortPhi> {
        &self.key_phi[((limb * 8 + part) * CHUNKS + b) * BLOCKS + a]
    }

    fn scalar(&mut self, x: i64) {
        if !self.scalars.contains_key(&x) {
            let mut c = [0i64; DEG];
            c[0] = (x as i128).rem_euclid(Q) as i64;
            self.scalars.insert(x, Arc::new(PolxBuf::from_int64(&[c])));
        }
    }

    pub fn scalar_phi(&self, x: i64) -> &Arc<PolxBuf> {
        self.scalars
            .get(&x)
            .expect("this constant was not prepared at key time")
    }

    /// `+base^d` at the previous diagonal, `-base^d X^SUB` at this one, and the `Phi_243` wrap of
    /// the last carry at diagonals `0` and `BLOCKS / 2`.
    fn carry(&mut self, base: i64, d: usize) {
        if self.carry_phi.contains_key(&(base, d)) {
            return;
        }
        let w = base.pow(d as u32) as i128;
        let buffers = (0..BLOCKS)
            .map(|a| {
                let polys: Vec<[i64; DEG]> = (0..BLOCKS)
                    .map(|x| {
                        let mut e = [0i128; DEG];
                        if a > 0 && x == a - 1 {
                            e[0] += w;
                        }
                        if x == a {
                            e[SUB] -= w;
                        }
                        if (a == 0 || a == BLOCKS / 2) && x == BLOCKS - 1 {
                            e[0] -= w;
                        }
                        core::array::from_fn(|t| e[t].rem_euclid(Q) as i64)
                    })
                    .collect();
                Arc::new(PolxBuf::from_int64(&polys))
            })
            .collect();
        self.carry_phi.insert((base, d), buffers);
    }

    pub fn carry_phi(&self, base: i64, d: usize, a: usize) -> &Arc<PolxBuf> {
        &self
            .carry_phi
            .get(&(base, d))
            .expect("this carry level was not prepared at key time")[a]
    }

    /// The `phi` of one group of binary lifts at chunk `b` and diagonal `a`. A lift's sub-chunk
    /// is a signed sum of the nine-bit windows [`chunk::taps`] names, so its [`SUB`]
    /// coefficients are read off the window bits directly.
    pub fn lift_phi(&self, windows: &[u16], len: usize, b: usize, a: usize) -> ShortPhi {
        let taps = &chunk::taps()[b][a];
        let mut c = vec![0i16; len * SUB];
        for i in 0..len {
            for t in taps {
                let w = windows[i * (N162 / SUB) + t.0 as usize];
                for (u, x) in c[i * SUB..(i + 1) * SUB].iter_mut().enumerate() {
                    *x += t.1 as i16 * ((w >> u) & 1) as i16;
                }
            }
        }
        ShortPhi::new(0, SUB, c)
    }

    /// Bytes held by everything above, the memory a key costs before any proof.
    pub fn footprint(&self) -> usize {
        let polx = sizeof_polx();
        self.key_phi.iter().map(|p| p.bytes()).sum::<usize>()
            + self
                .carry_phi
                .values()
                .flatten()
                .map(|p| p.len() * polx)
                .sum::<usize>()
            + self.blocks.len() * core::mem::size_of::<Blocks>()
            + (self.key_y.buf().len() + self.key_u.buf().len() + self.key_r.buf().len()) * polx
    }
}

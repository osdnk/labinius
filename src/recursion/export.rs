//! The statement and witness the LaBRADOR front end consumes.
//!
//! A block equation becomes one degree-1 constraint `sum_j phi_j . s_j = b` over runs of witness
//! polynomials; `phi` is listed in block order, one 64-coefficient element per witness polynomial,
//! as plain centred integers (LaBRADOR reduces them). Blocks whose `phi` is a function of the
//! commitment key alone carry `key_time` and a `tag`: two blocks with the same nonzero tag have
//! identical `phi`, so the FFI layer converts them to `polx` once and aliases them afterwards.
use super::chain::{At, Chain};
use super::{Instance, BLOCKS, CARRY, DEG, SPAN, SUB};

/// One witness vector.
#[derive(Clone, Debug)]
pub struct VectorSpec {
    pub name: String,
    /// Polynomials of `Z_Q[X]/(X^64 + 1)`.
    pub n: usize,
    /// The exact `‖s‖^2` the prover announces.
    pub betasq: u64,
    /// The cap the verifier enforces, as a squared norm.
    pub cap_betasq: u64,
    pub binary: bool,
}

/// A run of witness polynomials a constraint touches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Block {
    pub idx: usize,
    pub off: usize,
    pub len: usize,
    pub key_time: bool,
    /// Nonzero when the block's `phi` is shared: equal tags mean identical `phi`.
    pub tag: u64,
}

/// One LaBRADOR constraint.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub name: String,
    pub deg: usize,
    pub blocks: Vec<Block>,
    /// One element per witness polynomial of `blocks`, concatenated in block order.
    pub phi: Vec<[i64; DEG]>,
    pub b: Option<Vec<[i64; DEG]>>,
}

/// Everything the front end needs that is public.
#[derive(Clone, Debug)]
pub struct Statement {
    pub vectors: Vec<VectorSpec>,
    pub constraints: Vec<Constraint>,
    /// The witness vectors the pre-commitments of the plan's sections 1, 2c and 4 bind.
    pub residues: Vec<usize>,
    pub left_expansion: usize,
    pub rest: Vec<usize>,
}

/// The witness itself: `n * 64` centred coefficients per vector, polynomial-major.
#[derive(Clone, Debug)]
pub struct Witness {
    pub vectors: Vec<Vec<i16>>,
}

/// LaBRADOR's `int16` norm limit.
pub const INT16_LIMIT: i16 = 23170;

fn tag(first: usize, len: usize, chunk: usize, block: usize) -> u64 {
    (((first as u64 * 64 + len as u64) * 8 + chunk as u64) * 64 + block as u64) + 1
}

impl Instance {
    pub fn witness(&self) -> Witness {
        Witness {
            vectors: self
                .vectors
                .iter()
                .map(|v| {
                    let w: Vec<i16> = v.polys.iter().flatten().copied().collect();
                    assert!(w.iter().all(|&x| x.abs() <= INT16_LIMIT), "{} overflows int16", v.name);
                    w
                })
                .collect(),
        }
    }

    pub fn statement(&self) -> Statement {
        let vectors = self
            .vectors
            .iter()
            .map(|v| VectorSpec {
                name: v.name.clone(),
                n: v.polys.len(),
                betasq: v.betasq(),
                cap_betasq: (v.cap() * v.cap()).ceil() as u64,
                binary: v.binary,
            })
            .collect();
        let constraints = self
            .chains
            .iter()
            .flat_map(|c| (0..BLOCKS).map(move |a| self.constraint(c, a)))
            .collect();
        Statement {
            vectors,
            constraints,
            residues: self.hooks.residues.clone(),
            left_expansion: self.hooks.left_expansion,
            rest: self.hooks.rest.clone(),
        }
    }

    fn constraint(&self, c: &Chain, a: usize) -> Constraint {
        let mut blocks: Vec<Block> = Vec::new();
        let mut phi: Vec<[i64; DEG]> = Vec::new();
        let mut run: Option<(At, usize, usize, usize)> = None;
        for p in &c.products {
            let extend = match run {
                Some((at, len, first, chunk)) => {
                    at.vector == p.at.vector
                        && at.off + len == p.at.off
                        && first + len == p.blocks
                        && chunk == p.chunk
                }
                None => false,
            };
            if extend {
                let r = run.as_mut().unwrap();
                r.1 += 1;
            } else {
                if let Some((at, len, first, chunk)) = run {
                    blocks.push(Block {
                        idx: at.vector,
                        off: at.off,
                        len,
                        key_time: self.key_time[first],
                        tag: if self.key_time[first] { tag(first, len, chunk, a) } else { 0 },
                    });
                }
                run = Some((p.at, 1, p.blocks, p.chunk));
            }
            let mut e = [0i64; DEG];
            for u in 0..SUB {
                e[u] = self.public[p.blocks][p.chunk][a][u] as i64;
            }
            phi.push(e);
        }
        if let Some((at, len, first, chunk)) = run {
            blocks.push(Block {
                idx: at.vector,
                off: at.off,
                len,
                key_time: self.key_time[first],
                tag: if self.key_time[first] { tag(first, len, chunk, a) } else { 0 },
            });
        }
        for s in &c.scaled {
            let mut e = [0i64; DEG];
            if a == SPAN * s.chunk {
                e[0] = s.factor;
            }
            blocks.push(Block { idx: s.at.vector, off: s.at.off, len: 1, key_time: false, tag: 0 });
            phi.push(e);
        }
        for (d, at) in c.carries.at.iter().enumerate() {
            let w = c.carries.gadget.base.pow(d as u32);
            blocks.push(Block { idx: at.vector, off: at.off, len: BLOCKS, key_time: false, tag: 0 });
            for x in 0..BLOCKS {
                let mut e = [0i64; DEG];
                if a > 0 && x == a - 1 {
                    e[0] += w;
                }
                if x == a {
                    e[SUB] -= w;
                }
                if (a == 0 || a == BLOCKS / 2) && x == BLOCKS - 1 {
                    e[0] -= w;
                }
                phi.push(e);
            }
        }
        let mut b = [0i64; DEG];
        b[..SUB].copy_from_slice(&c.output[SUB * a..SUB * a + SUB]);
        Constraint {
            name: format!("{} diagonal {a}", c.name),
            deg: 1,
            blocks,
            phi,
            b: (b != [0i64; DEG]).then(|| vec![b]),
        }
    }
}

const _: () = assert!(CARRY + SUB <= DEG);

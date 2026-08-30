//! The witness the LaBRADOR front end consumes, and the same constraints as plain integers.
//!
//! A block equation becomes one degree-1 constraint `sum_j phi_j . s_j = b` over runs of witness
//! polynomials: [`Instance::runs`] cuts a chain's products into the longest runs that a single
//! `phi` block covers — consecutive polynomials of one witness vector against consecutive elements
//! of one public group, at one chunk — and both this module and [`super::statement`] work from
//! those. Here `phi` is listed as plain centred integers, one 64-coefficient element per witness
//! polynomial, which is the form the reference checker and the tests read.
use super::chain::{At, Chain};
use super::{Instance, Kind, BLOCKS, CARRY, DEG, SPAN, SUB};

/// One witness vector.
#[derive(Clone, PartialEq, Eq, Debug)]
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
}

/// One LaBRADOR constraint.
#[derive(Clone, PartialEq, Eq, Debug)]
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

/// Consecutive polynomials of one witness vector against consecutive elements of one public
/// group, at one chunk: what a single `phi` block covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Run {
    pub group: usize,
    /// Index of the first public element within its group.
    pub offset: usize,
    pub chunk: usize,
    pub at: At,
    pub len: usize,
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

    pub fn runs(&self, c: &Chain) -> Vec<Run> {
        let mut runs: Vec<Run> = Vec::new();
        for p in &c.products {
            let group = self.group_of[p.blocks];
            let first = self.groups[group].first;
            let extend = runs.last().is_some_and(|r| {
                r.group == group
                    && r.chunk == p.chunk
                    && r.at.vector == p.at.vector
                    && r.at.off + r.len == p.at.off
                    && first + r.offset + r.len == p.blocks
            });
            if extend {
                runs.last_mut().unwrap().len += 1;
            } else {
                runs.push(Run {
                    group,
                    offset: p.blocks - first,
                    chunk: p.chunk,
                    at: p.at,
                    len: 1,
                });
            }
        }
        runs
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
            .flat_map(|c| {
                let runs = self.runs(c);
                (0..BLOCKS).map(move |a| self.constraint(c, &runs, a))
            })
            .collect();
        Statement {
            vectors,
            constraints,
            residues: self.hooks.residues.clone(),
            left_expansion: self.hooks.left_expansion,
            rest: self.hooks.rest.clone(),
        }
    }

    fn constraint(&self, c: &Chain, runs: &[Run], a: usize) -> Constraint {
        let mut blocks: Vec<Block> = Vec::new();
        let mut phi: Vec<[i64; DEG]> = Vec::new();
        for r in runs {
            let group = &self.groups[r.group];
            blocks.push(Block {
                idx: r.at.vector,
                off: r.at.off,
                len: r.len,
                key_time: matches!(group.kind, Kind::Key { .. }),
            });
            for i in 0..r.len {
                let mut e = [0i64; DEG];
                for (u, x) in self.public[group.first + r.offset + i][r.chunk][a].iter().enumerate()
                {
                    e[u] = *x as i64;
                }
                phi.push(e);
            }
        }
        for s in &c.scaled {
            let mut e = [0i64; DEG];
            if a == SPAN * s.chunk {
                e[0] = s.factor;
            }
            blocks.push(Block { idx: s.at.vector, off: s.at.off, len: 1, key_time: false });
            phi.push(e);
        }
        for (d, at) in c.carries.at.iter().enumerate() {
            let w = c.carries.gadget.base.pow(d as u32);
            blocks.push(Block { idx: at.vector, off: at.off, len: BLOCKS, key_time: false });
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

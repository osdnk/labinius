//! The LaBRADOR statement of one round, built the same way by the prover and by the verifier.
//!
//! Nothing here reads the witness: the layout instance of [`Instance::layout`] already fixes every
//! block, and the only per-proof numbers are the three pre-commitments, the announced norms and
//! the verifier's mask scalars. The `phi` of a run of key rows is aliased out of [`Setup`]; the
//! challenges are converted in bulk once per `(chunk, diagonal)` and shared by every chain; the
//! binary lifts are assembled out of the nine-bit pattern table.
use std::sync::Arc;

use super::setup::Setup;
use super::{Instance, Kind, BLOCKS, CHUNKS, DEG, Q, SPAN, SUB};
use crate::challenge::Transcript;
use crate::labrador::{
    logq, BSource, Block, Constraint, PhiBlock, PhiSource, PolxBuf, ShortPhi, Statement, VectorSpec,
};

/// Independent repetitions of the zero-part test, `ceil(128 / LOGQ)`.
pub fn lifts() -> usize {
    128usize.div_ceil(logq())
}

/// The per-proof `phi`: one buffer per `(group, chunk, diagonal)` for every group that is not a
/// function of the key alone.
pub struct ProofPhi {
    buffers: Vec<Vec<Arc<ShortPhi>>>,
}

impl ProofPhi {
    pub fn new(setup: &Setup, layout: &Instance) -> ProofPhi {
        let buffers = layout
            .groups
            .iter()
            .enumerate()
            .map(|(g, group)| match group.kind {
                Kind::Key { .. } => Vec::new(),
                Kind::Lift => (0..CHUNKS)
                    .flat_map(|b| (0..BLOCKS).map(move |a| (b, a)))
                    .map(|(b, a)| Arc::new(setup.lift_phi(&layout.windows[g], group.len, b, a)))
                    .collect(),
                Kind::Challenge | Kind::Loose => (0..CHUNKS)
                    .flat_map(|b| (0..BLOCKS).map(move |a| (b, a)))
                    .map(|(b, a)| {
                        let mut c = vec![0i16; group.len * SUB];
                        for i in 0..group.len {
                            c[i * SUB..(i + 1) * SUB]
                                .copy_from_slice(&layout.public[group.first + i][b][a]);
                        }
                        Arc::new(ShortPhi::new(0, SUB, c))
                    })
                    .collect(),
            })
            .collect();
        ProofPhi { buffers }
    }

    fn get(&self, group: usize, b: usize, a: usize) -> &Arc<ShortPhi> {
        &self.buffers[group][b * BLOCKS + a]
    }

    pub fn footprint(&self) -> usize {
        self.buffers.iter().flatten().map(|p| p.bytes()).sum()
    }
}

/// The verifier's message: `lifts()` independent uniform scalars for every position at which a
/// witness chunk claims to be zero, already folded into one `phi` per repetition.
pub struct Masks {
    pub rows: Vec<Vec<[i64; DEG]>>,
}

impl Masks {
    /// `phi_xi = sum_j rho_{xi,j} X^{-j}` over the zero positions `j` of the chunk `xi`, so that
    /// the constant coefficient of `sum_xi phi_xi xi` is `sum rho_{xi,j} xi_j`.
    pub fn squeeze(setup: &Setup, transcript: &mut Transcript) -> Masks {
        let zeros = |i: usize, p: usize| {
            if p < setup.used[i] {
                setup.supports[i]
            } else {
                0
            }
        };
        let total: usize = (0..setup.ranks.len())
            .flat_map(|i| (0..setup.ranks[i]).map(move |p| (i, p)))
            .map(|(i, p)| DEG - zeros(i, p))
            .sum();
        let rows = (0..lifts())
            .map(|h| {
                let mut bytes = vec![0u8; 12 * total];
                transcript.fill(format!("labinius/recursion/mask/{h}").as_bytes(), &mut bytes);
                let mut at = 0;
                let mut row = Vec::with_capacity(setup.ranks.iter().sum());
                for (i, &n) in setup.ranks.iter().enumerate() {
                    for p in 0..n {
                        let mut e = [0i64; DEG];
                        for j in zeros(i, p)..DEG {
                            let mut w = [0u8; 16];
                            w[..12].copy_from_slice(&bytes[at..at + 12]);
                            at += 12;
                            let rho = (u128::from_le_bytes(w) % (Q as u128)) as i128;
                            let (at, x) = if j == 0 { (0, rho) } else { (DEG - j, -rho) };
                            e[at] = x.rem_euclid(Q) as i64;
                        }
                        row.push(e);
                    }
                }
                row
            })
            .collect();
        Masks { rows }
    }
}

/// The three pre-commitments and the exact norms the prover announces.
pub struct Opening<'a> {
    pub t_y: &'a Arc<PolxBuf>,
    pub t_u: &'a Arc<PolxBuf>,
    pub t_r: &'a Arc<PolxBuf>,
    pub norms: &'a [u64],
}

/// Chains whose block equations are emitted diagonal by diagonal rather than chain by chain.
///
/// A key `phi` buffer is read by exactly two constraints — the two output components whose
/// `(k, twist)` it is — at the same diagonal, and those two are `BLOCKS` constraints and 60 MB
/// apart in chain order. Interleaving a limb's four components brings them four constraints
/// apart, and the limb's whole diagonal (`2 * 4 * CHUNKS * n` polx, 12 MB) then stays in the
/// last-level cache across them: `labrador::prove` 394 -> 374 ms, `verify` 233 -> 221 ms.
/// Any order is sound; prover and verifier run this same function.
const FAMILY: usize = 4;

fn families(layout: &Instance) -> Vec<Vec<usize>> {
    (0..layout.chains.len())
        .step_by(FAMILY)
        .map(|at| (at..(at + FAMILY).min(layout.chains.len())).collect())
        .collect()
}

/// Dachshund reads `betasq == 0` as the binariness flag, so a witness vector that happens to be
/// all zero announces `1`; the no-wraparound bound reads the caps, not the announced norms.
pub fn build(
    setup: &Setup,
    layout: &Instance,
    phi: &ProofPhi,
    opening: Opening,
    masks: Masks,
    digest: [u8; 32],
) -> Statement {
    let vectors = (0..setup.ranks.len())
        .map(|i| {
            if setup.binary[i] {
                VectorSpec::binary(setup.ranks[i])
            } else {
                VectorSpec::norm_bounded(setup.ranks[i], opening.norms[i].max(1))
            }
        })
        .collect();
    let mut constraints: Vec<Constraint> = Vec::new();
    let runs: Vec<Vec<crate::recursion::export::Run>> =
        layout.chains.iter().map(|c| layout.runs(c)).collect();
    for family in families(layout) {
        for a in 0..BLOCKS {
            for &ci in &family {
                let c = &layout.chains[ci];
                let runs = &runs[ci];
                let mut blocks =
                    Vec::with_capacity(runs.len() + c.scaled.len() + c.carries.at.len());
                let mut parts = Vec::with_capacity(blocks.capacity());
                for r in runs {
                    blocks.push(Block::new(r.at.vector, r.at.off, r.len));
                    parts.push(match layout.groups[r.group].kind {
                        Kind::Key { limb, part } => PhiBlock::Short(
                            Arc::clone(setup.key_phi(limb, part, r.chunk, a)),
                            r.offset,
                        ),
                        _ => PhiBlock::Short(Arc::clone(phi.get(r.group, r.chunk, a)), r.offset),
                    });
                }
                for s in &c.scaled {
                    blocks.push(Block::new(s.at.vector, s.at.off, 1));
                    let factor = if a == SPAN * s.chunk { s.factor } else { 0 };
                    parts.push(PhiBlock::Polx(Arc::clone(setup.scalar_phi(factor)), 0));
                }
                for (d, at) in c.carries.at.iter().enumerate() {
                    blocks.push(Block::new(at.vector, at.off, BLOCKS));
                    parts.push(PhiBlock::Polx(
                        Arc::clone(setup.carry_phi(c.carries.gadget.base, d, a)),
                        0,
                    ));
                }
                let mut b = [0i64; DEG];
                for (u, x) in c.output[SUB * a..SUB * a + SUB].iter().enumerate() {
                    b[u] = (*x as i128).rem_euclid(Q) as i64;
                }
                let b = (b != [0i64; DEG]).then(|| BSource::Int64(vec![b]));
                constraints.push(Constraint::new(1, blocks, PhiSource::Blocks(parts), b));
            }
        }
    }

    let mut commitment =
        |key: &crate::labrador::CommitmentKey, group: &[usize], t: &Arc<PolxBuf>| {
            let blocks = group
                .iter()
                .map(|&i| Block::new(i, 0, setup.ranks[i]))
                .collect();
            constraints.push(Constraint::new(
                key.rank(),
                blocks,
                PhiSource::polx(key.buf_arc()),
                Some(BSource::Polx(Arc::clone(t))),
            ));
        };
    commitment(&setup.key_y, &setup.residues, opening.t_y);
    commitment(&setup.key_u, &[setup.left_expansion], opening.t_u);
    commitment(&setup.key_r, &setup.rest, opening.t_r);

    for row in masks.rows {
        let blocks = (0..setup.ranks.len())
            .map(|i| Block::new(i, 0, setup.ranks[i]))
            .collect();
        constraints.push(Constraint::new(0, blocks, PhiSource::Int64(row), None));
    }
    let statement = Statement::with_digest(vectors, constraints, digest);
    statement.precompute();
    statement
}

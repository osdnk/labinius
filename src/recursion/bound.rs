//! The no-wraparound condition: every block equation's integer left-hand side stays below `Q / 2`
//! for every witness the caps allow.
//!
//! Coefficient `t` of the left-hand side of a block equation is `sum_i <row_{t,i}, s_i>` over the
//! witness vectors, `row_{t,i}` collecting the `phi` coefficients that feed position `t` from
//! vector `i`. Cauchy-Schwarz gives `|.| <= sum_i ‖row_{t,i}‖_2 sqrt(betasq_i)`, exact over the
//! caps and attained by a sign-aligned witness, and that is what this module computes from the
//! generated blocks.
use super::{Instance, BLOCKS, CARRY, DEG, Q, SPAN, SUB};

/// The worst position of one chain.
#[derive(Clone, Debug)]
pub struct ChainBound {
    pub name: String,
    pub block: usize,
    pub position: usize,
    pub value: f64,
    /// The vector contributing most, and its share.
    pub worst: String,
    pub share: f64,
}

impl ChainBound {
    /// The factor by which the chain clears `Q / 2`.
    pub fn margin(&self) -> f64 {
        (Q as f64) / 2.0 / self.value
    }
}

impl Instance {
    /// The bound of the plan's section 5, one entry per chain.
    pub fn bound(&self) -> Vec<ChainBound> {
        self.chains
            .iter()
            .map(|c| {
                let mut best = ChainBound {
                    name: c.name.clone(),
                    block: 0,
                    position: 0,
                    value: 0.0,
                    worst: String::new(),
                    share: 0.0,
                };
                let mut sq = vec![0f64; self.vectors.len()];
                let mut acc = vec![[0f64; SUB]; self.vectors.len()];
                for a in 0..BLOCKS {
                    acc.iter_mut().for_each(|x| *x = [0f64; SUB]);
                    for p in &c.products {
                        let g = &self.public[p.blocks][p.chunk][a];
                        let row = &mut acc[p.at.vector];
                        for (u, &x) in g.iter().enumerate() {
                            row[u] += (x as f64) * (x as f64);
                        }
                    }
                    for t in 0..DEG {
                        for (i, row) in acc.iter().enumerate() {
                            let support = self.vectors[i].support;
                            sq[i] = (0..SUB)
                                .filter(|&u| t >= u && t - u < support)
                                .map(|u| row[u])
                                .sum();
                        }
                        for s in &c.scaled {
                            if a == SPAN * s.chunk && t < self.vectors[s.at.vector].support {
                                sq[s.at.vector] += (s.factor as f64) * (s.factor as f64);
                            }
                        }
                        for (d, at) in c.carries.at.iter().enumerate() {
                            let w = (c.carries.gadget.base.pow(d as u32) as f64).powi(2);
                            let mut n = 0.0;
                            if a > 0 && t < CARRY {
                                n += w;
                            }
                            if t >= SUB && t - SUB < CARRY {
                                n += w;
                            }
                            if (a == 0 || a == BLOCKS / 2) && t < CARRY {
                                n += w;
                            }
                            sq[at.vector] += n;
                        }
                        let mut total = 0.0;
                        let mut worst = (0.0, 0usize);
                        for (i, &x) in sq.iter().enumerate() {
                            let term = x.sqrt() * self.vectors[i].cap();
                            total += term;
                            if term > worst.0 {
                                worst = (term, i);
                            }
                        }
                        if total > best.value {
                            best.value = total;
                            best.block = a;
                            best.position = t;
                            best.worst = self.vectors[worst.1].name.clone();
                            best.share = worst.0 / total;
                        }
                    }
                }
                best
            })
            .collect()
    }

    /// Does every block equation clear `Q / 2`?
    pub fn clears(&self) -> bool {
        self.bound().iter().all(|b| b.value < (Q as f64) / 2.0)
    }
}

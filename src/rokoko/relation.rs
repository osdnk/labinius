//! The relation of [`crate::recursion`] over one committed vector of rokoko ring elements: the
//! layout, the honest witness, the block equations as public weights, and the no-wraparound
//! bound. `docs/rokoko.md` fixes the geometry.
//!
//! The vectors, in layout order: `v^0`, `v^1` (`v = v^0 + DIGIT v^1`, `S`-element `l n + i` is
//! component `l` of ring element `i`), `u` (`S`-element `j` the lift of row value `j`), then per
//! limb `q`: `C[q][m]^0`, `C[q][m]^1` for `m = 0..4` (`S`-element `j` the digits of the centred
//! residue of column `j`), `k[q]` (`S`-element `m levels + d` the digit `d` of the quotient of
//! component `m`), `e[q]` (element `(m levels + d) BLOCKS + a` the digit `d` of carry `a` of
//! component `m`); then `w`, `e[w]`, `w'`, `e[w']` for the binary fold and evaluation. An
//! `S`-element `s` of a vector occupies elements `CHUNKS s + b`, `b` its chunk.
//!
//! The weight polynomials of a round, `Relation::polys`: the key's block atlas of [`Setup`]
//! first (`fixed` of them), then `[1]`, `-Z^SUB`, and the blocks of the negated challenges and of
//! the lifts, twelve consecutive polys `(b, a)` per public `S`-element.
use super::{
    BlockEquations, Cap, Diagonal, Element, Entry, Gadget, Layout, Overflow, Poly, Region,
    Relation, SElem, Vector, Witness, BLOCKS, CARRY, CHUNK, CHUNKS, DEG, DIGIT, SPAN, SUB, SUPPORT,
};
use crate::api::{PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162};
use crate::eval::eq_table;
use crate::fields::scalar::F162;
use crate::params::N;
use crate::recursion::{binary, centre, chunk, limbs, FOLD_CAP};
use crate::scheme::{
    EvaluationPoint, FoldedWitness, FoldingChallenges, PublicParameters, RowEvaluation,
};

/// A quotient is the same `S`-element whatever the geometry, so its magnitude is the one
/// [`limbs`] and [`binary`] measured. A carry is a partial sum of the same products cut at a
/// multiple of `SUB` rather than of 9, so it keeps three times the constants measured there until
/// calibrated over rounds at this geometry (`GADGET_STATS`, as `examples/gadget.rs`); one round
/// at the basic shape measured `93 q sqrt(n r)` and `7.2 n r`, below the `SUB = 9` constants.
pub const CARRY_PER_ROOT: f64 = 3.0 * limbs::CARRY_PER_ROOT;
pub const CARRY_PER_TERM: f64 = 3.0 * binary::CARRY_PER_TERM;

const HALF: f64 = (DIGIT / 2) as f64;

const V0: usize = 0;
const V1: usize = 1;
const U: usize = 2;
const FIRST_LIMB: usize = 3;
const PER_LIMB: usize = 10;
/// Blocks of one public `S`-element, `(b, a)` at `b * BLOCKS + a`.
const PER_ELEMENT: usize = CHUNKS * BLOCKS;

/// The quotient and carry gadgets of one chain, base `DIGIT`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape {
    pub quotient: Gadget,
    pub carry: Gadget,
}

impl Shape {
    pub fn limb(q: u16, n: usize, r: usize) -> Shape {
        let root = ((n * r) as f64).sqrt();
        Shape {
            quotient: Gadget::covering(DIGIT, limbs::QUOTIENT_PER_ROOT * root),
            carry: Gadget::covering(DIGIT, CARRY_PER_ROOT * q as f64 * root),
        }
    }

    pub fn binary(n: usize, r: usize) -> Shape {
        let terms = (n * r) as f64;
        Shape {
            quotient: Gadget::covering(DIGIT, binary::QUOTIENT_PER_TERM * terms),
            carry: Gadget::covering(DIGIT, CARRY_PER_TERM * terms),
        }
    }
}

/// What both sides hold before any round.
pub struct Setup {
    /// Ring elements of one column, and columns.
    pub n: usize,
    pub r: usize,
    pub primes: Vec<u16>,
    /// The cap on `‖v‖^2` of the fold, [`FOLD_CAP`] per ring element and challenge.
    pub fold_cap: f64,
    pub limbs: Vec<Shape>,
    pub binary: Shape,
    /// The key's block atlas, `polys[block(limb, part, i, b, a)]`: block `(b, a)` of component
    /// `k` of key row `i` modulo the limb's prime, centred, times `-Z` when twisted,
    /// `part = 2 k + twist`. The fixed prefix of every round's `Relation::polys`.
    pub polys: Vec<Poly>,
}

impl Setup {
    pub fn new(pp: &PublicParameters) -> Setup {
        let (n, r) = (pp.key().len_ring(), pp.params().columns());
        let primes = pp.params().primes();
        let mut polys = Vec::with_capacity(primes.len() * 8 * n * PER_ELEMENT);
        for limb in 0..primes.len() {
            let rows = limbs::key_rows(pp, limb);
            for k in 0..4 {
                for twist in 0..2 {
                    for row in &rows.rows {
                        let mut g = row[k];
                        if twist == 1 {
                            g = chunk::shift(&g, 1);
                            g.iter_mut().for_each(|x| *x = -*x);
                        }
                        polys.extend(blocks(&g).iter().flatten().map(|b| b.to_vec()));
                    }
                }
            }
        }
        Setup {
            n,
            r,
            fold_cap: FOLD_CAP * (n * N * r) as f64,
            limbs: primes.iter().map(|&q| Shape::limb(q, n, r)).collect(),
            binary: Shape::binary(n, r),
            primes,
            polys,
        }
    }

    pub fn block(&self, limb: usize, part: usize, i: usize, b: usize, a: usize) -> usize {
        (((limb * 8 + part) * self.n + i) * CHUNKS + b) * BLOCKS + a
    }

    /// The first block of the multiplier of `v_{i,l}` in component `m` of a limb's identity:
    /// `F_{i,(m-l) mod 4}`, twisted iff `l > m`.
    pub fn key(&self, limb: usize, m: usize, l: usize, i: usize) -> usize {
        let k = (m + 4 - l) % 4;
        self.block(limb, 2 * k + usize::from(l > m), i, 0, 0)
    }

    /// Layout indices of `C[q][m]^0`, `C[q][m]^1` in the order of [`residue_vectors`].
    pub fn residue_vectors_at(&self) -> Vec<usize> {
        (0..self.primes.len())
            .flat_map(|limb| (0..8).map(move |x| FIRST_LIMB + PER_LIMB * limb + x))
            .collect()
    }

    /// Layout index of `u`.
    pub fn lift_vector_at(&self) -> usize {
        U
    }
}

// =============================================================================================
// S-elements, digits, chunks, blocks
// =============================================================================================

/// `x = low + DIGIT high`, `low` balanced.
fn pair(x: &SElem) -> [SElem; 2] {
    let low: SElem = core::array::from_fn(|t| centre(x[t], DIGIT));
    let high: SElem = core::array::from_fn(|t| (x[t] - low[t]) / DIGIT);
    [low, high]
}

/// The gadget digits of every coefficient, or the first coefficient out of reach.
fn split(x: &SElem, gadget: Gadget) -> Result<Vec<SElem>, i64> {
    let mut digits = vec![[0i64; N162]; gadget.levels];
    let mut d = vec![0i64; gadget.levels];
    for t in 0..N162 {
        if !gadget.try_split_into(x[t], &mut d) {
            return Err(x[t]);
        }
        for (l, digit) in digits.iter_mut().enumerate() {
            digit[t] = d[l];
        }
    }
    Ok(digits)
}

fn chunks(x: &SElem) -> [Element; CHUNKS] {
    core::array::from_fn(|b| {
        let mut e = [0i64; DEG];
        e[..CHUNK].copy_from_slice(&x[CHUNK * b..CHUNK * b + CHUNK]);
        e
    })
}

/// `blocks[b][a]` is coefficients `[SUB a, SUB a + SUB)` of `Z^{CHUNK b} g mod Phi_243`.
type Blocks = [[[i64; SUB]; BLOCKS]; CHUNKS];

fn blocks(g: &SElem) -> Blocks {
    core::array::from_fn(|b| {
        let h = chunk::shift(g, CHUNK * b);
        core::array::from_fn(|a| core::array::from_fn(|u| h[SUB * a + u]))
    })
}

/// The digits of the centred residues of one limb and component, low then high.
fn residue_digits(
    res: &limbs::Residues,
    limb: usize,
    m: usize,
    q: u16,
    r: usize,
) -> [Vec<Element>; 2] {
    let polys = &res.vectors[limb * 4 + m];
    let mut out = [
        Vec::with_capacity(CHUNKS * r),
        Vec::with_capacity(CHUNKS * r),
    ];
    for j in 0..r {
        let c = chunk::decode(&core::array::from_fn(|b| polys[b * r + j]));
        let x: SElem = core::array::from_fn(|t| centre(c[t], q as i64));
        for (digit, vector) in pair(&x).iter().zip(out.iter_mut()) {
            vector.extend(chunks(digit));
        }
    }
    out
}

/// The committed residue digit vectors, in layout order: per limb, `C[q][m]^0`, `C[q][m]^1` for
/// `m = 0..4`; `used` elements each.
pub fn residue_vectors(
    setup: &Setup,
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
) -> Vec<Vec<Element>> {
    let res = limbs::residues(matrix, &setup.primes);
    let mut out = Vec::with_capacity(8 * setup.primes.len());
    for (limb, &q) in setup.primes.iter().enumerate() {
        for m in 0..4 {
            out.extend(residue_digits(&res, limb, m, q, setup.r));
        }
    }
    out
}

/// The committed `u` vector: the chunks of the lifted row values.
pub fn lift_vector(setup: &Setup, row: &RowEvaluation) -> Vec<Element> {
    assert_eq!(row.values().len(), setup.r, "one row value per column");
    row.values()
        .iter()
        .flat_map(|x| chunks(&binary::lift(x)))
        .collect()
}

// =============================================================================================
// the diagonals and the carries
// =============================================================================================

type Sums = [[i64; DEG]; BLOCKS];
const WIDTH: usize = CHUNK + SUB - 1;
const _: () = assert!(WIDTH < DEG);
const _: () = assert!(SUB * (BLOCKS - 1) + WIDTH - N162 == CARRY);
const _: () = assert!(CARRY < SUPPORT);

/// `sum_a Z^{SUB a} D_a` as a plain polynomial.
fn assemble(sums: &Sums) -> Vec<i64> {
    let mut p = vec![0i64; SUB * (BLOCKS - 1) + WIDTH];
    for (a, d) in sums.iter().enumerate() {
        assert!(
            d[WIDTH..].iter().all(|&x| x == 0),
            "diagonal {a} exceeds its degree bound"
        );
        for j in 0..WIDTH {
            p[SUB * a + j] += d[j];
        }
    }
    p
}

/// `e_a = (D_a + e_{a-1} - [a = 0 or a = BLOCKS/2] e_last)[SUB .. SUB + CARRY]`, `e_last` the part
/// of the assembled polynomial at and above `Z^162`, which the recurrence reproduces.
fn honest_carries(sums: &Sums) -> [[i64; CARRY]; BLOCKS] {
    let p = assemble(sums);
    let mut last = [0i64; CARRY];
    last.copy_from_slice(&p[N162..]);
    let mut e = [[0i64; CARRY]; BLOCKS];
    let mut prev = [0i64; CARRY];
    for a in 0..BLOCKS {
        let mut t = [0i64; WIDTH];
        t.copy_from_slice(&sums[a][..WIDTH]);
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

fn stats(kind: &str, name: &str, values: impl Iterator<Item = i64>, gadget: Gadget) {
    let all: Vec<i64> = values.collect();
    let max = all.iter().map(|x| x.abs()).max().unwrap();
    let sd = (all.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / all.len() as f64).sqrt();
    eprintln!(
        "gadget-stats {kind:<8} {name:<24} max {max:>10} ({:.2} bits) sd {sd:>10.1} reach {} ({:.2} bits) fill {:.3}",
        (max as f64).log2(),
        gadget.reach(),
        (gadget.reach() as f64).log2(),
        max as f64 / gadget.reach() as f64
    );
}

// =============================================================================================
// the builder
// =============================================================================================

/// `scale` times the public element whose blocks start at `blocks`, against `S`-element `s` of
/// vector `vector`.
struct Term {
    blocks: usize,
    scale: i64,
    vector: usize,
    s: usize,
}

/// Quotient digits at `S`-elements `first + d` of `vector`.
struct Quotient {
    gadget: Gadget,
    vector: usize,
    first: usize,
}

/// Carries at elements `first + d * BLOCKS + a` of `vector`.
struct Carries {
    gadget: Gadget,
    vector: usize,
    first: usize,
}

struct Build {
    layout: Layout,
    witness: Option<Witness>,
    equations: Vec<BlockEquations>,
    polys: Vec<Poly>,
    /// `[1]` and `-Z^SUB`: the digit-term constants and the carry weights, under a scale.
    one: usize,
    minus_z: usize,
}

impl Build {
    fn new(setup: &Setup, witness: bool) -> Build {
        let mut polys = setup.polys.clone();
        let one = polys.len();
        polys.push(vec![1]);
        let mut p = vec![0i64; SUB + 1];
        p[SUB] = -1;
        polys.push(p);
        Build {
            layout: Layout {
                vectors: Vec::new(),
                regions: Vec::new(),
                len: 0,
            },
            witness: witness.then(Vec::new),
            equations: Vec::new(),
            polys,
            one,
            minus_z: one + 1,
        }
    }

    /// The blocks of a round's public element, appended; returns the first.
    fn blocks(&mut self, g: &SElem) -> usize {
        let first = self.polys.len();
        self.polys
            .extend(blocks(g).iter().flatten().map(|b| b.to_vec()));
        first
    }

    /// A vector of `used` elements, placed as `WitnessBuilder::push` places a power-of-two run.
    fn vector(&mut self, name: String, cap: Cap, binary: bool, used: usize) -> usize {
        let len = used.next_power_of_two();
        let start = self.layout.len.next_multiple_of(len);
        self.layout.vectors.push(Vector {
            name,
            cap,
            binary,
            used,
        });
        self.layout.regions.push(Region { start, len });
        self.layout.len = start + len;
        if let Some(w) = &mut self.witness {
            w.push(vec![[0i64; DEG]; used]);
        }
        self.layout.vectors.len() - 1
    }

    fn put(&mut self, vector: usize, elements: Vec<Element>) {
        if let Some(w) = &mut self.witness {
            assert_eq!(elements.len(), self.layout.vectors[vector].used);
            w[vector] = elements;
        }
    }

    fn set(&mut self, vector: usize, s: usize, x: &SElem) {
        if let Some(w) = &mut self.witness {
            w[vector][CHUNKS * s..CHUNKS * s + CHUNKS].copy_from_slice(&chunks(x));
        }
    }

    fn chain(
        &mut self,
        name: String,
        terms: &[Term],
        output: SElem,
        divisor: i64,
        quotient: Quotient,
        carries: Carries,
    ) -> Result<(), Overflow> {
        let stats_on = std::env::var_os("GADGET_STATS").is_some();
        let polys = &self.polys;
        if let Some(w) = self.witness.as_mut() {
            let mut sums = [[0i64; DEG]; BLOCKS];
            for t in terms {
                for b in 0..CHUNKS {
                    let x = &w[t.vector][CHUNKS * t.s + b];
                    for a in 0..BLOCKS {
                        for (u, &g) in polys[t.blocks + b * BLOCKS + a].iter().enumerate() {
                            if g != 0 {
                                let g = t.scale * g;
                                for j in 0..CHUNK {
                                    sums[a][u + j] += g * x[j];
                                }
                            }
                        }
                    }
                }
            }
            let value = chunk::reduce(&assemble(&sums));
            let k: SElem = core::array::from_fn(|t| {
                let x = value[t] - output[t];
                assert_eq!(
                    x % divisor,
                    0,
                    "{name}: the left side is not a multiple of {divisor}"
                );
                x / divisor
            });
            if stats_on {
                stats("quotient", &name, k.iter().copied(), quotient.gadget);
            }
            let digits = split(&k, quotient.gadget).map_err(|magnitude| Overflow {
                chain: name.clone(),
                magnitude,
                gadget: quotient.gadget,
            })?;
            for (d, digit) in digits.iter().enumerate() {
                let factor = -divisor * quotient.gadget.base.pow(d as u32);
                for (b, c) in chunks(digit).into_iter().enumerate() {
                    for j in 0..CHUNK {
                        sums[SPAN * b][j] += factor * c[j];
                    }
                    w[quotient.vector][CHUNKS * (quotient.first + d) + b] = c;
                }
            }
            let e = honest_carries(&sums);
            if stats_on {
                stats(
                    "carry",
                    &name,
                    e.iter().flat_map(|a| a.iter().copied()),
                    carries.gadget,
                );
            }
            let mut d = vec![0i64; carries.gadget.levels];
            for (a, carry) in e.iter().enumerate() {
                for (t, &x) in carry.iter().enumerate() {
                    if !carries.gadget.try_split_into(x, &mut d) {
                        return Err(Overflow {
                            chain: name,
                            magnitude: x,
                            gadget: carries.gadget,
                        });
                    }
                    for (l, &digit) in d.iter().enumerate() {
                        w[carries.vector][carries.first + l * BLOCKS + a][t] = digit;
                    }
                }
            }
        }
        let layout = &self.layout;
        let (one, minus_z) = (self.one, self.minus_z);
        let diagonals = (0..BLOCKS)
            .map(|a| {
                let mut entries = Vec::with_capacity(CHUNKS * terms.len() + 3 * BLOCKS);
                for t in terms {
                    for b in 0..CHUNKS {
                        entries.push(Entry {
                            element: layout.index(t.vector, CHUNKS * t.s + b),
                            poly: t.blocks + b * BLOCKS + a,
                            scale: t.scale,
                        });
                    }
                }
                if a % SPAN == 0 {
                    for d in 0..quotient.gadget.levels {
                        entries.push(Entry {
                            element: layout
                                .index(quotient.vector, CHUNKS * (quotient.first + d) + a / SPAN),
                            poly: one,
                            scale: -divisor * quotient.gadget.base.pow(d as u32),
                        });
                    }
                }
                for d in 0..carries.gadget.levels {
                    let power = carries.gadget.base.pow(d as u32);
                    let carry =
                        |x: usize| layout.index(carries.vector, carries.first + d * BLOCKS + x);
                    if a > 0 {
                        entries.push(Entry {
                            element: carry(a - 1),
                            poly: one,
                            scale: power,
                        });
                    }
                    entries.push(Entry {
                        element: carry(a),
                        poly: minus_z,
                        scale: power,
                    });
                    if a == 0 || a == BLOCKS / 2 {
                        entries.push(Entry {
                            element: carry(BLOCKS - 1),
                            poly: one,
                            scale: -power,
                        });
                    }
                }
                Diagonal {
                    entries,
                    output: output[SUB * a..SUB * a + SUB].to_vec(),
                }
            })
            .collect();
        self.equations.push(BlockEquations { name, diagonals });
        Ok(())
    }
}

// =============================================================================================
// the relation
// =============================================================================================

/// Verifier side: the layout and the block equations from public data alone. The layout is a
/// function of the shape (`n`, `r`, the primes, the gadget levels) and of nothing else, so any
/// challenges, point and claim give the same `Layout`; `polys[..fixed]` is `setup.polys`.
pub fn layout(
    setup: &Setup,
    challenges: &FoldingChallenges,
    point: &EvaluationPoint,
    claim: &F162,
) -> Relation {
    build(setup, None, challenges, point, claim)
        .expect("a layout-only build splits nothing")
        .0
}

/// Prover side: the same relation and the honest committed values. `Err` when a quotient or a
/// carry does not fit its gadget; the round is then retried with fresh challenges.
pub fn encode(
    setup: &Setup,
    commitment_matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    folded: &FoldedWitness,
    row: &RowEvaluation,
    challenges: &FoldingChallenges,
    point: &EvaluationPoint,
    claim: &F162,
) -> Result<(Relation, Witness), Overflow> {
    let (relation, witness) = build(
        setup,
        Some((commitment_matrix, folded, row)),
        challenges,
        point,
        claim,
    )?;
    Ok((relation, witness.expect("a witness was encoded")))
}

type Honest<'a> = (
    &'a VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    &'a FoldedWitness,
    &'a RowEvaluation,
);

fn build(
    setup: &Setup,
    honest: Option<Honest>,
    challenges: &FoldingChallenges,
    point: &EvaluationPoint,
    claim: &F162,
) -> Result<(Relation, Option<Witness>), Overflow> {
    let (n, r) = (setup.n, setup.r);
    assert_eq!(challenges.len(), r, "one folding challenge per column");
    let mut build = Build::new(setup, honest.is_some());
    let digit = Cap::PerCoefficient(HALF);

    let v0 = build.vector("v^0".into(), digit, false, CHUNKS * 4 * n);
    let v1 = build.vector("v^1".into(), digit, false, CHUNKS * 4 * n);
    let u = build.vector("u".into(), Cap::PerCoefficient(1.0), true, CHUNKS * r);
    assert_eq!([v0, v1, u], [V0, V1, U]);
    if let Some((_, folded, row)) = honest {
        assert_eq!(folded.len(), n, "the fold does not match the key");
        for (i, e) in folded.elements().iter().enumerate() {
            for (l, x) in limbs::components(e).iter().enumerate() {
                let [low, high] = pair(x);
                if let Some(&magnitude) = high.iter().find(|h| h.abs() > DIGIT / 2) {
                    return Err(Overflow {
                        chain: "v".into(),
                        magnitude: magnitude * DIGIT,
                        gadget: Gadget {
                            base: DIGIT,
                            levels: 2,
                        },
                    });
                }
                build.set(v0, l * n + i, &low);
                build.set(v1, l * n + i, &high);
            }
        }
        build.put(u, lift_vector(setup, row));
    }
    let negated: Vec<usize> = challenges
        .challenges()
        .iter()
        .map(|c| {
            let k = c.coeffs();
            build.blocks(&core::array::from_fn(|t| -(k[t] as i64)))
        })
        .collect();
    let both = |blocks: usize, vectors: [usize; 2], s: usize| {
        [
            Term {
                blocks,
                scale: 1,
                vector: vectors[0],
                s,
            },
            Term {
                blocks,
                scale: DIGIT,
                vector: vectors[1],
                s,
            },
        ]
    };

    let residues = honest.map(|(matrix, _, _)| limbs::residues(matrix, &setup.primes));
    for (limb, &q) in setup.primes.iter().enumerate() {
        let shape = setup.limbs[limb];
        let residue: Vec<[usize; 2]> = (0..4)
            .map(|m| {
                let low = build.vector(format!("C[{q}][{m}]^0"), digit, false, CHUNKS * r);
                let high = build.vector(
                    format!("C[{q}][{m}]^1"),
                    Cap::PerCoefficient(((q as i64 - 1) / 2 / DIGIT + 1) as f64),
                    false,
                    CHUNKS * r,
                );
                assert_eq!(low, FIRST_LIMB + PER_LIMB * limb + 2 * m);
                [low, high]
            })
            .collect();
        if let Some(res) = &residues {
            for m in 0..4 {
                let [low, high] = residue_digits(res, limb, m, q, r);
                build.put(residue[m][0], low);
                build.put(residue[m][1], high);
            }
        }
        let k = build.vector(
            format!("k[{q}]"),
            digit,
            false,
            CHUNKS * 4 * shape.quotient.levels,
        );
        let e = build.vector(
            format!("e[{q}]"),
            digit,
            false,
            4 * shape.carry.levels * BLOCKS,
        );
        for m in 0..4 {
            let mut terms = Vec::with_capacity(8 * n + 2 * r);
            for l in 0..4 {
                for i in 0..n {
                    terms.extend(both(setup.key(limb, m, l, i), [v0, v1], l * n + i));
                }
            }
            for (j, &c) in negated.iter().enumerate() {
                terms.extend(both(c, residue[m], j));
            }
            build.chain(
                format!("limb {q} component {m}"),
                &terms,
                [0i64; N162],
                q as i64,
                Quotient {
                    gadget: shape.quotient,
                    vector: k,
                    first: m * shape.quotient.levels,
                },
                Carries {
                    gadget: shape.carry,
                    vector: e,
                    first: m * shape.carry.levels * BLOCKS,
                },
            )?;
        }
    }

    let eq0 = eq_table(point.p0());
    assert_eq!(eq0.len(), 4 * n, "the row table does not match the key");
    let mut terms = Vec::with_capacity(8 * n + r);
    for l in 0..4 {
        for i in 0..n {
            let g = build.blocks(&binary::lift(&eq0[4 * i + l]));
            terms.extend(both(g, [v0, v1], l * n + i));
        }
    }
    for (j, &c) in negated.iter().enumerate() {
        terms.push(Term {
            blocks: c,
            scale: 1,
            vector: u,
            s: j,
        });
    }
    binary_chain(&mut build, setup, "binary fold", "w", &terms, [0i64; N162])?;

    let eq1 = eq_table(point.p1());
    assert_eq!(eq1.len(), r, "the column table does not match the columns");
    let terms: Vec<Term> = eq1
        .iter()
        .enumerate()
        .map(|(j, x)| Term {
            blocks: build.blocks(&binary::lift(x)),
            scale: 1,
            vector: u,
            s: j,
        })
        .collect();
    binary_chain(
        &mut build,
        setup,
        "binary evaluation",
        "w'",
        &terms,
        binary::lift(claim),
    )?;

    build.layout.len = build.layout.len.next_power_of_two();
    Ok((
        Relation {
            layout: build.layout,
            equations: build.equations,
            polys: build.polys,
            fixed: setup.polys.len(),
        },
        build.witness,
    ))
}

fn binary_chain(
    build: &mut Build,
    setup: &Setup,
    name: &str,
    tag: &str,
    terms: &[Term],
    output: SElem,
) -> Result<(), Overflow> {
    let Shape { quotient, carry } = setup.binary;
    let digit = Cap::PerCoefficient(HALF);
    let w = build.vector(tag.into(), digit, false, CHUNKS * quotient.levels);
    let e = build.vector(format!("e[{tag}]"), digit, false, carry.levels * BLOCKS);
    build.chain(
        name.into(),
        terms,
        output,
        2,
        Quotient {
            gadget: quotient,
            vector: w,
            first: 0,
        },
        Carries {
            gadget: carry,
            vector: e,
            first: 0,
        },
    )
}

// =============================================================================================
// the bound and the checker
// =============================================================================================

/// The vector each global index belongs to; `None` on padding.
fn owners(layout: &Layout) -> Vec<Option<usize>> {
    let mut owner = vec![None; layout.len];
    for (v, (vector, region)) in layout.vectors.iter().zip(&layout.regions).enumerate() {
        for e in 0..vector.used {
            owner[region.start + e] = Some(v);
        }
    }
    owner
}

fn l2_cap(vector: &Vector) -> f64 {
    match vector.cap {
        Cap::Betasq(b) => b.sqrt(),
        Cap::PerCoefficient(c) => c * ((vector.used * SUPPORT) as f64).sqrt(),
    }
}

/// Does no element repeat within the diagonal? The per-vector rows below are exact only then.
fn distinct(d: &Diagonal) -> bool {
    let mut elements: Vec<usize> = d.entries.iter().map(|e| e.element).collect();
    elements.sort_unstable();
    elements.windows(2).all(|w| w[0] != w[1])
}

/// The largest integer magnitude any coefficient of any block equation's left side can reach for
/// a witness within the caps: coefficient `t` is `sum_i <row_{t,i}, s_i>` over the vectors, at
/// most `sum_i ‖row_{t,i}‖ cap_i` by Cauchy-Schwarz, `row_{t,i}` collecting the weight
/// coefficients that feed position `t` from vector `i`. As `recursion::bound`, except that every
/// vector is capped over `SUPPORT` positions, carries included. The verifier requires the result
/// below `q / 2`.
pub fn no_wrap_bound(relation: &Relation) -> f64 {
    let layout = &relation.layout;
    let owner = owners(layout);
    let caps: Vec<f64> = layout.vectors.iter().map(l2_cap).collect();
    let mut worst = 0f64;
    let mut rows = vec![[0f64; SUB + 1]; layout.vectors.len()];
    for eq in &relation.equations {
        for d in &eq.diagonals {
            debug_assert!(distinct(d), "{}: an element repeats", eq.name);
            rows.iter_mut().for_each(|r| *r = [0f64; SUB + 1]);
            for e in &d.entries {
                let row = &mut rows[owner[e.element].expect("a claim reads a pad element")];
                for (u, &x) in relation.polys[e.poly].iter().enumerate() {
                    let w = (e.scale * x) as f64;
                    row[u] += w * w;
                }
            }
            for t in 0..DEG {
                let total: f64 = rows
                    .iter()
                    .zip(&caps)
                    .map(|(row, cap)| {
                        let sq: f64 = (0..=SUB)
                            .filter(|&u| t >= u && t - u < SUPPORT)
                            .map(|u| row[u])
                            .sum();
                        sq.sqrt() * cap
                    })
                    .sum();
                worst = worst.max(total);
            }
        }
    }
    worst
}

/// Every block equation holds exactly over `Z`, with plain polynomial products, and every element
/// is zero at positions `[SUPPORT, DEG)`.
pub fn check(relation: &Relation, witness: &Witness) -> Result<(), String> {
    let layout = &relation.layout;
    if witness.len() != layout.vectors.len() {
        return Err(format!(
            "{} witness vectors for {} of the layout",
            witness.len(),
            layout.vectors.len()
        ));
    }
    for (vector, w) in layout.vectors.iter().zip(witness) {
        if w.len() != vector.used {
            return Err(format!(
                "{} holds {} elements, not {}",
                vector.name,
                w.len(),
                vector.used
            ));
        }
        for (e, x) in w.iter().enumerate() {
            if x[SUPPORT..].iter().any(|&c| c != 0) {
                return Err(format!("{} element {e} outside its support", vector.name));
            }
        }
    }
    if relation.fixed > relation.polys.len() {
        return Err(format!(
            "{} fixed polys of {}",
            relation.fixed,
            relation.polys.len()
        ));
    }
    let owner = owners(layout);
    let at = |idx: usize| -> Option<&Element> {
        let v = owner[idx]?;
        Some(&witness[v][idx - layout.regions[v].start])
    };
    let mut acc = [0i128; DEG + SUB];
    for eq in &relation.equations {
        for (a, d) in eq.diagonals.iter().enumerate() {
            acc.fill(0);
            for e in &d.entries {
                let x = at(e.element).ok_or_else(|| {
                    format!("{} diagonal {a} reads pad element {}", eq.name, e.element)
                })?;
                let p = relation
                    .polys
                    .get(e.poly)
                    .ok_or_else(|| format!("{} diagonal {a} reads poly {}", eq.name, e.poly))?;
                if p.len() > SUB + 1 {
                    return Err(format!(
                        "{} diagonal {a}: a weight of degree {}",
                        eq.name,
                        p.len() - 1
                    ));
                }
                for (u, &c) in p.iter().enumerate() {
                    if c != 0 {
                        let w = (e.scale * c) as i128;
                        for j in 0..DEG {
                            acc[u + j] += w * x[j] as i128;
                        }
                    }
                }
            }
            for (j, &got) in acc.iter().enumerate() {
                let want = if j < SUB { d.output[j] as i128 } else { 0 };
                if got != want {
                    return Err(format!(
                        "{} diagonal {a} coefficient {j}: {got} != {want}",
                        eq.name
                    ));
                }
            }
        }
    }
    Ok(())
}

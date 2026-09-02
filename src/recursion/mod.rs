//! The post-fold relation, encoded in LaBRADOR's constraint language, plus a reference checker.
//!
//! # The ring, and the basis this module works in
//!
//! `R_648 = Z[X]/(X^648 - X^324 + 1)` is a free module of rank 4 over `S = Z[Y]/(Y^162 - Y^81 + 1)`
//! with `Y = X^4`, and `S = R_162 = Z[Z]/Phi_243(Z)` under `Z = -Y`. Everything here is in the
//! `Z`-basis, `Phi_243(Z) = Z^162 + Z^81 + 1`, i.e. `Z^162 = -Z^81 - 1`: the basis
//! [`crate::api::PowerOfThreeRingElement`] and the challenges of [`crate::challenge`] already use.
//! The `S`-components of `a in R_648` are `a_l = sum_m (-1)^m a_{4m+l} Z^m`, and
//!
//! ```text
//!     (a b)_m = sum_{k+l=m} a_k b_l + (-Z) sum_{k+l=m+4} a_k b_l,
//! ```
//! the `X^4 = Y = -Z` twist; for a fixed output component `m` and input component `l` exactly one
//! `k` contributes, `k = (m - l) mod 4`, twisted iff `l > m`.
//!
//! # The encoding
//!
//! LaBRADOR's linear constraint is `sum_j phi_j . s_j = b` with negacyclic products in `X^64 + 1`.
//! A witness `S`-element is cut into [`CHUNKS`] chunks of [`CHUNK`] coefficients and a public one
//! into [`BLOCKS`] sub-chunks of [`SUB`]; `CHUNK + SUB <= DEG + 1`, so every local product is the
//! plain polynomial product and `S`-arithmetic can be assembled out of them ([`chunk`]). The
//! `BLOCKS` diagonals of such an assembly are chained by carries of `CARRY` coefficients, with the
//! `Phi_243` wrap of the last carry entering at diagonals `0` and `BLOCKS / 2` ([`chain`]).
//!
//! # Module map
//!
//! - [`chunk`]  : `S` arithmetic, the chunking, the public blocks `Z^{CHUNK b} g mod Phi_243`.
//! - [`chain`]  : the diagonals, the carries, and the exact `i128` evaluation of every equation.
//! - [`limbs`]  : the key in coefficient form, the residues, and `F v - y = q k` per limb.
//! - [`binary`] : the two lifted binary identities of the plan's section 2c.
//! - [`bound`]  : the no-wraparound bound of the plan's section 5.
//! - [`export`] : the witness handed to the LaBRADOR front end.
//! - [`setup`]  : everything that depends on the commitment key alone, built once.
//! - [`statement`]: the LaBRADOR statement, built the same way by prover and verifier.
use crate::api::N162;
use crate::fields::scalar::F162;
use crate::params::{QS, QS_LARGE, QS_QUAD};
use crate::scheme::{EvaluationPoint, FoldedWitness, FoldingChallenges, RowEvaluation};
use chain::{At, Carries, Chain, Prepared, Product, Run, Scaled};
use setup::Setup;

pub mod binary;
pub mod bound;
pub mod chain;
pub mod chunk;
pub mod export;
pub mod limbs;
pub mod setup;
pub mod statement;

/// Degree of LaBRADOR's ring `Z_Q[X]/(X^DEG + 1)`.
pub const DEG: usize = 64;
/// Coefficients of one witness chunk; positions `CHUNK..DEG` of a chunk poly are zero.
pub const CHUNK: usize = 54;
/// Coefficients of one public sub-chunk.
pub const SUB: usize = 9;
/// Chunks per witness `S`-element.
pub const CHUNKS: usize = N162 / CHUNK;
/// Sub-chunks per public `S`-element, and diagonals per identity.
pub const BLOCKS: usize = N162 / SUB;
/// Coefficients of one carry: positions `SUB..CHUNK + SUB - 1` of a diagonal.
pub const CARRY: usize = CHUNK - 1;
/// Sub-chunks spanned by one witness chunk.
pub const SPAN: usize = CHUNK / SUB;

const _: () = assert!(N162 % CHUNK == 0 && N162 % SUB == 0 && CHUNK % SUB == 0);
const _: () = assert!(CHUNK + SUB <= DEG + 1);
const _: () = assert!(81 % SUB == 0 && 81 / SUB == BLOCKS / 2);
const _: () = assert!(CARRY + 81 < N162);
const _: () = assert!(SUB * (BLOCKS - 1) + CHUNK + SUB - 1 - N162 == CARRY);

/// The modulus of the LaBRADOR instance, `2^48 - 59`.
pub const Q: i128 = (1i128 << 48) - 59;

/// Witness ranks are rounded up to a multiple of this, so that a commitment constraint of any
/// rank the keys reach reads `extlen(len, kappa) = len` polynomials of every block it spans.
pub const PAD: usize = 32;

/// The cap on `‖v‖^2` per ring element and challenge, the plan's D5: the 95th percentile of the
/// honest fold at the recursive shape of [`crate::scheme::Params::basic`] — the shape at which
/// `prove_opening` enforces the cap — measured over 3000 rounds of the real sampler, so about
/// one fold in twenty is retried with fresh challenges (5.2%; 4.1% at the clear shape).
pub const FOLD_CAP: f64 = 55.7;

/// Largest witness coefficient: a residue of the widest limb, which is a centered residue modulo
/// 19441. LaBRADOR's own `int16` norm limit is 23170, so this clears it with a factor of 2.4.
pub const COEFF_LIMIT: i64 = ((QS_LARGE[1] - 1) / 2) as i64;
/// Largest public sub-chunk coefficient. A public block is `Z^{CHUNK b} g mod Phi_243` of a
/// centered key row, and the two foldings `Z^162 = -Z^81 - 1` can perform leave it inside
/// `2 |g|` — measured, 1.98 to 2.00 |g| for every prime (`tests/recursion.rs`).
pub const BLOCK_LIMIT: i64 = 2 * COEFF_LIMIT;
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!((QS[i] as i64 - 1) / 2 <= COEFF_LIMIT);
        assert!((QS_LARGE[i] as i64 - 1) / 2 <= COEFF_LIMIT);
        assert!((QS_QUAD[i] as i64 - 1) / 2 <= COEFF_LIMIT);
        i += 1;
    }
    assert!((QS_QUAD[2] as i64 - 1) / 2 <= COEFF_LIMIT);
};

/// An element of `S` over `Z`, coefficient `m` of `Z^m`.
pub type SElem = [i64; N162];
/// One LaBRADOR ring element as plain integers.
pub type Poly = [i16; DEG];
/// `blocks[b][a]` is sub-chunk `a` of `Z^{CHUNK b} g mod Phi_243`.
pub type Blocks = [[[i16; SUB]; BLOCKS]; CHUNKS];

/// `x` centred modulo `q`.
pub fn centre(x: i64, q: i64) -> i64 {
    let r = x.rem_euclid(q);
    if r > (q - 1) / 2 {
        r - q
    } else {
        r
    }
}

/// A signed base-`base` gadget of `levels` digits, `x = sum_d base^d d_d`, every digit in
/// `[-base/2, base/2)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Gadget {
    pub base: i64,
    pub levels: usize,
}

/// A chain's honest quotient or carry outside its gadget's reach: the round is retried with fresh
/// challenges, like a fold over its cap.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Overflow {
    pub chain: String,
    pub magnitude: i64,
    pub gadget: Gadget,
}

impl Gadget {
    /// The gadget of `base` with the fewest levels whose [`reach`](Self::reach) covers
    /// `magnitude`.
    pub fn covering(base: i64, magnitude: f64) -> Gadget {
        let mut levels = 1;
        while ((base.pow(levels as u32) / 2) as f64) < magnitude {
            levels += 1;
        }
        Gadget { base, levels }
    }

    pub fn split(&self, x: i64) -> Vec<i64> {
        let mut d = vec![0i64; self.levels];
        self.split_into(x, &mut d);
        d
    }

    /// The same into a caller-owned buffer, which the encoding reuses across the whole `S`-element
    /// rather than allocating one per coefficient and level.
    pub fn split_into(&self, x: i64, d: &mut [i64]) {
        assert!(
            self.try_split_into(x, d),
            "{x} does not fit {} base-{} digits",
            self.levels,
            self.base
        );
    }

    /// [`split_into`](Self::split_into), reporting instead of panicking when `x` is out of reach.
    pub fn try_split_into(&self, x: i64, d: &mut [i64]) -> bool {
        let mut r = x;
        for t in d.iter_mut().take(self.levels) {
            *t = centre(r, self.base);
            r = (r - *t) / self.base;
        }
        r == 0
    }
    /// The largest magnitude the gadget represents.
    pub fn reach(&self) -> i64 {
        self.base.pow(self.levels as u32) / 2
    }
}

/// What the verifier caps a witness vector at: either its whole `‖s‖^2`, or the magnitude of one
/// coefficient, from which the `l2` cap is `c sqrt(n)` over the claimed support.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cap {
    Betasq(f64),
    PerCoefficient(f64),
}

/// One LaBRADOR witness vector: a run of chunk polynomials with a common norm cap.
pub struct Vector {
    pub name: String,
    pub polys: Vec<Poly>,
    /// Polynomials a constraint reads; the rest pad the rank to a multiple of [`PAD`] and are
    /// zero at every position.
    pub used: usize,
    pub cap: Cap,
    /// Coefficients of one poly that the encoding claims may be nonzero.
    pub support: usize,
    pub binary: bool,
}

impl Vector {
    fn new(name: String, cap: Cap, support: usize, binary: bool) -> Vector {
        Vector {
            name,
            polys: Vec::new(),
            used: 0,
            cap,
            support,
            binary,
        }
    }

    /// The `l2` cap the verifier enforces on `‖s‖`.
    pub fn cap(&self) -> f64 {
        match self.cap {
            Cap::Betasq(b) => b.sqrt(),
            Cap::PerCoefficient(c) => c * ((self.used * self.support) as f64).sqrt(),
        }
    }
    fn push(&mut self, p: Poly) -> usize {
        assert!(
            p.iter().all(|&x| (x as i64).abs() <= COEFF_LIMIT),
            "{} holds a coefficient above {COEFF_LIMIT}",
            self.name
        );
        assert!(
            p[self.support..].iter().all(|&x| x == 0),
            "{} holds a coefficient outside its support",
            self.name
        );
        self.polys.push(p);
        self.polys.len() - 1
    }
    /// Append `count` zero polynomials: the layout of a witness the verifier does not hold.
    fn zeros(&mut self, count: usize) -> usize {
        let at = self.polys.len();
        self.polys.resize(at + count, [0i16; DEG]);
        at
    }
    fn push_s(&mut self, x: &SElem) -> usize {
        let c = chunk::chunks(x);
        let at = self.polys.len();
        for p in c {
            self.push(p);
        }
        at
    }
    /// The exact `‖s‖^2` the prover announces.
    pub fn betasq(&self) -> u64 {
        self.polys
            .iter()
            .flatten()
            .map(|&x| (x as i64 * x as i64) as u64)
            .sum()
    }
}

/// How a run of public multipliers becomes `polx`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Rows of the commitment key: converted once at key time and aliased by every proof.
    Key { limb: usize, part: usize },
    /// The negated folding challenges, shared by every chain of the round.
    Challenge,
    /// Binary lifts, whose blocks are signed sums of nine-bit windows of the lifted element.
    Lift,
    /// Anything else. Only [`Instance::of_identity`] makes these.
    Loose,
}

/// A run of public elements whose blocks are converted together and aliased as one `phi`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Group {
    pub kind: Kind,
    pub first: usize,
    pub len: usize,
}

/// Where the three pre-commitments bite: [`statement`] adds one degree-`kappa` constraint over
/// each of these groups of witness vectors.
pub struct Hooks {
    /// `T_Y`: the RNS residues, four vectors per limb.
    pub residues: Vec<usize>,
    /// `T_u`: the lifted left expansion.
    pub left_expansion: usize,
    /// `T_R`: everything else — `v`, the quotients, the carries.
    pub rest: Vec<usize>,
}

/// The whole encoded relation: the witness vectors, the public blocks every constraint points at,
/// and one [`chain::Chain`] per `S`-identity.
pub struct Instance {
    pub vectors: Vec<Vector>,
    pub public: Vec<Blocks>,
    pub groups: Vec<Group>,
    /// The group every public entry belongs to.
    pub group_of: Vec<usize>,
    /// For a [`Kind::Lift`] group, the `SUB`-bit windows of each of its elements; empty otherwise.
    pub windows: Vec<Vec<u16>>,
    pub chains: Vec<chain::Chain>,
    pub limbs: Vec<limbs::Shape>,
    pub hooks: Hooks,
}

impl Instance {
    /// Encode one honest round: the per-limb chains of [`limbs`] and the two binary chains of
    /// [`binary`], with every quotient and carry computed exactly over `Z`.
    pub fn new(
        setup: &Setup,
        residues: &limbs::Residues,
        folded: &FoldedWitness,
        row: &RowEvaluation,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        claim: &F162,
    ) -> Result<Instance, Overflow> {
        Instance::build(
            setup,
            Some((residues, folded, row)),
            challenges,
            point,
            claim,
        )
    }

    /// The same relation with every witness vector left zero: what the verifier can rebuild from
    /// public data alone, and the only thing [`statement`] reads.
    pub fn layout(
        setup: &Setup,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        claim: &F162,
    ) -> Instance {
        Instance::build(setup, None, challenges, point, claim)
            .expect("a layout-only build splits nothing")
    }

    fn build(
        setup: &Setup,
        witness: Option<(&limbs::Residues, &FoldedWitness, &RowEvaluation)>,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        claim: &F162,
    ) -> Result<Instance, Overflow> {
        let mut build = Build::new(setup, challenges, witness.map(|(_, f, r)| (f, r)));
        for limb in 0..setup.limbs.len() {
            limbs::encode(&mut build, setup, witness.map(|(r, _, _)| r), limb)?;
        }
        binary::encode(&mut build, point, claim)?;
        Ok(build.finish())
    }

    /// The exact left-hand side of every block equation, over `Z`, reduced negacyclically in
    /// `X^DEG + 1` exactly as LaBRADOR would: the reference checker.
    pub fn residuals(&self) -> Vec<[[i128; DEG]; BLOCKS]> {
        self.chains
            .iter()
            .map(|c| c.residuals(&self.public, &self.vectors))
            .collect()
    }

    /// Every block equation of every chain evaluates to zero.
    pub fn holds(&self) -> bool {
        self.residuals().iter().flatten().flatten().all(|&x| x == 0)
    }

    /// One identity `sum_nu g_nu x_nu = z` on its own, `z` computed honestly: the smallest thing
    /// the chain encoding says anything about.
    pub fn of_identity(g: &[SElem], x: &[SElem], carry: Gadget) -> Instance {
        assert_eq!(
            g.len(),
            x.len(),
            "one public multiplier per witness element"
        );
        let mut build = Build::bare();
        let xs = build.vector(
            "x".into(),
            Cap::PerCoefficient(COEFF_LIMIT as f64),
            CHUNK,
            false,
        );
        for e in x {
            build.vectors[xs].push_s(e);
        }
        let public: Vec<usize> = g.iter().map(|e| build.group(Kind::Loose, &[*e])).collect();
        let mut output = [0i64; N162];
        for (a, b) in g.iter().zip(x) {
            for (i, v) in chunk::mul(a, b).iter().enumerate() {
                output[i] += v;
            }
        }
        let products = (0..g.len())
            .flat_map(|t| {
                let p = public[t];
                (0..CHUNKS).map(move |b| Product {
                    blocks: p,
                    chunk: b,
                    at: At {
                        vector: xs,
                        off: t * CHUNKS + b,
                    },
                })
            })
            .collect();
        let runs = (0..CHUNKS)
            .map(|b| Run {
                g: chain::public_table(&build.public, public[0], g.len(), b),
                x: chain::witness_table(&build.vectors[xs].polys, b, g.len(), CHUNKS),
                terms: chain::padded(g.len()),
            })
            .collect();
        let carries = build.carry_vectors("x", carry);
        build
            .seal(
                Chain {
                    name: "identity".into(),
                    products,
                    scaled: Vec::new(),
                    output,
                    carries: Carries {
                        gadget: carry,
                        at: Vec::new(),
                    },
                },
                runs,
                1,
                (Gadget { base: 1, levels: 0 }, &[]),
                &carries,
            )
            .expect("the identity's carries are within the gadget");
        build.finish()
    }

    /// The first chain and block whose equation does not evaluate to zero.
    pub fn failure(&self) -> Option<(String, usize, usize)> {
        for (c, r) in self.residuals().iter().enumerate() {
            for (a, e) in r.iter().enumerate() {
                if let Some(t) = e.iter().position(|&x| x != 0) {
                    return Some((self.chains[c].name.clone(), a, t));
                }
            }
        }
        None
    }
}

/// The accumulator [`limbs`] and [`binary`] append to.
pub struct Build {
    pub vectors: Vec<Vector>,
    pub public: Vec<Blocks>,
    pub groups: Vec<Group>,
    pub group_of: Vec<usize>,
    pub windows: Vec<Vec<u16>>,
    pub chains: Vec<chain::Chain>,
    pub limbs: Vec<limbs::Shape>,
    pub binary_chains: binary::Shape,
    /// Ring elements of the folded witness, and columns of the commitment.
    pub n: usize,
    pub r: usize,
    /// The first public entry of the challenge group, shared by every chain.
    pub challenges: usize,
    /// The term-major challenge blocks at every chunk, the public side of every chain's tail.
    pub challenge_tables: Vec<chain::Table>,
    /// The term-major coefficients of the `(l, b)` runs of `v`, `v_tables[l * CHUNKS + b]`: the
    /// witness side of every limb chain's prefix and of the binary fold's.
    pub v_tables: Vec<chain::Table>,
    /// The same for the `CHUNKS` runs of `u`.
    pub u_tables: Vec<chain::Table>,
    /// Whether the witness is filled in: a layout-only build sizes every vector and leaves it zero.
    pub witness: bool,
    /// `residues` and `rest` of [`Hooks`], filled as the vectors are created.
    pub residues: Vec<usize>,
    pub left_expansion: usize,
    pub rest: Vec<usize>,
}

/// The witness vector holding the `S`-components of the folded witness, in the order
/// `(l, b, i)` — twelve runs of `n` polys, each one key-time public array of the plan's section 3.
pub const V: usize = 0;
/// The witness vector holding the lifted left expansion, in the order `(j, b)`.
pub const U: usize = 1;

impl Build {
    fn bare() -> Build {
        Build {
            vectors: Vec::new(),
            public: Vec::new(),
            groups: Vec::new(),
            group_of: Vec::new(),
            windows: Vec::new(),
            chains: Vec::new(),
            limbs: Vec::new(),
            binary_chains: binary::Shape::of(1, 1),
            n: 0,
            r: 0,
            challenges: 0,
            challenge_tables: Vec::new(),
            v_tables: Vec::new(),
            u_tables: Vec::new(),
            witness: true,
            residues: Vec::new(),
            left_expansion: U,
            rest: Vec::new(),
        }
    }

    fn new(
        setup: &Setup,
        challenges: &FoldingChallenges,
        witness: Option<(&FoldedWitness, &RowEvaluation)>,
    ) -> Build {
        let (n, r) = (setup.n, setup.r);
        assert_eq!(challenges.len(), r, "one folding challenge per column");
        let mut vv = Vector::new("v".into(), Cap::Betasq(setup.fold_cap), CHUNK, false);
        match witness {
            Some((folded, _)) => {
                assert_eq!(
                    folded.elements().len(),
                    n,
                    "the fold does not match the key"
                );
                let v: Vec<[SElem; 4]> = folded.elements().iter().map(limbs::components).collect();
                for l in 0..4 {
                    for b in 0..CHUNKS {
                        for i in 0..n {
                            let mut p = [0i16; DEG];
                            for j in 0..CHUNK {
                                p[j] = v[i][l][CHUNK * b + j] as i16;
                            }
                            vv.push(p);
                        }
                    }
                }
            }
            None => {
                vv.zeros(4 * CHUNKS * n);
            }
        }
        let mut uu = Vector::new("u".into(), Cap::PerCoefficient(1.0), CHUNK, true);
        match witness {
            Some((_, row)) => {
                let lifts: Vec<[Poly; CHUNKS]> = row
                    .values()
                    .iter()
                    .map(|x| chunk::chunks(&binary::lift(x)))
                    .collect();
                for b in 0..CHUNKS {
                    for l in lifts.iter() {
                        uu.push(l[b]);
                    }
                }
            }
            None => {
                uu.zeros(CHUNKS * r);
            }
        }
        let mut build = Build {
            vectors: vec![vv, uu],
            public: Vec::with_capacity((setup.limbs.len() * 8 + 4) * n + 2 * r),
            n,
            r,
            binary_chains: setup.binary_chains,
            witness: witness.is_some(),
            rest: vec![V],
            ..Build::bare()
        };
        let negated: Vec<SElem> = challenges
            .challenges()
            .iter()
            .map(|c| {
                let k = c.coeffs();
                core::array::from_fn(|m| -(k[m] as i64))
            })
            .collect();
        build.challenges = build.group(Kind::Challenge, &negated);
        if build.witness {
            build.challenge_tables = (0..CHUNKS)
                .map(|b| chain::public_table(&build.public, build.challenges, r, b))
                .collect();
            build.v_tables = (0..4 * CHUNKS)
                .map(|run| chain::witness_table(&build.vectors[V].polys, run * n, n, 1))
                .collect();
            build.u_tables = (0..CHUNKS)
                .map(|b| chain::witness_table(&build.vectors[U].polys, b * r, r, 1))
                .collect();
        }
        build
    }

    /// The poly of `v` holding chunk `b` of component `l` of ring element `i`.
    pub fn v_at(&self, l: usize, b: usize, i: usize) -> chain::At {
        chain::At {
            vector: V,
            off: (l * CHUNKS + b) * self.n + i,
        }
    }

    pub fn vector(&mut self, name: String, cap: Cap, support: usize, binary: bool) -> usize {
        self.vectors.push(Vector::new(name, cap, support, binary));
        self.vectors.len() - 1
    }

    /// One run of public elements, converted to blocks; returns its first public index.
    pub fn group(&mut self, kind: Kind, items: &[SElem]) -> usize {
        self.group_blocks(kind, &items.iter().map(chunk::blocks).collect::<Vec<_>>())
    }

    /// The same over blocks that are already computed — the key rows of [`setup`].
    pub fn group_blocks(&mut self, kind: Kind, items: &[Blocks]) -> usize {
        let first = self.public.len();
        self.groups.push(Group {
            kind,
            first,
            len: items.len(),
        });
        self.windows.push(Vec::new());
        self.group_of
            .resize(first + items.len(), self.groups.len() - 1);
        self.public.extend_from_slice(items);
        first
    }

    /// A run of binary lifts, which also records the `SUB`-bit windows [`setup`] assembles their
    /// `phi` from.
    pub fn group_lifts(&mut self, items: &[SElem]) -> usize {
        let first = self.group(Kind::Lift, items);
        self.windows[self.groups.len() - 1] = items
            .iter()
            .flat_map(|g| {
                (0..N162 / SUB).map(move |w| {
                    (0..SUB).fold(0u16, |acc, u| {
                        let c = g[SUB * w + u];
                        assert!(c == 0 || c == 1, "a lift holds the coefficient {c}");
                        acc | ((c as u16) << u)
                    })
                })
            })
            .collect();
        first
    }

    /// The witness vector the carries of `gadget` live in, returned once per level.
    ///
    /// The levels of one gadget share a cap and a support, so they share a vector and are told
    /// apart by their offsets. The no-wraparound bound then adds their rows in quadrature against
    /// one cap instead of one at a time, which costs about `sqrt(levels)` of the margin.
    pub fn carry_vectors(&mut self, tag: &str, gadget: Gadget) -> Vec<usize> {
        let v = self.vector(
            format!("e[{tag}]"),
            Cap::PerCoefficient(gadget.base as f64 / 2.0),
            CARRY,
            false,
        );
        self.rest.push(v);
        vec![v; gadget.levels]
    }

    /// The same for the digit levels of a quotient.
    pub fn digit_vectors(&mut self, tag: &str, gadget: Gadget) -> Vec<usize> {
        let v = self.vector(
            tag.to_string(),
            Cap::PerCoefficient(gadget.base as f64 / 2.0),
            CHUNK,
            false,
        );
        self.rest.push(v);
        vec![v; gadget.levels]
    }

    /// Close one chain: the exact quotient of its left side by `divisor` into `quotient`'s digit
    /// vectors, then the honest carries into `carry`'s.
    pub fn seal(
        &mut self,
        mut chain: Chain,
        prep: Prepared,
        divisor: i64,
        quotient: (Gadget, &[usize]),
        carry: &[usize],
    ) -> Result<(), Overflow> {
        let (gadget, levels) = quotient;
        if !self.witness {
            for (d, &vector) in levels.iter().enumerate() {
                let off = self.vectors[vector].zeros(CHUNKS);
                for b in 0..CHUNKS {
                    chain.scaled.push(Scaled {
                        factor: -divisor * gadget.base.pow(d as u32),
                        chunk: b,
                        at: At {
                            vector,
                            off: off + b,
                        },
                    });
                }
            }
            for &vector in carry {
                chain.carries.at.push(At {
                    vector,
                    off: self.vectors[vector].zeros(BLOCKS),
                });
            }
            self.chains.push(chain);
            return Ok(());
        }
        let mut sums = Chain::sums(&prep);
        let value = Chain::value(&sums);
        let k: SElem = core::array::from_fn(|t| {
            let x = value[t] - chain.output[t];
            assert_eq!(
                x % divisor,
                0,
                "{}: the left side is not a multiple of {divisor}",
                chain.name
            );
            x / divisor
        });
        if levels.is_empty() {
            assert_eq!(k, [0i64; N162], "{}: an unquotiented left side", chain.name);
        }
        if std::env::var_os("GADGET_STATS").is_some() {
            let max = k.iter().map(|x| x.abs()).max().unwrap();
            let sd = (k.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / N162 as f64).sqrt();
            eprintln!(
                "gadget-stats quotient {:<24} max {max:>10} ({:.2} bits) sd {sd:>10.1} reach {} ({:.2} bits) fill {:.3}",
                chain.name, (max as f64).log2(), gadget.reach(), (gadget.reach() as f64).log2(),
                max as f64 / gadget.reach() as f64
            );
        }
        let mut split = vec![0i64; gadget.levels.max(1)];
        let mut digits = vec![[0i64; N162]; levels.len()];
        for t in 0..N162 {
            if !gadget.try_split_into(k[t], &mut split) {
                return Err(Overflow {
                    chain: chain.name,
                    magnitude: k[t],
                    gadget,
                });
            }
            for (d, digit) in digits.iter_mut().enumerate() {
                digit[t] = split[d];
            }
        }
        for (d, &vector) in levels.iter().enumerate() {
            let off = self.vectors[vector].push_s(&digits[d]);
            for b in 0..CHUNKS {
                chain.scaled.push(Scaled {
                    factor: -divisor * gadget.base.pow(d as u32),
                    chunk: b,
                    at: At {
                        vector,
                        off: off + b,
                    },
                });
            }
        }
        chain.scaled_into(&mut sums, &self.vectors);
        let e = Chain::honest_carries(&sums);
        let g = chain.carries.gadget;
        if std::env::var_os("GADGET_STATS").is_some() {
            let all: Vec<i64> = e.iter().flat_map(|a| a.iter().copied()).collect();
            let max = all.iter().map(|x| x.abs()).max().unwrap();
            let sd =
                (all.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / all.len() as f64).sqrt();
            eprintln!(
                "gadget-stats carry    {:<24} max {max:>10} ({:.2} bits) sd {sd:>10.1} reach {} ({:.2} bits) fill {:.3}",
                chain.name, (max as f64).log2(), g.reach(), (g.reach() as f64).log2(),
                max as f64 / g.reach() as f64
            );
        }
        let mut polys = vec![[[0i16; DEG]; BLOCKS]; carry.len()];
        split.resize(g.levels.max(1), 0);
        for a in 0..BLOCKS {
            for t in 0..CARRY {
                if !g.try_split_into(e[a][t], &mut split) {
                    return Err(Overflow {
                        chain: chain.name,
                        magnitude: e[a][t],
                        gadget: g,
                    });
                }
                for (d, level) in polys.iter_mut().enumerate() {
                    level[a][t] = split[d] as i16;
                }
            }
        }
        for (d, &vector) in carry.iter().enumerate() {
            chain.carries.at.push(At {
                vector,
                off: self.vectors[vector].polys.len(),
            });
            for p in polys[d] {
                self.vectors[vector].push(p);
            }
        }
        self.chains.push(chain);
        Ok(())
    }

    fn finish(mut self) -> Instance {
        for v in self.vectors.iter_mut() {
            v.used = v.polys.len();
            v.zeros(v.used.next_multiple_of(PAD) - v.used);
        }
        Instance {
            vectors: self.vectors,
            public: self.public,
            groups: self.groups,
            group_of: self.group_of,
            windows: self.windows,
            chains: self.chains,
            limbs: self.limbs,
            hooks: Hooks {
                residues: self.residues,
                left_expansion: self.left_expansion,
                rest: self.rest,
            },
        }
    }
}

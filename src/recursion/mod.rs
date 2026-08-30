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
//! - [`export`] : the statement and witness handed to the LaBRADOR front end.
use crate::api::N162;
use crate::scheme::{
    Commitment, EvaluationPoint, FoldedWitness, FoldingChallenges, PublicParameters, RowEvaluation,
};
use chain::{At, Carries, Chain, Product, Scaled};
use bin_fields::scalar::F162;

pub mod binary;
pub mod bound;
pub mod chain;
pub mod chunk;
pub mod export;
pub mod limbs;

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

/// The modulus of the LaBRADOR instance, `2^40 - 195`.
pub const Q: i128 = (1i128 << 40) - 195;

/// Largest public sub-chunk coefficient, and largest witness coefficient, the `i16` dot product of
/// [`chain`] tolerates: `8 * 2 * BLOCK_LIMIT * COEFF_LIMIT < 2^31`.
pub const BLOCK_LIMIT: i64 = 16384;
/// See [`BLOCK_LIMIT`]; also LaBRADOR's own `int16` norm limit is 23170.
pub const COEFF_LIMIT: i64 = 8191;

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

impl Gadget {
    pub fn split(&self, x: i64) -> Vec<i64> {
        let mut r = x;
        let mut d = Vec::with_capacity(self.levels);
        for _ in 0..self.levels {
            let t = centre(r, self.base);
            d.push(t);
            r = (r - t) / self.base;
        }
        assert_eq!(r, 0, "{x} does not fit {} base-{} digits", self.levels, self.base);
        d
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
    pub cap: Cap,
    /// Coefficients of one poly that the encoding claims may be nonzero.
    pub support: usize,
    pub binary: bool,
}

impl Vector {
    fn new(name: String, cap: Cap, support: usize, binary: bool) -> Vector {
        Vector { name, polys: Vec::new(), cap, support, binary }
    }

    /// The `l2` cap the verifier enforces on `‖s‖`.
    pub fn cap(&self) -> f64 {
        match self.cap {
            Cap::Betasq(b) => b.sqrt(),
            Cap::PerCoefficient(c) => c * ((self.polys.len() * self.support) as f64).sqrt(),
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
        self.polys.iter().flatten().map(|&x| (x as i64 * x as i64) as u64).sum()
    }
}

/// Where the LaBRADOR-side commitments of the plan's sections 1 and 2c bite: the FFI layer adds the
/// degree-`kappa` constraints over exactly these witness vectors.
pub struct Hooks {
    /// `T_C`: the RNS residues, one vector per limb.
    pub residues: Vec<usize>,
    /// `T_u`: the lifted left expansion.
    pub left_expansion: usize,
    /// `T_R`: everything the zero-part masks must bind — `v`, the quotients, the carries.
    pub rest: Vec<usize>,
}

/// The whole encoded relation: the witness vectors, the public blocks every constraint points at,
/// and one [`chain::Chain`] per `S`-identity.
pub struct Instance {
    pub vectors: Vec<Vector>,
    pub public: Vec<Blocks>,
    /// Whether a public entry is a function of the commitment key alone.
    pub key_time: Vec<bool>,
    pub chains: Vec<chain::Chain>,
    pub prepared: Vec<chain::Prepared>,
    pub limbs: Vec<limbs::Shape>,
    pub hooks: Hooks,
}

impl Instance {
    /// Encode one honest round: the per-limb chains of [`limbs`] and the two binary chains of
    /// [`binary`], with every quotient and carry computed exactly over `Z`.
    pub fn new(
        pp: &PublicParameters,
        commitment: &Commitment,
        folded: &FoldedWitness,
        row: &RowEvaluation,
        challenges: &FoldingChallenges,
        point: &EvaluationPoint,
        claim: &F162,
    ) -> Instance {
        let mut build = Build::new(folded, row);
        for limb in 0..commitment.moduli().len() {
            limbs::encode(&mut build, pp, commitment, folded, challenges, limb);
        }
        binary::encode(&mut build, folded, row, challenges, point, claim);
        build.finish()
    }

    /// The exact left-hand side of every block equation, over `Z`, reduced negacyclically in
    /// `X^DEG + 1` exactly as LaBRADOR would: the reference checker.
    pub fn residuals(&self) -> Vec<[[i128; DEG]; BLOCKS]> {
        (0..self.chains.len())
            .map(|c| self.chains[c].residuals(&self.prepared[c], &self.vectors))
            .collect()
    }

    /// Every block equation of every chain evaluates to zero.
    pub fn holds(&self) -> bool {
        self.residuals().iter().flatten().flatten().all(|&x| x == 0)
    }

    /// One identity `sum_nu g_nu x_nu = z` on its own, `z` computed honestly: the smallest thing
    /// the chain encoding says anything about.
    pub fn of_identity(g: &[SElem], x: &[SElem], carry: Gadget) -> Instance {
        assert_eq!(g.len(), x.len(), "one public multiplier per witness element");
        let mut build = Build::bare();
        let xs = build.vector("x".into(), Cap::PerCoefficient(COEFF_LIMIT as f64), CHUNK, false);
        for e in x {
            build.vectors[xs].push_s(e);
        }
        let public: Vec<usize> = g.iter().map(|e| build.blocks(e, false)).collect();
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
                    at: At { vector: xs, off: t * CHUNKS + b },
                })
            })
            .collect();
        let carries = build.carry_vectors("x", carry);
        build.seal(
            Chain { name: "identity".into(), products, scaled: Vec::new(), output, carries: Carries { gadget: carry, at: Vec::new() } },
            1,
            (Gadget { base: 1, levels: 0 }, &[]),
            &carries,
        );
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
    pub key_time: Vec<bool>,
    pub chains: Vec<chain::Chain>,
    pub limbs: Vec<limbs::Shape>,
    /// The `S`-components of the folded witness, `v[(l * CHUNKS + b) * n + i]` in vector order.
    pub v: Vec<[SElem; 4]>,
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
            key_time: Vec::new(),
            chains: Vec::new(),
            limbs: Vec::new(),
            v: Vec::new(),
            residues: Vec::new(),
            left_expansion: U,
            rest: Vec::new(),
        }
    }

    fn new(folded: &FoldedWitness, row: &RowEvaluation) -> Build {
        let n = folded.elements().len();
        let v: Vec<[SElem; 4]> = folded.elements().iter().map(limbs::components).collect();
        let mut vv = Vector::new("v".into(), Cap::Betasq((2f64).powf(30.9)), CHUNK, false);
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
        let mut uu = Vector::new("u".into(), Cap::PerCoefficient(1.0), CHUNK, true);
        for x in row.values() {
            uu.push_s(&binary::lift(x));
        }
        Build { vectors: vec![vv, uu], v, rest: vec![V], ..Build::bare() }
    }

    /// The poly of `v` holding chunk `b` of component `l` of ring element `i`.
    pub fn v_at(&self, l: usize, b: usize, i: usize) -> chain::At {
        chain::At { vector: V, off: (l * CHUNKS + b) * self.v.len() + i }
    }

    pub fn vector(&mut self, name: String, cap: Cap, support: usize, binary: bool) -> usize {
        self.vectors.push(Vector::new(name, cap, support, binary));
        self.vectors.len() - 1
    }

    pub fn blocks(&mut self, g: &SElem, key_time: bool) -> usize {
        self.public.push(chunk::blocks(g));
        self.key_time.push(key_time);
        self.public.len() - 1
    }

    /// One witness vector per carry level of `gadget`.
    pub fn carry_vectors(&mut self, tag: &str, gadget: Gadget) -> Vec<usize> {
        (0..gadget.levels)
            .map(|d| {
                let v = self.vector(
                    format!("e[{tag}][{d}]"),
                    Cap::PerCoefficient(gadget.base as f64 / 2.0),
                    CARRY,
                    false,
                );
                self.rest.push(v);
                v
            })
            .collect()
    }

    /// One witness vector per digit level of `gadget`.
    pub fn digit_vectors(&mut self, tag: &str, gadget: Gadget) -> Vec<usize> {
        (0..gadget.levels)
            .map(|d| {
                let v = self.vector(
                    format!("{tag}[{d}]"),
                    Cap::PerCoefficient(gadget.base as f64 / 2.0),
                    CHUNK,
                    false,
                );
                self.rest.push(v);
                v
            })
            .collect()
    }

    /// Close one chain: the exact quotient of its left side by `divisor` into `quotient`'s digit
    /// vectors, then the honest carries into `carry`'s.
    pub fn seal(
        &mut self,
        mut chain: Chain,
        divisor: i64,
        quotient: (Gadget, &[usize]),
        carry: &[usize],
    ) {
        let prep = chain.prepare(&self.public);
        let mut sums = chain.sums(&prep, &self.vectors);
        let value = Chain::value(&sums);
        let (gadget, levels) = quotient;
        let k: SElem = core::array::from_fn(|t| {
            let x = value[t] - chain.output[t];
            assert_eq!(x % divisor, 0, "{}: the left side is not a multiple of {divisor}", chain.name);
            x / divisor
        });
        if levels.is_empty() {
            assert_eq!(k, [0i64; N162], "{}: an unquotiented left side", chain.name);
        }
        for (d, &vector) in levels.iter().enumerate() {
            let digit: SElem = core::array::from_fn(|t| gadget.split(k[t])[d]);
            let off = self.vectors[vector].push_s(&digit);
            for b in 0..CHUNKS {
                chain.scaled.push(Scaled {
                    factor: -divisor * gadget.base.pow(d as u32),
                    chunk: b,
                    at: At { vector, off: off + b },
                });
            }
        }
        chain.scaled_into(&mut sums, &self.vectors);
        let e = Chain::honest_carries(&sums);
        for (d, &vector) in carry.iter().enumerate() {
            chain.carries.at.push(At { vector, off: self.vectors[vector].polys.len() });
            for a in 0..BLOCKS {
                let mut p = [0i16; DEG];
                for t in 0..CARRY {
                    p[t] = chain.carries.gadget.split(e[a][t])[d] as i16;
                }
                self.vectors[vector].push(p);
            }
        }
        self.chains.push(chain);
    }

    fn finish(self) -> Instance {
        let prepared = self.chains.iter().map(|c| c.prepare(&self.public)).collect();
        Instance {
            vectors: self.vectors,
            public: self.public,
            key_time: self.key_time,
            chains: self.chains,
            prepared,
            limbs: self.limbs,
            hooks: Hooks {
                residues: self.residues,
                left_expansion: self.left_expansion,
                rest: self.rest,
            },
        }
    }
}

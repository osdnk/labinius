//! The recursive opening over rokoko instead of LaBRADOR: the same relation as [`crate::recursion`]
//! — `F v - sum_j c_j C_j = q k` per limb and the two `F162` checks, lifted to `Z` with quotients
//! and carries — encoded as sumcheck claims over one committed vector of `Z_q[X]/(X^128 + 1)`
//! elements. `docs/rokoko.md` is the design note; the modules are
//!
//! - [`relation`]: the chains, their gadgets, the layout of the committed vector, the honest
//!   witness, the block equations as public weights, and the no-wraparound bound;
//! - [`claims`]: the rokoko side — ring elements, the outer commitments, the claims, the proof;
//! - [`config`]: the hand-drafted exact-norm chains for the three shapes, with their security
//!   estimates and calibrated norm tables.
use crate::api::N162;

pub mod claims;
pub mod config;
pub mod relation;
pub mod round;

/// Degree of rokoko's ring `Z_q[X]/(X^DEG + 1)`.
pub const DEG: usize = 128;
/// Coefficients of an `S`-element one committed element carries: `S = Z[Z]/Phi_243` has degree
/// 162 = CHUNKS * CHUNK.
pub const CHUNK: usize = 81;
pub const CHUNKS: usize = N162 / CHUNK;
/// Width of a public block: a chunk times a block is a plain polynomial product, since
/// `CHUNK + SUB - 1 < DEG`.
pub const SUB: usize = 27;
/// Diagonals of one identity, `N162 / SUB`; `81 / SUB` of them make up `Z^81`, which the wrap
/// `Z^162 = -Z^81 - 1` of the last carry lands on.
pub const BLOCKS: usize = N162 / SUB;
/// Diagonals one chunk spans, `CHUNK / SUB`: the chunk `b` of a term enters at diagonal `SPAN b`.
pub const SPAN: usize = CHUNK / SUB;
/// Coefficients of one carry, `CHUNK - 1`.
pub const CARRY: usize = CHUNK - 1;
/// Positions every committed element is zero at: `[SUPPORT, DEG)`. Chunks use `CHUNK` of them
/// and carries `CARRY`; one pattern for both keeps the support claim a single random combination.
pub const SUPPORT: usize = CHUNK;
/// Every committed coefficient is a balanced digit of this base, `|x| <= DIGIT / 2`, except the
/// high digit of a residue, which reaches `(q - 1) / 2 / DIGIT + 1`.
pub const DIGIT_LOG: u32 = 7;
pub const DIGIT: i64 = 1 << DIGIT_LOG;

/// An `S`-element over `Z`: coefficient of `Z^p` at index `p`.
pub type SElem = [i64; N162];
/// One committed element in coefficient form, coefficients `[SUPPORT, DEG)` zero.
pub type Element = [i64; DEG];
/// A short public polynomial, low coefficient first: a block weight (`SUB` coefficients), a carry
/// weight (`+-Z^SUB` times a gadget power), or a scalar.
pub type Poly = Vec<i64>;

pub use crate::recursion::{Cap, Gadget, Overflow};

/// A run of committed elements under one cap and one norm claim; `binary` marks the lifted left
/// expansion, whose coefficients are additionally proven to be bits.
#[derive(Clone, PartialEq, Debug)]
pub struct Vector {
    pub name: String,
    pub cap: Cap,
    pub binary: bool,
    /// Real elements; the region is padded to a power of two with zero elements the claims
    /// never read.
    pub used: usize,
}

/// Where a vector lives in the committed vector: `[start, start + len)`, `len` a power of two
/// and `start` a multiple of it, as `rokoko`'s `WitnessBuilder` places pushes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: usize,
    pub len: usize,
}

/// The committed vector: vectors in order, their regions, and the total length (a power of two).
#[derive(Clone, PartialEq, Debug)]
pub struct Layout {
    pub vectors: Vec<Vector>,
    pub regions: Vec<Region>,
    pub len: usize,
}

impl Layout {
    /// Global index of element `e` of vector `v`.
    pub fn index(&self, v: usize, e: usize) -> usize {
        debug_assert!(e < self.vectors[v].used);
        self.regions[v].start + e
    }
}

/// The weighted elements of one diagonal of one identity: `sum (weight * element) = output` as
/// polynomials, every term of degree below `DEG`.
#[derive(Clone, Debug)]
pub struct Diagonal {
    /// `(global index, weight)`; an index may repeat.
    pub entries: Vec<(usize, Poly)>,
    /// The `SUB` output coefficients of this diagonal.
    pub output: Poly,
}

/// The `BLOCKS` block equations of one chain.
#[derive(Clone, Debug)]
pub struct BlockEquations {
    pub name: String,
    pub diagonals: Vec<Diagonal>,
}

/// Everything the verifier rebuilds from public data: the layout and the block equations.
#[derive(Clone, Debug)]
pub struct Relation {
    pub layout: Layout,
    pub equations: Vec<BlockEquations>,
}

/// The honest committed values, one `Vec<Element>` per vector of the layout, `used` elements each.
pub type Witness = Vec<Vec<Element>>;

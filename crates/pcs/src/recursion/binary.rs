//! The two `F162` checks lifted to identities over `Z` in `S`, with 2 as one more limb.
//!
//! Reduction mod 2 is a ring homomorphism `S -> F162` (the coefficients of an `S`-element are the
//! bits of the field element, and `Phi_243 mod 2` is the `F162` modulus), so
//!
//! ```text
//!     sum_{i,l} lift(B_{i,l}) v_{i,l} - sum_j c_j lift(u_j) = 2 w_b,
//!     sum_j lift(eq(p1)_j) lift(u_j) - lift(t)              = 2 w_e
//! ```
//!
//! over `Z` are exactly `eq(p0) . (v mod 2) = sum_j u_j (c_j mod 2)` and `u . eq(p1) = t`, and both
//! quotients exist precisely when those hold.
use super::chain::{padded, public_table, At, Carries, Chain, Prepared, Product, Run};
use super::{Build, Gadget, Overflow, SElem, CHUNKS, U};
use crate::api::N162;
use crate::eval::eq_table;
use crate::fields::scalar::F162;
use crate::scheme::EvaluationPoint;

/// The quotient and carry gadgets of the two binary chains, sized from the shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape {
    pub quotient: Gadget,
    pub carry: Gadget,
}

/// The binary fold sums in the `Z` basis, where a binary challenge against a binary lift brings no
/// sign cancellation at all, so its quotient and carry grow with the number of terms: the largest
/// quotient coefficient measured `7.05 n r` and the largest carry `10.5 n r` over 25 calibration
/// rounds (`examples/gadget.rs` under `GADGET_STATS`) across every limb list and shapes from
/// `(n, r) = (256, 64)` to `(1024, 256)`, with a spread of 6% and 30% between rounds. The
/// constants carry a 1.1 margin over those maxima, which keeps the basic shape at two levels of
/// each. The evaluation chain is a few hundredths of the fold chain and shares its gadgets.
pub const QUOTIENT_PER_TERM: f64 = 7.8;
pub const CARRY_PER_TERM: f64 = 12.0;

impl Shape {
    /// The gadgets at `n` ring elements per column and `r` columns.
    pub fn of(n: usize, r: usize) -> Shape {
        let terms = (n * r) as f64;
        Shape {
            quotient: Gadget::covering(1024, QUOTIENT_PER_TERM * terms),
            carry: Gadget::covering(2048, CARRY_PER_TERM * terms),
        }
    }
}

/// An `F162` element as the `S`-element with its bits as coefficients.
pub fn lift(x: &F162) -> SElem {
    core::array::from_fn(|p| ((x.0[p >> 6] >> (p & 63)) & 1) as i64)
}

/// `x mod 2` as an `F162`, the inverse of [`lift`] on a 0/1 element.
pub fn reduce_mod_2(x: &SElem) -> F162 {
    let mut out = F162::ZERO;
    for (p, &c) in x.iter().enumerate() {
        out.0[p >> 6] |= ((c & 1) as u64) << (p & 63);
    }
    out
}

/// Append the two lifted identities to `build`.
pub fn encode(build: &mut Build, point: &EvaluationPoint, claim: &F162) -> Result<(), Overflow> {
    let (n, r) = (build.n, build.r);
    let eq0 = eq_table(point.p0());
    assert_eq!(eq0.len(), 4 * n, "the row table does not match the key");

    let base_eq0 = build.public.len();
    for l in 0..4 {
        build.group_lifts(
            &(0..n)
                .map(|i| lift(&eq0[4 * i + l]))
                .collect::<Vec<SElem>>(),
        );
    }
    let base_eq1 = build.public.len();
    build.group_lifts(
        &eq_table(point.p1())
            .iter()
            .map(lift)
            .collect::<Vec<SElem>>(),
    );

    let mut products = Vec::with_capacity(4 * CHUNKS * n + CHUNKS * r);
    let mut runs = Vec::with_capacity(4 * CHUNKS + CHUNKS);
    for l in 0..4 {
        for b in 0..CHUNKS {
            for i in 0..n {
                products.push(Product {
                    blocks: base_eq0 + l * n + i,
                    chunk: b,
                    at: build.v_at(l, b, i),
                });
            }
            if build.witness {
                runs.push(Run {
                    g: public_table(&build.public, base_eq0 + l * n, n, b),
                    x: build.v_tables[l * CHUNKS + b].clone(),
                    terms: padded(n),
                });
            }
        }
    }
    for b in 0..CHUNKS {
        for j in 0..r {
            products.push(Product {
                blocks: build.challenges + j,
                chunk: b,
                at: At {
                    vector: U,
                    off: b * r + j,
                },
            });
        }
        if build.witness {
            runs.push(Run {
                g: build.challenge_tables[b].clone(),
                x: build.u_tables[b].clone(),
                terms: padded(r),
            });
        }
    }
    chain(
        build,
        "binary fold".into(),
        products,
        runs,
        [0i64; N162],
        "w",
    )?;

    let products = (0..CHUNKS)
        .flat_map(|b| {
            (0..r).map(move |j| Product {
                blocks: base_eq1 + j,
                chunk: b,
                at: At {
                    vector: U,
                    off: b * r + j,
                },
            })
        })
        .collect();
    let runs = (0..CHUNKS)
        .filter(|_| build.witness)
        .map(|b| Run {
            g: public_table(&build.public, base_eq1, r, b),
            x: build.u_tables[b].clone(),
            terms: padded(r),
        })
        .collect();
    chain(
        build,
        "binary evaluation".into(),
        products,
        runs,
        lift(claim),
        "w'",
    )
}

/// One lifted identity: quotient by 2, digits, carries.
fn chain(
    build: &mut Build,
    name: String,
    products: Vec<Product>,
    runs: Prepared,
    output: SElem,
    tag: &str,
) -> Result<(), Overflow> {
    let Shape { quotient, carry } = build.binary_chains;
    let digits = build.digit_vectors(tag, quotient);
    let carries = build.carry_vectors(tag, carry);
    build.seal(
        Chain {
            name,
            products,
            scaled: Vec::new(),
            output,
            carries: Carries {
                gadget: carry,
                at: Vec::new(),
            },
        },
        runs,
        2,
        (quotient, &digits),
        &carries,
    )
}

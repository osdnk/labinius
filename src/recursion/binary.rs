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
use super::chain::{At, Carries, Chain, Product};
use super::{Build, Gadget, SElem, CHUNKS, U};
use crate::api::N162;
use crate::eval::eq_table;
use crate::scheme::EvaluationPoint;
use bin_fields::scalar::F162;

/// The carries and quotients of the two binary chains.
pub const CARRY_GADGET: Gadget = Gadget { base: 1024, levels: 2 };
pub const QUOTIENT_GADGET: Gadget = Gadget { base: 1024, levels: 2 };

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
pub fn encode(build: &mut Build, point: &EvaluationPoint, claim: &F162) {
    let (n, r) = (build.n, build.r);
    let eq0 = eq_table(point.p0());
    assert_eq!(eq0.len(), 4 * n, "the row table does not match the key");

    let base_eq0 = build.public.len();
    for l in 0..4 {
        build.group_lifts(&(0..n).map(|i| lift(&eq0[4 * i + l])).collect::<Vec<SElem>>());
    }
    let base_eq1 = build.public.len();
    build.group_lifts(&eq_table(point.p1()).iter().map(lift).collect::<Vec<SElem>>());

    let mut products = Vec::with_capacity(4 * CHUNKS * n + CHUNKS * r);
    for l in 0..4 {
        for b in 0..CHUNKS {
            for i in 0..n {
                products.push(Product {
                    blocks: base_eq0 + l * n + i,
                    chunk: b,
                    at: build.v_at(l, b, i),
                });
            }
        }
    }
    for b in 0..CHUNKS {
        for j in 0..r {
            products.push(Product {
                blocks: build.challenges + j,
                chunk: b,
                at: At { vector: U, off: b * r + j },
            });
        }
    }
    chain(build, "binary fold".into(), products, [0i64; N162], "w");

    let products = (0..CHUNKS)
        .flat_map(|b| {
            (0..r).map(move |j| Product {
                blocks: base_eq1 + j,
                chunk: b,
                at: At { vector: U, off: b * r + j },
            })
        })
        .collect();
    chain(build, "binary evaluation".into(), products, lift(claim), "w'");
}

/// One lifted identity: quotient by 2, digits, carries.
fn chain(build: &mut Build, name: String, products: Vec<Product>, output: SElem, tag: &str) {
    let quotient = build.digit_vectors(tag, QUOTIENT_GADGET);
    let carry = build.carry_vectors(tag, CARRY_GADGET);
    build.seal(
        Chain {
            name,
            products,
            scaled: Vec::new(),
            output,
            carries: Carries { gadget: CARRY_GADGET, at: Vec::new() },
        },
        2,
        (QUOTIENT_GADGET, &quotient),
        &carry,
    );
}

//! `S = Z[Z]/(Z^162 + Z^81 + 1)` over the integers, and the chunk encoding of one element.
use super::{Blocks, Poly, SElem, BLOCK_LIMIT, CHUNK, CHUNKS, DEG, SUB};
use crate::api::N162;

/// `p mod Phi_243`, using `Z^162 = -Z^81 - 1`.
pub fn reduce(p: &[i64]) -> SElem {
    let mut c = p.to_vec();
    c.resize(c.len().max(N162), 0);
    for t in (N162..c.len()).rev() {
        let x = c[t];
        c[t] = 0;
        c[t - 81] -= x;
        c[t - N162] -= x;
    }
    let mut out = [0i64; N162];
    out.copy_from_slice(&c[..N162]);
    out
}

/// `Z^s a mod Phi_243`.
pub fn shift(a: &SElem, s: usize) -> SElem {
    let mut p = vec![0i64; s + N162];
    p[s..].copy_from_slice(a);
    reduce(&p)
}

/// `a b mod Phi_243`, schoolbook.
pub fn mul(a: &SElem, b: &SElem) -> SElem {
    let mut p = vec![0i64; 2 * N162];
    for (i, &x) in a.iter().enumerate() {
        if x != 0 {
            for (j, &y) in b.iter().enumerate() {
                p[i + j] += x * y;
            }
        }
    }
    reduce(&p)
}

/// The public blocks of `g`: `blocks[b][a][u]` is coefficient `SUB a + u` of
/// `Z^{CHUNK b} g mod Phi_243`, the multiplier chunk `b` of a witness meets at diagonal `a`.
pub fn blocks(g: &SElem) -> Blocks {
    core::array::from_fn(|b| {
        let h = shift(g, CHUNK * b);
        core::array::from_fn(|a| {
            core::array::from_fn(|u| {
                let x = h[SUB * a + u];
                assert!(x.abs() <= BLOCK_LIMIT, "public sub-chunk coefficient {x} is too large");
                x as i16
            })
        })
    })
}

/// The `CHUNKS` chunk polynomials of a witness element: chunk `b` holds coefficients
/// `CHUNK b .. CHUNK (b + 1)` at positions `0 .. CHUNK`, the rest zero.
pub fn chunks(x: &SElem) -> [Poly; CHUNKS] {
    core::array::from_fn(|b| {
        let mut p = [0i16; DEG];
        for j in 0..CHUNK {
            let c = x[CHUNK * b + j];
            p[j] = i16::try_from(c).unwrap_or_else(|_| panic!("witness coefficient {c} is too large"));
        }
        p
    })
}

/// `sum_b Z^{CHUNK b} chunk_b mod Phi_243` — the inverse of [`chunks`] on a well-formed encoding,
/// and the value a malformed one actually stands for.
pub fn decode(c: &[Poly; CHUNKS]) -> SElem {
    let mut p = vec![0i64; CHUNK * (CHUNKS - 1) + DEG];
    for (b, q) in c.iter().enumerate() {
        for (j, &x) in q.iter().enumerate() {
            p[CHUNK * b + j] += x as i64;
        }
    }
    reduce(&p)
}

/// Does every chunk keep to positions `0 .. CHUNK`?
pub fn supported(c: &[Poly; CHUNKS]) -> bool {
    c.iter().all(|q| q[CHUNK..].iter().all(|&x| x == 0))
}

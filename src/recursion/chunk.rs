//! `S = Z[Z]/(Z^162 + Z^81 + 1)` over the integers, and the chunk encoding of one element.
use super::{Blocks, Poly, SElem, BLOCKS, BLOCK_LIMIT, CHUNK, CHUNKS, DEG, SUB};
use crate::api::N162;
use std::sync::LazyLock;

/// `p mod Phi_243`, using `Z^162 = -Z^81 - 1`.
pub fn reduce(p: &[i64]) -> SElem {
    let mut c = p.to_vec();
    c.resize(c.len().max(N162), 0);
    reduce_in_place(&mut c)
}

/// The same over a caller-owned buffer, which [`blocks`] reuses across the shifts of one element.
/// The positions above `Z^161` are folded 81 at a time from the top, so that within one band no
/// fold lands on a position the band still has to read.
fn reduce_in_place(c: &mut [i64]) -> SElem {
    let mut top = c.len();
    while top > N162 {
        let low = top.saturating_sub(81).max(N162);
        let (rest, band) = c[..top].split_at_mut(low);
        for (t, x) in band.iter_mut().enumerate() {
            rest[low + t - 81] -= *x;
            rest[low + t - N162] -= *x;
            *x = 0;
        }
        top = low;
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
    let mut p = [0i64; CHUNK * (CHUNKS - 1) + N162];
    core::array::from_fn(|b| {
        p.fill(0);
        p[CHUNK * b..CHUNK * b + N162].copy_from_slice(g);
        let h = reduce_in_place(&mut p);
        core::array::from_fn(|a| {
            core::array::from_fn(|u| {
                let x = h[SUB * a + u];
                assert!(
                    x.abs() <= BLOCK_LIMIT,
                    "public sub-chunk coefficient {x} is too large"
                );
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
            p[j] =
                i16::try_from(c).unwrap_or_else(|_| panic!("witness coefficient {c} is too large"));
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

/// The windows of `g` that make up sub-chunk `a` of its block `b`, with signs.
///
/// `Z^{CHUNK b} g mod Phi_243` reduces through `Z^t = -Z^{t-81} - Z^{t-162}` and, above `Z^243`,
/// through `Z^t = Z^{t-243}`; every one of those shifts is a multiple of [`SUB`], so the map from
/// the coefficients of `g` to one sub-chunk of one block is a signed sum of whole `SUB`-windows of
/// `g`. `taps()[b][a]` is that sum, as `(window, sign)` pairs, read off the unit elements.
pub fn taps() -> &'static [[Vec<(u16, i8)>; BLOCKS]; CHUNKS] {
    static TAPS: LazyLock<[[Vec<(u16, i8)>; BLOCKS]; CHUNKS]> = LazyLock::new(|| {
        let windows = N162 / SUB;
        let mut t: [[Vec<(u16, i8)>; BLOCKS]; CHUNKS] =
            core::array::from_fn(|_| core::array::from_fn(|_| Vec::new()));
        for w in 0..windows {
            for u in 0..SUB {
                let mut g = [0i64; N162];
                g[SUB * w + u] = 1;
                let block = blocks(&g);
                for (b, row) in t.iter_mut().enumerate() {
                    for (a, taps) in row.iter_mut().enumerate() {
                        let sign = block[b][a][u];
                        assert!(
                            block[b][a]
                                .iter()
                                .enumerate()
                                .all(|(x, &c)| x == u || c == 0),
                            "block {b} diagonal {a} mixes window positions"
                        );
                        match taps.iter().find(|e| e.0 as usize == w) {
                            Some(e) => assert_eq!(e.1 as i16, sign, "window {w} is not uniform"),
                            None if sign != 0 => taps.push((w as u16, sign as i8)),
                            None => {}
                        }
                    }
                }
            }
        }
        t
    });
    &TAPS
}

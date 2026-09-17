//! Byte transposes of 64-word tiles, so that `vgf2p8affineqb` — `out.byte[t].bit[i] =
//! <A.byte[7 - i], x.byte[t]>`, an 8x8x8 block product per qword — sees eight different words in
//! each qword. Each `vpermt2b` index comes from the two bit layouts around it ([`round_idx`]);
//! the byte order inside a tile is the scramble [`word_at`].
use core::arch::x86_64::*;
use std::sync::OnceLock;

/// `affine(ident, D)` is the bit transpose of each qword of `D`.
pub const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

const E: [u8; 6] = [0, 1, 2, 3, 4, 5];
const K: [u8; 4] = [10, 11, 12, 13];
const C: [u8; 3] = [20, 21, 22];

/// `zmm[s]`: the logical bit in vector-index bit `s` from the top; `pos[b]`: in position bit `5 - b`.
#[derive(Clone)]
struct Layout {
    zmm: Vec<u8>,
    pos: [u8; 6],
}

impl Layout {
    fn slot_of(&self, id: u8) -> Option<usize> {
        self.zmm.iter().position(|&x| x == id)
    }

    fn value(&self, id: u8, z: usize, p: usize) -> usize {
        if let Some(s) = self.slot_of(id) {
            (z >> (self.zmm.len() - 1 - s)) & 1
        } else {
            let b = self.pos.iter().position(|&x| x == id).expect("a placed bit");
            (p >> (5 - b)) & 1
        }
    }
}

/// One round moves one bit between a vector slot and a byte position; the two vectors differing
/// in that slot feed the two outputs, and this is the index for the output whose new bit is `v`.
fn round_idx(from: &Layout, to: &Layout, v: usize) -> [u8; 64] {
    let slot = (0..to.zmm.len())
        .find(|&s| from.zmm[s] != to.zmm[s])
        .expect("one slot moves");
    let mut idx = [0u8; 64];
    let z = v << (to.zmm.len() - 1 - slot);
    for (p, i) in idx.iter_mut().enumerate() {
        let mut q = 0;
        for b in 0..6 {
            q |= to.value(from.pos[b], z, p) << (5 - b);
        }
        let s = to.value(from.zmm[slot], z, p);
        *i = ((s << 6) | q) as u8;
    }
    idx
}

fn rounds(mut layout: Layout, steps: &[(usize, usize)]) -> Vec<[[u8; 64]; 2]> {
    steps
        .iter()
        .map(|&(slot, pos_bit)| {
            let mut next = layout.clone();
            let out = next.zmm[slot];
            next.zmm[slot] = next.pos[pos_bit];
            next.pos[pos_bit] = out;
            let r = [round_idx(&layout, &next, 0), round_idx(&layout, &next, 1)];
            layout = next;
            r
        })
        .collect()
}

struct Tables {
    input: Vec<[[u8; 64]; 2]>,
    output: Vec<[[u8; 64]; 2]>,
    output_fix: [u8; 64],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let input = rounds(
            Layout {
                zmm: vec![E[5], E[4], E[3], E[2]],
                pos: [E[1], E[0], K[3], K[2], K[1], K[0]],
            },
            &[(0, 2), (1, 3), (2, 4), (3, 5)],
        );
        let after = Layout {
            zmm: vec![C[2], C[1], C[0]],
            pos: [E[1], E[0], E[5], E[4], E[3], E[2]],
        };
        let output = rounds(after, &[(0, 2), (1, 3), (2, 4)]);
        // pos is now [E1, E0, C2, C1, C0, E2]; the qword order is [E2, E1, E0, C2, C1, C0].
        let mut output_fix = [0u8; 64];
        for (p, f) in output_fix.iter_mut().enumerate() {
            let (e2, e1, e0, c) = ((p >> 5) & 1, (p >> 4) & 1, (p >> 3) & 1, p & 7);
            *f = ((e1 << 5) | (e0 << 4) | (c << 1) | e2) as u8;
        }
        Tables {
            input,
            output,
            output_fix,
        }
    })
}

/// The word at byte position `p` of a transposed tile.
#[inline]
pub fn word_at(p: usize) -> usize {
    (((p >> 5) & 1) << 1) | ((p >> 4) & 1) | ((p & 0xf) << 2)
}

#[inline(always)]
unsafe fn idx(v: &[u8; 64]) -> __m512i {
    _mm512_loadu_si512(v.as_ptr() as *const __m512i)
}

#[inline(always)]
unsafe fn apply_round(z: &mut [__m512i], slot_bit: usize, r: &[[u8; 64]; 2]) {
    let (i0, i1) = (idx(&r[0]), idx(&r[1]));
    for z0 in 0..z.len() {
        if z0 & slot_bit != 0 {
            continue;
        }
        let z1 = z0 | slot_bit;
        let (a, b) = (z[z0], z[z1]);
        z[z0] = _mm512_permutex2var_epi8(a, i0, b);
        z[z1] = _mm512_permutex2var_epi8(a, i1, b);
    }
}

/// `out[k].byte[p]` = byte `k` of word `word_at(p)` of the 64 words at `src`.
#[inline(always)]
pub unsafe fn transpose_tile(src: *const u8, out: &mut [__m512i; 16]) {
    let t = tables();
    for (i, o) in out.iter_mut().enumerate() {
        *o = _mm512_loadu_si512(src.add(64 * i) as *const __m512i);
    }
    for (r, round) in t.input.iter().enumerate() {
        apply_round(out, 8 >> r, round);
    }
}

/// `planes[c].byte[p]` = byte `c` of word `word_at(p)` becomes `out[g].qword[i].byte[c]` of word
/// `8g + i`.
#[inline(always)]
pub unsafe fn untranspose_planes(mut planes: [__m512i; 8]) -> [__m512i; 8] {
    let t = tables();
    for (r, round) in t.output.iter().enumerate() {
        apply_round(&mut planes, 4 >> r, round);
    }
    let fix = idx(&t.output_fix);
    planes.map(|z| _mm512_permutexvar_epi8(fix, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_transpose_and_back() {
        let words: Vec<u128> = (0..64u128)
            .map(|i| (i * 0x0123_4567_89AB_CDEF_1122_3344_5566_7788u128) ^ (i << 100))
            .collect();
        let mut t = [unsafe { _mm512_setzero_si512() }; 16];
        unsafe { transpose_tile(words.as_ptr() as *const u8, &mut t) };
        let bytes: Vec<[u8; 64]> = t
            .iter()
            .map(|&z| unsafe { core::mem::transmute::<__m512i, [u8; 64]>(z) })
            .collect();
        for k in 0..16 {
            for p in 0..64 {
                assert_eq!(bytes[k][p], words[word_at(p)].to_le_bytes()[k], "k {k} p {p}");
            }
        }
        let planes: [__m512i; 8] = core::array::from_fn(|c| t[c]);
        let back = unsafe { untranspose_planes(planes) };
        for g in 0..8 {
            let q: [u64; 8] = unsafe { core::mem::transmute(back[g]) };
            for i in 0..8 {
                assert_eq!(q[i], words[8 * g + i] as u64, "g {g} i {i}");
            }
        }
    }
}

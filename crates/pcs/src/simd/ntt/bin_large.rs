//! Forward NTT for **binary** inputs on the splitting tree for `q` in [`QS_LARGE`], in the
//! vertical batch-of-32 layout.
//!
//! The tree, the lookup tables and the block layout are
//! [`crate::simd::ntt::bin_small`]'s — 648 linear slots, levels 0, 1, 2 and the level-3 twiddles
//! folded into 96 byte-split 16-entry tables, then four radix-3 levels, handed out as 24 blocks
//! of 27 rows through [`BlockSink`]. What differs is the arithmetic budget.
//!
//! # Why a separate kernel
//!
//! A lane is an i16, so the whole transform lives inside `2^15 / q`: 8.43 at q = 3889 and 3.37 at
//! q = 9721, but only **1.873** at q = 17497 and **1.686** at q = 19441. A radix-3 butterfly
//! forms `a0 + t1 + t2`, and a Montgomery twiddle product never gets below `q/2` — the
//! `mulhi(m, q)` term is a full i16 times `q` — so three terms already cost `1.5 q` before any
//! slack. There is no schedule of the existing kernels, which reduce only the untwiddled `a0`,
//! that fits: at 19441 the un-Barretted level 3 alone leaves `3 q = 58323`.
//!
//! Everything a butterfly adds therefore has to be reduced, not just `a0`, and neither of the
//! crate's two reductions is dear enough to rule out. They are cheap on **different ports**,
//! which is the whole of the schedule below:
//!
//! * the shuffle-port **lookup Barrett** of
//!   [`crate::simd::ntt::bin_asm::barrett_lut_i16`] — `vpmultishiftqb`, `vpandd`, `vpord`,
//!   `vpermb`, `vpaddw`, of which LLVM makes the middle two one `vpternlogd`: **4 uops, two of
//!   them port-5 only and none on the multiply port**, `|r| <= q/2 + 2^10` = 0.53 q;
//! * the two-multiply **`vpmulhrsw` Barrett** of [`crate::params::barrett_i16`] — `vpmulhrsw`,
//!   `vpmullw`, `vpsubw`: **3 uops, two of them port-0 only**. `round(2^15/q)` is 2 for both
//!   primes, a quotient estimate with two significant bits, so it only reaches `|r| <= 0.60 q`
//!   at 17497 and 0.74 q at 19441 — but it is a uop cheaper.
//!
//! Nine of a butterfly's uops are the three Montgomery products' multiplies and are port-0 only,
//! and the level-3 loop carries 12 `vpermb` per row on port 5, so neither port is the bottleneck
//! by itself: the schedule is a port-balancing problem, not a "how few reductions" one, and
//! [`bin_sched`] solves it as one.
//!
//! # The unsigned alternative, and why this one
//!
//! The natural other answer is to leave the signed representation: keep the lanes in `[0, q)` as
//! u16, where the head-room is `2^16/q` = 3.75 / 3.37 and a radix-3 sum of three reduced values
//! fits with room. A butterfly is then 3 Shoup products with a conditional subtract
//! (`vpmullw` + `vpmulhuw` + `vpmullw` + `vpsubw` + `vpsubw` + `vpminuw`, 6 uops of which 3 on
//! the multiply port), three negations `q - t` so that the two twiddled outputs stay sums rather
//! than differences, six adds and one unsigned lookup Barrett on `a0`: 33 uops, 9 on port 0 and 2
//! on port 5, against the signed schedule's 29 (q = 17497) and 35 (q = 19441) with the same 9 on
//! port 0 and 2 or 8 on port 5.
//!
//! Measured over the same 216-butterfly level, both written out in intrinsics
//! (`tests/vertical_bin_large.rs::the_unsigned_alternative`), the signed one wins anyway:
//! **3.68 ns per butterfly against the unsigned form's 5.96 at q = 17497, and 4.40 against 6.08
//! at q = 19441** — 1.6x and 1.4x. The uop counts are a wash and the dependency chains are
//! not: a Shoup product is
//! `vpmullw -> vpmullw -> vpsubw -> vpsubw -> vpminuw` deep where a Montgomery one is
//! `vpmullw -> vpmulhw -> vpsubw`, and the unsigned butterfly serialises three of them behind
//! each other through `t1 - t2`.
//!
//! And the unsigned form has no output the rest of the crate can use. `Batch32` is i16, `A` is
//! centered, the base multiplication is `vpdpwssd` on signed pairs, and the fold, the
//! decomposition and every bound in [`crate::simd::commit`] are written on centered lanes; an
//! unsigned transform would have to be centered before any of that, a pass over the 648 output
//! vectors of every batch on top of a per-butterfly loss. Signed it is.
//!
//! # The schedule
//!
//! Per level there are three sites — the untwiddled `a0` input, the two twiddle products `t1`,
//! `t2`, and the `omega (t1 - t2)` — and each takes one of the three reductions above or none.
//! [`bin_sched`] walks all 3^12 placements over the four radix-3 levels, keeps those the exact
//! bound recursion [`bin_model`] proves stay inside i16, and among those the one [`bar_cost`]
//! scores cheapest on the two vector ports, so the reduction points are a compile-time
//! consequence of `q` and not a table:
//!
//! | level            | q = 17497           | after    | q = 19441                | after    |
//! |------------------|---------------------|---------:|--------------------------|---------:|
//! | levels 0+1+2     |                     |   1.00 q |                          |   1.00 q |
//! | level 3          | `a0` mul, `t` mul   |   1.85 q | `a0`, `t`, `u` lookup    |   1.58 q |
//! | level 4          | `a0` lookup, `t` mul|   1.79 q | `a0`, `t`, `u` lookup    |   1.58 q |
//! | level 5          | `a0` lookup, `t` mul|   1.79 q | `a0`, `t`, `u` lookup    |   1.58 q |
//! | level 6 (output) | `a0` lookup, `t` mul|   1.79 q | `a0`, `t`, `u` lookup    |   1.58 q |
//!
//! 17497 reduces `a0`, `t1` and `t2` — three reductions per butterfly, 2592 per batch — and can
//! leave `u` alone because `a0 + t2 + u` at `0.53 q + 0.60 q + 0.60 q` still clears 1.873 q.
//! 19441 cannot (that same sum is 1.77 q against a 1.686 q budget) and pays a fourth, 3456 per
//! batch; 0.74 q is too loose anywhere in its schedule, so all sixteen of its reductions are the
//! lookup. The `t1`, `t2` reduction is what earns its keep at either prime: without it the
//! level-3 sum `t1 + t2` of two table sums is `2 q`, and no amount of reducing `a0` brings
//! `a0 + t1 + t2` under 2.5 q.
//!
//! Every site 17497 gives the `vpmulhrsw` Barrett has an input small enough that the two
//! reductions return the same value — `|t| <= 0.73 q < 2^14` for a twiddle product, `|a0| <= q`
//! at level 3, and the two quotient estimates only part company above 24576 — so the kernel's
//! output is **bit-identical** to the all-lookup schedule it replaces, 6 % cheaper. The bound
//! model does not know that and does not need to: it carries the worst case of each, which is
//! why the declared output grows from 1.71 q to 1.79 q.
//!
//! # Levels 5 and 6 are software-pipelined
//!
//! A degree-9 sub-ring is exactly three degree-3 sub-rings, so its nine values never leave
//! registers between levels 5 and 6 — but level 6 consumes all three of level 5's butterflies,
//! and taking the three sub-rings of a block one after another leaves only three independent
//! butterflies in flight. Sampled per loop, that one ran 17.5 % above its port bound where
//! level 4, whose nine butterflies are independent, ran 1.4 % above. Running one sub-ring's
//! level 5 in the shadow of the previous one's level 6 holds 18 rows instead of 9 and still
//! keeps 32 registers without a spill: **431.1 to 415.7 cycles per ring element at 17497 and
//! 532.8 to 513.3 at 19441**, which leaves the whole kernel 6 % above the model. Holding all 27
//! rows of a block instead spills — uops issued per element 986 to 1180 — and costs 455.7.
//!
//! There is no `asm!` tail here as there is in [`crate::simd::ntt::bin_asm`]. That kernel
//! hand-schedules levels 4, 5 and 6 into one block per 27-slot block because 27 resident data
//! registers plus its three constants exactly fill the file; this one needs eight — `q`, omega
//! and its companion, the lookup Barrett's four and the `vpmulhrsw` one's `round(2^15/q)` — and
//! three or four reductions per butterfly, so the 27-row register-resident tail does not exist to
//! be written. The pipelined 18 rows above is what fits. It is otherwise the intrinsics form of
//! the same tree, block for block and sink for sink, which is also what
//! [`crate::simd::ntt::bin_quad`] does at 247 to 291 cycles per ring element.
use crate::params::*;
use crate::simd::ntt::r3_twiddles;
pub use crate::simd::transpose_f162::BinaryIndex32;
use crate::simd::ntt::bin_asm::barrett_lut_corr;
pub use crate::simd::ntt::bin_asm::BlockSink;
use crate::ring::element::*;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// bounds and the reduction schedule
// ---------------------------------------------------------------------------------------------

/// `|barrett_lut_i16(a, q)|` and `|barrett_i16(a, q)|` for the worst i16 `a`, per prime —
/// exhaustive sweeps, evaluated once (`q/2 + 2^10` is the lookup one's theoretical bound; the
/// sweep is a little tighter).
const BLM: [i32; 2] = [
    barrett_lut_sweep(QS_LARGE[0]),
    barrett_lut_sweep(QS_LARGE[1]),
];
const BMM: [i32; 2] = [
    barrett_mul_sweep(QS_LARGE[0]),
    barrett_mul_sweep(QS_LARGE[1]),
];

const fn qi(q: u16) -> usize {
    if q == QS_LARGE[0] {
        0
    } else {
        1
    }
}

/// Does `q` run this kernel rather than [`crate::simd::ntt::bin_asm`]?
pub const fn is_large(q: u16) -> bool {
    q == QS_LARGE[0] || q == QS_LARGE[1]
}

/// `|barrett_lut_i16(a, q)|` for the worst i16 `a`.
pub const fn barrett_lut_max(q: u16) -> i32 {
    BLM[qi(q)]
}

/// `|barrett_i16(a, q)|` for the worst i16 `a`.
pub const fn barrett_mul_max(q: u16) -> i32 {
    BMM[qi(q)]
}

const fn barrett_lut_sweep(q: u16) -> i32 {
    let mut m = 0i32;
    let mut a = -32768i32;
    while a < 32768 {
        let r = ((a + barrett_lut_corr(((a >> 11) & 31) as usize, q) as i32) as i16) as i32;
        let r = if r < 0 { -r } else { r };
        if r > m {
            m = r;
        }
        a += 1;
    }
    m
}

/// [`crate::params::barrett_i16`] as a `const fn`: `t = round(a v / 2^15)` (`vpmulhrsw`),
/// `r = a - t q`.
const fn barrett_mul_i16(a: i32, q: u16) -> i32 {
    let t = ((a * barrett_v(q) as i32 * 2 + (1 << 15)) >> 16) as i16;
    (a as i16).wrapping_sub(t.wrapping_mul(q as i16)) as i32
}

const fn barrett_mul_sweep(q: u16) -> i32 {
    let mut m = 0i32;
    let mut a = -32768i32;
    while a < 32768 {
        let r = barrett_mul_i16(a, q);
        let r = if r < 0 { -r } else { r };
        if r > m {
            m = r;
        }
        a += 1;
    }
    m
}

/// `|mont(a, w)| <= |a| q / 2^17 + q/2` (`params::mont_mul_i16`, |w| <= q/2).
const fn mont_bound(b: i32, q: u16) -> i32 {
    ((b as i64 * q as i64) >> 17) as i32 + (q as i32 + 1) / 2
}

/// A schedule is a base-3 word of 12 digits: digit `3 l + i` names the reduction the butterflies
/// of level `3 + l` apply at site `i` — 0 the untwiddled `a0`, 1 the two twiddle products `t1`
/// and `t2`, 2 the `omega (t1 - t2)`.
pub const SCHEDULES: u32 = 531441;
pub const RED_NONE: u8 = 0;
/// The shuffle-port lookup Barrett: 4 uops, two of them bound to port 5.
pub const RED_LUT: u8 = 1;
/// The two-multiply `vpmulhrsw` Barrett: 3 uops, two of them bound to port 0.
pub const RED_MUL: u8 = 2;

const POW3: [u32; 13] = {
    let mut p = [1u32; 13];
    let mut i = 1;
    while i < 13 {
        p[i] = 3 * p[i - 1];
        i += 1;
    }
    p
};

/// Which reduction level `3 + l` applies at site `i`.
pub const fn bar_kind(code: u32, l: usize, i: usize) -> u8 {
    (code / POW3[3 * l + i] % 3) as u8
}

/// The bound a site's reduction caps its input at, or the input itself when it reduces nothing.
const fn capped(b: i32, k: u8, q: u16) -> i32 {
    let r = match k {
        RED_LUT => barrett_lut_max(q),
        RED_MUL => barrett_mul_max(q),
        _ => return b,
    };
    if r < b {
        r
    } else {
        b
    }
}

/// One level of the bound recursion: the input bound `v` and the peak so far in, the output
/// bound and the new peak out. `l == 0` is level 3, whose twiddles are folded into the lookup
/// tables, so there `t1` and `t2` are table sums rather than Montgomery products.
const fn bin_level(q: u16, code: u32, l: usize, v: i32, peak: i32) -> (i32, i32) {
    let mut peak = peak;
    let t = capped(
        if l == 0 { v } else { mont_bound(v, q) },
        bar_kind(code, l, 1),
        q,
    );
    if 2 * t > peak {
        peak = 2 * t;
    }
    let u = capped(mont_bound(2 * t, q), bar_kind(code, l, 2), q);
    let b0 = capped(v, bar_kind(code, l, 0), q);
    let (o0, o1) = (b0 + 2 * t, b0 + t + u);
    if o0 > peak {
        peak = o0;
    }
    if o1 > peak {
        peak = o1;
    }
    (if o0 > o1 { o0 } else { o1 }, peak)
}

/// The kernel's schedule replayed on bounds: the maximum |lane| after the fused lookups and
/// after each of levels 3, 4, 5, 6, and the largest intermediate ever formed (which is what has
/// to stay inside i16).
///
/// Every position of a level carries the same bound — the tree is uniform and the schedule is
/// per level — so the recursion is the scalar one and its per-level maximum is exact.
pub const fn bin_model(q: u16, code: u32) -> ([i32; 5], i32) {
    let mut lm = [0i32; 5];
    let mut v = 2 * ((q as i32 + 1) / 2);
    let mut peak = v;
    lm[0] = v;
    let mut l = 0;
    while l < 4 {
        (v, peak) = bin_level(q, code, l, v, peak);
        lm[l + 1] = v;
        l += 1;
    }
    (lm, peak)
}

/// `(port-0-only, port-5-only, either)` uops one butterfly of level `3 + l` spends reducing.
const fn red_uops(code: u32, l: usize) -> (u32, u32, u32) {
    let (mut p0, mut p5, mut fx) = (0, 0, 0);
    let mut i = 0;
    while i < 3 {
        let n = if i == 1 { 2 } else { 1 };
        match bar_kind(code, l, i) {
            RED_LUT => {
                p5 += 2 * n;
                fx += 2 * n;
            }
            RED_MUL => {
                p0 += 2 * n;
                fx += n;
            }
            _ => {}
        }
        i += 1;
    }
    (p0, p5, fx)
}

/// Cycles a level's butterflies cost when nothing but the two vector ports binds: the flexible
/// uops go wherever they fit, so it is the larger of either port's own load and half the total —
/// except that the scheduler steers them at issue and does not quite reach the balance, so a
/// level whose port-0-only load comes within a sixteenth of half of it runs at that load
/// instead. That sixteenth and the per-level accounting of [`bar_cost`] are what make the model
/// rank six measured schedules of q = 17497 in the order the machine does (predicted 391.5,
/// 394.9, 401.3, 404.6, 420.8, 421.9 cycles per element; measured, before levels 5 and 6 were
/// pipelined, 432.6, 438.8, 441.6, 443.0, 444.4, 458.8; after, the winner and the runner-up are
/// 415.7 and 421.8).
const fn ports(p0: u32, p5: u32, fx: u32) -> u32 {
    let half = (p0 + p5 + fx).div_ceil(2);
    let m = if p0 > p5 { p0 } else { p5 };
    let m = m + m / 16;
    if m > half {
        m
    } else {
        half
    }
}

/// What one 162-row group costs: 54 butterflies at each of the four levels, balanced level by
/// level. Level 3's are the omega-only [`r3_folded`] — 3 multiply uops, not 9 — and its loop
/// also carries 27 rows of 12 `vpermb` and 6 adds, which is where the kernel's port-5 load is
/// concentrated and why that level can afford the port-0 Barrett everywhere.
///
/// Levels 5 and 6 share one loop body but are charged apart: level 6 consumes level 5's three
/// outputs, so a level-5 uop of the same sub-ring cannot move into level 6's shadow, and two
/// schedules with the same port totals over the pair measure 2.4% apart when the split between
/// them differs. What the body does overlap is the *next* sub-ring's level 5, which is a
/// different 9 rows and carries the same schedule.
const fn bar_cost(code: u32) -> u32 {
    let (a0, a5, af) = red_uops(code, 0);
    let mut c = ports(54 * (3 + a0), 324 + 54 * a5, 162 + 54 * (8 + af));
    let mut l = 1;
    while l < 4 {
        let (p0, p5, fx) = red_uops(code, l);
        c += ports(54 * (9 + p0), 54 * p5, 54 * (10 + fx));
        l += 1;
    }
    c
}

/// The cheapest schedule that keeps every intermediate inside i16, and the output bound it
/// leaves. Ties on cost go to the tighter output.
///
/// Exhaustive over all [`SCHEDULES`] of them, four nested passes over one level's 27 choices:
/// the peak only grows, so a prefix that has already left i16 prunes every schedule under it and
/// the search touches a few thousand instead of half a million.
const fn bin_sched(q: u16) -> (u32, i32) {
    let (mut best, mut cost, mut out) = (u32::MAX, u32::MAX, i32::MAX);
    let v0 = 2 * ((q as i32 + 1) / 2);
    let mut c0 = 0;
    while c0 < 27 {
        let (v1, p1) = bin_level(q, c0, 0, v0, v0);
        if p1 <= 32767 {
            let mut c1 = 0;
            while c1 < 27 {
                let (v2, p2) = bin_level(q, 27 * c1, 1, v1, p1);
                if p2 <= 32767 {
                    let mut c2 = 0;
                    while c2 < 27 {
                        let (v3, p3) = bin_level(q, 729 * c2, 2, v2, p2);
                        if p3 <= 32767 {
                            let mut c3 = 0;
                            while c3 < 27 {
                                let (v4, p4) = bin_level(q, 19683 * c3, 3, v3, p3);
                                if p4 <= 32767 {
                                    let code = c0 + 27 * c1 + 729 * c2 + 19683 * c3;
                                    let c = bar_cost(code);
                                    if c < cost || (c == cost && v4 < out) {
                                        best = code;
                                        cost = c;
                                        out = v4;
                                    }
                                }
                                c3 += 1;
                            }
                        }
                        c2 += 1;
                    }
                }
                c1 += 1;
            }
        }
        c0 += 1;
    }
    assert!(best != u32::MAX, "no i16 schedule for this prime");
    (best, out)
}

const BIN_SCHED: [(u32, i32); 2] = [bin_sched(QS_LARGE[0]), bin_sched(QS_LARGE[1])];

/// Which reduction each of levels 3, 4, 5, 6 applies at each of `a0`, `(t1, t2)` and `u`.
pub const fn bar_levels(q: u16) -> u32 {
    BIN_SCHED[qi(q)].0
}

/// Declared output bound: max |lane| of [`ntt_bin_batch32`].
pub const fn output_bound(q: u16) -> i32 {
    BIN_SCHED[qi(q)].1
}

/// The schedule that reduces `a0` with the lookup Barrett at every level and nothing else, which
/// is all the kernels below `2^14` ever do.
const fn lut_a0_only() -> u32 {
    let mut code = 0;
    let mut l = 0;
    while l < 4 {
        code += RED_LUT as u32 * 3u32.pow(3 * l);
        l += 1;
    }
    code
}

const _: () = {
    let mut i = 0;
    while i < 2 {
        let q = QS_LARGE[i];
        assert!(bin_model(q, bar_levels(q)).1 <= 32767);
        assert!(bin_model(q, lut_a0_only()).1 > 32767);
        i += 1;
    }
};
// 17497 reduces `a0` and the two twiddle products at every level and leaves `u` alone, and every
// level spends the three-uop Barrett on one of the two — the level's port-0 budget takes exactly
// one of them. 19441 pays for `u` as well, and 0.736 q is too loose anywhere in its schedule, so
// all sixteen of its reductions are the lookup.
const _: () = {
    let (c0, c1) = (bar_levels(QS_LARGE[0]), bar_levels(QS_LARGE[1]));
    let mut l = 0;
    while l < 4 {
        assert!(bar_kind(c0, l, 0) != RED_NONE && bar_kind(c0, l, 1) != RED_NONE);
        assert!(bar_kind(c0, l, 2) == RED_NONE);
        assert!(bar_kind(c0, l, 0) == RED_MUL || bar_kind(c0, l, 1) == RED_MUL);
        let mut i = 0;
        while i < 3 {
            assert!(bar_kind(c1, l, i) == RED_LUT);
            i += 1;
        }
        l += 1;
    }
};

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[((k * 2 + s2) * 3 + r) * 2 + ab]`: the 16 centered i16 values of
    /// `base_k(n) * zeta''_{2k+s2}^r * (ab == 1 ? zeta'_k : 1)`, **byte-split** so that a single
    /// `vpermb` (1 uop, port 5) does the lookup: byte n is the low half of entry n, byte 16+n the
    /// high half.
    lut: [[u8; 64]; 96],
    /// 512-bit constants of the lookup Barrett: `[ms, corr, and, or]` — the `vpmultishiftqb`
    /// control, the byte-split `-k q` table and the index fix-up masks.
    cv: [[i16; 32]; 4],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2), each i16
    /// duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw4: [[u32; 4]; 24],
    tw5: [[u32; 4]; 72],
    tw6: [[u32; 4]; 216],
    /// omega and its companion.
    om: [u32; 2],
    /// q and `round(2^15/q)`, duplicated.
    qd: u32,
    bvd: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

const fn r3_pair<const Q: u16>(z: u16) -> [u32; 4] {
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let (a, b) = mont_pair::<Q>(z);
    let (c, d) = mont_pair::<Q>(z2);
    [a, b, c, d]
}

/// Byte `u` of the 64-byte `vpermb` correction table: the low halves of `-k(s) q` at u = s < 32,
/// the high halves at u = 32 + s.
pub const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = Params::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    let mut lut = [[0u8; 64]; 96];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        let mut s2 = 0;
        while s2 < 2 {
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            let mut r = 0;
            while r < 3 {
                let f = pow_mod(z3, r as u64, q);
                let mut ab = 0;
                while ab < 2 {
                    let extra = if ab == 0 { 1 } else { zp };
                    let mut n = 0;
                    while n < 16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let e = center(base * f % q * extra % q, q) as u16;
                        let ti = ((k * 2 + s2) * 3 + r) * 2 + ab;
                        lut[ti][n] = e as u8;
                        lut[ti][16 + n] = (e >> 8) as u8;
                        n += 1;
                    }
                    ab += 1;
                }
                r += 1;
            }
            s2 += 1;
        }
        k += 1;
    }

    let tw4 = r3_twiddles!(24, r3_pair::<Q>, Params::<Q>::ZETA_L4);
    let tw5 = r3_twiddles!(72, r3_pair::<Q>, Params::<Q>::ZETA_L5);
    let tw6 = r3_twiddles!(216, r3_pair::<Q>, Params::<Q>::ZETA_L6);
    let (oa, ob) = mont_pair::<Q>(Params::<Q>::OMEGA);

    let mut cv = [[0i16; 32]; 4];
    let mut i = 0;
    while i < 32 {
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that word.
        cv[0][i] = ((16 * (i % 4) + 11) * 257) as i16;
        let (b0, b1) = (lut_byte(2 * i, Q), lut_byte(2 * i + 1, Q));
        cv[1][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[2][i] = 0x1f1f;
        cv[3][i] = 0x2000;
        i += 1;
    }
    Tables {
        lut,
        cv,
        tw4,
        tw5,
        tw6,
        om: [oa, ob],
        qd: dup(Q as i16),
        bvd: dup(barrett_v(Q)),
    }
}

static T17497: Tables = build_tables::<17497>();
static T19441: Tables = build_tables::<19441>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    if Q == QS_LARGE[0] {
        &T17497
    } else {
        &T19441
    }
}

// ---------------------------------------------------------------------------------------------
// arithmetic helpers
// ---------------------------------------------------------------------------------------------

/// `vpbroadcastd zmm, m32` - one pure load uop. Written as `asm!` because LLVM otherwise
/// "recognises" the duplicated-u32 splat and rebuilds it with vpmovsxwd/vpmovdw/vinserti64x4.
#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// The four twiddle broadcasts of one sub-ring off a single base register.
#[inline(always)]
unsafe fn bc4(p: *const u32) -> (__m512i, __m512i, __m512i, __m512i) {
    let (a, b, c, d): (__m512i, __m512i, __m512i, __m512i);
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{4}]",
        "vpbroadcastd {1}, dword ptr [{4} + 4]",
        "vpbroadcastd {2}, dword ptr [{4} + 8]",
        "vpbroadcastd {3}, dword ptr [{4} + 12]",
        out(zmm_reg) a,
        out(zmm_reg) b,
        out(zmm_reg) c,
        out(zmm_reg) d,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    (a, b, c, d)
}

/// `vpmulhw`. stdarch's `_mm512_mulhi_epi16` is written as sext -> mul -> shr -> trunc; LLVM
/// folds the multiply but leaves a `vpmovsxwd`/`vpmovdw`/`vinserti64x4` round trip on operands it
/// cannot see through (broadcast constants), and rematerialises it inside the hot loops.
#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpmulhw {0}, {1}, {2}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// 3-uop signed Montgomery twiddle multiply: a * x mod q in (-q, q).
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm512_sub_epi16(hi, t)
}

struct C {
    q: __m512i,
    bv: __m512i,
    om: __m512i,
    omp: __m512i,
    ms: __m512i,
    corr: __m512i,
    andm: __m512i,
    orm: __m512i,
}

/// The shuffle-port lookup Barrett. `vpandd` and `vpord` are one `vpternlogd`, so it is
/// `vpmultishiftqb` + `vpternlogd` + `vpermb` + `vpaddw`: **4 uops, two of them port-5 only and
/// no multiply-port slot at all**, `|r| <= q/2 + 2^10`.
#[inline(always)]
unsafe fn barrett_lut(a: __m512i, c: &C) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(c.ms, a);
    let s = _mm512_and_si512(s, c.andm);
    let s = _mm512_or_si512(s, c.orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, c.corr))
}

/// The two-multiply Barrett of [`crate::params::barrett_i16`]: `vpmulhrsw` + `vpmullw` +
/// `vpsubw`, **3 uops, two of them port-0 only**. The quotient estimate `round(2^15/q)` is 2 for
/// both primes, so `|r|` only comes down to `0.60 q` where the lookup reaches `0.53 q` — but it
/// is one uop cheaper and it spends it on the port the lookup leaves idle.
#[inline(always)]
unsafe fn barrett_mul(a: __m512i, c: &C) -> __m512i {
    let t = _mm512_mulhrs_epi16(a, c.bv);
    _mm512_sub_epi16(a, _mm512_mullo_epi16(t, c.q))
}

#[inline(always)]
unsafe fn red(a: __m512i, c: &C, k: u8) -> __m512i {
    match k {
        RED_LUT => barrett_lut(a, c),
        RED_MUL => barrett_mul(a, c),
        _ => a,
    }
}

/// The radix-3 butterfly of a level whose twiddles are already in `t1`, `t2` (level 3, where the
/// tables carry them), reducing whichever of `a0`, `(t1, t2)` and `u` the schedule names.
#[inline(always)]
unsafe fn r3_folded(
    c: &C,
    a0: __m512i,
    t1: __m512i,
    t2: __m512i,
    bar: [u8; 3],
) -> (__m512i, __m512i, __m512i) {
    let t1 = red(t1, c, bar[1]);
    let t2 = red(t2, c, bar[1]);
    let u = red(mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q), c, bar[2]);
    let a0 = red(a0, c, bar[0]);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// The same butterfly with the twiddles `[w, w', w2, w2']` applied first.
#[inline(always)]
unsafe fn r3(
    c: &C,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
    bar: [u8; 3],
) -> (__m512i, __m512i, __m512i) {
    let (w1, w1p, w2, w2p) = bc4(tw);
    let t1 = mont(a1, w1, w1p, c.q);
    let t2 = mont(a2, w2, w2p, c.q);
    r3_folded(c, a0, t1, t2, bar)
}

#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m512i {
    _mm512_load_si512(p.add(64 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn ld(p: *const i16, j: usize) -> __m512i {
    _mm512_load_si512(p.add(32 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn st(p: *mut i16, j: usize, v: __m512i) {
    _mm512_store_si512(p.add(32 * j) as *mut __m512i, v);
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);

/// The per-prime schedule as compile-time constants of the kernel: a `const fn` call on its own
/// is not folded before instruction selection, an associated const is.
struct Sched<const Q: u16>;

impl<const Q: u16> Sched<Q> {
    const BAR: [[u8; 3]; 4] = {
        let m = bar_levels(Q);
        let mut b = [[RED_NONE; 3]; 4];
        let mut l = 0;
        while l < 4 {
            b[l] = [bar_kind(m, l, 0), bar_kind(m, l, 1), bar_kind(m, l, 2)];
            l += 1;
        }
        b
    };
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_core<const Q: u16, S: BlockSink>(input: &BinaryIndex32, sink: &mut S) {
    let t = tables::<Q>();
    let cvp = t.cv.as_ptr() as *const __m512i;
    let c = C {
        q: bc(&t.qd),
        bv: bc(&t.bvd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
        ms: _mm512_load_si512(cvp),
        corr: _mm512_load_si512(cvp.add(1)),
        andm: _mm512_load_si512(cvp.add(2)),
        orm: _mm512_load_si512(cvp.add(3)),
    };
    let bar = Sched::<Q>::BAR;

    // The caller already holds the `vpermb` byte-index rows; the kernel reads them straight.
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;

    for k in 0..4 {
        let lut = t.lut.as_ptr().add(12 * k) as *const u8;
        let l = |s2: usize, r: usize, ab: usize| -> __m512i {
            _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
        };
        let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
        let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
        let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));
        let (l110, l111) = (l(1, 1, 0), l(1, 1, 1));
        let (l120, l121) = (l(1, 2, 0), l(1, 2, 1));

        // levels 0+1+2 (table lookups) fused with level 3 (omega multiply only).
        for i in 0..27 {
            let n0 = ldb(ip, i);
            let n0h = ldb(ip, i + 81);
            let n1 = ldb(ip, i + 27);
            let n1h = ldb(ip, i + 108);
            let n2 = ldb(ip, i + 54);
            let n2h = ldb(ip, i + 135);

            let x = _mm512_permutexvar_epi8(n0, l000);
            let y = _mm512_permutexvar_epi8(n0h, l001);
            let a0 = _mm512_add_epi16(x, y);
            let b0 = _mm512_sub_epi16(x, y);

            let x = _mm512_permutexvar_epi8(n1, l010);
            let y = _mm512_permutexvar_epi8(n1h, l011);
            let a1 = _mm512_add_epi16(x, y);
            let x = _mm512_permutexvar_epi8(n1, l110);
            let y = _mm512_permutexvar_epi8(n1h, l111);
            let b1 = _mm512_sub_epi16(x, y);

            let x = _mm512_permutexvar_epi8(n2, l020);
            let y = _mm512_permutexvar_epi8(n2h, l021);
            let a2 = _mm512_add_epi16(x, y);
            let x = _mm512_permutexvar_epi8(n2, l120);
            let y = _mm512_permutexvar_epi8(n2h, l121);
            let b2 = _mm512_sub_epi16(x, y);

            let (u0, u1, u2) = r3_folded(&c, a0, a1, a2, bar[0]);
            let (v0, v1, v2) = r3_folded(&c, b0, b1, b2, bar[0]);
            st(bp, i, u0);
            st(bp, i + 27, u1);
            st(bp, i + 54, u2);
            st(bp, i + 81, v0);
            st(bp, i + 108, v1);
            st(bp, i + 135, v2);
        }

        // levels 4, 5 and 6, one 27-block at a time: level 4 over the L1 scratch, then levels 5
        // and 6 fused over the nine values of a degree-9 sub-ring, straight to the sink.
        for j in 0..6 {
            let kk = 6 * k + j;
            let op = sink.dst(kk);
            let base = 27 * j;
            let t4 = t.tw4[kk].as_ptr();
            for i in 0..9 {
                let (b0, b1, b2) = (base + i, base + i + 9, base + i + 18);
                let (o0, o1, o2) = r3(&c, ld(bp, b0), ld(bp, b1), ld(bp, b2), t4, bar[1]);
                st(bp, b0, o0);
                st(bp, b1, o1);
                st(bp, b2, o2);
            }
            let lv5 = |g: usize| -> [__m512i; 9] {
                let t5 = t.tw5[3 * kk + g].as_ptr();
                let b = base + 9 * g;
                let mut v = [_mm512_setzero_si512(); 9];
                for i in 0..9 {
                    v[i] = ld(bp, b + i);
                }
                for i in 0..3 {
                    let (o0, o1, o2) = r3(&c, v[i], v[3 + i], v[6 + i], t5, bar[2]);
                    v[i] = o0;
                    v[3 + i] = o1;
                    v[6 + i] = o2;
                }
                v
            };
            let lv6 = |g: usize, v: &[__m512i; 9]| {
                for i in 0..3 {
                    let t6 = t.tw6[9 * kk + 3 * g + i].as_ptr();
                    let (o0, o1, o2) = r3(&c, v[3 * i], v[3 * i + 1], v[3 * i + 2], t6, bar[3]);
                    let o = 9 * g + 3 * i;
                    st(op, o, o0);
                    st(op, o + 1, o1);
                    st(op, o + 2, o2);
                }
            };
            // one group's level 5 runs in the shadow of the previous group's level 6, which
            // depends on it and would otherwise leave only three independent butterflies.
            let g0 = lv5(0);
            let g1 = lv5(1);
            lv6(0, &g0);
            let g2 = lv5(2);
            lv6(1, &g1);
            lv6(2, &g2);
            sink.block(kk, op);
        }
    }
}

/// Forward NTT of 32 binary polynomials: `out.v[j][p] = a_p(psi^SLOT_EXP[j]) mod q`, lazily
/// reduced (`|lane| <= output_bound(Q)`).
///
/// # Safety
/// The host must have AVX-512 F/BW/VL/VBMI; `out` is 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    let mut sink = crate::simd::ntt::bin_asm::OutSink(out.v.as_mut_ptr() as *mut i16);
    ntt_core::<Q, _>(input, &mut sink);
    out.representation = Representation::Ntt;
}

/// The same transform with the output handed to `sink` 27 rows at a time instead of being written
/// to a `Batch32`, for consumers that want each block while it is still in L1.
///
/// # Safety
/// See [`BlockSink`]: `sink.dst` must give 27 writable 64-byte aligned vectors per block.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32_sink<const Q: u16, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
    ntt_core::<Q, S>(input, sink);
}

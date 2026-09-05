//! `simd::ntt::bin_large` against the scalar NTT, against an i32 shadow of its own schedule,
//! and against the `const` bound recursion that chose that schedule.
//!
//! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
//! element (`f162::pack4`) and sliced by the production front end, so the kernel is fed exactly
//! what a commitment feeds it.
use bin_ntt::f162;
use bin_ntt::params::*;
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::transpose_f162::{self as tf, BinaryIndex32};
use bin_ntt::simd::ntt::bin_asm::{barrett_lut_corr, barrett_lut_i16};
use bin_ntt::simd::ntt::bin_large as vl;
use bin_ntt::simd::ntt::bin_large::{RED_LUT, RED_MUL};
use bin_ntt::ring::*;
use bin_ntt::F162;

type Bin = [u32; N];

fn random_bin(rng: &mut Rng) -> Bin {
    let mut c = [0u32; N];
    for w in 0..N.div_ceil(64) {
        let x = rng.next_u64();
        for b in 0..64 {
            if 64 * w + b < N {
                c[64 * w + b] = ((x >> b) & 1) as u32;
            }
        }
    }
    c
}

fn adversarial() -> Vec<Bin> {
    let mut v = vec![[0u32; N], [1u32; N]];
    for phase in 0..2 {
        v.push(core::array::from_fn(|i| (i % 2 == phase) as u32));
        v.push(core::array::from_fn(|i| (i % 3 == phase) as u32));
    }
    for b in 0..4 {
        let mut p = [0u32; N];
        for i in 0..162 {
            p[i + 162 * b] = 1;
        }
        v.push(p);
    }
    for d in [
        0usize, 1, 80, 81, 161, 162, 163, 323, 324, 325, 485, 486, 646, 647,
    ] {
        let mut p = [0u32; N];
        p[d] = 1;
        v.push(p);
    }
    v
}

fn batches(count: usize, seed: u64) -> Vec<[Bin; 32]> {
    let mut rng = Rng::new(seed);
    let adv = adversarial();
    let mut out = Vec::new();
    let mut b0: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().take(32).enumerate() {
        b0[i] = *p;
    }
    out.push(b0);
    let mut b1: [Bin; 32] = [[0u32; N]; 32];
    for (i, p) in adv.iter().skip(32).enumerate() {
        b1[i] = *p;
    }
    for i in adv.len().saturating_sub(32)..32 {
        b1[i] = random_bin(&mut rng);
    }
    out.push(b1);
    for _ in 0..count {
        out.push(core::array::from_fn(|_| random_bin(&mut rng)));
    }
    out
}

fn elems_of(polys: &[Bin; 32]) -> [F162; 128] {
    let mut e = [F162([0; 3]); 128];
    for p in 0..32 {
        e[4 * p..4 * p + 4].copy_from_slice(&f162::pack4(&polys[p]));
    }
    e
}

fn idx_of(elems: &[F162; 128]) -> BinaryIndex32 {
    let mut out = BinaryIndex32::zero();
    unsafe { tf::slice_f162_into(elems, &mut out) };
    out
}

// ------------------------------------------------------------------ the i32 shadow

/// The kernel's schedule replayed one polynomial at a time in i32, with every value the kernel
/// would hold in an i16 lane checked against `2^15` as it is formed.
struct Shadow<const Q: u16> {
    /// max |value| after the fused lookups and after each of levels 3, 4, 5, 6.
    max: [i32; 5],
    max_any: i32,
}

impl<const Q: u16> Shadow<Q> {
    fn new() -> Self {
        Shadow {
            max: [0; 5],
            max_any: 0,
        }
    }
    fn see(&mut self, x: i32) -> i16 {
        let a = x.abs();
        self.max_any = self.max_any.max(a);
        assert!(a < 32768, "i16 overflow: {x}");
        x as i16
    }
    fn mont(&mut self, a: i16, x: u16) -> i16 {
        let w = Params::<Q>::to_mont(x);
        self.see(mont_mul_i16(a, w, Params::<Q>::mont_pre(w), Q) as i32)
    }
    fn red(&mut self, a: i16, k: u8) -> i16 {
        match k {
            RED_LUT => self.see(barrett_lut_i16(a, Q) as i32),
            RED_MUL => self.see(barrett_i16(a, Q) as i32),
            _ => a,
        }
    }
    fn r3_folded(&mut self, a0: i16, t1: i16, t2: i16, l: usize) -> (i16, i16, i16) {
        let m = vl::bar_levels(Q);
        let f = |i| vl::bar_kind(m, l, i);
        let t1 = self.red(t1, f(1));
        let t2 = self.red(t2, f(1));
        let d = self.see(t1 as i32 - t2 as i32);
        let u = self.mont(d, Params::<Q>::OMEGA);
        let u = self.red(u, f(2));
        let a0 = self.red(a0, f(0));
        (
            self.see(a0 as i32 + t1 as i32 + t2 as i32),
            self.see(a0 as i32 - t2 as i32 + u as i32),
            self.see(a0 as i32 - t1 as i32 - u as i32),
        )
    }
    fn r3(&mut self, a0: i16, a1: i16, a2: i16, zeta: u16, l: usize) -> (i16, i16, i16) {
        let z2 = (zeta as u64 * zeta as u64 % Q as u64) as u16;
        let t1 = self.mont(a1, zeta);
        let t2 = self.mont(a2, z2);
        self.r3_folded(a0, t1, t2, l)
    }

    fn run(&mut self, poly: &Bin) -> [i16; N] {
        let q = Q as u64;
        let z6 = Params::<Q>::ZETA6 as u64;
        let kappa = [z6, (1 + q - z6) % q];
        let base = |k: usize, m: usize| -> u64 {
            let (s0, s1) = (k / 2, k % 2);
            let z1 = Params::<Q>::ZETA_L1[s0] as u64;
            let n = [m, m + 162, m + 324, m + 486].map(|c| poly[c] as u64);
            let inner = z1 * ((n[1] + kappa[s0] * n[3]) % q) % q;
            let t = if s1 == 0 { inner } else { (q - inner) % q };
            ((n[0] + kappa[s0] * n[2]) % q + t) % q
        };
        let mut v = [0i16; N];
        for k in 0..4 {
            let z2 = Params::<Q>::ZETA_L2[k] as u64;
            // levels 0, 1, 2 and the level-3 twiddle out of the tables, then the omega-only
            // radix-3 of level 3.
            for i in 0..27 {
                let entry = |s2: usize, r: usize, ab: usize, m: usize| -> i32 {
                    let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
                    let f = pow_mod(z3, r as u64, q) * if ab == 1 { z2 } else { 1 } % q;
                    center(base(k, m) * f % q, q) as i32
                };
                let mut a = [0i16; 3];
                let mut b = [0i16; 3];
                for c in 0..3 {
                    let (m, mh) = (i + 27 * c, i + 27 * c + 81);
                    let x = entry(0, c, 0, m);
                    let y = entry(0, c, 1, mh);
                    a[c] = self.see(x + y);
                    let x = entry(c.min(1), c, 0, m);
                    let y = entry(c.min(1), c, 1, mh);
                    b[c] = self.see(x - y);
                    self.max[0] = self.max[0]
                        .max((a[c] as i32).abs())
                        .max((b[c] as i32).abs());
                }
                let (u0, u1, u2) = self.r3_folded(a[0], a[1], a[2], 0);
                let (w0, w1, w2) = self.r3_folded(b[0], b[1], b[2], 0);
                v[162 * k + i] = u0;
                v[162 * k + i + 27] = u1;
                v[162 * k + i + 54] = u2;
                v[162 * k + i + 81] = w0;
                v[162 * k + i + 108] = w1;
                v[162 * k + i + 135] = w2;
            }
            self.level_max(&v[162 * k..162 * k + 162], 1);
            for j in 0..6 {
                let kk = 6 * k + j;
                let o = 162 * k + 27 * j;
                for i in 0..9 {
                    let (r0, r1, r2) = self.r3(
                        v[o + i],
                        v[o + i + 9],
                        v[o + i + 18],
                        Params::<Q>::ZETA_L4[kk],
                        1,
                    );
                    v[o + i] = r0;
                    v[o + i + 9] = r1;
                    v[o + i + 18] = r2;
                }
            }
            self.level_max(&v[162 * k..162 * k + 162], 2);
        }
        for kk in 0..24 {
            let o = 27 * kk;
            for g in 0..3 {
                let b = o + 9 * g;
                for i in 0..3 {
                    let (r0, r1, r2) = self.r3(
                        v[b + i],
                        v[b + i + 3],
                        v[b + i + 6],
                        Params::<Q>::ZETA_L5[3 * kk + g],
                        2,
                    );
                    v[b + i] = r0;
                    v[b + i + 3] = r1;
                    v[b + i + 6] = r2;
                }
            }
        }
        self.level_max(&v, 3);
        for kk in 0..24 {
            for g in 0..3 {
                for i in 0..3 {
                    let b = 27 * kk + 9 * g + 3 * i;
                    let (r0, r1, r2) = self.r3(
                        v[b],
                        v[b + 1],
                        v[b + 2],
                        Params::<Q>::ZETA_L6[9 * kk + 3 * g + i],
                        3,
                    );
                    v[b] = r0;
                    v[b + 1] = r1;
                    v[b + 2] = r2;
                }
            }
        }
        self.level_max(&v, 4);
        v
    }

    fn level_max(&mut self, v: &[i16], l: usize) {
        for x in v {
            self.max[l] = self.max[l].max((*x as i32).abs());
        }
    }
}

fn kernel<const Q: u16>() {
    let bound = vl::output_bound(Q);
    let mut sh = Shadow::<Q>::new();
    for polys in batches(24, 0xB16 ^ Q as u64) {
        let elems = elems_of(&polys);
        let idx = idx_of(&elems);
        let mut out = Batch32::zero(Representation::Coefficients);
        unsafe { vl::ntt_bin_batch32::<Q>(&idx, &mut out) };
        assert_eq!(out.representation, Representation::Ntt);
        for p in 0..32 {
            let coeffs = f162::lift4(&elems[4 * p..4 * p + 4].try_into().unwrap());
            assert_eq!(coeffs, polys[p]);
            let want = scalar::ntt::<Q>(&coeffs);
            let shadow = sh.run(&coeffs);
            for j in 0..N {
                let got = out.v[j][p];
                assert_eq!(
                    (got as i32).rem_euclid(Q as i32) as u32,
                    want[j],
                    "q={Q} slot {j} lane {p}"
                );
                assert_eq!(got, shadow[j], "q={Q} slot {j} lane {p}: shadow model");
                assert!(
                    (got as i32).abs() <= bound,
                    "q={Q} slot {j} lane {p}: |{got}| over the declared bound {bound}"
                );
            }
        }
    }
    let (model, peak) = vl::bin_model(Q, vl::bar_levels(Q));
    assert!(sh.max_any < 32768);
    assert!(
        sh.max_any <= peak,
        "shadow peak {} > model {peak}",
        sh.max_any
    );
    for l in 0..5 {
        assert!(
            sh.max[l] <= model[l],
            "level {l}: {} > {}",
            sh.max[l],
            model[l]
        );
    }
    let m = vl::bar_levels(Q);
    let flags: Vec<String> = (0..4)
        .map(|l| {
            let n = ["a0", "t12", "u"];
            let kind = |i| {
                if vl::bar_kind(m, l, i) == RED_LUT {
                    "L"
                } else {
                    "M"
                }
            };
            let on: Vec<String> = (0..3)
                .filter(|&i| vl::bar_kind(m, l, i) != 0)
                .map(|i| format!("{}{}", n[i], kind(i)))
                .collect();
            format!(
                "L{}: {}",
                3 + l,
                if on.is_empty() {
                    "-".into()
                } else {
                    on.join("+")
                }
            )
        })
        .collect();
    println!(
        "q={Q}  {}  |  per level {:?} q (model {:?}), peak {} = {:.3} q of 32767",
        flags.join("  "),
        sh.max
            .map(|x| (x as f64 / Q as f64 * 1000.0).round() / 1000.0),
        model,
        sh.max_any,
        sh.max_any as f64 / Q as f64
    );
}

#[test]
fn kernel_17497() {
    kernel::<17497>();
}

#[test]
fn kernel_19441() {
    kernel::<19441>();
}

/// The block sink the commitment consumes the transform through sees exactly the 648 rows the
/// plain entry point writes, 27 at a time, in block order.
fn sink<const Q: u16>() {
    let mut rng = Rng::new(7 + Q as u64);
    let polys: [Bin; 32] = core::array::from_fn(|_| random_bin(&mut rng));
    let idx = idx_of(&elems_of(&polys));
    let mut out = Batch32::zero(Representation::Ntt);
    unsafe { vl::ntt_bin_batch32::<Q>(&idx, &mut out) };

    #[repr(C, align(64))]
    struct Blk27([i16; 27 * 32]);
    struct Collect {
        buf: Vec<Blk27>,
        seen: Vec<usize>,
    }
    impl vl::BlockSink for Collect {
        unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
            self.buf[blk].0.as_mut_ptr()
        }
        unsafe fn block(&mut self, blk: usize, _dst: *const i16) {
            self.seen.push(blk);
        }
    }
    let mut c = Collect {
        buf: (0..24).map(|_| Blk27([0i16; 27 * 32])).collect(),
        seen: Vec::new(),
    };
    unsafe { vl::ntt_bin_batch32_sink::<Q, _>(&idx, &mut c) };
    assert_eq!(c.seen, (0..24).collect::<Vec<_>>());
    for blk in 0..24 {
        for r in 0..27 {
            for p in 0..32 {
                assert_eq!(
                    c.buf[blk].0[32 * r + p],
                    out.v[27 * blk + r][p],
                    "block {blk} row {r}"
                );
            }
        }
    }
}

#[test]
fn sink_matches_the_plain_kernel() {
    sink::<17497>();
    sink::<19441>();
}

// ------------------------------------------------------------------ the unsigned alternative

// The butterfly this kernel could have used instead: lanes in `[0, q)` as u16, Shoup products
// with a conditional subtract, and three negations `q - t` so that the two twiddled outputs stay
// sums. Head-room is then `2^16/q` = 3.75 / 3.37 and an output lands inside `3 q + 2^11`, which
// is the point — but `Batch32`, `A`, `vpdpwssd` and every bound in `simd::commit` are signed and
// centered, so the transform would have to be centered before anything downstream could read it.
// Both forms are timed here over the same 864 butterflies, which is one batch of the kernel.

use core::arch::x86_64::*;

/// `w' = floor(w 2^16 / q)`, the Shoup companion.
fn shoup_pre(w: u16, q: u16) -> u16 {
    (((w as u32) << 16) / q as u32) as u16
}

/// `a w mod q` in `[0, q)` for `a < 2^16`: `vpmullw` + `vpmulhuw` + `vpmullw` + `vpsubw` and the
/// unsigned conditional subtract `vpsubw` + `vpminuw`.
#[inline(always)]
unsafe fn shoup(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let quot = _mm512_mulhi_epu16(a, wp);
    let r = _mm512_sub_epi16(_mm512_mullo_epi16(a, w), _mm512_mullo_epi16(quot, q));
    _mm512_min_epu16(r, _mm512_sub_epi16(r, q))
}

/// The unsigned lookup Barrett: the same five uops as the signed one on a table of
/// `-floor(2^11 s / q) q`, taking `[0, 2^16)` to `[0, q + 2^11)`.
#[inline(always)]
unsafe fn ubarrett(a: __m512i, ms: __m512i, corr: __m512i, andm: __m512i, orm: __m512i) -> __m512i {
    let s = _mm512_multishift_epi64_epi8(ms, a);
    let s = _mm512_and_si512(s, andm);
    let s = _mm512_or_si512(s, orm);
    _mm512_add_epi16(a, _mm512_permutexvar_epi8(s, corr))
}

struct U {
    q: __m512i,
    om: __m512i,
    omp: __m512i,
    ms: __m512i,
    corr: __m512i,
    andm: __m512i,
    orm: __m512i,
}

impl U {
    fn new<const Q: u16>() -> U {
        let mut corr = [0i16; 32];
        let mut ms = [0i16; 32];
        for (i, c) in corr.iter_mut().enumerate() {
            let k = (2048 * i as u32) / Q as u32;
            *c = (k * Q as u32).wrapping_neg() as u16 as i16;
        }
        for (i, m) in ms.iter_mut().enumerate() {
            *m = ((16 * (i % 4) + 11) * 257) as i16;
        }
        // byte-split so one `vpermb` does the lookup: byte u the low half, byte 32 + u the high
        let mut bytes = [0u8; 64];
        for i in 0..32 {
            bytes[i] = corr[i] as u16 as u8;
            bytes[32 + i] = (corr[i] as u16 >> 8) as u8;
        }
        let om = Params::<Q>::OMEGA;
        unsafe {
            U {
                q: _mm512_set1_epi16(Q as i16),
                om: _mm512_set1_epi16(om as i16),
                omp: _mm512_set1_epi16(shoup_pre(om, Q) as i16),
                ms: _mm512_loadu_si512(ms.as_ptr() as *const __m512i),
                corr: _mm512_loadu_si512(bytes.as_ptr() as *const __m512i),
                andm: _mm512_set1_epi16(0x1f1f),
                orm: _mm512_set1_epi16(0x2000),
            }
        }
    }
}

/// `(a0 + t1 + t2, a0 - t2 + u, a0 - t1 - u)` with every value unsigned, `t1 = z a1`,
/// `t2 = z^2 a2`, `u = omega (t1 - t2)`.
#[inline(always)]
unsafe fn ur3(
    c: &U,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    w1: __m512i,
    w1p: __m512i,
    w2: __m512i,
    w2p: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let t1 = shoup(a1, w1, w1p, c.q);
    let t2 = shoup(a2, w2, w2p, c.q);
    let n1 = _mm512_sub_epi16(c.q, t1);
    let n2 = _mm512_sub_epi16(c.q, t2);
    let u = shoup(_mm512_add_epi16(t1, n2), c.om, c.omp, c.q);
    let nu = _mm512_sub_epi16(c.q, u);
    let a0 = ubarrett(a0, c.ms, c.corr, c.andm, c.orm);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_add_epi16(a0, u), n2),
        _mm512_add_epi16(_mm512_add_epi16(a0, n1), nu),
    )
}

fn unsigned_is_the_same_butterfly<const Q: u16>() {
    let mut rng = Rng::new(0x5150 ^ Q as u64);
    let c = U::new::<Q>();
    let q = Q as u64;
    let z = Params::<Q>::ZETA_L6[7];
    let z2 = (z as u64 * z as u64 % q) as u16;
    let om = Params::<Q>::OMEGA as u64;
    let (mut a0, mut a1, mut a2) = ([0i16; 32], [0i16; 32], [0i16; 32]);
    for p in 0..32 {
        a0[p] = rng.below(3 * Q as u32 + 2048) as i16;
        a1[p] = rng.below(Q as u32) as i16;
        a2[p] = rng.below(Q as u32) as i16;
    }
    let mut out = [[0i16; 32]; 3];
    unsafe {
        let ld = |v: &[i16; 32]| _mm512_loadu_si512(v.as_ptr() as *const __m512i);
        let (o0, o1, o2) = ur3(
            &c,
            ld(&a0),
            ld(&a1),
            ld(&a2),
            _mm512_set1_epi16(z as i16),
            _mm512_set1_epi16(shoup_pre(z, Q) as i16),
            _mm512_set1_epi16(z2 as i16),
            _mm512_set1_epi16(shoup_pre(z2, Q) as i16),
        );
        _mm512_storeu_si512(out[0].as_mut_ptr() as *mut __m512i, o0);
        _mm512_storeu_si512(out[1].as_mut_ptr() as *mut __m512i, o1);
        _mm512_storeu_si512(out[2].as_mut_ptr() as *mut __m512i, o2);
    }
    for p in 0..32 {
        let (x0, x1, x2) = (a0[p] as u16 as u64, a1[p] as u64, a2[p] as u64);
        let (t1, t2) = (x1 * z as u64 % q, x2 * z2 as u64 % q);
        let u = om * ((t1 + q - t2) % q) % q;
        let want = [
            (x0 + t1 + t2) % q,
            (x0 + q - t2 + u) % q,
            (x0 + 2 * q - t1 + q - u) % q,
        ];
        for k in 0..3 {
            let got = out[k][p] as u16 as u64;
            assert!(
                got < 3 * q + 2048,
                "q={Q} lane {p} output {k}: {got} over 3q + 2^11"
            );
            assert_eq!(got % q, want[k], "q={Q} lane {p} output {k}");
        }
    }
}

/// The signed butterfly the kernel runs, written out here so the two forms are timed over the
/// same loop: Montgomery products and the Barretts the prime's own level-6 schedule asks for.
#[inline(always)]
unsafe fn sr3<const Q: u16>(
    c: &U,
    corr: __m512i,
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    w1: __m512i,
    w1p: __m512i,
    w2: __m512i,
    w2p: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let m = vl::bar_levels(Q);
    let mont = |a: __m512i, w: __m512i, wp: __m512i| {
        let m = _mm512_mullo_epi16(a, wp);
        _mm512_sub_epi16(_mm512_mulhi_epi16(a, w), _mm512_mulhi_epi16(m, c.q))
    };
    let red = |a: __m512i, k: u8| match k {
        RED_LUT => {
            let x = _mm512_multishift_epi64_epi8(c.ms, a);
            let x = _mm512_or_si512(_mm512_and_si512(x, c.andm), c.orm);
            _mm512_add_epi16(a, _mm512_permutexvar_epi8(x, corr))
        }
        RED_MUL => {
            let t = _mm512_mulhrs_epi16(a, _mm512_set1_epi16(barrett_v(Q)));
            _mm512_sub_epi16(a, _mm512_mullo_epi16(t, c.q))
        }
        _ => a,
    };
    let t1 = red(mont(a1, w1, w1p), vl::bar_kind(m, 3, 1));
    let t2 = red(mont(a2, w2, w2p), vl::bar_kind(m, 3, 1));
    let u = red(
        mont(_mm512_sub_epi16(t1, t2), c.om, c.omp),
        vl::bar_kind(m, 3, 2),
    );
    let a0 = red(a0, vl::bar_kind(m, 3, 0));
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

/// 216 butterflies — one level — in each form over the same L1-resident block, best of five.
/// The ratio is what the module comment of `ntt::bin_large` quotes.
fn compare<const Q: u16>() {
    const REPS: usize = 40000;
    let c = U::new::<Q>();
    let mut buf = Batch32::zero(Representation::Ntt);
    let mut rng = Rng::new(3);
    for row in buf.v.iter_mut() {
        for x in row.iter_mut() {
            *x = rng.below(Q as u32) as i16;
        }
    }
    let z = Params::<Q>::ZETA_L6[7];
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let p = buf.v.as_mut_ptr() as *mut i16;
    let mut bytes = [0u8; 64];
    for i in 0..32 {
        bytes[i] = barrett_lut_corr(i, Q) as u16 as u8;
        bytes[32 + i] = (barrett_lut_corr(i, Q) as u16 >> 8) as u8;
    }
    let mw = Params::<Q>::to_mont(z);
    let mw2 = Params::<Q>::to_mont(z2);

    let mut best = [f64::INFINITY; 2];
    for _ in 0..5 {
        for form in 0..2 {
            let t = std::time::Instant::now();
            unsafe {
                let corr = _mm512_loadu_si512(bytes.as_ptr() as *const __m512i);
                let (w1, w1p) = if form == 0 {
                    (
                        _mm512_set1_epi16(z as i16),
                        _mm512_set1_epi16(shoup_pre(z, Q) as i16),
                    )
                } else {
                    (
                        _mm512_set1_epi16(mw),
                        _mm512_set1_epi16(Params::<Q>::mont_pre(mw)),
                    )
                };
                let (w2, w2p) = if form == 0 {
                    (
                        _mm512_set1_epi16(z2 as i16),
                        _mm512_set1_epi16(shoup_pre(z2, Q) as i16),
                    )
                } else {
                    (
                        _mm512_set1_epi16(mw2),
                        _mm512_set1_epi16(Params::<Q>::mont_pre(mw2)),
                    )
                };
                for _ in 0..REPS {
                    for i in 0..216 {
                        let b = p.add(32 * (3 * i));
                        let ld = |k: usize| _mm512_load_si512(b.add(32 * k) as *const __m512i);
                        let (o0, o1, o2) = if form == 0 {
                            ur3(&c, ld(0), ld(1), ld(2), w1, w1p, w2, w2p)
                        } else {
                            sr3::<Q>(&c, corr, ld(0), ld(1), ld(2), w1, w1p, w2, w2p)
                        };
                        _mm512_store_si512(b as *mut __m512i, o0);
                        _mm512_store_si512(b.add(32) as *mut __m512i, o1);
                        _mm512_store_si512(b.add(64) as *mut __m512i, o2);
                    }
                }
            }
            best[form] = best[form].min(t.elapsed().as_secs_f64() / (REPS * 216) as f64 * 1e9);
            core::hint::black_box(buf.v[0][0]);
        }
    }
    println!(
        "q={Q}: {:.3} ns/butterfly unsigned, {:.3} signed (unsigned is {:.2}x)",
        best[0],
        best[1],
        best[0] / best[1]
    );
}

#[test]
fn the_unsigned_alternative() {
    unsigned_is_the_same_butterfly::<17497>();
    unsigned_is_the_same_butterfly::<19441>();
    compare::<17497>();
    compare::<19441>();
}

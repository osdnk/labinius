//! The seven NTT kernels against each other and against the scalar reference.

mod common;

mod bin_small {
    //! The pure-intrinsics reference kernel (`simd::ntt::bin_small`) against the generated one
    //! (`simd::ntt::bin_asm`) and against the scalar NTT.
    //!
    //! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
    //! element (`f162::pack4`) and sliced by the production front end, so both kernels are fed
    //! exactly what a commitment feeds them.
    //!
    //! For q = 3889 neither kernel reduces at all and they run the same operation order per
    //! butterfly, so the two outputs are **bit-identical**. For q = 9721 they reduce at different
    //! levels and with a different Barrett (reference: `params::barrett_i16` at levels 4, 5 and 6;
    //! asm: the lookup Barrett at level 4 and `barrett_i16` at level 6), so they agree only modulo q.
    use crate::common::*;
    use labinius::f162;
    use labinius::params::*;

    use labinius::scalar;

    use labinius::simd::ntt::bin_small as vb;
    use labinius::simd::ntt::bin_asm as vba;
    use labinius::ring::*;
    use labinius::F162;

    // ------------------------------------------------------------------ inputs

    /// The 648 binary coefficients of one ring element.

    // ------------------------------------------------------------------ the two kernels

    /// `EXACT`: the two kernels must agree lane by lane in i16 (q = 3889, where neither reduces);
    /// otherwise only modulo q.
    fn check_against_asm<const Q: u16, const EXACT: bool>() {
        for polys in batches(64, 0x5EED ^ Q as u64) {
            let idx = idx_of(&elems_of(&polys));
            let mut got = Batch32::zero(Representation::Coefficients);
            let mut want = Batch32::zero(Representation::Coefficients);
            unsafe {
                vb::ntt_bin_batch32::<Q>(&idx, &mut got);
                vba::ntt_bin_batch32::<Q>(&idx, &mut want);
            }
            assert_eq!(got.representation, Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    let (a, b) = (got.v[j][p] as i32, want.v[j][p] as i32);
                    if EXACT {
                        assert_eq!(a, b, "q={Q} slot={j} poly={p}");
                    } else {
                        assert_eq!((a - b) % Q as i32, 0, "q={Q} slot={j} poly={p}: {a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn matches_asm_3889() {
        check_against_asm::<3889, true>();
    }

    #[test]
    fn matches_asm_9721() {
        check_against_asm::<9721, false>();
    }

    // ------------------------------------------------------------------ the scalar reference

    /// Output against `scalar::ntt` of the lift of the very `F162` elements the front end sliced,
    /// plus the declared output bound.
    fn check_against_scalar<const Q: u16>() {
        let bound = (vb::output_bound_milli_q(Q) as i64 * Q as i64 / 1000) as i32;
        let mut worst = 0i32;
        for polys in batches(64, 0xABCD ^ Q as u64) {
            let elems = elems_of(&polys);
            let idx = idx_of(&elems);
            let mut out = Batch32::zero(Representation::Coefficients);
            unsafe { vb::ntt_bin_batch32::<Q>(&idx, &mut out) };
            for p in 0..32 {
                let coeffs: [F162; 4] = elems[4 * p..4 * p + 4].try_into().unwrap();
                let coeffs = f162::lift4(&coeffs);
                assert_eq!(coeffs, polys[p], "lift4 {p}");
                let want = scalar::ntt::<Q>(&coeffs);
                let e = out.get(p);
                let got = e.normalized(Q);
                for j in 0..N {
                    assert_eq!(got[j], want[j], "q={Q} poly={p} slot={j}");
                    let a = (e.v[j] as i32).abs();
                    worst = worst.max(a);
                    assert!(
                        a <= bound,
                        "q={Q} output {a} exceeds declared bound {bound}"
                    );
                }
            }
        }
        println!(
            "q={Q}: reference kernel max |output| {worst} = {:.3}q (declared {:.3}q)",
            worst as f64 / Q as f64,
            bound as f64 / Q as f64
        );
    }

    #[test]
    fn matches_scalar_3889() {
        check_against_scalar::<3889>();
    }

    #[test]
    fn matches_scalar_9721() {
        check_against_scalar::<9721>();
    }
}

mod bin_asm {
    //! Correctness and bound tests for the binary vertical NTT.
    //!
    //! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
    //! element (`f162::pack4`) and sliced by the production front end, so the kernel is fed exactly
    //! what a commitment feeds it.
    use crate::common::*;
    use labinius::f162;
    use labinius::params::*;

    use labinius::scalar;
    use labinius::simd::transpose_f162::{self as tf, BinaryIndex32};
    use labinius::simd::ntt::bin_asm as vb;
    use labinius::simd::ntt::bin_large as vl;
    use labinius::ring::*;

    // ------------------------------------------------------------------ inputs

    /// The 648 binary coefficients of one ring element.

    // ------------------------------------------------------------------ the front end

    /// The AVX-512 slicer against the scalar definition of the index rows.
    #[test]
    fn transpose_matches_scalar() {
        for polys in batches(64, 11) {
            let e = elems_of(&polys);
            let want = f162::index_rows_scalar(&e);
            let mut got = BinaryIndex32::zero();
            unsafe { tf::slice_f162_into(&e, &mut got) };
            for i in 0..162 {
                assert!(want.rows[i] == got.rows[i], "slice_f162_into row {i}");
            }
            // and the lift really is the coefficient vector the test built
            for p in 0..32 {
                assert_eq!(f162::lift_elem(&e, p), polys[p], "lift_elem {p}");
            }
        }
    }

    // ------------------------------------------------------------------ i32 shadow model
    //
    // Mirrors the kernel's exact schedule lane-wise in i32 so that every intermediate can be checked
    // against the 2^15 invariant, and the *exact* (not just mod q) output can be compared.

    struct Shadow<const Q: u16> {
        /// max |value| seen after: [0] levels 0+1+2 (table combines), [1..=4] levels 3, 4, 5, 6.
        max: [i32; 5],
        /// max |value| of any intermediate whatsoever (must stay < 2^15).
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
            if a > self.max_any {
                self.max_any = a;
            }
            assert!(a < 32768, "i16 overflow: {x}");
            x as i16
        }
        fn mont(&mut self, a: i16, x: u16) -> i16 {
            let w = Params::<Q>::to_mont(x);
            let r = mont_mul_i16(a, w, Params::<Q>::mont_pre(w), Q);
            self.see(r as i32)
        }
        /// The 16-entry table value used by the kernel for block k, level-3 child s2, role r, side ab.
        fn table(k: usize, s2: usize, r: usize, ab: usize, n: usize) -> i16 {
            let q = Q as u64;
            let (s0, s1) = (k / 2, k % 2);
            let z6 = Params::<Q>::ZETA6 as u64;
            let ka = if s0 == 0 { z6 } else { (1 + q - z6) % q };
            let z1 = Params::<Q>::ZETA_L1[s0] as u64;
            let zp = Params::<Q>::ZETA_L2[k] as u64;
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            let f = pow_mod(z3, r as u64, q);
            let extra = if ab == 0 { 1 } else { zp };
            let (n0, n1, n2, n3) = (
                (n & 1) as u64,
                ((n >> 1) & 1) as u64,
                ((n >> 2) & 1) as u64,
                ((n >> 3) & 1) as u64,
            );
            let inner = z1 * ((n1 + ka * n3) % q) % q;
            let t = if s1 == 0 { inner } else { (q - inner) % q };
            let base = ((n0 + ka * n2) % q + t) % q;
            center(base * f % q * extra % q, q)
        }
        /// a0 + t1 + t2, a0 - t2 + u, a0 - t1 - u with u = omega * (t1 - t2).
        fn r3(&mut self, a0: i16, t1: i16, t2: i16) -> (i16, i16, i16) {
            let d = self.see(t1 as i32 - t2 as i32);
            let u = self.mont(d, Params::<Q>::OMEGA);
            let o0 = self.see(a0 as i32 + t1 as i32 + t2 as i32);
            let o1 = self.see(a0 as i32 - t2 as i32 + u as i32);
            let o2 = self.see(a0 as i32 - t1 as i32 - u as i32);
            (o0, o1, o2)
        }
        /// `red`: 0 = no reduction of the un-twiddled a0, 1 = the lookup Barrett, 2 = the
        /// two-multiply `vpmulhrsw` one.
        fn r3_tw(&mut self, a0: i16, a1: i16, a2: i16, z: u16, red: u8) -> (i16, i16, i16) {
            let z2 = (z as u64 * z as u64 % Q as u64) as u16;
            let t1 = self.mont(a1, z);
            let t2 = self.mont(a2, z2);
            let a0 = match red {
                1 => self.see(vb::barrett_lut_i16(a0, Q) as i32),
                2 => self.see(barrett_i16(a0, Q) as i32),
                _ => a0,
            };
            self.r3(a0, t1, t2)
        }

        fn run(&mut self, poly: &Bin) -> [i16; N] {
            let bar = vb::needs_barrett(Q);
            let mut v = [0i16; N];
            let nib: Vec<usize> = (0..162)
                .map(|i| {
                    (poly[i] | poly[i + 162] << 1 | poly[i + 324] << 2 | poly[i + 486] << 3) as usize
                })
                .collect();
            for k in 0..4 {
                let base = 162 * k;
                for ii in 0..27 {
                    // levels 0+1+2 with the level-3 twiddles folded in
                    let mut six = [[0i16; 3]; 2];
                    for r in 0..3 {
                        let (lo, hi) = (nib[ii + 27 * r], nib[ii + 27 * r + 81]);
                        for h in 0..2 {
                            let s2 = if r == 0 { 0 } else { h };
                            let x = Self::table(k, s2, r, 0, lo) as i32;
                            let y = Self::table(k, s2, r, 1, hi) as i32;
                            let val = if h == 0 { x + y } else { x - y };
                            six[h][r] = self.see(val);
                            if six[h][r].unsigned_abs() as i32 > self.max[0] {
                                self.max[0] = six[h][r].unsigned_abs() as i32;
                            }
                        }
                    }
                    // level 3 (only the omega multiply is left)
                    for h in 0..2 {
                        let (o0, o1, o2) = self.r3(six[h][0], six[h][1], six[h][2]);
                        v[base + 81 * h + ii] = o0;
                        v[base + 81 * h + ii + 27] = o1;
                        v[base + 81 * h + ii + 54] = o2;
                    }
                }
            }
            for x in v.iter() {
                self.max[1] = self.max[1].max(x.unsigned_abs() as i32);
            }
            for (li, level) in [4usize, 5, 6].into_iter().enumerate() {
                let red = match (bar, vb::LUT_BARRETT_LEVELS[li], vb::MUL_BARRETT_LEVELS[li]) {
                    (true, true, _) => 1,
                    (true, _, true) => 2,
                    _ => 0,
                };
                let n = DEGREE[level];
                let m = n / 3;
                for k in 0..SUBRINGS[level] {
                    let b = k * n;
                    let z = Params::<Q>::zeta(level, k);
                    for i in 0..m {
                        let (o0, o1, o2) = self.r3_tw(v[b + i], v[b + m + i], v[b + 2 * m + i], z, red);
                        v[b + i] = o0;
                        v[b + m + i] = o1;
                        v[b + 2 * m + i] = o2;
                    }
                }
                for x in v.iter() {
                    self.max[2 + li] = self.max[2 + li].max(x.unsigned_abs() as i32);
                }
            }
            v
        }
    }

    // ------------------------------------------------------------------ main kernel test

    fn check_kernel<const Q: u16>() {
        let bound = (vb::output_bound_milli_q(Q) as i64 * Q as i64 / 1000) as i32;
        let mut sh = Shadow::<Q>::new();
        let mut worst_out = 0i32;
        for polys in batches(64, 0xABCD ^ Q as u64) {
            let bb = idx_of_polys(&polys);
            let mut out = Batch32::zero(Representation::Coefficients);
            unsafe { vb::ntt_bin_batch32::<Q>(&bb, &mut out) };
            assert_eq!(out.representation, Representation::Ntt);
            for p in 0..32 {
                let want = scalar::ntt::<Q>(&polys[p]);
                let e = out.get(p);
                let got = e.normalized(Q);
                for j in 0..N {
                    assert_eq!(got[j], want[j], "q={Q} poly={p} slot={j}");
                }
                // exact agreement with the i32 shadow schedule (not just modulo q)
                let shadow = sh.run(&polys[p]);
                for j in 0..N {
                    assert_eq!(e.v[j], shadow[j], "shadow mismatch q={Q} poly={p} slot={j}");
                }
                for j in 0..N {
                    let a = (e.v[j] as i32).abs();
                    worst_out = worst_out.max(a);
                    assert!(
                        a <= bound,
                        "q={Q} output {a} exceeds declared bound {bound}"
                    );
                }
            }
        }
        let q = Q as f64;
        println!(
            "q={Q}: bounds (multiples of q) levels 0-2 {:.3}, l3 {:.3}, l4 {:.3}, l5 {:.3}, l6 {:.3}; \
             max intermediate {:.3}q = {}; declared output bound {:.3}q",
            sh.max[0] as f64 / q,
            sh.max[1] as f64 / q,
            sh.max[2] as f64 / q,
            sh.max[3] as f64 / q,
            sh.max[4] as f64 / q,
            sh.max_any as f64 / q,
            sh.max_any,
            bound as f64 / q
        );
        assert!(sh.max_any < 32768);
        assert_eq!(worst_out, sh.max[4]);
    }

    #[test]
    fn kernel_3889() {
        check_kernel::<3889>();
    }
    #[test]
    fn kernel_9721() {
        check_kernel::<9721>();
    }

    // ------------------------------------------------------------------ the lookup Barrett

    /// The reduction the kernel's `vpmultishiftqb` + `vpermb` + `vpaddw` sequence performs, checked
    /// exhaustively over every i16 input: it subtracts a multiple of q and leaves |r| <= 0.579 q,
    /// where the two-multiply `barrett_i16` only guarantees 0.809 q.
    #[test]
    fn barrett_lut_exhaustive() {
        for &q in QS.iter().chain(QS_LARGE.iter()) {
            let (mut worst, mut worst_mul) = (0i32, 0i32);
            for a in i16::MIN..=i16::MAX {
                let r = vb::barrett_lut_i16(a, q);
                // same residue class
                assert_eq!(
                    (a as i32).rem_euclid(q as i32),
                    (r as i32).rem_euclid(q as i32),
                    "q={q} a={a}"
                );
                // the correction really is a multiple of q, and the i16 addition is the one the
                // kernel does — above 2^14, `-k q` itself can leave i16 (k reaches 2 at 17497) and
                // only the result has to land back inside it, which the residue check above pins
                let corr = vb::barrett_lut_corr(((a >> 11) & 31) as usize, q);
                assert_eq!((a as i32 - r as i32) % q as i32, 0);
                assert_eq!(a.wrapping_add(corr), r, "q={q} a={a}");
                worst = worst.max((r as i32).abs());
                worst_mul = worst_mul.max((barrett_i16(a, q) as i32).abs());
            }
            println!(
                "q={q}: lookup Barrett |r| <= {worst} = {:.4} q (two-multiply {worst_mul} = {:.4} q)",
                worst as f64 / q as f64,
                worst_mul as f64 / q as f64
            );
            assert!(worst < q as i32);
            if q == 9721 {
                assert_eq!(worst, 5625);
            }
            // what `ntt::bin_large` reduces to, and what its `barrett_lut_max` sweep returns
            if vl::is_large(q) {
                assert_eq!(worst, vl::barrett_lut_max(q));
                assert!(worst as f64 / q as f64 <= 0.532);
            }
        }
    }

    /// The *proved* bound of the schedule (the shadow model only sees the inputs the test feeds it):
    /// worst case over all i16 lane values, level by level, with
    /// `|mont(a, w)| <= (|a| (q-1)/2 + 2^15 q) / 2^16` and `|barrett| <= 5625 / 7864`.
    #[test]
    fn proved_bounds_9721() {
        const Q: i64 = 9721;
        let mont = |b: i64| (b * (Q - 1) / 2 + 32768 * Q) / 65536;
        // one radix-3 butterfly: a0 bounded by `a`, a1 / a2 by `b`; returns (y0, y1 = y2, max
        // intermediate).
        let bf = |a: i64, b: i64| {
            let m = mont(b);
            let u = mont(2 * m);
            (a + 2 * m, a + m + u, (2 * m).max(a + 2 * m).max(a + m + u))
        };
        let lut = 5625; // barrett_lut_exhaustive
        let mul = 7864; // params::barrett_i16, exhaustive (see the same test)
                        // levels 0-2: two centred table entries, |T| <= (q-1)/2
        let l2 = Q - 1;
        // level 3: the twiddles are folded, so y0 = a0 + t1 + t2 with all three <= l2
        let l3 = 3 * l2;
        // the level-3 intermediates: t1 - t2 <= 2 l2, y0 = a0 + t1 + t2 <= 3 l2, y1 = y2 <= 2 l2 + u
        assert!(l3.max(2 * l2).max(2 * l2 + mont(2 * l2)) < 32768);
        // level 4: lookup Barrett on a0
        let (l4y0, l4y1, mi4) = bf(lut, l3);
        // level 5: unreduced; the three 9-blocks of a 27-block hold y0-, y1- and y2-values, so each
        // butterfly's three inputs are all of one kind
        let (l5y0a, l5y1a, mi5a) = bf(l4y0, l4y0);
        let (l5y0b, l5y1b, mi5b) = bf(l4y1, l4y1);
        let (l5y0, l5y1) = (l5y0a.max(l5y0b), l5y1a.max(l5y1b));
        // level 6: two-multiply Barrett on a0 (which is a level-5 y0), a1 / a2 are level-5 y1 / y2
        let (l6y0, l6y1, mi6) = bf(mul, l5y1);
        let out = l6y0.max(l6y1);
        let declared = vb::output_bound_milli_q(9721) as i64 * Q / 1000;
        println!(
            "q=9721 proved: l3 {:.3}q  l4 {:.3}q  l5 {:.3}q  out {:.4}q = {out}  (declared {:.3}q);\
             \n  max intermediate {} of 32767, i.e. {} to spare",
            l3 as f64 / Q as f64,
            l4y0 as f64 / Q as f64,
            l5y0 as f64 / Q as f64,
            out as f64 / Q as f64,
            declared as f64 / Q as f64,
            mi4.max(mi5a).max(mi5b).max(mi6),
            32767 - mi4.max(mi5a).max(mi5b).max(mi6)
        );
        for m in [mi4, mi5a, mi5b, mi6] {
            assert!(m < 32768, "i16 overflow in the proved bound: {m}");
        }
        assert!(
            out <= declared,
            "output {out} exceeds the declared bound {declared}"
        );
        // and the reason level 5 can be skipped at all: with the two-multiply Barrett at level 4 it
        // could not.
        let (w0, _, _) = bf(mul, l3);
        assert!(
            bf(w0, w0).2 >= 32768,
            "the level-5 reduction would not have been needed"
        );
    }
}

mod bin_large {
    //! `simd::ntt::bin_large` against the scalar NTT, against an i32 shadow of its own schedule,
    //! and against the `const` bound recursion that chose that schedule.
    //!
    //! Inputs are built as 648 binary coefficients, packed back into the four `F162` of a ring
    //! element (`f162::pack4`) and sliced by the production front end, so the kernel is fed exactly
    //! what a commitment feeds it.
    use crate::common::*;
    use labinius::f162;
    use labinius::params::*;
    use labinius::rng::Rng;
    use labinius::scalar;

    use labinius::simd::ntt::bin_asm::{barrett_lut_corr, barrett_lut_i16};
    use labinius::simd::ntt::bin_large as vl;
    use labinius::simd::ntt::bin_large::{RED_LUT, RED_MUL};
    use labinius::ring::*;

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
    // is the point — but `Batch32`, `A`, `vpmaddwd` and every bound in `simd::commit` are signed and
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
}

mod gen_small {
    //! Correctness and bound tests for `simd::ntt::gen_small`.

    use labinius::params::*;
    use labinius::rng::Rng;
    use labinius::scalar;
    use labinius::simd::ntt::gen_small::{intt_gen_batch32, ntt_gen_batch32, Tw, TwI};
    use labinius::ring::*;

    /// Exact i32 mirror of the kernel: same operation order, same Barrett placement, but every value
    /// kept as i32 so that an i16 overflow is observable. Returns the output and the per-level maximum
    /// absolute value (index l = after level l).
    fn shadow<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 7]) -> [i32; N] {
        let q = Q as u64;
        let mut v = [0i32; N];
        for i in 0..N {
            v[i] = input[i] as i32;
            assert!(v[i].abs() <= Q as i32, "input bound");
        }
        let mont = |a: i32, x: u16| -> i32 {
            assert!(
                a.abs() < 32768,
                "i16 overflow feeding a multiplication: {a}"
            );
            let w = Params::<Q>::to_mont(x);
            params_mont(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
        };
        // level 0: Phi_6 split, with the pass-A Barrett on the a0 half of the second child.
        let z6 = Params::<Q>::ZETA6;
        for i in 0..324 {
            let (a0, a1) = (v[i], v[i + 324]);
            let t = mont(a1, z6);
            v[i] = a0 + t;
            let mut x = a0 + a1 - t;
            if Tw::<Q>::BAR_A && i < 162 {
                assert!(x.abs() < 32768, "i16 overflow before barrett: {x}");
                x = barrett_i16(x as i16, Q) as i32;
            }
            v[i + 324] = x;
        }
        lmax[0] = v.iter().map(|x| x.abs()).max().unwrap();
        let w1 = Params::<Q>::OMEGA;
        for level in 1..=6 {
            let n = DEGREE[level];
            let p = RADIX[level];
            let m = n / p;
            for k in 0..SUBRINGS[level] {
                let base = k * n;
                let zeta = pow_mod(Params::<Q>::PSI as u64, twiddle_exp(level, k) as u64, q) as u16;
                let zeta2 = (zeta as u64 * zeta as u64 % q) as u16;
                for i in 0..m {
                    let mut a0 = v[base + i];
                    if Tw::<Q>::BAR_L[level] {
                        assert!(a0.abs() < 32768, "i16 overflow before barrett: {a0}");
                        a0 = barrett_i16(a0 as i16, Q) as i32;
                    }
                    if p == 2 {
                        let t = mont(v[base + m + i], zeta);
                        v[base + i] = a0 + t;
                        v[base + m + i] = a0 - t;
                    } else {
                        let t1 = mont(v[base + m + i], zeta);
                        let t2 = mont(v[base + 2 * m + i], zeta2);
                        let u = mont(t1 - t2, w1);
                        v[base + i] = a0 + t1 + t2;
                        v[base + m + i] = a0 - t2 + u;
                        v[base + 2 * m + i] = a0 - t1 - u;
                    }
                }
            }
            lmax[level] = v.iter().map(|x| x.abs()).max().unwrap();
        }
        v
    }

    fn params_mont(a: i16, w: i16, wp: i16, q: u16) -> i16 {
        mont_mul_i16(a, w, wp, q)
    }

    fn to_batch(cols: &[[i16; N]; 32]) -> Batch32 {
        let mut b = Batch32::zero(Representation::Coefficients);
        for p in 0..32 {
            for j in 0..N {
                b.v[j][p] = cols[p][j];
            }
        }
        b
    }

    fn check<const Q: u16>(cols: &[[i16; N]; 32], what: &str) {
        let mut b = to_batch(cols);
        unsafe { ntt_gen_batch32::<Q>(&mut b) };
        assert_eq!(b.representation, Representation::Ntt);
        let bound = Tw::<Q>::OUTPUT_BOUND;
        for p in 0..32 {
            let mut lmax = [0i32; 7];
            let want_shadow = shadow::<Q>(&cols[p], &mut lmax);
            let mut coeffs = [0u32; N];
            for j in 0..N {
                coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
            }
            let want = scalar::ntt::<Q>(&coeffs);
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert_eq!(
                    got, want_shadow[j],
                    "{what} q={Q} poly {p} slot {j}: shadow mismatch"
                );
                assert_eq!(
                    got.rem_euclid(Q as i32) as u32,
                    want[j],
                    "{what} q={Q} poly {p} slot {j}"
                );
                assert!(
                    got.abs() <= bound,
                    "{what} q={Q} poly {p} slot {j}: |{got}| > {bound}"
                );
            }
        }
    }

    fn random_cols<const Q: u16>(rng: &mut Rng) -> [[i16; N]; 32] {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16)
        })
    }

    /// Binary columns: the input a commitment's generic-side kernel sees when it is fed a lift.
    fn binary_cols(rng: &mut Rng) -> [[i16; N]; 32] {
        std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() & 1) as i16))
    }

    fn adversarial<const Q: u16>() -> Vec<[[i16; N]; 32]> {
        let q = Q as i16;
        let mut out = Vec::new();
        out.push([[0i16; N]; 32]);
        out.push([[q; N]; 32]);
        out.push([[-q; N]; 32]);
        out.push(std::array::from_fn(|_| {
            std::array::from_fn(|j| if j % 2 == 0 { q } else { -q })
        }));
        out.push(std::array::from_fn(|p| {
            std::array::from_fn(|j| if (j + p) % 2 == 0 { q } else { -q })
        }));
        for &m in &[0usize, 161, 162, 323, 324, 647] {
            for &val in &[1i16, q, -q] {
                let mut c = [[0i16; N]; 32];
                for p in 0..32 {
                    c[p][m] = val;
                }
                out.push(c);
            }
        }
        // one monomial per polynomial, all different positions
        let mut c = [[0i16; N]; 32];
        for p in 0..32 {
            c[p][(p * 21) % N] = q;
        }
        out.push(c);
        out
    }

    fn run<const Q: u16>() {
        let mut rng = Rng::new(0x5eed ^ Q as u64);
        for (i, c) in adversarial::<Q>().iter().enumerate() {
            check::<Q>(c, &format!("adversarial#{i}"));
        }
        for i in 0..32 {
            check::<Q>(&binary_cols(&mut rng), &format!("binary#{i}"));
        }
        for i in 0..32 {
            check::<Q>(&random_cols::<Q>(&mut rng), &format!("random#{i}"));
        }
    }

    #[test]
    fn ntt_3889() {
        run::<3889>();
    }

    #[test]
    fn ntt_9721() {
        run::<9721>();
    }

    /// Per-level bounds proven in the module comment of `ntt::gen_small`, as `ceil(bound * q)`.
    fn level_bounds<const Q: u16>() -> [i32; 7] {
        if Q == 9721 {
            [25024, 21298, 27738, 21700, 20804, 20671, 20652]
        } else {
            [9838, 12075, 14378, 19120, 24143, 8819, 13231]
        }
    }

    fn bounds<const Q: u16>() {
        let mut rng = Rng::new(0xb0 ^ Q as u64);
        let mut worst = [0i32; 7];
        let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
        for _ in 0..8 {
            cases.push(random_cols::<Q>(&mut rng));
            cases.push(binary_cols(&mut rng));
        }
        for c in &cases {
            for p in 0..32 {
                let mut lmax = [0i32; 7];
                shadow::<Q>(&c[p], &mut lmax);
                for l in 0..7 {
                    worst[l] = worst[l].max(lmax[l]);
                }
            }
        }
        let claim = level_bounds::<Q>();
        for l in 0..7 {
            let c = claim[l];
            assert!(
                worst[l] <= c,
                "q={Q} level {l}: observed {} > claimed {c}",
                worst[l]
            );
            assert!(c < 32768, "q={Q} level {l}: claimed bound {c} exceeds i16");
            println!(
                "q={Q} level {l}: observed {} ({:.3} q), claimed {c} ({:.4} q)",
                worst[l],
                worst[l] as f64 / Q as f64,
                c as f64 / Q as f64
            );
        }
        assert_eq!(Tw::<Q>::OUTPUT_BOUND, claim[6]);
    }

    #[test]
    fn bounds_3889() {
        bounds::<3889>();
    }

    #[test]
    fn bounds_9721() {
        bounds::<9721>();
    }

    // ------------------------------------------------------------------ the inverse transform

    /// Exact i32 mirror of `intt_gen_batch32`: same butterfly order, same Barrett placement, every
    /// value kept as i32 so an i16 overflow is observable. `lmax[l]` = max |value| after level `l`.
    fn shadow_inv<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 7]) -> [i32; N] {
        let q = Q as u64;
        let mut v = [0i32; N];
        for i in 0..N {
            v[i] = input[i] as i32;
            assert!(v[i].abs() <= TwI::<Q>::IN_BOUND, "input bound: {}", v[i]);
        }
        let mont = |a: i32, x: u16| -> i32 {
            assert!(
                a.abs() < 32768,
                "i16 overflow feeding a multiplication: {a}"
            );
            let w = Params::<Q>::to_mont(x);
            mont_mul_i16(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
        };
        let bar = |a: i32| -> i32 {
            assert!(a.abs() < 32768, "i16 overflow before barrett: {a}");
            barrett_i16(a as i16, Q) as i32
        };
        let ck = |a: i32| -> i32 {
            assert!(a.abs() < 32768, "i16 overflow: {a}");
            a
        };
        let w1 = Params::<Q>::OMEGA;
        let zi = |level: usize, k: usize| -> u16 {
            inv_mod(
                pow_mod(Params::<Q>::PSI as u64, twiddle_exp(level, k) as u64, q),
                q,
            ) as u16
        };
        // one inverse radix-3 butterfly on the three positions, returning the untwiddled sum first
        let r3i = |v: &mut [i32; N], i0: usize, i1: usize, i2: usize, z: u16, bar_s: bool| {
            let (y0, y1, y2) = (v[i0], v[i1], v[i2]);
            let u = mont(ck(y2 - y1), w1);
            let s = ck(y0 + y1 + y2);
            let z2 = (z as u64 * z as u64 % q) as u16;
            let a1 = mont(ck(y0 - y1 + u), z);
            let a2 = mont(ck(y0 - y2 - u), z2);
            v[i0] = if bar_s { bar(s) } else { s };
            v[i1] = a1;
            v[i2] = a2;
        };
        for k4 in 0..24 {
            for g in 0..9 {
                let b = 27 * k4 + 3 * g;
                if TwI::<Q>::BAR_IN {
                    for t in 0..3 {
                        v[b + t] = bar(v[b + t]);
                    }
                }
                r3i(&mut v, b, b + 1, b + 2, zi(6, 9 * k4 + g), TwI::<Q>::BAR_S6);
            }
        }
        lmax[6] = v.iter().map(|x| x.abs()).max().unwrap();
        for k4 in 0..24 {
            for bb in 0..3 {
                for j in 0..3 {
                    let b = 27 * k4 + 9 * bb + j;
                    r3i(
                        &mut v,
                        b,
                        b + 3,
                        b + 6,
                        zi(5, 3 * k4 + bb),
                        TwI::<Q>::BAR_S5[j],
                    );
                }
            }
        }
        lmax[5] = v.iter().map(|x| x.abs()).max().unwrap();
        for k4 in 0..24 {
            for i in 0..9 {
                let b = 27 * k4 + i;
                r3i(&mut v, b, b + 9, b + 18, zi(4, k4), TwI::<Q>::BAR_S4[i]);
            }
        }
        lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
        for k in 0..8 {
            for j in 0..27 {
                let b = 81 * k + j;
                r3i(&mut v, b, b + 27, b + 54, zi(3, k), TwI::<Q>::BAR_S3);
            }
        }
        lmax[3] = v.iter().map(|x| x.abs()).max().unwrap();
        for blk in 0..4 {
            for a in 0..3 {
                for j in 0..27 {
                    let b = 162 * blk + 27 * a + j;
                    let (y0, y1) = (v[b], v[b + 81]);
                    let s = ck(y0 + y1);
                    v[b] = if TwI::<Q>::BAR_S2[a] { bar(s) } else { s };
                    v[b + 81] = mont(ck(y0 - y1), zi(2, blk));
                }
            }
        }
        lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
        for c in 0..2 {
            for i in 0..162 {
                let b = 324 * c + i;
                let (y0, y1) = (v[b], v[b + 162]);
                let s = ck(y0 + y1);
                v[b] = if TwI::<Q>::BAR_S1 { bar(s) } else { s };
                v[b + 162] = mont(ck(y0 - y1), zi(1, c));
            }
        }
        lmax[1] = v.iter().map(|x| x.abs()).max().unwrap();
        // level 0: the Phi_6 recombination carries the whole 1/648 * det normalisation, then centering.
        let det = inv_mod((2 * Params::<Q>::ZETA6 as u64 + q - 1) % q, q);
        let ka = (det * inv_mod(324, q) % q) as u16;
        let kb = inv_mod(648, q) as u16;
        let kc = ((q - det * inv_mod(648, q) % q) % q) as u16;
        let half = (Q as i32 - 1) / 2;
        let mut raw = 0i32;
        let mut out = [0i32; N];
        for i in 0..324 {
            let (y0, y1) = (v[i], v[i + 324]);
            let d = ck(y0 - y1);
            let s = ck(y0 + y1);
            let a1 = mont(d, ka);
            let a0 = ck(mont(s, kb) + mont(d, kc));
            raw = raw.max(a0.abs()).max(a1.abs());
            let center = |mut x: i32| {
                if x > half {
                    x -= Q as i32;
                }
                if x < -half {
                    x += Q as i32;
                }
                x
            };
            out[i] = center(a0);
            out[i + 324] = center(a1);
        }
        lmax[0] = raw;
        out
    }

    fn check_inv<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut [i32; 7]) {
        let mut b = Batch32::zero(Representation::Ntt);
        for p in 0..32 {
            for j in 0..N {
                b.v[j][p] = cols[p][j];
            }
        }
        unsafe { intt_gen_batch32::<Q>(&mut b) };
        assert_eq!(b.representation, Representation::Coefficients);
        let half = TwI::<Q>::OUT_BOUND;
        for p in 0..32 {
            let mut lmax = [0i32; 7];
            let want_shadow = shadow_inv::<Q>(&cols[p], &mut lmax);
            for l in 0..7 {
                worst[l] = worst[l].max(lmax[l]);
            }
            let want = scalar::intt::<Q>(&scalar::normalize_i16(&cols[p], Q));
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert_eq!(
                    got, want_shadow[j],
                    "{what} q={Q} poly {p} coeff {j}: shadow mismatch"
                );
                assert!(
                    got.abs() <= half,
                    "{what} q={Q} poly {p} coeff {j}: |{got}| > {half}"
                );
                assert_eq!(
                    got.rem_euclid(Q as i32) as u32,
                    want[j],
                    "{what} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }

    /// NTT-domain columns with `|v| <= bound`.
    fn random_ntt_cols<const Q: u16>(rng: &mut Rng, bound: i32) -> [[i16; N]; 32] {
        std::array::from_fn(|_| {
            std::array::from_fn(|_| (rng.below(2 * bound as u32 + 1) as i32 - bound) as i16)
        })
    }

    fn adversarial_ntt<const Q: u16>() -> Vec<[[i16; N]; 32]> {
        let m = TwI::<Q>::IN_BOUND as i16;
        let mut out = Vec::new();
        out.push([[0i16; N]; 32]);
        out.push([[m; N]; 32]);
        out.push([[-m; N]; 32]);
        out.push(std::array::from_fn(|_| {
            std::array::from_fn(|j| if j % 2 == 0 { m } else { -m })
        }));
        out.push(std::array::from_fn(|_| {
            std::array::from_fn(|j| if j % 3 == 0 { m } else { -m })
        }));
        out.push(std::array::from_fn(|p| {
            std::array::from_fn(|j| if (j / 27 + p) % 2 == 0 { m } else { -m })
        }));
        for &u in &[0usize, 1, 2, 26, 27, 80, 81, 323, 324, 647] {
            let mut c = [[0i16; N]; 32];
            for p in 0..32 {
                c[p][u] = if p % 2 == 0 { m } else { -m };
            }
            out.push(c);
        }
        out
    }

    fn run_inv<const Q: u16>() {
        let mut rng = Rng::new(0x1117 ^ Q as u64);
        let mut worst = [0i32; 7];
        for (i, c) in adversarial_ntt::<Q>().iter().enumerate() {
            check_inv::<Q>(c, &format!("adversarial#{i}"), &mut worst);
        }
        for i in 0..12 {
            let c = random_ntt_cols::<Q>(&mut rng, TwI::<Q>::IN_BOUND);
            check_inv::<Q>(&c, &format!("lazy#{i}"), &mut worst);
        }
        for i in 0..12 {
            let c = random_ntt_cols::<Q>(&mut rng, (Q as i32 - 1) / 2);
            check_inv::<Q>(&c, &format!("centered#{i}"), &mut worst);
        }
        // the real inputs: the forward kernel's own output
        for i in 0..8 {
            let cols = random_cols::<Q>(&mut rng);
            let mut b = to_batch(&cols);
            unsafe { ntt_gen_batch32::<Q>(&mut b) };
            let ntt: [[i16; N]; 32] = std::array::from_fn(|p| std::array::from_fn(|j| b.v[j][p]));
            check_inv::<Q>(&ntt, &format!("forward#{i}"), &mut worst);
        }
        let claim = TwI::<Q>::BOUND;
        for l in 0..7 {
            assert!(
                worst[l] <= claim[l],
                "q={Q} inverse level {l}: {} > claimed {}",
                worst[l],
                claim[l]
            );
            assert!(
                claim[l] < 32768,
                "q={Q} inverse level {l}: claimed {} exceeds i16",
                claim[l]
            );
            println!(
                "q={Q} inverse level {l}: observed {} ({:.3} q), claimed {} ({:.4} q)",
                worst[l],
                worst[l] as f64 / Q as f64,
                claim[l],
                claim[l] as f64 / Q as f64
            );
        }
        println!(
            "q={Q} inverse: peak intermediate {} ({:.4} q, budget {:.4} q), {} Barretts per batch",
            TwI::<Q>::PEAK,
            TwI::<Q>::PEAK as f64 / Q as f64,
            32767.0 / Q as f64,
            216 * (3 * TwI::<Q>::BAR_IN as u32 + TwI::<Q>::BAR_S6 as u32 + TwI::<Q>::BAR_S3 as u32)
                + 72 * (TwI::<Q>::BAR_S5[0] as u32
                    + TwI::<Q>::BAR_S5[1] as u32
                    + TwI::<Q>::BAR_S5[2] as u32)
                + 24 * TwI::<Q>::BAR_S4.iter().filter(|x| **x).count() as u32
                + 108
                    * (TwI::<Q>::BAR_S2[0] as u32
                        + TwI::<Q>::BAR_S2[1] as u32
                        + TwI::<Q>::BAR_S2[2] as u32)
                + 324 * TwI::<Q>::BAR_S1 as u32
        );
    }

    #[test]
    fn intt_3889() {
        run_inv::<3889>();
    }

    #[test]
    fn intt_9721() {
        run_inv::<9721>();
    }

    /// `intt(ntt(x)) == x` for the centered representative of `x mod q`.
    fn round_trip<const Q: u16>() {
        let mut rng = Rng::new(0x0d0d ^ Q as u64);
        let half = (Q as i32 - 1) / 2;
        let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
        for _ in 0..8 {
            cases.push(binary_cols(&mut rng));
            cases.push(random_cols::<Q>(&mut rng));
        }
        for (c, cols) in cases.iter().enumerate() {
            let mut b = to_batch(cols);
            unsafe {
                ntt_gen_batch32::<Q>(&mut b);
                intt_gen_batch32::<Q>(&mut b);
            }
            for p in 0..32 {
                for j in 0..N {
                    let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                    if want > half {
                        want -= Q as i32;
                    }
                    assert_eq!(
                        b.v[j][p] as i32, want,
                        "round trip #{c} q={Q} poly {p} coeff {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn round_trip_3889() {
        round_trip::<3889>();
    }

    #[test]
    fn round_trip_9721() {
        round_trip::<9721>();
    }
}

mod gen_large {
    //! `simd::ntt::gen_large` against `scalar::ntt` / `scalar::intt`, against its own declared
    //! bounds and against the `const` recursions that prove them.

    use labinius::params::*;
    use labinius::rng::Rng;
    use labinius::scalar;
    use labinius::simd::ntt::gen_large as vgl;
    use labinius::ring::{Batch32, Representation};

    /// Adversarial coefficient batches at the declared input bound `|x| <= q`, then random ones.
    fn inputs<const Q: u16>(count: usize, seed: u64) -> Vec<Batch32> {
        let mut rng = Rng::new(seed);
        let q = Q as i16;
        let mut out = Vec::new();
        for &f in &[q, -q] {
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = f;
                }
            }
            out.push(b);
        }
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = if (j + p) % 2 == 0 { q } else { -q };
            }
        }
        out.push(b);
        for _ in 0..count {
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(2 * Q as u32 + 1) as i16 - q;
                }
            }
            out.push(b);
        }
        out
    }

    fn forward<const Q: u16>() {
        let bound = vgl::output_bound(Q);
        let mut worst = 0i32;
        for mut b in inputs::<Q>(8, 0x6EA1 ^ Q as u64) {
            let coeffs: Vec<[u32; N]> = (0..32).map(|p| b.get(p).normalized(Q)).collect();
            unsafe { vgl::ntt_gen_batch32::<Q>(&mut b) };
            assert_eq!(b.representation, Representation::Ntt);
            for p in 0..32 {
                let want = scalar::ntt::<Q>(&coeffs[p]);
                for j in 0..N {
                    let got = b.v[j][p] as i32;
                    assert_eq!(
                        got.rem_euclid(Q as i32) as u32,
                        want[j],
                        "q={Q} slot {j} lane {p}"
                    );
                    worst = worst.max(got.abs());
                    assert!(
                        got.abs() <= bound,
                        "q={Q} |{got}| over the declared bound {bound}"
                    );
                }
            }
        }
        let (lm, peak) = vgl::fwd_model(Q);
        println!(
            "q={Q} forward: max |output| {worst} = {:.3} q (declared {:.3} q), \
             model {lm:?} peak {peak}",
            worst as f64 / Q as f64,
            bound as f64 / Q as f64
        );
    }

    #[test]
    fn forward_matches_scalar() {
        forward::<17497>();
        forward::<19441>();
    }

    /// The inverse is fed exactly what `recursion::limbs::columns_split` feeds it: a fully reduced
    /// centered transform.
    fn inverse<const Q: u16>() {
        let mut rng = Rng::new(0x1177 ^ Q as u64);
        let half = ((Q - 1) / 2) as i32;
        for _ in 0..6 {
            let coeffs: Vec<[u32; N]> = (0..32)
                .map(|_| core::array::from_fn(|_| rng.below(Q as u32)))
                .collect();
            let mut b = Batch32::zero(Representation::Ntt);
            for p in 0..32 {
                let s = scalar::ntt::<Q>(&coeffs[p]);
                for j in 0..N {
                    let x = s[j] as i32;
                    b.v[j][p] = if x > half {
                        (x - Q as i32) as i16
                    } else {
                        x as i16
                    };
                }
            }
            unsafe { vgl::intt_gen_batch32::<Q>(&mut b) };
            assert_eq!(b.representation, Representation::Coefficients);
            for p in 0..32 {
                for j in 0..N {
                    let got = b.v[j][p] as i32;
                    assert!(
                        got.abs() <= half,
                        "q={Q} slot {j} lane {p}: |{got}| not centered"
                    );
                    assert_eq!(
                        got.rem_euclid(Q as i32) as u32,
                        coeffs[p][j],
                        "q={Q} coefficient {j} lane {p}"
                    );
                }
            }
        }
        let (lm, peak) = vgl::inv_model(Q);
        println!("q={Q} inverse: model {lm:?} peak {peak} of 32767");
    }

    #[test]
    fn inverse_matches_scalar() {
        inverse::<17497>();
        inverse::<19441>();
    }

    #[test]
    fn round_trip() {
        fn go<const Q: u16>() {
            let mut rng = Rng::new(0x2A2A ^ Q as u64);
            let half = ((Q - 1) / 2) as i16;
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(Q as u32) as i16 - half;
                }
            }
            let want = b.clone();
            unsafe {
                vgl::ntt_gen_batch32::<Q>(&mut b);
                vgl::intt_gen_batch32::<Q>(&mut b);
            }
            assert_eq!(b.v, want.v, "q={Q}");
        }
        go::<17497>();
        go::<19441>();
    }
}

mod quad {
    //! Correctness, bound and product tests for the quadratic-slot tree (q in `QS_QUAD`): the scalar
    //! reference against the definition, all three SIMD kernels — binary, generic and the generic
    //! inverse — against the scalar reference slot for slot, the declared bounds, an i32 shadow model
    //! of the binary kernel's schedule, and the `R_162` decomposition of a transform.
    use crate::common::*;
    use labinius::f162;
    use labinius::params::*;
    use labinius::recursion::limbs;
    use labinius::rng::Rng;
    use labinius::scalar::{self, Coeffs};

    use labinius::simd::ntt::bin_quad as vq;
    use labinius::simd::ntt::gen_quad as vgq;
    use labinius::ring::*;

    // ------------------------------------------------------------------ inputs

    /// The 648 binary coefficients of one ring element.

    fn random_coeffs(rng: &mut Rng, q: u16) -> Coeffs {
        std::array::from_fn(|_| (rng.next_u64() % q as u64) as u32)
    }

    // ------------------------------------------------------------------ the tree itself

    #[test]
    fn tree_shape() {
        // the leaves are the units mod 972, each exactly once
        let mut seen = [false; 972];
        for j in 0..QUAD_SLOTS {
            let u = QUAD_SLOT_EXP[j] as usize;
            assert!(
                u % 2 == 1 && u % 3 != 0,
                "leaf {j} exponent {u} is not a unit"
            );
            assert!(!seen[u]);
            seen[u] = true;
        }
        // sub-ring invariant: X^n - psi'^e with (n/2) | e, and every split is integral
        for level in 1..=6 {
            for k in 0..SUBRINGS_Q[level] {
                let e = subring_exp_quad(level, k);
                assert_eq!(
                    e as usize % (DEGREE_Q[level] / 2),
                    0,
                    "level {level} sub-ring {k}"
                );
                if level < 6 {
                    assert_eq!(e % RADIX_Q[level] as u32, 0);
                }
            }
        }
        // the R_162 classes of the two trees agree, and each class owns a + and a - leaf
        for s in 0..162 {
            let v = QUAD_POW3_CLASS[s] as usize;
            assert_eq!(v, labinius::ring::POW3_SLOT_EXP[s] as usize);
            let (jp, jm) = (
                QUAD_CLASS_SLOT[0][s] as usize,
                QUAD_CLASS_SLOT[1][s] as usize,
            );
            assert_eq!(QUAD_SLOT_EXP[jp] as usize, v);
            assert_eq!(QUAD_SLOT_EXP[jm] as usize, v + 486);
        }
    }

    fn roots<const Q: u16>() {
        let q = Q as u64;
        let psi = ParamsQ::<Q>::PSI972 as u64;
        assert_eq!(pow_mod(psi, 972, q), 1);
        assert_ne!(pow_mod(psi, 486, q), 1);
        assert_ne!(pow_mod(psi, 324, q), 1);
        assert_eq!(pow_mod(psi, 486, q), q - 1);
        let z6 = ParamsQ::<Q>::ZETA6 as u64;
        assert_eq!((z6 * z6 + q - z6 + 1) % q, 0);
        let om = ParamsQ::<Q>::OMEGA as u64;
        assert_eq!((om * om + om + 1) % q, 0);
        // Phi_1944 = prod_j (X^2 - psi'^u_j), checked at random points
        let mut rng = Rng::new(7);
        for _ in 0..4 {
            let x = rng.next_u64() % q;
            let mut p = 1u64;
            for j in 0..QUAD_SLOTS {
                p = p * ((x * x % q + q - ParamsQ::<Q>::LEAF_C[j] as u64) % q) % q;
            }
            let want = (pow_mod(x, 648, q) + q - pow_mod(x, 324, q) + 1) % q;
            assert_eq!(p, want, "factorisation at x = {x}");
        }
    }

    #[test]
    fn roots_and_factorisation() {
        roots::<2917>();
        roots::<4861>();
        roots::<12637>();
    }

    // ------------------------------------------------------------------ scalar reference

    fn scalar_matches_definition<const Q: u16>() {
        let mut rng = Rng::new(3 + Q as u64);
        for _ in 0..4 {
            let a = random_coeffs(&mut rng, Q);
            let out = scalar::ntt_quad::<Q>(&a);
            for j in 0..QUAD_SLOTS {
                let (r0, r1) = scalar::eval_quad_at::<Q>(&a, QUAD_SLOT_EXP[j] as u32);
                assert_eq!((out[2 * j], out[2 * j + 1]), (r0, r1), "leaf {j}");
            }
        }
    }

    #[test]
    fn scalar_ntt_quad_is_the_leaf_reduction() {
        scalar_matches_definition::<2917>();
        scalar_matches_definition::<4861>();
        scalar_matches_definition::<12637>();
    }

    fn scalar_product<const Q: u16>() {
        let mut rng = Rng::new(11 + Q as u64);
        for _ in 0..3 {
            let a = random_coeffs(&mut rng, Q);
            let b = random_coeffs(&mut rng, Q);
            let want = scalar::ntt_quad::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
            let got =
                scalar::mul_quad_slots::<Q>(&scalar::ntt_quad::<Q>(&a), &scalar::ntt_quad::<Q>(&b));
            assert_eq!(want, got);
        }
    }

    #[test]
    fn scalar_product_identity() {
        scalar_product::<2917>();
        scalar_product::<4861>();
        scalar_product::<12637>();
    }

    /// `y_k(theta^v)` computed directly: component k is `sum_m y[4m+k] Y^m`, evaluated at
    /// `theta^v = psi'^{2v}` by Horner.
    fn component_eval<const Q: u16>(y: &Coeffs, k: usize, v: u32) -> u32 {
        let q = Q as u64;
        let th = pow_mod(ParamsQ::<Q>::PSI972 as u64, 2 * v as u64 % 972, q);
        let mut acc = 0u64;
        for m in (0..162).rev() {
            acc = (acc * th + y[4 * m + k] as u64) % q;
        }
        acc as u32
    }

    fn decompose<const Q: u16>() {
        let mut rng = Rng::new(23 + Q as u64);
        for _ in 0..3 {
            let a = random_coeffs(&mut rng, Q);
            let got = scalar::decompose_quad_648_to_4x162::<Q>(&scalar::ntt_quad::<Q>(&a));
            for k in 0..4 {
                for s in 0..162 {
                    assert_eq!(
                        got[k][s],
                        component_eval::<Q>(&a, k, QUAD_POW3_CLASS[s] as u32),
                        "component {k} slot {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn decomposition_matches_the_definition() {
        decompose::<2917>();
        decompose::<4861>();
        decompose::<12637>();
    }

    // ------------------------------------------------------------------ i32 shadow model
    //
    // Replays the binary kernel's exact schedule lane-wise in i32, so every intermediate can be
    // checked against the 2^15 invariant and the exact (not just mod q) output compared.

    struct Shadow<const Q: u16> {
        /// max |value| after: [0] the fused lookups + level 2, [1..=3] levels 3, 4, 5.
        max: [i32; 4],
        max_any: i32,
    }

    impl<const Q: u16> Shadow<Q> {
        fn new() -> Self {
            Shadow {
                max: [0; 4],
                max_any: 0,
            }
        }
        fn see(&mut self, x: i32) -> i16 {
            let a = x.abs();
            if a > self.max_any {
                self.max_any = a;
            }
            assert!(a < 32768, "i16 overflow: {x}");
            x as i16
        }
        fn mont(&mut self, a: i16, x: u16) -> i16 {
            let w = Params::<Q>::to_mont(x);
            let r = mont_mul_i16(a, w, Params::<Q>::mont_pre(w), Q);
            self.see(r as i32)
        }
        fn bar(&mut self, a: i16) -> i16 {
            self.see(labinius::simd::ntt::bin_asm::barrett_lut_i16(a, Q) as i32)
        }
        fn r3(&mut self, a0: i16, a1: i16, a2: i16, zeta: u16, bar: bool) -> (i16, i16, i16) {
            let q = Q as u64;
            let z2 = (zeta as u64 * zeta as u64 % q) as u16;
            let t1 = self.mont(a1, zeta);
            let t2 = self.mont(a2, z2);
            let d = self.see(t1 as i32 - t2 as i32);
            let u = self.mont(d, ParamsQ::<Q>::OMEGA);
            let a0 = if bar { self.bar(a0) } else { a0 };
            (
                self.see(a0 as i32 + t1 as i32 + t2 as i32),
                self.see(a0 as i32 - t2 as i32 + u as i32),
                self.see(a0 as i32 - t1 as i32 - u as i32),
            )
        }
        /// One polynomial through the binary kernel's schedule.
        fn run(&mut self, poly: &Bin) -> [i16; N] {
            let q = Q as u64;
            let z6 = ParamsQ::<Q>::ZETA6 as u64;
            let kappa = [z6, (1 + q - z6) % q];
            let bar = vq::bar_levels(Q);
            let mut v = [0i16; N];
            let om = ParamsQ::<Q>::OMEGA as u64;
            // levels 0 + 1 as the 16-entry table gives them, before any folded twiddle
            let base_of = |k: usize, i: usize| -> u64 {
                let (s0, s1) = (k / 2, k % 2);
                let z1 = ParamsQ::<Q>::ZETA_L1[s0] as u64;
                let n = [i, i + 162, i + 324, i + 486].map(|c| poly[c] as u64);
                let inner = z1 * ((n[1] + kappa[s0] * n[3]) % q) % q;
                let t = if s1 == 0 { inner } else { (q - inner) % q };
                ((n[0] + kappa[s0] * n[2]) % q + t) % q
            };
            if vq::fold3(Q) {
                // level 2 straight out of the tables: output s of position i is the sum of three
                // lookups carrying zeta2^r omega^{r s} zeta3^{i/18}
                for k in 0..4 {
                    let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
                    for s in 0..3 {
                        let z3 = ParamsQ::<Q>::ZETA_L3[3 * k + s] as u64;
                        for i in 0..54 {
                            let e = |r: usize| -> i32 {
                                let g = pow_mod(z2, r as u64, q) * pow_mod(om, (r * s) as u64, q) % q
                                    * pow_mod(z3, (i / 18) as u64, q)
                                    % q;
                                center(base_of(k, i + 54 * r) * g % q, q) as i32
                            };
                            let x = self.see(e(0) + e(1));
                            v[162 * k + 54 * s + i] = self.see(x as i32 + e(2));
                        }
                    }
                }
            } else {
                for k in 0..4 {
                    let z2 = ParamsQ::<Q>::ZETA_L2[k] as u64;
                    for i in 0..162 {
                        let f = pow_mod(z2, (i / 54) as u64, q);
                        v[162 * k + i] = self.see(center(base_of(k, i) * f % q, q) as i32);
                    }
                }
                // level 2: omega-only radix-3 on (i, i+54, i+108)
                for k in 0..4 {
                    for i in 0..54 {
                        let b = 162 * k + i;
                        let (t1, t2) = (v[b + 54], v[b + 108]);
                        let d = self.see(t1 as i32 - t2 as i32);
                        let u = self.mont(d, ParamsQ::<Q>::OMEGA);
                        let a0 = v[b];
                        v[b] = self.see(a0 as i32 + t1 as i32 + t2 as i32);
                        v[b + 54] = self.see(a0 as i32 - t2 as i32 + u as i32);
                        v[b + 108] = self.see(a0 as i32 - t1 as i32 - u as i32);
                    }
                }
            }
            self.level_max(0, &v);
            if vq::fold3(Q) {
                // level 3: omega-only radix-3 on (i, i+18, i+36) of each 54-block
                for base in (0..N).step_by(54) {
                    for i in 0..18 {
                        let (t1, t2) = (v[base + i + 18], v[base + i + 36]);
                        let d = self.see(t1 as i32 - t2 as i32);
                        let u = self.mont(d, ParamsQ::<Q>::OMEGA);
                        let a0 = v[base + i];
                        v[base + i] = self.see(a0 as i32 + t1 as i32 + t2 as i32);
                        v[base + i + 18] = self.see(a0 as i32 - t2 as i32 + u as i32);
                        v[base + i + 36] = self.see(a0 as i32 - t1 as i32 - u as i32);
                    }
                }
                self.level_max(1, &v);
                for (l, (blk, level)) in [(18usize, 4usize), (6, 5)].iter().enumerate() {
                    let m = blk / 3;
                    for base in (0..N).step_by(*blk) {
                        let zeta = ParamsQ::<Q>::zeta(*level, base / blk);
                        for i in 0..m {
                            let (o0, o1, o2) = self.r3(
                                v[base + i],
                                v[base + i + m],
                                v[base + i + 2 * m],
                                zeta,
                                bar[l + 1],
                            );
                            v[base + i] = o0;
                            v[base + i + m] = o1;
                            v[base + i + 2 * m] = o2;
                        }
                    }
                    self.level_max(l + 2, &v);
                }
                return v;
            }
            for (l, (blk, level)) in [(54usize, 3usize), (18, 4), (6, 5)].iter().enumerate() {
                let m = blk / 3;
                for base in (0..N).step_by(*blk) {
                    let zeta = ParamsQ::<Q>::zeta(*level, base / blk);
                    for i in 0..m {
                        let (o0, o1, o2) = self.r3(
                            v[base + i],
                            v[base + i + m],
                            v[base + i + 2 * m],
                            zeta,
                            bar[l],
                        );
                        v[base + i] = o0;
                        v[base + i + m] = o1;
                        v[base + i + 2 * m] = o2;
                    }
                }
                self.level_max(l + 1, &v);
            }
            v
        }
        fn level_max(&mut self, l: usize, v: &[i16; N]) {
            for x in v.iter() {
                let a = (*x as i32).abs();
                if a > self.max[l] {
                    self.max[l] = a;
                }
            }
        }
    }

    // ------------------------------------------------------------------ the binary kernel

    fn bin_kernel<const Q: u16>() {
        let mut sh = Shadow::<Q>::new();
        let bound = vq::output_bound(Q);
        let mut out = Batch32::zero(Representation::Ntt);
        for polys in batches_quad(24, 5 + Q as u64) {
            let idx = idx_of_polys(&polys);
            unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
            for p in 0..32 {
                let want = scalar::ntt_quad::<Q>(&polys[p]);
                let shadow = sh.run(&polys[p]);
                for j in 0..N {
                    let got = out.v[j][p];
                    assert_eq!(
                        (got as i32).rem_euclid(Q as i32) as u32,
                        want[j],
                        "q={Q} row {j} lane {p}"
                    );
                    assert_eq!(got, shadow[j], "q={Q} row {j} lane {p}: shadow model");
                    assert!(
                        (got as i32).abs() <= bound,
                        "q={Q} row {j} lane {p}: |{got}| over the declared bound {bound}"
                    );
                }
            }
        }
        let model = vq::bin_model(Q, vq::bar_levels(Q));
        assert!(sh.max_any < 32768);
        assert!(
            sh.max_any <= model.1,
            "shadow peak {} > model {}",
            sh.max_any,
            model.1
        );
        for l in 0..4 {
            assert!(
                sh.max[l] <= model.0[l],
                "level {l}: {} > {}",
                sh.max[l],
                model.0[l]
            );
        }
        println!(
            "q={Q} binary: observed per level {:?} (model {:?}), output {:.3} q of the declared {:.3} q",
            sh.max,
            model.0,
            sh.max[3] as f64 / Q as f64,
            bound as f64 / Q as f64
        );
    }

    #[test]
    fn binary_kernel_matches_scalar() {
        bin_kernel::<2917>();
        bin_kernel::<4861>();
        bin_kernel::<12637>();
    }

    /// The block sink the commitment consumes the transform through sees exactly the 648 rows the
    /// plain entry point writes, 18 at a time, in block order.
    fn bin_sink<const Q: u16>() {
        let mut rng = Rng::new(99 + Q as u64);
        let polys: [Bin; 32] = std::array::from_fn(|_| random_bin(&mut rng));
        let e = elems_of(&polys);
        let idx = idx_of_polys(&polys);
        let mut out = Batch32::zero(Representation::Ntt);
        unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut out) };
        for p in 0..32 {
            let want = scalar::ntt_quad::<Q>(&f162::lift_elem(&e, p));
            for j in 0..N {
                assert_eq!(
                    (out.v[j][p] as i32).rem_euclid(Q as i32) as u32,
                    want[j],
                    "q={Q} row {j} lane {p}"
                );
            }
        }

        #[repr(C, align(64))]
        struct Blk18([i16; 18 * 32]);
        struct Collect {
            buf: Vec<Blk18>,
            seen: Vec<usize>,
        }
        impl vq::BlockSink for Collect {
            unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
                self.buf[blk].0.as_mut_ptr()
            }
            unsafe fn block(&mut self, blk: usize, _dst: *const i16) {
                self.seen.push(blk);
            }
        }
        let mut c = Collect {
            buf: (0..36).map(|_| Blk18([0i16; 18 * 32])).collect(),
            seen: Vec::new(),
        };
        unsafe { vq::ntt_quad_bin_batch32_sink::<Q, _>(&idx, &mut c) };
        assert_eq!(c.seen, (0..36).collect::<Vec<_>>());
        for blk in 0..36 {
            for r in 0..18 {
                for p in 0..32 {
                    assert_eq!(
                        c.buf[blk].0[32 * r + p],
                        out.v[18 * blk + r][p],
                        "block {blk} row {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn binary_sink_matches_the_plain_kernel() {
        bin_sink::<2917>();
        bin_sink::<4861>();
        bin_sink::<12637>();
    }

    // ------------------------------------------------------------------ the generic kernel

    fn gen_inputs<const Q: u16>(count: usize, seed: u64) -> Vec<Batch32> {
        let mut rng = Rng::new(seed);
        let q = Q as i16;
        let mut out = Vec::new();
        // adversarial: every lane at +-q, alternating, the binary patterns, then random in [-q, q]
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = if (j + p) % 2 == 0 { q } else { -q };
            }
        }
        out.push(b);
        let mut b = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                b.v[j][p] = if p % 3 == 0 {
                    q
                } else if p % 3 == 1 {
                    -q
                } else {
                    0
                };
            }
        }
        out.push(b);
        for polys in batches_quad(0, seed).into_iter().take(2) {
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = polys[p][j] as i16;
                }
            }
            out.push(b);
        }
        for _ in 0..count {
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = (rng.next_u64() % (2 * Q as u64 + 1)) as i16 - q;
                }
            }
            out.push(b);
        }
        out
    }

    fn gen_kernel<const Q: u16>() {
        let bound = vgq::output_bound(Q);
        let mut peak = 0i32;
        for b in gen_inputs::<Q>(12, 17 + Q as u64) {
            let mut got = b.clone();
            unsafe { vgq::ntt_quad_gen_batch32::<Q>(&mut got) };
            for p in 0..32 {
                let a: Coeffs = std::array::from_fn(|j| (b.v[j][p] as i32).rem_euclid(Q as i32) as u32);
                let want = scalar::ntt_quad::<Q>(&a);
                for j in 0..N {
                    let g = got.v[j][p];
                    assert_eq!(
                        (g as i32).rem_euclid(Q as i32) as u32,
                        want[j],
                        "generic q={Q} row {j} lane {p}"
                    );
                    assert!(
                        (g as i32).abs() <= bound,
                        "generic q={Q} row {j} lane {p}: |{g}| over the declared bound {bound}"
                    );
                    peak = peak.max((g as i32).abs());
                }
            }
        }
        println!(
            "q={Q} generic: output {:.3} q observed of the declared {:.3} q",
            peak as f64 / Q as f64,
            bound as f64 / Q as f64
        );
    }

    #[test]
    fn generic_kernel_matches_scalar() {
        gen_kernel::<2917>();
        gen_kernel::<4861>();
        gen_kernel::<12637>();
    }

    // ------------------------------------------------------------------ products through the SIMD

    /// `ntt_quad(a * b mod Phi) == mul_quad_slots(NTT(a), NTT(b))` with both transforms produced by
    /// the SIMD kernels (the binary one for `a`, the generic one for `b`).
    fn simd_product<const Q: u16>() {
        let mut rng = Rng::new(41 + Q as u64);
        let polys: [Bin; 32] = std::array::from_fn(|_| random_bin(&mut rng));
        let idx = idx_of_polys(&polys);
        let mut ta = Batch32::zero(Representation::Ntt);
        unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx, &mut ta) };

        let mut bb = Batch32::zero(Representation::Coefficients);
        for j in 0..N {
            for p in 0..32 {
                bb.v[j][p] = (rng.next_u64() % Q as u64) as i16;
            }
        }
        let coeffs_b = bb.clone();
        let mut tb = bb;
        unsafe { vgq::ntt_quad_gen_batch32::<Q>(&mut tb) };

        for p in [0usize, 1, 17, 31] {
            let a: Coeffs = polys[p];
            let b: Coeffs = std::array::from_fn(|j| coeffs_b.v[j][p] as u32);
            let na: Coeffs = std::array::from_fn(|j| (ta.v[j][p] as i32).rem_euclid(Q as i32) as u32);
            let nb: Coeffs = std::array::from_fn(|j| (tb.v[j][p] as i32).rem_euclid(Q as i32) as u32);
            let want = scalar::ntt_quad::<Q>(&scalar::mul_mod_phi(&a, &b, Q));
            assert_eq!(
                want,
                scalar::mul_quad_slots::<Q>(&na, &nb),
                "product q={Q} lane {p}"
            );
        }
    }

    #[test]
    fn product_through_the_simd_outputs() {
        simd_product::<2917>();
        simd_product::<4861>();
        simd_product::<12637>();
    }

    // ------------------------------------------------------------------ the vectorised inverse

    /// A fully reduced transform written into a batch, centered — what the fold hands the inverse.
    fn centered_slots<const Q: u16>(slots: &[Coeffs]) -> Batch32 {
        let half = ((Q - 1) / 2) as i32;
        let mut b = Batch32::zero(Representation::Ntt);
        for (p, s) in slots.iter().enumerate() {
            for j in 0..N {
                let x = s[j] as i32;
                b.v[j][p] = if x > half {
                    (x - Q as i32) as i16
                } else {
                    x as i16
                };
            }
        }
        b
    }

    /// `intt_quad_gen_batch32` against the scalar [`limbs::intt_quad`], slot for slot, on random
    /// transforms fed in the centered form the fold produces.
    fn inverse_kernel<const Q: u16>() {
        let mut rng = Rng::new(0x11D5 ^ Q as u64);
        let half = ((Q - 1) / 2) as i32;
        for _ in 0..6 {
            let coeffs: Vec<Coeffs> = (0..32).map(|_| random_coeffs(&mut rng, Q)).collect();
            let slots: Vec<Coeffs> = coeffs.iter().map(scalar::ntt_quad::<Q>).collect();
            let mut b = centered_slots::<Q>(&slots);
            unsafe { vgq::intt_quad_gen_batch32::<Q>(&mut b) };
            assert_eq!(b.representation, Representation::Coefficients);
            for p in 0..32 {
                let want = limbs::intt_quad::<Q>(&slots[p]);
                assert_eq!(
                    want, coeffs[p],
                    "q={Q} the scalar inverse is not the inverse"
                );
                for j in 0..N {
                    let got = b.v[j][p] as i32;
                    assert!(
                        got.abs() <= half,
                        "q={Q} coefficient {j} lane {p}: |{got}| not centered"
                    );
                    assert_eq!(
                        got.rem_euclid(Q as i32) as u32,
                        want[j],
                        "q={Q} coefficient {j} lane {p}"
                    );
                }
            }
        }
        println!(
            "q={Q} inverse: reductions {:?}, per-level bound {:?}, input {:.3} q",
            vgq::inv_flags(Q),
            vgq::inv_bound(Q),
            vgq::in_bound(Q) as f64 / Q as f64
        );
    }

    #[test]
    fn vectorised_inverse_matches_the_scalar_one() {
        inverse_kernel::<2917>();
        inverse_kernel::<4861>();
        inverse_kernel::<12637>();
    }

    /// The same on a lazily reduced transform: the binary kernel's output, inverted straight back to
    /// the bits.
    fn inverse_at_the_declared_bound<const Q: u16>() {
        let bound = vgq::in_bound(Q);
        let half = ((Q - 1) / 2) as i32;
        for polys in batches_quad(3, 0x5EED ^ Q as u64) {
            let mut b = Batch32::zero(Representation::Ntt);
            unsafe { vq::ntt_quad_bin_batch32::<Q>(&idx_of_polys(&polys), &mut b) };
            for j in 0..N {
                for p in 0..32 {
                    let x = b.v[j][p] as i32;
                    assert!(
                        x.abs() <= bound,
                        "q={Q} row {j} lane {p}: |{x}| over the input bound"
                    );
                }
            }
            unsafe { vgq::intt_quad_gen_batch32::<Q>(&mut b) };
            for j in 0..N {
                for p in 0..32 {
                    let got = b.v[j][p] as i32;
                    assert!(
                        got.abs() <= half,
                        "q={Q} coefficient {j} lane {p} not centered"
                    );
                    assert_eq!(got, polys[p][j] as i32, "q={Q} coefficient {j} lane {p}");
                }
            }
        }
    }

    #[test]
    fn the_inverse_takes_a_lazily_reduced_transform() {
        inverse_at_the_declared_bound::<2917>();
        inverse_at_the_declared_bound::<4861>();
        inverse_at_the_declared_bound::<12637>();
    }

    #[test]
    fn generic_round_trip() {
        fn go<const Q: u16>() {
            let mut rng = Rng::new(0x3C3C ^ Q as u64);
            let half = ((Q - 1) / 2) as i16;
            let mut b = Batch32::zero(Representation::Coefficients);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(Q as u32) as i16 - half;
                }
            }
            let want = b.clone();
            unsafe {
                vgq::ntt_quad_gen_batch32::<Q>(&mut b);
                vgq::intt_quad_gen_batch32::<Q>(&mut b);
            }
            assert_eq!(b.v, want.v, "q={Q}");
        }
        go::<2917>();
        go::<4861>();
        go::<12637>();
    }

    /// A folding challenge enters `R_648` as `c(-X^4)`, a polynomial in `X^4`; modulo the leaf
    /// `X^2 - psi'^u` that is `X^4 = psi'^{2u}`, a scalar, so the transform's odd rows vanish. This is
    /// what lets the fold multiply a quadratic-slot base row by row instead of leaf by leaf.
    fn subring_element_is_a_leaf_scalar<const Q: u16>() {
        let mut rng = Rng::new(0x4A4A ^ Q as u64);
        for _ in 0..4 {
            let mut a = [0u32; N];
            for m in 0..162 {
                a[4 * m] = rng.below(Q as u32);
            }
            let y = scalar::ntt_quad::<Q>(&a);
            for j in 0..QUAD_SLOTS {
                assert_eq!(y[2 * j + 1], 0, "q={Q} leaf {j} has an X coefficient");
            }
        }
    }

    #[test]
    fn an_embedded_subring_element_has_no_x_coefficient() {
        subring_element_is_a_leaf_scalar::<2917>();
        subring_element_is_a_leaf_scalar::<4861>();
        subring_element_is_a_leaf_scalar::<12637>();
    }
}

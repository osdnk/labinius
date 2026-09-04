use crate::api::N162;
use crate::challenge::{ShortChallenge, MAX_WEIGHT};
use core::arch::x86_64::*;

const PAD64: usize = 168;
const PAD32: usize = 176;
const BUF: usize = 512;
const MAX_STEPS: usize = 64;

pub struct Sparse {
    pos: [u8; MAX_WEIGHT],
    bar: [u8; MAX_WEIGHT],
    signs: u32,
    weight: usize,
}

impl Sparse {
    pub fn of(c: &ShortChallenge) -> Self {
        let mut bar = [0u8; MAX_WEIGHT];
        for i in 0..c.weight {
            bar[i] = ((243 - c.positions[i] as usize) % 243) as u8;
        }
        Sparse {
            pos: c.positions,
            bar,
            signs: c.signs,
            weight: c.weight,
        }
    }
}

#[repr(align(64))]
#[derive(Clone, Copy)]
struct V64([f64; PAD64]);

#[repr(align(64))]
struct B64([f64; BUF]);

#[repr(align(64))]
#[derive(Clone, Copy)]
struct V32([f32; PAD32]);

#[repr(align(64))]
struct B32([f32; BUF]);

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mul64(s: &Sparse, bar: bool, x: &V64, buf: &mut B64, y: &mut V64) {
    let b = buf.0.as_mut_ptr();
    let z = _mm512_setzero_pd();
    let (zn, fn64) = if bar { (52, 21) } else { (42, 11) };
    for j in 0..zn {
        _mm512_store_pd(b.add(8 * j), z);
    }
    let xp = x.0.as_ptr();
    let src = if bar { &s.bar } else { &s.pos };
    for i in 0..s.weight {
        let d = b.add(src[i] as usize);
        if (s.signs >> i) & 1 == 0 {
            for j in 0..PAD64 / 8 {
                let a = _mm512_loadu_pd(d.add(8 * j));
                let v = _mm512_load_pd(xp.add(8 * j));
                _mm512_storeu_pd(d.add(8 * j), _mm512_add_pd(a, v));
            }
        } else {
            for j in 0..PAD64 / 8 {
                let a = _mm512_loadu_pd(d.add(8 * j));
                let v = _mm512_load_pd(xp.add(8 * j));
                _mm512_storeu_pd(d.add(8 * j), _mm512_sub_pd(a, v));
            }
        }
    }
    for j in 0..fn64 {
        let a = _mm512_load_pd(b.add(8 * j));
        let c = _mm512_loadu_pd(b.add(243 + 8 * j));
        _mm512_store_pd(b.add(8 * j), _mm512_add_pd(a, c));
    }
    let yp = y.0.as_mut_ptr();
    for j in 0..10 {
        let t = _mm512_loadu_pd(b.add(162 + 8 * j));
        let lo = _mm512_load_pd(b.add(8 * j));
        let mid = _mm512_loadu_pd(b.add(81 + 8 * j));
        _mm512_store_pd(yp.add(8 * j), _mm512_sub_pd(lo, t));
        _mm512_storeu_pd(yp.add(81 + 8 * j), _mm512_sub_pd(mid, t));
    }
    let m: __mmask8 = 0x01;
    let t = _mm512_maskz_loadu_pd(m, b.add(242));
    let lo = _mm512_maskz_loadu_pd(m, b.add(80));
    let mid = _mm512_maskz_loadu_pd(m, b.add(161));
    _mm512_mask_storeu_pd(yp.add(80), m, _mm512_sub_pd(lo, t));
    _mm512_mask_storeu_pd(yp.add(161), m, _mm512_sub_pd(mid, t));
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mix64(y: &mut V64, a: f64, b: f64, c: f64, d: f64, div: f64) {
    let p = y.0.as_mut_ptr();
    let (va, vb, vc, vd) = (
        _mm512_set1_pd(a),
        _mm512_set1_pd(b),
        _mm512_set1_pd(c),
        _mm512_set1_pd(d),
    );
    let vq = _mm512_set1_pd(div);
    for j in 0..11 {
        let m: __mmask8 = if j < 10 { 0xff } else { 0x01 };
        let lo = _mm512_maskz_loadu_pd(m, p.add(8 * j));
        let hi = _mm512_maskz_loadu_pd(m, p.add(81 + 8 * j));
        let nl = _mm512_add_pd(_mm512_mul_pd(va, lo), _mm512_mul_pd(vb, hi));
        let nh = _mm512_add_pd(_mm512_mul_pd(vc, lo), _mm512_mul_pd(vd, hi));
        _mm512_mask_storeu_pd(p.add(8 * j), m, _mm512_div_pd(nl, vq));
        _mm512_mask_storeu_pd(p.add(81 + 8 * j), m, _mm512_div_pd(nh, vq));
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn apply64(s: &Sparse, x: &V64, buf: &mut B64, t: &mut V64, out: &mut V64) {
    mul64(s, false, x, buf, t);
    mix64(t, 2.0, 1.0, 1.0, 2.0, 1.0);
    mul64(s, true, t, buf, out);
    mix64(out, 2.0, -1.0, -1.0, 2.0, 3.0);
}

#[inline(always)]
unsafe fn hsum64(v: __m512d) -> f64 {
    let mut l = [0.0f64; 8];
    _mm512_storeu_pd(l.as_mut_ptr(), v);
    ((((((l[0] + l[1]) + l[2]) + l[3]) + l[4]) + l[5]) + l[6]) + l[7]
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot64(a: &V64, b: &V64) -> f64 {
    let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
    let mut acc = _mm512_setzero_pd();
    for j in 0..PAD64 / 8 {
        acc = _mm512_fmadd_pd(
            _mm512_load_pd(pa.add(8 * j)),
            _mm512_load_pd(pb.add(8 * j)),
            acc,
        );
    }
    hsum64(acc)
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn axpy64(v: &mut V64, s: f64, q: &V64) {
    let (pv, pq) = (v.0.as_mut_ptr(), q.0.as_ptr());
    let vs = _mm512_set1_pd(s);
    for j in 0..PAD64 / 8 {
        let r = _mm512_fmadd_pd(
            vs,
            _mm512_load_pd(pq.add(8 * j)),
            _mm512_load_pd(pv.add(8 * j)),
        );
        _mm512_store_pd(pv.add(8 * j), r);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn scale64(q: &mut V64, v: &V64, d: f64) {
    let (pq, pv) = (q.0.as_mut_ptr(), v.0.as_ptr());
    let vd = _mm512_set1_pd(d);
    for j in 0..PAD64 / 8 {
        _mm512_store_pd(
            pq.add(8 * j),
            _mm512_div_pd(_mm512_load_pd(pv.add(8 * j)), vd),
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mul32(s: &Sparse, bar: bool, x: &V32, buf: &mut B32, y: &mut V32) {
    let b = buf.0.as_mut_ptr();
    let z = _mm512_setzero_ps();
    let (zn, fn32) = if bar { (27, 11) } else { (23, 7) };
    for j in 0..zn {
        _mm512_store_ps(b.add(16 * j), z);
    }
    let xp = x.0.as_ptr();
    let src = if bar { &s.bar } else { &s.pos };
    for i in 0..s.weight {
        let d = b.add(src[i] as usize);
        if (s.signs >> i) & 1 == 0 {
            for j in 0..PAD32 / 16 {
                let a = _mm512_loadu_ps(d.add(16 * j));
                let v = _mm512_load_ps(xp.add(16 * j));
                _mm512_storeu_ps(d.add(16 * j), _mm512_add_ps(a, v));
            }
        } else {
            for j in 0..PAD32 / 16 {
                let a = _mm512_loadu_ps(d.add(16 * j));
                let v = _mm512_load_ps(xp.add(16 * j));
                _mm512_storeu_ps(d.add(16 * j), _mm512_sub_ps(a, v));
            }
        }
    }
    for j in 0..fn32 {
        let a = _mm512_load_ps(b.add(16 * j));
        let c = _mm512_loadu_ps(b.add(243 + 16 * j));
        _mm512_store_ps(b.add(16 * j), _mm512_add_ps(a, c));
    }
    let yp = y.0.as_mut_ptr();
    for j in 0..5 {
        let t = _mm512_loadu_ps(b.add(162 + 16 * j));
        let lo = _mm512_load_ps(b.add(16 * j));
        let mid = _mm512_loadu_ps(b.add(81 + 16 * j));
        _mm512_store_ps(yp.add(16 * j), _mm512_sub_ps(lo, t));
        _mm512_storeu_ps(yp.add(81 + 16 * j), _mm512_sub_ps(mid, t));
    }
    let m: __mmask16 = 0x0001;
    let t = _mm512_maskz_loadu_ps(m, b.add(242));
    let lo = _mm512_maskz_loadu_ps(m, b.add(80));
    let mid = _mm512_maskz_loadu_ps(m, b.add(161));
    _mm512_mask_storeu_ps(yp.add(80), m, _mm512_sub_ps(lo, t));
    _mm512_mask_storeu_ps(yp.add(161), m, _mm512_sub_ps(mid, t));
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mix32(y: &mut V32, a: f32, b: f32, c: f32, d: f32, div: f32) {
    let p = y.0.as_mut_ptr();
    let (va, vb, vc, vd) = (
        _mm512_set1_ps(a),
        _mm512_set1_ps(b),
        _mm512_set1_ps(c),
        _mm512_set1_ps(d),
    );
    let vq = _mm512_set1_ps(div);
    for j in 0..6 {
        let m: __mmask16 = if j < 5 { 0xffff } else { 0x0001 };
        let lo = _mm512_maskz_loadu_ps(m, p.add(16 * j));
        let hi = _mm512_maskz_loadu_ps(m, p.add(81 + 16 * j));
        let nl = _mm512_add_ps(_mm512_mul_ps(va, lo), _mm512_mul_ps(vb, hi));
        let nh = _mm512_add_ps(_mm512_mul_ps(vc, lo), _mm512_mul_ps(vd, hi));
        _mm512_mask_storeu_ps(p.add(16 * j), m, _mm512_div_ps(nl, vq));
        _mm512_mask_storeu_ps(p.add(81 + 16 * j), m, _mm512_div_ps(nh, vq));
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn apply32(s: &Sparse, x: &V32, buf: &mut B32, t: &mut V32, out: &mut V32) {
    mul32(s, false, x, buf, t);
    mix32(t, 2.0, 1.0, 1.0, 2.0, 1.0);
    mul32(s, true, t, buf, out);
    mix32(out, 2.0, -1.0, -1.0, 2.0, 3.0);
}

#[inline(always)]
unsafe fn hsum32(v: __m512) -> f64 {
    let mut l = [0.0f32; 16];
    _mm512_storeu_ps(l.as_mut_ptr(), v);
    let mut s = 0.0f64;
    for k in 0..16 {
        s += l[k] as f64;
    }
    s
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot32(a: &V32, b: &V32) -> f64 {
    let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
    let mut acc = _mm512_setzero_ps();
    for j in 0..PAD32 / 16 {
        acc = _mm512_fmadd_ps(
            _mm512_load_ps(pa.add(16 * j)),
            _mm512_load_ps(pb.add(16 * j)),
            acc,
        );
    }
    hsum32(acc)
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn axpy32(v: &mut V32, s: f32, q: &V32) {
    let (pv, pq) = (v.0.as_mut_ptr(), q.0.as_ptr());
    let vs = _mm512_set1_ps(s);
    for j in 0..PAD32 / 16 {
        let r = _mm512_fmadd_ps(
            vs,
            _mm512_load_ps(pq.add(16 * j)),
            _mm512_load_ps(pv.add(16 * j)),
        );
        _mm512_store_ps(pv.add(16 * j), r);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn scale32(q: &mut V32, v: &V32, d: f32) {
    let (pq, pv) = (q.0.as_mut_ptr(), v.0.as_ptr());
    let vd = _mm512_set1_ps(d);
    for j in 0..PAD32 / 16 {
        _mm512_store_ps(
            pq.add(16 * j),
            _mm512_div_ps(_mm512_load_ps(pv.add(16 * j)), vd),
        );
    }
}

fn tri_above(alpha: &[f64; MAX_STEPS], beta: &[f64; MAX_STEPS], k: usize, x: f64) -> bool {
    let mut d = x - alpha[0];
    if !(d > 0.0) {
        return false;
    }
    for i in 1..k {
        d = (x - alpha[i]) - beta[i - 1] * beta[i - 1] / d;
        if !(d > 0.0) {
            return false;
        }
    }
    true
}

fn tri_eval(
    alpha: &[f64; MAX_STEPS],
    beta: &[f64; MAX_STEPS],
    k: usize,
    x: f64,
) -> Option<(f64, f64)> {
    let d0 = x - alpha[0];
    if !(d0 > 0.0) {
        return None;
    }
    let mut inv = 1.0 / d0;
    let mut e = 1.0f64;
    let mut f = 0.0f64;
    let mut u = inv;
    let mut g = u;
    let mut l2 = -u * u;
    for i in 1..k {
        let bb = beta[i - 1] * beta[i - 1];
        let inv2 = inv * inv;
        let dn = (x - alpha[i]) - bb * inv;
        if !(dn > 0.0) {
            return None;
        }
        let en = 1.0 + bb * e * inv2;
        let fx = bb * inv2 * (f - 2.0 * e * e * inv);
        e = en;
        f = fx;
        inv = 1.0 / dn;
        u = e * inv;
        g += u;
        l2 += f * inv - u * u;
    }
    Some((g, -l2))
}

fn tri_lambda_max(
    alpha: &[f64; MAX_STEPS],
    beta: &[f64; MAX_STEPS],
    k: usize,
    hint: f64,
    inc: f64,
) -> f64 {
    if k == 1 {
        return alpha[0];
    }
    let mut x = 0.0f64;
    let mut gh = None;
    if hint > 0.0 {
        let mut d = 4.0 * inc + 1e-9 * hint + 1e-12;
        for _ in 0..2 {
            let t = hint + d;
            gh = tri_eval(alpha, beta, k, t);
            if gh.is_some() {
                x = t;
                break;
            }
            d *= 4096.0;
        }
    }
    if gh.is_none() {
        let mut hi = f64::NEG_INFINITY;
        for i in 0..k {
            let l = if i > 0 { beta[i - 1].abs() } else { 0.0 };
            let r = if i + 1 < k { beta[i].abs() } else { 0.0 };
            let g = alpha[i] + l + r;
            if g > hi {
                hi = g;
            }
        }
        x = hi + 1e-9 * hi.abs() + 1e-12;
        gh = tri_eval(alpha, beta, k, x);
        if gh.is_none() {
            return x;
        }
    }
    let n = k as f64;
    for _ in 0..80 {
        let (g, h) = match gh {
            Some(v) => v,
            None => return x,
        };
        let disc = (n - 1.0) * (n * h - g * g);
        let sq = if disc > 0.0 { disc.sqrt() } else { 0.0 };
        let denom = g + sq;
        if !(denom > 0.0) {
            return x;
        }
        let step = n / denom;
        let xn = x - step;
        if !(xn < x) || !xn.is_finite() {
            return x;
        }
        if step <= 1e-15 * xn.abs() {
            return xn;
        }
        gh = tri_eval(alpha, beta, k, xn);
        if gh.is_none() {
            return xn;
        }
        x = xn;
    }
    x
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn decide_f64(c: &ShortChallenge, bound_sq: f64) -> (bool, f64) {
    let s = Sparse::of(c);
    let mut buf = B64([0.0; BUF]);
    let mut tmp = V64([0.0; PAD64]);
    let mut v = V64([0.0; PAD64]);
    let mut q = V64([0.0; PAD64]);
    let mut qp = V64([0.0; PAD64]);
    let e = 1.0 / (N162 as f64).sqrt();
    for i in 0..81 {
        q.0[i] = e;
        q.0[81 + i] = -e;
    }
    let mut alpha = [0.0f64; MAX_STEPS];
    let mut beta = [0.0f64; MAX_STEPS];
    let mut bc = 0.0f64;
    let mut theta_prev = 0.0f64;
    let mut inc = 0.0f64;
    let tol = bound_sq + 1e-12;
    for k in 0..MAX_STEPS {
        apply64(&s, &q, &mut buf, &mut tmp, &mut v);
        if k > 0 {
            axpy64(&mut v, -bc, &qp);
        }
        let a = dot64(&q, &v);
        axpy64(&mut v, -a, &q);
        alpha[k] = a;
        let theta = tri_lambda_max(&alpha, &beta, k + 1, theta_prev, inc);
        if theta > tol && !tri_above(&alpha, &beta, k + 1, tol) {
            return (false, theta);
        }
        if k >= 2 && (theta - theta_prev).abs() <= 1e-12 * theta {
            return (tri_above(&alpha, &beta, k + 1, tol), theta);
        }
        inc = theta - theta_prev;
        theta_prev = theta;
        let b = dot64(&v, &v).sqrt();
        if b < 1e-14 {
            return (tri_above(&alpha, &beta, k + 1, tol), theta);
        }
        beta[k] = b;
        bc = b;
        qp = q;
        scale64(&mut q, &v, b);
    }
    (tri_above(&alpha, &beta, MAX_STEPS, tol), theta_prev)
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn decide_f32(c: &ShortChallenge, bound_sq: f64) -> (bool, f64) {
    let s = Sparse::of(c);
    let mut buf = B32([0.0; BUF]);
    let mut tmp = V32([0.0; PAD32]);
    let mut v = V32([0.0; PAD32]);
    let mut q = V32([0.0; PAD32]);
    let mut qp = V32([0.0; PAD32]);
    let e = (1.0 / (N162 as f64).sqrt()) as f32;
    for i in 0..81 {
        q.0[i] = e;
        q.0[81 + i] = -e;
    }
    let mut alpha = [0.0f64; MAX_STEPS];
    let mut beta = [0.0f64; MAX_STEPS];
    let mut bc = 0.0f64;
    let mut theta_prev = 0.0f64;
    let mut inc = 0.0f64;
    for k in 0..MAX_STEPS {
        apply32(&s, &q, &mut buf, &mut tmp, &mut v);
        if k > 0 {
            axpy32(&mut v, -bc as f32, &qp);
        }
        let a = dot32(&q, &v);
        axpy32(&mut v, -a as f32, &q);
        alpha[k] = a;
        let theta = tri_lambda_max(&alpha, &beta, k + 1, theta_prev, inc);
        if theta > bound_sq && !tri_above(&alpha, &beta, k + 1, bound_sq) {
            return (false, theta);
        }
        if k >= 2 && (theta - theta_prev).abs() <= 1e-6 * theta {
            return (tri_above(&alpha, &beta, k + 1, bound_sq), theta);
        }
        inc = theta - theta_prev;
        theta_prev = theta;
        let b = dot32(&v, &v).sqrt();
        if b < 1e-6 {
            return (tri_above(&alpha, &beta, k + 1, bound_sq), theta);
        }
        beta[k] = b;
        bc = b;
        qp = q;
        scale32(&mut q, &v, b as f32);
    }
    (tri_above(&alpha, &beta, MAX_STEPS, bound_sq), theta_prev)
}

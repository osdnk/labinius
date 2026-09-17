use super::f162;
use super::scalar::F162;
use std::arch::x86_64::*;

#[derive(Clone)]
pub struct Poly {
    pub w: [Vec<u64>; 3],
    pub n: usize,
}

impl Poly {
    pub fn zero(n: usize) -> Self {
        Self {
            w: [vec![0u64; n], vec![0u64; n], vec![0u64; n]],
            n,
        }
    }

    pub fn from_scalars(v: &[F162]) -> Self {
        let n = v.len();
        let mut w = [vec![0u64; n], vec![0u64; n], vec![0u64; n]];
        for (i, x) in v.iter().enumerate() {
            for k in 0..3 {
                w[k][i] = x.0[k];
            }
        }
        Self { w, n }
    }

    pub fn get(&self, i: usize) -> F162 {
        F162([self.w[0][i], self.w[1][i], self.w[2][i]])
    }

    fn set(&mut self, i: usize, x: F162) {
        for k in 0..3 {
            self.w[k][i] = x.0[k];
        }
    }
}

#[inline(always)]
unsafe fn load(p: &Poly, i: usize) -> [__m512i; 3] {
    [
        _mm512_loadu_si512(p.w[0].as_ptr().add(i) as *const __m512i),
        _mm512_loadu_si512(p.w[1].as_ptr().add(i) as *const __m512i),
        _mm512_loadu_si512(p.w[2].as_ptr().add(i) as *const __m512i),
    ]
}

#[inline(always)]
unsafe fn store(p: &mut Poly, i: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_storeu_si512(p.w[k].as_mut_ptr().add(i) as *mut __m512i, v[k]);
    }
}

#[inline(always)]
unsafe fn xor3(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    [
        _mm512_xor_si512(a[0], b[0]),
        _mm512_xor_si512(a[1], b[1]),
        _mm512_xor_si512(a[2], b[2]),
    ]
}

#[inline(always)]
unsafe fn bcast(x: F162) -> [__m512i; 3] {
    [
        _mm512_set1_epi64(x.0[0] as i64),
        _mm512_set1_epi64(x.0[1] as i64),
        _mm512_set1_epi64(x.0[2] as i64),
    ]
}

unsafe fn horiz(acc: [__m512i; 12]) -> F162 {
    let r = f162::reduce_soa8(acc);
    let w: [[u64; 8]; 3] = [
        std::mem::transmute(r[0]),
        std::mem::transmute(r[1]),
        std::mem::transmute(r[2]),
    ];
    let mut out = F162::ZERO;
    for i in 0..8 {
        out += F162([w[0][i], w[1][i], w[2][i]]);
    }
    out
}

pub(super) fn msg(a: &Poly, p: &Poly, half: usize) -> [F162; 2] {
    let mut e0 = F162::ZERO;
    let mut einf = F162::ZERO;
    let blocks = half / 8;
    unsafe {
        let mut acc0 = [_mm512_setzero_si512(); 12];
        let mut acci = [_mm512_setzero_si512(); 12];
        for b in 0..blocks {
            let j = b * 8;
            let a0 = load(a, j);
            let a1 = load(a, j + half);
            let p0 = load(p, j);
            let p1 = load(p, j + half);
            f162::mac_soa8(&mut acc0, a0, p0);
            f162::mac_soa8(&mut acci, xor3(a0, a1), xor3(p0, p1));
        }
        if blocks > 0 {
            e0 = horiz(acc0);
            einf = horiz(acci);
        }
    }
    for j in blocks * 8..half {
        let (a0, a1) = (a.get(j), a.get(j + half));
        let (p0, p1) = (p.get(j), p.get(j + half));
        e0 += a0 * p0;
        einf += (a0 + a1) * (p0 + p1);
    }
    [e0, einf]
}

pub(super) fn fold(a: &mut Poly, p: &mut Poly, half: usize, r: F162) {
    let blocks = half / 8;
    unsafe {
        let rb = bcast(r);
        for b in 0..blocks {
            let j = b * 8;
            let a0 = load(a, j);
            let a1 = load(a, j + half);
            let p0 = load(p, j);
            let p1 = load(p, j + half);
            store(a, j, xor3(a0, f162::mul_soa8(rb, xor3(a0, a1))));
            store(p, j, xor3(p0, f162::mul_soa8(rb, xor3(p0, p1))));
        }
    }
    for j in blocks * 8..half {
        let (a0, a1) = (a.get(j), a.get(j + half));
        let (p0, p1) = (p.get(j), p.get(j + half));
        a.set(j, a0 + r * (a0 + a1));
        p.set(j, p0 + r * (p0 + p1));
    }
}

pub(super) fn round(a: &mut Poly, p: &mut Poly, half: usize, r: F162) -> [F162; 2] {
    let m = msg(a, p, half);
    fold(a, p, half, r);
    m
}

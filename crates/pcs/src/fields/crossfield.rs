use super::bitmat::{transpose_tile, untranspose_planes, GF_IDENT};
use super::scalar::{B128, F162};
use crate::simd::transpose_f162::transpose8x8_q;
use std::arch::x86_64::*;

pub const LOG_PACK: usize = 7;
pub const PACK: usize = 1 << LOG_PACK;

#[derive(Clone)]
struct Tensor {
    pub c: Vec<F162>,
}

impl Tensor {
    fn from_vertical(a: F162) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[0] = a;
        Self { c }
    }

    fn scale_vertical(&self, a: F162) -> Self {
        Self {
            c: self.c.iter().map(|&x| x * a).collect(),
        }
    }

    fn shift_x(&self) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[1..].copy_from_slice(&self.c[..127]);
        let top = self.c[127];
        for k in [0usize, 1, 2, 7] {
            c[k] += top;
        }
        Self { c }
    }

    fn scale_horizontal(&self, h: B128) -> Self {
        let mut acc = Self {
            c: vec![F162::ZERO; 128],
        };
        for k in (0..128).rev() {
            acc = acc.shift_x();
            if h.bit(k) {
                for i in 0..128 {
                    acc.c[i] += self.c[i];
                }
            }
        }
        acc
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            c: self.c.iter().zip(&o.c).map(|(&a, &b)| a + b).collect(),
        }
    }

    fn fold_vertical(&self, batch: &[F162]) -> F162 {
        self.c
            .iter()
            .zip(batch)
            .fold(F162::ZERO, |a, (&x, &y)| a + x * y)
    }
}

pub fn eq_expand_f162(r: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &ri in r {
        let mut n = Vec::with_capacity(t.len() * 2);
        for &x in &t {
            let hi = x * ri;
            n.push(x + hi);
            n.push(hi);
        }
        t = n;
    }
    t
}

pub fn eq_expand_b128(r: &[B128]) -> Vec<B128> {
    let mut t = vec![B128::ZERO; 1 << r.len()];
    eq_expand_b128_into(r, &mut t);
    t
}

/// The expansion in place, doubling from the back so no entry is read after it is written.
pub fn eq_expand_b128_into(r: &[B128], t: &mut [B128]) {
    assert_eq!(t.len(), 1 << r.len());
    t[0] = B128::ONE;
    for (round, &ri) in r.iter().enumerate() {
        for i in (0..1 << round).rev() {
            let hi = t[i] * ri;
            t[2 * i + 1] = hi;
            t[2 * i] = t[i] + hi;
        }
    }
}

pub fn transparent_coeff(r_hi: &[B128], r_pp: &[F162], batch: &[F162]) -> F162 {
    assert_eq!(r_hi.len(), r_pp.len());
    let mut t = Tensor::from_vertical(F162::ONE);
    for (&v, &h) in r_pp.iter().zip(r_hi) {
        let vs = t.scale_vertical(v);
        let hs = t.scale_horizontal(h);
        t = t.add(&vs).add(&hs);
    }
    t.fold_vertical(batch)
}

pub fn psi_table(batch: &[F162]) -> Vec<Vec<F162>> {
    (0..16)
        .map(|chunk| {
            let mut t = vec![F162::ZERO; 256];
            for m in 1..256usize {
                let b = m.trailing_zeros() as usize;
                t[m] = t[m ^ (1 << b)] + batch[chunk * 8 + b];
            }
            t
        })
        .collect()
}

#[inline]
pub fn psi(tab: &[Vec<F162>], x: B128) -> F162 {
    let mut acc = F162::ZERO;
    for (c, t) in tab.iter().enumerate() {
        acc += t[((x.0 >> (8 * c)) & 0xff) as usize];
    }
    acc
}

#[inline(always)]
unsafe fn affine(x: __m512i, a: __m512i) -> __m512i {
    _mm512_gf2p8affine_epi64_epi8::<0>(x, a)
}

#[inline(always)]
unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

const TILE: usize = 64;

/// `v[k] = sum_j bit_k(pi0[j]) eq_hi[j]`, a `128 x n` by `n x 128` bit product: per tile the
/// bit-transposed bytes of `pi0` are the matrices, those of `eq_hi` the vectors, and
/// `acc[h][kb].qword[i].byte[t].bit[k']` sums `V[8kb + 7 - k'][8(8h + i) + t]`.
fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let mut v = vec![B128::ZERO; PACK];
    unsafe {
        let ident = _mm512_set1_epi64(GF_IDENT);
        let mut acc = [[_mm512_setzero_si512(); 16]; 2];
        let mut xe = [_mm512_setzero_si512(); 16];
        let mut xp = [_mm512_setzero_si512(); 16];
        let mut ap = [[0u64; 8]; 16];
        for t in 0..tiles {
            transpose_tile(eq_hi.as_ptr().add(TILE * t) as *const u8, &mut xe);
            transpose_tile(pi0.as_ptr().add(TILE * t) as *const u8, &mut xp);
            for k in 0..16 {
                _mm512_storeu_si512(ap[k].as_mut_ptr() as *mut __m512i, affine(ident, xp[k]));
            }
            let te: [__m512i; 16] = core::array::from_fn(|b| affine(ident, xe[b]));
            for (h, acc) in acc.iter_mut().enumerate() {
                let tep = transpose8x8_q(core::array::from_fn(|i| te[8 * h + i]));
                for (k, a) in acc.iter_mut().enumerate() {
                    for j in (0..8).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(tep[j], _mm512_set1_epi64(ap[k][j] as i64)),
                            affine(tep[j + 1], _mm512_set1_epi64(ap[k][j + 1] as i64)),
                        );
                    }
                }
            }
        }
        for (h, acc) in acc.iter().enumerate() {
            for (k, a) in acc.iter().enumerate() {
                let q: [u64; 8] = core::mem::transmute(*a);
                for (i, qi) in q.iter().enumerate() {
                    for t in 0..8 {
                        let byte = (qi >> (8 * t)) & 0xff;
                        for kp in 0..8 {
                            if (byte >> kp) & 1 == 1 {
                                v[8 * k + 7 - kp].0 ^= 1u128 << (8 * (8 * h + i) + t);
                            }
                        }
                    }
                }
            }
        }
    }
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

/// The eight `B128` at `x` as their low words and their high words.
#[inline(always)]
unsafe fn split_words(x: *const B128) -> (__m512i, __m512i) {
    let a = _mm512_loadu_si512(x as *const __m512i);
    let b = _mm512_loadu_si512(x.add(4) as *const __m512i);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    (
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
    )
}

/// `pi0` in the sumcheck's layout: the bits of a `B128` are the low coefficients of an `F162`.
fn poly_from_b128(pi0: &[B128], out: &mut super::sumcheck::Poly) {
    let n = pi0.len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    w[2].fill(0);
    let blocks = n / 8;
    unsafe {
        for b in 0..blocks {
            let (lo, hi) = split_words(pi0.as_ptr().add(8 * b));
            _mm512_storeu_si512(w[0].as_mut_ptr().add(8 * b) as *mut __m512i, lo);
            _mm512_storeu_si512(w[1].as_mut_ptr().add(8 * b) as *mut __m512i, hi);
        }
    }
    for j in 8 * blocks..n {
        w[0][j] = pi0[j].0 as u64;
        w[1][j] = (pi0[j].0 >> 64) as u64;
    }
}

/// `m[cb][kb].byte[7 - c].bit[k] = batch[8kb + k].bit(8cb + c)`: the matrix taking byte `kb` of
/// a word to byte `cb` of its `F162` image.
fn eq_matrices(batch: &[F162]) -> [[u64; 16]; 24] {
    let mut m = [[0u64; 16]; 24];
    for (cb, row) in m.iter_mut().enumerate() {
        for (kb, q) in row.iter_mut().enumerate() {
            for c in 0..8 {
                let mut byte = 0u64;
                for k in 0..8 {
                    byte |= ((batch[8 * kb + k].0[cb / 8] >> (8 * (cb % 8) + c)) & 1) << k;
                }
                *q |= byte << (8 * (7 - c));
            }
        }
    }
    m
}

/// `a[j] = sum_i sum_k bit_k(eq[i][j]) batch[i][k]`, straight into the sumcheck's layout: per
/// tile the byte planes of `eq` through [`eq_matrices`], and the 24 output planes back to qwords.
fn poly_from_eq(eqs: &[&[B128]], batches: &[Vec<F162>], out: &mut super::sumcheck::Poly) {
    let n = eqs[0].len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    let tiles = n / TILE;
    let ms: Vec<[[u64; 16]; 24]> = batches.iter().map(|b| eq_matrices(b)).collect();
    unsafe {
        let mut xe = [_mm512_setzero_si512(); 16];
        for t in 0..tiles {
            let mut acc = [_mm512_setzero_si512(); 24];
            for (eq, m) in eqs.iter().zip(&ms) {
                transpose_tile(eq.as_ptr().add(TILE * t) as *const u8, &mut xe);
                for (a, row) in acc.iter_mut().zip(m) {
                    for k in (0..16).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                            affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
                        );
                    }
                }
            }
            for (l, wl) in w.iter_mut().enumerate() {
                let out = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
                for (g, o) in out.iter().enumerate() {
                    _mm512_storeu_si512(wl.as_mut_ptr().add(TILE * t + 8 * g) as *mut __m512i, *o);
                }
            }
        }
    }
    if TILE * tiles < n {
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for j in TILE * tiles..n {
            let x = eqs
                .iter()
                .zip(&tabs)
                .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
            for l in 0..3 {
                w[l][j] = x.0[l];
            }
        }
    }
}

pub struct SwitchProof {
    pub v: Vec<B128>,
    pub rounds: Vec<[F162; 2]>,
    pub final_eval: F162,
}

pub struct Transcript {
    pub r_prime: Vec<F162>,
    pub r_pp: Vec<F162>,
}

pub fn prove(pi0: &[B128], r_lo: &[B128], r_hi: &[B128], challenges: &Transcript) -> SwitchProof {
    assert_eq!(r_lo.len(), LOG_PACK);
    let l = r_hi.len();
    assert_eq!(pi0.len(), 1 << l);

    let eq_hi = eq_expand_b128(r_hi);
    let v = partial_evals(pi0, &eq_hi);

    let batch = eq_expand_f162(&challenges.r_prime);
    let mut ap = super::sumcheck::Poly::zero(pi0.len());
    let mut pp = super::sumcheck::Poly::zero(pi0.len());
    poly_from_eq(&[&eq_hi], &[batch], &mut ap);
    poly_from_b128(pi0, &mut pp);
    let mut rounds = Vec::with_capacity(l);
    let mut half = 1usize << l;
    for round in 0..l {
        half /= 2;
        rounds.push(super::sumcheck::round(
            &mut ap,
            &mut pp,
            half,
            challenges.r_pp[round],
        ));
    }
    let final_eval = pp.get(0);

    SwitchProof {
        v,
        rounds,
        final_eval,
    }
}

pub fn verify(
    proof: &SwitchProof,
    claim: B128,
    r_lo: &[B128],
    r_hi: &[B128],
    challenges: &Transcript,
) -> Result<F162, &'static str> {
    let eq_lo = eq_expand_b128(r_lo);
    let recomputed = proof
        .v
        .iter()
        .zip(&eq_lo)
        .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
    if recomputed != claim {
        return Err("partial evaluation mismatch");
    }

    let mut u = vec![0u128; PACK];
    for (i, &vi) in proof.v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    let batch = eq_expand_f162(&challenges.r_prime);
    let mut s = F162::ZERO;
    for k in 0..PACK {
        s += F162::from_b128(B128(u[k])) * batch[k];
    }

    for (round, msg) in proof.rounds.iter().enumerate() {
        let [e0, einf] = *msg;
        let e1 = s + e0;
        let r = challenges.r_pp[round];
        s = e0 + r * (e0 + e1 + einf) + r * r * einf;
    }

    let d = transparent_coeff(r_hi, &challenges.r_pp, &batch);
    if s != d * proof.final_eval {
        return Err("sumcheck final check failed");
    }
    Ok(proof.final_eval)
}

/// The switch's buffers, sized once and reused: fresh pages cost more than the kernels do.
pub struct Workspace {
    eq: Vec<Vec<B128>>,
    a: super::sumcheck::Poly,
    p: super::sumcheck::Poly,
}

impl Workspace {
    pub fn new(l: usize, claims: usize) -> Self {
        Self {
            eq: (0..claims).map(|_| vec![B128::ZERO; 1 << l]).collect(),
            a: super::sumcheck::Poly::zero(1 << l),
            p: super::sumcheck::Poly::zero(1 << l),
        }
    }

    pub fn fits(&self, l: usize, claims: usize) -> bool {
        self.eq.len() == claims && self.a.n == 1 << l
    }

    /// The partial evaluations of claim `i` at `r_hi`; its `eq` stays for the sumcheck.
    pub fn partial_evals(&mut self, i: usize, pi0: &[B128], r_hi: &[B128]) -> Vec<B128> {
        eq_expand_b128_into(r_hi, &mut self.eq[i]);
        partial_evals(pi0, &self.eq[i])
    }
}

pub struct SwitchProver {
    ws: Workspace,
    half: usize,
}

impl SwitchProver {
    pub fn partial_evals_and_eq(pi0: &[B128], r_hi: &[B128]) -> (Vec<B128>, Vec<B128>) {
        let eq_hi = eq_expand_b128(r_hi);
        let v = partial_evals(pi0, &eq_hi);
        (v, eq_hi)
    }

    pub fn new(pi0: &[B128], eq_hi: &[B128], batch: &[F162]) -> Self {
        Self::batched(pi0, &[eq_hi], &[F162::ONE], batch)
    }

    pub fn batched(pi0: &[B128], eq_his: &[&[B128]], gammas: &[F162], batch: &[F162]) -> Self {
        let mut ws = Workspace::new(pi0.len().trailing_zeros() as usize, eq_his.len());
        for (dst, src) in ws.eq.iter_mut().zip(eq_his) {
            dst.copy_from_slice(src);
        }
        Self::from_workspace(ws, pi0, gammas, batch)
    }

    /// The sumcheck over the `eq` the workspace holds from [`Workspace::partial_evals`].
    pub fn from_workspace(
        mut ws: Workspace,
        pi0: &[B128],
        gammas: &[F162],
        batch: &[F162],
    ) -> Self {
        let batches: Vec<Vec<F162>> = gammas
            .iter()
            .map(|&g| batch.iter().map(|&x| g * x).collect())
            .collect();
        let views: Vec<&[B128]> = ws.eq.iter().map(Vec::as_slice).collect();
        poly_from_eq(&views, &batches, &mut ws.a);
        poly_from_b128(pi0, &mut ws.p);
        Self {
            ws,
            half: pi0.len(),
        }
    }

    pub fn into_workspace(self) -> Workspace {
        self.ws
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        super::sumcheck::msg(&self.ws.a, &self.ws.p, self.half)
    }

    pub fn fold(&mut self, r: F162) {
        super::sumcheck::fold(&mut self.ws.a, &mut self.ws.p, self.half, r);
    }

    pub fn final_eval(&self) -> F162 {
        self.ws.p.get(0)
    }
}

pub fn slice_sum(v: &[B128], batch: &[F162]) -> F162 {
    let mut u = vec![0u128; PACK];
    for (i, &vi) in v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    (0..PACK).fold(F162::ZERO, |s, k| {
        s + F162::from_b128(B128(u[k])) * batch[k]
    })
}

pub struct SwitchVerifier {
    pub s: F162,
    round: usize,
}

impl SwitchVerifier {
    pub fn start(
        v: &[B128],
        claim: B128,
        r_lo: &[B128],
        batch: &[F162],
    ) -> Result<Self, &'static str> {
        let eq_lo = eq_expand_b128(r_lo);
        let recomputed = v
            .iter()
            .zip(&eq_lo)
            .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
        if recomputed != claim {
            return Err("partial evaluation mismatch");
        }
        Ok(Self::from_sum(slice_sum(v, batch)))
    }

    pub const fn from_sum(s: F162) -> Self {
        Self { s, round: 0 }
    }

    pub fn round(&mut self, msg: [F162; 2], r: F162) {
        let [e0, einf] = msg;
        let e1 = self.s + e0;
        self.s = e0 + r * (e0 + e1 + einf) + r * r * einf;
        self.round += 1;
    }

    pub fn finish(
        &self,
        r_hi: &[B128],
        r_pp: &[F162],
        batch: &[F162],
        opened: F162,
    ) -> Result<(), &'static str> {
        let d = transparent_coeff(r_hi, r_pp, batch);
        if self.s != d * opened {
            return Err("sumcheck final check failed");
        }
        Ok(())
    }
}

pub fn eval_pi1(pi0: &[B128], r_pp: &[F162]) -> F162 {
    let eq = eq_expand_f162(r_pp);
    pi0.iter()
        .zip(&eq)
        .fold(F162::ZERO, |a, (&p, &e)| a + F162::from_b128(p) * e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
        let mut v = vec![B128::ZERO; PACK];
        for (&p, &e) in pi0.iter().zip(eq_hi) {
            for (k, vk) in v.iter_mut().enumerate() {
                if p.bit(k) {
                    *vk = *vk + e;
                }
            }
        }
        v
    }

    #[test]
    fn poly_from_eq_matches_psi() {
        let mut x = 0x2545F4914F6CDD1Du128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let batches: Vec<Vec<F162>> = (0..2)
            .map(|_| {
                (0..PACK)
                    .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
                    .collect()
            })
            .collect();
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for n in [1usize, 8, 13, 64, 200, 1024] {
            let eqs: Vec<Vec<B128>> = (0..2)
                .map(|_| (0..n).map(|_| B128(next())).collect())
                .collect();
            let views: Vec<&[B128]> = eqs.iter().map(|e| e.as_slice()).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_eq(&views, &batches, &mut poly);
            for j in 0..n {
                let want = views
                    .iter()
                    .zip(&tabs)
                    .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
                assert_eq!(poly.get(j), want);
            }
            let p: Vec<B128> = (0..n).map(|_| B128(next())).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_b128(&p, &mut poly);
            for j in 0..n {
                assert_eq!(poly.get(j), F162::from_b128(p[j]));
            }
        }
    }

    const M34_TEST: u64 = (1 << 34) - 1;

    #[test]
    fn workspace_reuse_is_clean() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let l = 8;
        let pi0: Vec<B128> = (0..1 << l).map(|_| B128(next())).collect();
        let r_hi: Vec<B128> = (0..l).map(|_| B128(next())).collect();
        let batch: Vec<F162> = (0..PACK)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let r: Vec<F162> = (0..l)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let run = |ws: Workspace| {
            let mut ws = ws;
            let v = ws.partial_evals(0, &pi0, &r_hi);
            let mut p = SwitchProver::from_workspace(ws, &pi0, &[F162::ONE], &batch);
            let mut msgs = Vec::new();
            for &ri in &r {
                msgs.push(p.msg());
                p.fold(ri);
            }
            (v, msgs, p.final_eval(), p.into_workspace())
        };
        let (v1, m1, f1, ws) = run(Workspace::new(l, 1));
        let (v2, m2, f2, _) = run(ws);
        assert_eq!((v1, m1, f1), (v2, m2, f2));
    }

    #[test]
    fn partial_evals_match_the_bitwise_sum() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            B128(x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (x >> 64))
        };
        for n in [1usize, 7, 8, 64, 1000, 4096] {
            let pi0: Vec<B128> = (0..n).map(|_| next()).collect();
            let eq: Vec<B128> = (0..n).map(|_| next()).collect();
            assert_eq!(partial_evals(&pi0, &eq), reference(&pi0, &eq));
        }
    }
}

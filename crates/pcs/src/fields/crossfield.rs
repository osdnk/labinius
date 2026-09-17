use super::scalar::{B128, F162};
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
    let mut t = vec![B128::ONE];
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

/// Byte `4g + i` of the mask spread over 128-bit lane `i`, for group `g` of four bits.
fn spread_idx() -> [__m512i; 16] {
    core::array::from_fn(|g| unsafe {
        let base = _mm512_set_epi8(
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        );
        _mm512_add_epi8(base, _mm512_set1_epi8((4 * g) as i8))
    })
}

/// `v[k] = sum_j bit_k(pi0[j]) eq_hi[j]`: the 64 bits of one word half become 64 byte masks,
/// each group of four is spread over the four lanes of one accumulator, and the lane takes
/// `eq_hi[j]` under its mask.
fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let idx = spread_idx();
    let mut v = vec![B128::ZERO; PACK];
    for half in 0..2 {
        unsafe {
            let mut acc = [_mm512_setzero_si512(); 16];
            for (&p, &e) in pi0.iter().zip(eq_hi) {
                let bits = (p.0 >> (64 * half)) as u64;
                let m = _mm512_movm_epi8(bits);
                let eq =
                    _mm512_broadcast_i32x4(_mm_loadu_si128(&e.0 as *const u128 as *const __m128i));
                for g in 0..16 {
                    let mask = _mm512_permutexvar_epi8(idx[g], m);
                    acc[g] = _mm512_ternarylogic_epi64::<0x78>(acc[g], mask, eq);
                }
            }
            for g in 0..16 {
                _mm512_storeu_si512(
                    v.as_mut_ptr().add(64 * half + 4 * g) as *mut __m512i,
                    acc[g],
                );
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
fn poly_from_b128(pi0: &[B128]) -> super::sumcheck::Poly {
    let n = pi0.len();
    let mut w = [vec![0u64; n], vec![0u64; n], vec![0u64; n]];
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
    super::sumcheck::Poly { w, n }
}

/// Nibble `c` of a word picks one of sixteen sums of `batch[4c..4c + 4]`, one table per limb.
fn nibble_tables(batch: &[F162]) -> Vec<[[__m512i; 2]; 3]> {
    (0..32)
        .map(|c| {
            let mut t = [F162::ZERO; 16];
            for m in 1..16usize {
                let b = m.trailing_zeros() as usize;
                t[m] = t[m ^ (1 << b)] + batch[4 * c + b];
            }
            core::array::from_fn(|l| unsafe {
                let limb = |m: usize| t[m].0[l] as i64;
                [
                    _mm512_setr_epi64(
                        limb(0),
                        limb(1),
                        limb(2),
                        limb(3),
                        limb(4),
                        limb(5),
                        limb(6),
                        limb(7),
                    ),
                    _mm512_setr_epi64(
                        limb(8),
                        limb(9),
                        limb(10),
                        limb(11),
                        limb(12),
                        limb(13),
                        limb(14),
                        limb(15),
                    ),
                ]
            })
        })
        .collect()
}

/// `a[j] = sum_i sum_k bit_k(eq[i][j]) batch[i][k]`, straight into the sumcheck's layout: eight
/// words at a time, nibble by nibble through [`nibble_tables`].
fn poly_from_eq(eqs: &[&[B128]], batches: &[Vec<F162>]) -> super::sumcheck::Poly {
    let n = eqs[0].len();
    let mut w = [vec![0u64; n], vec![0u64; n], vec![0u64; n]];
    let tables: Vec<_> = batches.iter().map(|b| nibble_tables(b)).collect();
    let blocks = n / 8;
    unsafe {
        let nib = _mm512_set1_epi64(0xf);
        for b in 0..blocks {
            let mut acc = [_mm512_setzero_si512(); 3];
            for (eq, tab) in eqs.iter().zip(&tables) {
                let (mut lo, mut hi) = split_words(eq.as_ptr().add(8 * b));
                for c in 0..16 {
                    let i_lo = _mm512_and_si512(lo, nib);
                    let i_hi = _mm512_and_si512(hi, nib);
                    lo = _mm512_srli_epi64::<4>(lo);
                    hi = _mm512_srli_epi64::<4>(hi);
                    for l in 0..3 {
                        let [t0, t1] = tab[c][l];
                        let [u0, u1] = tab[16 + c][l];
                        acc[l] = _mm512_ternarylogic_epi64::<0x96>(
                            acc[l],
                            _mm512_permutex2var_epi64(t0, i_lo, t1),
                            _mm512_permutex2var_epi64(u0, i_hi, u1),
                        );
                    }
                }
            }
            for l in 0..3 {
                _mm512_storeu_si512(w[l].as_mut_ptr().add(8 * b) as *mut __m512i, acc[l]);
            }
        }
    }
    if blocks * 8 < n {
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for j in 8 * blocks..n {
            let x = eqs
                .iter()
                .zip(&tabs)
                .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
            for l in 0..3 {
                w[l][j] = x.0[l];
            }
        }
    }
    super::sumcheck::Poly { w, n }
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
    let mut ap = poly_from_eq(&[&eq_hi], &[batch]);
    let mut pp = poly_from_b128(pi0);
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

pub struct SwitchProver {
    a: super::sumcheck::Poly,
    p: super::sumcheck::Poly,
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
        let batches: Vec<Vec<F162>> = gammas
            .iter()
            .map(|&g| batch.iter().map(|&x| g * x).collect())
            .collect();
        Self {
            a: poly_from_eq(eq_his, &batches),
            p: poly_from_b128(pi0),
            half: pi0.len(),
        }
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        super::sumcheck::msg(&self.a, &self.p, self.half)
    }

    pub fn fold(&mut self, r: F162) {
        super::sumcheck::fold(&mut self.a, &mut self.p, self.half, r);
    }

    pub fn final_eval(&self) -> F162 {
        self.p.get(0)
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
        for n in [1usize, 8, 13, 64] {
            let eqs: Vec<Vec<B128>> = (0..2)
                .map(|_| (0..n).map(|_| B128(next())).collect())
                .collect();
            let views: Vec<&[B128]> = eqs.iter().map(|e| e.as_slice()).collect();
            let poly = poly_from_eq(&views, &batches);
            for j in 0..n {
                let want = views
                    .iter()
                    .zip(&tabs)
                    .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
                assert_eq!(poly.get(j), want);
            }
            let p: Vec<B128> = (0..n).map(|_| B128(next())).collect();
            let poly = poly_from_b128(&p);
            for j in 0..n {
                assert_eq!(poly.get(j), F162::from_b128(p[j]));
            }
        }
    }

    const M34_TEST: u64 = (1 << 34) - 1;

    #[test]
    fn partial_evals_match_the_bitwise_sum() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            B128(x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (x >> 64))
        };
        for n in [1usize, 7, 8, 1000] {
            let pi0: Vec<B128> = (0..n).map(|_| next()).collect();
            let eq: Vec<B128> = (0..n).map(|_| next()).collect();
            assert_eq!(partial_evals(&pi0, &eq), reference(&pi0, &eq));
        }
    }
}

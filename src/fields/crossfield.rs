use super::scalar::{B128, F162};

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


fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    let mut acc = vec![[0u128; 16]; 32];
    for (&p, &e) in pi0.iter().zip(eq_hi) {
        let w = p.0;
        let ev = e.0;
        for (c, t) in acc.iter_mut().enumerate() {
            t[((w >> (4 * c)) & 0xf) as usize] ^= ev;
        }
    }
    let mut v = vec![B128::ZERO; PACK];
    for (c, t) in acc.iter().enumerate() {
        for b in 0..4 {
            let mut s = 0u128;
            for (m, &x) in t.iter().enumerate() {
                if (m >> b) & 1 == 1 {
                    s ^= x;
                }
            }
            v[4 * c + b] = B128(s);
        }
    }
    v
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

pub fn prove(
    pi0: &[B128],
    r_lo: &[B128],
    r_hi: &[B128],
    challenges: &Transcript,
) -> SwitchProof {
    assert_eq!(r_lo.len(), LOG_PACK);
    let l = r_hi.len();
    assert_eq!(pi0.len(), 1 << l);

    let eq_hi = eq_expand_b128(r_hi);
    let v = partial_evals(pi0, &eq_hi);

    let batch = eq_expand_f162(&challenges.r_prime);
    let tab = psi_table(&batch);
    let a: Vec<F162> = eq_hi.iter().map(|&e| psi(&tab, e)).collect();
    let p: Vec<F162> = pi0.iter().map(|&x| F162::from_b128(x)).collect();

    let mut ap = super::sumcheck::Poly::from_scalars(&a);
    let mut pp = super::sumcheck::Poly::from_scalars(&p);
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
        let tab = psi_table(batch);
        let a: Vec<F162> = eq_hi.iter().map(|&e| psi(&tab, e)).collect();
        let p: Vec<F162> = pi0.iter().map(|&x| F162::from_b128(x)).collect();
        let half = pi0.len();
        Self {
            a: super::sumcheck::Poly::from_scalars(&a),
            p: super::sumcheck::Poly::from_scalars(&p),
            half,
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

pub struct SwitchVerifier {
    pub s: F162,
    round: usize,
}

impl SwitchVerifier {
    pub fn start(v: &[B128], claim: B128, r_lo: &[B128], batch: &[F162]) -> Result<Self, &'static str> {
        let eq_lo = eq_expand_b128(r_lo);
        let recomputed = v
            .iter()
            .zip(&eq_lo)
            .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
        if recomputed != claim {
            return Err("partial evaluation mismatch");
        }
        let mut u = vec![0u128; PACK];
        for (i, &vi) in v.iter().enumerate() {
            for k in 0..PACK {
                if vi.bit(k) {
                    u[k] |= 1u128 << i;
                }
            }
        }
        let mut s = F162::ZERO;
        for k in 0..PACK {
            s += F162::from_b128(B128(u[k])) * batch[k];
        }
        Ok(Self { s, round: 0 })
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

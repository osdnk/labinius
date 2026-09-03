use crate::fields::crossfield as cf;
use crate::fields::scalar::{B128 as SB, F162};
use flock_core::pcs::ring_switch::build_claim_weights_from_skip;
use flock_core::proof::ZClaim;
use flock_field::F128;
use flock_transcript::challenger::Challenger;

pub const LOG_PACKING: usize = cf::LOG_PACK;
pub const K_SKIP: usize = LOG_PACKING - 1;

pub struct Proof {
    pub v: Vec<Vec<F128>>,
    pub rounds: Vec<[F162; 2]>,
}

pub struct Output {
    pub opened: F162,
    pub r_pp: Vec<F162>,
}

pub struct Check {
    pub s: F162,
    pub d: F162,
    pub r_pp: Vec<F162>,
}

#[inline]
fn lift(x: F128) -> SB {
    SB((x.lo as u128) | ((x.hi as u128) << 64))
}

#[inline]
fn drop_(x: SB) -> F128 {
    F128::new(x.0 as u64, (x.0 >> 64) as u64)
}

fn compose(a: F128, b: F128) -> F162 {
    F162([a.lo, a.hi, b.lo & ((1 << 34) - 1)])
}

fn sample<Ch: Challenger>(ch: &mut Ch, n: usize) -> Vec<F162> {
    let raw = ch.sample_f128_vec(2 * n);
    (0..n)
        .map(|i| compose(raw[2 * i], raw[2 * i + 1]))
        .collect()
}

fn observe<Ch: Challenger>(ch: &mut Ch, xs: &[F162]) {
    let words: Vec<F128> = xs
        .iter()
        .flat_map(|x| [F128::new(x.0[0], x.0[1]), F128::new(x.0[2], 0)])
        .collect();
    ch.observe_f128_slice(&words);
}

fn suffix(claim: &ZClaim) -> Vec<F128> {
    let point = &claim.point;
    point
        .x_inner_rest
        .iter()
        .chain(&point.x_outer)
        .skip(1)
        .copied()
        .collect()
}

fn weights(claim: &ZClaim) -> Vec<F128> {
    let point = &claim.point;
    let head = *point
        .x_inner_rest
        .first()
        .or_else(|| point.x_outer.first())
        .expect("the claim point has at least the seventh packing coordinate");
    build_claim_weights_from_skip(&point.z_skip.weights(K_SKIP), head)
}

fn descending(x: &[F128]) -> Vec<SB> {
    x.iter().rev().map(|&y| lift(y)).collect()
}

pub fn prove<Ch: Challenger>(trace: &[SB], claims: &[ZClaim], ch: &mut Ch) -> (Proof, Output) {
    let points: Vec<Vec<SB>> = claims.iter().map(|c| descending(&suffix(c))).collect();
    let l = points[0].len();
    assert_eq!(trace.len(), 1 << l, "the trace does not match the claims");

    let mut v = Vec::with_capacity(claims.len());
    let mut eq = Vec::with_capacity(claims.len());
    for point in &points {
        let (vi, eqi) = cf::SwitchProver::partial_evals_and_eq(trace, point);
        v.push(vi.iter().map(|&x| drop_(x)).collect::<Vec<_>>());
        eq.push(eqi);
    }
    for vi in &v {
        ch.observe_f128_slice(vi);
    }

    let batch = cf::eq_expand_f162(&sample(ch, LOG_PACKING));
    let gammas = combiners(ch, claims.len());
    let views: Vec<&[SB]> = eq.iter().map(|e| e.as_slice()).collect();
    let mut prover = cf::SwitchProver::batched(trace, &views, &gammas, &batch);

    let mut rounds = Vec::with_capacity(l);
    let mut r_pp = Vec::with_capacity(l);
    for _ in 0..l {
        let msg = prover.msg();
        observe(ch, &msg);
        let r = sample(ch, 1)[0];
        prover.fold(r);
        rounds.push(msg);
        r_pp.push(r);
    }
    (
        Proof { v, rounds },
        Output {
            opened: prover.final_eval(),
            r_pp,
        },
    )
}

pub fn verify<Ch: Challenger>(
    claims: &[ZClaim],
    proof: &Proof,
    ch: &mut Ch,
) -> Result<Check, &'static str> {
    if proof.v.len() != claims.len() || proof.v.iter().any(|v| v.len() != 1 << LOG_PACKING) {
        return Err("the partial evaluations do not match the claims");
    }
    let points: Vec<Vec<SB>> = claims.iter().map(|c| descending(&suffix(c))).collect();
    let l = points[0].len();
    if proof.rounds.len() != l {
        return Err("truncated sumcheck");
    }

    for (claim, v) in claims.iter().zip(&proof.v) {
        let w = weights(claim);
        let recomputed = v
            .iter()
            .zip(&w)
            .fold(SB::ZERO, |acc, (&vi, &wi)| acc + lift(vi) * lift(wi));
        if recomputed != lift(claim.value) {
            return Err("partial evaluation mismatch");
        }
        ch.observe_f128_slice(v);
    }

    let batch = cf::eq_expand_f162(&sample(ch, LOG_PACKING));
    let gammas = combiners(ch, claims.len());
    let mut s = F162::ZERO;
    for (v, &g) in proof.v.iter().zip(&gammas) {
        let lifted: Vec<SB> = v.iter().map(|&x| lift(x)).collect();
        s += g * cf::slice_sum(&lifted, &batch);
    }

    let mut verifier = cf::SwitchVerifier::from_sum(s);
    let mut r_pp = Vec::with_capacity(l);
    for msg in &proof.rounds {
        observe(ch, msg);
        let r = sample(ch, 1)[0];
        verifier.round(*msg, r);
        r_pp.push(r);
    }

    let d = points
        .iter()
        .zip(&gammas)
        .fold(F162::ZERO, |acc, (point, &g)| {
            acc + g * cf::transparent_coeff(point, &r_pp, &batch)
        });
    Ok(Check {
        s: verifier.s,
        d,
        r_pp,
    })
}

fn combiners<Ch: Challenger>(ch: &mut Ch, n: usize) -> Vec<F162> {
    let mut gammas = vec![F162::ONE];
    gammas.extend(sample(ch, n - 1));
    gammas
}

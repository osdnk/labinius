//! The cross-field switch, on top of [`crate::fields::crossfield`]: binius64's `B128` claim
//! `w~(r) = s` on the packed trace becomes an `F162` evaluation claim on the same trace lifted by
//! `phi` (the `beta` basis of `B128` onto `{1, X, ..., X^127} ⊂ F162`, so a zero-extension).
//!
//! The prover sends the 128 partial evaluations `v_i`, both sides draw `r' ∈ F162^7` after them,
//! and an `l`-round sumcheck over `F162` reduces to `pi1~(r'')`, which the PCS opens.
use crate::fields::crossfield as cf;
use crate::fields::scalar::{B128 as SB, F162};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_verifier::config::B128;

/// `log2` of the number of `B1` coordinates one `B128` element packs.
pub const LOG_PACKING: usize = cf::LOG_PACK;

/// A `B128` challenge pair read as one `F162`, the low 34 bits of the second kept.
fn compose(a: u128, b: u128) -> F162 {
    F162([a as u64, (a >> 64) as u64, (b as u64) & ((1 << 34) - 1)])
}

fn sample_prover<C: IPProverChannel<B128>>(channel: &mut C, n: usize) -> Vec<F162> {
    let raw = channel.sample_many(2 * n);
    (0..n).map(|i| compose(u128::from(raw[2 * i]), u128::from(raw[2 * i + 1]))).collect()
}

fn sample_verifier<C: IPVerifierChannel<B128, Elem = B128>>(channel: &mut C, n: usize) -> Vec<F162> {
    let raw = channel.sample_many(2 * n);
    (0..n).map(|i| compose(u128::from(raw[2 * i]), u128::from(raw[2 * i + 1]))).collect()
}

/// Two `B128` per `F162`, the way a round message travels on a `B128` channel.
fn encode(xs: &[F162]) -> Vec<B128> {
    xs.iter()
        .flat_map(|x| {
            [
                B128::from((x.0[0] as u128) | ((x.0[1] as u128) << 64)),
                B128::from(x.0[2] as u128),
            ]
        })
        .collect()
}

fn decode(e: &[B128]) -> Vec<F162> {
    e.chunks_exact(2)
        .map(|c| {
            let (a, b) = (u128::from(c[0]), u128::from(c[1]));
            F162([a as u64, (a >> 64) as u64, b as u64])
        })
        .collect()
}

/// The claim the switch leaves for the PCS: `pi1~(r_pp) = opened`.
pub struct Output {
    pub opened: F162,
    pub r_pp: Vec<F162>,
}

/// `trace` is the packed non-public trace as `B128`, `eval_point` binius64's claim point.
pub fn prove<Channel: IPProverChannel<B128>>(
    trace: &[SB],
    eval_point: &[B128],
    channel: &mut Channel,
) -> Output {
    let l = eval_point.len() - LOG_PACKING;
    assert_eq!(trace.len(), 1 << l, "the trace does not match the claim point");

    let r_hi: Vec<SB> = eval_point[LOG_PACKING..].iter().rev().map(|&x| SB(u128::from(x))).collect();
    let (v, eq_hi) = cf::SwitchProver::partial_evals_and_eq(trace, &r_hi);
    channel.send_many(&v.iter().map(|&x| B128::from(x.0)).collect::<Vec<_>>());

    let batch = cf::eq_expand_f162(&sample_prover(channel, LOG_PACKING));
    let mut prover = cf::SwitchProver::new(trace, &eq_hi, &batch);

    let mut r_pp = Vec::with_capacity(l);
    for _ in 0..l {
        channel.send_many(&encode(&prover.msg()));
        let r = sample_prover(channel, 1)[0];
        prover.fold(r);
        r_pp.push(r);
    }
    Output {
        opened: prover.final_eval(),
        r_pp,
    }
}

/// The verifier's half: it stops one step short, returning the running sum `s` and the
/// transparent coefficient `d`, so the caller checks `s == d * opened` against the PCS's opening.
pub struct Check {
    pub s: F162,
    pub d: F162,
    pub r_pp: Vec<F162>,
}

pub fn verify<Channel: IPVerifierChannel<B128, Elem = B128>>(
    claim: B128,
    eval_point: &[B128],
    channel: &mut Channel,
) -> Result<Check, &'static str> {
    let l = eval_point.len() - LOG_PACKING;
    let coordinate = |x: &[B128]| -> Vec<SB> { x.iter().rev().map(|&y| SB(u128::from(y))).collect() };
    let r_lo = coordinate(&eval_point[..LOG_PACKING]);
    let r_hi = coordinate(&eval_point[LOG_PACKING..]);

    let v: Vec<SB> = channel
        .recv_many(1 << LOG_PACKING)
        .map_err(|_| "truncated partial evaluations")?
        .iter()
        .map(|&x| SB(u128::from(x)))
        .collect();

    let batch = cf::eq_expand_f162(&sample_verifier(channel, LOG_PACKING));
    let mut verifier = cf::SwitchVerifier::start(&v, SB(u128::from(claim)), &r_lo, &batch)?;

    let mut r_pp = Vec::with_capacity(l);
    for _ in 0..l {
        let msg = decode(&channel.recv_many(4).map_err(|_| "truncated round message")?);
        let r = sample_verifier(channel, 1)[0];
        verifier.round([msg[0], msg[1]], r);
        r_pp.push(r);
    }
    let d = cf::transparent_coeff(&r_hi, &r_pp, &batch);
    Ok(Check {
        s: verifier.s,
        d,
        r_pp,
    })
}

//! One recursive round over rokoko: setup, the outer commitments, the opening proof, its check.
use std::time::{Duration, Instant};

use rokoko::common::ring_arithmetic::RingElement;
use rokoko::protocol::config::{Config, SumcheckConfig};

use super::{claims, config, relation, Cap, Element, Layout, DIGIT, SUPPORT};
use crate::api::{PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162};
use crate::challenge::{ShortChallenge, Transcript};
use crate::fields::scalar::F162;
use crate::params::N;
use crate::scheme::{
    EvaluationPoint, FoldedWitness, FoldingChallenges, OpeningError, PublicParameters,
    RowEvaluation, VerificationError,
};

/// The layout is a function of the shape alone, so it is fixed here, at key time.
pub struct Setup {
    pub relation: relation::Setup,
    pub layout: Layout,
    pub keys: claims::Keys,
    pub shape: config::Shape,
    pub config: Config,
    pub crs: claims::Crs,
}

impl Setup {
    pub fn new(pp: &PublicParameters, seed: [u8; 32]) -> Setup {
        let params = pp.params();
        let relation = relation::Setup::new(pp);
        let quiet = FoldingChallenges::of(vec![
            ShortChallenge::from_coeffs(&[0i8; N162]);
            params.columns()
        ]);
        let origin = EvaluationPoint::of(
            vec![F162::ZERO; params.row_log_len() as usize],
            vec![F162::ZERO; params.column_log_len as usize],
        );
        let layout = relation::layout(&relation, &quiet, &origin, &F162::ZERO).layout;
        let shape = config::Shape::of_len(layout.len);
        let residues = relation.residue_vectors_at();
        let lift = relation.lift_vector_at();
        let rank = |vectors: &[usize]| {
            let (m, capsq) = vectors.iter().fold((0usize, 0f64), |(m, c), &v| {
                let cap = l2_cap(&layout, v);
                (m + layout.regions[v].len, c + cap * cap)
            });
            config::outer_rank(m, capsq.sqrt())
        };
        let keys = claims::Keys::new(
            &layout,
            &residues,
            lift,
            seed,
            rank(&residues),
            rank(&[lift]),
        );
        let config = config::chain(shape);
        let crs = match &config {
            Config::Sumcheck(c) => claims::Crs::new(c),
            _ => panic!("the chain starts with a sumcheck round"),
        };
        Setup {
            relation,
            layout,
            keys,
            shape,
            config,
            crs,
        }
    }

    pub fn sumcheck(&self) -> &SumcheckConfig {
        match &self.config {
            Config::Sumcheck(c) => c,
            _ => panic!("the chain starts with a sumcheck round"),
        }
    }
}

/// The cap on the fold, per ring element and challenge, on the triangle bound
/// `(‖v^0‖ + DIGIT ‖v^1‖)^2 >= ‖v‖^2` the verifier forms from the two digit norms it is shipped:
/// about three times `‖v‖^2` itself, 38 to 92 over eight rounds at the basic shape, so this sits
/// where [`crate::recursion::FOLD_CAP`] sits over the plain norm, at a few retries in a hundred.
pub const FOLD_CAP: f64 = 200.0;

fn normsq(elements: &[Element]) -> f64 {
    elements
        .iter()
        .flat_map(|e| e.iter())
        .map(|&x| (x as f64) * (x as f64))
        .sum()
}

/// `(‖v^0‖ + DIGIT ‖v^1‖)^2` from the two squared norms.
pub fn fold_bound(v0: f64, v1: f64) -> f64 {
    let b = v0.sqrt() + DIGIT as f64 * v1.sqrt();
    b * b
}

fn fold_cap(setup: &Setup) -> f64 {
    FOLD_CAP * (setup.relation.n * N * setup.relation.r) as f64
}

pub fn l2_cap(layout: &Layout, v: usize) -> f64 {
    match layout.vectors[v].cap {
        Cap::PerCoefficient(c) => c * ((layout.vectors[v].used * SUPPORT) as f64).sqrt(),
        Cap::Betasq(b) => b.sqrt(),
    }
}

pub fn commit(
    setup: &Setup,
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
) -> Vec<RingElement> {
    claims::commit_residues(
        &setup.keys,
        &relation::residue_vectors(&setup.relation, matrix),
    )
}

pub fn commit_lift(setup: &Setup, row: &RowEvaluation) -> Vec<RingElement> {
    claims::commit_lift(&setup.keys, &relation::lift_vector(&setup.relation, row))
}

pub fn bytes(elements: &[RingElement]) -> Vec<u8> {
    let mut out = Vec::with_capacity(elements.len() * super::DEG * 8);
    for e in elements {
        for x in e.v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// `ceil(log2 q) = 50` bits per slot.
pub fn wire_bytes(elements: &[RingElement]) -> usize {
    (elements.len() * super::DEG * 50).div_ceil(8)
}

pub struct OpeningProof {
    pub proof: claims::Proof,
    pub timings: OpeningTimings,
}

impl OpeningProof {
    pub fn wire_bytes(&self) -> usize {
        self.proof.wire_bytes()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct OpeningTimings {
    pub encoding: Duration,
    pub rokoko: Duration,
}

fn absorb_claim(transcript: &mut Transcript, claim: &F162) -> [u8; 32] {
    transcript.absorb_bytes(b"bin-ntt/claim");
    for limb in claim.0 {
        transcript.absorb_u64(limb);
    }
    let mut digest = [0u8; 32];
    transcript.fill(b"bin-ntt/rokoko/statement", &mut digest);
    digest
}

#[allow(clippy::too_many_arguments)]
pub fn prove(
    setup: &Setup,
    transcript: &mut Transcript,
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    folded: &FoldedWitness,
    row: &RowEvaluation,
    challenges: &FoldingChallenges,
    point: &EvaluationPoint,
    claim: &F162,
    t_y: &[RingElement],
    t_u: &[RingElement],
) -> Result<OpeningProof, OpeningError> {
    let mut timings = OpeningTimings::default();
    let clock = Instant::now();
    let (relation, witness) = relation::encode(
        &setup.relation,
        matrix,
        folded,
        row,
        challenges,
        point,
        claim,
    )
    .map_err(OpeningError::GadgetOverflow)?;
    timings.encoding = clock.elapsed();
    let bound = fold_bound(normsq(&witness[0]), normsq(&witness[1]));
    let cap = fold_cap(setup);
    if std::env::var_os("GADGET_STATS").is_some() {
        eprintln!(
            "gadget-stats fold bound {bound:.0} cap {cap:.0} fill {:.3} per-element {:.2}",
            bound / cap,
            bound / (setup.relation.n * N * setup.relation.r) as f64
        );
    }
    if bound > cap {
        return Err(OpeningError::FoldTooLong {
            normsq: bound as u64,
            cap: cap as u64,
        });
    }
    let digest = absorb_claim(transcript, claim);
    let clock = Instant::now();
    let proof = claims::prove(
        &relation,
        &witness,
        &setup.keys,
        t_y,
        t_u,
        &setup.crs,
        setup.sumcheck(),
        digest,
    )
    .map_err(OpeningError::Rokoko)?;
    timings.rokoko = clock.elapsed();
    Ok(OpeningProof { proof, timings })
}

#[allow(clippy::too_many_arguments)]
pub fn verify(
    setup: &Setup,
    transcript: &mut Transcript,
    t_y: &[RingElement],
    t_u: &[RingElement],
    point: &EvaluationPoint,
    claim: &F162,
    challenges: &FoldingChallenges,
    proof: &OpeningProof,
) -> Result<(), VerificationError> {
    let relation = relation::layout(&setup.relation, challenges, point, claim);
    if relation.layout != setup.layout {
        return Err(VerificationError::Rejected);
    }
    let digest = absorb_claim(transcript, claim);
    claims::verify(
        &relation,
        &setup.keys,
        t_y,
        t_u,
        &setup.crs,
        setup.sumcheck(),
        digest,
        &proof.proof,
    )
    .map_err(|_| VerificationError::Rejected)?;
    let norm = |v: usize| claims::constant_term(&proof.proof.norms[v]) as f64;
    if fold_bound(norm(0), norm(1)) > fold_cap(setup) {
        return Err(VerificationError::Rejected);
    }
    Ok(())
}

//! The exact-norm chains for the three shapes of the committed vector, their calibrated norm
//! tables, and the outer Ajtai ranks. Every round projects fine: rokoko's coarse projection
//! reads `projection_height * projection_ratio` rows per block and is empty below height
//! `2^13`, while the fine one works from height `2 * projection_ratio`.
use std::sync::LazyLock;

use rokoko::common::estimator::{estimate_rsis_security, RSISParameters};
use rokoko::protocol::config::{Config, SimpleConfig, SumcheckConfig};
use rokoko::protocol::config_generator::{
    AuxConfig, AuxProjection, AuxRecursionConfig, AuxSumcheckConfig,
};
use rokoko::protocol::params::NORM_MARGIN;

/// Measured with `examples/rokoko_chain.rs` under `rokoko-hardness`: the basic commitment of
/// every round, then the recursion levels (rank 2, then rank 1) of every commitment, opening
/// and projection tree; sizes from the plain build, times on one core with AVX-512.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `2^14 -> 2^12 -> 2^11 -> 2^10`: 139 / 137 / 158 / 155 bits at ranks 6 / 5 / 5 / 4;
    /// levels 170 / 156, 212 / 157, 197. Proof 89.0 KB, prover 0.22 s, verifier 5 ms.
    N14,
    /// `2^15 -> 2^13 -> 2^12 -> 2^11 -> 2^10`: 134 / 146 / 139 / 158 / 155 bits at ranks
    /// 6 / 6 / 5 / 5 / 4; levels 160 / 157, 188 / 157, 218 / 157, 197. Proof 96.2 KB, prover
    /// 0.41 s, verifier 6 ms.
    N15,
    /// `2^16 -> 2^13 -> 2^12 -> 2^11 -> 2^10`: 134 / 144 / 139 / 158 / 155 bits at ranks
    /// 6 / 6 / 5 / 5 / 4; levels 158 / 157, 184 / 157, 218 / 157, 197. Proof 96.3 KB, prover
    /// 0.69 s, verifier 7 ms.
    N16,
    /// `2^17 -> 2^14 -> 2^12 -> 2^11 -> 2^10`: 130 / 141 / 137 / 158 / 154 bits at ranks
    /// 6 / 6 / 5 / 5 / 4; levels 149 / 157, 175 / 157, 211 / 157, 197. Proof 96.4 KB, prover
    /// 1.27 s, verifier 8 ms.
    N17,
}

impl Shape {
    pub const fn log_len(self) -> u32 {
        match self {
            Shape::N14 => 14,
            Shape::N15 => 15,
            Shape::N16 => 16,
            Shape::N17 => 17,
        }
    }

    pub const fn len(self) -> usize {
        1 << self.log_len()
    }

    pub fn of_len(len: usize) -> Shape {
        [Shape::N14, Shape::N15, Shape::N16, Shape::N17]
            .into_iter()
            .find(|shape| shape.len() >= len)
            .unwrap_or_else(|| panic!("{len} elements exceed the large chain"))
    }
}

const SECURITY: f64 = 128.0;

fn levels(
    base_log: usize,
    chunks: usize,
    rank: usize,
    next: Option<AuxRecursionConfig>,
) -> AuxRecursionConfig {
    AuxRecursionConfig {
        decomposition_base_log: base_log,
        decomposition_chunks: chunks,
        rank,
        next: next.map(Box::new),
    }
}

fn last() -> AuxRecursionConfig {
    levels(7, 8, 1, None)
}

fn base7(rank: usize) -> AuxRecursionConfig {
    levels(7, 8, rank, Some(last()))
}

fn fine(constant_term: AuxRecursionConfig, batched: AuxRecursionConfig) -> AuxProjection {
    AuxProjection::Fine {
        nof_batches: 2,
        recursion_constant_term: constant_term,
        recursion_batched_projection: batched,
    }
}

struct Round {
    height_log: u32,
    width_log: u32,
    ratio_log: u32,
    rank: usize,
    commitment: AuxRecursionConfig,
    opening: AuxRecursionConfig,
    projection: AuxProjection,
    witness_base_log: usize,
    witness_chunks: usize,
    exact: bool,
}

impl Round {
    fn into_aux(self, next: AuxConfig) -> AuxSumcheckConfig {
        AuxSumcheckConfig {
            exact_projection_norm: self.exact,
            witness_height: 1 << self.height_log,
            witness_width: 1 << self.width_log,
            projection_ratio: 1 << self.ratio_log,
            projection_height: 1 << 8,
            basic_commitment_rank: self.rank,
            nof_openings: 2,
            commitment_recursion: self.commitment,
            opening_recursion: self.opening,
            projection_recursion: self.projection,
            witness_decomposition_base_log: self.witness_base_log,
            witness_decomposition_chunks: self.witness_chunks,
            next: Some(Box::new(next)),
        }
    }
}

fn tail_last() -> SimpleConfig {
    SimpleConfig {
        witness_height: 1 << 8,
        witness_width: 1 << 2,
        projection_ratio: 1 << 7,
        projection_height: 1 << 8,
        basic_commitment_rank: 4,
        projection_nof_batches: 2,
        witness_norm_bound: f64::INFINITY,
        projection_norm_bound: f64::INFINITY,
    }
}

/// `2^11 -> 2^10`, composed `512 + 240 + 112 + 64 + 96`: rank 5 fits only with six base-9
/// commitment digits.
fn tail_11() -> Round {
    Round {
        height_log: 8,
        width_log: 3,
        ratio_log: 6,
        rank: 5,
        commitment: levels(9, 6, 2, None),
        opening: levels(8, 7, 2, None),
        projection: fine(levels(9, 2, 2, None), levels(9, 6, 2, None)),
        witness_base_log: 7,
        witness_chunks: 2,
        exact: false,
    }
}

/// `2^12 -> 2^11`
fn tail_12() -> Round {
    Round {
        height_log: 9,
        width_log: 3,
        ratio_log: 5,
        rank: 5,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(9, 2, 2, Some(last())), base7(2)),
        witness_base_log: 7,
        witness_chunks: 2,
        exact: false,
    }
}

/// `2^13 -> 2^12`
fn tail_13() -> Round {
    Round {
        height_log: 9,
        width_log: 4,
        ratio_log: 6,
        rank: 6,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(10, 2, 2, Some(last())), base7(2)),
        witness_base_log: 8,
        witness_chunks: 2,
        exact: false,
    }
}

fn from_12() -> AuxConfig {
    AuxConfig::Sumcheck(tail_12().into_aux(AuxConfig::Sumcheck(
        tail_11().into_aux(AuxConfig::Simple(tail_last())),
    )))
}

fn from_13() -> AuxConfig {
    AuxConfig::Sumcheck(tail_13().into_aux(from_12()))
}

/// `2^14 -> 2^12`
fn round_14(exact: bool) -> Round {
    Round {
        height_log: 10,
        width_log: 4,
        ratio_log: 6,
        rank: 6,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(10, 2, 2, Some(last())), base7(2)),
        witness_base_log: 8,
        witness_chunks: 2,
        exact,
    }
}

fn root_small() -> AuxSumcheckConfig {
    round_14(true).into_aux(from_12())
}

/// `2^15 -> 2^13`
fn root_medium() -> AuxSumcheckConfig {
    Round {
        height_log: 10,
        width_log: 5,
        ratio_log: 6,
        rank: 6,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(10, 2, 2, Some(last())), base7(2)),
        witness_base_log: 8,
        witness_chunks: 2,
        exact: true,
    }
    .into_aux(from_13())
}

/// `2^16 -> 2^13`
fn root_large() -> AuxSumcheckConfig {
    Round {
        height_log: 11,
        width_log: 5,
        ratio_log: 7,
        rank: 6,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(10, 2, 2, Some(last())), base7(2)),
        witness_base_log: 8,
        witness_chunks: 2,
        exact: true,
    }
    .into_aux(from_13())
}

/// `2^17 -> 2^14`
fn root_huge() -> AuxSumcheckConfig {
    Round {
        height_log: 11,
        width_log: 6,
        ratio_log: 7,
        rank: 6,
        commitment: base7(2),
        opening: base7(2),
        projection: fine(levels(10, 2, 2, Some(last())), base7(2)),
        witness_base_log: 8,
        witness_chunks: 2,
        exact: true,
    }
    .into_aux(AuxConfig::Sumcheck(round_14(false).into_aux(from_12())))
}

/// Raw maxima of one `rokoko-calibration` run, one row per round: `[norm, most inner norm,
/// projection image]` for a sumcheck round, `[folded witness, projection image]` for the simple
/// one; bounds are these times `NORM_MARGIN`. `N14`, `N15` and `N17` are measured on a real
/// recursive round (`--features rokoko-calibration,sizes`, `,sizem`, `,sizel`); `N16`, which no
/// shape of `main.rs` uses, on `examples/rokoko_chain.rs`.
const NB_N14: [[f64; 3]; 4] = [
    [53001.89368315061, 3124.462993859905, 320994.8826321068],
    [21637.617636884148, 3162.719873779529, f64::INFINITY],
    [31335.354457864363, 30596.542745872448, f64::INFINITY],
    [146901.68424493982, 364429.44947959407, f64::INFINITY],
];

const NB_N15: [[f64; 3]; 5] = [
    [69396.465226984, 3107.31298713213, 456313.71600468026],
    [36660.02418166142, 3150.5778835001047, f64::INFINITY],
    [19787.452312008234, 3154.1041517362737, f64::INFINITY],
    [31405.55707195782, 30678.110404651718, f64::INFINITY],
    [147573.84737818554, 372770.9135662277, f64::INFINITY],
];

const NB_N16: [[f64; 3]; 5] = [
    [87096.0231181654, 3151.570878149498, 976784.4307046463],
    [42590.31153912824, 3125.0787190085307, f64::INFINITY],
    [20010.68382139901, 3133.12048922476, f64::INFINITY],
    [31358.225507831274, 30626.841054865585, f64::INFINITY],
    [146920.83620099636, 345119.3229275927, f64::INFINITY],
];

const NB_N17: [[f64; 3]; 5] = [
    [84370.96536131372, 3088.404442426542, 706014.0193473498],
    [49631.784976968134, 3124.9385593960083, f64::INFINITY],
    [22559.642727667477, 3121.4935527724547, f64::INFINITY],
    [31383.214255394556, 30648.150237820228, f64::INFINITY],
    [147369.7926238617, 370274.5780566092, f64::INFINITY],
];

/// Rokoko's private `assign_norm_bounds`.
fn assign_norm_bounds(config: &mut Config, bounds: &[[f64; 3]]) {
    fn rec(config: &mut Config, bounds: &[[f64; 3]], i: usize) -> usize {
        let row = bounds[i].map(|b| b * NORM_MARGIN);
        match config {
            Config::Sumcheck(c) => {
                c.norm_bound = row[0];
                c.most_inner_norm_bound = row[1];
                c.projection_norm_bound = row[2];
                c.next
                    .as_deref_mut()
                    .map_or(i + 1, |n| rec(n, bounds, i + 1))
            }
            Config::Intermediate(c) => {
                c.norm_bound = row[0];
                c.projection_norm_bound = row[1];
                c.next
                    .as_deref_mut()
                    .map_or(i + 1, |n| rec(n, bounds, i + 1))
            }
            Config::Simple(c) => {
                c.witness_norm_bound = row[0];
                c.projection_norm_bound = row[1];
                i + 1
            }
        }
    }
    assert_eq!(rec(config, bounds, 0), bounds.len());
}

pub fn chain(shape: Shape) -> Config {
    let (root, table): (AuxSumcheckConfig, &[[f64; 3]]) = match shape {
        Shape::N14 => (root_small(), &NB_N14),
        Shape::N15 => (root_medium(), &NB_N15),
        Shape::N16 => (root_large(), &NB_N16),
        Shape::N17 => (root_huge(), &NB_N17),
    };
    assert_eq!(root.witness_height * root.witness_width, shape.len());
    let mut config = root.generate_config();
    assign_norm_bounds(&mut config, table);
    config
}

static N14: LazyLock<Config> = LazyLock::new(|| chain(Shape::N14));
static N15: LazyLock<Config> = LazyLock::new(|| chain(Shape::N15));
static N16: LazyLock<Config> = LazyLock::new(|| chain(Shape::N16));
static N17: LazyLock<Config> = LazyLock::new(|| chain(Shape::N17));

pub fn sumcheck(shape: Shape) -> &'static SumcheckConfig {
    let config = match shape {
        Shape::N14 => &*N14,
        Shape::N15 => &*N15,
        Shape::N16 => &*N16,
        Shape::N17 => &*N17,
    };
    match config {
        Config::Sumcheck(c) => c,
        _ => unreachable!(),
    }
}

/// The smallest Module-SIS rank binding a commitment to `m` ring elements of `l2` norm `cap`: a
/// collision is a vector of norm `2 cap`.
pub fn outer_rank(m: usize, cap: f64) -> usize {
    let bound = (2.0 * cap).ceil() as u64;
    (1..=64)
        .find(|&n| {
            estimate_rsis_security(&RSISParameters {
                n,
                m: m as u64,
                length_bound: bound,
            })
            .map_or(false, |r| r.secpar >= SECURITY)
        })
        .unwrap_or_else(|| panic!("no rank up to 64 binds {m} elements at l2 norm {cap}"))
        as usize
}

use crate::challenge::DEFAULT_WEIGHT;
use crate::params::N;
use crate::wire::residue_bits;

pub const BD_CAP: f64 = 4.0;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Dropped {
    primes: Vec<u16>,
    columns: usize,
    dropped_bits: u32,
    top: Vec<u16>,
    digits: Vec<Vec<u16>>,
}

impl Dropped {
    pub fn of(
        primes: Vec<u16>,
        columns: usize,
        dropped_bits: u32,
        top: Vec<u16>,
        digits: Vec<Vec<u16>>,
    ) -> Dropped {
        Dropped {
            primes,
            columns,
            dropped_bits,
            top,
            digits,
        }
    }

    pub fn primes(&self) -> &[u16] {
        &self.primes
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn dropped_bits(&self) -> u32 {
        self.dropped_bits
    }

    pub fn top(&self) -> &[u16] {
        &self.top
    }

    pub fn digits(&self) -> &[Vec<u16>] {
        &self.digits
    }

    pub fn wire_bytes(&self) -> usize {
        bytes(&self.primes, self.columns, self.dropped_bits)
    }
}

pub const fn top_bound(q: u16, dropped_bits: u32) -> u32 {
    ((q as u32 - 1) + (1 << (dropped_bits - 1))) >> dropped_bits
}

pub const fn top_bits(q: u16, dropped_bits: u32) -> u32 {
    u32::BITS - top_bound(q, dropped_bits).leading_zeros()
}

pub fn coefficient_bits(primes: &[u16], dropped_bits: u32) -> u32 {
    top_bits(primes[0], dropped_bits) + primes[1..].iter().map(|&q| residue_bits(q)).sum::<u32>()
}

pub fn bytes(primes: &[u16], columns: usize, dropped_bits: u32) -> usize {
    (columns * N * coefficient_bits(primes, dropped_bits) as usize).div_ceil(8)
}

pub fn cap(columns: usize, dropped_bits: u32) -> u64 {
    let spread = ((1u64 << (2 * dropped_bits)) - 1) as f64 / 12.0;
    (BD_CAP * (N * columns * DEFAULT_WEIGHT) as f64 * spread).ceil() as u64
}

pub fn expected_normsq(columns: usize, dropped_bits: u32) -> f64 {
    let spread = ((1u64 << (2 * dropped_bits)) - 1) as f64 / 12.0;
    (N * columns * DEFAULT_WEIGHT) as f64 * spread
}

mod digits;
mod residual;

pub use digits::*;
pub use residual::*;

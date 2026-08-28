//! Data types. Naming follows rokoko (`RingElement { v, representation }`), adapted to the
//! degree-648 ring, i16 lanes and batches of 32 polynomials.
use crate::params::N;
use crate::rng::Rng;

/// A binary polynomial of degree < 648: coefficient i is bit i (bit i of `bits[i / 64]`).
/// Bits 648..703 must be zero. This is the storage form any producer can write cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BinaryPoly {
    pub bits: [u64; 11],
}

impl BinaryPoly {
    pub const WORDS: usize = 11;
    #[inline]
    pub fn coeff(&self, i: usize) -> u8 {
        ((self.bits[i >> 6] >> (i & 63)) & 1) as u8
    }
    #[inline]
    pub fn set(&mut self, i: usize, b: bool) {
        let m = 1u64 << (i & 63);
        if b {
            self.bits[i >> 6] |= m;
        } else {
            self.bits[i >> 6] &= !m;
        }
    }
    pub fn random(rng: &mut Rng) -> Self {
        let mut bits = [0u64; 11];
        for w in bits.iter_mut() {
            *w = rng.next_u64();
        }
        bits[10] &= (1u64 << (N - 640)) - 1;
        BinaryPoly { bits }
    }
    pub fn to_coeffs(&self) -> [u32; N] {
        let mut c = [0u32; N];
        for i in 0..N {
            c[i] = self.coeff(i) as u32;
        }
        c
    }
}

/// 32 binary polynomials in the kernel-side "nibble-sliced" form:
/// `idx[i][p]` = b_i | b_{i+162} << 1 | b_{i+324} << 2 | b_{i+486} << 3 of polynomial p,
/// i.e. exactly the 4-bit index that the fused levels 0+1 table lookup consumes for coefficient i.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct BinaryBatch32 {
    pub idx: [[u8; 32]; 162],
}

impl BinaryBatch32 {
    pub fn zero() -> Self {
        BinaryBatch32 { idx: [[0u8; 32]; 162] }
    }
    /// Scalar reference conversion (the SIMD transpose lives in `simd`).
    pub fn from_polys_scalar(polys: &[BinaryPoly; 32]) -> Self {
        let mut b = Self::zero();
        for (p, poly) in polys.iter().enumerate() {
            for i in 0..162 {
                b.idx[i][p] = poly.coeff(i)
                    | poly.coeff(i + 162) << 1
                    | poly.coeff(i + 324) << 2
                    | poly.coeff(i + 486) << 3;
            }
        }
        b
    }
    pub fn poly(&self, p: usize) -> BinaryPoly {
        let mut poly = BinaryPoly::default();
        for i in 0..162 {
            let n = self.idx[i][p];
            for j in 0..4 {
                poly.set(i + 162 * j, (n >> j) & 1 == 1);
            }
        }
        poly
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Representation {
    /// Plain coefficients a_0..a_647 of a(X).
    Coefficients,
    /// NTT domain in tree order: v[j] = a(psi^SLOT_EXP[j]) (see `params`), lazily reduced.
    Ntt,
}

/// One ring element, rokoko-style: flat inline array + representation tag. Values are signed
/// 16-bit residues; in NTT form they are only lazily reduced (|v| < 4q at most, see each kernel).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingElement {
    pub v: [i16; N],
    pub representation: Representation,
}

impl RingElement {
    pub fn zero(representation: Representation) -> Self {
        RingElement { v: [0i16; N], representation }
    }
    pub fn from_binary(p: &BinaryPoly) -> Self {
        let mut e = Self::zero(Representation::Coefficients);
        for i in 0..N {
            e.v[i] = p.coeff(i) as i16;
        }
        e
    }
    /// Fully reduced non-negative coefficients in [0, q).
    pub fn normalized(&self, q: u16) -> [u32; N] {
        let mut out = [0u32; N];
        for i in 0..N {
            out[i] = ((self.v[i] as i32).rem_euclid(q as i32)) as u32;
        }
        out
    }
}

/// 32 ring elements in "vertical" layout: `v[j][p]` is coefficient/slot j of polynomial p, so one
/// 512-bit vector holds one slot of all 32 polynomials. 64-byte aligned, 41472 bytes.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Batch32 {
    pub v: [[i16; 32]; N],
    pub representation: Representation,
}

impl Batch32 {
    pub fn zero(representation: Representation) -> Self {
        Batch32 { v: [[0i16; 32]; N], representation }
    }
    pub fn get(&self, p: usize) -> RingElement {
        let mut e = RingElement::zero(self.representation);
        for j in 0..N {
            e.v[j] = self.v[j][p];
        }
        e
    }
    pub fn set(&mut self, p: usize, e: &RingElement) {
        debug_assert_eq!(e.representation, self.representation);
        for j in 0..N {
            self.v[j][p] = e.v[j];
        }
    }
    pub fn from_binary(polys: &[BinaryPoly; 32]) -> Self {
        let mut b = Self::zero(Representation::Coefficients);
        for (p, poly) in polys.iter().enumerate() {
            for j in 0..N {
                b.v[j][p] = poly.coeff(j) as i16;
            }
        }
        b
    }
}

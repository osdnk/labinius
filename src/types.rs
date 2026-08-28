//! Data types. Naming follows rokoko (`RingElement { v, representation }`), adapted to the
//! degree-648 ring, i16 lanes and batches of 32 polynomials.
use crate::params::N;

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
}

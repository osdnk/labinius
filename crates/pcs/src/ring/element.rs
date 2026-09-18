use crate::params::N;
use crate::ring::N162;

// =============================================================================================
// ring elements
// =============================================================================================

/// One element of `R_162 = Z_q[Z]/Phi_243(Z)` for a single prime, in the NTT domain: 162 slots,
/// centered signed residues in `[-(q-1)/2, (q-1)/2]`, slot `s` holding the evaluation at the
/// primitive 243-rd root of unity indexed by [`POW3_SLOT_EXP`]`[s]` (see the module documentation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PowerOfThreeRingElement {
    pub v: [i16; N162],
}

impl PowerOfThreeRingElement {
    pub fn zero() -> Self {
        PowerOfThreeRingElement { v: [0i16; N162] }
    }
    /// The canonical non-negative representatives in `[0, q)`.
    pub fn normalized(&self, q: u16) -> [u32; N162] {
        let mut out = [0u32; N162];
        for s in 0..N162 {
            out[s] = (self.v[s] as i32).rem_euclid(q as i32) as u32;
        }
        out
    }
}

impl Default for PowerOfThreeRingElement {
    fn default() -> Self {
        Self::zero()
    }
}

/// One element of `R_162` given by its residues modulo the key's limbs: `limbs[0]` is the residue
/// modulo the key's base limb and `limbs[1 + i]` the one modulo the key's `i`-th additional
/// [`Modulus`], in the order the key was built with.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PowerOfThreeRingElementWithLimbs {
    pub limbs: Vec<PowerOfThreeRingElement>,
}

impl PowerOfThreeRingElementWithLimbs {
    /// `n` zero limbs.
    pub fn zero(n: usize) -> Self {
        PowerOfThreeRingElementWithLimbs {
            limbs: vec![PowerOfThreeRingElement::zero(); n],
        }
    }
    /// The residue modulo the key's base limb.
    pub fn base(&self) -> &PowerOfThreeRingElement {
        &self.limbs[0]
    }
    /// The residue modulo the key's `i`-th additional limb.
    pub fn additional(&self, i: usize) -> &PowerOfThreeRingElement {
        &self.limbs[1 + i]
    }
    /// Number of limbs (`1 + additional`).
    pub fn len(&self) -> usize {
        self.limbs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.limbs.is_empty()
    }
}

// =============================================================================================
// the matrix
// =============================================================================================

/// A `rows x cols` matrix stored column by column, one column being one commitment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VerticallyAlignedMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> VerticallyAlignedMatrix<T> {
    /// `data` in column-major order: entry `(row, col)` at `data[col * rows + row]`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "column-major data of the wrong length"
        );
        VerticallyAlignedMatrix { rows, cols, data }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) out of range"
        );
        &self.data[col * self.rows + row]
    }
    /// One whole column — one commitment, its `rows` components in order.
    pub fn column(&self, col: usize) -> &[T] {
        assert!(col < self.cols, "column {col} out of range");
        &self.data[col * self.rows..(col + 1) * self.rows]
    }
    pub fn columns(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.rows)
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
    pub fn as_slice(&self) -> &[T] {
        &self.data
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
        RingElement {
            v: [0i16; N],
            representation,
        }
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
/// 512-bit vector holds one slot of all 32 polynomials. 64-byte aligned, 41472 bytes of slots
/// (41536 with the representation tag and the alignment padding).
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Batch32 {
    pub v: [[i16; 32]; N],
    pub representation: Representation,
}

impl Batch32 {
    pub fn zero(representation: Representation) -> Self {
        Batch32 {
            v: [[0i16; 32]; N],
            representation,
        }
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

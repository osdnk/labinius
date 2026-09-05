//! The wire forms of the clear-text round: what a verifier actually receives, and its exact
//! inverse.
//!
//! Two kinds of object travel, and they get two different codes.
//!
//! * **Uniform data is bit-packed.** A commitment slot is a residue modulo its limb, so it
//!   carries `ceil(log2 q)` bits and no more — 12 for 3889, 14 for 9721, 15 for 19441 — and the
//!   slots are written back to back with no per-slot padding. An `F162` is a uniform 162-bit
//!   field element, so the row evaluation is 162 bits per element, contiguously: 5184 bytes for
//!   256 elements rather than 6144. Neither carries a header; both shapes come from the
//!   parameters the verifier already holds. No code can beat those floors on honest data — if
//!   one measures below them, it is a bug, not a win. The unpacker reads sixteen residues at a
//!   time with AVX-512 — a byte permute into dword lanes, a variable shift, a mask — and
//!   range-checks and centres them in the lanes; without it, a bit reader refilled 64 bits at a
//!   time.
//! * **The folded witness is entropy-coded.** `v = sum_j c_j W_j` is a sum of a few hundred
//!   binary columns against short binary challenges, so its coefficients are a discrete
//!   Gaussian of a few tens — sigma 53 to 120 across the crate's limb lists, and 56 at
//!   [`Params::basic`] — well inside the bound `(q-1)/2`, which is 7.8 to 8.8 bits of entropy
//!   against the 16 bits an `i16` spends. [`encode`] measures the distribution of the message it
//!   is given, transmits it in the header, and codes against it with a static rANS.
//!
//! The coder is a 32-bit rANS with 16-bit renormalisation and a 2^12 frequency total. The
//! histogram covers exactly the occupied range `[offset, offset + length)` and is quantised to
//! 4096 by largest remainder with every occupied symbol floored at 1; when the occupied range is
//! wider than the table can hold — an adversarial witness spread over the whole of `[-(q-1)/2,
//! (q-1)/2]` at a large limb — the rarest symbols are folded into an escape symbol and coded as
//! raw bits after it, so any `i16` message encodes. The quantised counts themselves are Elias
//! gamma codes of `count + 1`, which spends one bit on each of the empty symbols in the tails.
//!
//! The message is coded as [`LANES`] interleaved rANS streams: coefficient `i` of the flattened
//! witness belongs to lane `i mod LANES`, and the decoder walks the coefficients in groups of
//! `LANES`, advancing every lane's state once per group, which turns one dependency chain of
//! `n` steps into `LANES` chains of `n / LANES`. All lanes share one word sequence: the encoder
//! runs the groups backwards and pushes its renormalisation words onto a single vector, whose
//! reversal is exactly the order a forward decoder pulls them in, so the wire carries the `LANES`
//! final states (four bytes each, after the histogram) and then the words, with no per-lane
//! lengths. Within a group the steps come in lane order, a lane's raw bits directly after its
//! escape. A decoder with AVX-512 keeps the states in vectors — the slot lookup is a gather on a
//! 4096-entry table, the renormalisation a compare mask and an expand of the next words — and
//! redoes a group lane by lane on the rare escape; the scalar decoder pulls the same layout lane
//! by lane throughout.
//!
//! Everything here is off the hot path: the recursive mode sends `T_Y`, `T_u`, `T_R` and one
//! LaBRADOR proof, which are true wire widths already.
use crate::labrador::PolxBuf;
use crate::ring::{
    PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162,
};
use crate::bd::{self, Dropped};
use crate::fields::scalar::F162;
use crate::params::N;
use crate::scheme::{Commitment, CommitmentValue, Params, RowEvaluation};
use std::sync::Arc;

/// Significant bits of an `F162`.
pub const F162_BITS: u32 = 162;
/// log2 of the rANS frequency total.
const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;
/// Lower end of the rANS state interval; renormalisation moves 16 bits at a time.
const RANS_LOW: u32 = 1 << 16;
const HEADER: usize = 16;
/// A cap on the element count a header may announce, so that a corrupt one cannot ask for an
/// arbitrary allocation before the stream runs out.
const MAX_ELEMENTS: usize = 1 << 24;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireError {
    /// The byte string ran out before the object did.
    Truncated,
    /// The byte string is not a coding of any object of the expected shape.
    Malformed,
}

/// Bits one residue modulo `q` occupies: `ceil(log2 q)`.
pub const fn residue_bits(q: u16) -> u32 {
    u32::BITS - (q as u32 - 1).leading_zeros()
}

// =============================================================================================
// bit stream
// =============================================================================================

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn with_capacity(bytes: usize) -> BitWriter {
        BitWriter {
            bytes: Vec::with_capacity(bytes),
            acc: 0,
            n: 0,
        }
    }

    /// `bits` low bits of `value`, least significant first, `bits <= 32`.
    fn put(&mut self, value: u64, bits: u32) {
        debug_assert!(bits <= 32);
        self.acc |= (value & ((1u64 << bits) - 1)) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// The same for `bits <= 64`.
    fn put_wide(&mut self, value: u64, bits: u32) {
        if bits <= 32 {
            self.put(value, bits);
        } else {
            self.put(value, 32);
            self.put(value >> 32, bits - 32);
        }
    }

    /// Elias gamma of `value >= 1`, least significant bit first inside each field: `k` zeros,
    /// the leading one, then the `k` bits below it.
    fn put_gamma(&mut self, value: u64) {
        let k = 63 - value.leading_zeros();
        self.put(0, k);
        self.put(1, 1);
        self.put_wide(value & ((1u64 << k) - 1), k);
    }

    /// The bytes, the last one zero-padded.
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader {
            bytes,
            pos: 0,
            acc: 0,
            n: 0,
        }
    }

    /// Tops the accumulator up to at least 56 bits from a 64-bit load of the bytes at `pos`;
    /// the bits above the counted ones are the stream's next bytes, which the following refill
    /// ORs in again, identically.
    #[inline]
    fn refill(&mut self) {
        let rest = &self.bytes[self.pos..];
        let word = if rest.len() >= 8 {
            u64::from_le_bytes(rest[..8].try_into().unwrap())
        } else {
            let mut b = [0u8; 8];
            b[..rest.len()].copy_from_slice(rest);
            u64::from_le_bytes(b)
        };
        self.acc |= word << self.n;
        let take = (((63 - self.n) >> 3) as usize).min(rest.len());
        self.pos += take;
        self.n += 8 * take as u32;
    }

    #[inline]
    fn get(&mut self, bits: u32) -> Result<u64, WireError> {
        debug_assert!(bits <= 32);
        if self.n < bits {
            self.refill();
            if self.n < bits {
                return Err(WireError::Truncated);
            }
        }
        let value = self.acc & ((1u64 << bits) - 1);
        self.acc >>= bits;
        self.n -= bits;
        Ok(value)
    }

    fn get_wide(&mut self, bits: u32) -> Result<u64, WireError> {
        if bits <= 32 {
            self.get(bits)
        } else {
            let low = self.get(32)?;
            Ok(low | (self.get(bits - 32)? << 32))
        }
    }

    fn get_gamma(&mut self) -> Result<u64, WireError> {
        let mut k = 0;
        while self.get(1)? == 0 {
            k += 1;
            if k > 32 {
                return Err(WireError::Malformed);
            }
        }
        Ok((1u64 << k) | self.get_wide(k)?)
    }

    /// Bytes consumed, the current partial byte included.
    fn consumed(&self) -> usize {
        self.pos - (self.n / 8) as usize
    }
}

// =============================================================================================
// uniform objects: the commitment matrix and the row evaluation
// =============================================================================================

/// The commitment matrix, `ceil(log2 q)` bits per slot per limb, in the order
/// [`VerticallyAlignedMatrix::iter`] hands the entries out (column by column, four rows each).
/// Header-free: the shape is [`Params`]. Panics on a recursive commitment, which travels as
/// `T_Y`.
pub fn pack_commitment(commitment: &Commitment) -> Vec<u8> {
    let primes = commitment.moduli();
    if let CommitmentValue::Dropped(dropped) = commitment.value() {
        return pack_dropped(dropped);
    }
    let matrix = commitment.matrix();
    let mut w = BitWriter::with_capacity(commitment_bytes(primes, matrix.cols()));
    for element in matrix.iter() {
        for (k, limb) in element.limbs.iter().enumerate() {
            let q = primes[k] as i32;
            let bits = residue_bits(primes[k]);
            for slot in limb.v {
                w.put((slot as i32).rem_euclid(q) as u64, bits);
            }
        }
    }
    w.finish()
}

/// The inverse of [`pack_commitment`], against the parameters the verifier holds.
pub fn unpack_commitment(params: &Params, bytes: &[u8]) -> Result<Commitment, WireError> {
    let primes = params.primes();
    let columns = params.columns();
    if params.dropped_bits() > 0 {
        return unpack_dropped(params, bytes).map(|d| {
            Commitment::of(
                primes,
                columns,
                CommitmentValue::Dropped(std::sync::Arc::new(d)),
            )
        });
    }
    if params.recursion() || bytes.len() != commitment_bytes(&primes, columns) {
        return Err(WireError::Malformed);
    }
    let mut data: Vec<PowerOfThreeRingElementWithLimbs> = (0..4 * columns)
        .map(|_| PowerOfThreeRingElementWithLimbs::zero(primes.len()))
        .collect();
    #[cfg(target_arch = "x86_64")]
    let vectorised = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi");
    #[cfg(not(target_arch = "x86_64"))]
    let vectorised = false;
    if vectorised {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx512::unpack_residues(&primes, bytes, &mut data)?
        };
    } else {
        let mut r = BitReader::new(bytes);
        for element in data.iter_mut() {
            for (limb, &q) in element.limbs.iter_mut().zip(&primes) {
                let bits = residue_bits(q);
                let half = ((q - 1) / 2) as i32;
                for slot in limb.v.iter_mut() {
                    let value = r.get(bits)? as i32;
                    if value >= q as i32 {
                        return Err(WireError::Malformed);
                    }
                    *slot = if value > half {
                        (value - q as i32) as i16
                    } else {
                        value as i16
                    };
                }
            }
        }
    }
    Ok(Commitment::of(
        primes,
        columns,
        CommitmentValue::Matrix(VerticallyAlignedMatrix::new(4, columns, data)),
    ))
}

/// The commitment as tagged bytes, for a transcript that carries the proof itself: the `i16`
/// slots of the matrix in column-major order, the `polx` image of `T_Y`, or the bit-dropped
/// digits, after a one-byte tag and the element count. Wider than [`pack_commitment`] for `T_Y`,
/// which is `LOGQ` bits per coefficient rather than a whole `polx`.
pub fn pack_tagged_commitment(commitment: &Commitment) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + commitment.wire_bytes());
    match &commitment.value {
        CommitmentValue::Matrix(m) => {
            out.push(0);
            out.extend_from_slice(&(m.cols() as u32).to_le_bytes());
            for element in m.iter() {
                for limb in &element.limbs {
                    for slot in limb.v {
                        out.extend_from_slice(&slot.to_le_bytes());
                    }
                }
            }
        }
        CommitmentValue::Recursive(t) => {
            out.push(1);
            out.extend_from_slice(&(t.len() as u32).to_le_bytes());
            out.extend_from_slice(t.as_bytes());
        }
        CommitmentValue::Dropped(d) => {
            out.push(2);
            out.extend_from_slice(&(d.columns() as u32).to_le_bytes());
            out.extend_from_slice(&pack_dropped(d));
        }
    }
    out
}

/// The inverse of [`pack_tagged_commitment`], against the parameters the verifier holds.
pub fn unpack_tagged_commitment(params: &Params, bytes: &[u8]) -> Option<Commitment> {
    let primes = params.primes();
    let (&tag, rest) = bytes.split_first()?;
    if rest.len() < 4
        || (tag == 1) != params.recursion()
        || (tag == 2) != (params.dropped_bits() > 0)
    {
        return None;
    }
    let count = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
    let body = &rest[4..];
    let value = if tag == 2 {
        if count != params.columns() {
            return None;
        }
        CommitmentValue::Dropped(Arc::new(unpack_dropped(params, body).ok()?))
    } else if tag == 0 {
        let slots = 4 * count * primes.len() * N162;
        if count != params.columns() || body.len() != 2 * slots {
            return None;
        }
        let mut slot = body
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]));
        let data = (0..4 * count)
            .map(|_| PowerOfThreeRingElementWithLimbs {
                limbs: (0..primes.len())
                    .map(|_| PowerOfThreeRingElement {
                        v: core::array::from_fn(|_| slot.next().unwrap()),
                    })
                    .collect(),
            })
            .collect();
        CommitmentValue::Matrix(VerticallyAlignedMatrix::new(4, count, data))
    } else {
        CommitmentValue::Recursive(Arc::new(PolxBuf::from_bytes(count, body)?))
    };
    Some(Commitment {
        primes,
        columns: params.columns(),
        value,
    })
}

/// Bytes [`pack_commitment`] produces for a key over `primes` with `columns` columns.
pub fn commitment_bytes(primes: &[u16], columns: usize) -> usize {
    let bits: u32 = primes.iter().map(|&q| residue_bits(q)).sum();
    (4 * columns * N162 * bits as usize).div_ceil(8)
}

pub fn pack_dropped(dropped: &Dropped) -> Vec<u8> {
    let primes = dropped.primes();
    let top = bd::top_bits(primes[0], dropped.dropped_bits());
    let mut w = BitWriter::with_capacity(dropped.wire_bytes());
    for i in 0..dropped.columns() * N {
        w.put(dropped.top()[i] as u64, top);
        for (k, digit) in dropped.digits().iter().enumerate() {
            w.put(digit[i] as u64, residue_bits(primes[k + 1]));
        }
    }
    w.finish()
}

pub fn unpack_dropped(params: &Params, bytes: &[u8]) -> Result<Dropped, WireError> {
    let primes = params.primes();
    let columns = params.columns();
    let dropped_bits = params.dropped_bits();
    if params.recursion()
        || dropped_bits == 0
        || bytes.len() != bd::bytes(&primes, columns, dropped_bits)
    {
        return Err(WireError::Malformed);
    }
    let bits = bd::top_bits(primes[0], dropped_bits);
    let bound = bd::top_bound(primes[0], dropped_bits);
    let count = columns * N;
    let mut top = vec![0u16; count];
    let mut digits: Vec<Vec<u16>> = (1..primes.len()).map(|_| vec![0u16; count]).collect();
    let mut r = BitReader::new(bytes);
    for i in 0..count {
        let value = r.get(bits)? as u32;
        if value > bound {
            return Err(WireError::Malformed);
        }
        top[i] = value as u16;
        for (k, digit) in digits.iter_mut().enumerate() {
            let q = primes[k + 1];
            let value = r.get(residue_bits(q))? as u32;
            if value >= q as u32 {
                return Err(WireError::Malformed);
            }
            digit[i] = value as u16;
        }
    }
    Ok(Dropped::of(primes, columns, dropped_bits, top, digits))
}

/// The row evaluation, 162 bits per element back to back. Header-free: the length is
/// [`Params::columns`].
pub fn pack_row_evaluation(row: &RowEvaluation) -> Vec<u8> {
    pack_f162(row.values())
}

/// The inverse of [`pack_row_evaluation`] for `count` elements.
pub fn unpack_row_evaluation(bytes: &[u8], count: usize) -> Result<RowEvaluation, WireError> {
    Ok(RowEvaluation::of(unpack_f162(bytes, count)?))
}

/// `F162` elements at 162 bits each, contiguously: `ceil(162 n / 8)` bytes.
pub fn pack_f162(values: &[F162]) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(f162_bytes(values.len()));
    for x in values {
        w.put_wide(x.0[0], 64);
        w.put_wide(x.0[1], 64);
        w.put_wide(x.0[2], F162_BITS - 128);
    }
    w.finish()
}

/// The inverse of [`pack_f162`] for `count` elements.
pub fn unpack_f162(bytes: &[u8], count: usize) -> Result<Vec<F162>, WireError> {
    if bytes.len() != f162_bytes(count) {
        return Err(WireError::Malformed);
    }
    let mut r = BitReader::new(bytes);
    (0..count)
        .map(|_| {
            Ok(F162([
                r.get_wide(64)?,
                r.get_wide(64)?,
                r.get_wide(F162_BITS - 128)?,
            ]))
        })
        .collect()
}

/// Bytes [`pack_f162`] produces for `count` elements.
pub fn f162_bytes(count: usize) -> usize {
    (count * F162_BITS as usize).div_ceil(8)
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{residue_bits, WireError};
    use crate::ring::{PowerOfThreeRingElementWithLimbs, N162};
    use core::arch::x86_64::*;

    /// Sixteen `bits`-wide fields starting `phase` bits into a 64-byte load: the byte permute
    /// that brings each field's four bytes into its dword lane, and the shift that then aligns
    /// it.
    #[target_feature(enable = "avx512f")]
    unsafe fn field_lanes(bits: u32, phase: u32) -> (__m512i, __m512i) {
        let mut bytes = [0u8; 64];
        let mut shifts = [0i32; 16];
        for lane in 0..16 {
            let start = phase + lane as u32 * bits;
            for b in 0..4 {
                bytes[4 * lane + b] = (start / 8) as u8 + b as u8;
            }
            shifts[lane] = (start % 8) as i32;
        }
        (
            _mm512_loadu_si512(bytes.as_ptr() as *const __m512i),
            _mm512_loadu_epi32(shifts.as_ptr()),
        )
    }

    /// The residues of every limb of every element out of the bit-packed `bytes`, sixteen
    /// fields per step, range-checked and centred in the lanes.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    pub(super) unsafe fn unpack_residues(
        primes: &[u16],
        bytes: &[u8],
        data: &mut [PowerOfThreeRingElementWithLimbs],
    ) -> Result<(), WireError> {
        const GROUPS: usize = N162.div_ceil(16);
        const TAIL: __mmask16 = (1 << (N162 % 16)) - 1;
        let lanes: Vec<[(__m512i, __m512i); 8]> = primes
            .iter()
            .map(|&q| core::array::from_fn(|phase| field_lanes(residue_bits(q), phase as u32)))
            .collect();
        let mut bad: __mmask16 = 0;
        let mut bit = 0usize;
        for element in data.iter_mut() {
            for (limb, (&q, lanes)) in element.limbs.iter_mut().zip(primes.iter().zip(&lanes)) {
                let bits = residue_bits(q);
                let (permute, shift) = lanes[bit % 8];
                let field = _mm512_set1_epi32((1 << bits) - 1);
                let modulus = _mm512_set1_epi32(q as i32);
                let half = _mm512_set1_epi32(((q - 1) / 2) as i32);
                for g in 0..GROUPS {
                    let at = bit / 8 + 2 * bits as usize * g;
                    let src = if at + 64 <= bytes.len() {
                        _mm512_loadu_si512(bytes.as_ptr().add(at) as *const __m512i)
                    } else {
                        let avail = (1u64 << (bytes.len() - at)) - 1;
                        _mm512_maskz_loadu_epi8(avail, bytes.as_ptr().add(at) as *const i8)
                    };
                    let v = _mm512_and_si512(
                        _mm512_srlv_epi32(_mm512_permutexvar_epi8(permute, src), shift),
                        field,
                    );
                    let live = if g + 1 < GROUPS { 0xFFFF } else { TAIL };
                    bad |= _mm512_mask_cmpge_epu32_mask(live, v, modulus);
                    let centred =
                        _mm512_mask_sub_epi32(v, _mm512_cmpgt_epu32_mask(v, half), v, modulus);
                    _mm256_mask_storeu_epi16(
                        limb.v.as_mut_ptr().add(16 * g),
                        live,
                        _mm512_cvtepi32_epi16(centred),
                    );
                }
                bit += N162 * bits as usize;
            }
        }
        if bad != 0 {
            return Err(WireError::Malformed);
        }
        Ok(())
    }
}

mod rans;
pub use rans::*;

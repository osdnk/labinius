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
//!   one measures below them, it is a bug, not a win.
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
//! Everything here is off the hot path: the recursive mode sends `T_Y`, `T_u`, `T_R` and one
//! LaBRADOR proof, which are true wire widths already.
use crate::api::{
    PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162,
};
use crate::fields::scalar::F162;
use crate::params::N;
use crate::scheme::{Commitment, CommitmentValue, FoldedWitness, Params, RowEvaluation};
use crate::types::{Representation, RingElement};

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

    fn get(&mut self, bits: u32) -> Result<u64, WireError> {
        debug_assert!(bits <= 32);
        while self.n < bits {
            let byte = *self.bytes.get(self.pos).ok_or(WireError::Truncated)? as u64;
            self.pos += 1;
            self.acc |= byte << self.n;
            self.n += 8;
        }
        let mask = (1u64 << bits) - 1;
        let value = self.acc & mask;
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
        self.pos
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
    if params.recursion || bytes.len() != commitment_bytes(&primes, columns) {
        return Err(WireError::Malformed);
    }
    let mut r = BitReader::new(bytes);
    let mut data = Vec::with_capacity(4 * columns);
    for _ in 0..4 * columns {
        let mut limbs = Vec::with_capacity(primes.len());
        for &q in &primes {
            let bits = residue_bits(q);
            let half = ((q - 1) / 2) as i32;
            let mut element = PowerOfThreeRingElement::zero();
            for slot in element.v.iter_mut() {
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
            limbs.push(element);
        }
        data.push(PowerOfThreeRingElementWithLimbs { limbs });
    }
    Ok(Commitment::of(
        primes,
        columns,
        CommitmentValue::Matrix(VerticallyAlignedMatrix::new(4, columns, data)),
    ))
}

/// Bytes [`pack_commitment`] produces for a key over `primes` with `columns` columns.
pub fn commitment_bytes(primes: &[u16], columns: usize) -> usize {
    let bits: u32 = primes.iter().map(|&q| residue_bits(q)).sum();
    (4 * columns * N162 * bits as usize).div_ceil(8)
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

// =============================================================================================
// the folded witness: a static rANS over a transmitted histogram
// =============================================================================================

/// The folded witness, entropy-coded against its own histogram. `base_q` is the modulus its
/// coefficients are centered modulo; it is carried in the header, so [`decode`] needs nothing
/// but the bytes.
pub fn encode(folded: &FoldedWitness, base_q: u16) -> Vec<u8> {
    let elements = folded.elements();
    let coefficients = elements.len() * N;
    let representation = elements
        .first()
        .map(|e| e.representation)
        .unwrap_or(Representation::Coefficients);
    let mut out = Vec::with_capacity(HEADER + coefficients);
    out.push(1);
    out.push(u8::from(representation == Representation::Ntt));
    out.extend_from_slice(&base_q.to_le_bytes());
    out.extend_from_slice(&(elements.len() as u32).to_le_bytes());
    if coefficients == 0 {
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        return out;
    }

    let low = elements.iter().flat_map(|e| e.v).min().unwrap() as i32;
    let high = elements.iter().flat_map(|e| e.v).max().unwrap() as i32;
    let length = (high - low + 1) as usize;
    out.extend_from_slice(&low.to_le_bytes());
    out.extend_from_slice(&(length as u32).to_le_bytes());
    debug_assert_eq!(out.len(), HEADER);

    let mut counts = vec![0u64; length + 1];
    for e in elements {
        for &c in &e.v {
            counts[(c as i32 - low) as usize] += 1;
        }
    }
    let frequency = quantize(&counts);
    let mut w = BitWriter::with_capacity(length / 4 + 16);
    for &f in &frequency {
        w.put_gamma(f as u64 + 1);
    }
    out.extend_from_slice(&w.finish());

    let start = starts(&frequency);
    let escape = length;
    let raw = raw_bits(length);
    let mut x = RANS_LOW;
    let mut words: Vec<u16> = Vec::with_capacity(coefficients / 2 + 4);
    for e in elements.iter().rev() {
        for &c in e.v.iter().rev() {
            let s = (c as i32 - low) as usize;
            if frequency[s] == 0 {
                put_bits(&mut x, &mut words, s as u32, raw);
                put_symbol(&mut x, &mut words, start[escape], frequency[escape]);
            } else {
                put_symbol(&mut x, &mut words, start[s], frequency[s]);
            }
        }
    }
    words.push(x as u16);
    words.push((x >> 16) as u16);
    words.reverse();
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

/// The exact inverse of [`encode`].
pub fn decode(bytes: &[u8]) -> Result<FoldedWitness, WireError> {
    if bytes.len() < HEADER {
        return Err(WireError::Truncated);
    }
    if bytes[0] != 1 || bytes[1] > 1 {
        return Err(WireError::Malformed);
    }
    let representation = if bytes[1] == 1 {
        Representation::Ntt
    } else {
        Representation::Coefficients
    };
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let low = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let length = u32::from_le_bytes(bytes[12..HEADER].try_into().unwrap()) as usize;
    if count == 0 {
        return if bytes.len() == HEADER && length == 0 {
            Ok(FoldedWitness::of(Vec::new()))
        } else {
            Err(WireError::Malformed)
        };
    }
    if length == 0 || length > 1 << 16 || count > MAX_ELEMENTS {
        return Err(WireError::Malformed);
    }

    let mut r = BitReader::new(&bytes[HEADER..]);
    let mut frequency = Vec::with_capacity(length + 1);
    let mut total = 0u64;
    for _ in 0..length + 1 {
        let f = r.get_gamma()? - 1;
        total += f;
        if total > SCALE as u64 {
            return Err(WireError::Malformed);
        }
        frequency.push(f as u32);
    }
    if total != SCALE as u64 {
        return Err(WireError::Malformed);
    }
    let start = starts(&frequency);
    let mut symbol = vec![0u32; SCALE as usize];
    for (s, &f) in frequency.iter().enumerate() {
        for slot in start[s]..start[s] + f {
            symbol[slot as usize] = s as u32;
        }
    }

    let stream = &bytes[HEADER + r.consumed()..];
    if stream.len() % 2 != 0 || stream.len() < 4 {
        return Err(WireError::Malformed);
    }
    let mut words = stream
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]));
    let mut x = ((words.next().unwrap() as u32) << 16) | words.next().unwrap() as u32;
    let escape = length as u32;
    let raw = raw_bits(length);
    let mut elements = Vec::with_capacity(count);
    for _ in 0..count {
        let mut e = RingElement::zero(representation);
        for slot in e.v.iter_mut() {
            let mut s = get_symbol(&mut x, &mut words, &symbol, &start, &frequency)?;
            if s == escape {
                s = get_bits(&mut x, &mut words, raw)?;
                if s as usize >= length {
                    return Err(WireError::Malformed);
                }
            }
            let value = low + s as i32;
            *slot = i16::try_from(value).map_err(|_| WireError::Malformed)?;
        }
        elements.push(e);
    }
    if words.next().is_some() {
        return Err(WireError::Malformed);
    }
    Ok(FoldedWitness::of(elements))
}

/// Bits one out-of-histogram symbol index spends after the escape.
fn raw_bits(length: usize) -> u32 {
    usize::BITS - (length - 1).leading_zeros()
}

/// Exclusive prefix sums of the frequency table.
fn starts(frequency: &[u32]) -> Vec<u32> {
    let mut start = Vec::with_capacity(frequency.len());
    let mut acc = 0;
    for &f in frequency {
        start.push(acc);
        acc += f;
    }
    start
}

/// The raw counts scaled to a total of exactly `SCALE`: every occupied symbol keeps at least one
/// slot, the rest is shared out by largest remainder, and — when the occupied range does not fit
/// — the rarest symbols are folded into the escape symbol at the end of the table.
fn quantize(counts: &[u64]) -> Vec<u32> {
    let escape = counts.len() - 1;
    let mut w = counts.to_vec();
    let mut occupied: Vec<usize> = (0..escape).filter(|&i| w[i] > 0).collect();
    if occupied.len() > SCALE as usize - 1 {
        occupied.sort_by(|&a, &b| w[b].cmp(&w[a]).then(a.cmp(&b)));
        for &i in &occupied[SCALE as usize - 1..] {
            w[escape] += w[i];
            w[i] = 0;
        }
    }
    let total: u64 = w.iter().sum();
    let occupied: Vec<usize> = (0..w.len()).filter(|&i| w[i] > 0).collect();
    let mut frequency = vec![0u32; w.len()];
    if occupied.is_empty() {
        return frequency;
    }
    let spare = SCALE as u64 - occupied.len() as u64;
    let mut used = 0u64;
    let mut remainder: Vec<(u64, usize)> = Vec::with_capacity(occupied.len());
    for &i in &occupied {
        let share = w[i] * spare / total;
        frequency[i] = 1 + share as u32;
        used += share;
        remainder.push((w[i] * spare - share * total, i));
    }
    remainder.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, i) in remainder.iter().take((spare - used) as usize) {
        frequency[i] += 1;
    }
    frequency
}

fn put_symbol(x: &mut u32, words: &mut Vec<u16>, start: u32, frequency: u32) {
    debug_assert!(frequency > 0);
    if *x as u64 >= (frequency as u64) << (32 - SCALE_BITS) {
        words.push(*x as u16);
        *x >>= 16;
    }
    *x = ((*x / frequency) << SCALE_BITS) + (*x % frequency) + start;
}

fn put_bits(x: &mut u32, words: &mut Vec<u16>, value: u32, bits: u32) {
    if bits == 0 {
        return;
    }
    debug_assert!(bits <= 16);
    if *x as u64 >= 1u64 << (32 - bits) {
        words.push(*x as u16);
        *x >>= 16;
    }
    *x = (*x << bits) | value;
}

fn get_symbol(
    x: &mut u32,
    words: &mut impl Iterator<Item = u16>,
    symbol: &[u32],
    start: &[u32],
    frequency: &[u32],
) -> Result<u32, WireError> {
    let slot = *x & (SCALE - 1);
    let s = symbol[slot as usize];
    *x = frequency[s as usize] * (*x >> SCALE_BITS) + slot - start[s as usize];
    if *x < RANS_LOW {
        *x = (*x << 16) | words.next().ok_or(WireError::Truncated)? as u32;
    }
    Ok(s)
}

fn get_bits(
    x: &mut u32,
    words: &mut impl Iterator<Item = u16>,
    bits: u32,
) -> Result<u32, WireError> {
    if bits == 0 {
        return Ok(0);
    }
    let value = *x & ((1 << bits) - 1);
    *x >>= bits;
    if *x < RANS_LOW {
        *x = (*x << 16) | words.next().ok_or(WireError::Truncated)? as u32;
    }
    Ok(value)
}

/// The zeroth-order entropy of a folded witness in bytes: what a perfect coder against the exact
/// per-message distribution would spend, and what [`encode`] is measured against.
pub fn entropy_bytes(folded: &FoldedWitness) -> f64 {
    let mut counts = std::collections::BTreeMap::new();
    let mut total = 0u64;
    for e in folded.elements() {
        for &c in &e.v {
            *counts.entry(c).or_insert(0u64) += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let bits: f64 = counts
        .values()
        .map(|&n| {
            let p = n as f64 / total as f64;
            -(n as f64) * p.log2()
        })
        .sum();
    bits / 8.0
}

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
use crate::api::{PowerOfThreeRingElementWithLimbs, VerticallyAlignedMatrix, N162};
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
// the folded witness: interleaved static rANS over a transmitted histogram
// =============================================================================================

/// Interleaved rANS states; coefficient `i` of the flattened witness is coded by lane
/// `i mod LANES`.
pub const LANES: usize = 64;

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

    let (mut low, mut high) = (i16::MAX, i16::MIN);
    for e in elements {
        for &c in &e.v {
            low = low.min(c);
            high = high.max(c);
        }
    }
    let low = low as i32;
    let length = (high as i32 - low + 1) as usize;
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
    let symbols: Vec<EncoderSymbol> = frequency
        .iter()
        .zip(&start)
        .map(|(&f, &s)| EncoderSymbol::new(f, s))
        .collect();
    let escaped: u64 = (0..length)
        .filter(|&s| frequency[s] == 0)
        .map(|s| counts[s])
        .sum();
    let raw = raw_bits(length);
    let mut x = [RANS_LOW; LANES];
    let mut words = vec![0u16; coefficients + escaped as usize + 1];
    let mut top = 0usize;
    let mut group = [0i16; LANES];
    for g in (0..coefficients.div_ceil(LANES)).rev() {
        let lanes = LANES.min(coefficients - g * LANES);
        gather(elements, g * LANES, &mut group[..lanes]);
        for k in (0..lanes).rev() {
            let s = (group[k] as i32 - low) as usize;
            let symbol = &symbols[s];
            if symbol.frequency == 0 {
                put_bits(&mut x[k], &mut words, &mut top, s as u32, raw);
                put_symbol(&mut x[k], &mut words, &mut top, &symbols[length]);
            } else {
                put_symbol(&mut x[k], &mut words, &mut top, symbol);
            }
        }
    }
    for state in x {
        out.extend_from_slice(&state.to_le_bytes());
    }
    let base = out.len();
    out.resize(base + 2 * top, 0);
    for (i, &word) in words[..top].iter().rev().enumerate() {
        out[base + 2 * i..base + 2 * i + 2].copy_from_slice(&word.to_le_bytes());
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
    if length == 0
        || length > 1 << 16
        || count > MAX_ELEMENTS
        || low < i16::MIN as i32
        || low + length as i32 - 1 > i16::MAX as i32
    {
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
    let mut table = vec![0u64; SCALE as usize];
    for (s, &f) in frequency.iter().enumerate() {
        for slot in start[s]..start[s] + f {
            table[slot as usize] =
                f as u64 | (((slot - start[s]) as u64) << 16) | ((s as u64) << 32);
        }
    }

    let rest = &bytes[HEADER + r.consumed()..];
    if rest.len() < 4 * LANES {
        return Err(WireError::Truncated);
    }
    let mut x = [0u32; LANES];
    for (k, state) in rest[..4 * LANES].chunks_exact(4).enumerate() {
        x[k] = u32::from_le_bytes(state.try_into().unwrap());
        if x[k] < RANS_LOW {
            return Err(WireError::Malformed);
        }
    }
    let stream = &rest[4 * LANES..];
    if stream.len() % 2 != 0 {
        return Err(WireError::Malformed);
    }
    let coefficients = count * N;
    let mut elements = vec![RingElement::zero(representation); count];
    let decoder = Decoder {
        table: &table,
        stream,
        raw: raw_bits(length),
        length: length as u32,
        low,
    };
    let mut pos = 0usize;
    let full = coefficients / LANES;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if LANES % 16 == 0
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
    {
        unsafe { avx512::decode_groups(&decoder, &mut x, &mut pos, full, &mut elements)? };
        done = full;
    }
    let mut group = [0i16; LANES];
    for g in done..coefficients.div_ceil(LANES) {
        let lanes = LANES.min(coefficients - g * LANES);
        decoder.group(&mut x, &mut pos, &mut group[..lanes])?;
        scatter(&mut elements, g * LANES, &group[..lanes]);
    }
    if 2 * pos != stream.len() || x.iter().any(|&state| state != RANS_LOW) {
        return Err(WireError::Malformed);
    }
    Ok(FoldedWitness::of(elements))
}

fn gather(elements: &[RingElement], i: usize, values: &mut [i16]) {
    let (e, j) = (i / N, i % N);
    let first = values.len().min(N - j);
    let rest = values.len() - first;
    values[..first].copy_from_slice(&elements[e].v[j..j + first]);
    if rest > 0 {
        values[first..].copy_from_slice(&elements[e + 1].v[..rest]);
    }
}

fn scatter(elements: &mut [RingElement], i: usize, values: &[i16]) {
    let (e, j) = (i / N, i % N);
    let first = values.len().min(N - j);
    let rest = values.len() - first;
    elements[e].v[j..j + first].copy_from_slice(&values[..first]);
    if rest > 0 {
        elements[e + 1].v[..rest].copy_from_slice(&values[first..]);
    }
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

/// A symbol as the encoder divides by it: `x / f = (x * reciprocal) >> 64` exactly for
/// `x < 2^32`, `f >= 2`; `f = 1` gets the all-ones reciprocal, whose quotient is one short, and a
/// bias of `start + SCALE - 1` puts the step back. A folded symbol keeps frequency 0.
struct EncoderSymbol {
    reciprocal: u64,
    frequency: u32,
    bias: u32,
}

impl EncoderSymbol {
    fn new(frequency: u32, start: u32) -> EncoderSymbol {
        match frequency {
            0 => EncoderSymbol {
                reciprocal: 0,
                frequency: 0,
                bias: 0,
            },
            1 => EncoderSymbol {
                reciprocal: u64::MAX,
                frequency: 1,
                bias: start + SCALE - 1,
            },
            _ => EncoderSymbol {
                reciprocal: u64::MAX / frequency as u64 + 1,
                frequency,
                bias: start,
            },
        }
    }
}

/// Emits the low word of `x` when it must, branch-free: the word is always written at `top`,
/// which only advances on a renormalisation.
#[inline(always)]
fn renormalise(x: &mut u32, words: &mut [u16], top: &mut usize, needed: bool) {
    debug_assert!(*top < words.len());
    unsafe { *words.get_unchecked_mut(*top) = *x as u16 };
    *top += needed as usize;
    *x = if needed { *x >> 16 } else { *x };
}

#[inline(always)]
fn put_symbol(x: &mut u32, words: &mut [u16], top: &mut usize, symbol: &EncoderSymbol) {
    renormalise(x, words, top, *x >> (32 - SCALE_BITS) >= symbol.frequency);
    let q = ((*x as u128 * symbol.reciprocal as u128) >> 64) as u32;
    *x = (q << SCALE_BITS) + (*x - q * symbol.frequency) + symbol.bias;
}

#[inline(always)]
fn put_bits(x: &mut u32, words: &mut [u16], top: &mut usize, value: u32, bits: u32) {
    if bits == 0 {
        return;
    }
    debug_assert!(bits <= 16);
    renormalise(x, words, top, *x >> (32 - bits) != 0);
    *x = (*x << bits) | value;
}

/// The decoder's static side: the per-slot table `frequency | bias << 16 | symbol << 32`, the
/// shared word stream, and the header's constants; the escape symbol is `length`.
struct Decoder<'a> {
    table: &'a [u64],
    stream: &'a [u8],
    raw: u32,
    length: u32,
    low: i32,
}

impl Decoder<'_> {
    /// Word `pos` of the stream, or zero past its end; the group that read past the end is
    /// refused before its output is used.
    #[inline(always)]
    fn word(&self, pos: usize) -> u32 {
        match self.stream.get(2 * pos..2 * pos + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]) as u32,
            None => 0,
        }
    }

    #[inline(always)]
    fn words(&self) -> usize {
        self.stream.len() / 2
    }

    #[inline(always)]
    fn get_symbol(&self, x: &mut u32, pos: &mut usize) -> u32 {
        let e = self.table[(*x & (SCALE - 1)) as usize];
        let y = (e as u32 & 0xFFFF) * (*x >> SCALE_BITS) + ((e >> 16) as u32 & 0xFFFF);
        let w = self.word(*pos);
        let renormalise = y < RANS_LOW;
        *x = if renormalise { (y << 16) | w } else { y };
        *pos += renormalise as usize;
        (e >> 32) as u32
    }

    #[inline(always)]
    fn get_bits(&self, x: &mut u32, pos: &mut usize) -> u32 {
        if self.raw == 0 {
            return 0;
        }
        let value = *x & ((1 << self.raw) - 1);
        let y = *x >> self.raw;
        let w = self.word(*pos);
        let renormalise = y < RANS_LOW;
        *x = if renormalise { (y << 16) | w } else { y };
        *pos += renormalise as usize;
        value
    }

    #[inline(always)]
    fn group(
        &self,
        x: &mut [u32; LANES],
        pos: &mut usize,
        out: &mut [i16],
    ) -> Result<(), WireError> {
        for (k, slot) in out.iter_mut().enumerate() {
            let mut s = self.get_symbol(&mut x[k], pos);
            if s == self.length {
                s = self.get_bits(&mut x[k], pos);
                if s >= self.length {
                    return Err(WireError::Malformed);
                }
            }
            *slot = (self.low + s as i32) as i16;
        }
        if *pos > self.words() {
            return Err(WireError::Truncated);
        }
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{residue_bits, scatter, Decoder, WireError, LANES, RANS_LOW, SCALE, SCALE_BITS};
    use crate::api::{PowerOfThreeRingElementWithLimbs, N162};
    use crate::params::N;
    use crate::types::RingElement;
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

    const VECTORS: usize = LANES / 16;

    /// `groups` full groups: sixteen states per vector, the slot lookup two eight-lane gathers
    /// on the table, the renormalisation a compare mask and an expand of the next sixteen words.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(super) unsafe fn decode_groups(
        d: &Decoder,
        x: &mut [u32; LANES],
        pos: &mut usize,
        groups: usize,
        elements: &mut [RingElement],
    ) -> Result<(), WireError> {
        let table = d.table.as_ptr() as *const i64;
        let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        let slot_mask = _mm512_set1_epi32(SCALE as i32 - 1);
        let half_mask = _mm512_set1_epi32(0xFFFF);
        let low_state = _mm512_set1_epi32(RANS_LOW as i32);
        let escape = _mm512_set1_epi32(d.length as i32);
        let low = _mm512_set1_epi32(d.low);
        let words = d.words();
        let mut xv = [_mm512_setzero_si512(); VECTORS];
        for (v, chunk) in x.chunks_exact(16).enumerate() {
            xv[v] = _mm512_loadu_epi32(chunk.as_ptr() as *const i32);
        }
        let mut out = [0i16; LANES];
        for g in 0..groups {
            let (e, j) = ((g * LANES) / N, (g * LANES) % N);
            let straddles = j + LANES > N;
            let dst = if straddles {
                out.as_mut_ptr()
            } else {
                elements[e].v.as_mut_ptr().add(j)
            };
            let before = (xv, *pos);
            let mut escaped = false;
            for v in 0..VECTORS {
                let slot = _mm512_and_si512(xv[v], slot_mask);
                let e_lo = _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(slot), table);
                let e_hi = _mm512_i32gather_epi64::<8>(_mm512_extracti64x4_epi64::<1>(slot), table);
                let fb = _mm512_permutex2var_epi32(e_lo, even, e_hi);
                let s = _mm512_permutex2var_epi32(e_lo, odd, e_hi);
                let y = _mm512_add_epi32(
                    _mm512_mullo_epi32(
                        _mm512_and_si512(fb, half_mask),
                        _mm512_srli_epi32::<{ SCALE_BITS }>(xv[v]),
                    ),
                    _mm512_srli_epi32::<16>(fb),
                );
                let renormalise = _mm512_cmplt_epu32_mask(y, low_state);
                let at = d.stream.as_ptr().add(2 * *pos);
                let next = if *pos + 16 <= words {
                    _mm256_loadu_si256(at as *const __m256i)
                } else {
                    let avail = (1u32 << words.saturating_sub(*pos)) - 1;
                    _mm256_maskz_loadu_epi16(avail as __mmask16, at as *const i16)
                };
                let w = _mm512_maskz_expand_epi32(renormalise, _mm512_cvtepu16_epi32(next));
                xv[v] = _mm512_or_si512(_mm512_mask_slli_epi32::<16>(y, renormalise, y), w);
                *pos += renormalise.count_ones() as usize;
                _mm256_storeu_si256(
                    dst.add(16 * v) as *mut __m256i,
                    _mm512_cvtepi32_epi16(_mm512_add_epi32(s, low)),
                );
                escaped |= _mm512_cmpeq_epi32_mask(s, escape) != 0;
            }
            if escaped || *pos > words {
                (xv, *pos) = before;
                for v in 0..VECTORS {
                    _mm512_storeu_epi32(x.as_mut_ptr().add(16 * v) as *mut i32, xv[v]);
                }
                d.group(x, pos, &mut out)?;
                for v in 0..VECTORS {
                    xv[v] = _mm512_loadu_epi32(x.as_ptr().add(16 * v) as *const i32);
                }
                scatter(elements, g * LANES, &out);
            } else if straddles {
                scatter(elements, g * LANES, &out);
            }
        }
        for v in 0..VECTORS {
            _mm512_storeu_epi32(x.as_mut_ptr().add(16 * v) as *mut i32, xv[v]);
        }
        Ok(())
    }
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

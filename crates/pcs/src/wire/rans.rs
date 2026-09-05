use super::{BitReader, BitWriter, WireError, HEADER, MAX_ELEMENTS, RANS_LOW, SCALE, SCALE_BITS};
use crate::params::N;
use crate::ring::{Representation, RingElement};
use crate::scheme::FoldedWitness;

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
    use super::{scatter, Decoder, WireError, LANES, RANS_LOW, SCALE, SCALE_BITS};
    use crate::params::N;
    use crate::ring::RingElement;
    use core::arch::x86_64::*;
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

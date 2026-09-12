//! The Keccak-256, SHA-256 and BLAKE3 example circuits of binius64 and their witnesses, built
//! against the upstream crates the way `binius_examples` builds them: the message as `inout`
//! words, the digest as more of them, and one `assert_eq` per digest word against the gadget's
//! output.
use binius_circuits::blake3::blake3_fixed;
use binius_circuits::keccak::{fixed_length::keccak256, ref_keccak_f1600, RATE_BYTES};
use binius_circuits::sha256::{compress::ref_compress, sha256_fixed};
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_core::word::Word;
use binius_frontend::{Circuit as FrontendCircuit, CircuitBuilder, Wire};

use labinius::Suite;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hash {
    Keccak256,
    Sha256,
    Blake3,
}

/// Per hash and suite, the longest message whose circuit has at most `2^(witness_log_len + 1)`
/// private words; one more byte tips it over. `examples/words.rs` measures a candidate.
pub const MESSAGE_LEN: [[usize; 4]; 3] = [
    [118727, 475319, 1901415, 7605663],
    [72567, 290487, 1162039, 4648183],
    [123840, 495488, 1982120, 7928448],
];

impl Hash {
    pub const ALL: [Hash; 3] = [Hash::Keccak256, Hash::Sha256, Hash::Blake3];

    pub const fn name(self) -> &'static str {
        match self {
            Hash::Keccak256 => "keccak-256",
            Hash::Sha256 => "sha-256",
            Hash::Blake3 => "blake3",
        }
    }

    pub fn message_len(self, suite: &Suite) -> usize {
        let len = MESSAGE_LEN[self as usize][suite.index()];
        assert!(
            len != 0,
            "no message length is measured for this hash at this suite"
        );
        len
    }

    pub const fn compressions(self, len_bytes: usize) -> usize {
        match self {
            Hash::Keccak256 => (len_bytes + 1).div_ceil(RATE_BYTES),
            Hash::Sha256 => (len_bytes + 9).div_ceil(64),
            Hash::Blake3 => len_bytes.div_ceil(64) + len_bytes.div_ceil(1024) - 1,
        }
    }

    pub const fn unit(self) -> &'static str {
        match self {
            Hash::Keccak256 => "permutations",
            Hash::Sha256 | Hash::Blake3 => "compressions",
        }
    }

    const fn word(self) -> (usize, bool) {
        match self {
            Hash::Keccak256 => (8, false),
            Hash::Sha256 => (4, true),
            Hash::Blake3 => (4, false),
        }
    }
}

/// The compiled circuit together with the wires its witness is filled through.
pub struct Circuit {
    hash: Hash,
    circuit: FrontendCircuit,
    message: Vec<Wire>,
    digest: Vec<Wire>,
    len_bytes: usize,
}

impl Circuit {
    pub fn new(hash: Hash, len_bytes: usize) -> Circuit {
        let builder = CircuitBuilder::new();
        let (width, _) = hash.word();
        let message: Vec<Wire> = (0..len_bytes.div_ceil(width))
            .map(|_| builder.add_inout())
            .collect();
        let computed: Vec<Wire> = match hash {
            Hash::Keccak256 => keccak256(&builder, &message, len_bytes).to_vec(),
            Hash::Sha256 => sha256_fixed(&builder, &message, len_bytes).to_vec(),
            Hash::Blake3 => blake3_fixed(&builder, &message, len_bytes).to_vec(),
        };
        let digest: Vec<Wire> = computed.iter().map(|_| builder.add_inout()).collect();
        for (i, (computed, digest)) in computed.iter().zip(&digest).enumerate() {
            builder.assert_eq(format!("digest[{i}]"), *computed, *digest);
        }
        Circuit {
            hash,
            circuit: builder.build(),
            message,
            digest,
            len_bytes,
        }
    }

    pub const fn hash(&self) -> Hash {
        self.hash
    }

    pub fn constraint_system(&self) -> &ConstraintSystem {
        self.circuit.constraint_system()
    }

    /// The satisfying witness for `message`, whose length must be the circuit's.
    pub fn witness(&self, message: &[u8]) -> ValueVec {
        assert_eq!(
            message.len(),
            self.len_bytes,
            "the message length is fixed by the circuit"
        );
        let mut filler = self.circuit.new_witness_filler();
        let (width, big_endian) = self.hash.word();
        let digest = match self.hash {
            Hash::Keccak256 => keccak256_digest(message),
            Hash::Sha256 => sha256_digest(message),
            Hash::Blake3 => *blake3::hash(message).as_bytes(),
        };
        for (wire, word) in self.message.iter().zip(pack(message, width, big_endian)) {
            filler[*wire] = Word(word);
        }
        for (wire, word) in self.digest.iter().zip(pack(&digest, width, big_endian)) {
            filler[*wire] = Word(word);
        }
        self.circuit
            .populate_wire_witness(&mut filler)
            .expect("the witness satisfies the circuit");
        filler.into_value_vec()
    }
}

fn pack(bytes: &[u8], width: usize, big_endian: bool) -> Vec<u64> {
    (0..bytes.len().div_ceil(width))
        .map(|i| {
            let mut buf = [0u8; 8];
            let chunk = &bytes[width * i..(width * (i + 1)).min(bytes.len())];
            buf[..chunk.len()].copy_from_slice(chunk);
            let word = u64::from_le_bytes(buf);
            if big_endian {
                word.swap_bytes() >> (8 * (8 - width))
            } else {
                word
            }
        })
        .collect()
}

/// Keccak-256 out of circuit, on [`ref_keccak_f1600`]: the sponge at rate 136 with `pad10*1`.
fn keccak256_digest(message: &[u8]) -> [u8; 32] {
    fn absorb(state: &mut [u64; 25], block: &[u8]) {
        for (i, lane) in block.chunks_exact(8).enumerate() {
            state[i] ^= u64::from_le_bytes(lane.try_into().unwrap());
        }
        ref_keccak_f1600(state);
    }
    let mut state = [0u64; 25];
    let mut blocks = message.chunks_exact(RATE_BYTES);
    for block in blocks.by_ref() {
        absorb(&mut state, block);
    }
    let tail = blocks.remainder();
    let mut block = [0u8; RATE_BYTES];
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] |= 0x01;
    block[RATE_BYTES - 1] |= 0x80;
    absorb(&mut state, &block);
    let mut out = [0u8; 32];
    for (i, lane) in state[..4].iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

fn sha256_digest(message: &[u8]) -> [u8; 32] {
    const IV: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut state = IV;
    let mut compress = |block: &[u8]| {
        state = ref_compress(
            state,
            core::array::from_fn(|i| {
                u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
            }),
        );
    };
    let mut blocks = message.chunks_exact(64);
    for block in blocks.by_ref() {
        compress(block);
    }
    let tail = blocks.remainder();
    let mut block = [0u8; 128];
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] = 0x80;
    let padded = if tail.len() < 56 { 64 } else { 128 };
    block[padded - 8..padded].copy_from_slice(&(8 * message.len() as u64).to_be_bytes());
    for block in block[..padded].chunks_exact(64) {
        compress(block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

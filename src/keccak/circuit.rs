//! The Keccak-256 example circuit of binius64 and its witness, built against the upstream crates
//! the way `binius_examples::circuits::keccak::KeccakExample` builds it: the message as `inout`
//! words, the digest as four more, and one `assert_eq` per digest word against
//! [`keccak256`](binius_circuits::keccak::fixed_length::keccak256)'s output.
use binius_circuits::keccak::{RATE_BYTES, fixed_length::keccak256, ref_keccak_f1600};
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_core::word::Word;
use binius_frontend::{Circuit as FrontendCircuit, CircuitBuilder, Wire};

/// The compiled circuit together with the wires its witness is filled through.
pub struct Circuit {
    circuit: FrontendCircuit,
    message: Vec<Wire>,
    digest: [Wire; 4],
    len_bytes: usize,
}

impl Circuit {
    /// The fixed-length Keccak-256 circuit over a `len_bytes`-byte message.
    pub fn new(len_bytes: usize) -> Circuit {
        let builder = CircuitBuilder::new();
        let message: Vec<Wire> = (0..len_bytes.div_ceil(8)).map(|_| builder.add_inout()).collect();
        let computed = keccak256(&builder, &message, len_bytes);
        let digest: [Wire; 4] = core::array::from_fn(|_| builder.add_inout());
        for i in 0..4 {
            builder.assert_eq(format!("digest[{i}]"), computed[i], digest[i]);
        }
        Circuit {
            circuit: builder.build(),
            message,
            digest,
            len_bytes,
        }
    }

    pub fn constraint_system(&self) -> &ConstraintSystem {
        self.circuit.constraint_system()
    }

    /// The satisfying witness for `message`, whose length must be the circuit's.
    pub fn witness(&self, message: &[u8]) -> ValueVec {
        assert_eq!(message.len(), self.len_bytes, "the message length is fixed by the circuit");
        let mut filler = self.circuit.new_witness_filler();
        for (i, wire) in self.message.iter().enumerate() {
            let mut word = [0u8; 8];
            let chunk = &message[8 * i..(8 * i + 8).min(message.len())];
            word[..chunk.len()].copy_from_slice(chunk);
            filler[*wire] = Word(u64::from_le_bytes(word));
        }
        let digest = digest(message);
        for i in 0..4 {
            filler[self.digest[i]] =
                Word(u64::from_le_bytes(digest[8 * i..8 * i + 8].try_into().unwrap()));
        }
        self.circuit
            .populate_wire_witness(&mut filler)
            .expect("the witness satisfies the circuit");
        filler.into_value_vec()
    }
}

/// Keccak-256 out of circuit, on [`ref_keccak_f1600`]: the sponge at rate 136 with `pad10*1`.
fn digest(message: &[u8]) -> [u8; 32] {
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

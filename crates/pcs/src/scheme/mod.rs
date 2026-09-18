//! The scheme: parameters, a prover, a verifier, and the five values that pass between them.
//!
//! One round is
//!
//! ```text
//!     (C, opening)   = prover.commit(w)
//!     p = (p0, p1)   = verifier.derive_evaluation_point(transcript, C)
//!     t              = w.mle_evaluate(p)                    the statement
//!     u              = w.row_evaluate(p)                    the prover's message
//!     c              = verifier.derive_folding_challenges(transcript, u)
//!     v              = prover.fold(opening, c)
//! ```
//!
//! and the verifier accepts when `u . eq(p1) = t`, `v` is short, `A v = sum_j c_j C_j` modulo
//! every modulus, and `eq(p0) . (v mod 2) = sum_j u_j (c_j mod 2)` over `F162`.
//!
//! With [`Params::recursion`] the last three messages become one LaBRADOR proof of
//! [`crate::recursion`]'s relation:
//!
//! ```text
//!     (T_Y, opening) = prover.commit(w)
//!     p              = verifier.derive_evaluation_point(transcript, T_Y)
//!     T_u            = prover.commit_left_expansion(w.row_evaluate(p))
//!     c              = verifier.derive_folding_challenges(transcript, T_u)
//!     (T_R, eta, pi) = prover.prove_opening(transcript, opening, c, p, T_u, u, t, T_Y)
//! ```
//!
//! and the verifier accepts when every `eta_i` is under its cap, the no-wraparound bound of
//! [`crate::recursion::bound`] clears `Q / 2`, and LaBRADOR accepts `pi` against the statement it
//! rebuilds from those same public values. Both identities above are inside the proof.
mod commitment;
mod params;
mod prover;
mod verifier;
mod witness;

pub use commitment::*;
pub use params::*;
pub use prover::*;
pub use verifier::*;
pub use witness::*;

use crate::challenge::Transcript;
use crate::fields::scalar::F162;
use crate::labrador::PolxBuf;

/// The claimed value, `T_R` and the announced norms, in the transcript position `v` had.
pub(super) fn absorb_opening(transcript: &mut Transcript, claim: &F162, t_r: &PolxBuf, norms: &[u64]) {
    transcript.absorb_bytes(b"labinius/claim");
    for limb in claim.0 {
        transcript.absorb_u64(limb);
    }
    transcript.absorb_bytes(b"labinius/rest-commitment");
    transcript.absorb_bytes(t_r.as_bytes());
    transcript.absorb_bytes(b"labinius/norms");
    for &n in norms {
        transcript.absorb_u64(n);
    }
}

/// The LaBRADOR statement is a deterministic function of everything absorbed so far, so its
/// digest is one derivation of the transcript rather than a hash of the constraints.
pub(super) fn statement_digest(transcript: &mut Transcript) -> [u8; 32] {
    let mut digest = [0u8; 32];
    transcript.fill(b"labinius/recursion/statement", &mut digest);
    digest
}

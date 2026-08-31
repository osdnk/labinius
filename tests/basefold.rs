//! binius64's own commitment and BaseFold opening at 2^12 `B128`: one honest round, and one whose
//! opening speaks about a vector the Merkle root does not bind.
#[allow(dead_code)]
#[path = "../src/bin/basefold.rs"]
mod basefold;

use basefold::{Pcs, prove, random_words, verify};
use binius_compute::BufferPool;

const LOG_LEN: usize = 12;
const WITNESS_SEED: u64 = 0x5A;

#[test]
fn the_opening_verifies_and_a_tampered_one_does_not() {
    let pool = BufferPool::new();
    let pcs = Pcs::new(LOG_LEN);
    let words = random_words(LOG_LEN, WITNESS_SEED);

    let (proof, claim, _) = prove(&pcs, &pool, LOG_LEN, &words, None);
    verify(&pcs, LOG_LEN, &proof, claim).expect("the honest opening verifies");

    let (tampered, tampered_claim, _) = prove(&pcs, &pool, LOG_LEN, &words, Some(0));
    assert!(
        verify(&pcs, LOG_LEN, &tampered, tampered_claim).is_err(),
        "an opening of a vector the root does not bind must be rejected"
    );
}

use super::{median_of, once, rate_label, Row, SECURITY_BITS};
use binius_verifier::fri::calculate_n_test_queries;
use bolt_rs::ligerito_recursive::{
    ligero_commit_pub, recursive_prover, recursive_verifier, EncodeMethod, HashMethod, ProverConfig,
    VerifierConfig,
};
use bolt_rs::{BinaryElem, BinaryElem128, ReedSolomonEncoding, Sha256Hasher};

const ROOT_BYTES: usize = 32;

pub struct Shape {
    pub log_msg_cols: usize,
    pub initial_k: usize,
    pub log_dims: Vec<usize>,
    pub ks: Vec<usize>,
}

pub fn shape(log_len: usize) -> Shape {
    let initial_k = 4;
    let mut log_msg_cols = log_len - initial_k;
    let mut log_dims = Vec::new();
    let mut ks = Vec::new();
    while log_msg_cols > 10 {
        log_msg_cols -= 4;
        log_dims.push(log_msg_cols);
        ks.push(4);
    }
    Shape {
        log_msg_cols: log_len - initial_k,
        initial_k,
        log_dims,
        ks,
    }
}

fn prover_config(shape: &Shape, inv_rate: usize) -> ProverConfig<BinaryElem128, BinaryElem128> {
    let msg_cols = 1usize << shape.log_msg_cols;
    ProverConfig {
        recursive_steps: shape.log_dims.len(),
        initial_dims: (msg_cols, 1usize << shape.initial_k),
        dims: shape
            .log_dims
            .iter()
            .zip(&shape.ks)
            .map(|(&cols, &k)| (1usize << cols, 1usize << k))
            .collect(),
        initial_k: shape.initial_k,
        ks: shape.ks.clone(),
        initial_rs: ReedSolomonEncoding::new(msg_cols, msg_cols * inv_rate),
        recursive_rs: shape
            .log_dims
            .iter()
            .map(|&cols| {
                let m = 1usize << cols;
                ReedSolomonEncoding::new(m, m * inv_rate)
            })
            .collect(),
    }
}

fn verifier_config(shape: &Shape) -> VerifierConfig {
    VerifierConfig {
        recursive_steps: shape.log_dims.len(),
        initial_log_msg_cols: shape.log_msg_cols,
        log_dims: shape.log_dims.clone(),
        initial_k: shape.initial_k,
        ks: shape.ks.clone(),
    }
}

pub fn elements(u64s: &[u64]) -> Vec<BinaryElem128> {
    u64s.chunks_exact(2)
        .map(|pair| BinaryElem128::from_u128((pair[1] as u128) << 64 | pair[0] as u128))
        .collect()
}

pub fn run(log_len: usize, log_inv_rate: usize, u64s: &[u64]) -> Row {
    let inv_rate = 1usize << log_inv_rate;
    let queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
    let shape = shape(log_len);
    let config = prover_config(&shape, inv_rate);
    let verifier = verifier_config(&shape);
    let poly = elements(u64s);

    let (commit_ms, _) = median_of(3, || {
        ligero_commit_pub::<BinaryElem128, Sha256Hasher>(
            &poly,
            1usize << shape.log_msg_cols,
            1usize << shape.initial_k,
            &config.initial_rs,
        )
    });

    let (prove_ms, proof) = once(|| {
        recursive_prover::<BinaryElem128, BinaryElem128, Sha256Hasher>(
            &config,
            &poly,
            queries,
            HashMethod::Cpu,
            EncodeMethod::Cpu,
        )
    });

    let (verify_ms, verified) = median_of(3, || {
        recursive_verifier::<BinaryElem128, BinaryElem128, Sha256Hasher>(
            &verifier, &proof, queries, inv_rate,
        )
    });
    assert!(verified, "the honest opening verifies");

    Row {
        scheme: "bolt-rs Ligerito",
        rate: rate_label(log_inv_rate),
        security: format!(
            "{SECURITY_BITS} bits, {queries} queries, flat rate, built config k={} ks={:?}",
            shape.initial_k, shape.ks
        ),
        claim: "element-MLE, transcript point",
        commit_ms,
        open_ms: prove_ms - commit_ms,
        verify_ms,
        commitment: ROOT_BYTES,
        proof: proof.size_bytes() - ROOT_BYTES,
    }
}

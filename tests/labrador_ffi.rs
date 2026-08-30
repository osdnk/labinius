//! The LaBRADOR FFI layer: statement/witness construction in `polx` form, proving,
//! verifying, tampering, and the timings that matter for the recursion.
//!
//! Run with `cargo test --release --offline --test labrador_ffi -- --nocapture --test-threads 1`
//! to see the benchmark lines and LaBRADOR's own prover output.

use std::sync::Arc;
use std::time::Instant;

use bin_ntt::labrador::{
    self, warm_comkey, BSource, Block, CommitmentKey, Constraint, PhiSource, PolxBuf, Statement,
    VectorSpec, Witness, N,
};

// ---------------------------------------------------------------------------------------
// deterministic sampling
// ---------------------------------------------------------------------------------------

struct Xof(blake3::OutputReader);

impl Xof {
    fn new(label: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(label.as_bytes());
        Xof(h.finalize_xof())
    }

    fn fill(&mut self, out: &mut [u8]) {
        self.0.fill(out);
    }

    fn u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.fill(&mut b);
        u64::from_le_bytes(b)
    }

    /// A uniform residue in `[0, q)`.
    fn zq(&mut self) -> i64 {
        let q = labrador::compiled_q();
        let mask = (1u64 << labrador::logq()) - 1;
        loop {
            let x = self.u64() & mask;
            if x < q {
                return x as i64;
            }
        }
    }

    fn uniform_polys(&mut self, len: usize) -> Vec<[i64; N]> {
        (0..len).map(|_| std::array::from_fn(|_| self.zq())).collect()
    }

    fn ternary(&mut self, coeffs: usize) -> Vec<i16> {
        let mut bytes = vec![0u8; coeffs];
        self.fill(&mut bytes);
        bytes.iter().map(|&b| (b % 3) as i16 - 1).collect()
    }

    fn binary(&mut self, coeffs: usize) -> Vec<i16> {
        let mut bytes = vec![0u8; coeffs];
        self.fill(&mut bytes);
        bytes.iter().map(|&b| (b & 1) as i16).collect()
    }
}

// ---------------------------------------------------------------------------------------
// shape (a): test_off's run_shape, rebuilt through the Rust layer
// ---------------------------------------------------------------------------------------

const SHAPE_N: [usize; 3] = [1024, 1024, 256];
const SHAPE_K1: usize = 64;
const SHAPE_K2: usize = 6;

fn shape_witness() -> Witness {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/shape/witness");
    Witness::new(SHAPE_N.iter().map(|&n| xof.ternary(n * N)).collect())
}

/// `prepare_shape()` of `labrador/test_off.c` with `PROBE_EXACT`: `r = 3` vectors of ranks
/// `1024, 1024, 256`, exact l2-norm bounds, 64 two-block constraints at offsets that walk
/// across both long vectors, and 6 constraints spanning everything.
fn shape_statement(wit: &Witness) -> (Statement, ShapeTimings) {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/shape/phi");
    let sx = wit.to_sx();

    let vectors: Vec<VectorSpec> = (0..3)
        .map(|i| VectorSpec::norm_bounded(SHAPE_N[i], wit.normsq(i)))
        .collect();

    let mut sample = std::time::Duration::ZERO;
    let mut convert = std::time::Duration::ZERO;
    let mut evaluate = std::time::Duration::ZERO;
    let mut phi_polys = 0usize;
    let mut constraints = Vec::with_capacity(SHAPE_K1 + SHAPE_K2);

    for c in 0..SHAPE_K1 + SHAPE_K2 {
        let (blocks, len) = if c < SHAPE_K1 {
            (
                vec![
                    Block::new(c & 1, (32 * (c / 2)) % 1024, 32),
                    Block::new(2, (4 * (c / 2)) % SHAPE_N[2], 4),
                ],
                36,
            )
        } else {
            (
                (0..3).map(|j| Block::new(j, 0, SHAPE_N[j])).collect(),
                SHAPE_N[0] + SHAPE_N[1] + SHAPE_N[2],
            )
        };

        let t = Instant::now();
        let phi = xof.uniform_polys(len);
        sample += t.elapsed();
        phi_polys += len;

        let cnst = Constraint::new(1, blocks, PhiSource::Int64(phi), None);
        let t = Instant::now();
        cnst.precompute();
        convert += t.elapsed();

        let t = Instant::now();
        let b = cnst.eval(&sx);
        evaluate += t.elapsed();

        let mut cnst = cnst;
        cnst.b = Some(BSource::Polx(Arc::new(b)));
        constraints.push(cnst);
    }

    let t = Instant::now();
    let stmt = Statement::new(vectors, constraints);
    let digest = t.elapsed();

    (stmt, ShapeTimings { sample, convert, evaluate, digest, phi_polys })
}

struct ShapeTimings {
    sample: std::time::Duration,
    convert: std::time::Duration,
    evaluate: std::time::Duration,
    digest: std::time::Duration,
    phi_polys: usize,
}

#[test]
fn shape_prove_and_verify() {
    let total_rank: usize = SHAPE_N.iter().sum();
    let warm = warm_comkey(labrador::comkey_len_for_rank(total_rank));

    let wit = shape_witness();
    let build = Instant::now();
    let (stmt, timings) = shape_statement(&wit);
    let build = build.elapsed();

    let warm = warm.join().unwrap();
    println!("--- shape (a): r=3, n={SHAPE_N:?}, k={} ---", stmt.constraints.len());
    println!("  comkey warm-up (background)     {:>9.3?}  -> {} polx", warm, labrador::comkey_len());
    println!("  statement build (total)         {:>9.3?}", build);
    println!("    phi sampling                  {:>9.3?}", timings.sample);
    println!(
        "    phi -> polx ({} polys)      {:>9.3?}  = {:.3?} / 1000 polys",
        timings.phi_polys,
        timings.convert,
        timings.convert / (timings.phi_polys as u32) * 1000
    );
    println!("    b = <phi, s> (70 constraints)  {:>9.3?}", timings.evaluate);
    println!("    content digest                {:>9.3?}", timings.digest);

    let t = Instant::now();
    let proof = labrador::prove(&stmt, &wit).expect("prove");
    let prove_time = t.elapsed();

    let t = Instant::now();
    labrador::verify(&stmt, &proof).expect("verify");
    let verify_time = t.elapsed();

    println!("  composite_prove_simple          {:>9.3?}", prove_time);
    println!("  composite_verify_simple         {:>9.3?}", verify_time);
    println!("  proof size                      {:>9.2} KB", proof.size_kb());
    println!("  comkey after proving            {} polx", labrador::comkey_len());
    assert!(proof.size_kb() > 0.0);
}

// ---------------------------------------------------------------------------------------
// (b) a degree-kappa commitment constraint against an expanded key
// ---------------------------------------------------------------------------------------

const COMMIT_N: usize = 1024;
const COMMIT_KAPPA: usize = 8;

#[test]
fn commitment_constraint() {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/commit");
    let s = xof.ternary(COMMIT_N * N);
    let wit = Witness::new(vec![s.clone()]);

    let seed = [7u8; 16];
    let t = Instant::now();
    let key = CommitmentKey::expand(COMMIT_N, COMMIT_KAPPA, &seed, 0);
    let expand_time = t.elapsed();
    assert_eq!(key.buf().len(), labrador::extlen(COMMIT_N, COMMIT_KAPPA));

    let t = Instant::now();
    let u = key.commit_i16(&s);
    let commit_time = t.elapsed();
    assert_eq!(u.len(), COMMIT_KAPPA);

    // The same commitment through the shared polx form of the witness.
    let sx = wit.to_sx();
    assert_eq!(key.commit_sx(&sx, 0, 0), u);

    let cnst = Constraint::new(
        COMMIT_KAPPA,
        vec![Block::new(0, 0, COMMIT_N)],
        PhiSource::polx(key.buf_arc()),
        Some(BSource::Polx(Arc::new(u))),
    );
    // The degree-kappa linear form of the constraint is exactly the commitment.
    assert_eq!(&cnst.eval(&sx), match &cnst.b {
        Some(BSource::Polx(b)) => b.as_ref(),
        _ => unreachable!(),
    });

    let stmt = Statement::new(vec![VectorSpec::norm_bounded(COMMIT_N, wit.normsq(0))], vec![cnst]);

    println!("--- (b) degree-{COMMIT_KAPPA} commitment constraint, n={COMMIT_N} ---");
    println!("  key expansion ({} polx)       {:>9.3?}", key.buf().len(), expand_time);
    println!("  Ajtai commit                    {:>9.3?}", commit_time);

    let t = Instant::now();
    let proof = labrador::prove(&stmt, &wit).expect("prove");
    let prove_time = t.elapsed();
    let t = Instant::now();
    labrador::verify(&stmt, &proof).expect("verify");
    let verify_time = t.elapsed();
    println!("  prove                           {:>9.3?}", prove_time);
    println!("  verify                          {:>9.3?}", verify_time);
    println!("  proof size                      {:>9.2} KB", proof.size_kb());
}

/// The same key aliased as the `phi` of two constraints at once, over two witness vectors:
/// nothing is copied and nothing is double-freed when the statement is torn down.
#[test]
fn commitment_key_shared_across_constraints() {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/commit/shared");
    let n = 512;
    let kappa = 4;
    let wit = Witness::new(vec![xof.ternary(n * N), xof.ternary(n * N)]);
    let sx = wit.to_sx();
    let key = CommitmentKey::expand(n, kappa, &[11u8; 16], 3);

    let constraints = (0..2)
        .map(|i| {
            let u = key.commit_sx(&sx, i, 0);
            Constraint::new(
                kappa,
                vec![Block::new(i, 0, n)],
                PhiSource::polx(key.buf_arc()),
                Some(BSource::Polx(Arc::new(u))),
            )
        })
        .collect();
    let vectors = (0..2).map(|i| VectorSpec::norm_bounded(n, wit.normsq(i))).collect();
    let stmt = Statement::new(vectors, constraints);

    let proof = labrador::prove(&stmt, &wit).expect("prove");
    labrador::verify(&stmt, &proof).expect("verify");
    println!("--- (b') shared key, 2 x degree-{kappa} constraints: {:.2} KB", proof.size_kb());
}

// ---------------------------------------------------------------------------------------
// (c) a binary vector (betasq == 0)
// ---------------------------------------------------------------------------------------

#[test]
fn binary_vector() {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/binary");
    let n = 256;
    let wit = Witness::new(vec![xof.binary(n * N)]);
    let sx = wit.to_sx();

    let cnst = Constraint::new(1, vec![Block::new(0, 0, n)], PhiSource::Int64(xof.uniform_polys(n)), None);
    let b = cnst.eval(&sx);
    let mut cnst = cnst;
    cnst.b = Some(BSource::Polx(Arc::new(b)));
    let stmt = Statement::new(vec![VectorSpec::binary(n)], vec![cnst]);

    println!("--- (c) binary vector, n={n} (betasq = 0) ---");
    match labrador::prove(&stmt, &wit) {
        Ok(proof) => match labrador::verify(&stmt, &proof) {
            Ok(()) => println!("  binary path OK, proof size {:.2} KB", proof.size_kb()),
            Err(e) => println!("  REPORT: binariness path verifies FALSE: {e}"),
        },
        Err(e) => println!("  REPORT: binariness path fails to prove: {e}"),
    }
}

// ---------------------------------------------------------------------------------------
// (d) tampering
// ---------------------------------------------------------------------------------------

const TAMPER_N: usize = 256;

fn tamper_setup() -> (Statement, Witness) {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/tamper");
    let wit = Witness::new(vec![xof.ternary(TAMPER_N * N), xof.ternary(TAMPER_N * N)]);
    let sx = wit.to_sx();

    let mut constraints = Vec::new();
    for c in 0..4 {
        let blocks = vec![Block::new(0, 0, TAMPER_N), Block::new(1, 64 * c, 64)];
        let phi = xof.uniform_polys(TAMPER_N + 64);
        let cnst = Constraint::new(1, blocks, PhiSource::Int64(phi), None);
        let b = cnst.eval(&sx);
        let mut cnst = cnst;
        cnst.b = Some(BSource::Polx(Arc::new(b)));
        constraints.push(cnst);
    }
    let vectors = (0..2).map(|i| VectorSpec::norm_bounded(TAMPER_N, wit.normsq(i))).collect();
    (Statement::new(vectors, constraints), wit)
}

#[test]
fn tamper_wrong_b() {
    let (stmt, wit) = tamper_setup();
    let proof = labrador::prove(&stmt, &wit).expect("honest prove");

    let mut bad = tamper_setup().0;
    let wrong = PolxBuf::expand(1, &[3u8; 16], 99);
    bad.constraints[2].b = Some(BSource::Polx(Arc::new(wrong)));
    bad.reseal();
    let err = labrador::verify(&bad, &proof).expect_err("wrong b must not verify");
    println!("--- (d) wrong b: {err}");

    // With the original digest kept, so only b itself differs, it must still fail.
    let mut bad = tamper_setup().0;
    bad.constraints[2].b = Some(BSource::Polx(Arc::new(PolxBuf::expand(1, &[5u8; 16], 1))));
    let err = labrador::verify(&bad, &proof).expect_err("wrong b (same digest) must not verify");
    println!("--- (d) wrong b, digest unchanged: {err}");
}

#[test]
fn tamper_wrong_witness_coefficient() {
    let (stmt, wit) = tamper_setup();

    // Norm-preserving: negate one non-zero coefficient, so the tamper has to be caught by
    // the constraints rather than by the l2-norm bound.
    let mut flipped = wit.clone();
    let i = flipped.vectors[1].iter().position(|&c| c != 0).unwrap();
    flipped.vectors[1][i] = -flipped.vectors[1][i];
    assert_eq!(flipped.normsq(1), wit.normsq(1));
    let err = labrador::prove_verified(&stmt, &flipped).expect_err("tampered witness must not prove");
    println!("--- (d) wrong witness coefficient (norm preserved): {err}");
    assert!(err.contains("simple_verify"), "unexpected error: {err}");

    // Norm-raising: caught before LaBRADOR is ever called.
    let mut grown = wit.clone();
    let i = grown.vectors[1].iter().position(|&c| c == 0).unwrap();
    grown.vectors[1][i] = 1;
    let err = labrador::prove(&stmt, &grown).expect_err("over-long witness must not prove");
    println!("--- (d) wrong witness coefficient (norm raised): {err}");
    assert!(err.contains("normsq"), "unexpected error: {err}");
}

#[test]
fn tamper_wrong_digest() {
    let (stmt, wit) = tamper_setup();
    let proof = labrador::prove(&stmt, &wit).expect("honest prove");

    let mut bad = tamper_setup().0;
    bad.digest[0] ^= 1;
    let err = labrador::verify(&bad, &proof).expect_err("wrong digest must not verify");
    println!("--- (d) wrong digest: {err}");
}

// ---------------------------------------------------------------------------------------
// conversion benchmarks
// ---------------------------------------------------------------------------------------

#[test]
fn bench_polx_conversion() {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/bench");
    const LEN: usize = 8192;
    let i64s = xof.uniform_polys(LEN);
    let i16s: Vec<[i16; N]> = (0..LEN).map(|_| std::array::from_fn(|_| (xof.u64() % 3) as i16 - 1)).collect();

    let t = Instant::now();
    let a = PolxBuf::from_int64(&i64s);
    let d64 = t.elapsed();
    let t = Instant::now();
    let b = PolxBuf::from_int16(&i16s);
    let d16 = t.elapsed();
    assert_eq!(a.len(), LEN);
    assert_eq!(b.len(), LEN);

    println!("--- polx conversion, {LEN} polys ---");
    println!("  polxvec_fromint64vec            {:>9.3?}  = {:.3?} / 1000 polys", d64, d64 / (LEN as u32) * 1000);
    println!("  int16 -> polxvec_frompolyvec    {:>9.3?}  = {:.3?} / 1000 polys", d16, d16 / (LEN as u32) * 1000);
    println!("  sizeof(polx) = {} bytes, LOGQ = {}, q = {}", labrador::sizeof_polx(), labrador::logq(), labrador::compiled_q());
}

#[test]
fn bench_comkey_warmup() {
    let len = labrador::comkey_len_for_rank(SHAPE_N.iter().sum::<usize>());
    labrador::free_comkey();
    let t = Instant::now();
    labrador::ensure_comkey(len);
    let cold = t.elapsed();
    let t = Instant::now();
    labrador::ensure_comkey(len);
    let warm = t.elapsed();
    println!("--- init_comkey ---");
    println!("  cold expansion to {len} polx   {:>9.3?}", cold);
    println!("  already expanded                {:>9.3?}", warm);
    assert!(labrador::comkey_len() >= len);
}

// ---------------------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------------------

#[test]
fn rejects_out_of_range_blocks() {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/validate");
    let wit = Witness::new(vec![xof.ternary(64 * N)]);

    let stmt = Statement::new(
        vec![VectorSpec::norm_bounded(64, wit.normsq(0))],
        vec![Constraint::new(1, vec![Block::new(0, 32, 64)], PhiSource::Int64(xof.uniform_polys(64)), None)],
    );
    assert!(labrador::prove(&stmt, &wit).unwrap_err().contains("spans"));

    // A degree-8 constraint reads extlen(len, 8) coefficients, so the block must leave room.
    let stmt = Statement::new(
        vec![VectorSpec::norm_bounded(64, wit.normsq(0))],
        vec![Constraint::new(8, vec![Block::new(0, 60, 3)], PhiSource::Int64(xof.uniform_polys(8)), None)],
    );
    assert!(labrador::prove(&stmt, &wit).unwrap_err().contains("spans"));
}

// ---------------------------------------------------------------------------------------
// (e) degree 0, 1 and kappa in one statement, as the recursion mixes them
// ---------------------------------------------------------------------------------------

const MIXED_N: usize = 64;
const MIXED_KAPPA: usize = 8;
/// Coefficients at and above this position are zero, and only the degree-0 constraints say so.
const MIXED_SUPPORT: usize = 32;

/// Two vectors whose top coefficients are zero, three constraint degrees interleaved, and the
/// zero-part test of the recursion: `phi = sum_j rho_j X^{-j}` over the positions that must
/// vanish, so the constant coefficient of the linear form is `sum_j rho_j s_j = 0`.
fn mixed_setup(tamper: bool) -> (Statement, Witness) {
    let mut xof = Xof::new("bin-ntt/labrador_ffi/mixed");
    let mut vectors: Vec<Vec<i16>> = (0..2)
        .map(|_| {
            let mut v = xof.ternary(MIXED_N * N);
            for p in 0..MIXED_N {
                v[p * N + MIXED_SUPPORT..(p + 1) * N].fill(0);
            }
            v
        })
        .collect();
    if tamper {
        vectors[1][MIXED_SUPPORT + 3] = 1;
    }
    let wit = Witness::new(vectors);
    let sx = wit.to_sx();
    let key = CommitmentKey::expand(MIXED_N, MIXED_KAPPA, &[19u8; 16], 5);

    let mut constraints = Vec::new();
    for c in 0..3 {
        constraints.push(Constraint::new(
            0,
            (0..2).map(|i| Block::new(i, 0, MIXED_N)).collect(),
            PhiSource::Int64(
                (0..2 * MIXED_N)
                    .map(|_| {
                        let mut e = [0i64; N];
                        for j in MIXED_SUPPORT..N {
                            e[N - j] = xof.zq();
                        }
                        e
                    })
                    .collect(),
            ),
            None,
        ));

        let mut linear = Constraint::new(
            1,
            vec![Block::new(c % 2, 0, MIXED_N)],
            PhiSource::Int64(xof.uniform_polys(MIXED_N)),
            None,
        );
        linear.b = Some(BSource::Polx(Arc::new(linear.eval(&sx))));
        constraints.push(linear);

        constraints.push(Constraint::new(
            MIXED_KAPPA,
            vec![Block::new(c % 2, 0, MIXED_N)],
            PhiSource::polx(key.buf_arc()),
            Some(BSource::Polx(Arc::new(key.commit_sx(&sx, c % 2, 0)))),
        ));
    }
    let vectors = (0..2).map(|i| VectorSpec::norm_bounded(MIXED_N, wit.normsq(i))).collect();
    (Statement::new(vectors, constraints), wit)
}

#[test]
fn mixed_degrees() {
    let (stmt, wit) = mixed_setup(false);
    let proof = labrador::prove(&stmt, &wit).expect("prove");
    labrador::verify(&stmt, &proof).expect("verify");
    println!("--- (e) degrees 0/1/{MIXED_KAPPA} interleaved: {:.2} KB", proof.size_kb());

    let (bad, wit) = mixed_setup(true);
    let err = labrador::prove_verified(&bad, &wit).expect_err("a coefficient outside the support");
    println!("--- (e) coefficient outside its support: {err}");
    assert!(err.contains("simple_verify"), "unexpected error: {err}");
}

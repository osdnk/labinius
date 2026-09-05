//! LaBRADOR (Dachshund) backend: statements, witnesses, `composite_prove_simple` and
//! `composite_verify_simple`, over the fork in the `labrador/` submodule built at
//! `LOGQ = 48`.
//!
//! # Shape of a statement
//!
//! A statement is `r` witness vectors -- vector `i` has rank `n[i]` (that many ring
//! elements of `Z_q[X]/(X^64 + 1)`) and either an exact l2-norm bound `betasq[i]` or, when
//! [`VectorSpec::binary`] is set, the requirement that every coefficient is `0` or `1`
//! (Dachshund encodes "binary" as `betasq == 0`) -- together with `k` dot-product
//! constraints. Constraint `c` of extension degree `deg` is
//!
//! ```text
//! sum over blocks j of  < phi_j , s[idx_j][off_j .. off_j + len_j] >  ==  b
//! ```
//!
//! where the inner product is LaBRADOR's truncated degree-`deg` extension product, `phi_j`
//! is `extlen(len_j, deg)` ring elements and `b` is `max(1, deg)` ring elements. `deg == 1`
//! is the ordinary linear constraint.
//!
//! # Everything is `polx`
//!
//! Constraints are handed to LaBRADOR already in `polx` form. [`PhiSource::Polx`] aliases a
//! caller-owned [`PolxBuf`] by pointer -- nothing is copied, which is what makes a
//! commitment key usable directly as the `phi` of the matching commitment constraint. The
//! owned variants ([`PhiSource::Int64`], [`PhiSource::Int16`]) are converted once and cached
//! inside the [`Constraint`], so re-deriving a statement for verification is free.
//!
//! Aliasing is sound because `free_sparsecnst()` frees only the single allocation made by
//! `init_sparsecnst_half()` (which here holds nothing but `idx/off/len/mult/phi` and the
//! `b` slot); the `phi` blocks themselves are never freed by LaBRADOR.
//!
//! # The statement digest binds everything
//!
//! LaBRADOR's Fiat-Shamir transcript starts from the 16-byte statement hash `st->h`. The
//! upstream setters absorb every `phi` and `b` polynomial into that hash as they are set;
//! this backend does not, because re-serialising 14 MB of constraint data per proof is the
//! single most expensive thing in statement construction and the caller has already hashed
//! the same data upstream. Instead `st->h = shake128(digest, 16)` for a 32-byte digest the
//! caller supplies in [`Statement::digest`].
//!
//! **That digest is the only binding of the statement.** The caller MUST commit, in it, to
//! every input the verifier is trusted to agree on: the number of vectors, each rank, each
//! norm bound and binariness flag, and for every constraint its degree, every block
//! `(idx, off, len)`, every `phi` coefficient and `b`. [`Statement::content_digest`]
//! computes exactly such a digest and is the safe default; a caller who already derives the
//! statement from a transcript may substitute its own, as long as it is injective in all of
//! the above.
//!
//! # Threading
//!
//! LaBRADOR keeps its commitment key in the mutable globals `comkey` / `comkey_len`, which
//! `init_comkey()` reallocates, so proving and verifying are serialised here behind a
//! process-wide lock. [`warm_comkey`] expands the key on a background thread so that the
//! first proof does not pay for it.
//!
//! # What the prover does not do
//!
//! [`prove`] does not run `simple_verify`. It converts the whole witness to `polx` and evaluates
//! every constraint against it -- an aggregation pass' worth of work, 60 ms of a 375 ms proof at
//! `Params::basic()` -- to tell an honest prover what it already knows. [`prove_verified`] keeps
//! it for the tests. The library's own stdout chatter is redirected to `/dev/null` around every
//! entry point unless `BIN_NTT_LABRADOR_VERBOSE` is set; on a terminal it cost 48 ms of a proof
//! and 27 ms of a verification.

pub mod ffi;
pub mod polx;

use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard, OnceLock};

pub use polx::{
    comkey_len, compiled_q, extlen, logq, philen, sis_rank, sizeof_polx, CommitmentKey, PolxBuf,
    Sx, N,
};

/// The largest witness coefficient magnitude LaBRADOR's `polyvec_sprodz` can square without
/// overflowing its `int32` accumulator lanes (`4 * 23170^2 < 2^31`).
pub const WITNESS_COEFF_MAX: i16 = 23170;

/// One witness vector: `n` ring elements, either l2-norm bounded or binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSpec {
    /// Rank: number of ring elements, i.e. `n * 64` integer coefficients.
    pub n: usize,
    /// Exact squared l2-norm bound. Ignored (and passed to LaBRADOR as `0`) when `binary`.
    pub betasq: u64,
    /// Binary vector: every coefficient in `{0, 1}`. Dachshund encodes this as `betasq == 0`.
    pub binary: bool,
}

impl VectorSpec {
    pub fn norm_bounded(n: usize, betasq: u64) -> Self {
        Self {
            n,
            betasq,
            binary: false,
        }
    }

    pub fn binary(n: usize) -> Self {
        Self {
            n,
            betasq: 0,
            binary: true,
        }
    }

    fn c_betasq(&self) -> u64 {
        if self.binary {
            0
        } else {
            self.betasq
        }
    }
}

/// A contiguous slice `s[idx][off .. off + len]` of one witness vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub idx: usize,
    pub off: usize,
    pub len: usize,
}

impl Block {
    pub fn new(idx: usize, off: usize, len: usize) -> Self {
        Self { idx, off, len }
    }
}

/// Where a constraint's `phi` comes from.
///
/// The owned variants hold the concatenation of the blocks' coefficient polynomials, each
/// block padded to `extlen(len, deg)` (no padding when `deg == 1`); they are converted to
/// `polx` once and cached. [`PhiSource::Polx`] aliases an existing buffer starting at
/// `base`, with block `j` at `base + sum of extlen(len_i, deg) for i < j` -- the layout
/// LaBRADOR itself uses when it points a constraint at rows of the commitment key.
#[derive(Debug)]
pub enum PhiSource {
    Int64(Vec<[i64; N]>),
    Int16(Vec<[i16; N]>),
    Polx(std::sync::Arc<PolxBuf>, usize),
    /// One block descriptor per block, so that blocks of the same constraint can alias
    /// buffers that were converted independently and are shared with other constraints.
    Blocks(Vec<PhiBlock>),
}

/// A `phi` block given in the coefficient domain: element `i` of the block is
/// `sum_t c[i * wid + t] X^(off + t)`, with `off + wid <= N`.
///
/// A chain constraint's `phi` is a public sub-chunk: nine consecutive `i16` coefficients of
/// a polynomial of degree 64. Handing LaBRADOR those nine numbers instead of the `1 KB`
/// `polx` image is a 57-fold cut in what the aggregation streams, and lets it multiply the
/// block by the aggregation challenge over `int32` coefficient lanes -- one `polx`
/// conversion per destination element at the end, instead of one per contribution.
#[derive(Debug)]
pub struct ShortPhi {
    pub off: usize,
    pub wid: usize,
    pub c: Vec<i16>,
}

impl ShortPhi {
    pub fn new(off: usize, wid: usize, c: Vec<i16>) -> Self {
        assert!(
            wid > 0 && off + wid <= N,
            "a short phi must fit the ring degree"
        );
        assert_eq!(
            c.len() % wid,
            0,
            "coefficients are element major, {wid} per element"
        );
        Self { off, wid, c }
    }

    /// Elements the buffer holds.
    pub fn len(&self) -> usize {
        self.c.len() / self.wid
    }

    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// Bytes held, for a memory report.
    pub fn bytes(&self) -> usize {
        self.c.len() * 2
    }
}

/// Where one block of a [`PhiSource::Blocks`] constraint reads its `phi`: either
/// `polx` at an offset into a shared buffer, or `wid` coefficients per element at an
/// element offset into a shared [`ShortPhi`].
#[derive(Debug)]
pub enum PhiBlock {
    Polx(std::sync::Arc<PolxBuf>, usize),
    Short(std::sync::Arc<ShortPhi>, usize),
}

impl PhiSource {
    pub fn polx(buf: std::sync::Arc<PolxBuf>) -> Self {
        PhiSource::Polx(buf, 0)
    }

    fn source_len(&self) -> usize {
        match self {
            PhiSource::Int64(v) => v.len(),
            PhiSource::Int16(v) => v.len(),
            PhiSource::Polx(buf, base) => buf.len().saturating_sub(*base),
            PhiSource::Blocks(_) => usize::MAX,
        }
    }
}

/// A constraint's right-hand side, `max(1, deg)` ring elements.
#[derive(Debug)]
pub enum BSource {
    Int64(Vec<[i64; N]>),
    Polx(std::sync::Arc<PolxBuf>),
}

impl BSource {
    fn len(&self) -> usize {
        match self {
            BSource::Int64(v) => v.len(),
            BSource::Polx(buf) => buf.len(),
        }
    }
}

/// One dot-product constraint.
///
/// Build with [`Constraint::new`]; the remaining field is a cache for the converted `phi`.
#[derive(Debug)]
pub struct Constraint {
    /// Extension degree. `0` is a constant-coefficient constraint: only the constant
    /// coefficient of the linear form has to match `b`, and `phi` is one ring element per
    /// witness polynomial.
    pub deg: usize,
    pub blocks: Vec<Block>,
    pub phi: PhiSource,
    pub b: Option<BSource>,
    phi_cache: OnceLock<PolxBuf>,
}

/// The per-block pointer arrays the shim takes: one `polx` pointer per block, and, for the
/// blocks given in the coefficient domain, one `i16` pointer with its offset and width.
struct BlockPtrs {
    phi: Vec<*const c_void>,
    sphi: Vec<*const i16>,
    soff: Vec<usize>,
    swid: Vec<usize>,
    short: bool,
}

impl BlockPtrs {
    fn short_ptr(&self) -> *const *const i16 {
        if self.short {
            self.sphi.as_ptr()
        } else {
            std::ptr::null()
        }
    }
}

impl Constraint {
    pub fn new(deg: usize, blocks: Vec<Block>, phi: PhiSource, b: Option<BSource>) -> Self {
        Self {
            deg,
            blocks,
            phi,
            b,
            phi_cache: OnceLock::new(),
        }
    }

    /// Total `phi` length: the blocks' lengths, each padded to `philen(len, deg)`.
    pub fn phi_len(&self) -> usize {
        self.blocks
            .iter()
            .map(|blk| philen(blk.len, self.deg))
            .sum()
    }

    /// Offset of block `j` within `phi`.
    pub fn phi_offset(&self, j: usize) -> usize {
        self.blocks[..j]
            .iter()
            .map(|blk| philen(blk.len, self.deg))
            .sum()
    }

    /// The `phi` buffer, converting and caching the owned forms on first use.
    fn phi_buf(&self) -> &PolxBuf {
        match &self.phi {
            PhiSource::Int64(v) => self.phi_cache.get_or_init(|| PolxBuf::from_int64(v)),
            PhiSource::Int16(v) => self.phi_cache.get_or_init(|| PolxBuf::from_int16(v)),
            PhiSource::Polx(buf, _) => buf,
            PhiSource::Blocks(_) => unreachable!("a per-block phi has no single buffer"),
        }
    }

    fn phi_base(&self) -> usize {
        match &self.phi {
            PhiSource::Polx(_, base) => *base,
            _ => 0,
        }
    }

    /// Pointers to each block's `phi`, aliased into the (cached or borrowed) buffers.
    /// Short blocks contribute a null `polx` pointer and a non-null coefficient pointer.
    fn phi_ptrs(&self) -> BlockPtrs {
        let mut p = BlockPtrs {
            phi: Vec::with_capacity(self.blocks.len()),
            sphi: vec![std::ptr::null(); self.blocks.len()],
            soff: vec![0; self.blocks.len()],
            swid: vec![0; self.blocks.len()],
            short: false,
        };
        if let PhiSource::Blocks(parts) = &self.phi {
            for (j, part) in parts.iter().enumerate() {
                match part {
                    PhiBlock::Polx(buf, off) => p.phi.push(buf.ptr_at(*off)),
                    PhiBlock::Short(sp, off) => {
                        p.phi.push(std::ptr::null());
                        p.sphi[j] = sp.c[off * sp.wid..].as_ptr();
                        p.soff[j] = sp.off;
                        p.swid[j] = sp.wid;
                        p.short = true;
                    }
                }
            }
            return p;
        }
        let buf = self.phi_buf();
        let base = self.phi_base();
        p.phi = (0..self.blocks.len())
            .map(|j| buf.ptr_at(base + self.phi_offset(j)))
            .collect();
        p
    }

    /// Force the `phi` conversion now, so that it is not charged to the first proof.
    pub fn precompute(&self) {
        if !matches!(self.phi, PhiSource::Blocks(_)) {
            let _ = self.phi_buf();
        }
    }

    /// Evaluate this constraint's linear form against a witness already in `polx` form.
    ///
    /// Returns the `max(1, deg)` ring elements that `b` must equal for the constraint to
    /// hold -- the Rust equivalent of `sparsecnst_eval`, but without needing the statement
    /// to exist yet, so a caller can build `b` and the statement in one pass.
    pub fn eval(&self, sx: &Sx) -> PolxBuf {
        let deg2 = self.deg.max(1);
        let out = PolxBuf::zeroed(deg2);
        let idx: Vec<usize> = self.blocks.iter().map(|b| b.idx).collect();
        let off: Vec<usize> = self.blocks.iter().map(|b| b.off).collect();
        let len: Vec<usize> = self.blocks.iter().map(|b| b.len).collect();
        let p = self.phi_ptrs();
        unsafe {
            ffi::bn_eval_blocks(
                out.as_ptr() as *mut c_void,
                self.deg,
                self.blocks.len(),
                idx.as_ptr(),
                off.as_ptr(),
                len.as_ptr(),
                p.phi.as_ptr(),
                p.short_ptr(),
                p.soff.as_ptr(),
                p.swid.as_ptr(),
                sx.raw(),
            );
        }
        out
    }
}

/// A Dachshund simple statement.
#[derive(Debug)]
pub struct Statement {
    pub vectors: Vec<VectorSpec>,
    pub constraints: Vec<Constraint>,
    /// The 32-byte digest that binds the whole statement; see the module documentation.
    pub digest: [u8; 32],
}

impl Statement {
    pub fn new(vectors: Vec<VectorSpec>, constraints: Vec<Constraint>) -> Self {
        let mut st = Self {
            vectors,
            constraints,
            digest: [0u8; 32],
        };
        st.digest = st.content_digest();
        st
    }

    /// The same with a digest the caller derives instead: for a statement that is already a
    /// deterministic function of a transcript, hashing hundreds of megabytes of `phi` again
    /// binds nothing new and costs more than the proof.
    pub fn with_digest(
        vectors: Vec<VectorSpec>,
        constraints: Vec<Constraint>,
        digest: [u8; 32],
    ) -> Self {
        Self {
            vectors,
            constraints,
            digest,
        }
    }

    /// Total rank over all vectors.
    pub fn total_rank(&self) -> usize {
        self.vectors.iter().map(|v| v.n).sum()
    }

    /// A digest binding every input of the statement: the vector specs, and for each
    /// constraint its degree, blocks, `phi` and `b`. Owned `phi`/`b` are hashed in their
    /// canonical coefficient form; aliased `polx` buffers are hashed as the `polx` bytes
    /// the prover will actually read.
    pub fn content_digest(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"bin-ntt/labrador/statement/v1");
        h.update(&(logq() as u64).to_le_bytes());
        h.update(&(self.vectors.len() as u64).to_le_bytes());
        for v in &self.vectors {
            h.update(&(v.n as u64).to_le_bytes());
            h.update(&v.c_betasq().to_le_bytes());
            h.update(&[u8::from(v.binary)]);
        }
        h.update(&(self.constraints.len() as u64).to_le_bytes());
        for c in &self.constraints {
            h.update(&(c.deg as u64).to_le_bytes());
            h.update(&(c.blocks.len() as u64).to_le_bytes());
            for blk in &c.blocks {
                h.update(&(blk.idx as u64).to_le_bytes());
                h.update(&(blk.off as u64).to_le_bytes());
                h.update(&(blk.len as u64).to_le_bytes());
            }
            match &c.phi {
                PhiSource::Int64(v) => {
                    h.update(b"i64");
                    h.update(&(v.len() as u64).to_le_bytes());
                    for p in v {
                        for &x in p {
                            h.update(&x.to_le_bytes());
                        }
                    }
                }
                PhiSource::Int16(v) => {
                    h.update(b"i16");
                    h.update(&(v.len() as u64).to_le_bytes());
                    for p in v {
                        for &x in p {
                            h.update(&x.to_le_bytes());
                        }
                    }
                }
                PhiSource::Polx(buf, base) => {
                    h.update(b"plx");
                    let start = base * sizeof_polx();
                    let end = (base + c.phi_len()) * sizeof_polx();
                    let bytes = buf.as_bytes();
                    h.update(&bytes[start.min(bytes.len())..end.min(bytes.len())]);
                }
                PhiSource::Blocks(parts) => {
                    h.update(b"blk");
                    for (part, blk) in parts.iter().zip(&c.blocks) {
                        match part {
                            PhiBlock::Polx(buf, off) => {
                                h.update(b"p");
                                let start = off * sizeof_polx();
                                let end = (off + philen(blk.len, c.deg)) * sizeof_polx();
                                let bytes = buf.as_bytes();
                                h.update(&bytes[start.min(bytes.len())..end.min(bytes.len())]);
                            }
                            PhiBlock::Short(sp, off) => {
                                h.update(b"s");
                                h.update(&(sp.off as u64).to_le_bytes());
                                h.update(&(sp.wid as u64).to_le_bytes());
                                for &x in &sp.c[off * sp.wid..(off + blk.len) * sp.wid] {
                                    h.update(&x.to_le_bytes());
                                }
                            }
                        }
                    }
                }
            }
            match &c.b {
                None => {
                    h.update(b"hom");
                }
                Some(BSource::Int64(v)) => {
                    h.update(b"bi64");
                    for p in v {
                        for &x in p {
                            h.update(&x.to_le_bytes());
                        }
                    }
                }
                Some(BSource::Polx(buf)) => {
                    h.update(b"bplx");
                    h.update(buf.as_bytes());
                }
            }
        }
        *h.finalize().as_bytes()
    }

    /// Recompute [`Statement::digest`] from the statement contents.
    pub fn reseal(&mut self) {
        self.digest = self.content_digest();
    }

    /// Convert every owned `phi` to `polx` now.
    pub fn precompute(&self) {
        for c in &self.constraints {
            c.precompute();
        }
    }
}

/// A witness: one `Vec<i16>` of `n * 64` coefficients per vector.
#[derive(Clone, Debug, Default)]
pub struct Witness {
    pub vectors: Vec<Vec<i16>>,
}

impl Witness {
    pub fn new(vectors: Vec<Vec<i16>>) -> Self {
        Self { vectors }
    }

    /// Squared l2-norm of vector `i`, the value LaBRADOR will recompute.
    pub fn normsq(&self, i: usize) -> u64 {
        self.vectors[i]
            .iter()
            .map(|&c| (c as i64 * c as i64) as u64)
            .sum()
    }

    /// Convert to `polx` once, for constraint evaluation.
    pub fn to_sx(&self) -> Sx {
        Sx::new(&self.vectors)
    }
}

// ---------------------------------------------------------------------------------------
// commitment key warm-up
// ---------------------------------------------------------------------------------------

/// LaBRADOR prints a page of statement and proof-size chatter per recursion level. Every
/// entry point into the library takes one of these, which redirects fd 1 to `/dev/null`
/// for as long as it lives; `BIN_NTT_LABRADOR_VERBOSE=1` leaves it alone. Errors go to
/// stderr and are never suppressed.
struct Quiet(bool);

impl Quiet {
    fn new() -> Quiet {
        let verbose = std::env::var_os("BIN_NTT_LABRADOR_VERBOSE").is_some();
        if !verbose {
            unsafe { ffi::bn_mute_stdout() };
        }
        Quiet(!verbose)
    }
}

impl Drop for Quiet {
    fn drop(&mut self) {
        if self.0 {
            unsafe { ffi::bn_unmute_stdout() };
        }
    }
}

fn labrador_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// A commitment-key length that covers a statement of this total witness rank.
///
/// `init_statement` sizes the key as `r * extlen(fu * kappa, kappa1)` over the *output*
/// statement, which is bounded above by roughly `3/2` of the input rank for the parameter
/// sets Dachshund picks; the key is rounded to the 32-`polx` chunk `init_comkey` expands in.
pub fn comkey_len_for_rank(total_rank: usize) -> usize {
    (total_rank * 3 / 2).div_ceil(32) * 32
}

/// Expand the global commitment key to at least `len` `polx`, if it is not already.
pub fn ensure_comkey(len: usize) {
    let _guard = labrador_lock();
    unsafe { ffi::labrador48_init_comkey(len) };
}

/// Expand the commitment key on a background thread; returns the time it took.
///
/// Proving expands the key itself if it has to, so this only moves the cost off the
/// critical path.
pub fn warm_comkey(len: usize) -> std::thread::JoinHandle<std::time::Duration> {
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        ensure_comkey(len);
        start.elapsed()
    })
}

/// Release the global commitment key. Not safe to call while a proof is in flight.
pub fn free_comkey() {
    let _guard = labrador_lock();
    unsafe { ffi::labrador48_free_comkey() };
}

// ---------------------------------------------------------------------------------------
// RAII wrappers
// ---------------------------------------------------------------------------------------

struct RawStatement(*mut c_void);

impl Drop for RawStatement {
    fn drop(&mut self) {
        unsafe {
            ffi::labrador48_free_smplstmnt(self.0);
            ffi::bn_free(self.0);
        }
    }
}

struct RawWitness(*mut c_void);

impl Drop for RawWitness {
    fn drop(&mut self) {
        unsafe {
            ffi::labrador48_free_witness(self.0);
            ffi::bn_free(self.0);
        }
    }
}

struct RawComposite(*mut c_void);

impl Drop for RawComposite {
    fn drop(&mut self) {
        unsafe {
            ffi::labrador48_free_composite(self.0);
            ffi::bn_free(self.0);
        }
    }
}

struct RawCommitment(*mut c_void);

impl Drop for RawCommitment {
    fn drop(&mut self) {
        unsafe {
            ffi::labrador48_free_commitment(self.0);
            ffi::bn_free(self.0);
        }
    }
}

/// A produced proof: the composite proof and the Dachshund commitment it opens.
#[derive(Debug)]
pub struct ProofHandle {
    composite: *mut c_void,
    commitment: *mut c_void,
    size_kb: f64,
}

unsafe impl Send for ProofHandle {}

impl ProofHandle {
    /// The analytic proof size in KB that LaBRADOR reports in `composite->size`.
    pub fn size_kb(&self) -> f64 {
        self.size_kb
    }
}

impl Drop for ProofHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::labrador48_free_composite(self.composite);
            ffi::bn_free(self.composite);
            ffi::labrador48_free_commitment(self.commitment);
            ffi::bn_free(self.commitment);
        }
    }
}

// ---------------------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------------------

fn check_statement(stmt: &Statement) -> Result<(), String> {
    if stmt.vectors.is_empty() {
        return Err("statement has no witness vectors".into());
    }
    if stmt.constraints.is_empty() {
        return Err("statement has no constraints".into());
    }
    let q = compiled_q() as i64;
    for (i, v) in stmt.vectors.iter().enumerate() {
        if v.n == 0 {
            return Err(format!("vector {i}: rank 0"));
        }
        if !v.binary && v.betasq == 0 {
            return Err(format!(
                "vector {i}: betasq 0 on a non-binary vector (use VectorSpec::binary)"
            ));
        }
        if v.betasq >= 1u64 << (logq() - 1) {
            return Err(format!(
                "vector {i}: betasq {} exceeds 2^(LOGQ-1)",
                v.betasq
            ));
        }
    }
    for (ci, c) in stmt.constraints.iter().enumerate() {
        if c.blocks.is_empty() {
            return Err(format!("constraint {ci}: no blocks"));
        }
        for (j, blk) in c.blocks.iter().enumerate() {
            if blk.idx >= stmt.vectors.len() {
                return Err(format!(
                    "constraint {ci}: block {j} idx {} out of range",
                    blk.idx
                ));
            }
            if blk.len == 0 {
                return Err(format!("constraint {ci}: block {j} has length 0"));
            }
            let end = blk
                .off
                .checked_add(extlen(blk.len, c.deg.max(1)))
                .ok_or_else(|| format!("constraint {ci}: block {j} off+extlen overflowed usize"))?;
            if end > stmt.vectors[blk.idx].n {
                return Err(format!(
                    "constraint {ci}: block {j} spans [{}, {end}) of vector {} (rank {}); \
                     a degree-{} constraint reads and writes extlen(len, deg) = {} elements",
                    blk.off,
                    blk.idx,
                    stmt.vectors[blk.idx].n,
                    c.deg,
                    extlen(blk.len, c.deg.max(1))
                ));
            }
        }
        if let PhiSource::Blocks(parts) = &c.phi {
            if parts.len() != c.blocks.len() {
                return Err(format!(
                    "constraint {ci}: {} phi blocks for {} witness blocks",
                    parts.len(),
                    c.blocks.len()
                ));
            }
            for (j, (part, blk)) in parts.iter().zip(&c.blocks).enumerate() {
                match part {
                    PhiBlock::Polx(buf, off) => {
                        if off + philen(blk.len, c.deg) > buf.len() {
                            return Err(format!(
                                "constraint {ci}: phi block {j} runs past its buffer"
                            ));
                        }
                    }
                    PhiBlock::Short(sp, off) => {
                        if sp.off + sp.wid > N {
                            return Err(format!(
                                "constraint {ci}: short phi block {j} spans coefficients [{}, {}) of {N}",
                                sp.off,
                                sp.off + sp.wid
                            ));
                        }
                        if off + philen(blk.len, c.deg) > sp.len() {
                            return Err(format!(
                                "constraint {ci}: short phi block {j} runs past its buffer"
                            ));
                        }
                    }
                }
            }
        }
        let want = c.phi_len();
        if c.phi.source_len() < want {
            return Err(format!(
                "constraint {ci}: phi has {} polynomials, needs {want} (sum of extlen(len, deg))",
                c.phi.source_len()
            ));
        }
        if let PhiSource::Int64(v) = &c.phi {
            if v.iter().flatten().any(|&x| x < 0 || x >= q) {
                return Err(format!("constraint {ci}: phi coefficient outside [0, q)"));
            }
        }
        match &c.b {
            None => {}
            Some(b) => {
                if b.len() != c.deg.max(1) {
                    return Err(format!(
                        "constraint {ci}: b has {} polynomials, needs max(1, deg) = {}",
                        b.len(),
                        c.deg.max(1)
                    ));
                }
                if let BSource::Int64(v) = b {
                    if v.iter().flatten().any(|&x| x < 0 || x >= q) {
                        return Err(format!("constraint {ci}: b coefficient outside [0, q)"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_witness(stmt: &Statement, wit: &Witness) -> Result<(), String> {
    if stmt.vectors.len() != wit.vectors.len() {
        return Err(format!(
            "statement has {} vectors, witness has {}",
            stmt.vectors.len(),
            wit.vectors.len()
        ));
    }
    for (i, (v, coeffs)) in stmt.vectors.iter().zip(wit.vectors.iter()).enumerate() {
        let expected =
            v.n.checked_mul(N)
                .ok_or_else(|| format!("vector {i}: n*64 overflowed usize"))?;
        if coeffs.len() != expected {
            return Err(format!(
                "vector {i}: witness has {} coefficients, statement rank {} wants {expected}",
                coeffs.len(),
                v.n
            ));
        }
        if v.binary {
            if let Some(&c) = coeffs.iter().find(|&&c| c != 0 && c != 1) {
                return Err(format!("vector {i}: binary vector has coefficient {c}"));
            }
        } else {
            for &c in coeffs {
                if c == i16::MIN || c.abs() > WITNESS_COEFF_MAX {
                    return Err(format!(
                        "vector {i}: |coeff| exceeds {WITNESS_COEFF_MAX}, LaBRADOR's norm \
                         accumulator would overflow"
                    ));
                }
            }
            let normsq = wit.normsq(i);
            if normsq > v.betasq {
                return Err(format!(
                    "vector {i}: normsq {normsq} exceeds betasq {}",
                    v.betasq
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------------------

fn build_raw_statement(stmt: &Statement) -> Result<RawStatement, String> {
    let n: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();
    let betasq: Vec<u64> = stmt.vectors.iter().map(VectorSpec::c_betasq).collect();
    let ptr = unsafe { ffi::bn_alloc_smplstmnt() };
    if ptr.is_null() {
        return Err("out of memory allocating smplstmnt".into());
    }
    let raw = RawStatement(ptr);
    let ret = unsafe {
        ffi::labrador48_init_smplstmnt_raw(
            raw.0,
            n.len(),
            n.as_ptr(),
            betasq.as_ptr(),
            stmt.constraints.len(),
        )
    };
    if ret != 0 {
        return Err(format!("init_smplstmnt_raw failed with code {ret}"));
    }
    unsafe { ffi::bn_smplstmnt_set_digest(raw.0, stmt.digest.as_ptr()) };

    for (ci, c) in stmt.constraints.iter().enumerate() {
        let idx: Vec<usize> = c.blocks.iter().map(|b| b.idx).collect();
        let off: Vec<usize> = c.blocks.iter().map(|b| b.off).collect();
        let len: Vec<usize> = c.blocks.iter().map(|b| b.len).collect();
        let p = c.phi_ptrs();
        let b_owned: Option<PolxBuf> = match &c.b {
            Some(BSource::Int64(v)) => Some(PolxBuf::from_int64(v)),
            _ => None,
        };
        let b_ptr = match (&c.b, &b_owned) {
            (_, Some(buf)) => buf.as_ptr(),
            (Some(BSource::Polx(buf)), _) => buf.as_ptr(),
            (None, _) => std::ptr::null(),
            (Some(BSource::Int64(_)), None) => unreachable!(),
        };
        let ret = unsafe {
            ffi::bn_smplstmnt_set_constraint(
                raw.0,
                ci,
                c.deg,
                c.blocks.len(),
                idx.as_ptr(),
                off.as_ptr(),
                len.as_ptr(),
                p.phi.as_ptr(),
                p.short_ptr(),
                p.soff.as_ptr(),
                p.swid.as_ptr(),
                b_ptr,
            )
        };
        if ret != 0 {
            return Err(format!(
                "bn_smplstmnt_set_constraint({ci}) failed with code {ret}"
            ));
        }
    }
    Ok(raw)
}

fn build_raw_witness(stmt: &Statement, wit: &Witness) -> Result<RawWitness, String> {
    let n: Vec<usize> = stmt.vectors.iter().map(|v| v.n).collect();
    let ptr = unsafe { ffi::bn_alloc_witness() };
    if ptr.is_null() {
        return Err("out of memory allocating witness".into());
    }
    let raw = RawWitness(ptr);
    unsafe { ffi::labrador48_init_witness_raw(raw.0, n.len(), n.as_ptr()) };
    for (i, (v, coeffs)) in stmt.vectors.iter().zip(wit.vectors.iter()).enumerate() {
        let ret = unsafe { ffi::bn_set_witness_i16(raw.0, i, v.n, coeffs.as_ptr()) };
        if ret != 0 {
            return Err(format!("bn_set_witness_i16({i}) failed with code {ret}"));
        }
        if !v.binary {
            let normsq = unsafe { ffi::bn_witness_normsq(raw.0, i) };
            if normsq > v.betasq {
                return Err(format!(
                    "vector {i}: LaBRADOR recomputed normsq {normsq}, statement betasq {}",
                    v.betasq
                ));
            }
        }
    }
    Ok(raw)
}

// ---------------------------------------------------------------------------------------
// prove / verify
// ---------------------------------------------------------------------------------------

/// Check the statement and witness, then `composite_prove_simple`.
pub fn prove(stmt: &Statement, wit: &Witness) -> Result<ProofHandle, String> {
    prove_inner(stmt, wit, false)
}

/// The same with LaBRADOR's `simple_verify` first: it converts the whole witness to `polx` and
/// evaluates every constraint against it, which costs as much as an aggregation pass and tells
/// an honest prover nothing it does not already know. Tests use it; [`prove`] does not.
pub fn prove_verified(stmt: &Statement, wit: &Witness) -> Result<ProofHandle, String> {
    prove_inner(stmt, wit, true)
}

fn prove_inner(stmt: &Statement, wit: &Witness, verify: bool) -> Result<ProofHandle, String> {
    check_statement(stmt)?;
    check_witness(stmt, wit)?;

    let _guard = labrador_lock();
    let _quiet = Quiet::new();
    let raw_stmt = build_raw_statement(stmt)?;
    let raw_wit = build_raw_witness(stmt, wit)?;

    if verify {
        let ret = unsafe { ffi::labrador48_simple_verify(raw_stmt.0, raw_wit.0) };
        if ret != 0 {
            return Err(format!("simple_verify: FAIL (code {ret})"));
        }
    }

    let composite = unsafe { ffi::bn_alloc_composite() };
    let commitment = unsafe { ffi::bn_alloc_commitment() };
    if composite.is_null() || commitment.is_null() {
        unsafe {
            ffi::bn_free(composite);
            ffi::bn_free(commitment);
        }
        return Err("out of memory allocating composite/commitment".into());
    }
    let composite_guard = RawComposite(composite);
    let commitment_guard = RawCommitment(commitment);
    let ret = unsafe {
        ffi::labrador48_composite_prove_simple(
            composite_guard.0,
            commitment_guard.0,
            raw_stmt.0,
            raw_wit.0,
        )
    };
    if ret != 0 {
        return Err(format!("composite_prove_simple: FAIL (code {ret})"));
    }
    let size_kb = unsafe { ffi::bn_composite_size(composite_guard.0) };

    let composite = composite_guard.0;
    let commitment = commitment_guard.0;
    std::mem::forget(composite_guard);
    std::mem::forget(commitment_guard);
    Ok(ProofHandle {
        composite,
        commitment,
        size_kb,
    })
}

/// Rebuild the statement and run `composite_verify_simple` against it.
pub fn verify(stmt: &Statement, proof: &ProofHandle) -> Result<(), String> {
    check_statement(stmt)?;

    let _guard = labrador_lock();
    let _quiet = Quiet::new();
    let raw_stmt = build_raw_statement(stmt)?;
    let ret = unsafe {
        ffi::labrador48_composite_verify_simple(proof.composite, proof.commitment, raw_stmt.0)
    };
    if ret != 0 {
        return Err(format!("composite_verify_simple: FAIL (code {ret})"));
    }
    Ok(())
}

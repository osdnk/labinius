//! `polx` buffers owned by Rust.
//!
//! `polx` is LaBRADOR's working representation: one ring element held as `K` NTT images,
//! one per RNS prime (`K = 7` at `LOGQ = 40`, so `sizeof(polx) = 896`). Constraints are
//! consumed by the prover in exactly this form, so everything the caller can hoist out of
//! the proving loop -- constraint coefficients, commitment keys, commitments -- is built
//! once as a [`PolxBuf`] and afterwards only ever aliased by pointer.

use std::ffi::c_void;
use std::os::raw::c_int;

use super::ffi;

/// Ring degree `N` of `Z_q[X]/(X^N + 1)`.
pub const N: usize = 64;

/// `sizeof(polx)` for the linked LaBRADOR build.
pub fn sizeof_polx() -> usize {
    unsafe { ffi::bn_sizeof_polx() }
}

/// `LOGQ` of the linked LaBRADOR build.
pub fn logq() -> usize {
    unsafe { ffi::bn_logq() }
}

/// The prime `q = 2^LOGQ - QOFF` of the linked LaBRADOR build.
pub fn compiled_q() -> u64 {
    unsafe { ffi::bn_compiled_q() }
}

/// LaBRADOR's `extlen(len, deg)`: `len` rounded up to a multiple of the next power of two
/// at or above `deg` (and `len` itself when `deg == 1`).
///
/// A degree-`deg` constraint consumes its `phi` block in chunks of `next2power(deg)`, so a
/// block of `len` witness polynomials needs `extlen(len, deg)` coefficient polynomials.
pub fn extlen(len: usize, deg: usize) -> usize {
    unsafe { ffi::bn_extlen(len, deg) }
}

/// Length of LaBRADOR's global commitment key, in `polx`.
pub fn comkey_len() -> usize {
    unsafe { ffi::bn_comkey_len() }
}

/// A 64-byte aligned, Rust-owned array of `len` `polx`.
pub struct PolxBuf {
    ptr: *mut c_void,
    len: usize,
}

// The buffer is a plain owned allocation; every mutation goes through `&mut self`.
unsafe impl Send for PolxBuf {}
unsafe impl Sync for PolxBuf {}

impl PolxBuf {
    /// A zeroed buffer of `len` `polx`.
    pub fn zeroed(len: usize) -> Self {
        let ptr = unsafe { ffi::bn_polx_alloc(len) };
        assert!(!ptr.is_null(), "out of memory allocating {len} polx");
        unsafe { ffi::bn_polx_setzero(ptr, len) };
        Self { ptr, len }
    }

    /// Convert `polys.len()` coefficient polynomials, each `N` coefficients in `[0, q)` or
    /// centred, into `polx`. This is the bulk `polxvec_fromint64vec` path.
    pub fn from_int64(polys: &[[i64; N]]) -> Self {
        let len = polys.len();
        let buf = Self::alloc(len);
        unsafe { ffi::bn_polx_from_int64(buf.ptr, len, polys.as_ptr().cast::<i64>()) };
        buf
    }

    /// Convert `polys.len()` small coefficient polynomials into `polx` without an `int64`
    /// detour (`polyvec_fromint64vec` is skipped in favour of a direct `int16` fill plus
    /// `polxvec_frompolyvec`). Coefficients are taken as centred representatives.
    pub fn from_int16(polys: &[[i16; N]]) -> Self {
        let len = polys.len();
        let buf = Self::alloc(len);
        unsafe { ffi::bn_polx_from_int16(buf.ptr, len, polys.as_ptr().cast::<i16>()) };
        buf
    }

    /// The same flat form, as one `i16` slice of `len * N` coefficients.
    pub fn from_int16_flat(len: usize, coeffs: &[i16]) -> Self {
        assert_eq!(coeffs.len(), len * N, "expected {} coefficients", len * N);
        let buf = Self::alloc(len);
        unsafe { ffi::bn_polx_from_int16(buf.ptr, len, coeffs.as_ptr()) };
        buf
    }

    /// Expand `len` almost-uniform `polx` from a 16-byte seed and a nonce, with the routine
    /// (`polxvec_almostuniform`) that LaBRADOR's own `init_comkey` uses.
    pub fn expand(len: usize, seed: &[u8; 16], nonce: u64) -> Self {
        let buf = Self::alloc(len);
        unsafe { ffi::bn_polx_expand(buf.ptr, len, seed.as_ptr(), nonce) };
        buf
    }

    fn alloc(len: usize) -> Self {
        let ptr = unsafe { ffi::bn_polx_alloc(len) };
        assert!(!ptr.is_null(), "out of memory allocating {len} polx");
        Self { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Pointer to element `off`. Panics unless `off <= len`.
    pub fn ptr_at(&self, off: usize) -> *const c_void {
        assert!(off <= self.len, "polx offset {off} out of range (len {})", self.len);
        unsafe { self.ptr.cast::<u8>().add(off * sizeof_polx()).cast::<c_void>() }
    }

    /// The raw bytes of the buffer, for hashing it into a statement digest.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len * sizeof_polx()) }
    }

    pub fn copy_from(&mut self, src: &PolxBuf) {
        assert_eq!(self.len, src.len, "polx length mismatch");
        unsafe { ffi::bn_polx_copy(self.ptr, src.ptr, self.len) };
    }
}

impl Drop for PolxBuf {
    fn drop(&mut self) {
        unsafe { ffi::bn_polx_free(self.ptr) };
    }
}

impl PartialEq for PolxBuf {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && unsafe { ffi::bn_polx_eq(self.ptr, other.ptr, self.len) as c_int != 0 }
    }
}

impl Eq for PolxBuf {}

impl std::fmt::Debug for PolxBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolxBuf({} polx)", self.len)
    }
}

/// A witness in `polx` form: every vector converted once, so that constraint evaluation
/// (`b = <phi, s>`) over many constraints pays the `poly -> polx` NTT only once.
pub struct Sx {
    ptr: *mut c_void,
    n: Vec<usize>,
}

unsafe impl Send for Sx {}

impl Sx {
    /// Convert a witness (one `Vec<i16>` of `n * N` coefficients per vector) to `polx`.
    pub fn new(vectors: &[Vec<i16>]) -> Self {
        let n: Vec<usize> = vectors.iter().map(|v| v.len() / N).collect();
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(v.len(), n[i] * N, "witness vector {i} is not a whole number of polynomials");
        }
        let ptrs: Vec<*const i16> = vectors.iter().map(|v| v.as_ptr()).collect();
        let ptr = unsafe { ffi::bn_sx_new(n.len(), n.as_ptr(), ptrs.as_ptr()) };
        assert!(!ptr.is_null() || n.is_empty(), "out of memory allocating sx");
        Self { ptr, n }
    }

    /// Pointer to polynomial `off` of vector `i`.
    pub fn ptr(&self, i: usize, off: usize) -> *const c_void {
        assert!(i < self.n.len(), "sx vector {i} out of range");
        assert!(off <= self.n[i], "sx offset {off} out of range for vector {i}");
        unsafe { ffi::bn_sx_ptr(self.ptr, i, off) }
    }

    pub fn ranks(&self) -> &[usize] {
        &self.n
    }

    /// The raw `polx *[]` array LaBRADOR's evaluators take.
    pub(crate) fn raw(&self) -> *const c_void {
        self.ptr
    }
}

impl Drop for Sx {
    fn drop(&mut self) {
        unsafe { ffi::bn_sx_free(self.ptr) };
    }
}

/// An Ajtai commitment key: `kappa` rows of `n` ring elements, laid out exactly as
/// LaBRADOR's global `comkey` is, i.e. as `extlen(n, kappa)` `polx` consumed by the
/// truncated extension product.
#[derive(Debug)]
pub struct CommitmentKey {
    buf: std::sync::Arc<PolxBuf>,
    n: usize,
    kappa: usize,
}

impl CommitmentKey {
    /// Expand a key for rank `kappa` and length `n` from a 16-byte seed and a nonce.
    pub fn expand(n: usize, kappa: usize, seed: &[u8; 16], nonce: u64) -> Self {
        assert!(kappa >= 1, "commitment rank must be at least 1");
        Self { buf: std::sync::Arc::new(PolxBuf::expand(extlen(n, kappa), seed, nonce)), n, kappa }
    }

    pub fn rank(&self) -> usize {
        self.kappa
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn buf(&self) -> &PolxBuf {
        &self.buf
    }

    /// A shared handle to the key, to alias as a constraint's `phi`.
    pub fn buf_arc(&self) -> std::sync::Arc<PolxBuf> {
        std::sync::Arc::clone(&self.buf)
    }

    /// `u = A s`, the truncated extension product `polxvec_mul_extension(u, key, s, n, kappa, 1)`
    /// that LaBRADOR's `commit_raw` forms. Returns `kappa` `polx`.
    ///
    /// `s` is a witness vector of `n * N` `i16` coefficients.
    pub fn commit_i16(&self, s: &[i16]) -> PolxBuf {
        assert_eq!(s.len(), self.n * N, "expected {} coefficients", self.n * N);
        let out = PolxBuf::alloc(self.kappa);
        unsafe { ffi::bn_commit_i16(out.ptr, self.buf.ptr, s.as_ptr(), self.n, self.kappa) };
        out
    }

    /// The same product against a slice already in `polx` form (`sx[i][off..off+n]`).
    pub fn commit_sx(&self, sx: &Sx, i: usize, off: usize) -> PolxBuf {
        let out = PolxBuf::alloc(self.kappa);
        unsafe { ffi::bn_commit_polx(out.ptr, self.buf.ptr, sx.ptr(i, off), self.n, self.kappa) };
        out
    }
}

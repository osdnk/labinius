//! Raw declarations for `liblabrador.a` (built at `LOGQ = 48`, symbol prefix `labrador48_`)
//! and for the `csrc/bn_labrador.c` shim.
//!
//! Only the symbols LaBRADOR marks `visibility("default")` carry the `labrador48_` prefix;
//! everything else (`init_sparsecnst_half`, `polxvec_*`, `sparsecnst_eval`, ...) keeps its
//! plain name and is reached through the shim rather than from here.

use std::ffi::c_void;
use std::os::raw::c_int;

extern "C" {
    pub fn labrador48_init_comkey(n: usize);
    pub fn labrador48_free_comkey();
    pub fn labrador48_init_witness_raw(wt: *mut c_void, r: usize, n: *const usize);
    pub fn labrador48_free_witness(wt: *mut c_void);
    pub fn labrador48_init_smplstmnt_raw(
        st: *mut c_void,
        r: usize,
        n: *const usize,
        betasq: *const u64,
        k: usize,
    ) -> c_int;
    pub fn labrador48_free_smplstmnt(st: *mut c_void);
    pub fn labrador48_free_commitment(com: *mut c_void);
    pub fn labrador48_free_composite(p: *mut c_void);
    pub fn labrador48_simple_verify(st: *const c_void, wt: *const c_void) -> c_int;
    pub fn labrador48_composite_prove_simple(
        p: *mut c_void,
        com: *mut c_void,
        st: *const c_void,
        wt: *const c_void,
    ) -> c_int;
    pub fn labrador48_composite_verify_simple(
        p: *const c_void,
        com: *const c_void,
        st: *const c_void,
    ) -> c_int;

    pub fn bn_sizeof_polx() -> usize;
    pub fn bn_polx_align() -> usize;
    pub fn bn_ring_degree() -> usize;
    pub fn bn_logq() -> usize;
    pub fn bn_compiled_q() -> u64;
    pub fn bn_extlen(len: usize, deg: usize) -> usize;
    pub fn bn_comkey_len() -> usize;

    pub fn bn_polx_alloc(len: usize) -> *mut c_void;
    pub fn bn_polx_free(p: *mut c_void);
    pub fn bn_polx_setzero(r: *mut c_void, len: usize);
    pub fn bn_polx_copy(r: *mut c_void, a: *const c_void, len: usize);
    pub fn bn_polx_eq(a: *const c_void, b: *const c_void, len: usize) -> c_int;
    pub fn bn_polx_from_int64(r: *mut c_void, len: usize, a: *const i64);
    pub fn bn_polx_from_int16(r: *mut c_void, len: usize, a: *const i16);
    pub fn bn_polx_expand(r: *mut c_void, len: usize, seed: *const u8, nonce: u64);

    pub fn bn_sx_new(r: usize, n: *const usize, s: *const *const i16) -> *mut c_void;
    pub fn bn_sx_free(sx: *mut c_void);
    pub fn bn_sx_ptr(sx: *const c_void, i: usize, off: usize) -> *const c_void;

    pub fn bn_commit_polx(out: *mut c_void, key: *const c_void, s: *const c_void, len: usize, deg: usize);
    pub fn bn_commit_i16(out: *mut c_void, key: *const c_void, s: *const i16, len: usize, deg: usize);
    pub fn bn_commit_blocks(
        out: *mut c_void,
        key: *const c_void,
        deg: usize,
        nb: usize,
        len: *const usize,
        s: *const *const i16,
    );
    pub fn bn_polx_table_sum(
        out: *mut c_void,
        len: usize,
        table: *const c_void,
        terms: usize,
        idx: *const u16,
        sign: *const i8,
    );
    pub fn bn_sis_secure(rank: usize, norm: f64) -> c_int;

    pub fn bn_eval_blocks(
        out: *mut c_void,
        deg: usize,
        nz: usize,
        idx: *const usize,
        off: *const usize,
        len: *const usize,
        phi: *const *const c_void,
        sx: *const c_void,
    );

    pub fn bn_alloc_smplstmnt() -> *mut c_void;
    pub fn bn_alloc_witness() -> *mut c_void;
    pub fn bn_alloc_composite() -> *mut c_void;
    pub fn bn_alloc_commitment() -> *mut c_void;
    pub fn bn_free(p: *mut c_void);

    pub fn bn_smplstmnt_set_digest(st: *mut c_void, digest: *const u8);
    pub fn bn_smplstmnt_set_constraint(
        st: *mut c_void,
        ci: usize,
        deg: usize,
        nz: usize,
        idx: *const usize,
        off: *const usize,
        len: *const usize,
        phi: *const *const c_void,
        b: *const c_void,
    ) -> c_int;

    pub fn bn_set_witness_i16(wt: *mut c_void, i: usize, n: usize, s: *const i16) -> c_int;
    pub fn bn_witness_normsq(wt: *const c_void, i: usize) -> u64;
    pub fn bn_composite_size(p: *const c_void) -> f64;
}

#ifndef BN_LABRADOR_H
#define BN_LABRADOR_H

#include <stdint.h>
#include <stddef.h>

/* Compile-time facts about the LaBRADOR build this shim is linked against. */
size_t bn_sizeof_polx(void);
size_t bn_polx_align(void);
size_t bn_ring_degree(void);
size_t bn_logq(void);
uint64_t bn_compiled_q(void);
size_t bn_extlen(size_t len, size_t deg);
size_t bn_comkey_len(void);

/* Opaque, 64-byte aligned polx buffers owned by Rust. */
void *bn_polx_alloc(size_t len);
void bn_polx_free(void *p);
void bn_polx_setzero(void *r, size_t len);
void bn_polx_copy(void *r, const void *a, size_t len);
int bn_polx_eq(const void *a, const void *b, size_t len);

/* Bulk conversions into polx. */
void bn_polx_from_int64(void *r, size_t len, const int64_t *a);
void bn_polx_from_int16(void *r, size_t len, const int16_t *a);
/* Expand `len` almost-uniform polx from a 16-byte seed; same routine as init_comkey(). */
void bn_polx_expand(void *r, size_t len, const uint8_t seed[16], uint64_t nonce);

/* Witness slices in polx form, converted once and shared by every constraint. */
void *bn_sx_new(size_t r, const size_t *n, const int16_t *const *s);
void bn_sx_free(void *sx);
const void *bn_sx_ptr(const void *sx, size_t i, size_t off);

/* Ajtai commitment: out[deg] = <key, s> in the truncated degree-`deg` extension,
 * the exact product commit_raw() forms against the global commitment key. */
void bn_commit_polx(void *out, const void *key, const void *s, size_t len, size_t deg);
void bn_commit_i16(void *out, const void *key, const int16_t *s, size_t len, size_t deg);

/* Linear form of one constraint against a prepared sx; writes MAX(1,deg) polx. */
void bn_eval_blocks(void *out, size_t deg, size_t nz, const size_t *idx, const size_t *off,
                    const size_t *len, const void *const *phi, const void *sx);

/* Opaque allocation of the LaBRADOR objects (Rust never sees their layout). */
void *bn_alloc_smplstmnt(void);
void *bn_alloc_witness(void);
void *bn_alloc_composite(void);
void *bn_alloc_commitment(void);
void bn_free(void *p);

/* Statement. The digest is the only binding of the statement contents. */
void bn_smplstmnt_set_digest(void *st, const uint8_t digest[32]);
int bn_smplstmnt_set_constraint(void *st, size_t ci, size_t deg, size_t nz,
                                const size_t *idx, const size_t *off, const size_t *len,
                                const void *const *phi, const void *b);

/* Witness. */
int bn_set_witness_i16(void *wt, size_t i, size_t n, const int16_t *s);
uint64_t bn_witness_normsq(const void *wt, size_t i);

double bn_composite_size(const void *composite);

#endif

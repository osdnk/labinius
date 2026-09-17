#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include "data.h"
#include "malloc.h"
#include "fips202.h"
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "sparsemat.h"
#include "labrador.h"
#include "chihuahua.h"
#include "dachshund.h"
#include "pack.h"
#include "bn_labrador.h"

size_t bn_sizeof_polx(void) { return sizeof(polx); }
size_t bn_polx_align(void) { return 64; }
size_t bn_ring_degree(void) { return N; }
size_t bn_logq(void) { return LOGQ; }
uint64_t bn_compiled_q(void) { return ((uint64_t)1 << LOGQ) - QOFF; }
size_t bn_extlen(size_t len, size_t deg) { return extlen(len,deg); }
size_t bn_comkey_len(void) { return comkey_len; }

void *bn_polx_alloc(size_t len) {
  if(!len) len = 1;
  return aligned_alloc(64,len*sizeof(polx));
}

void bn_polx_free(void *p) { free(p); }

void bn_polx_setzero(void *r, size_t len) { polxvec_setzero((polx*)r,len); }

void bn_polx_copy(void *r, const void *a, size_t len) { polxvec_copy((polx*)r,(const polx*)a,len); }

int bn_polx_eq(const void *a, const void *b, size_t len) {
  size_t i;
  polx t[1];

  for(i=0;i<len;i++) {
    polx_sub(t,&((const polx*)a)[i],&((const polx*)b)[i]);
    if(!polx_iszero(t)) return 0;
  }
  return 1;
}

void bn_polx_from_int64(void *r, size_t len, const int64_t *a) {
  polxvec_fromint64vec((polx*)r,len,1,a);
}

void bn_polx_from_int16(void *r, size_t len, const int16_t *a) {
  size_t i,k,m;
  polx *rr = (polx*)r;
  __attribute__((aligned(64)))
  poly t[32];

  while(len) {
    m = MIN(len,32);
    for(i=0;i<m;i++)
      for(k=0;k<N;k++)
        t[i].vec->c[k] = a[i*N+k];
    polxvec_frompolyvec(rr,t,m);
    rr += m;
    a += m*N;
    len -= m;
  }
}

void bn_polx_expand(void *r, size_t len, const uint8_t seed[16], uint64_t nonce) {
  polxvec_almostuniform((polx*)r,len,seed,nonce);
}

void *bn_sx_new(size_t r, const size_t *n, const int16_t *const *s) {
  size_t i,k;
  polx **sx;
  polx *buf;

  if(!r) return NULL;
  k = 0;
  for(i=0;i<r;i++)
    k += n[i];

  sx = _malloc(r*sizeof(polx*));
  buf = _aligned_alloc(64,MAX(k,1)*sizeof(polx));
  for(i=0;i<r;i++) {
    sx[i] = buf;
    bn_polx_from_int16(buf,n[i],s[i]);
    buf += n[i];
  }
  return sx;
}

void bn_sx_free(void *sxp) {
  polx **sx = sxp;

  if(!sx) return;
  free(sx[0]);
  free(sx);
}

const void *bn_sx_ptr(const void *sxp, size_t i, size_t off) {
  polx *const *sx = sxp;

  return &sx[i][off];
}

void bn_commit_polx(void *out, const void *key, const void *s, size_t len, size_t deg) {
  polxvec_mul_extension((polx*)out,(const polx*)key,(const polx*)s,len,deg,1,SPROD_WORST);
}

void bn_commit_i16(void *out, const void *key, const int16_t *s, size_t len, size_t deg) {
  polx *sx = _aligned_alloc(64,MAX(len,1)*sizeof(polx));

  bn_polx_from_int16(sx,len,s);
  polxvec_mul_extension((polx*)out,(const polx*)key,sx,len,deg,1,SPROD_WORST);
  free(sx);
}

void bn_eval_blocks(void *out, size_t deg, size_t nz, const size_t *idx, const size_t *off,
                    const size_t *len, const void *const *phi,
                    const int16_t *const *sphi, const size_t *soff, const size_t *swid,
                    const void *sxp)
{
  size_t j,elen,cap = 0;
  const size_t deg2 = MAX(1,deg);
  polx *const *sx = sxp;
  polx *o = out;
  polx *scratch = NULL;
  polx t[deg2];

  polxvec_setzero(o,deg2);
  for(j=0;j<nz;j++) {
    const polx *p = (const polx*)phi[j];
    if(sphi && sphi[j]) {
      shortphi sp = {sphi[j],soff[j],swid[j]};
      elen = extlen(len[j],deg2);
      if(elen > cap) {
        free(scratch);
        scratch = _aligned_alloc(64,elen*sizeof(polx));
        cap = elen;
      }
      polxvec_setzero(scratch,elen);
      shortphi_topolxvec(scratch,&sp,0,len[j]);
      p = scratch;
    }
    polxvec_mul_extension(t,p,&sx[idx[j]][off[j]],len[j],deg2,1,SPROD_WORST);
    polxvec_add(o,o,t,deg2);
  }
  free(scratch);
}

void *bn_alloc_smplstmnt(void) { return calloc(1,sizeof(smplstmnt)); }
void *bn_alloc_witness(void) { return calloc(1,sizeof(witness)); }
void *bn_alloc_composite(void) { return calloc(1,sizeof(composite)); }
void *bn_alloc_commitment(void) { return calloc(1,sizeof(commitment)); }
void bn_free(void *p) { free(p); }

void bn_smplstmnt_set_digest(void *stp, const uint8_t digest[32]) {
  shake128(((smplstmnt*)stp)->h,16,digest,32);
}

int bn_smplstmnt_set_constraint(void *stp, size_t ci, size_t deg, size_t nz,
                                const size_t *idx, const size_t *off, const size_t *len,
                                const void *const *phi,
                                const int16_t *const *sphi, const size_t *soff, const size_t *swid,
                                const void *b)
{
  smplstmnt *st = stp;
  sparsecnst *cnst;
  size_t j;

  if(ci >= st->k) return 1;
  cnst = &st->cnst[ci];
  if(cnst->idx) return 2;
  for(j=0;j<nz;j++)
    if(idx[j] >= st->r) return 3;

  /* buflen 0: the only storage init_sparsecnst_half() reserves is the MAX(1,deg) polx
   * for b; every phi block below aliases a caller-owned buffer, and free_sparsecnst()
   * frees nothing but cnst->idx (and the quadratic part, which is absent here). */
  init_sparsecnst_half(cnst,st->r,nz,0,deg,0,b == NULL);
  for(j=0;j<nz;j++) {
    cnst->idx[j] = idx[j];
    cnst->off[j] = off[j];
    cnst->len[j] = len[j];
    cnst->mult[j] = 1;
    cnst->phi[j] = (polx*)phi[j];
    if(sphi && sphi[j]) {
      cnst->sphi[j].c = sphi[j];
      cnst->sphi[j].off = soff[j];
      cnst->sphi[j].wid = swid[j];
    }
  }
  cnst->a->len = 0;
  if(b)
    polxvec_copy(cnst->b,(const polx*)b,MAX(1,deg));
  return 0;
}

int bn_set_witness_i16(void *wtp, size_t i, size_t n, const int16_t *s) {
  witness *wt = wtp;
  size_t j,k;

  if(i >= wt->r) return 1;
  if(n != wt->n[i]) return 2;
  for(j=0;j<n;j++)
    for(k=0;k<N;k++)
      wt->s[i][j].vec->c[k] = s[j*N+k];
  wt->normsq[i] = polyvec_sprodz(wt->s[i],wt->s[i],n);
  return 0;
}

uint64_t bn_witness_normsq(const void *wtp, size_t i) {
  return ((const witness*)wtp)->normsq[i];
}

int bn_sis_secure(size_t rank, double norm) { return sis_secure(rank,norm); }

void bn_commit_blocks(void *out, const void *key, size_t deg, size_t nb, const size_t *len,
                      const int16_t *const *s)
{
  size_t j,mx = 0,off = 0;
  polx *o = out;
  polx *sx;
  polx t[deg];

  for(j=0;j<nb;j++)
    mx = MAX(mx,len[j]);
  sx = _aligned_alloc(64,MAX(mx,1)*sizeof(polx));
  polxvec_setzero(o,deg);
  for(j=0;j<nb;j++) {
    bn_polx_from_int16(sx,len[j],s[j]);
    polxvec_mul_extension(t,(const polx*)key + off,sx,len[j],deg,1,SPROD_WORST);
    polxvec_add(o,o,t,deg);
    off += extlen(len[j],deg);
  }
  free(sx);
}

void bn_polx_table_sum(void *out, size_t len, const void *table, size_t terms,
                       const uint16_t *idx, const int8_t *sign)
{
  size_t i,k;
  polx *o = out;
  const polx *t = table;

  for(i=0;i<len;i++) {
    polxvec_setzero(&o[i],1);
    for(k=0;k<terms;k++) {
      if(sign[k] > 0) polx_add(&o[i],&o[i],&t[idx[i*terms+k]]);
      else if(sign[k] < 0) polx_sub(&o[i],&o[i],&t[idx[i*terms+k]]);
    }
  }
}

double bn_composite_size(const void *cp) {
  return ((const composite*)cp)->size;
}

/* -------------------------------------------------------------------------------------
 * stdout muting
 *
 * LaBRADOR prints a page of statement and proof-size chatter per recursion level. The
 * crate reports its own timings, so the shim redirects fd 1 to /dev/null around the FFI
 * calls and restores it afterwards; stderr, which carries the library's error messages,
 * is untouched.
 * ----------------------------------------------------------------------------------- */

static int bn_saved_stdout = -1;

void bn_mute_stdout(void) {
  int devnull;

  if(bn_saved_stdout >= 0) return;
  fflush(stdout);
  bn_saved_stdout = dup(1);
  if(bn_saved_stdout < 0) return;
  devnull = open("/dev/null",O_WRONLY);
  if(devnull < 0) { close(bn_saved_stdout); bn_saved_stdout = -1; return; }
  dup2(devnull,1);
  close(devnull);
}

void bn_unmute_stdout(void) {
  if(bn_saved_stdout < 0) return;
  fflush(stdout);
  dup2(bn_saved_stdout,1);
  close(bn_saved_stdout);
  bn_saved_stdout = -1;
}

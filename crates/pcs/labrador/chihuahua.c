#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include "malloc.h"
#include "fips202.h"
#include "aesctr.h"
#include "data.h"
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "sparsemat.h"
#include "jlproj.h"
#include "labrador.h"
#include "chihuahua.h"

polx *init_sparsecnst_half(sparsecnst *cnst, size_t r, size_t nz, size_t buflen, size_t deg,
                           int quadratic, int homogeneous)
{
  void *buf;

  cnst->deg = deg;
  cnst->nz = nz;
  cnst->a->len = 0;
  if(quadratic) {
    buf = _malloc((r*r+r)*sizeof(size_t));
    cnst->a->rows = (size_t*)buf;
    cnst->a->cols = (size_t*)buf + (r*r+r)/2;

    buf = _aligned_alloc(64,r*sizeof(polx));
    cnst->a->coeffs = (polx*)buf;
  }
  else {
    cnst->a->rows = NULL;
    cnst->a->cols = NULL;
    cnst->a->coeffs = NULL;
  }

  if(!homogeneous)
    buflen += MAX(1,deg);

  buf = _malloc(4*nz*sizeof(size_t) + nz*sizeof(polx*) + nz*sizeof(shortphi) + 63 + buflen*sizeof(polx));
  cnst->idx = (size_t*)buf;
  cnst->off = &cnst->idx[nz];
  cnst->len = &cnst->off[nz];
  cnst->mult = &cnst->len[nz];
  cnst->phi = (polx**)&cnst->mult[nz];
  cnst->sphi = (shortphi*)&cnst->phi[nz];
  memset(cnst->sphi,0,nz*sizeof(shortphi));
  buf = (void*)(((uintptr_t)&cnst->sphi[nz] + 63) & -64ULL);

  if(homogeneous)
    cnst->b = NULL;
  else {
    cnst->b = buf;
    buf = (polx*)buf + MAX(1,deg);
  }

  return buf;
}

void init_sparsecnst_raw(sparsecnst *cnst, size_t r, size_t nz, const size_t idx[nz], const size_t n[nz], size_t deg,
                         int quadratic, int homogeneous)
{
  size_t i;
  size_t buflen;
  size_t elen[nz];
  polx *buf;

  buflen = 0;
  for(i=0;i<nz;i++) {
    elen[i] = extlen(n[i]*deg,deg);
    buflen += elen[i];
  }

  buf = init_sparsecnst_half(cnst,r,nz,buflen,deg,quadratic,homogeneous);
  for(i=0;i<nz;i++) {
    cnst->idx[i] = idx[i];
    cnst->off[i] = 0;
    cnst->len[i] = n[i]*deg;
    cnst->mult[i] = 1;
    cnst->phi[i] = (polx*)buf;
    buf += elen[i];
  }
}

int set_sparsecnst_raw(sparsecnst *cnst, uint8_t h[16], size_t nz, const size_t idx[nz],
                       const size_t n[nz], size_t deg, int64_t *phi, int64_t *b)
{
  size_t j,k;
  polz t[deg];
  __attribute__((aligned(16)))
  uint8_t hashbuf[deg*N*QBYTES];
  shake128incctx shakectx;

  if(nz != cnst->nz) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in number of affected vectors\n");
    return 1;
  }
  if(deg != cnst->deg) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in extension degree\n");
    return 2;
  }
  if((b && !cnst->b) || (!b && cnst->b)) {
    fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in homogeneity\n");
    return 3;
  }
  for(j=0;j<nz;j++) {
    if(idx[j] != cnst->idx[j]) {
      fprintf(stderr,"ERROR in set_sparsecnst_raw(): Mismatch in index of %zu-th affected vector\n",j);
      return 4;
    }
    if(cnst->off[j])  //FIXME
      return 5;
    if(cnst->mult[j] != 1)
      return 6;
  }

  cnst->a->len = 0;

  shake128_inc_init(&shakectx);
  shake128_inc_absorb(&shakectx,h,16);

  if(b) {
    polzvec_fromint64vec(t,1,deg,b);
    polzvec_topolxvec(cnst->b,t,deg);
    polzvec_bitpack(hashbuf,t,deg);
    shake128_inc_absorb(&shakectx,hashbuf,deg*N*QBYTES);
  }

  for(j=0;j<nz;j++) {
    for(k=0;k<n[j];k++) {
      polzvec_fromint64vec(t,1,deg,phi);
      polzvec_topolxvec(&cnst->phi[j][k*deg],t,deg);
      polzvec_bitpack(hashbuf,t,deg);
      shake128_inc_absorb(&shakectx,hashbuf,deg*N*QBYTES);
      phi += deg*N;
    }
  }

  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(h,16,&shakectx);
  return 0;
}

void free_sparsecnst(sparsecnst *cnst) {
  free(cnst->idx);
  free(cnst->a->rows);
  free(cnst->a->coeffs);
  cnst->idx = cnst->a->rows = NULL;
  cnst->a->coeffs = NULL;
}

/* The dense polx image of elements [off, off+len) of a short block. */
void shortphi_topolxvec(polx *r, const shortphi *sp, size_t off, size_t len) {
  size_t i,t,m;
  __attribute__((aligned(64)))
  poly p[32];

  while(len) {
    m = MIN(len,32);
    memset(p,0,m*sizeof(poly));
    for(i=0;i<m;i++)
      for(t=0;t<sp->wid;t++)
        p[i].vec->c[sp->off+t] = sp->c[(off+i)*sp->wid+t];
    polxvec_frompolyvec(r,p,m);
    r += m;
    off += m;
    len -= m;
  }
}

/* A dense pointer to block j, materialising a short block into *scratch (grown as
 * needed). Used by the paths that are not worth a coefficient-domain kernel of their
 * own: constraint evaluation and the constant-term collapse. */
static const polx *dense_block(const sparsecnst *cnst, size_t j, polx **scratch, size_t *cap) {
  size_t elen;

  if(!cnst->sphi || !cnst->sphi[j].c)
    return cnst->phi[j];

  elen = extlen(cnst->len[j],MAX(1,cnst->deg));
  if(elen > *cap) {
    free(*scratch);
    *scratch = _aligned_alloc(64,elen*sizeof(polx));
    *cap = elen;
  }
  polxvec_setzero(*scratch,elen);
  shortphi_topolxvec(*scratch,&cnst->sphi[j],0,cnst->len[j]);
  return *scratch;
}

void sparsecnst_eval(polx *b, const sparsecnst *cnst, polx *sx[], const witness *wt) {
  const size_t r = wt->r;
  const size_t *n = wt->n;

  size_t i,j,k;
  size_t deg2 = MAX(1,cnst->deg);
  polx t[deg2];

  if(!*sx) {
    k = 0;
    for(i=0;i<r;i++)
      k += n[i];
    sx[0] = _aligned_alloc(64,k*sizeof(polx));
    for(i=1;i<r;i++)
      sx[i] = &sx[i-1][n[i-1]];
    for(i=0;i<r;i++)
      polxvec_frompolyvec(sx[i],wt->s[i],n[i]);
  }

  polxvec_setzero(b,deg2);

  /* quadratic term */
  for(i=0;i<cnst->a->len;i++) {
    j = cnst->a->rows[i];
    k = cnst->a->cols[i];
    polxvec_sprod(t,sx[j],sx[k],MIN(n[j],n[k]),SPROD_WORST);  // TODO: store?
    polx_mul(t,t,&cnst->a->coeffs[i]);
    polx_add(b,b,t);
  }

  /* linear term */
  polx *scratch = NULL;
  size_t cap = 0;
  for(i=0;i<cnst->nz;i++) {
    j = cnst->idx[i];
    k = cnst->off[i];
    polxvec_mul_extension(t,dense_block(cnst,i,&scratch,&cap),&sx[j][k],cnst->len[i],deg2/cnst->mult[i],cnst->mult[i],SPROD_WORST);
    polxvec_add(b,b,t,deg2);
  }
  free(scratch);
}

int sparsecnst_check(const sparsecnst *cnst, polx *sx[], const witness *wt) {
  int ret;
  size_t deg2 = MAX(1,cnst->deg);
  polx b[deg2];

  sparsecnst_eval(b,cnst,sx,wt);
  if(cnst->b)
    polxvec_sub(b,b,cnst->b,deg2);

  if(cnst->deg)
    ret = polxvec_iszero(b,deg2);
  else
    ret = polx_iszero_constcoeff(b);

  return ret;
}

/* One (phi buffer, witness slice) block of one constraint, ready to be evaluated
 * in an order of our choosing. */
typedef struct {
  const polx *phi;
  const shortphi *sp;
  const polx *s;
  polx *acc;
  size_t len;
  size_t deg2;
  size_t mult;
} blockent;

static const void *blockent_key(const blockent *e) {
  return e->sp ? (const void*)e->sp->c : (const void*)e->phi;
}

static int blockent_cmp(const void *x, const void *y) {
  const void *const a = blockent_key(x);
  const void *const b = blockent_key(y);
  return (a > b) - (a < b);
}

/* Checks every constraint at once. Constraints alias each other's phi buffers, so
 * the blocks are evaluated phi-major: each distinct buffer is streamed once
 * instead of once per referencing constraint. Returns k if all hold, else the
 * index of the first that does not. */
size_t sparsecnst_check_batch(const sparsecnst *cnst, size_t k, polx *sx[], const witness *wt) {
  const size_t r = wt->r;
  const size_t *n = wt->n;

  size_t i,j,u,v,ne,tot,ret;
  polx *acc;
  blockent *ent;

  if(!*sx) {
    u = 0;
    for(i=0;i<r;i++)
      u += n[i];
    sx[0] = _aligned_alloc(64,u*sizeof(polx));
    for(i=1;i<r;i++)
      sx[i] = &sx[i-1][n[i-1]];
    for(i=0;i<r;i++)
      polxvec_frompolyvec(sx[i],wt->s[i],n[i]);
  }

  tot = 0;
  ne = 0;
  for(i=0;i<k;i++) {
    tot += MAX(1,cnst[i].deg);
    ne += cnst[i].nz;
  }
  acc = _aligned_alloc(64,MAX(tot,1)*sizeof(polx));
  polxvec_setzero(acc,tot);
  ent = _malloc(MAX(ne,1)*sizeof(blockent));

  ne = 0;
  tot = 0;
  for(i=0;i<k;i++) {
    const size_t deg2 = MAX(1,cnst[i].deg);
    for(j=0;j<cnst[i].nz;j++) {
      ent[ne].phi = cnst[i].phi[j];
      ent[ne].sp = (cnst[i].sphi && cnst[i].sphi[j].c) ? &cnst[i].sphi[j] : NULL;
      ent[ne].s = &sx[cnst[i].idx[j]][cnst[i].off[j]];
      ent[ne].acc = &acc[tot];
      ent[ne].len = cnst[i].len[j];
      ent[ne].deg2 = deg2;
      ent[ne].mult = cnst[i].mult[j];
      ne += 1;
    }
    tot += deg2;
  }

  qsort(ent,ne,sizeof(blockent),blockent_cmp);
  polx *scratch = NULL;
  size_t cap = 0;
  for(i=0;i<ne;i++) {
    polx t[ent[i].deg2];
    const polx *phi = ent[i].phi;
    if(ent[i].sp) {
      const size_t elen = extlen(ent[i].len,ent[i].deg2);
      if(elen > cap) {
        free(scratch);
        scratch = _aligned_alloc(64,elen*sizeof(polx));
        cap = elen;
      }
      polxvec_setzero(scratch,elen);
      shortphi_topolxvec(scratch,ent[i].sp,0,ent[i].len);
      phi = scratch;
    }
    polxvec_mul_extension(t,phi,ent[i].s,ent[i].len,ent[i].deg2/ent[i].mult,ent[i].mult,SPROD_WORST);
    polxvec_add(ent[i].acc,ent[i].acc,t,ent[i].deg2);
  }
  free(scratch);

  ret = k;
  tot = 0;
  for(i=0;i<k;i++) {
    const size_t deg2 = MAX(1,cnst[i].deg);
    polx *b = &acc[tot];
    polx t[deg2];
    tot += deg2;

    for(j=0;j<cnst[i].a->len;j++) {
      u = cnst[i].a->rows[j];
      v = cnst[i].a->cols[j];
      polxvec_sprod(t,sx[u],sx[v],MIN(n[u],n[v]),SPROD_WORST);
      polx_mul(t,t,&cnst[i].a->coeffs[j]);
      polx_add(b,b,t);
    }
    if(cnst[i].b)
      polxvec_sub(b,b,cnst[i].b,deg2);

    if(!(cnst[i].deg ? polxvec_iszero(b,deg2) : polx_iszero_constcoeff(b))) {
      ret = i;
      break;
    }
  }

  free(ent);
  free(acc);
  return ret;
}

int init_prncplstmnt_raw(prncplstmnt *st, size_t r, const size_t n[r],
                         uint64_t betasq, size_t k, int quadratic)
{
  size_t i;
  void *buf;

  if(betasq > JLMAXNORMSQ) {
    fprintf(stderr,"ERROR in init_prcplstmnt_raw(): Total witness norm too big for JL projection\n");
    return 1;
  }

  buf = _malloc(r*sizeof(size_t) + k*sizeof(sparsecnst));
  st->n = (size_t*)buf;
  st->cnst = (sparsecnst*)&st->n[r];

  st->r = r;
  st->k = k;
  st->quadratic = quadratic;
  st->betasq = betasq;
  for(i=0;i<r;i++)
    st->n[i] = n[i];
  for(i=0;i<k;i++)
    st->cnst[i].idx = NULL;

  memset(st->h,0,16);
  return 0;
}

int set_prncplstmnt_lincnst_raw(prncplstmnt *st, size_t i, size_t nz, const size_t idx[nz],
                                const size_t n[nz], size_t deg, int64_t *phi, int64_t *b)
{
  size_t j,k;

  if(i >= st->k) {
    fprintf(stderr,"ERROR in set_prcplstmnt_lincnst_raw(): Constraint %zu does not exist\n",i);
    return 1;
  }

  sparsecnst *cnst = &st->cnst[i];
  if(cnst->idx) {
    fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Constraint has already been set\n");
    return 2;
  }
  for(j=0;j<nz;j++) {
    k = idx[j];
    if(k >= st->r) {
      fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Witness vector %zu does not exist\n",k);
      return 3;
    }
    if(n[j]*deg != st->n[k]) {
      fprintf(stderr,"ERROR in set_prncplstmnt_lincnst_raw(): Mismatch in witness vector length\n");
      return 4;
    }
  }

  init_sparsecnst_raw(cnst,st->r,nz,idx,n,deg,0,b == NULL);
  (void)set_sparsecnst_raw(cnst,st->h,nz,idx,n,deg,phi,b);  // errors can not happen
  return 0;
}

void free_prncplstmnt(prncplstmnt *st) {
  size_t i;

  if(!st->n) return;
  for(i=0;i<st->k;i++)
    free_sparsecnst(&st->cnst[i]);
  free(st->n);  // one buffer for n,cnst
  st->n = NULL;
}

void print_prncplstmnt_pp(const prncplstmnt *st) {
  size_t i;

  printf("Chihuahua statement ");
  for(i=0;i<5;i++)
    printf("%02hhX",st->h[i]);
  printf(":\n");

  printf("  Witness multiplicity: %zu\n",st->r);
  printf("  Witness ranks: ");
  for(i=0;i<st->r;i++) {
    printf("%zu",st->n[i]);
    if(i<st->r-1) printf(", ");
  }
  printf("\n");

  printf("  Number of dot-product constraints: %zu\n",st->k);
  printf("  Norm constraint: %.2f\n",sqrt(st->betasq));
  printf("\n");
}

#define AGGCHUNK 16

/* A destination run written by the deferred constant-term collapse. */
typedef struct {
  polx *p;
  size_t len;
} phirun;

static int phirun_cmp(const void *x, const void *y) {
  polx *const a = ((const phirun*)x)->p;
  polx *const b = ((const phirun*)y)->p;
  return (a > b) - (a < b);
}

/* Degree-0 constraints of the input statement: the ones the LIFTS rounds collapse. */
size_t sparsecnst_zqcount(const sparsecnst *cnst, size_t k) {
  size_t i,m = 0;

  for(i=0;i<k;i++)
    if(!cnst[i].deg) m += 1;
  return m;
}

/* Flat offset into a statement's r*n phi (and into the matching sx) of every witness
 * index, walked once instead of once per block. */
static size_t *zqbase(const statement *ost, const proof *pi, const sparsecnst *icnst, size_t k) {
  size_t i,j,u,v,row,mx;
  size_t *base;

  mx = 0;
  for(i=0;i<k;i++)
    if(!icnst[i].deg)
      for(j=0;j<icnst[i].nz;j++)
        if(icnst[i].idx[j] > mx) mx = icnst[i].idx[j];

  base = _malloc((mx+1)*sizeof(size_t));
  row = 0;
  v = 0;
  for(u=0;u<=mx;u++) {
    base[u] = row*ost->n + v;
    row += pi->nu[u];
    v = (pi->nu[u]) ? 0 : v + pi->n[u];
  }
  return base;
}

/* The constant-term collapse of one lift round, without its phi.
 *
 * The scalars, b and a belong to the round they are drawn in, but the phi accumulation
 * does not: over the LIFTS rounds it amounts to sum_c (sum_i alpha_i s_c^(i)) phi_c, and
 * carrying the k0 scalars instead of scaling phi by them turns LIFTS passes over phi --
 * each of which blew the CRT representatives up to ~q^2 and had to be refreshed -- into
 * the one Montgomery pass of collaps_sparsecnst_apply(). */
void collaps_sparsecnst_scalars(constraint *ocnst, statement *ost, const sparsecnst *icnst, size_t k,
                                int64_t s[])
{
  size_t i,j,m;

  m = sparsecnst_zqcount(icnst,k);

  __attribute__((aligned(16)))
  uint8_t hashbuf[16+m*QBYTES];

  shake128(hashbuf,sizeof(hashbuf),ost->h,16);
  memcpy(ost->h,hashbuf,16);

  m = 0;  // deg 0 constraints get consecutive challenges
  for(i=0;i<k;i++) {
    if(icnst[i].deg) continue;

    s[m] = 0;
    for(j=0;j<QBYTES;j++)
      s[m] |= (int64_t)hashbuf[16+m*QBYTES+j] << 8*j;
    s[m] &= ((int64_t)1 << LOGQ) - 1;

    if(icnst[i].b)
      polx_scale_add(ocnst->b,icnst[i].b,s[m]);
    sparsemat_scale_add(ocnst->a,icnst[i].a,s[m]);
    m += 1;
  }
}

/* u[c] = <phi_c, s>, the part of the round's lifted inner product that the deferred phi
 * would have contributed. Prover only, and the same for every round. */
void collaps_sparsecnst_eval(polx *u, const statement *ost, const proof *pi, const sparsecnst *icnst,
                             size_t k, const polx *sx)
{
  size_t i,j,m;
  size_t *base;
  polx t[1];
  polx *scratch = NULL;
  size_t cap = 0;

  base = zqbase(ost,pi,icnst,k);
  m = 0;
  for(i=0;i<k;i++) {
    if(icnst[i].deg) continue;
    polxvec_setzero(&u[m],1);
    for(j=0;j<icnst[i].nz;j++) {
      polxvec_sprod(t,dense_block(&icnst[i],j,&scratch,&cap),
                    &sx[base[icnst[i].idx[j]]+icnst[i].off[j]],icnst[i].len[j],SPROD_WALK);
      polx_add(&u[m],&u[m],t);
    }
    /* The round scales this by a full width challenge, which multiplies the CRT
     * representative by ~q; only a canonical one leaves room for that. */
    polx_refresh(&u[m]);
    m += 1;
  }
  free(scratch);
  free(base);
}

/* ost->cnst->phi += sum_c t[c] phi_c, the one pass the rounds deferred.
 *
 * The multiplier is a product of a full width challenge and an aggregation challenge, so
 * the products blow the CRT representatives up to ~q^2 and amortize() would overflow the
 * product of the polx primes on them; the runs that were written are refreshed once here,
 * where LIFTS separate collapses used to refresh them once each. */
void collaps_sparsecnst_apply(statement *ost, const proof *pi, const sparsecnst *icnst, size_t k,
                              const polx t[])
{
  size_t i,j,m,nb,nc;
  size_t *base;
  phirun *cov;
  polx tc[1];
  polx *scratch = NULL;
  size_t cap = 0;

  nb = 0;
  for(i=0;i<k;i++)
    if(!icnst[i].deg) nb += icnst[i].nz;
  cov = _malloc(MAX(nb,1)*sizeof(phirun));

  base = zqbase(ost,pi,icnst,k);
  nb = 0;
  m = 0;
  for(i=0;i<k;i++) {
    if(icnst[i].deg) continue;
    polxvec_copy(tc,&t[m],1);
    polx_refresh(tc);
    for(j=0;j<icnst[i].nz;j++) {
      polx *dst = &ost->cnst->phi[base[icnst[i].idx[j]]+icnst[i].off[j]];
      polxvec_polx_mul_add(dst,tc,dense_block(&icnst[i],j,&scratch,&cap),icnst[i].len[j]);
      cov[nb].p = dst;
      cov[nb].len = icnst[i].len[j];
      nb += 1;
    }
    m += 1;
  }

  qsort(cov,nb,sizeof(phirun),phirun_cmp);
  for(i=0;i<nb;) {
    polx *lo = cov[i].p;
    polx *hi = lo + cov[i].len;
    for(j=i+1;j<nb && cov[j].p <= hi;j++)
      if(cov[j].p + cov[j].len > hi) hi = cov[j].p + cov[j].len;
    nc = hi - lo;
    polxvec_refresh(lo,nc);
    i = j;
  }

  free(cov);
  free(scratch);
  free(base);
}

/* One (destination, phi buffer, challenge) contribution to the aggregated phi. */
typedef struct {
  polx *dst;
  const polx *src;
  const polx *alpha;
  size_t len;
} aggent;

static int aggent_cmp(const void *x, const void *y) {
  const polx *const a = ((const aggent*)x)->src;
  const polx *const b = ((const aggent*)y)->src;
  return (a > b) - (a < b);
}

/* The same contribution with the phi block still in the coefficient domain. */
typedef struct {
  size_t dst;            /* first destination element, indexed off ost->cnst->phi */
  const int16_t *c;      /* len*wid coefficients, element major */
  const int16_t *alpha;  /* N coefficients of the challenge */
  size_t len;
  size_t off;
  size_t wid;
} shortent;

static int shortent_cmp(const void *x, const void *y) {
  const size_t a = ((const shortent*)x)->dst;
  const size_t b = ((const shortent*)y)->dst;
  return (a > b) - (a < b);
}

#ifdef __AVX512VNNI__
#define DPWSSD(a,x,y) _mm512_dpwssd_epi32(a,x,y)
#else
#define DPWSSD(a,x,y) _mm512_add_epi32(a,_mm512_madd_epi16(x,y))
#endif

/* acc[i] += sum_t c[i*wid+t] X^(off+t) alpha over Z[X]/(X^N+1), int32 lanes.
 *
 * With C[m] = -alpha[m] for m < N and alpha[m-N] for m >= N, coefficient j of
 * X^k alpha is C[N + j - k], so every term is a shifted read of one 2N table. Terms are
 * taken two at a time and the table is stored as the interleaving D[2p] = C[p],
 * D[2p+1] = C[p-1], which turns a term pair into a single multiply-accumulate against a
 * broadcast of the two coefficients. */
static void shortmac(int32_t *acc, const shortent *e) {
  __attribute__((aligned(64)))
  int16_t D[4*N];
  size_t i,u,t;
  const size_t wid = e->wid;
  const int16_t *v = e->c;

  for(t=0;t<N;t++) {
    D[2*t] = -e->alpha[t];
    D[2*N+2*t] = e->alpha[t];
  }
  D[1] = 0;
  for(t=1;t<2*N;t++)
    D[2*t+1] = D[2*t-2];

  for(i=0;i<e->len;i++) {
    __m512i a0 = _mm512_load_si512((const __m512i*)&acc[0]);
    __m512i a1 = _mm512_load_si512((const __m512i*)&acc[16]);
    __m512i a2 = _mm512_load_si512((const __m512i*)&acc[32]);
    __m512i a3 = _mm512_load_si512((const __m512i*)&acc[48]);
    for(u=0;u<wid;u+=2) {
      uint32_t w = (uint16_t)v[u];
      if(u+1 < wid)
        w |= (uint32_t)(uint16_t)v[u+1] << 16;
      const __m512i vw = _mm512_set1_epi32((int32_t)w);
      const int16_t *d = &D[2*(N - e->off - u)];
      a0 = DPWSSD(a0,vw,_mm512_loadu_si512((const __m512i*)(d+0)));
      a1 = DPWSSD(a1,vw,_mm512_loadu_si512((const __m512i*)(d+32)));
      a2 = DPWSSD(a2,vw,_mm512_loadu_si512((const __m512i*)(d+64)));
      a3 = DPWSSD(a3,vw,_mm512_loadu_si512((const __m512i*)(d+96)));
    }
    _mm512_store_si512((__m512i*)&acc[0],a0);
    _mm512_store_si512((__m512i*)&acc[16],a1);
    _mm512_store_si512((__m512i*)&acc[32],a2);
    _mm512_store_si512((__m512i*)&acc[48],a3);
    acc += N;
    v += wid;
  }
}

/* polxvec_quarternary(), keeping the coefficient polynomials the short path needs. */
static void quarternary_keep(polx *r, poly *t, size_t len, const uint8_t seed[16], uint64_t nonce) {
  while(len >= 32) {
    polyvec_quarternary(t,32,seed,nonce);
    polxvec_frompolyvec(r,t,32);
    nonce += (uint64_t)1 << 32;
    r += 32;
    t += 32;
    len -= 32;
  }

  polyvec_quarternary(t,len,seed,nonce);
  polxvec_frompolyvec(r,t,len);
}

void aggregate_sparsecnst(statement *ost, const proof *pi, const sparsecnst *cnst, size_t k) {
  size_t i,j,u,v,ne,ns,degtot,maxidx;
  __attribute__((aligned(16)))
  uint8_t hashbuf[32];
  polx (*phi)[ost->n];
  polx **dbase;
  polx *alphabuf;
  poly *alphac;
  aggent *ent;
  shortent *sent;
  polx *scratch = NULL;
  size_t cap = 0;

  shake128(hashbuf,32,ost->h,16);
  memcpy(ost->h,hashbuf,16);

  /* Destination base of every witness index, walked once instead of once per block. */
  maxidx = 0;
  degtot = 0;
  ne = 0;
  for(i=0;i<k;i++) {
    if(!cnst[i].deg) continue;
    degtot += cnst[i].deg;
    ne += cnst[i].nz;
    for(j=0;j<cnst[i].nz;j++)
      if(cnst[i].idx[j] > maxidx) maxidx = cnst[i].idx[j];
  }
  dbase = _malloc((maxidx+1)*sizeof(polx*));
  phi = (polx(*)[ost->n])ost->cnst->phi;
  v = 0;
  for(u=0;u<=maxidx;u++) {
    dbase[u] = &(*phi)[v];
    phi += pi->nu[u];
    v = (pi->nu[u]) ? 0 : v + pi->n[u];
  }

  alphabuf = _aligned_alloc(64,MAX(degtot,1)*sizeof(polx));
  alphac = _aligned_alloc(64,MAX(degtot,1)*sizeof(poly));
  ent = _malloc(MAX(ne,1)*sizeof(aggent));
  sent = _malloc(MAX(ne,1)*sizeof(shortent));

  /* The challenges, b and a are consumed in the original order; only the phi
   * accumulation is deferred and regrouped. */
  ne = 0;
  ns = 0;
  degtot = 0;
  for(i=0;i<k;i++) {
    if(cnst[i].deg == 0) continue;
    polx *alpha = &alphabuf[degtot];
    poly *alphacoeff = &alphac[degtot];
    degtot += cnst[i].deg;
    quarternary_keep(alpha,alphacoeff,cnst[i].deg,&hashbuf[16],i);  // nonce is the global index

    if(cnst[i].b)
      polxvec_sprod_add(ost->cnst->b,alpha,cnst[i].b,cnst[i].deg,SPROD_WORST);

    for(j=0;j<cnst[i].nz;j++) {
      polx *dst = dbase[cnst[i].idx[j]] + cnst[i].off[j];
      const shortphi *sp = cnst[i].sphi[j].c ? &cnst[i].sphi[j] : NULL;
      if(cnst[i].deg == 1 && cnst[i].mult[j] == 1 && sp) {
        sent[ns].dst = dst - ost->cnst->phi;
        sent[ns].c = sp->c;
        sent[ns].alpha = alphacoeff->vec->c;
        sent[ns].len = cnst[i].len[j];
        sent[ns].off = sp->off;
        sent[ns].wid = sp->wid;
        ns += 1;
      }
      else if(cnst[i].deg == 1 && cnst[i].mult[j] == 1) {
        ent[ne].dst = dst;
        ent[ne].src = cnst[i].phi[j];
        ent[ne].alpha = alpha;
        ent[ne].len = cnst[i].len[j];
        ne += 1;
      }
      else
        polxvec_collaps_add_extension(dst,alpha,dense_block(&cnst[i],j,&scratch,&cap),cnst[i].len[j],
                                      cnst[i].deg/cnst[i].mult[j],cnst[i].mult[j],SPROD_WORST);
    }

    sparsemat_polx_mul_add(ost->cnst->a,alpha,cnst[i].a);
  }
  free(scratch);

  /* Constraints alias each other's phi buffers but write few distinct destination
   * runs; visiting the contributions destination-major keeps the accumulator hot
   * instead of streaming it once per constraint. */
  qsort(ent,ne,sizeof(aggent),aggent_cmp);
  for(i=0;i<ne;) {
    size_t e,off,maxlen = ent[i].len;
    for(j=i+1;j<ne && ent[j].src == ent[i].src;j++)
      if(ent[j].len > maxlen) maxlen = ent[j].len;
    /* Tiled so the destination slice stays in L1 while every contribution to it
     * is streamed in. */
    for(off=0;off<maxlen;off+=AGGCHUNK)
      for(e=i;e<j;e++)
        if(off < ent[e].len)
          polxvec_polx_mul_add(ent[e].dst+off,ent[e].alpha,ent[e].src+off,
                               MIN(AGGCHUNK,ent[e].len-off));
    i = j;
  }

  /* The coefficient-domain blocks. Their phi never becomes a polx: the challenge is
   * multiplied in over int32 coefficient lanes, and only the accumulated sum of a
   * destination element pays one conversion. Destination major, because the BLOCKS
   * diagonals of one chain write the same run and so keep the accumulator in L1. */
  if(ns) {
    size_t lo,hi,span,mx,wmx;
    int64_t bound;
    int32_t *acc;
    int32_t *cov;

    lo = sent[0].dst;
    hi = 0;
    wmx = 0;
    for(i=0;i<ns;i++) {
      if(sent[i].dst < lo) lo = sent[i].dst;
      if(sent[i].dst + sent[i].len > hi) hi = sent[i].dst + sent[i].len;
      if(sent[i].wid > wmx) wmx = sent[i].wid;
    }
    span = hi - lo;

    /* Coverage of every destination element, for the accumulator bound and for the
     * runs that have to be converted back. */
    cov = _malloc((span+1)*sizeof(int32_t));
    memset(cov,0,(span+1)*sizeof(int32_t));
    for(i=0;i<ns;i++) {
      cov[sent[i].dst-lo] += 1;
      cov[sent[i].dst-lo+sent[i].len] -= 1;
    }
    mx = 0;
    v = 0;
    for(i=0;i<span;i++) {
      v += cov[i];
      cov[i] = v;
      if(v > mx) mx = v;
    }

    /* |acc| <= (blocks covering the element) * wid * max|coefficient| * max|alpha|. */
    bound = (int64_t)mx*wmx*32768*2;
    if(bound < (int64_t)INT32_MAX - 32768) {
      acc = _aligned_alloc(64,span*N*sizeof(int32_t));
      memset(acc,0,span*N*sizeof(int32_t));
      qsort(sent,ns,sizeof(shortent),shortent_cmp);
      for(i=0;i<ns;i++)
        shortmac(&acc[(sent[i].dst-lo)*N],&sent[i]);

      polx t[32];
      for(i=0;i<span;) {
        if(!cov[i]) { i += 1; continue; }
        for(j=i;j<span && cov[j];j++);
        while(i<j) {
          u = MIN(j-i,32);
          polxvec_fromint32vec(t,u,&acc[i*N]);
          polxvec_add(&ost->cnst->phi[lo+i],&ost->cnst->phi[lo+i],t,u);
          i += u;
        }
      }
      free(acc);
    }
    else {
      /* Out of int32 headroom: materialise and take the dense path. */
      for(i=0;i<ns;i++) {
        shortphi sp = {sent[i].c,sent[i].off,sent[i].wid};
        polx alpha[1];
        if(sent[i].len > cap) {
          free(scratch);
          scratch = _aligned_alloc(64,sent[i].len*sizeof(polx));
          cap = sent[i].len;
        }
        shortphi_topolxvec(scratch,&sp,0,sent[i].len);
        polx_frompoly(alpha,(const poly*)sent[i].alpha);
        polxvec_polx_mul_add(&ost->cnst->phi[sent[i].dst],alpha,scratch,sent[i].len);
      }
      free(scratch);
    }
    free(cov);
  }

  free(sent);
  free(ent);
  free(alphac);
  free(alphabuf);
  free(dbase);

  ost->cnst->a->coeffs = realloc(ost->cnst->a->coeffs,ost->cnst->a->len*sizeof(polx));
}

int principle_prove(statement *ost, witness *owt, proof *pi, const prncplstmnt *ist, const witness *iwt, int tail) {
  int ret;
  size_t i;
  constraint cnst[1] = {};
  void *buf = NULL;
  int64_t *zqs = NULL;
  polx *zqbuf = NULL;

  ret = init_proof(pi,iwt,ist->quadratic,tail);
  if(ret)  // commitments not secure (1/2)
    return ret;
  init_statement(ost,pi,ist->h);
  init_witness(owt,ost);
  printf("Predicted witness norm: %.2f\n\n",sqrt(pi->normsq));

  {
    buf = _aligned_alloc(64,ost->r*ost->n*(sizeof(polx)+256*N/8));
    polx (*sx)[ost->n] = (polx(*)[ost->n])buf;
    uint8_t (*jlmat)[ost->n][256*N/8] = (uint8_t(*)[ost->n][256*N/8])sx[ost->r];
commit(ost,owt,pi,sx,iwt);
ret = project(ost,pi,jlmat,iwt);
    if(ret) {
      ret += 10;
      goto err;
    }

    init_constraint(cnst,ost);
    const size_t k0 = sparsecnst_zqcount(ist->cnst,ist->k);
    zqs = _malloc(MAX(k0,1)*sizeof(int64_t));
    zqbuf = _aligned_alloc(64,2*MAX(k0,1)*sizeof(polx));
    zqdefer df = {k0,zqs,&zqbuf[k0],zqbuf};
    polxvec_setzero(zqbuf,2*MAX(k0,1));
    collaps_sparsecnst_eval(&zqbuf[k0],ost,pi,ist->cnst,ist->k,*sx);
    for(i=0;i<LIFTS;i++) {
      collaps_jlproj(cnst,ost,pi,jlmat);
      collaps_sparsecnst_scalars(cnst,ost,ist->cnst,ist->k,zqs);
      lift_aggregate_zqcnst(ost,pi,i,cnst,sx,&df);
    }
    collaps_sparsecnst_apply(ost,pi,ist->cnst,ist->k,zqbuf);
    free(zqbuf);
    free(zqs);
    zqbuf = NULL;
    zqs = NULL;
    free_constraint(cnst);

aggregate_sparsecnst(ost,pi,ist->cnst,ist->k);
    amortize(ost,owt,pi,sx);
    free(buf);
    buf = NULL;
  }

  polx_refresh(ost->cnst->b);
  polxvec_refresh(ost->cnst->phi,ost->n);
  return 0;

err:
  free(zqbuf);
  free(zqs);
  free(buf);
  free_constraint(cnst);
  free_proof(pi);
  free_statement(ost);
  free_witness(owt);
  return ret;
}

int principle_reduce(statement *ost, const proof *pi, const prncplstmnt *ist) {
  size_t i;
  int ret;
  uint8_t (*jlmat)[ost->n][256*N/8];
  constraint cnst[1] = {};
  int64_t *zqs = NULL;
  polx *zqbuf = NULL;

  init_statement(ost,pi,ist->h);
  jlmat = _aligned_alloc(64,ost->r*ost->n*256*N/8);

reduce_commit(ost,pi);
ret = reduce_project(ost,jlmat,pi,pi->r,ist->betasq);
  if(ret) goto err;  // projection too long

  init_constraint(cnst,ost);
  const size_t k0 = sparsecnst_zqcount(ist->cnst,ist->k);
  zqs = _malloc(MAX(k0,1)*sizeof(int64_t));
  zqbuf = _aligned_alloc(64,MAX(k0,1)*sizeof(polx));
  zqdefer df = {k0,zqs,NULL,zqbuf};
  polxvec_setzero(zqbuf,MAX(k0,1));
  for(i=0;i<LIFTS;i++) {
    collaps_jlproj(cnst,ost,pi,jlmat);
    collaps_sparsecnst_scalars(cnst,ost,ist->cnst,ist->k,zqs);
    reduce_lift_aggregate_zqcnst(ost,pi,i,cnst,&df);
  }
  collaps_sparsecnst_apply(ost,pi,ist->cnst,ist->k,zqbuf);
  free(zqbuf);
  free(zqs);
  zqbuf = NULL;
  zqs = NULL;
  free_constraint(cnst);
  free(jlmat);
  jlmat = NULL;

aggregate_sparsecnst(ost,pi,ist->cnst,ist->k);
  ret = reduce_amortize(ost,pi);
  if(ret) {  // commitments not secure (1/2)
    ret += 10;
    goto err;
  }

  polx_refresh(ost->cnst->b);
  polxvec_refresh(ost->cnst->phi,ost->n);
  return 0;

err:
  free(zqbuf);
  free(zqs);
  free_statement(ost);
  free(jlmat);
  free_constraint(cnst);
  return ret;
}

int principle_verify(const prncplstmnt *st, const witness *wt) {
  int ret;
  size_t i;
  uint64_t normsq = 0;
  polx *sx[wt->r];

  if(wt->r != st->r) {
    fprintf(stderr,"ERROR in principle_verify(): Mismatch in witness multiplicity\n");
    return 1;
  }
  for(i=0;i<wt->r;i++)
    if(wt->n[i] != st->n[i]) {
      fprintf(stderr,"ERROR in principle_verify(): Mismatch in length of witness vector %zu\n",i);
      return 2;
    }
  for(i=0;i<wt->r;i++)
    normsq += polyvec_sprodz(wt->s[i],wt->s[i],wt->n[i]);
  if(normsq > st->betasq) {
    fprintf(stderr,"ERROR in principle_verify(): Total witness vector norm too big\n");
    return 3;
  }

  *sx = NULL;
  for(i=0;i<st->k;i++) {
    ret = !sparsecnst_check(&st->cnst[i],sx,wt);
    if(ret) {
      fprintf(stderr,"ERROR in principle_verify(): Sparse dot-product constraint %zu does not hold\n",i);
      ret = 10+i;
      goto end;
    }
  }

end:
  free(*sx);
  return ret;
}

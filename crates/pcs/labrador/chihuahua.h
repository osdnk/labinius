#ifndef CHIHUAHUA_H
#define CHIHUAHUA_H

#include <stdint.h>
#include <stddef.h>
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "sparsemat.h"
#include "labrador.h"

/* A phi block handed over in the coefficient domain instead of as NTT images: element i
 * of the block is sum_{t<wid} c[i*wid + t] X^(off + t), with off + wid <= N. Aggregation
 * multiplies such a block by a quarternary challenge without ever forming its polx. */
typedef struct {
  const int16_t *c;  /* len*wid coefficients, element major; NULL: block is dense */
  size_t off;
  size_t wid;
} shortphi;

typedef struct {
  size_t deg;
  sparsemat a[1];
  size_t nz;
  size_t *idx;
  size_t *off;
  size_t *len;
  size_t *mult;
  polx **phi;
  shortphi *sphi;  /* nz entries; sphi[j].c != NULL means block j is short */
  polx *b;
} sparsecnst;

typedef struct {
  size_t r;
  size_t *n;
  int quadratic;
  size_t k;
  sparsecnst *cnst;
  uint64_t betasq;
  uint8_t h[16];
} prncplstmnt;

polx *init_sparsecnst_half(sparsecnst *cnst, size_t r, size_t nz, size_t buflen, size_t deg,
                           int quadratic, int homogeneous);
void init_sparsecnst_raw(sparsecnst *cnst, size_t r, size_t nz, const size_t idx[nz], const size_t n[nz], size_t deg,
                         int quadratic, int homogeneous);
int set_sparsecnst_raw(sparsecnst *cnst, uint8_t h[16], size_t nz, const size_t idx[nz], const size_t n[nz],
                       size_t deg, int64_t *phi, int64_t *b);
void free_sparsecnst(sparsecnst *cnst);
void shortphi_topolxvec(polx *r, const shortphi *sp, size_t off, size_t len);
void sparsecnst_eval(polx *b, const sparsecnst *cnst, polx *sx[], const witness *wt);
int sparsecnst_check(const sparsecnst *cnst, polx *sx[], const witness *wt);
size_t sparsecnst_check_batch(const sparsecnst *cnst, size_t k, polx *sx[], const witness *wt);
int init_prncplstmnt_raw(prncplstmnt *st, size_t r, const size_t n[r],
                         uint64_t betasq, size_t k, int quadratic);
int set_prncplstmnt_lincnst_raw(prncplstmnt *st, size_t i, size_t nz, const size_t idx[nz],
                                const size_t n[nz], size_t deg, int64_t *phi, int64_t *b);
void free_prncplstmnt(prncplstmnt *st);
void print_prncplstmnt_pp(const prncplstmnt *st);

size_t sparsecnst_zqcount(const sparsecnst *cnst, size_t k);
void collaps_sparsecnst_scalars(constraint *ocnst, statement *ost, const sparsecnst *icnst, size_t k,
                                int64_t s[]);
void collaps_sparsecnst_eval(polx *u, const statement *ost, const proof *pi, const sparsecnst *icnst,
                             size_t k, const polx *sx);
void collaps_sparsecnst_apply(statement *ost, const proof *pi, const sparsecnst *icnst, size_t k,
                              const polx t[]);
void aggregate_sparsecnst(statement *ost, const proof *pi, const sparsecnst *cnst, size_t k);

int principle_prove(statement *ost, witness *owt, proof *pi, const prncplstmnt *ist, const witness *iwt, int tail);
int principle_reduce(statement *ost, const proof *pi, const prncplstmnt *ist);
int principle_verify(const prncplstmnt *st, const witness *wt);

#endif

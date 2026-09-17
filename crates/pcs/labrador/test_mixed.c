#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "fips202.h"
#include "data.h"
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "labrador.h"
#include "chihuahua.h"
#include "dachshund.h"
#include "pack.h"

#define R 3
#define K0 4  /* deg 0 constraints, dense over the whole witness */
#define K1 3  /* deg 1 constraints, vectors 0 and 1 only         */
#define K8 2  /* deg 8 constraints, phi is a commitment key      */
#define KTOT (K0+K1+K8)

/* interleaved: deg 0, deg 1, deg 8, deg 0, deg 1, deg 8, deg 0, deg 1, deg 0 */
static const size_t degs[KTOT] = {0,1,8,0,1,8,0,1,0};

#define PERT_NONE 0
#define PERT_B    1  /* add 1 to the LAST deg 0 constraint's constant term */
#define PERT_S    2  /* zero a coefficient only the deg 0 constraints see */

static void fill_binary(poly *s, size_t n, const uint8_t seed[16], uint64_t nonce) {
  size_t i,j,t;
  uint8_t in[24];
  uint8_t *buf;

  memcpy(in,seed,16);
  for(i=0;i<8;i++)
    in[16+i] = nonce >> 8*i;
  buf = malloc((n*N+7)/8);
  shake128(buf,(n*N+7)/8,in,sizeof(in));
  for(i=0;i<n;i++)
    for(j=0;j<N;j++) {
      t = i*N+j;
      s[i].vec->c[j] = (buf[t/8] >> (t%8)) & 1;
    }
  free(buf);
}

/* b := the constant coefficient of the evaluation, as a polx */
static void set_constcoeff(polx *b, const polx *eval) {
  size_t j;
  polz z[1];

  polz_frompolx(z,eval);
  polz_center(z);
  for(j=1;j<N;j++)
    polz_setcoeff_fromint64(z,0,j);
  polz_topolx(b,z);
}

static void prepare(smplstmnt *st, witness *wt, const size_t n[R], uint8_t seedbyte, int pert) {
  size_t i,j,c,off,total,l;
  __attribute__((aligned(16)))
  uint8_t seed[16];
  uint64_t nonce = 0;
  shake128incctx shakectx;
  sparsecnst *cnst;
  polx *buf;
  uint64_t betasq[R];
  __attribute__((aligned(16)))
  uint8_t hashbuf[8*N*QBYTES];
  polx *sx[R];
  polz t[8];
  polx bb[8];

  memset(seed,seedbyte,sizeof(seed));
  init_witness_raw(wt,R,n);
  fill_binary(wt->s[0],n[0],seed,nonce++);
  betasq[0] = 0;
  for(i=1;i<R;i++) {
    polyvec_ternary(wt->s[i],n[i],seed,nonce++);
    betasq[i] = polyvec_sprodz(wt->s[i],wt->s[i],n[i]);
  }
  for(i=0;i<R;i++)
    wt->normsq[i] = polyvec_sprodz(wt->s[i],wt->s[i],n[i]);

  total = 0;
  for(i=0;i<R;i++)
    total += n[i];

  *sx = NULL;
  shake128_inc_init(&shakectx);
  init_smplstmnt_raw(st,R,n,betasq,KTOT);
  for(c=0;c<KTOT;c++) {
    cnst = &st->cnst[c];
    switch(degs[c]) {
    case 0:  /* dense over all R vectors, uniform phi */
      buf = init_sparsecnst_half(cnst,R,R,total,0,0,0);
      off = 0;
      for(j=0;j<R;j++) {
        cnst->idx[j] = j;
        cnst->off[j] = 0;
        cnst->len[j] = n[j];
        cnst->mult[j] = 1;
        cnst->phi[j] = buf + off;
        off += n[j];
      }
      polxvec_almostuniform(buf,total,seed,nonce++);
      break;
    case 1:  /* vectors 0 and 1 only, uniform phi */
      buf = init_sparsecnst_half(cnst,R,2,n[0]+n[1],1,0,0);
      for(j=0;j<2;j++) {
        cnst->idx[j] = j;
        cnst->off[j] = 0;
        cnst->len[j] = n[j];
        cnst->mult[j] = 1;
        cnst->phi[j] = buf + j*n[0];
      }
      polxvec_almostuniform(buf,n[0]+n[1],seed,nonce++);
      break;
    default: /* deg 8: phi is a commitment key over one vector */
      j = (c/3)&1;
      l = extlen(n[j],8);
      buf = init_sparsecnst_half(cnst,R,1,l,8,0,0);
      cnst->idx[0] = j;
      cnst->off[0] = 0;
      cnst->len[0] = n[j];
      cnst->mult[0] = 1;
      cnst->phi[0] = buf;
      polxvec_almostuniform(buf,l,seed,nonce++);
      break;
    }

    sparsecnst_eval(bb,cnst,sx,wt);
    if(degs[c])
      polxvec_copy(cnst->b,bb,degs[c]);
    else
      set_constcoeff(cnst->b,bb);

    polzvec_frompolxvec(t,cnst->b,MAX(1,degs[c]));
    polzvec_bitpack(hashbuf,t,MAX(1,degs[c]));
    shake128_inc_absorb(&shakectx,hashbuf,MAX(1,degs[c])*N*QBYTES);
    free(*sx);
    *sx = NULL;
  }

  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(st->h,16,&shakectx);

  if(pert == PERT_B) {
    /* the LAST deg 0 constraint: a loop that stops advancing at the first
     * constraint of the other degree never reaches it */
    polx one[1];
    polx_monomial(one,1,0);
    polx_add(st->cnst[KTOT-1].b,st->cnst[KTOT-1].b,one);
  }
  else if(pert == PERT_S) {  /* vector 2 is touched by the deg 0 constraints only */
    wt->s[2][n[2]/2].vec->c[N/3] = 0;
    wt->normsq[2] = polyvec_sprodz(wt->s[2],wt->s[2],n[2]);
  }
}

static int run(const char *name, const size_t n[R], uint8_t seedbyte, int pert) {
  int ret,fail = 0;
  const int bad = (pert != PERT_NONE);
  smplstmnt st = {};
  witness wt = {};
  commitment com = {};
  composite p = {};

  printf("\n########## %s ##########\n",name);
  printf("ranks: %zu %zu %zu, constraints: %d deg0 + %d deg1 + %d deg8\n\n",
         n[0],n[1],n[2],K0,K1,K8);

  prepare(&st,&wt,n,seedbyte,pert);
  print_smplstmnt_pp(&st);

  ret = simple_verify(&st,&wt);
  if(!!ret != bad) {
    fprintf(stderr,"FAIL[%s]: simple_verify returned %d, expected %s\n",
            name,ret,bad ? "failure" : "success");
    fail = 1;
    goto end;
  }
  printf("simple_verify: %s (as expected)\n",ret ? "rejected" : "accepted");

  ret = composite_prove_simple(&p,&com,&st,&wt);
  if(ret) {
    if(bad) {
      printf("composite_prove_simple rejected (%d) (as expected)\n",ret);
      goto end;
    }
    fprintf(stderr,"FAIL[%s]: composite_prove_simple failed: %d\n",name,ret);
    fail = 1;
    goto end;
  }

  ret = composite_verify_simple(&p,&com,&st);
  if(!!ret != bad) {
    fprintf(stderr,"FAIL[%s]: composite_verify_simple returned %d, expected %s\n",
            name,ret,bad ? "failure" : "success");
    if(!ret)
      fprintf(stderr,"FAIL[%s]: *** constraint not covered by the proof ***\n",name);
    fail = 1;
    goto end;
  }
  printf("composite_verify_simple: %s (as expected)\n",ret ? "rejected" : "accepted");

end:
  free_smplstmnt(&st);
  free_commitment(&com);
  free_composite(&p);
  free_witness(&wt);
  return fail;
}

int main(int argc, char *argv[]) {
  int fail = 0;
  uint8_t so = (argc > 1) ? (uint8_t)atoi(argv[1]) : 0;
  const size_t n[R] = {512,512,128};
  const size_t nb[R] = {256,384,64};

  fail |= run("mixed/honest",n,71+so,PERT_NONE);
  fail |= run("mixed/honest small",nb,72+so,PERT_NONE);
  fail |= run("mixed/perturbed last deg0 b",n,73+so,PERT_B);
  fail |= run("mixed/perturbed witness seen only by deg0",n,74+so,PERT_S);

  free_comkey();
  printf("\n=== test_mixed %s ===\n",fail ? "FAILED" : "PASSED");
  return fail;
}

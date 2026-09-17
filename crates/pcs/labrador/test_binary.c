#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "randombytes.h"
#include "fips202.h"
#include "data.h"
#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "labrador.h"
#include "chihuahua.h"
#include "dachshund.h"
#include "pack.h"

/* content of the vector declared binary; BIN_SIGNFLIP is non-binary at the
 * norm of a binary vector, so only the binariness identity can reject it */
#define BIN_RAND     0
#define BIN_ZERO     1
#define BIN_ONE      2
#define BIN_TWO      3
#define BIN_MINUS1   4
#define BIN_TERNARY  5
#define BIN_SIGNFLIP 6

static const char *binname[] = {"random 0/1","all zeros","all ones",
                                "one coeff = 2","one coeff = -1","ternary",
                                "sign-flipped 1s (same norm)"};

static void fill_binary(poly *s, size_t n, int mode, const uint8_t seed[16], uint64_t nonce) {
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

  switch(mode) {
  case BIN_ZERO:
    polyvec_setzero(s,n);
    break;
  case BIN_ONE:
    for(i=0;i<n;i++)
      for(j=0;j<N;j++)
        s[i].vec->c[j] = 1;
    break;
  case BIN_TWO:
    s[n/2].vec->c[N/3] = 2;
    break;
  case BIN_MINUS1:
    s[n/2].vec->c[N/3] = -1;
    break;
  case BIN_TERNARY:
    polyvec_ternary(s,n,seed,nonce);
    break;
  case BIN_SIGNFLIP:
    for(i=0,t=0;i<n && t<16;i++)
      for(j=0;j<N && t<16;j++)
        if(s[i].vec->c[j] == 1) {
          s[i].vec->c[j] = -1;
          t += 1;
        }
    break;
  default:
    break;
  }
}

/* vectors 0..nbin-1 are declared binary (betasq = 0), the rest are ternary
 * with betasq[i] = exact norm; k linear (deg = 1) constraints over all */
static void prepare(smplstmnt *st, witness *wt, size_t r, const size_t n[], size_t k,
                    int binmode, uint8_t seedbyte, size_t nbin)
{
  size_t i,j,c,off,total;
  __attribute__((aligned(16)))
  uint8_t seed[16];
  uint64_t nonce = 0;
  shake128incctx shakectx;
  sparsecnst *cnst;
  polx *buf;
  uint64_t betasq[r];
  __attribute__((aligned(16)))
  uint8_t hashbuf[N*QBYTES];
  polx *sx[r];
  polz t[1];

  memset(seed,seedbyte,sizeof(seed));

  init_witness_raw(wt,r,n);
  for(i=0;i<r;i++) {
    if(i < nbin) {
      fill_binary(wt->s[i],n[i],i ? BIN_RAND : binmode,seed,nonce++);
      betasq[i] = 0;
    }
    else {
      polyvec_ternary(wt->s[i],n[i],seed,nonce++);
      betasq[i] = 0;
    }
    wt->normsq[i] = polyvec_sprodz(wt->s[i],wt->s[i],n[i]);
    if(i >= nbin)
      betasq[i] = wt->normsq[i];
  }

  total = 0;
  for(i=0;i<r;i++)
    total += n[i];

  *sx = NULL;
  shake128_inc_init(&shakectx);
  init_smplstmnt_raw(st,r,n,betasq,k);
  for(c=0;c<k;c++) {
    cnst = &st->cnst[c];
    buf = init_sparsecnst_half(cnst,r,r,total,1,0,0);
    off = 0;
    for(j=0;j<r;j++) {
      cnst->idx[j] = j;
      cnst->off[j] = 0;
      cnst->len[j] = n[j];
      cnst->mult[j] = 1;
      cnst->phi[j] = buf + off;
      off += n[j];
    }
    for(j=0;j<total;j++) {
      polzvec_almostuniform(t,1,seed,nonce++);
      polzvec_bitpack(hashbuf,t,1);
      shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
      polzvec_topolxvec(&buf[j],t,1);
    }
    sparsecnst_eval(cnst->b,cnst,sx,wt);
    polzvec_frompolxvec(t,cnst->b,1);
    polzvec_bitpack(hashbuf,t,1);
    shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
    free(*sx);
    *sx = NULL;
  }

  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(st->h,16,&shakectx);
}

/* bypass runs the prover even though simple_verify rejects, to see whether
 * the proof system itself catches the non-binary witness */
static int run(const char *name, size_t r, const size_t n[], size_t k, int binmode,
               uint8_t seedbyte, int expect_verify, int bypass, int expect_proof, size_t nbin)
{
  int ret,fail = 0;
  smplstmnt st = {};
  witness wt = {};
  commitment com = {};
  composite p = {};

  printf("\n########## %s ##########\n",name);
  printf("binary vector 0: %s, ranks:",binname[binmode]);
  for(size_t i=0;i<r;i++)
    printf(" %zu",n[i]);
  printf(", %zu constraints, %zu binary\n\n",k,nbin);

  prepare(&st,&wt,r,n,k,binmode,seedbyte,nbin);
  print_smplstmnt_pp(&st);

  ret = simple_verify(&st,&wt);
  if(!!ret != !!expect_verify) {
    fprintf(stderr,"FAIL[%s]: simple_verify returned %d, expected %s\n",
            name,ret,expect_verify ? "failure" : "success");
    fail = 1;
    goto end;
  }
  printf("simple_verify: %s (as expected)\n",ret ? "rejected" : "accepted");
  if(ret && !bypass)
    goto end;

  ret = composite_prove_simple(&p,&com,&st,&wt);
  if(ret) {
    if(expect_proof) {
      printf("composite_prove_simple rejected (%d) (as expected)\n",ret);
      goto end;
    }
    fprintf(stderr,"FAIL[%s]: composite_prove_simple failed: %d\n",name,ret);
    fail = 1;
    goto end;
  }

  ret = composite_verify_simple(&p,&com,&st);
  if(!!ret != !!expect_proof) {
    fprintf(stderr,"FAIL[%s]: composite_verify_simple returned %d, expected %s\n",
            name,ret,expect_proof ? "failure" : "success");
    if(!ret)
      fprintf(stderr,"FAIL[%s]: *** UNSOUND: non-binary witness accepted ***\n",name);
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
  const size_t n3[3] = {512,512,128};
  const size_t nodd[3] = {509,383,129};
  const size_t nsmall[3] = {64,64,17};

  fail |= run("binary/random",3,n3,3,BIN_RAND,42+so,0,0,0,1);
  fail |= run("binary/zeros",3,n3,3,BIN_ZERO,43+so,0,0,0,1);
  fail |= run("binary/ones",3,n3,3,BIN_ONE,44+so,0,0,0,1);
  fail |= run("binary/odd ranks",3,nodd,4,BIN_RAND,45+so,0,0,0,1);
  fail |= run("binary/small ranks",3,nsmall,2,BIN_RAND,46+so,0,0,0,1);

  /* declared binary but is not: simple_verify must reject */
  fail |= run("nonbinary/coeff 2",3,n3,3,BIN_TWO,47+so,1,0,1,1);
  fail |= run("nonbinary/coeff -1",3,n3,3,BIN_MINUS1,48+so,1,0,1,1);

  /* soundness: prove anyway and see whether the verifier accepts */
  fail |= run("soundness/coeff 2 bypass",3,n3,3,BIN_TWO,49+so,1,1,1,1);
  fail |= run("soundness/ternary bypass",3,n3,3,BIN_TERNARY,50+so,1,1,1,1);
  fail |= run("soundness/sign-flip bypass",3,n3,3,BIN_SIGNFLIP,51+so,1,1,1,1);

  /* every vector binary: no norm slack vector */
  fail |= run("allbinary/random",3,n3,3,BIN_RAND,52+so,0,0,0,3);
  fail |= run("allbinary/odd ranks",3,nodd,2,BIN_RAND,53+so,0,0,0,3);
  fail |= run("allbinary/soundness bypass",3,n3,3,BIN_SIGNFLIP,54+so,1,1,1,3);

  free_comkey();
  if(fail)
    printf("\n=== test_binary FAILED ===\n");
  else
    printf("\n=== test_binary PASSED ===\n");
  return fail;
}

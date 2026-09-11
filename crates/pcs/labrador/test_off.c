#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "randombytes.h"
#include "fips202.h"
#include "dachshund.h"
#include "pack.h"

static void prepare(smplstmnt *st, witness *wt, int split) {
  size_t i,j;
  __attribute__((aligned(16)))
  uint8_t seed[16];
  uint64_t nonce = 0;
  shake128incctx shakectx;
  sparsecnst *cnst;
  polx *buf;

  size_t r = 2;
  size_t n[r];
  for(i=0;i<r;i++)
    n[i] = 1024;
  if(getenv("PROBE_N0")) n[0] = (size_t)atoi(getenv("PROBE_N0"));
  if(getenv("PROBE_N1")) n[1] = (size_t)atoi(getenv("PROBE_N1"));
  size_t k = r;
  size_t deg = 1;
  uint64_t betasq[r];
  for(i=0;i<r;i++)
    betasq[i] = 1.15*10/16*n[i]*N;

  __attribute__((aligned(16)))
  uint8_t hashbuf[N*QBYTES];
  polx *sx[r];
  polz t[1];

  randombytes(seed,sizeof(seed));
  init_witness_raw(wt,r,n);
  for(i=0;i<r;i++) {
    polyvec_ternary(wt->s[i],wt->n[i],seed,nonce++);
    wt->normsq[i] = polyvec_sprodz(wt->s[i],wt->s[i],wt->n[i]);
  }

  *sx = NULL;
  shake128_inc_init(&shakectx);
  init_smplstmnt_raw(st,r,n,betasq,k);
  for(i=0;i<k;i++) {
    cnst = &st->cnst[i];
    size_t nz = split ? 2 : 1;
    buf = init_sparsecnst_half(cnst,r,nz,n[i],deg,0,0);

    if(split) {
      cnst->idx[0] = i;
      cnst->off[0] = 0;
      cnst->len[0] = n[i]/2;
      cnst->mult[0] = 1;
      cnst->phi[0] = buf;
      cnst->idx[1] = i;
      cnst->off[1] = n[i]/2;
      cnst->len[1] = n[i]/2;
      cnst->mult[1] = 1;
      cnst->phi[1] = buf + n[i]/2;
    }
    else {
      cnst->idx[0] = i;
      cnst->off[0] = 0;
      cnst->len[0] = n[i];
      cnst->mult[0] = 1;
      cnst->phi[0] = buf;
    }

    for(j=0;j<n[i];j++) {
      polzvec_almostuniform(t,1,seed,nonce++);
      polzvec_bitpack(hashbuf,t,1);
      shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
      polzvec_topolxvec(&buf[j],t,1);
    }

    sparsecnst_eval(cnst->b,cnst,sx,wt);
    polzvec_frompolxvec(t,cnst->b,deg);
    polzvec_bitpack(hashbuf,t,deg);
    shake128_inc_absorb(&shakectx,hashbuf,deg*N*QBYTES);
  }

  free(*sx);
  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(st->h,16,&shakectx);
}

static int run(int split) {
  int ret;
  smplstmnt st = {};
  witness wt = {};
  commitment com = {};
  composite p = {};

  printf("=== split=%d ===\n",split);
  prepare(&st,&wt,split);
  ret = simple_verify(&st,&wt);
  if(ret) {
    fprintf(stderr,"simple_verify failed: %d\n",ret);
    goto end;
  }
  printf("simple_verify OK\n");

  ret = composite_prove_simple(&p,&com,&st,&wt);
  if(ret) {
    fprintf(stderr,"composite_prove_simple failed: %d\n",ret);
    goto end;
  }
  ret = composite_verify_simple(&p,&com,&st);
  if(ret)
    fprintf(stderr,"composite_verify_simple failed: %d\n",ret);
  else
    printf("composite_verify OK\n");

end:
  free_smplstmnt(&st);
  free_commitment(&com);
  free_composite(&p);
  free_witness(&wt);
  return ret;
}

static void prepare_shape(smplstmnt *st, witness *wt) {
  size_t i,j,c;
  __attribute__((aligned(16)))
  uint8_t seed[16];
  uint64_t nonce = 0;
  shake128incctx shakectx;
  sparsecnst *cnst;
  polx *buf;

  size_t r = 3;
  size_t n2 = getenv("PROBE_N2") ? (size_t)atoi(getenv("PROBE_N2")) : 256;
  size_t k1 = getenv("PROBE_K1") ? (size_t)atoi(getenv("PROBE_K1")) : 64;
  size_t k2 = getenv("PROBE_K2") ? (size_t)atoi(getenv("PROBE_K2")) : 6;
  size_t n[] = {1024,1024,n2};
  size_t k = k1+k2;
  uint64_t betasq[r];
  for(i=0;i<r;i++)
    betasq[i] = 1.15*10/16*n[i]*N;

  __attribute__((aligned(16)))
  uint8_t hashbuf[N*QBYTES];
  polx *sx[r];
  polz t[1];

  memset(seed,42,sizeof(seed));
  if(getenv("PROBE_SEED")) seed[0] = (uint8_t)atoi(getenv("PROBE_SEED"));
  init_witness_raw(wt,r,n);
  for(i=0;i<r;i++) {
    polyvec_ternary(wt->s[i],wt->n[i],seed,nonce++);
    if(getenv("PROBE_BIG")) {
      int sc = atoi(getenv("PROBE_BIG"));
      for(j=0;j<wt->n[i]*N;j++)
        wt->s[i][j/N].vec->c[j%N] *= sc;
    }
    wt->normsq[i] = polyvec_sprodz(wt->s[i],wt->s[i],wt->n[i]);
    if(getenv("PROBE_EXACT")) betasq[i] = wt->normsq[i];
  }

  *sx = NULL;
  shake128_inc_init(&shakectx);
  init_smplstmnt_raw(st,r,n,betasq,k);
  for(c=0;c<k;c++) {
    cnst = &st->cnst[c];
    if(c < k1) {
      int homog = (getenv("PROBE_HOMOG") != NULL);
      int nz1 = (getenv("PROBE_NZ1") != NULL);
      buf = init_sparsecnst_half(cnst,r,homog ? 3 : (nz1 ? 1 : 2),homog ? 68 : (nz1 ? 32 : 36),1,0,homog);
      cnst->idx[0] = c&1;
      cnst->off[0] = (32*(c/2))%1024;
      cnst->len[0] = 32;
      cnst->mult[0] = 1;
      cnst->phi[0] = buf;
      if(homog) {
        cnst->idx[1] = c&1;
        cnst->off[1] = 32*(c/2);
        cnst->len[1] = 32;
        cnst->mult[1] = 1;
        cnst->phi[1] = buf+32;
        cnst->idx[2] = 2;
        cnst->off[2] = (4*(c/2))%n2;
        cnst->len[2] = 4;
        cnst->mult[2] = 1;
        cnst->phi[2] = buf+64;
      }
      else if(!nz1) {
        cnst->idx[1] = 2;
        cnst->off[1] = (4*(c/2))%n2;
        cnst->len[1] = 4;
        cnst->mult[1] = 1;
        cnst->phi[1] = buf+32;
      }
      for(j=0;j<32;j++) {
        polzvec_almostuniform(t,1,seed,nonce++);
        polzvec_bitpack(hashbuf,t,1);
        shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
        polzvec_topolxvec(&buf[j],t,1);
      }
      if(homog) {
        polxvec_setzero(buf+32,36);
        polxvec_sub(buf+32,buf+32,buf,32);
        polx bb[1];
        sparsecnst_eval(bb,cnst,sx,wt);
        if(!polxvec_iszero(bb,1))
          fprintf(stderr,"probe constraint %zu not homogeneous\n",c);
      }
      else {
        if(!nz1) for(j=32;j<36;j++) {
          polzvec_almostuniform(t,1,seed,nonce++);
          polzvec_bitpack(hashbuf,t,1);
          shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
          polzvec_topolxvec(&buf[j],t,1);
        }
        sparsecnst_eval(cnst->b,cnst,sx,wt);
        polzvec_frompolxvec(t,cnst->b,1);
        polzvec_bitpack(hashbuf,t,1);
        shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
      }
    }
    else {
      buf = init_sparsecnst_half(cnst,r,3,2048+n2,1,0,0);
      size_t offs[] = {0,0,0};
      for(j=0;j<3;j++) {
        cnst->idx[j] = j;
        cnst->off[j] = offs[j];
        cnst->len[j] = n[j];
        cnst->mult[j] = 1;
      }
      cnst->phi[0] = buf;
      cnst->phi[1] = buf+1024;
      cnst->phi[2] = buf+2048;
      for(j=0;j<2048+n2;j++) {
        polzvec_almostuniform(t,1,seed,nonce++);
        polzvec_bitpack(hashbuf,t,1);
        shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
        polzvec_topolxvec(&buf[j],t,1);
      }
      sparsecnst_eval(cnst->b,cnst,sx,wt);
      polzvec_frompolxvec(t,cnst->b,1);
      polzvec_bitpack(hashbuf,t,1);
      shake128_inc_absorb(&shakectx,hashbuf,N*QBYTES);
    }
    free(*sx);
    *sx = NULL;
  }

  shake128_inc_finalize(&shakectx);
  shake128_inc_squeeze(st->h,16,&shakectx);
}

static int run_shape(void) {
  int ret;
  smplstmnt st = {};
  witness wt = {};
  commitment com = {};
  composite p = {};

  printf("=== shape ===\n");
  prepare_shape(&st,&wt);
  ret = simple_verify(&st,&wt);
  if(ret) {
    fprintf(stderr,"simple_verify failed: %d\n",ret);
    goto end;
  }
  printf("simple_verify OK\n");

  ret = composite_prove_simple(&p,&com,&st,&wt);
  if(ret) {
    fprintf(stderr,"composite_prove_simple failed: %d\n",ret);
    goto end;
  }
  ret = composite_verify_simple(&p,&com,&st);
  if(ret)
    fprintf(stderr,"composite_verify_simple failed: %d\n",ret);
  else
    printf("composite_verify OK\n");

end:
  free_smplstmnt(&st);
  free_commitment(&com);
  free_composite(&p);
  free_witness(&wt);
  return ret;
}

int main(void) {
  int ret;
  ret = run(0);
  if(ret) goto end;
  ret = run(1);
  if(ret) goto end;
  ret = run_shape();
end:
  free_comkey();
  return ret;
}

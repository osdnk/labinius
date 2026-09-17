#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include "data.h"
#include "polx.h"
#include "poly.h"
#include "polz.h"

/* Every polx pass over a long vector is K per-prime sweeps of the same elements; walking
 * the vector in chunks keeps a chunk's K images in L1 across them instead of streaming
 * the whole vector once per prime. 16 polx is 16 KB of source and as much destination. */
#define POLXCHUNK 16

static int64_t cmodq(int64_t a) {
  int64_t t;
  const int64_t mask = ((int64_t)1 << LOGQ) - 1;
  const int64_t q = ((int64_t)1 << LOGQ) - QOFF;

  t = a >> LOGQ;
  a &= mask;
  a += t*QOFF;
  t = q/2 - a;
  a -= (t >> 63)&q;
  return a;
}

void polx_print(const polx *a) {
  polz t[1];
  polz_frompolx(t,a);
  polz_center(t);
  polz_print(t);
}

void polx_getcoeff(zz *r, const polx *a, int k) {
  polz z;

  polz_frompolx(&z,a);
  polz_getcoeff(r,&z,k);
}

void polxvec_setzero(polx *r, size_t len) {
  polyvec_setzero(&r->vec[0],len*K);
}

void polxvec_copy(polx *r, const polx *a, size_t len) {
  polyvec_copy(&r->vec[0],&a->vec[0],len*K);
}

void polxvec_fromint64vec(polx *r, size_t len, size_t deg, const int64_t a[len*deg*N]) {
  size_t i;
  polz t[deg];

  for(i=0;i<len;i++) {
    polzvec_fromint64vec(t,1,deg,&a[i*deg*N]);
    polzvec_topolxvec(&r[i*deg],t,deg);
  }
}

/* The polx of len coefficient vectors of int32, |a| < 2^31 - 2^15: the same map as
 * polxvec_frompolyvec(), which is a * s_k * 2^-16 mod p_k followed by the NTT. The value
 * is split as a = lo + 2^16 hi with both halves int16, so that each half can go through
 * poly_scale()'s int16 Montgomery multiplier; the two halves are combined before the
 * single NTT pass rather than after, which is what makes this cheaper than converting
 * both halves as polx and adding. */
void polxvec_fromint32vec(polx *r, size_t len, const int32_t *a) {
  size_t i,j,k,m;
  int16_t c;
  __attribute__((aligned(64)))
  poly lo[32], hi[32];
  const __m512i half = _mm512_set1_epi32(1 << 15);

  while(len) {
    m = MIN(len,32);
    for(i=0;i<m;i++) {
      for(j=0;j<N/16;j++) {
        __m512i x = _mm512_loadu_si512((const __m512i*)&a[i*N + 16*j]);
        __m512i h = _mm512_srai_epi32(_mm512_add_epi32(x,half),16);
        _mm256_store_si256((__m256i*)&lo[i].vec->c[16*j],_mm512_cvtepi32_epi16(x));
        _mm256_store_si256((__m256i*)&hi[i].vec->c[16*j],_mm512_cvtepi32_epi16(h));
      }
    }
    for(k=0;k<K;k++) {
      /* c = s * 2^16 mod p, so that poly_scale(hi,c) = hi * s: the 2^16 weight of the
       * high half and poly_scale()'s own 2^-16 cancel. */
      c = ((int64_t)primes[k].s * primes[k].mont) % primes[k].p;
      for(i=0;i<m;i++) {
        poly_scale(&r[i].vec[k],&lo[i],primes[k].s,&primes[k]);
        poly_scale_add(&r[i].vec[k],&hi[i],c,&primes[k]);
      }
      polyvec_reduce(&r->vec[k],m,K,&primes[k]);
      polyvec_ntt(&r->vec[k],&r->vec[k],m,K,&primes[k]);
    }
    r += m;
    a += m*N;
    len -= m;
  }
}

int polx_iszero(const polx *a) {
  polz t;

  polz_frompolx(&t,a);
  polz_center(&t);
  return polz_iszero(&t);
}

int polxvec_iszero(const polx *a, size_t len) {
  polz t[len];

  polzvec_frompolxvec(t,a,len);
  polzvec_center(t,len);
  return polzvec_iszero(t,len);
}

int polx_iszero_constcoeff(const polx *a) {
  polz t;

  polz_frompolx(&t,a);
  polz_center(&t);
  return polz_iszero_constcoeff(&t);
}

void polx_monomial(polx *r, int64_t v, int k) {
  size_t i;

  v = cmodq(v);
  for(i=0;i<K;i++)
    poly_monomial_ntt(&r->vec[i],v % primes[i].p,k,&primes[i]);
}

void polxvec_ternary(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce) {
  poly t[32];

  while(len >= 32) {
    polyvec_ternary(t,32,seed,nonce);
    polxvec_frompolyvec(r,t,32);
    nonce += (uint64_t)1 << 32;
    r += 32;
    len -= 32;
  }

  polyvec_ternary(t,len,seed,nonce);
  polxvec_frompolyvec(r,t,len);
}

void polxvec_quarternary(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce) {
  poly t[32];

  while(len >= 32) {
    polyvec_quarternary(t,32,seed,nonce);
    polxvec_frompolyvec(r,t,32);
    nonce += (uint64_t)1 << 32;
    r += 32;
    len -= 32;
  }

  polyvec_quarternary(t,len,seed,nonce);
  polxvec_frompolyvec(r,t,len);
}

void polxvec_challenge(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce) {
  poly t[10];

  while(len >= 10) {
    polyvec_challenge(t,10,seed,nonce);
    polxvec_frompolyvec(r,t,10);
    nonce += (uint64_t)1 << 32;
    r += 10;
    len -= 10;
  }

  polyvec_challenge(t,len,seed,nonce);
  polxvec_frompolyvec(r,t,len);
}

void polxvec_almostuniform(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce) {
  polz t[32];

  while(len >= 32) {
    polzvec_almostuniform(t,32,seed,nonce);
    polzvec_topolxvec(r,t,32);
    nonce += (uint64_t)1 << 32;
    r += 32;
    len -= 32;
  }

  polzvec_almostuniform(t,len,seed,nonce);
  polzvec_topolxvec(r,t,len);
}

void polx_frompoly(polx *r, const poly *a) {
  size_t i;

  for(i=0;i<K;i++)
    poly_scale(&r->vec[i],a,primes[i].s,&primes[i]);

  polx_ntt(r,r);
}

void polxvec_frompolyvec(polx *r, const poly *a, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_scale_widening(&r[i].vec[j],&a[i],l,K,primes[j].s,&primes[j]);
    polxvec_ntt(&r[i],&r[i],l);
  }
}

void polx_refresh(polx *r) {
  polz a;

  polz_frompolx(&a,r);
  polz_center(&a);
  polz_topolx(r,&a);
}

/* Batched, so that the inverse NTT, the CRT lift and the forward NTT all run over 16
 * elements at a time with the per-prime stride, instead of once per element. Bit
 * identical to polx_refresh() in a loop: polzvec_frompolxvec() and polz_frompolx() are
 * the same computation. */
void polxvec_refresh(polx *r, size_t len) {
  size_t m;
  __attribute__((aligned(64)))
  polz t[16];

  while(len) {
    m = MIN(len,16);
    polzvec_frompolxvec(t,r,m);
    polzvec_center(t,m);
    polzvec_topolxvec(r,t,m);
    r += m;
    len -= m;
  }
}

void polx_reduce(polx *r) {
  size_t i;

  for(i=0;i<K;i++)
    poly_reduce(&r->vec[i],&primes[i]);
}

void polxvec_reduce(polx *r, size_t len) {
  size_t i;

  for(i=0;i<K;i++)
    polyvec_reduce(&r->vec[i],len,K,&primes[i]);
}

void polx_neg(polx *r, const polx *a) {
  polyvec_neg(&r->vec[0],&a->vec[0],K);
}

void polxvec_neg(polx *r, const polx *a, size_t len) {
  polyvec_neg(&r->vec[0],&a->vec[0],K*len);
}

void polx_add(polx *r, const polx *a, const polx *b) {
  size_t i;

  for(i=0;i<K;i++)
    poly_add_reduce(&r->vec[i],&a->vec[i],&b->vec[i],&primes[i]);
}

void polxvec_add(polx *r, const polx *a, const polx *b, size_t len) {
  size_t i,j;

  for(i=0;i<len;i++)
    for(j=0;j<K;j++)
      poly_add_reduce(&r[i].vec[j],&a[i].vec[j],&b[i].vec[j],&primes[j]);
}

void polx_sub(polx *r,const polx *a,const polx *b) {
  size_t i;

  for(i=0;i<K;i++)
    poly_sub_reduce(&r->vec[i],&a->vec[i],&b->vec[i],&primes[i]);
}

void polxvec_sub(polx *r, const polx *a, const polx *b, size_t len) {
  size_t i,j;

  for(i=0;i<len;i++)
    for(j=0;j<K;j++)
      poly_sub_reduce(&r[i].vec[j],&a[i].vec[j],&b[i].vec[j],&primes[j]);
}

void polx_ntt(polx *r, const polx *a) {
  size_t i;

  for(i=0;i<K;i++)
    poly_ntt(&r->vec[i],&a->vec[i],&primes[i]);
}

void polxvec_ntt(polx *r, const polx *a, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_ntt(&r[i].vec[j],&a[i].vec[j],l,K,&primes[j]);
  }
}

void polx_invntt(polx *r, const polx *a) {
  size_t i;

  for(i=0;i<K;i++)
    poly_invntt(&r->vec[i],&a->vec[i],&primes[i]);
}

void polxvec_invntt(polx *r, const polx *a, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_invntt(&r[i].vec[j],&a[i].vec[j],l,K,&primes[j]);
  }
}

void polx_mul(polx *r, const polx *a, const polx *b) {
  size_t i;

  for(i=0;i<K;i++)
    poly_pointwise(&r->vec[i],&a->vec[i],&b->vec[i],&primes[i]);
}

void polx_poly_mul(polx *r, const polx *a, const poly *b) {
  polx t;

  polx_frompoly(&t,b);
  polx_mul(r,a,&t);
}

void polxvec_mul(polx *r, const polx *a, const polx *b, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_pointwise(&r[i].vec[j],&a[i].vec[j],&b[i].vec[j],l,K,&primes[j]);
  }
}

void polxvec_polx_mul(polx *r, const polx *a, const polx *b, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_poly_pointwise(&r[i].vec[j],&a->vec[j],&b[i].vec[j],l,K,&primes[j]);
  }
}

void polx_mul_add(polx *r, const polx *a, const polx *b) {
  size_t i;

  for(i=0;i<K;i++)
    poly_pointwise_add_reduce(&r->vec[i],&a->vec[i],&b->vec[i],&primes[i]);
}

void polxvec_mul_add(polx *r, const polx *a, const polx *b, size_t len) {
  size_t i,j,l;

  for(i=0;i<len;i+=POLXCHUNK) {
    l = MIN(POLXCHUNK,len-i);
    for(j=0;j<K;j++)
      polyvec_pointwise_add(&r[i].vec[j],&a[i].vec[j],&b[i].vec[j],l,K,&primes[j]);
    polxvec_reduce(&r[i],l);
  }
}

void polxvec_polx_mul_add(polx *r, const polx *a, const polx *b, size_t len) {
  /* Blocked so that the source and destination slices of one chunk stay in L1
   * across the K per-prime passes. */
  const size_t chunk = POLXCHUNK;
  size_t i,j,l;

  for(i=0;i<len;i+=chunk) {
    l = MIN(chunk,len-i);
    for(j=0;j<K;j++)
      polyvec_poly_pointwise_add_reduce(&r[i].vec[j],&a->vec[j],&b[i].vec[j],l,K,&primes[j]);
  }
}

/* The fixed point of |z| <= |mont|*(|z|/2^16 + p/2) + chunk*(3p/2)^2 against the int16 the
 * Montgomery step has to land in, |z|/2^16 + p/2 <= 2^15-1; rounded down to a pair count. */
static size_t sprod_chunk(size_t j, int bound) {
  static size_t chunk[2][K];
  const double p = primes[j].p, m = primes[j].mont < 0 ? -primes[j].mont : primes[j].mont;

  if(!chunk[bound][j])
    chunk[bound][j] = (bound == SPROD_WALK)
                    ? (size_t)floor(pow(ldexp(1/3.2,31)/(p*p),2)) & ~(size_t)1
                    : (size_t)floor((ldexp(32767.0 - p/2,16)*(1 - m/65536) - m*p/2)/(2.25*p*p)) & ~(size_t)1;

  return chunk[bound][j];
}

void polxvec_sprod(polx *r, const polx *a, const polx *b, size_t len, int bound) {
  /* Blocked so that both operand slices of one chunk stay in L1 across the K
   * per-prime passes: two 16-polx slices are 32 KB, two 64-polx ones were 128 KB. */
  const size_t chunk = POLXCHUNK;
  size_t i,j,l;

  if(len == 0) {
    polxvec_setzero(r,1);
    return;
  }

  l = MIN(chunk,len);
  for(j=0;j<K;j++)
    polyvec_sprod_pointwise(&r->vec[j],&a->vec[j],&b->vec[j],l,K,sprod_chunk(j,bound),&primes[j]);
  for(i=chunk;i<len;i+=chunk) {
    l = MIN(chunk,len-i);
    for(j=0;j<K;j++)
      polyvec_sprod_pointwise_add(&r->vec[j],&a[i].vec[j],&b[i].vec[j],l,K,sprod_chunk(j,bound),&primes[j]);
  }
}

void polxvec_sprod_add(polx *r, const polx *a, const polx *b, size_t len, int bound) {
  size_t i;

  for(i=0;i<K;i++)
    polyvec_sprod_pointwise_add(&r->vec[i],&a->vec[i],&b->vec[i],len,K,sprod_chunk(i,bound),&primes[i]);
}

void polx_scale(polx *r, const polx *a, int64_t s) {
  size_t i;

  s = cmodq(s);  // s<<16 can overflow int64_t at |s|~q/2 for LOGQ=50; shift per-prime instead
  for(i=0;i<K;i++)
    poly_scale(&r->vec[i],&a->vec[i],((s % primes[i].p) << 16) % primes[i].p,&primes[i]);
}

void polx_scale_frompoly(polx *r, const poly *a, int64_t s) {
  size_t i;

  s = cmodq(s);
  for(i=0;i<K;i++)
    poly_scale(&r->vec[i],a,s*primes[i].s % primes[i].p,&primes[i]);

  polx_ntt(r,r);
}

void polxvec_scale(polx *r, const polx *a, size_t len, int64_t s) {
  size_t i;

  s = cmodq(s);
  for(i=0;i<K;i++)
    polyvec_scale(&r->vec[i],&a->vec[i],len,K,((s % primes[i].p) << 16) % primes[i].p,&primes[i]);
}

void polxvec_scale_frompolyvec(polx *r, const poly *a, size_t len, int64_t s) {
  size_t i;

  s = cmodq(s);
  for(i=0;i<K;i++)
    polyvec_scale_widening(&r->vec[i],a,len,K,s*primes[i].s % primes[i].p,&primes[i]);

  polxvec_ntt(r,r,len);
}

void polx_scale_add(polx *r, const polx *a, int64_t s) {
  size_t i;

  s = cmodq(s);
  for(i=0;i<K;i++)
    poly_scale_add(&r->vec[i],&a->vec[i],((s % primes[i].p) << 16) % primes[i].p,&primes[i]);

  polx_reduce(r);
}

void polxvec_scale_add(polx *r, const polx *a, size_t len, int64_t s) {
  const size_t chunk = POLXCHUNK;
  size_t i,j,l;

  s = cmodq(s);
  for(i=0;i<len;i+=chunk) {
    l = MIN(chunk,len-i);
    for(j=0;j<K;j++)
      polyvec_scale_add_reduce(&r[i].vec[j],&a[i].vec[j],l,K,((s % primes[j].p) << 16) % primes[j].p,&primes[j]);
  }
}

size_t polxvec_mul_extension(polx *c, const polx *a, const polx *b, size_t len, size_t deg, size_t mult, int bound) {
  size_t i,j,k = 0;

  if(deg == 1 && mult == 1) {
    polxvec_sprod(c,a,b,len,bound);
    return len;
  }

  for(i=0;i<mult;i++)
    for(j=0;j<K;j++)
      k = polyvec_pointwise_extension(&c[i].vec[j],&a->vec[j],&b[i].vec[j],len,K*mult,deg,sprod_chunk(j,bound),
                                      &primes[j]);

  return k;
}

size_t polxvec_collaps_add_extension(polx *c, const polx *a, const polx *b, size_t len, size_t deg, size_t mult,
                                     int bound)
{
  size_t i,j,k = 0;

  if(deg == 1 && mult == 1) {
    polxvec_polx_mul_add(c,a,b,len);
    return len;
  }

  for(i=0;i<mult;i++)
    for(j=0;j<K;j++)
      k = polyvec_collaps_add_extension(&c[i].vec[j],&a[i].vec[j],&b->vec[j],len,K*mult,deg,sprod_chunk(j,bound),
                                        &primes[j]);

  return k;
}

void polxvec_decompose(poly *r, const polx *a, size_t len, size_t t, size_t d) {
  size_t i;
  const size_t stride = len;
  polz b[16];

  while(len >= 16) {
    polzvec_frompolxvec(b,a,16);
    polzvec_center(b,16);
    for(i=0;i<16;i++)
      polz_decompose(&r[i],&b[i],stride,t,d);
    r += 16;
    a += 16;
    len -= 16;
  }

  polzvec_frompolxvec(b,a,len);
  polzvec_center(b,len);
  for(i=0;i<len;i++)
    polz_decompose(&r[i],&b[i],stride,t,d);
}

void polxvec_reconstruct(polx *r, const poly *a, size_t len, size_t t, size_t d) {
  size_t i;
  const size_t stride = len;
  polz b[16];

  while(len >= 16) {
    for(i=0;i<16;i++)
      polz_reconstruct(&b[i],&a[i],stride,t,d);
    polzvec_topolxvec(r,b,16);
    r += 16;
    a += 16;
    len -= 16;
  }

  for(i=0;i<len;i++)
    polz_reconstruct(&b[i],&a[i],stride,t,d);
  polzvec_topolxvec(r,b,len);
}

void polx_sigmam1(polx *r, const polx *a) {
  polyvec_sigmam1_ntt(&r->vec[0],&a->vec[0],K);
}

void polxvec_sigmam1(polx *r, const polx *a, size_t len) {
  polyvec_sigmam1_ntt(&r->vec[0],&a->vec[0],K*len);
}

/* FIXME: NTT rep! */
void polx_sigma5(polx *r, const polx *a) {
  polyvec_sigma5(&r->vec[0],&a->vec[0],K);
}

void polxvec_sigma5(polx *r, const polx *a, size_t len) {
  polyvec_sigma5(&r->vec[0],&a->vec[0],K*len);
}

void polx_sigma5inv(polx *r, const polx *a) {
  polyvec_sigma5inv(&r->vec[0],&a->vec[0],K);
}

void polxvec_sigma5inv(polx *r, const polx *a, size_t len) {
  polyvec_sigma5inv(&r->vec[0],&a->vec[0],K*len);
}

void polx_flip(polx *r, const polx *a) {
  size_t i;

  for(i=0;i<K;i++)
    poly_flip_ntt(&r->vec[i],&a->vec[i],&primes[i]);
}

void polxvec_flip(polx *r, const polx *a, size_t len) {
  size_t i;

  for(i=0;i<K;i++)
    polyvec_flip_ntt(&r->vec[i],&a->vec[i],len,K,&primes[i]);
}


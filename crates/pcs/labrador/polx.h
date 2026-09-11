#ifndef POLX_H
#define POLX_H

#include "data.h"
#include "poly.h"

typedef struct {
  poly vec[K];
} polx;

/* How many products may pile up in an int32 sprod lane before the next Montgomery
 * reduction. SPROD_WALK is Lazer's random-walk bound, which the prover may use on its own
 * witness; SPROD_WORST is the bound that survives any operands the invariant admits and is
 * what every product reading a proof has to use. */
enum { SPROD_WALK, SPROD_WORST };

void polx_print(const polx *a);
void polx_getcoeff(zz *r, const polx *a, int k);
void polxvec_setzero(polx *r, size_t len);
void polxvec_fromint64vec(polx *r, size_t len, size_t deg, const int64_t a[len*deg*N]);
void polxvec_fromint32vec(polx *r, size_t len, const int32_t a[len*N]);
int polx_iszero(const polx *a);
int polxvec_iszero(const polx *a, size_t len);
int polx_iszero_constcoeff(const polx *a);
void polx_monomial(polx *r, int64_t v, int k);
void polxvec_copy(polx *r, const polx *a, size_t len);
void polxvec_almostuniform(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce);
void polxvec_ternary(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce);
void polxvec_quarternary(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce);
void polxvec_challenge(polx *r, size_t len, const uint8_t seed[16], uint64_t nonce);
void polx_frompoly(polx *r, const poly *a);
void polxvec_frompolyvec(polx *r, const poly *a, size_t len);

void polx_refresh(polx *r);
void polxvec_refresh(polx *r, size_t len);
void polx_reduce(polx *r);
void polxvec_reduce(polx *r, size_t len);

void polx_neg(polx *r, const polx *a);
void polxvec_neg(polx *r, const polx *a, size_t len);
void polx_add(polx *r, const polx *a, const polx *b);
void polxvec_add(polx *r, const polx *a, const polx *b, size_t len);
void polx_sub(polx *r, const polx *a, const polx *b);
void polxvec_sub(polx *r, const polx *a, const polx *b, size_t len);

void polx_ntt(polx *r, const polx *a);
void polxvec_ntt(polx *r, const polx *a, size_t len);
void polx_invntt(polx *r, const polx *a);
void polxvec_invntt(polx *r, const polx *a, size_t len);

void polx_mul(polx *r, const polx *a, const polx *b);
void polx_poly_mul(polx *r, const polx *a, const poly *b);
void polxvec_mul(polx *r, const polx *a, const polx *b, size_t len);
void polxvec_polx_mul(polx *r, const polx *a, const polx *b, size_t len);
void polx_mul_add(polx *r, const polx *a, const polx *b);
void polxvec_mul_add(polx *r, const polx *a, const polx *b, size_t len);
void polxvec_polx_mul_add(polx *r, const polx *a, const polx *b, size_t len);
void polxvec_sprod(polx *r, const polx *a, const polx *b, size_t len, int bound);
void polxvec_sprod_add(polx *r, const polx *a, const polx *b, size_t len, int bound);
size_t polxvec_mul_extension(polx *c, const polx *a, const polx *b, size_t len, size_t deg, size_t mult, int bound);
size_t polxvec_collaps_add_extension(polx *c, const polx *a, const polx *b, size_t len, size_t deg, size_t mult,
                                     int bound);

void polx_scale(polx *r, const polx *a, int64_t s);
void polx_scale_frompoly(polx *r, const poly *a, int64_t s);
void polxvec_scale(polx *r, const polx *a, size_t len, int64_t s);
void polxvec_scale_frompolyvec(polx *r, const poly *a, size_t len, int64_t s);
void polx_scale_add(polx *r, const polx *a, int64_t s);
void polxvec_scale_add(polx *r, const polx *a, size_t len, int64_t s);

void polxvec_decompose(poly *r, const polx *a, size_t len, size_t t, size_t d);
void polxvec_reconstruct(polx *r, const poly *a, size_t len, size_t t, size_t d);

void polx_sigmam1(polx *r, const polx *a);
void polxvec_sigmam1(polx *r, const polx *a, size_t len);
void polx_sigma5(polx *r, const polx *a);
void polxvec_sigma5(polx *r, const polx *a, size_t len);
void polx_sigma5inv(polx *r, const polx *a);
void polxvec_sigma5inv(polx *r, const polx *a, size_t len);

void polx_flip(polx *r, const polx *a);
void polxvec_flip(polx *r, const polx *a, size_t len);

#endif

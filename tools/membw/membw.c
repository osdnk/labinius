// Single-core DRAM write/read bandwidth floor for the materialised NTT output (2^18 x 1296 B).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(void){
  size_t bytes=(size_t)(1<<18)*1296; void*buf=aligned_alloc(64,bytes); memset(buf,1,bytes);
  __m512i v=_mm512_set1_epi16(7);
  for(int rep=0;rep<3;rep++){
    double t0=now(); for(char*p=buf;p<(char*)buf+bytes;p+=64)_mm512_store_si512((void*)p,v); double t1=now();
    printf("regular zmm store : %6.2f GB/s (%5.1f ms)\n",bytes/(t1-t0)/1e9,(t1-t0)*1e3);
    t0=now(); for(char*p=buf;p<(char*)buf+bytes;p+=64)_mm512_stream_si512((void*)p,v); _mm_sfence(); t1=now();
    printf("non-temporal store: %6.2f GB/s (%5.1f ms)\n",bytes/(t1-t0)/1e9,(t1-t0)*1e3);
    t0=now(); memset(buf,rep,bytes); t1=now();
    printf("memset            : %6.2f GB/s (%5.1f ms)\n",bytes/(t1-t0)/1e9,(t1-t0)*1e3);
    __m512i acc=_mm512_setzero_si512(); t0=now(); for(char*p=buf;p<(char*)buf+bytes;p+=64)acc=_mm512_add_epi64(acc,_mm512_load_si512((void*)p)); t1=now();
    printf("read              : %6.2f GB/s (%5.1f ms) [%lld]\n",bytes/(t1-t0)/1e9,(t1-t0)*1e3,(long long)_mm512_reduce_add_epi64(acc));
  }
  return 0;}

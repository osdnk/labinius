// Results (i7-11850H, taskset -c 6, gcc -O2 -march=native, best of 9, +-0.005 cyc):
//
//   variant           cyc/bf   p0/bf   p5/bf   note
//   1 nat             10.713  10.707   8.294  natural order            <- what the kernel does
//   2 grp             10.694  10.688   8.313  mults grouped per butterfly
//   3 pipe            10.609  10.270   8.731  block pipeline P1(k)|P2(k-1)
//   4 mix11           10.453  10.446   8.555  1 mul : 1 add, pipelined <- fastest
//   5 mix22           10.473  10.399   8.602  2 mul : 2 add, pipelined
//   6 memnat          10.757  10.690   8.312  memory operands for a1,a2
//   7 memmix          10.851  10.427   8.574  memory operands + 1:1
//   9 deep            10.553   9.757   9.245  3-stage pipe, [9mul][10add] blocks
//  12 deep sprd       10.533   9.738   9.263  same + ld/st spread      <- lowest p0
//  13 deep -st         9.770   9.764   9.237  stores removed
//  14 nat -st         10.714  10.708   8.293  stores removed
//   8 ymm             14.596  13.809  10.309  (p1 13.884) 256-bit, much worse
//   L mul only         9.007   9.001   0.001  9 mul + 3 ld + 3 st
//   L add only         5.007   5.001   5.001  10 add + 3 ld + 3 st
//
// Synthetic allocator probes (independent uops, no dependences):
//   9 mul | 10 add             9.507  p0 9.501  p5 9.501   <- allocator is OPTIMAL
//   ... + 3 loads              9.507  p0 9.501  p5 9.501   loads are free
//   ... + 3 zmm stores        10.267  p0 9.501  p5 9.501   +0.77 cyc, ports idle
//   ... + 3 loads + 3 stores  10.695  p0 9.501  p5 9.501   +1.19 cyc (placement-invariant)
//   9 mul / 10 DEPENDENT add  10.083  p0 10.077 p5 8.925   <- the leak appears
//
// Model that fits every row within 0.1 cycle:  cycles = max(p0_uops, ~10.5).
// The 10.5 floor is 9.5 (ALU) + ~0.8 for the three 512-bit stores.  Ordering only
// buys the 0.26 cyc/butterfly (2.4%) by which the natural order's p0 leak exceeds
// that floor; see the report at the bottom of this file.
//
// Port-allocation microbenchmark for the radix-3 NTT butterfly on Tiger Lake.
//
// One butterfly, matching src/simd/vertical_gen.rs::r3:
//     t1 = mont(a1,w1) = mulhi(a1,w1) - mulhi(mullo(a1,w1'),q)
//     t2 = mont(a2,w2)
//     d  = t1 - t2 ;  u = mont(d,om)
//     y0 = a0 + (t1+t2) ; y1 = (a0-t2) + u ; y2 = (a0-t1) - u
// = 3 loads + 9 multiply uops (p0 only, 512-bit) + 10 add/sub uops (p0 or p5) + 3 stores.
//
// Ideal cycles/butterfly = max(9 + x, 10 - x) minimised at x = 0.5 -> 9.5.
// Measured p0/p5 split tells how many of the 10 adds the allocator leaked onto p0.
//
// Data: 27 zmm (1728 B, L1-resident), 9 independent butterflies (i, i+9, i+18) per pass.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

/* ------------------------------------------------------------------ perf */
#define NEV 5
static int evfd[NEV];
static const char *evname[NEV] = { "cyc", "p0", "p1", "p5", "uops" };
static const uint32_t evtype[NEV] = { PERF_TYPE_HARDWARE, PERF_TYPE_RAW, PERF_TYPE_RAW,
                                      PERF_TYPE_RAW, PERF_TYPE_RAW };
/* Ice Lake / Tiger Lake: UOPS_DISPATCHED.PORT_x = 0xa1, umask 0x01/0x02/0x20.
   UOPS_ISSUED.ANY = 0x0e umask 0x01 (fused domain, at rename). */
static uint64_t evcfg[NEV] = { PERF_COUNT_HW_CPU_CYCLES, 0x01a1, 0x02a1, 0x20a1, 0x010e };
/* EV=1: memory ports + front-end.  PORT_2_3 0x04a1, PORT_4_9 0x10a1, PORT_7_8 0x80a1,
   IDQ_UOPS_NOT_DELIVERED.CORE 0x019c (issue slots the front end failed to fill). */
static const uint64_t evcfg2[NEV] = { PERF_COUNT_HW_CPU_CYCLES, 0x04a1, 0x10a1, 0x80a1, 0x019c };
static const char *evname2[NEV] = { "cyc", "p23", "p49", "p78", "fe_gap" };
/* EV3: front-end delivery.  IDQ.DSB_UOPS 0x0879, IDQ.MITE_UOPS 0x0479,
   LSD.UOPS 0x01a8, UOPS_EXECUTED.THREAD 0x01b1 (unfused). */
static const uint64_t evcfg3[NEV] = { PERF_COUNT_HW_CPU_CYCLES, 0x0879, 0x0479, 0x01a8, 0x01b1 };
static const char *evname3[NEV] = { "cyc", "dsb", "mite", "lsd", "exec" };

static void perf_init(void) {
    if (getenv("EV2")) { memcpy(evcfg, evcfg2, sizeof(evcfg)); memcpy(evname, evname2, sizeof(evname)); }
    if (getenv("EV3")) { memcpy(evcfg, evcfg3, sizeof(evcfg)); memcpy(evname, evname3, sizeof(evname)); }
    for (int i = 0; i < NEV; i++) {
        struct perf_event_attr pe; memset(&pe, 0, sizeof(pe));
        pe.type = evtype[i]; pe.size = sizeof(pe); pe.config = evcfg[i];
        pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        evfd[i] = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        if (evfd[i] < 0) { fprintf(stderr, "event %s: ", evname[i]); perror("perf_event_open"); exit(1); }
    }
}
static void perf_start(void) {
    for (int i = 0; i < NEV; i++) { ioctl(evfd[i], PERF_EVENT_IOC_RESET, 0); ioctl(evfd[i], PERF_EVENT_IOC_ENABLE, 0); }
}
static void perf_stop(double out[NEV]) {
    for (int i = 0; i < NEV; i++) {
        uint64_t v[3];
        ioctl(evfd[i], PERF_EVENT_IOC_DISABLE, 0);
        if (read(evfd[i], v, sizeof(v)) != sizeof(v)) { perror("read"); exit(2); }
        out[i] = v[2] ? (double)v[0] * (double)v[1] / (double)v[2] : 0.0;
    }
}

/* ------------------------------------------------------- constant registers */
#define Q   "%%zmm25"
#define W1P "%%zmm26"
#define W1  "%%zmm27"
#define W2P "%%zmm28"
#define W2  "%%zmm29"
#define OMP "%%zmm30"
#define OM  "%%zmm31"
#define LOADCONST \
    "vmovdqa64   0(%1)," Q   "\n" "vmovdqa64  64(%1)," W1P "\n" "vmovdqa64 128(%1)," W1  "\n" \
    "vmovdqa64 192(%1)," W2P "\n" "vmovdqa64 256(%1)," W2  "\n" "vmovdqa64 320(%1)," OMP "\n" \
    "vmovdqa64 384(%1)," OM  "\n"

#define YQ   "%%ymm25"
#define YW1P "%%ymm26"
#define YW1  "%%ymm27"
#define YW2P "%%ymm28"
#define YW2  "%%ymm29"
#define YOMP "%%ymm30"
#define YOM  "%%ymm31"

/* --------------------------------------------------------- butterfly steps
   Arg pack: A0,A1,A2,M1,M2,U,O0,O1,O2   (6 zmm registers + 3 vector indices)
   A1 becomes t1, A2 becomes t2, M2 becomes d then is dead, M1 becomes s.        */
#define E_L0(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O0 "*64(%0),%%zmm" #A0 "\n"
#define E_L1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O1 "*64(%0),%%zmm" #A1 "\n"
#define E_L2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O2 "*64(%0),%%zmm" #A2 "\n"
#define E_X1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " W1P ",%%zmm" #A1 ",%%zmm" #M1 "\n"
#define E_X2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " W1  ",%%zmm" #A1 ",%%zmm" #A1 "\n"
#define E_X3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " Q   ",%%zmm" #M1 ",%%zmm" #M1 "\n"
#define E_S1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #M1 ",%%zmm" #A1 ",%%zmm" #A1 "\n"
#define E_X4(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " W2P ",%%zmm" #A2 ",%%zmm" #M2 "\n"
#define E_X5(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " W2  ",%%zmm" #A2 ",%%zmm" #A2 "\n"
#define E_X6(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " Q   ",%%zmm" #M2 ",%%zmm" #M2 "\n"
#define E_S2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #M2 ",%%zmm" #A2 ",%%zmm" #A2 "\n"
#define E_D(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #A2 ",%%zmm" #A1 ",%%zmm" #M2 "\n"
#define E_X7(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " OMP ",%%zmm" #M2 ",%%zmm" #M1 "\n"
#define E_X8(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " OM  ",%%zmm" #M2 ",%%zmm" #U  "\n"
#define E_X9(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " Q   ",%%zmm" #M1 ",%%zmm" #M1 "\n"
#define E_S3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #M1 ",%%zmm" #U  ",%%zmm" #U  "\n"
#define E_A1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%zmm" #A2 ",%%zmm" #A1 ",%%zmm" #M1 "\n"
#define E_A2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #A2 ",%%zmm" #A0 ",%%zmm" #A2 "\n"
#define E_A3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%zmm" #U  ",%%zmm" #A2 ",%%zmm" #A2 "\n"
#define E_A4(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #A1 ",%%zmm" #A0 ",%%zmm" #A1 "\n"
#define E_A5(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%zmm" #U  ",%%zmm" #A1 ",%%zmm" #A1 "\n"
#define E_A6(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%zmm" #M1 ",%%zmm" #A0 ",%%zmm" #A0 "\n"
#define E_T0(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%zmm" #A0 "," #O0 "*64(%0)\n"
#define E_T1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%zmm" #A2 "," #O1 "*64(%0)\n"
#define E_T2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%zmm" #A1 "," #O2 "*64(%0)\n"

/* memory-operand forms: a1 / a2 read straight out of L1 (multiply is commutative,
   so the memory operand can sit in the src2 slot); no separate load uop. */
#define M_X1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " #O1 "*64(%0)," W1P ",%%zmm" #M1 "\n"
#define M_X2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " #O1 "*64(%0)," W1  ",%%zmm" #A1 "\n"
#define M_X4(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " #O2 "*64(%0)," W2P ",%%zmm" #M2 "\n"
#define M_X5(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " #O2 "*64(%0)," W2  ",%%zmm" #A2 "\n"

/* multiply-only / add-only skeletons (same loads, same stores) */
#define O_X7(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " OMP ",%%zmm" #A1 ",%%zmm" #M2 "\n"
#define O_X8(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " OM  ",%%zmm" #A1 ",%%zmm" #U  "\n"
#define O_X9(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " Q   ",%%zmm" #M2 ",%%zmm" #M2 "\n"
#define O_T0(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%zmm" #U  "," #O0 "*64(%0)\n"
#define N_S1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw " Q ",%%zmm" #A1 ",%%zmm" #A1 "\n"
#define N_S2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw " Q ",%%zmm" #A2 ",%%zmm" #A2 "\n"
#define N_S3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw " Q ",%%zmm" #M2 ",%%zmm" #U  "\n"

/* 256-bit forms */
#define Y_L0(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O0 "*32(%0),%%ymm" #A0 "\n"
#define Y_L1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O1 "*32(%0),%%ymm" #A1 "\n"
#define Y_L2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 " #O2 "*32(%0),%%ymm" #A2 "\n"
#define Y_X1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " YW1P ",%%ymm" #A1 ",%%ymm" #M1 "\n"
#define Y_X2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YW1  ",%%ymm" #A1 ",%%ymm" #A1 "\n"
#define Y_X3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YQ   ",%%ymm" #M1 ",%%ymm" #M1 "\n"
#define Y_S1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #M1 ",%%ymm" #A1 ",%%ymm" #A1 "\n"
#define Y_X4(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " YW2P ",%%ymm" #A2 ",%%ymm" #M2 "\n"
#define Y_X5(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YW2  ",%%ymm" #A2 ",%%ymm" #A2 "\n"
#define Y_X6(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YQ   ",%%ymm" #M2 ",%%ymm" #M2 "\n"
#define Y_S2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #M2 ",%%ymm" #A2 ",%%ymm" #A2 "\n"
#define Y_D(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #A2 ",%%ymm" #A1 ",%%ymm" #M2 "\n"
#define Y_X7(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmullw " YOMP ",%%ymm" #M2 ",%%ymm" #M1 "\n"
#define Y_X8(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YOM  ",%%ymm" #M2 ",%%ymm" #U  "\n"
#define Y_X9(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpmulhw " YQ   ",%%ymm" #M1 ",%%ymm" #M1 "\n"
#define Y_S3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #M1 ",%%ymm" #U  ",%%ymm" #U  "\n"
#define Y_A1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%ymm" #A2 ",%%ymm" #A1 ",%%ymm" #M1 "\n"
#define Y_A2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #A2 ",%%ymm" #A0 ",%%ymm" #A2 "\n"
#define Y_A3(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%ymm" #U  ",%%ymm" #A2 ",%%ymm" #A2 "\n"
#define Y_A4(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #A1 ",%%ymm" #A0 ",%%ymm" #A1 "\n"
#define Y_A5(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpsubw %%ymm" #U  ",%%ymm" #A1 ",%%ymm" #A1 "\n"
#define Y_A6(A0,A1,A2,M1,M2,U,O0,O1,O2) "vpaddw %%ymm" #M1 ",%%ymm" #A0 ",%%ymm" #A0 "\n"
#define Y_T0(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%ymm" #A0 "," #O0 "*32(%0)\n"
#define Y_T1(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%ymm" #A2 "," #O1 "*32(%0)\n"
#define Y_T2(A0,A1,A2,M1,M2,U,O0,O1,O2) "vmovdqa64 %%ymm" #A1 "," #O2 "*32(%0)\n"

/* ------------------------------------------------------------ orderings
   X is a butterfly argument pack (expands to the 9 numbers above).           */

/* (1) natural: source order, mont-by-mont */
#define NAT(X) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_S1(X) E_X4(X) E_X5(X) E_X6(X) E_S2(X) E_D(X) \
    E_X7(X) E_X8(X) E_X9(X) E_S3(X) \
    E_A1(X) E_A2(X) E_A3(X) E_A4(X) E_A5(X) E_A6(X) E_T0(X) E_T1(X) E_T2(X)

/* (2) grouped: 6 mul | 3 sub | 3 mul | 1 sub | 6 add, one butterfly at a time */
#define GRP(X) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) E_S1(X) E_S2(X) E_D(X) \
    E_X7(X) E_X8(X) E_X9(X) E_S3(X) \
    E_A1(X) E_A2(X) E_A3(X) E_A4(X) E_A5(X) E_A6(X) E_T0(X) E_T1(X) E_T2(X)

/* (3) block software pipeline: P1 = loads+9 mul+4 sub, P2 = 6 add+3 stores */
#define P1(...) E_L0(__VA_ARGS__) E_L1(__VA_ARGS__) E_L2(__VA_ARGS__) \
    E_X1(__VA_ARGS__) E_X2(__VA_ARGS__) E_X3(__VA_ARGS__) E_X4(__VA_ARGS__) E_X5(__VA_ARGS__) E_X6(__VA_ARGS__) E_S1(__VA_ARGS__) E_S2(__VA_ARGS__) E_D(__VA_ARGS__) \
    E_X7(__VA_ARGS__) E_X8(__VA_ARGS__) E_X9(__VA_ARGS__) E_S3(__VA_ARGS__)
#define P2(...) E_A1(__VA_ARGS__) E_A2(__VA_ARGS__) E_A3(__VA_ARGS__) E_A4(__VA_ARGS__) E_A5(__VA_ARGS__) E_A6(__VA_ARGS__) E_T0(__VA_ARGS__) E_T1(__VA_ARGS__) E_T2(__VA_ARGS__)
#define PIPE(X,Y) P1(X) P2(Y)

/* (4) 1:1 interleave -- one p0-only multiply, one p0/p5 add, alternating */
#define MIX11(X,Y) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_A1(Y) E_X2(X) E_A2(Y) E_X3(X) E_S1(X) E_X4(X) E_A3(Y) \
    E_X5(X) E_T1(Y) E_X6(X) E_S2(X) E_D(X) E_X7(X) E_A4(Y) E_X8(X) E_A5(Y) \
    E_X9(X) E_T2(Y) E_S3(X) E_A6(Y) E_T0(Y)

/* (5) 2:2 interleave */
#define MIX22(X,Y) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_A1(Y) E_A2(Y) E_X3(X) E_X4(X) E_S1(X) E_A3(Y) \
    E_X5(X) E_X6(X) E_S2(X) E_T1(Y) E_D(X) E_A4(Y) E_X7(X) E_X8(X) \
    E_A5(Y) E_T2(Y) E_X9(X) E_S3(X) E_A6(Y) E_T0(Y)

/* (6) memory operands for a1/a2, natural order */
#define MEMNAT(X) E_L0(X) \
    M_X1(X) M_X2(X) E_X3(X) E_S1(X) M_X4(X) M_X5(X) E_X6(X) E_S2(X) E_D(X) \
    E_X7(X) E_X8(X) E_X9(X) E_S3(X) \
    E_A1(X) E_A2(X) E_A3(X) E_A4(X) E_A5(X) E_A6(X) E_T0(X) E_T1(X) E_T2(X)

/* (7) memory operands + 1:1 interleave */
#define MEMMIX(X,Y) E_L0(X) \
    M_X1(X) E_A1(Y) M_X2(X) E_A2(Y) E_X3(X) E_S1(X) M_X4(X) E_A3(Y) \
    M_X5(X) E_T1(Y) E_X6(X) E_S2(X) E_D(X) E_X7(X) E_A4(Y) E_X8(X) E_A5(Y) \
    E_X9(X) E_T2(Y) E_S3(X) E_A6(Y) E_T0(Y)

/* (8) 256-bit, natural order */
#define YNAT(X) Y_L0(X) Y_L1(X) Y_L2(X) \
    Y_X1(X) Y_X2(X) Y_X3(X) Y_S1(X) Y_X4(X) Y_X5(X) Y_X6(X) Y_S2(X) Y_D(X) \
    Y_X7(X) Y_X8(X) Y_X9(X) Y_S3(X) \
    Y_A1(X) Y_A2(X) Y_A3(X) Y_A4(X) Y_A5(X) Y_A6(X) Y_T0(X) Y_T1(X) Y_T2(X)

/* (9) lower bound: 9 multiplies, no adds */
#define MULONLY(X) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) O_X7(X) O_X8(X) O_X9(X) \
    O_T0(X) E_T1(X) E_T2(X)

/* (10) lower bound: 10 adds, no multiplies */
#define ADDONLY(X) E_L0(X) E_L1(X) E_L2(X) \
    N_S1(X) N_S2(X) E_D(X) N_S3(X) \
    E_A1(X) E_A2(X) E_A3(X) E_A4(X) E_A5(X) E_A6(X) E_T0(X) E_T1(X) E_T2(X)

/* ------------------------------------------------- register sets / packs */
#define RS0 0,1,2,3,4,5
#define RS1 6,7,8,9,10,11
#define RS2 12,13,14,15,16,17
#define B0 RS0,0,9,18
#define B1 RS1,1,10,19
#define B2 RS2,2,11,20
#define B3 RS0,3,12,21
#define B4 RS1,4,13,22
#define B5 RS2,5,14,23
#define B6 RS0,6,15,24
#define B7 RS1,7,16,25
#define B8 RS2,8,17,26
#define ALL9(F) F(B0) F(B1) F(B2) F(B3) F(B4) F(B5) F(B6) F(B7) F(B8)
#define PIPE9(F) P1(B0) F(B1,B0) F(B2,B1) F(B3,B2) F(B4,B3) F(B5,B4) F(B6,B5) \
                 F(B7,B6) F(B8,B7) P2(B8)
/* memory-operand pipeline: prologue must use the memory-operand P1 too */
#define MP1(...) E_L0(__VA_ARGS__) M_X1(__VA_ARGS__) M_X2(__VA_ARGS__) E_X3(__VA_ARGS__) M_X4(__VA_ARGS__) M_X5(__VA_ARGS__) E_X6(__VA_ARGS__) \
    E_S1(__VA_ARGS__) E_S2(__VA_ARGS__) E_D(__VA_ARGS__) E_X7(__VA_ARGS__) E_X8(__VA_ARGS__) E_X9(__VA_ARGS__) E_S3(__VA_ARGS__)
#define MPIPE9(F) MP1(B0) F(B1,B0) F(B2,B1) F(B3,B2) F(B4,B3) F(B5,B4) F(B6,B5) \
                  F(B7,B6) F(B8,B7) P2(B8)


/* (11) 3-stage software pipeline: step k issues the loads and first six multiplies of
   butterfly k, the last three multiplies of k-1, then all ten add/sub of k-1 / k-2 as
   one contiguous block, then k-2's stores.  Uop layout per butterfly is
   [3 load][9 multiply][10 add-sub][3 store] with no multiply inside the add block. */
#define DEEP(X,Y,Z) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) E_X7(Y) E_X8(Y) E_X9(Y) \
    E_S1(X) E_S2(X) E_D(X) E_S3(Y) \
    E_A1(Z) E_A2(Z) E_A3(Z) E_A4(Z) E_A5(Z) E_A6(Z) E_T0(Z) E_T1(Z) E_T2(Z)
/* same, but the three stores are spread through the add block */
#define DEEPS(X,Y,Z) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) E_X7(Y) E_X8(Y) E_X9(Y) \
    E_S1(X) E_S2(X) E_D(X) E_S3(Y) \
    E_A1(Z) E_A2(Z) E_A3(Z) E_T1(Z) E_A4(Z) E_A5(Z) E_T2(Z) E_A6(Z) E_T0(Z)
/* same, but add block first, then the multiplies */
#define DEEPA(X,Y,Z) E_L0(X) E_L1(X) E_L2(X) \
    E_A1(Z) E_A2(Z) E_A3(Z) E_A4(Z) E_A5(Z) E_A6(Z) E_T0(Z) E_T1(Z) E_T2(Z) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) E_S1(X) E_S2(X) E_D(X) \
    E_X7(Y) E_X8(Y) E_X9(Y) E_S3(Y)
/* (12) deep pipeline with the loads and stores spread through the ALU stream */
#define DEEPSL(X,Y,Z) E_L1(X) E_X1(X) E_X2(X) E_L2(X) E_X4(X) E_X5(X) E_L0(X) \
    E_X3(X) E_X6(X) E_X7(Y) E_X8(Y) E_X9(Y) \
    E_S1(X) E_S2(X) E_D(X) E_S3(Y) \
    E_A1(Z) E_A2(Z) E_A3(Z) E_T1(Z) E_A4(Z) E_A5(Z) E_T2(Z) E_A6(Z) E_T0(Z)
/* (13) same schedules with the three stores removed, to price the stores in situ */
#define DEEPNS(X,Y,Z) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_X4(X) E_X5(X) E_X6(X) E_X7(Y) E_X8(Y) E_X9(Y) \
    E_S1(X) E_S2(X) E_D(X) E_S3(Y) \
    E_A1(Z) E_A2(Z) E_A3(Z) E_A4(Z) E_A5(Z) E_A6(Z)
#define NATNS(X) E_L0(X) E_L1(X) E_L2(X) \
    E_X1(X) E_X2(X) E_X3(X) E_S1(X) E_X4(X) E_X5(X) E_X6(X) E_S2(X) E_D(X) \
    E_X7(X) E_X8(X) E_X9(X) E_S3(X) \
    E_A1(X) E_A2(X) E_A3(X) E_A4(X) E_A5(X) E_A6(X)
#define DEEP9(F) F(B0,B8,B7) F(B1,B0,B8) F(B2,B1,B0) F(B3,B2,B1) F(B4,B3,B2) \
                 F(B5,B4,B3) F(B6,B5,B4) F(B7,B6,B5) F(B8,B7,B6)

/* 18 ymm butterflies = the same 9 zmm butterflies' worth of work */
#define Y0  RS0,0,18,36
#define Y1  RS1,1,19,37
#define Y2  RS2,2,20,38
#define Y3  RS0,3,21,39
#define Y4  RS1,4,22,40
#define Y5  RS2,5,23,41
#define Y6  RS0,6,24,42
#define Y7  RS1,7,25,43
#define Y8  RS2,8,26,44
#define Y9  RS0,9,27,45
#define Y10 RS1,10,28,46
#define Y11 RS2,11,29,47
#define Y12 RS0,12,30,48
#define Y13 RS1,13,31,49
#define Y14 RS2,14,32,50
#define Y15 RS0,15,33,51
#define Y16 RS1,16,34,52
#define Y17 RS2,17,35,53
#define ALL18(F) F(Y0) F(Y1) F(Y2) F(Y3) F(Y4) F(Y5) F(Y6) F(Y7) F(Y8) \
                 F(Y9) F(Y10) F(Y11) F(Y12) F(Y13) F(Y14) F(Y15) F(Y16) F(Y17)

/* ------------------------------------------------------------- test bodies */
#define ZCLOB "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9", \
              "zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17", \
              "zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31"

#define RUNNER(name, BODY) \
static void run_##name(void *p, const void *c, long n) { \
    asm volatile(LOADCONST "1:\n" BODY "subq $1,%2\n jnz 1b\n" \
                 : "+r"(p), "+r"(c), "+r"(n) : : "memory", "cc", ZCLOB); \
}

RUNNER(nat,     ALL9(NAT))
RUNNER(grp,     ALL9(GRP))
RUNNER(pipe,    PIPE9(PIPE))
RUNNER(mix11,   PIPE9(MIX11))
RUNNER(mix22,   PIPE9(MIX22))
RUNNER(memnat,  ALL9(MEMNAT))
RUNNER(memmix,  MPIPE9(MEMMIX))
RUNNER(deep,    DEEP9(DEEP))
RUNNER(deeps,   DEEP9(DEEPS))
RUNNER(deepa,   DEEP9(DEEPA))
RUNNER(deepsl,  DEEP9(DEEPSL))
RUNNER(deepns,  DEEP9(DEEPNS))
RUNNER(natns,   ALL9(NATNS))
RUNNER(ymm,     ALL18(YNAT))
RUNNER(mulonly, ALL9(MULONLY))
RUNNER(addonly, ALL9(ADDONLY))

/* ---------------------------------------------------------------- harness */
static int16_t data[54 * 16] __attribute__((aligned(64)));
static int16_t consts[7 * 32] __attribute__((aligned(64)));

typedef void (*fn_t)(void *, const void *, long);

/* =================================================================== */
/* Synthetic allocator probe: a stream of independent p0-only multiplies */
/* and independent p0/p5 adds in a fixed repeating pattern, no memory    */
/* traffic and no dependences, so only the rename-time port assignment   */
/* can explain the p0/p5 split.                                          */
#define MU(n) "vpmullw %%zmm20,%%zmm21,%%zmm" #n "\n"
#define AD(n) "vpaddw %%zmm22,%%zmm23,%%zmm" #n "\n"
#define LDU(n) "vmovdqa64 " #n "*64(%0),%%zmm" #n "\n"
#define STU(n) "vmovdqa64 %%zmm" #n "," #n "*64(%0)\n"

#define U_1_1   MU(0) AD(1)
#define U_1_2   MU(0) AD(1) AD(2)
#define U_2_1   MU(0) MU(1) AD(2)
#define U_2_2   MU(0) MU(1) AD(2) AD(3)
#define U_3_3   MU(0) MU(1) MU(2) AD(3) AD(4) AD(5)
#define U_5_5   MU(0) MU(1) MU(2) MU(3) MU(4) AD(5) AD(6) AD(7) AD(8) AD(9)
#define U_9_10  MU(0) MU(1) MU(2) MU(3) MU(4) MU(5) MU(6) MU(7) MU(8) \
                AD(0) AD(1) AD(2) AD(3) AD(4) AD(5) AD(6) AD(7) AD(8) AD(9)
#define U_9_10i MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9)
#define U_9_10m MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                LDU(10) LDU(11) LDU(12) STU(13) STU(14) STU(15)
#define U_9_10s MU(0) MU(1) MU(2) MU(3) MU(4) MU(5) MU(6) MU(7) MU(8) \
                AD(0) AD(1) AD(2) AD(3) AD(4) AD(5) AD(6) AD(7) AD(8) AD(9) \
                LDU(10) LDU(11) LDU(12) STU(13) STU(14) STU(15)
/* one add per five multiplies: p0 must be the only sensible choice for none */
#define U_9_5   MU(0) AD(0) MU(1) MU(2) AD(1) MU(3) MU(4) AD(2) MU(5) MU(6) AD(3) \
                MU(7) MU(8) AD(4)
#define U_9_20  MU(0) AD(0) AD(1) MU(1) AD(2) AD(3) MU(2) AD(4) AD(5) MU(3) AD(6) AD(7) \
                MU(4) AD(8) AD(9) MU(5) AD(10) AD(11) MU(6) AD(12) AD(13) \
                MU(7) AD(14) AD(15) MU(8) AD(0) AD(1) AD(2) AD(3)

#define U_L3    MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                LDU(10) LDU(11) LDU(12)
#define U_S3    MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                STU(13) STU(14) STU(15)
#define U_LS1   MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                LDU(10) STU(13)
#define U_LS2   MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                LDU(10) LDU(11) STU(13) STU(14)
#define U_NOP6  MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                "nopw 0x0(%%rax,%%rax,1)\n nopw 0x0(%%rax,%%rax,1)\n nopw 0x0(%%rax,%%rax,1)\n" \
                "nopw 0x0(%%rax,%%rax,1)\n nopw 0x0(%%rax,%%rax,1)\n nopw 0x0(%%rax,%%rax,1)\n"
#define U_MOV6  MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) \
                "vmovdqa64 %%zmm20,%%zmm10\n vmovdqa64 %%zmm20,%%zmm11\n vmovdqa64 %%zmm20,%%zmm12\n" \
                "vmovdqa64 %%zmm20,%%zmm13\n vmovdqa64 %%zmm20,%%zmm14\n vmovdqa64 %%zmm20,%%zmm15\n"

/* distinct-cache-line variants: unit k touches lines 3k..3k+2 */
#define STO(k,j) "vmovdqa64 %%zmm13,(" #k "*3+" #j ")*64(%0)\n"
#define LDO(k,j) "vmovdqa64 (" #k "*3+" #j ")*64(%0),%%zmm10\n"
#define UV_S3(k)  MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) STO(k,0) STO(k,1) STO(k,2)
#define UV_LS3(k) MU(0) AD(0) MU(1) AD(1) MU(2) AD(2) MU(3) AD(3) MU(4) AD(4) \
                MU(5) AD(5) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9) LDO(k,0) LDO(k,1) LDO(k,2) STO(k,0) STO(k,1) STO(k,2)
#define REP8I(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define PRUNNERI(name, U) \
static void pat_##name(void *p, const void *c, long n) { \
    asm volatile(LOADCONST "vmovdqa64 %%zmm25,%%zmm20\n vmovdqa64 %%zmm26,%%zmm21\n" \
                 "vmovdqa64 %%zmm27,%%zmm22\n vmovdqa64 %%zmm28,%%zmm23\n" \
                 "1:\n" REP8I(U) "subq $1,%2\n jnz 1b\n" \
                 : "+r"(p), "+r"(c), "+r"(n) : : "memory", "cc", PCLOB, \
                   "zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31"); \
}

/* dependence probe: identical 9 multiply / 10 add uop mix, but now every add reads a
   multiply result produced in the same unit, so the adds sit in the scheduler unready. */
#define MUD(n) "vpmullw %%zmm20,%%zmm21,%%zmm" #n "\n"
#define ADD_(n) "vpaddw %%zmm22,%%zmm" #n ",%%zmm" #n "\n"
#define U_DEP  MUD(0) MUD(1) MUD(2) MUD(3) MUD(4) MUD(5) MUD(6) MUD(7) MUD(8) \
               ADD_(0) ADD_(1) ADD_(2) ADD_(3) ADD_(4) ADD_(5) ADD_(6) ADD_(7) ADD_(8) ADD_(0)
#define U_DEPi MUD(0) MUD(1) MUD(2) MUD(3) MUD(4) MUD(5) MUD(6) MUD(7) MUD(8) \
               ADD_(0) ADD_(1) ADD_(2) ADD_(3) ADD_(4) ADD_(5) ADD_(6) ADD_(7) ADD_(8) ADD_(0) \
               LDU(10) LDU(11) LDU(12) STU(13) STU(14) STU(15)
/* half dependent, half independent */
#define U_DEPh MUD(0) MUD(1) MUD(2) MUD(3) MUD(4) MUD(5) MUD(6) MUD(7) MUD(8) \
               ADD_(0) ADD_(1) ADD_(2) ADD_(3) ADD_(4) AD(5) AD(6) AD(7) AD(8) AD(9)

/* memory ops spread through the ALU stream instead of clustered */
#define U_LSSPR MU(0) AD(0) LDU(10) MU(1) AD(1) STU(13) MU(2) AD(2) LDU(11) MU(3) AD(3) \
                STU(14) MU(4) AD(4) LDU(12) MU(5) AD(5) STU(15) MU(6) AD(6) MU(7) AD(7) \
                MU(8) AD(8) AD(9)
#define U_STSPR MU(0) AD(0) MU(1) AD(1) STU(13) MU(2) AD(2) MU(3) AD(3) STU(14) \
                MU(4) AD(4) MU(5) AD(5) STU(15) MU(6) AD(6) MU(7) AD(7) MU(8) AD(8) AD(9)

#define REP8(X) X X X X X X X X
#define PCLOB "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9", \
              "zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm20","zmm21","zmm22","zmm23"
#define PRUNNER(name, U) \
static void pat_##name(void *p, const void *c, long n) { \
    asm volatile(LOADCONST "vmovdqa64 %%zmm25,%%zmm20\n vmovdqa64 %%zmm26,%%zmm21\n" \
                 "vmovdqa64 %%zmm27,%%zmm22\n vmovdqa64 %%zmm28,%%zmm23\n" \
                 "1:\n" REP8(U) "subq $1,%2\n jnz 1b\n" \
                 : "+r"(p), "+r"(c), "+r"(n) : : "memory", "cc", PCLOB, \
                   "zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31"); \
}
PRUNNER(p11,   U_1_1)
PRUNNER(p12,   U_1_2)
PRUNNER(p21,   U_2_1)
PRUNNER(p22,   U_2_2)
PRUNNER(p33,   U_3_3)
PRUNNER(p55,   U_5_5)
PRUNNER(p910,  U_9_10)
PRUNNER(p910i, U_9_10i)
PRUNNER(p910m, U_9_10m)
PRUNNER(p910s, U_9_10s)
PRUNNER(p95,   U_9_5)
PRUNNER(p920,  U_9_20)
PRUNNER(l3,    U_L3)
PRUNNER(s3,    U_S3)
PRUNNER(ls1,   U_LS1)
PRUNNER(ls2,   U_LS2)
PRUNNER(nop6,  U_NOP6)
PRUNNER(mov6,  U_MOV6)
PRUNNERI(vs3,  UV_S3)
PRUNNERI(vls3, UV_LS3)
PRUNNER(dep,   U_DEP)
PRUNNER(depi,  U_DEPi)
PRUNNER(deph,  U_DEPh)
PRUNNER(lsspr, U_LSSPR)
PRUNNER(stspr, U_STSPR)

static const struct { const char *name; fn_t f; double mul, add; } PAT[] = {
    { "1mul:1add",   pat_p11,   1, 1 },
    { "1mul:2add",   pat_p12,   1, 2 },
    { "2mul:1add",   pat_p21,   2, 1 },
    { "2mul:2add",   pat_p22,   2, 2 },
    { "3mul:3add",   pat_p33,   3, 3 },
    { "5mul:5add",   pat_p55,   5, 5 },
    { "9mul|10add",  pat_p910,  9, 10 },
    { "9:10 alt",    pat_p910i, 9, 10 },
    { "9:10 alt+ls", pat_p910m, 9, 10 },
    { "9|10 blk+ls", pat_p910s, 9, 10 },
    { "9mul:5add",   pat_p95,   9, 5 },
    { "9mul:20add",  pat_p920,  9, 20 },
    { "9:10 +3ld",   pat_l3,    9, 10 },
    { "9:10 +3st",   pat_s3,    9, 10 },
    { "9:10 +1ld1st",pat_ls1,   9, 10 },
    { "9:10 +2ld2st",pat_ls2,   9, 10 },
    { "9:10 +6nop",  pat_nop6,  9, 10 },
    { "9:10 +6vmov", pat_mov6,  9, 10 },
    { "9:10 +3st/ln",pat_vs3,   9, 10 },
    { "9:10 +3l3s/ln",pat_vls3, 9, 10 },
    { "9:10 dep",    pat_dep,   9, 10 },
    { "9:10 dep+ls", pat_depi,  9, 10 },
    { "9:10 half dep",pat_deph, 9, 10 },
    { "9:10 ls sprd",pat_lsspr, 9, 10 },
    { "9:10 3st sprd",pat_stspr,9, 10 },
};
#define NPAT ((int)(sizeof(PAT)/sizeof(PAT[0])))
static const struct { const char *name; fn_t f; double bf; const char *note; } V[] = {
    { "1 nat",        run_nat,     9, "natural, mont by mont" },
    { "2 grp",        run_grp,     9, "6mul|3sub|3mul|1sub|6add" },
    { "3 pipe",       run_pipe,    9, "block pipeline P1(k)|P2(k-1)" },
    { "4 mix11",      run_mix11,   9, "1 mul : 1 add, pipelined" },
    { "5 mix22",      run_mix22,   9, "2 mul : 2 add, pipelined" },
    { "6 memnat",     run_memnat,  9, "mem operands for a1,a2" },
    { "7 memmix",     run_memmix,  9, "mem operands + 1:1" },
    { "9 deep",       run_deep,    9, "[3ld][9mul][10add][3st]" },
    { "10 deep+st",   run_deeps,   9, "deep, stores spread in add block" },
    { "11 deep add1", run_deepa,   9, "deep, add block before mul block" },
    { "12 deep sprd", run_deepsl, 9, "deep + ld/st spread" },
    { "13 deep -st",  run_deepns,  9, "deep, stores removed" },
    { "14 nat -st",   run_natns,   9, "natural, stores removed" },
    { "8 ymm",        run_ymm,     9, "256-bit, 18 butterflies" },
    { "L mul only",   run_mulonly, 9, "9 mul, 0 add" },
    { "L add only",   run_addonly, 9, "0 mul, 10 add" },
};
#define NV ((int)(sizeof(V)/sizeof(V[0])))
#define ITERS 20000L
#define RUNS 9

int main(void) {
    perf_init();
    for (int i = 0; i < 54 * 16; i++) data[i] = (int16_t)(i * 2654435761u >> 3);
    static const int16_t cv[7] = { 3889, -1809, 1200, 901, -1500, 777, -333 };
    for (int j = 0; j < 7; j++) for (int i = 0; i < 32; i++) consts[j * 32 + i] = cv[j];

    for (int w = 0; w < 30; w++) run_nat(data, consts, 2000);   /* AVX-512 licence + freq */

    printf("%-12s %8s %8s %8s %8s %8s  %s\n", "variant",
           evname[0], evname[1], evname[2], evname[3], evname[4], "note");
    for (int v = 0; v < NV; v++) {
        double best[NEV]; best[0] = 1e30;
        for (int r = 0; r < RUNS; r++) {
            double s[NEV];
            V[v].f(data, consts, 200);
            perf_start();
            V[v].f(data, consts, ITERS);
            perf_stop(s);
            if (s[0] < best[0]) memcpy(best, s, sizeof(best));
        }
        double n = ITERS * V[v].bf;
        printf("%-12s %8.3f %8.3f %8.3f %8.3f %8.3f  %s\n", V[v].name,
               best[0]/n, best[1]/n, best[2]/n, best[3]/n, best[4]/n, V[v].note);
    }

    printf("\n%-12s %8s %8s %8s %8s %8s %8s %8s\n", "pattern/unit",
           evname[0], evname[1], evname[2], evname[3], evname[4], "ideal", "leak%");
    for (int v = 0; v < NPAT; v++) {
        double best[NEV]; best[0] = 1e30;
        for (int r = 0; r < RUNS; r++) {
            double s[NEV];
            PAT[v].f(data, consts, 200);
            perf_start();
            PAT[v].f(data, consts, ITERS);
            perf_stop(s);
            if (s[0] < best[0]) memcpy(best, s, sizeof(best));
        }
        double n = ITERS * 8.0;                 /* REP8 units per iteration */
        double cyc = best[0]/n, p0 = best[1]/n, p5 = best[3]/n;
        double ideal = PAT[v].mul > (PAT[v].mul+PAT[v].add)/2 ? PAT[v].mul
                                                             : (PAT[v].mul+PAT[v].add)/2;
        (void)p5;
        printf("%-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.1f\n", PAT[v].name,
               cyc, p0, best[2]/n, best[3]/n, best[4]/n, ideal,
               100.0*(p0-PAT[v].mul)/PAT[v].add);
    }
    return 0;
}

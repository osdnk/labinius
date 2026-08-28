// Instruction throughput/latency microbenchmark for the local Tiger Lake core.
// Counts real core cycles via perf_event_open (PERF_COUNT_HW_CPU_CYCLES).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <immintrin.h>

static int fd;
static void perf_init(void) {
    struct perf_event_attr pe; memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES; pe.disabled = 1;
    pe.exclude_kernel = 1; pe.exclude_hv = 1;
    fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd < 0) { perror("perf_event_open"); exit(1); }
}
static inline uint64_t cyc(void) { uint64_t c; if (read(fd, &c, 8) != 8) exit(2); return c; }

#define ITERS 20000
static uint8_t buf[1 << 16] __attribute__((aligned(64)));
static uint32_t masks[64] __attribute__((aligned(64)));

// THROUGHPUT: 12 independent copies per iteration. LATENCY: 12 dependent copies.
#define TP_TEST(name, insn) \
static double tp_##name(void) { \
    uint64_t t0, t1; void *p = buf; void *m = masks; (void)p; (void)m; \
    asm volatile("vpxord %%zmm0,%%zmm0,%%zmm0\n vpxord %%zmm1,%%zmm1,%%zmm1\n vpxord %%zmm2,%%zmm2,%%zmm2\n vpxord %%zmm3,%%zmm3,%%zmm3\n" \
                 "vpxord %%zmm4,%%zmm4,%%zmm4\n vpxord %%zmm5,%%zmm5,%%zmm5\n vpxord %%zmm6,%%zmm6,%%zmm6\n vpxord %%zmm7,%%zmm7,%%zmm7\n" \
                 "vpxord %%zmm8,%%zmm8,%%zmm8\n vpxord %%zmm9,%%zmm9,%%zmm9\n vpxord %%zmm10,%%zmm10,%%zmm10\n vpxord %%zmm11,%%zmm11,%%zmm11\n" \
                 "vpxord %%zmm20,%%zmm20,%%zmm20\n vpxord %%zmm21,%%zmm21,%%zmm21\n vpxord %%zmm22,%%zmm22,%%zmm22\n kxnord %%k1,%%k1,%%k1\n" ::: "memory"); \
    ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); \
    t0 = cyc(); \
    for (int i = 0; i < ITERS; i++) { \
        asm volatile( \
            insn(0) insn(1) insn(2) insn(3) insn(4) insn(5) insn(6) insn(7) insn(8) insn(9) insn(10) insn(11) \
            : : "r"(p), "r"(m) : "memory", "k1","k2","k3","k4","k5","k6", "rax"); \
    } \
    t1 = cyc(); ioctl(fd, PERF_EVENT_IOC_DISABLE, 0); \
    return (double)(t1 - t0) / (ITERS * 12.0); \
}

#define R(n) "%%zmm" #n
#define Y(n) "%%ymm" #n
// independent: dst = zmm n, sources zmm20/21/22
#define I_VPMULLW(n)   "vpmullw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULHW(n)   "vpmulhw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULHUW(n)  "vpmulhuw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULHRSW(n) "vpmulhrsw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMADDWD(n)  "vpmaddwd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPDPWSSD(n)  "vpdpwssd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPDPBUSD(n)  "vpdpbusd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMADDUBSW(n) "vpmaddubsw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULLD(n)   "vpmulld %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULLW_Y(n) "vpmullw %%ymm20,%%ymm21," Y(n) "\n"
#define I_VPMULHW_Y(n) "vpmulhw %%ymm20,%%ymm21," Y(n) "\n"
#define I_VPADDW(n)    "vpaddw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPADDW_Y(n)  "vpaddw %%ymm20,%%ymm21," Y(n) "\n"
#define I_VPSUBW(n)    "vpsubw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMINUW(n)   "vpminuw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPMAXSW(n)   "vpmaxsw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPSRAW(n)    "vpsraw $3,%%zmm20," R(n) "\n"
#define I_VPSRLW(n)    "vpsrlw $3,%%zmm20," R(n) "\n"
#define I_VPSLLW(n)    "vpsllw $3,%%zmm20," R(n) "\n"
#define I_VPANDD(n)    "vpandd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPTERNLOGD(n) "vpternlogd $0x96,%%zmm20,%%zmm21," R(n) "\n"
#define I_VPADDW_K(n)  "vpaddw %%zmm20,%%zmm21," R(n) "%{%%k1%}\n"
#define I_VPADDW_KZ(n) "vpaddw %%zmm20,%%zmm21," R(n) "%{%%k1%}%{z%}\n"
#define I_VPBLENDMW(n) "vpblendmw %%zmm20,%%zmm21," R(n) "%{%%k1%}\n"
#define I_VPERMW(n)    "vpermw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMI2W(n)  "vpermi2w %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMT2W(n)  "vpermt2w %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMB(n)    "vpermb %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMI2B(n)  "vpermi2b %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMD(n)    "vpermd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMQ(n)    "vpermq %%zmm20,%%zmm21," R(n) "\n"
#define I_VPERMI2Q(n)  "vpermi2q %%zmm20,%%zmm21," R(n) "\n"
#define I_VPSHUFB(n)   "vpshufb %%zmm20,%%zmm21," R(n) "\n"
#define I_VPSHUFD(n)   "vpshufd $0x4e,%%zmm20," R(n) "\n"
#define I_VPUNPCKLWD(n) "vpunpcklwd %%zmm20,%%zmm21," R(n) "\n"
#define I_VPUNPCKLBW(n) "vpunpcklbw %%zmm20,%%zmm21," R(n) "\n"
#define I_VPALIGNR(n)  "vpalignr $2,%%zmm20,%%zmm21," R(n) "\n"
#define I_VSHUFI64X2(n) "vshufi64x2 $0x4e,%%zmm20,%%zmm21," R(n) "\n"
#define I_VPMOVM2W(n)  "vpmovm2w %%k1," R(n) "\n"
#define I_VPMOVW2M(n)  "vpmovw2m %%zmm" #n ",%%k" "2\n"
#define I_VPCMPUW(n)   "vpcmpuw $1,%%zmm20,%%zmm" #n ",%%k2\n"
#define I_VPTESTMW(n)  "vptestmw %%zmm20,%%zmm" #n ",%%k2\n"
#define I_GF2P8AFF(n)  "vgf2p8affineqb $0,%%zmm20,%%zmm21," R(n) "\n"
#define I_VPMULTISHIFT(n) "vpmultishiftqb %%zmm20,%%zmm21," R(n) "\n"
#define I_VPSHLDW(n)   "vpshldw $3,%%zmm20,%%zmm21," R(n) "\n"
#define I_VPSHRDW(n)   "vpshrdw $3,%%zmm20,%%zmm21," R(n) "\n"
#define I_VPOPCNTW(n)  "vpopcntw %%zmm20," R(n) "\n"
#define I_VPMOVZXBW(n) "vpmovzxbw %%ymm20," R(n) "\n"
#define I_VPMOVWB(n)   "vpmovwb %%zmm20," Y(n) "\n"
#define I_VPACKUSWB(n) "vpackuswb %%zmm20,%%zmm21," R(n) "\n"
#define I_VPBCASTW_M(n) "vpbroadcastw 2(%0)," R(n) "\n"
#define I_VPBCASTW_R(n) "vpbroadcastw %%xmm20," R(n) "\n"
#define I_VPBCASTD_M(n) "vpbroadcastd 4(%0)," R(n) "\n"
#define I_VPADDW_BCASTD(n) "vpaddd 4(%0)%{1to16%},%%zmm21," R(n) "\n"
#define I_VPMULLW_M(n) "vpmullw 64(%0),%%zmm21," R(n) "\n"
#define I_KMOVD_M(n)   "kmovd 4(%1),%%k2\n"
#define I_KMOVD_R(n)   "kmovd %%eax,%%k2\n"
#define I_VPADDW_KM(n) "kmovd " #n "*4(%1),%%k2\n vpaddw %%zmm20,%%zmm21," R(n) "%{%%k2%}\n"
#define I_LOADZ(n)     "vmovdqu64 " #n "*64(%0)," R(n) "\n"
#define I_STOREZ(n)    "vmovdqu64 %%zmm20," #n "*64(%0)\n"
#define I_STORENT(n)   "vmovntdq %%zmm20," #n "*64(%0)\n"
// mixed: 1 mul + 1 add (tests whether adds co-issue on p5 with muls on p0)
#define I_MIX_MULADD(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpaddw %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MULPERM(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpermw %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MULPERMI2W(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vmovdqa64 %%zmm22,%%zmm" #n "\n vpermi2w %%zmm20,%%zmm21,%%zmm" #n "\n"
#define I_MIX_MULADDADD(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpaddw %%zmm20,%%zmm22,%%zmm" #n "\n vpaddw %%zmm21,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MUL_MUL_ADD_ADD_ADD(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpmulhw %%zmm20,%%zmm21," R(n) "\n vpaddw %%zmm20,%%zmm22,%%zmm" #n "\n vpaddw %%zmm21,%%zmm22,%%zmm" #n "\n vpsubw %%zmm21,%%zmm22,%%zmm" #n "\n"
#define I_MIX_GF_PERMB(n) "vgf2p8affineqb $0,%%zmm20,%%zmm21," R(n) "\n vpermb %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MUL_GF(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vgf2p8affineqb $0,%%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MUL_MINUW(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpminuw %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MUL_SRAW(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpsraw $3,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MUL_MOVM2W(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpmovm2w %%k1,%%zmm" #n "\n"
#define I_MIX_MUL_KADD(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpaddw %%zmm20,%%zmm22,%%zmm" #n "%{%%k1%}\n"
#define I_MIX_MUL_BCASTW(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpbroadcastw 2(%0),%%zmm" #n "\n"
#define I_MIX_MUL_MULHRS(n) "vpmullw %%zmm20,%%zmm21," R(n) "\n vpmulhrsw %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MADDWD_ADD(n) "vpmaddwd %%zmm20,%%zmm21," R(n) "\n vpaddd %%zmm20,%%zmm22,%%zmm" #n "\n"
#define I_MIX_MULY_ADDY(n) "vpmullw %%ymm20,%%ymm21," Y(n) "\n vpaddw %%ymm20,%%ymm22,%%ymm" #n "\n"
#define I_MIX_MULY_MULY_ADDY(n) "vpmullw %%ymm20,%%ymm21," Y(n) "\n vpmulhw %%ymm20,%%ymm21," Y(n) "\n vpaddw %%ymm20,%%ymm22,%%ymm" #n "\n"

// latency: chain through zmm0
#define L_VPMULLW(n)   "vpmullw %%zmm20,%%zmm0,%%zmm0\n"
#define L_VPMULHW(n)   "vpmulhw %%zmm20,%%zmm0,%%zmm0\n"
#define L_VPADDW(n)    "vpaddw %%zmm20,%%zmm0,%%zmm0\n"
#define L_VPERMW(n)    "vpermw %%zmm0,%%zmm21,%%zmm0\n"
#define L_VPERMI2W(n)  "vpermi2w %%zmm20,%%zmm21,%%zmm0\n"
#define L_VPERMB(n)    "vpermb %%zmm0,%%zmm21,%%zmm0\n"
#define L_VPERMI2B(n)  "vpermi2b %%zmm20,%%zmm21,%%zmm0\n"
#define L_GF2P8AFF(n)  "vgf2p8affineqb $0,%%zmm20,%%zmm0,%%zmm0\n"
#define L_VPMADDWD(n)  "vpmaddwd %%zmm20,%%zmm0,%%zmm0\n"
#define L_VPDPWSSD(n)  "vpdpwssd %%zmm20,%%zmm21,%%zmm0\n"
#define L_VPMINUW(n)   "vpminuw %%zmm20,%%zmm0,%%zmm0\n"
#define L_VPMULHRSW(n) "vpmulhrsw %%zmm20,%%zmm0,%%zmm0\n"

#define X(name, insn) TP_TEST(name, insn)
#include "tests.inc"
#undef X

int main(void) {
    perf_init();
    for (int i = 0; i < 64; i++) masks[i] = 0x12345678u * (i + 1);
    // warm up (frequency ramp / AVX-512 license)
    for (int i = 0; i < 20; i++) tp_VPMULLW();
    printf("%-26s %8s\n", "test", "cyc/op");
#define X(name, insn) { double best = 1e9; for (int r = 0; r < 7; r++) { double v = tp_##name(); if (v < best) best = v; } printf("%-26s %8.3f\n", #name, best); }
#include "tests.inc"
#undef X
    return 0;
}

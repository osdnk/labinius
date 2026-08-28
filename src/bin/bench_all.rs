//! Headline benchmark: 2^18 binary polynomials (a plain `Vec<BinaryPoly>`) -> preallocated NTT
//! output, both primes, all kernels, single thread pinned to one core. Prints total milliseconds
//! and per-ring-element cycles / instructions / uops (perf counters), plus two "multiply in the
//! NTT domain" benches on 2^18 elements. Usage: `bench_all [cpu] [--quick]`.
use bin_ntt::params::{Params, N};
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::horizontal_gen as hg;
use bin_ntt::simd::pointwise::{self, MontElement};
use bin_ntt::simd::{vertical_bin as vb, vertical_gen as vg};
use bin_ntt::types::*;
use std::time::Instant;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

struct Meter {
    perf: Option<PerfGroup>,
}

struct Result {
    ms: f64,
    counts: Option<Counts>,
}

impl Meter {
    fn new() -> Self {
        let perf = match PerfGroup::new() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("perf counters unavailable ({e}); reporting wall time only");
                None
            }
        };
        Meter { perf }
    }
    fn run(&self, f: impl FnOnce()) -> Result {
        if let Some(p) = &self.perf {
            p.start();
        }
        let t0 = Instant::now();
        f();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let counts = self.perf.as_ref().map(|p| p.stop());
        Result { ms, counts }
    }
    /// Best (minimum wall time) of `reps` runs of `f`, which does its own untimed preparation
    /// and then calls `m.run(..)` for the timed part.
    fn best(&self, reps: usize, mut f: impl FnMut(&Meter) -> Result) -> Result {
        let mut best: Option<Result> = None;
        for _ in 0..reps {
            let r = f(self);
            if best.as_ref().map_or(true, |b| r.ms < b.ms) {
                best = Some(r);
            }
        }
        best.unwrap()
    }
}

fn report(label: &str, r: &Result, elements: usize) {
    let n = elements as f64;
    print!("{label:<58} {:8.2} ms", r.ms);
    if let Some(c) = &r.counts {
        print!(
            "   per element: {:6.1} cyc {:7.1} ins {:7.1} uops (p0 {:6.1}, p5 {:6.1})   {:.2} GHz",
            c.cycles as f64 / n,
            c.instructions as f64 / n,
            c.uops as f64 / n,
            c.port0 as f64 / n,
            c.port5 as f64 / n,
            c.cycles as f64 / (r.ms * 1e6)
        );
    }
    println!();
}

fn random_polys(n: usize, seed: u64) -> Vec<BinaryPoly> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| BinaryPoly::random(&mut rng)).collect()
}

/// y += sum over the batch of a[j][p] * w[j][p] (Montgomery product, then pairwise 16-bit ->
/// 32-bit accumulation with vpmaddwd against ones). acc[j] holds 16 i32 partial sums of slot j.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_products<const Q: u16>(w: &Batch32, a: &Batch32, acc: &mut [[i32; 16]; N]) {
    use core::arch::x86_64::*;
    let ones = _mm512_set1_epi16(1);
    for j in 0..N {
        let x = _mm512_load_si512(w.v[j].as_ptr() as *const __m512i);
        let y = _mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
        let p = pointwise::mont_mul_epi16::<Q>(x, y);
        let s = _mm512_madd_epi16(p, ones);
        let acc_j = _mm512_loadu_si512(acc[j].as_ptr() as *const __m512i);
        _mm512_storeu_si512(acc[j].as_mut_ptr() as *mut __m512i, _mm512_add_epi32(acc_j, s));
    }
}

/// Finishes the accumulation: y[j] = (sum of the 16 lanes) * 2^16 mod q (undoing the Montgomery
/// factor of `mont_mul_epi16`), fully reduced.
fn finish_accumulator<const Q: u16>(acc: &[[i32; 16]; N]) -> [u32; N] {
    let q = Q as i64;
    let mut y = [0u32; N];
    for j in 0..N {
        let s: i64 = acc[j].iter().map(|&x| x as i64).sum();
        y[j] = ((s.rem_euclid(q) * 65536) % q) as u32;
    }
    y
}

fn headline<const Q: u16>(m: &Meter, polys: &[BinaryPoly], reps: usize) {
    let n = polys.len();
    let batches = n / 32;
    println!("\n=== q = {Q}: 2^{} binary polynomials -> preallocated NTT output ===", n.trailing_zeros());
    // --- vertical binary kernel (the deliverable): Vec<BinaryPoly> -> Vec<Batch32>, NT stores.
    let mut out: Vec<Batch32> = (0..batches).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let r = m.best(reps, |m| m.run(|| vb::ntt_bin_polys::<Q>(polys, &mut out)));
    report("vertical_bin: BinaryPoly -> Batch32 (materialised, NT)", &r, n);
    // spot-check against the scalar reference
    let mut rng = Rng::new(99);
    for _ in 0..4 {
        let i = rng.below(n as u32) as usize;
        let want = scalar::ntt::<Q>(&scalar::lift(&polys[i]));
        let got = scalar::normalize_i16(&out[i / 32].get(i % 32).v, Q);
        assert_eq!(got, want, "vertical_bin mismatch at element {i}");
    }
    // --- streamed to a trivial consumer
    let mut sink = 0i16;
    let r = m.best(reps, |m| {
        m.run(|| vb::ntt_bin_stream::<Q>(polys, |_, b| sink ^= b.v[0][0] ^ b.v[647][31]))
    });
    report("vertical_bin: streamed (closure per batch of 32)", &r, n);
    std::hint::black_box(sink);
    // --- generic vertical kernel: in place on a preallocated Batch32 array of i16 coefficients.
    let pristine: Vec<Batch32> = polys
        .chunks(32)
        .map(|c| Batch32::from_binary(c.try_into().unwrap()))
        .collect();
    let mut work = pristine.clone();
    let r = m.best(reps, |m| {
        work.clone_from(&pristine);
        m.run(|| vg::ntt_gen_batches::<Q>(&mut work))
    });
    report("vertical_gen: generic i16 Batch32 in place (for comparison)", &r, n);
    let i = 12345;
    let want = scalar::ntt::<Q>(&scalar::lift(&polys[i]));
    assert_eq!(scalar::normalize_i16(&work[i / 32].get(i % 32).v, Q), want);
    drop(work);
    drop(pristine);
    // --- horizontal generic kernel (Gregor's layout): in place on HBatch4.
    let pristine: Vec<hg::HBatch4> = polys
        .chunks(4)
        .map(|c| hg::HBatch4::from_binary(c.try_into().unwrap()))
        .collect();
    let mut work = pristine.clone();
    let r = m.best(reps, |m| {
        work.clone_from(&pristine);
        m.run(|| unsafe { hg::ntt_gen_hbatch4_many::<Q>(&mut work) })
    });
    report("horizontal_gen: generic, 4 polys/zmm, in place (comparison)", &r, n);
    assert_eq!(scalar::normalize_i16(&work[i / 4].get(i % 4).v, Q), want);
}

fn multiplication<const Q: u16>(m: &Meter, polys: &[BinaryPoly], reps: usize) {
    let n = polys.len();
    let batches = n / 32;
    println!("\n=== q = {Q}: multiplication in the NTT domain on 2^{} elements ===", n.trailing_zeros());
    // (i) c_i = NTT(w_i) * b for one fixed ring element b (batch x element, 3 mults per slot).
    let mut rng = Rng::new(5);
    let mut b = RingElement::zero(Representation::Ntt);
    for j in 0..N {
        b.v[j] = rng.below(Q as u32) as i16;
    }
    let bm = MontElement::new::<Q>(&b);
    let mut out: Vec<Batch32> = (0..batches).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let r = m.best(reps, |m| {
        m.run(|| vb::ntt_bin_stream::<Q>(polys, |i, w| unsafe { pointwise::mul_batch_element::<Q>(w, &bm, &mut out[i]) }))
    });
    report("NTT(w_i) * b, fixed b, streamed, materialised", &r, n);
    let i = 777;
    let want = scalar::pointwise_mul(&scalar::ntt::<Q>(&scalar::lift(&polys[i])), &b.normalized(Q), Q);
    assert_eq!(scalar::normalize_i16(&out[i / 32].get(i % 32).v, Q), want);
    drop(out);
    // (ii) y = sum_i a_i * w_i with 2^18 distinct NTT-domain a_i streamed from memory (340 MB).
    let a: Vec<Batch32> = (0..batches)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(Q as u32) as i16;
                }
            }
            b
        })
        .collect();
    let mut acc = vec![[[0i32; 16]; N]; 1];
    let r = m.best(reps, |m| {
        acc[0] = [[0i32; 16]; N];
        m.run(|| vb::ntt_bin_stream::<Q>(polys, |i, w| unsafe { accumulate_products::<Q>(w, &a[i], &mut acc[0]) }))
    });
    report("y = sum_i a_i * NTT(w_i), 2^18 distinct a_i streamed", &r, n);
    // verify on the first 2 batches with the scalar reference
    let mut acc_small = [[0i32; 16]; N];
    let mut want = [0u64; N];
    for (bi, chunk) in polys[..64].chunks(32).enumerate() {
        let mut nt = Batch32::zero(Representation::Ntt);
        vb::ntt_bin_polys::<Q>(chunk, std::slice::from_mut(&mut nt));
        unsafe { accumulate_products::<Q>(&nt, &a[bi], &mut acc_small) };
        for p in 0..32 {
            let w = scalar::ntt::<Q>(&scalar::lift(&chunk[p]));
            for j in 0..N {
                want[j] = (want[j] + w[j] as u64 * a[bi].v[j][p] as u64) % Q as u64;
            }
        }
    }
    let got = finish_accumulator::<Q>(&acc_small);
    for j in 0..N {
        assert_eq!(got[j] as u64, want[j], "accumulate mismatch at slot {j}");
    }
    println!("  (accumulation verified against the scalar reference on 64 elements)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let quick = args.iter().any(|a| a == "--quick");
    pin(cpu);
    let m = Meter::new();
    let log_n = if quick { 14 } else { 18 };
    let reps = if quick { 2 } else { 3 };
    println!("bin-ntt bench_all: cpu {cpu}, 2^{log_n} elements, best of {reps} runs, single thread");
    println!("psi(3889) = {}, psi(9721) = {}", Params::<3889>::PSI, Params::<9721>::PSI);
    let polys = random_polys(1 << log_n, 42);
    headline::<3889>(&m, &polys, reps);
    headline::<9721>(&m, &polys, reps);
    multiplication::<3889>(&m, &polys, reps);
    multiplication::<9721>(&m, &polys, reps);
}

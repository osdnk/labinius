//! Headline benchmark for the `F162` front end: 2^18 `bin_fields::scalar::F162` (a plain
//! `&[F162]`, 6 MB) -> 2^16 ring elements in the vertical NTT layout, both primes, one core.
//! Usage: `taskset -c 2 bench_f162 [cpu] [--quick]`.
use bin_fields::scalar::F162;
use bin_ntt::f162::RandomF162;
use bin_ntt::params::N;
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::pointwise;
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::transpose_f162 as tf;
use bin_ntt::simd::vertical_bin_asm as va;
use bin_ntt::simd::ntt_f162 as nf;
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

struct Run {
    ms: f64,
    c: Counts,
}

/// Per ring element and per F162 (four elements make one ring element).
fn report(label: &str, r: &Run, elements: f64) {
    let e = elements;
    println!(
        "{label:<44} {:8.2} ms | per ring elt: {:6.1} cyc {:7.1} ins {:7.1} uops (p0 {:6.1}, p5 {:6.1}) \
         | per F162: {:5.1} cyc {:5.1} ins | {:.2} GHz",
        r.ms,
        r.c.cycles as f64 / e,
        r.c.instructions as f64 / e,
        r.c.uops as f64 / e,
        r.c.port0 as f64 / e,
        r.c.port5 as f64 / e,
        r.c.cycles as f64 / (4.0 * e),
        r.c.instructions as f64 / (4.0 * e),
        r.c.cycles as f64 / (r.ms * 1e6)
    );
}

fn best(pg: &PerfGroup, reps: usize, mut f: impl FnMut()) -> Run {
    let mut best: Option<Run> = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        pg.start();
        f();
        let c = pg.stop();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if best.as_ref().map_or(true, |b| ms < b.ms) {
            best = Some(Run { ms, c });
        }
    }
    best.unwrap()
}

fn random_elems(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| F162::random(&mut rng)).collect()
}

/// y += sum over the batch of a[j][p] * w[j][p].
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

const NB: usize = 8; // 256 ring elements: input 24 KB, index rows 83 KB, output 331 KB -> L2

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let quick = args.iter().any(|a| a == "--quick");
    pin(cpu);
    bin_ntt::f162::assert_layout();
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");
    let log_e = if quick { 14 } else { 18 }; // log2 of the number of F162
    let nf162 = 1usize << log_e;
    let nring = nf162 / 4;
    let nbatch = nring / 32;
    println!(
        "bin-ntt bench_f162: cpu {cpu}, 2^{log_e} F162 = 2^{} ring elements, best of 3, one thread",
        nring.trailing_zeros()
    );

    // ------------------------------------------------------------- cache-resident components
    let small = random_elems(128 * NB, 0xF162);
    let mut idxs: Vec<BinaryIndex32> = (0..NB).map(|_| BinaryIndex32::zero()).collect();
    let mut souts: Vec<Batch32> = (0..NB).map(|_| Batch32::zero(Representation::Ntt)).collect();

    const REP: usize = 400;
    let slice_all = |idxs: &mut Vec<BinaryIndex32>| {
        for b in 0..NB {
            unsafe {
                let c = &*(small.as_ptr().add(128 * b) as *const [F162; 128]);
                tf::slice_f162_into(c, &mut idxs[b]);
            }
        }
    };
    slice_all(&mut idxs);
    let r = best(&pg, 5, || {
        for _ in 0..REP {
            slice_all(&mut idxs);
        }
    });
    report("transpose_f162 alone (L1/L2 resident)", &r, (REP * NB * 32) as f64);
    std::hint::black_box(&idxs);

    macro_rules! both {
        ($q:expr) => {{
            let f = |idxs: &mut Vec<BinaryIndex32>, souts: &mut Vec<Batch32>| unsafe {
                for b in 0..NB {
                    let c = &*(small.as_ptr().add(128 * b) as *const [F162; 128]);
                    tf::slice_f162_into(c, &mut idxs[b]);
                    va::ntt_bin_batch32::<$q>(&idxs[b], &mut souts[b]);
                }
            };
            f(&mut idxs, &mut souts);
            let r = best(&pg, 3, || {
                for _ in 0..REP / 4 {
                    f(&mut idxs, &mut souts);
                }
            });
            report(
                concat!("transpose + kernel, resident, q = ", stringify!($q)),
                &r,
                (REP / 4 * NB * 32) as f64,
            );
            let g = |idxs: &Vec<BinaryIndex32>, souts: &mut Vec<Batch32>| unsafe {
                for b in 0..NB {
                    va::ntt_bin_batch32::<$q>(&idxs[b], &mut souts[b]);
                }
            };
            g(&idxs, &mut souts);
            let r = best(&pg, 3, || {
                for _ in 0..REP / 4 {
                    g(&idxs, &mut souts);
                }
            });
            report(
                concat!("kernel alone, resident,       q = ", stringify!($q)),
                &r,
                (REP / 4 * NB * 32) as f64,
            );
        }};
    }
    both!(3889);
    both!(9721);

    std::hint::black_box(&souts);
    drop(souts);
    drop(idxs);

    // -------------------------------------------------------------------------- the headline
    println!();
    let elems = random_elems(nf162, 42);
    let mut out_a: Vec<Batch32> = (0..nbatch).map(|_| Batch32::zero(Representation::Ntt)).collect();
    println!(
        "input {:.0} MB, output {:.0} MB per prime",
        nf162 as f64 * 24.0 / 1e6,
        nbatch as f64 * 41472.0 / 1e6
    );

    macro_rules! headline {
        ($q:expr) => {{
            nf::ntt_f162::<$q>(&elems[..128 * 64], &mut out_a[..64]);
            let r = best(&pg, 3, || nf::ntt_f162::<$q>(&elems, &mut out_a));
            report(concat!("ntt_f162, materialised NT, q = ", stringify!($q)), &r, nring as f64);
            let mut rng = Rng::new(99);
            for _ in 0..4 {
                let i = rng.below(nring as u32) as usize;
                let want = scalar::ntt::<$q>(&bin_ntt::f162::lift_elem(&elems, i));
                let got = scalar::normalize_i16(&out_a[i / 32].get(i % 32).v, $q);
                assert_eq!(got, want, "mismatch at ring element {i}");
            }
            let r = best(&pg, 3, || nf::ntt_f162_pf::<$q, 2>(&elems, &mut out_a));
            report(concat!("  + input prefetch,        q = ", stringify!($q)), &r, nring as f64);
        }};
    }
    headline!(3889);
    headline!(9721);

    // ------------------------------------------------------------------ both primes, one slice
    let mut out_b: Vec<Batch32> = (0..nbatch).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let r = best(&pg, 3, || nf::ntt_f162_2q::<3889, 9721>(&elems, &mut out_a, &mut out_b));
    report("ntt_f162_2q, both primes, one slice", &r, nring as f64);
    println!(
        "  (per ring element and prime: {:.1} cyc)",
        r.c.cycles as f64 / (2.0 * nring as f64)
    );
    drop(out_b);

    // ------------------------------------------------------------------------------- streamed
    let mut sink = 0i16;
    let r = best(&pg, 3, || {
        nf::ntt_f162_stream::<3889>(&elems, |_, b| sink ^= b.v[0][0] ^ b.v[647][31])
    });
    report("ntt_f162_stream, trivial consumer, q = 3889", &r, nring as f64);
    std::hint::black_box(sink);

    // ------------------------------------------------------ accumulate y = sum_i a_i o NTT(w_i)
    let mut rng = Rng::new(5);
    let a: Vec<Batch32> = (0..nbatch)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(3889) as i16;
                }
            }
            b
        })
        .collect();
    let mut acc = vec![[[0i32; 16]; N]; 1];
    let r = best(&pg, 3, || {
        acc[0] = [[0i32; 16]; N];
        nf::ntt_f162_stream::<3889>(&elems, |i, w| unsafe {
            accumulate_products::<3889>(w, &a[i], &mut acc[0])
        });
    });
    report("y = sum_i a_i o NTT(w_i), q = 3889", &r, nring as f64);
    let mut acc_small = [[0i32; 16]; N];
    let mut want = [0u64; N];
    let mut nt = Batch32::zero(Representation::Ntt);
    nf::ntt_f162::<3889>(&elems[..128], std::slice::from_mut(&mut nt));
    unsafe { accumulate_products::<3889>(&nt, &a[0], &mut acc_small) };
    for p in 0..32 {
        let w = scalar::ntt::<3889>(&bin_ntt::f162::lift_elem(&elems, p));
        for j in 0..N {
            want[j] = (want[j] + w[j] as u64 * a[0].v[j][p] as u64) % 3889;
        }
    }
    let got = finish_accumulator::<3889>(&acc_small);
    for j in 0..N {
        assert_eq!(got[j] as u64, want[j], "accumulate mismatch at slot {j}");
    }
    println!("  (accumulation verified against the scalar reference on one batch)");

    println!(
        "\nDRAM implied: input {:.0} MB read, {:.0} MB written per prime (NT stores; measured \
         37 GB/s NT, 19.5 GB/s read on this core)",
        nf162 as f64 * 24.0 / 1e6,
        nbatch as f64 * 41472.0 / 1e6
    );
}

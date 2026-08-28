//! Headline-style measurements for everything *outside* the binary NTT kernel: two primes from one
//! transpose, producer-side index rows, prefetch tuning, non-temporal store placement, huge pages
//! for the 340 MB output, and the streamed form with a real consumer.
//!
//! `taskset -c 6 bench_vertical_bin_io [cpu] [--quick]`. `--only <name>` runs one variant once so
//! the run can be wrapped in `perf stat -e dtlb_store_misses.walk_completed,...`.
use bin_ntt::params::N;
use bin_ntt::perf::PerfGroup;
use bin_ntt::rng::Rng;
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::vertical_bin as vb;
use bin_ntt::simd::vertical_bin_io as io;
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


/// Two extra raw PMU counters (`perf.rs` fixes its own six), so the TLB experiment can be read
/// inside the process instead of over the whole program: dtlb_store_misses.walk_completed
/// (event 0x49 umask 0x0e) and dtlb_load_misses.walk_completed (0x08 / 0x0e).
struct Tlb {
    fds: [i32; 2],
}
extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
}
impl Tlb {
    fn new() -> Option<Self> {
        let mut fds = [0i32; 2];
        for (i, cfg) in [0x0e49u64, 0x0e08u64].iter().enumerate() {
            let mut attr = [0u64; 16];
            attr[0] = 4 | ((std::mem::size_of::<[u64; 16]>() as u64) << 32); // type = RAW, size
            attr[1] = *cfg;
            attr[5] = (1 << 5) | (1 << 6); // flags: exclude_kernel | exclude_hv
            let fd = unsafe { syscall(298, attr.as_ptr(), 0i32, -1i32, -1i32, 0u64) };
            if fd < 0 {
                return None;
            }
            fds[i] = fd as i32;
        }
        Some(Tlb { fds })
    }
    fn start(&self) {
        for f in self.fds {
            unsafe {
                ioctl(f, 0x2403, 0u64);
                ioctl(f, 0x2400, 0u64);
            }
        }
    }
    fn stop(&self) -> [u64; 2] {
        let mut v = [0u64; 2];
        for (i, f) in self.fds.iter().enumerate() {
            unsafe {
                ioctl(*f, 0x2401, 0u64);
                let mut b = 0u64;
                read(*f, &mut b as *mut u64 as *mut u8, 8);
                v[i] = b;
            }
        }
        v
    }
}

struct M {
    pg: Option<PerfGroup>,
    reps: usize,
    only: Option<String>,
    n: usize,
}

impl M {
    /// Best of `reps`; `elems` is the number of polynomials the reported per-element figures
    /// divide by (2 * n for the two-prime drivers).
    /// Same as `run` but also reports the two TLB page-walk counters per polynomial.
    fn run_tlb(&self, label: &str, elems: usize, mut f: impl FnMut()) {
        if self.only.is_some() {
            return;
        }
        f();
        let t = Tlb::new();
        let mut best = f64::INFINITY;
        let mut bt = [0u64; 2];
        for _ in 0..self.reps {
            if let Some(x) = &t {
                x.start();
            }
            let t0 = Instant::now();
            f();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let c = t.as_ref().map(|x| x.stop()).unwrap_or([0; 2]);
            if ms < best {
                best = ms;
                bt = c;
            }
        }
        let e = elems as f64;
        println!(
            "{label:<52} {best:8.2} ms   store walks/poly {:7.4}   load walks/poly {:7.4}",
            bt[0] as f64 / e,
            bt[1] as f64 / e
        );
    }

    fn run(&self, label: &str, elems: usize, mut f: impl FnMut()) {
        if let Some(o) = &self.only {
            if !label.contains(o.as_str()) {
                return;
            }
            f();
            println!("{label}: one run");
            return;
        }
        f(); // warm
        let mut best = f64::INFINITY;
        let mut bc = None;
        for _ in 0..self.reps {
            if let Some(p) = &self.pg {
                p.start();
            }
            let t0 = Instant::now();
            f();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let c = self.pg.as_ref().map(|p| p.stop());
            if ms < best {
                best = ms;
                bc = c;
            }
        }
        let e = elems as f64;
        print!("{label:<52} {best:8.2} ms");
        if let Some(c) = bc {
            print!(
                "  {:6.1} cyc {:7.1} ins {:7.1} uops (p0 {:6.1} p5 {:6.1})  {:.2} GHz",
                c.cycles as f64 / e,
                c.instructions as f64 / e,
                c.uops as f64 / e,
                c.port0 as f64 / e,
                c.port5 as f64 / e,
                c.cycles as f64 / (best * 1e6)
            );
        }
        println!();
        let _ = self.n;
    }
}

fn anon_huge() -> String {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("AnonHugePages:"))
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let quick = args.iter().any(|a| a == "--quick");
    let only = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1))
        .cloned();
    pin(cpu);
    let log_n = if quick { 15 } else { 18 };
    let n = 1usize << log_n;
    let nb = n / 32;
    let reps = if quick { 2 } else { 3 };
    let m = M { pg: PerfGroup::new().ok(), reps, only: only.clone(), n };
    println!(
        "bench_vertical_bin_io: cpu {cpu}, 2^{log_n} polynomials ({nb} batches), best of {reps}"
    );

    let mut rng = Rng::new(42);
    let polys: Vec<BinaryPoly> = (0..n).map(|_| BinaryPoly::random(&mut rng)).collect();

    // ------------------------------------------------------------------ single prime, Vec output
    {
        let mut out: Vec<Batch32> = (0..nb).map(|_| Batch32::zero(Representation::Ntt)).collect();
        println!("\n--- baseline driver (Vec<Batch32> output, malloc'd, 4 KB pages) ---");
        m.run("base ntt_bin_polys q=3889", n, || vb::ntt_bin_polys::<3889>(&polys, &mut out));
        m.run("base ntt_bin_polys q=9721", n, || vb::ntt_bin_polys::<9721>(&polys, &mut out));
        println!("\n--- store placement / fence / grouping (q=3889) ---");
        m.run("cached stores (no NT) q=3889", n, || {
            io::ntt_bin_polys_cached::<3889>(&polys, &mut out)
        });
        m.run("NT via separate copy pass q=3889", n, || {
            io::ntt_bin_polys_copy::<3889>(&polys, &mut out)
        });
        m.run("sfence per batch q=3889", n, || {
            io::ntt_bin_polys_fence::<3889>(&polys, &mut out)
        });
        m.run("grouped transpose G=4 q=3889", n, || {
            io::ntt_bin_polys_grouped::<3889, 4>(&polys, &mut out)
        });
        m.run("grouped transpose G=16 q=3889", n, || {
            io::ntt_bin_polys_grouped::<3889, 16>(&polys, &mut out)
        });
        println!("\n--- input prefetch distance (q=3889) ---");
        m.run("prefetch polys d=1 q=3889", n, || io::ntt_bin_polys_pf::<3889, 1>(&polys, &mut out));
        m.run("prefetch polys d=2 q=3889", n, || io::ntt_bin_polys_pf::<3889, 2>(&polys, &mut out));
        m.run("prefetch polys d=4 q=3889", n, || io::ntt_bin_polys_pf::<3889, 4>(&polys, &mut out));
        m.run("prefetch polys d=8 q=3889", n, || io::ntt_bin_polys_pf::<3889, 8>(&polys, &mut out));
    }

    // ------------------------------------------------------------------ where the overhead lives
    println!("\n--- decomposition of the out-of-cache overhead (q=3889) ---");
    {
        let mut out = io::Batches::new(nb, true);
        let mut one = Batch32::zero(Representation::Ntt);
        let small = 32 * 1024.min(nb); // 32k polynomials: input fits L2/L3
        m.run("big in, big out (the driver)", n, || vb::ntt_bin_polys::<3889>(&polys, &mut out));
        m.run("small in, big out (input from cache)", n, || {
            for c in out.chunks_mut(small / 32) {
                vb::ntt_bin_polys::<3889>(&polys[..32 * c.len()], c);
            }
        });
        m.run("big in, one out (NT traffic, no walk)", n, || {
            io::ntt_bin_polys_1out::<3889>(&polys, &mut one)
        });
        m.run("small in, one out (~cache resident)", n, || {
            for _ in 0..(nb / (small / 32)) {
                io::ntt_bin_polys_1out::<3889>(&polys[..small], &mut one);
            }
        });
        std::hint::black_box(&one);
        println!("\n--- prefetch placement: head of loop vs between transpose and kernel ---");
        m.run("pf polys at head  d=2 (T0)", n, || io::ntt_bin_polys_pf::<3889, 2>(&polys, &mut out));
        m.run("pf polys mid      d=1 (T0)", n, || {
            io::ntt_bin_polys_pfmid::<3889, 1, 3>(&polys, &mut out)
        });
        m.run("pf polys mid      d=2 (T0)", n, || {
            io::ntt_bin_polys_pfmid::<3889, 2, 3>(&polys, &mut out)
        });
        m.run("pf polys mid      d=4 (T0)", n, || {
            io::ntt_bin_polys_pfmid::<3889, 4, 3>(&polys, &mut out)
        });
        m.run("pf polys mid      d=2 (T1)", n, || {
            io::ntt_bin_polys_pfmid::<3889, 2, 2>(&polys, &mut out)
        });
        m.run("pf polys mid      d=2 (NTA)", n, || {
            io::ntt_bin_polys_pfmid::<3889, 2, 0>(&polys, &mut out)
        });
        m.run("pf polys mid      d=2 (T0) q=9721", n, || {
            io::ntt_bin_polys_pfmid::<9721, 2, 3>(&polys, &mut out)
        });
    }

    // ------------------------------------------------------------------ huge pages for the output
    println!("\n--- output buffer: mmap 4 KB vs MADV_HUGEPAGE ---");
    for huge in [false, true] {
        let tag = if huge { "huge" } else { "4KB " };
        let mut out = io::Batches::new(nb, huge);
        println!("  ({tag}) {}", anon_huge());
        m.run(&format!("mmap {tag} q=3889"), n, || vb::ntt_bin_polys::<3889>(&polys, &mut out));
        m.run_tlb(&format!("mmap {tag} q=3889 [TLB]"), n, || {
            vb::ntt_bin_polys::<3889>(&polys, &mut out)
        });
        m.run(&format!("mmap {tag} q=9721"), n, || vb::ntt_bin_polys::<9721>(&polys, &mut out));
        m.run(&format!("mmap {tag} + pf d=2 q=3889"), n, || {
            io::ntt_bin_polys_pf::<3889, 2>(&polys, &mut out)
        });
        m.run(&format!("mmap {tag} + pf d=2 q=9721"), n, || {
            io::ntt_bin_polys_pf::<9721, 2>(&polys, &mut out)
        });
    }

    // ------------------------------------------------------------------ both primes, one transpose
    println!("\n--- both primes (ms/cycles are for 2 x 2^{log_n} transforms) ---");
    {
        let mut a = io::Batches::new(nb, true);
        let mut b = io::Batches::new(nb, true);
        m.run("2 x ntt_bin_polys (separate transposes)", 2 * n, || {
            vb::ntt_bin_polys::<3889>(&polys, &mut a);
            vb::ntt_bin_polys::<9721>(&polys, &mut b);
        });
        m.run("ntt_bin_polys_2q (one transpose)", 2 * n, || {
            io::ntt_bin_polys_2q::<3889, 9721>(&polys, &mut a, &mut b)
        });
        m.run("ntt_bin_polys_2q_pf d=2", 2 * n, || {
            io::ntt_bin_polys_2q_pf::<3889, 9721, 2>(&polys, &mut a, &mut b)
        });
    }

    // ------------------------------------------------------------------ producer-side index rows
    println!("\n--- producer-side input forms ---");
    {
        let mut idx = io::Indices::new(nb, true);
        let mut nib: Vec<BinaryBatch32> = (0..nb).map(|_| BinaryBatch32::zero()).collect();
        io::transpose_polys(&polys, &mut idx);
        io::nibble_polys(&polys, &mut nib);
        let mut out = io::Batches::new(nb, true);
        println!("  index rows {} MB, nibbles {} MB, polys {} MB",
            nb * std::mem::size_of::<BinaryIndex32>() / (1 << 20),
            nb * std::mem::size_of::<BinaryBatch32>() / (1 << 20),
            n * std::mem::size_of::<BinaryPoly>() / (1 << 20));
        m.run("idx rows in  q=3889", n, || io::ntt_bin_idx::<3889>(&idx, &mut out));
        m.run("idx rows in  q=9721", n, || io::ntt_bin_idx::<9721>(&idx, &mut out));
        m.run("idx rows + pf 162 d=1 q=3889", n, || {
            io::ntt_bin_idx_pf::<3889, 1, 162>(&idx, &mut out)
        });
        m.run("idx rows + pf 64 d=1 q=3889", n, || {
            io::ntt_bin_idx_pf::<3889, 1, 64>(&idx, &mut out)
        });
        m.run("idx rows + pf 162 d=2 q=3889", n, || {
            io::ntt_bin_idx_pf::<3889, 2, 162>(&idx, &mut out)
        });
        m.run("idx rows + pf 162 d=1 (T1) q=3889", n, || {
            io::ntt_bin_idx_pfh::<3889, 1, 162, 2>(&idx, &mut out)
        });
        m.run("idx rows + pf 162 d=1 (NTA) q=3889", n, || {
            io::ntt_bin_idx_pfh::<3889, 1, 162, 0>(&idx, &mut out)
        });
        m.run("nibbles in   q=3889", n, || io::ntt_bin_nib::<3889>(&nib, &mut out));
        m.run("nibbles in   q=9721", n, || io::ntt_bin_nib::<9721>(&nib, &mut out));
        let mut out2 = io::Batches::new(nb, true);
        m.run("idx rows 2q (2 x 2^18)", 2 * n, || {
            io::ntt_bin_idx_2q::<3889, 9721>(&idx, &mut out, &mut out2)
        });
        m.run("nibbles 2q  (2 x 2^18)", 2 * n, || {
            io::ntt_bin_nib_2q::<3889, 9721>(&nib, &mut out, &mut out2)
        });
    }

    // ------------------------------------------------------------------ pipelined transpose
    println!("\n--- transpose software-pipelined into the kernel ---");
    {
        let mut out = io::Batches::new(nb, true);
        m.run("base   ntt_bin_polys        q=3889", n, || vb::ntt_bin_polys::<3889>(&polys, &mut out));
        m.run("copy   pipelined_off        q=3889", n, || {
            io::ntt_bin_polys_pipelined_off::<3889>(&polys, &mut out)
        });
        m.run("PIPE   pipelined            q=3889", n, || {
            io::ntt_bin_polys_pipelined::<3889>(&polys, &mut out)
        });
        m.run("P34    pipelined P3+P4 only  q=3889", n, || {
            io::ntt_bin_polys_pipelined_p34::<3889>(&polys, &mut out)
        });
        m.run("base   ntt_bin_polys        q=9721", n, || vb::ntt_bin_polys::<9721>(&polys, &mut out));
        m.run("copy   pipelined_off        q=9721", n, || {
            io::ntt_bin_polys_pipelined_off::<9721>(&polys, &mut out)
        });
        m.run("PIPE   pipelined            q=9721", n, || {
            io::ntt_bin_polys_pipelined::<9721>(&polys, &mut out)
        });
        // cache-resident: 8 batches (input 90 KB, output 332 KB) repeated
        const R: usize = 400;
        let sm = 8;
        m.run("P34    pipelined P3+P4 only  q=9721", n, || {
            io::ntt_bin_polys_pipelined_p34::<9721>(&polys, &mut out)
        });
        m.run("resident base      (8 batches x400)", 32 * sm * R, || {
            for _ in 0..R {
                vb::ntt_bin_polys::<3889>(&polys[..32 * sm], &mut out[..sm]);
            }
        });
        m.run("resident PIPE      (8 batches x400)", 32 * sm * R, || {
            for _ in 0..R {
                io::ntt_bin_polys_pipelined::<3889>(&polys[..32 * sm], &mut out[..sm]);
            }
        });
        m.run("resident P34       (8 batches x400)", 32 * sm * R, || {
            for _ in 0..R {
                io::ntt_bin_polys_pipelined_p34::<3889>(&polys[..32 * sm], &mut out[..sm]);
            }
        });
        m.run("resident base 9721 (8 batches x400)", 32 * sm * R, || {
            for _ in 0..R {
                vb::ntt_bin_polys::<9721>(&polys[..32 * sm], &mut out[..sm]);
            }
        });
        m.run("resident PIPE 9721 (8 batches x400)", 32 * sm * R, || {
            for _ in 0..R {
                io::ntt_bin_polys_pipelined::<9721>(&polys[..32 * sm], &mut out[..sm]);
            }
        });
    }

    // ------------------------------------------------------------------ streamed + real consumer
    println!("\n--- streamed with an accumulating consumer (340 MB of operands read) ---");
    {
        let mut a = io::Batches::new(nb, true);
        for b in a.iter_mut() {
            for j in 0..N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(3889) as i16;
                }
            }
        }
        let mut acc = vec![[[0i32; 16]; N]; 1];
        m.run("stream+accumulate, no prefetch q=3889", n, || {
            acc[0] = [[0i32; 16]; N];
            io::ntt_bin_accumulate::<3889, 0>(&polys, &a, &mut acc[0])
        });
        m.run("stream+accumulate, pf 1 batch  q=3889", n, || {
            acc[0] = [[0i32; 16]; N];
            io::ntt_bin_accumulate::<3889, 1>(&polys, &a, &mut acc[0])
        });
        m.run("stream+accumulate, pf 2 batch  q=3889", n, || {
            acc[0] = [[0i32; 16]; N];
            io::ntt_bin_accumulate::<3889, 2>(&polys, &a, &mut acc[0])
        });
        m.run("stream, trivial consumer       q=3889", n, || {
            let mut s = 0i16;
            vb::ntt_bin_stream::<3889>(&polys, |_, b| s ^= b.v[0][0] ^ b.v[647][31]);
            std::hint::black_box(s);
        });
        std::hint::black_box(&acc);
    }
}

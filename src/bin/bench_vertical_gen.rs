//! Benchmark for `simd::vertical_gen` (generic-input vertical NTT). Pin with `taskset -c 4`.
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::vertical_gen::{
    ntt_gen_batch32, ntt_gen_batch32_plan, ntt_gen_batch32_r27, ntt_gen_batches, Tw,
};
use bin_ntt::types::{Batch32, Representation};
use std::time::Instant;

fn fill<const Q: u16>(b: &mut Batch32, rng: &mut Rng) {
    for j in 0..bin_ntt::N {
        for p in 0..32 {
            b.v[j][p] = (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16;
        }
    }
    b.representation = Representation::Coefficients;
}

fn report(name: &str, c: Counts, polys: f64) {
    println!(
        "{name:<32} {:6.1} cyc {:7.1} ins {:7.1} uops  p0 {:6.1}  p1 {:5.1}  p5 {:6.1}  IPC {:.2}",
        c.cycles as f64 / polys,
        c.instructions as f64 / polys,
        c.uops as f64 / polys,
        c.port0 as f64 / polys,
        c.port1 as f64 / polys,
        c.port5 as f64 / polys,
        c.instructions as f64 / c.cycles as f64,
    );
}

fn make<const Q: u16>(n: usize, rng: &mut Rng) -> Vec<Batch32> {
    (0..n)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Coefficients);
            fill::<Q>(&mut b, rng);
            b
        })
        .collect()
}

fn survey<const Q: u16>() {
    let mut rng = Rng::new(1);
    let mut set = make::<Q>(8, &mut rng);
    let reps = 400;
    let names = [
        "A|B(2+3)|C(4+5)|D(6)",
        "A|B(2+3)|4|5|6",
        "A|2|3|C(4+5)|D(6)",
        "A|2|3|4|5|6",
        "A|B(2+3)|4|56",
        "A|2|3|4|56",
        "A|B(2+3)|r27(4+5+6)",
        "shipped ntt_gen_batch32",
    ];
    for v in 0..8u32 {
        for _ in 0..30 {
            for b in set.iter_mut() {
                run_one::<Q>(v, b);
            }
        }
        let g = PerfGroup::new().expect("perf_event_open (perf_event_paranoid <= 1?)");
        g.start();
        for _ in 0..reps {
            for b in set.iter_mut() {
                run_one::<Q>(v, b);
            }
        }
        let c = g.stop();
        report(&format!("q={Q} L2 {}", names[v as usize]), c, (reps * 8 * 32) as f64);
    }
}

#[inline(always)]
fn run_one<const Q: u16>(v: u32, b: &mut Batch32) {
    unsafe {
        match v {
            0 => ntt_gen_batch32_plan::<Q, 0>(b),
            1 => ntt_gen_batch32_plan::<Q, 1>(b),
            2 => ntt_gen_batch32_plan::<Q, 2>(b),
            3 => ntt_gen_batch32_plan::<Q, 3>(b),
            4 => ntt_gen_batch32_plan::<Q, 4>(b),
            5 => ntt_gen_batch32_plan::<Q, 5>(b),
            6 => ntt_gen_batch32_r27::<Q>(b),
            _ => ntt_gen_batch32::<Q>(b),
        }
    }
}

fn dram<const Q: u16>() {
    let mut rng = Rng::new(2);
    let batches = 8192;
    let mut big: Vec<Batch32> = vec![Batch32::zero(Representation::Coefficients); batches];
    for b in big.iter_mut().take(64) {
        fill::<Q>(b, &mut rng);
    }
    let g = PerfGroup::new().unwrap();
    let bytes = (batches * std::mem::size_of::<Batch32>() * 2) as f64;
    for pf in [false, true, false, true] {
        let t0 = Instant::now();
        g.start();
        if pf {
            ntt_gen_batches::<Q>(&mut big);
        } else {
            for b in big.iter_mut() {
                unsafe { ntt_gen_batch32::<Q>(b) };
            }
        }
        let c = g.stop();
        let secs = t0.elapsed().as_secs_f64();
        let tag = if pf { "prefetch" } else { "plain   " };
        report(&format!("q={Q} 2^18 polys {tag}"), c, (batches * 32) as f64);
        println!(
            "{:<32} {:6.2} ms   {:.1} GB/s (r+w)   {:.2} Mpoly/s",
            "",
            secs * 1e3,
            bytes / secs / 1e9,
            (batches * 32) as f64 / secs / 1e6
        );
    }
    // memory floor: the same in-place read+write with no arithmetic
    for _ in 0..2 {
        let t0 = Instant::now();
        for b in big.iter_mut() {
            unsafe {
                let p = b.v.as_mut_ptr() as *mut std::arch::x86_64::__m512i;
                for j in 0..bin_ntt::N {
                    let x = std::arch::x86_64::_mm512_load_si512(p.add(j));
                    std::arch::x86_64::_mm512_store_si512(
                        p.add(j),
                        std::arch::x86_64::_mm512_add_epi16(x, x),
                    );
                }
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "{:<32} {:6.2} ms   {:.1} GB/s (r+w)   in-place copy floor",
            format!("q={Q} 2^18 polys memory floor"),
            secs * 1e3,
            bytes / secs / 1e9
        );
    }
}

/// Static instruction counts of the hot loop bodies, read off
/// `objdump -d --no-show-raw-insn` of `ntt_gen_batch32` (pass A, C4, C5, D are inlined; pass B is
/// out of line): (insns, p0-only multiplies, adds/subs, trips per batch).
fn statics<const Q: u16>() {
    let rows: [(&str, u32, u32, u32, u32); 5] = if Q == 9721 {
        [
            ("A  levels 0+1", 40, 14, 15, 162),
            ("B  levels 2+3", 76, 31, 33, 108),
            ("C4 level 4", 86, 33, 33, 72),
            ("C5 level 5", 92, 33, 33, 72),
            ("D  level 6", 36, 11, 11, 216),
        ]
    } else {
        [
            ("A  levels 0+1", 37, 12, 14, 162),
            ("B  levels 2+3", 70, 27, 29, 108),
            ("C4 level 4", 77, 27, 30, 72),
            ("C5 level 5", 92, 33, 33, 72),
            ("D  level 6", 33, 9, 10, 216),
        ]
    };
    let (mut i, mut m, mut a) = (0, 0, 0);
    for (n, ins, mul, add, trips) in rows {
        println!(
            "  static q={Q} {n:<14} {ins:3} insns x {trips:3} = {:6}   p0 {:6}   add/sub {:6}",
            ins * trips,
            mul * trips,
            add * trips
        );
        i += ins * trips;
        m += mul * trips;
        a += add * trips;
    }
    println!(
        "  static q={Q} total/batch    {i:6} insns   p0-only {m:6}   add/sub {a:6}  (= {:.1} + {:.1} + {:.1} per poly)",
        i as f64 / 32.0,
        m as f64 / 32.0,
        a as f64 / 32.0
    );
    println!(
        "  static q={Q} floors: p0 {:.1} cyc/poly, balanced (p0+p5)/2 {:.1} cyc/poly",
        m as f64 / 32.0,
        (m + a) as f64 / 64.0
    );
}

fn main() {
    println!(
        "output bounds: q=3889 |v| <= {}, q=9721 |v| <= {}",
        Tw::<3889>::OUTPUT_BOUND,
        Tw::<9721>::OUTPUT_BOUND
    );
    statics::<3889>();
    statics::<9721>();
    survey::<3889>();
    survey::<9721>();
    dram::<3889>();
    dram::<9721>();
}

//! Bench for the horizontal generic-input NTT. Run pinned: `taskset -c 6 ./bench_horizontal_gen`.
//!
//! The kernel is in place and the numbers are data independent, so the repeated passes over a
//! working set keep NTT-ing their own output (garbage after the first pass, same instruction
//! stream). A correctness check on a fresh batch runs first.
use bin_ntt::params::*;
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::scalar;
use bin_ntt::simd::horizontal_gen::*;
use bin_ntt::types::*;

fn random_batch<const Q: u16>(rng: &mut Rng) -> HBatch4 {
    let es: [RingElement; 4] = std::array::from_fn(|_| RingElement {
        v: std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16),
        representation: Representation::Coefficients,
    });
    HBatch4::from_elements(&es)
}

fn check<const Q: u16>() {
    let mut rng = Rng::new(1);
    let mut b = random_batch::<Q>(&mut rng);
    let inputs: [RingElement; 4] =
        std::array::from_fn(|p| b.get_as(p, Representation::Coefficients));
    unsafe { ntt_gen_hbatch4::<Q>(&mut b) };
    for (p, input) in inputs.iter().enumerate() {
        let want = scalar::ntt::<Q>(&scalar::normalize_i16(&input.v, Q));
        assert_eq!(scalar::normalize_i16(&b.get(p).v, Q), want);
    }
}

fn report(label: &str, c: Counts, polys: f64) {
    println!(
        "{label:<28} {:8.1} cyc  {:8.1} instr  {:8.1} uops  p0 {:7.1}  p1 {:7.1}  p5 {:7.1}  \
         (IPC {:.2}, uops/cyc {:.2})",
        c.cycles as f64 / polys,
        c.instructions as f64 / polys,
        c.uops as f64 / polys,
        c.port0 as f64 / polys,
        c.port1 as f64 / polys,
        c.port5 as f64 / polys,
        c.instructions as f64 / c.cycles as f64,
        c.uops as f64 / c.cycles as f64,
    );
}

fn run<const Q: u16>(g: &PerfGroup, label: &str, bs: &mut [HBatch4], reps: usize) {
    unsafe { ntt_gen_hbatch4_many::<Q>(bs) };
    unsafe { ntt_gen_hbatch4_many::<Q>(bs) };
    g.start();
    for _ in 0..reps {
        unsafe { ntt_gen_hbatch4_many::<Q>(bs) };
    }
    let c = g.stop();
    report(label, c, (reps * bs.len() * HBatch4::POLYS) as f64);
}

fn bench<const Q: u16>(g: &PerfGroup, what: &str) {
    let mut rng = Rng::new(Q as u64);
    println!("--- q = {Q} (barrett per radix-3 level: {}) ---", uses_barrett::<Q>());

    if what == "all" || what == "l1" {
        let mut l1: Vec<HBatch4> = (0..4).map(|_| random_batch::<Q>(&mut rng)).collect();
        run::<Q>(g, "16 polys (L1, 20 KB)", &mut l1, 4096);
    }
    if what == "all" || what == "l2" {
        let mut l2: Vec<HBatch4> = (0..64).map(|_| random_batch::<Q>(&mut rng)).collect();
        run::<Q>(g, "256 polys (L2, 331 KB)", &mut l2, 256);
    }
    if what == "all" || what == "big" {
        let mut big: Vec<HBatch4> = (0..65536).map(|_| HBatch4::zero()).collect();
        for (i, b) in big.iter_mut().enumerate() {
            if i < 64 {
                *b = random_batch::<Q>(&mut rng);
            }
        }
        run::<Q>(g, "2^18 polys (340 MB)", &mut big, 2);
    }
}

fn main() {
    check::<3889>();
    check::<9721>();
    println!("psi(3889) = {}, psi(9721) = {}", Params::<3889>::PSI, Params::<9721>::PSI);
    for q in QS {
        let (b, o) = if q == 3889 {
            (level_bounds::<3889>(), out_bound::<3889>())
        } else {
            (level_bounds::<9721>(), out_bound::<9721>())
        };
        let lv: Vec<String> =
            (1..8).map(|l| format!("{:.2}q", b[l] as f64 / BSCALE as f64)).collect();
        println!("q = {q}: per-level bounds {lv:?}, output |v| <= {o}");
    }
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let g = PerfGroup::new().expect("perf_event_open (need perf_event_paranoid <= 2)");
    bench::<3889>(&g, &what);
    bench::<9721>(&g, &what);
}


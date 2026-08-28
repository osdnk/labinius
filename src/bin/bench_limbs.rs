//! The five limb primes side by side: the cache-resident transform, the whole 2^18-`F162`
//! commitment against 85 MB of `A`, and the base multiplication alone that sits inside it.
//! Pin with `taskset -c 2`.
//!
//! The commitment is one limb of a key: front end, transform, and the multiply-accumulate the
//! block sink performs — `mac27` on 27-slot blocks for a splitting prime, `mac_quad18` on 18-row
//! blocks (9 quadratic leaves, the Karatsuba or schoolbook `P_2`) for a quadratic-slot one.
use bin_fields::scalar::F162;
use bin_ntt::f162::{self, RandomF162};
use bin_ntt::perf::{Counts, PerfGroup};
use bin_ntt::rng::Rng;
use bin_ntt::simd::commit::{self as cm, Acc, QuadAcc};
use bin_ntt::simd::transpose::BinaryIndex32;
use bin_ntt::simd::transpose_f162 as tf;
use bin_ntt::simd::{vertical_bin_asm as va, vertical_bin_quad as vq};
use bin_ntt::types::{Batch32, Representation};
use std::time::Instant;

const BIG: usize = 1 << 18; // F162 = 2^16 ring elements
const BIGB: usize = BIG / 128;
const NB: usize = 8; // cache-resident batches
const REP: usize = 100;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

fn best(pg: &PerfGroup, rounds: usize, mut f: impl FnMut()) -> (Counts, f64) {
    let mut b: Option<(Counts, f64)> = None;
    for _ in 0..rounds {
        let t0 = Instant::now();
        pg.start();
        f();
        let c = pg.stop();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if b.as_ref().map_or(true, |x| c.cycles < x.0.cycles) {
            b = Some((c, ms));
        }
    }
    b.unwrap()
}

/// 85 MB of uniform centered `A` for one prime.
fn key(q: u16, nb: usize, seed: u64) -> Vec<Batch32> {
    let mut rng = Rng::new(seed);
    let half = ((q - 1) / 2) as i16;
    (0..nb)
        .map(|_| {
            let mut b = Batch32::zero(Representation::Ntt);
            for j in 0..bin_ntt::N {
                for p in 0..32 {
                    b.v[j][p] = rng.below(q as u32) as i16 - half;
                }
            }
            b
        })
        .collect()
}

struct Row {
    q: u16,
    tree: &'static str,
    transform: f64,
    basemul: f64,
    commit_ms: f64,
    commit_cyc: f64,
    period: (usize, usize),
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    pin(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2));
    f162::assert_layout();
    let pg = PerfGroup::new().expect("perf_event_open failed (perf_event_paranoid?)");

    let mut rng = Rng::new(0xF162);
    let small: Vec<F162> = (0..128 * NB).map(|_| F162::random(&mut rng)).collect();
    let mut idx: Vec<BinaryIndex32> = (0..NB).map(|_| BinaryIndex32::zero()).collect();
    let mut out: Vec<Batch32> = (0..NB)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    let elems: Vec<F162> = f162::random_elems(BIG, 0x5EED);

    let w = Batch32::zero(Representation::Ntt);
    let a1 = Batch32::zero(Representation::Ntt);
    let (wp, ap) = (w.v.as_ptr() as *const i16, a1.v.as_ptr() as *const i16);
    let mut acc = Acc::zero();
    let mut qacc = QuadAcc::zero();
    let accp = acc.v.as_mut_ptr() as *mut i32;
    let (q01, q2) = (
        qacc.p01.as_mut_ptr() as *mut i32,
        qacc.p2.as_mut_ptr() as *mut i32,
    );
    const MREP: usize = 2048;

    macro_rules! transform {
        ($tf:expr) => {{
            let (c, _) = best(&pg, 3, || unsafe {
                for _ in 0..REP {
                    for b in 0..NB {
                        let ch = &*(small.as_ptr().add(128 * b) as *const [F162; 128]);
                        tf::slice_f162_into(ch, &mut idx[b]);
                        $tf(&idx[b], &mut out[b]);
                    }
                }
            });
            std::hint::black_box(&out);
            c.cycles as f64 / (REP * NB * 32) as f64
        }};
    }

    macro_rules! row {
        ($q:literal, $quad:literal, $tf:expr, $commit:expr, $mac:expr) => {{
            let transform = transform!($tf);
            let (mc, _) = best(&pg, 3, || unsafe {
                for _ in 0..MREP {
                    $mac
                }
            });
            let basemul = mc.cycles as f64 / (MREP * 32) as f64;
            let a = key($q, BIGB, 0xA11CE ^ $q as u64);
            let (cc, ms) = best(&pg, 3, || {
                std::hint::black_box($commit(&elems, &a));
            });
            drop(a);
            Row {
                q: $q,
                tree: if $quad { "quadratic (324 x 2)" } else { "split (648)" },
                transform,
                basemul,
                commit_ms: ms,
                commit_cyc: cc.cycles as f64 / (BIG / 4) as f64,
                period: if $quad {
                    (cm::red_period_quad01($q), cm::red_period_quad2($q))
                } else {
                    (cm::red_period($q), cm::red_period($q))
                },
            }
        }};
    }

    let rows = vec![
        row!(3889, false, |i: &BinaryIndex32, o: &mut Batch32| va::ntt_bin_batch32::<3889>(i, o), cm::commit::<3889>, {
            for k in 0..24 {
                cm::mac27::<false>(
                    std::hint::black_box(wp),
                    std::hint::black_box(ap).add(32 * 27 * k),
                    ap as *const i8,
                    accp.add(16 * cm::ACC_PER_BLK * k),
                );
            }
        }),
        row!(9721, false, |i: &BinaryIndex32, o: &mut Batch32| va::ntt_bin_batch32::<9721>(i, o), cm::commit::<9721>, {
            for k in 0..24 {
                cm::mac27::<false>(
                    std::hint::black_box(wp),
                    std::hint::black_box(ap).add(32 * 27 * k),
                    ap as *const i8,
                    accp.add(16 * cm::ACC_PER_BLK * k),
                );
            }
        }),
        row!(2917, true, |i: &BinaryIndex32, o: &mut Batch32| vq::ntt_quad_bin_batch32::<2917>(i, o), cm::commit_quad::<2917>, {
            cm::mac_quad_batch::<2917, false>(
                std::hint::black_box(wp),
                std::hint::black_box(ap),
                ap as *const i8,
                q01,
                q2,
            );
        }),
        row!(4861, true, |i: &BinaryIndex32, o: &mut Batch32| vq::ntt_quad_bin_batch32::<4861>(i, o), cm::commit_quad::<4861>, {
            cm::mac_quad_batch::<4861, false>(
                std::hint::black_box(wp),
                std::hint::black_box(ap),
                ap as *const i8,
                q01,
                q2,
            );
        }),
        row!(12637, true, |i: &BinaryIndex32, o: &mut Batch32| vq::ntt_quad_bin_batch32::<12637>(i, o), cm::commit_quad::<12637>, {
            cm::mac_quad_batch::<12637, false>(
                std::hint::black_box(wp),
                std::hint::black_box(ap),
                ap as *const i8,
                q01,
                q2,
            );
        }),
    ];
    std::hint::black_box((&acc, &qacc));

    println!(
        "one limb of a commitment, 2^18 F162 = {} ring elements, 85 MB of A, one core\n",
        BIG / 4
    );
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| a.commit_cyc.partial_cmp(&b.commit_cyc).unwrap());
    println!(
        "| q | tree | transform | commitment ms | cyc/element | of which basemul | fold-back |"
    );
    println!("|---|------|----------:|--------------:|------------:|-----------------:|----------:|");
    for r in sorted {
        println!(
            "| {} | {} | {:.0} | {:.1} | {:.0} | {:.0} | {} |",
            r.q,
            r.tree,
            r.transform,
            r.commit_ms,
            r.commit_cyc,
            r.basemul,
            if r.period.0 == r.period.1 {
                format!("{}", r.period.0)
            } else {
                format!("{} / {}", r.period.0, r.period.1)
            }
        );
    }
    println!(
        "\ntransform = front end + binary kernel, cache-resident, cycles per ring element;\n\
         basemul = the block sink's multiply-accumulate alone, W in L1 and A in L2;\n\
         fold-back = batches between two accumulator fold-backs (P_0|P_1 / P_2 for a quadratic limb)."
    );

    println!("\nbounds per quadratic limb (|W| is the kernel's declared output bound):");
    for q in [2917u16, 4861, 12637] {
        println!(
            "  q = {q}: |W| <= {} = {:.2} q, karatsuba {}, per batch P01 {} P2 {}, periods {} / {}",
            cm::w_bound_quad(q),
            cm::w_bound_quad(q) as f64 / q as f64,
            cm::karatsuba(q),
            cm::acc_per_batch_quad01(q),
            cm::acc_per_batch_quad2(q),
            cm::red_period_quad01(q),
            cm::red_period_quad2(q),
        );
    }
    for q in [3889u16, 9721] {
        println!(
            "  q = {q}: |W| <= {} = {:.2} q, per batch {}, period {}",
            cm::w_bound(q),
            cm::w_bound(q) as f64 / q as f64,
            cm::acc_per_batch(q),
            cm::red_period(q),
        );
    }
}

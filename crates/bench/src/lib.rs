use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub use labinius::rng::Rng;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

static PINNED: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Pin the thread to `cpu`, or to `$BENCH_CPU` when set; a pin that fails aborts the bench
/// rather than let it run wherever the scheduler puts it.
pub fn pin(cpu: usize) {
    let cpu = std::env::var("BENCH_CPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cpu);
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    let rc = unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
    assert!(rc == 0, "cannot pin to cpu {cpu}; set BENCH_CPU to a cpu of this allocation");
    PINNED.store(cpu, Ordering::Relaxed);
}

/// The cpu [`pin`] pinned to.
pub fn pinned() -> usize {
    PINNED.load(Ordering::Relaxed)
}

pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// One measurement of a step that may only be run once.
pub fn once<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t0 = Instant::now();
    let value = std::hint::black_box(f());
    (ms(t0), value)
}

/// How many times every timed step runs; each reported number is the median of its samples.
pub const REPS: usize = 10;

/// A timing, or a struct of them, reduced sample-wise to its median.
pub trait Medians: Sized {
    fn medians(samples: Vec<Self>) -> Self;
}

impl Medians for f64 {
    fn medians(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }
}

impl Medians for Duration {
    fn medians(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }
}

impl<const N: usize> Medians for [f64; N] {
    fn medians(samples: Vec<[f64; N]>) -> [f64; N] {
        std::array::from_fn(|i| f64::medians(samples.iter().map(|s| s[i]).collect()))
    }
}

impl<A: Medians + Clone, B: Medians + Clone> Medians for (A, B) {
    fn medians(samples: Vec<(A, B)>) -> (A, B) {
        (
            A::medians(samples.iter().map(|s| s.0.clone()).collect()),
            B::medians(samples.iter().map(|s| s.1.clone()).collect()),
        )
    }
}

/// `impl Medians` for a struct, field by field; every field must be listed.
#[macro_export]
macro_rules! medians_by_field {
    ($t:ty { $($f:ident),* $(,)? }) => {
        impl $crate::Medians for $t {
            fn medians(samples: Vec<Self>) -> Self {
                Self { $($f: $crate::Medians::medians(samples.iter().map(|s| s.$f).collect())),* }
            }
        }
    };
}

medians_by_field!(labinius::OpeningTimings {
    fold, encoding, witness, t_r, masks, phi, statement, labrador
});
medians_by_field!(labinius::VerifyTimings {
    rebuild, layout, bound, phi, statement, labrador
});

/// `reps` runs of `step`, each on a fresh `input`, which is not timed: the medians of what the
/// steps measured, and the last value produced.
pub fn medians<I, S: Medians, V>(
    reps: usize,
    mut input: impl FnMut() -> I,
    mut step: impl FnMut(I) -> (S, V),
) -> (S, V) {
    let mut samples = Vec::with_capacity(reps);
    let mut out = None;
    for _ in 0..reps {
        drop(out.take());
        let (sample, value) = step(input());
        samples.push(sample);
        out = Some(value);
    }
    (S::medians(samples), out.unwrap())
}

/// Median of `reps` wall milliseconds of `f`, and the last value produced.
pub fn median_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    medians(reps, || (), |()| once(&mut f))
}

pub fn row(name: &str, milliseconds: f64) {
    println!("  {name:<28}{milliseconds:>9.2} ms");
}

/// One column per mode, `None` where that mode has no such stage.
pub fn table_row(name: &str, values: impl AsRef<[Option<f64>]>, unit: &str) {
    print!("  {name:<32}");
    for value in values.as_ref() {
        match value {
            Some(v) => print!("{v:>13.2}"),
            None => print!("{:>13}", "—"),
        }
    }
    println!(" {unit}");
}

/// Peak resident set size in MB, from `/proc/self/status`.
pub fn peak_rss() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
        / 1024.0
}

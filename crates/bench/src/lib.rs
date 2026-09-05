use std::time::{Duration, Instant};

pub use bin_ntt::rng::Rng;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

pub fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
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

/// Median of `reps` wall milliseconds, and the last value produced. `reps` is odd, so the median
/// is a measured sample rather than an average of two.
pub fn median_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut samples = Vec::with_capacity(reps);
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let value = std::hint::black_box(f());
        samples.push(ms(t0));
        out = Some(value);
    }
    samples.sort_by(f64::total_cmp);
    (samples[reps / 2], out.unwrap())
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

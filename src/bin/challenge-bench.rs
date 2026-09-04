use bin_ntt::api::N162;
use bin_ntt::challenge::{
    op_norm_sq_f32, op_norm_sq_f64, sample_attempt, sample_short_challenge,
    sample_short_challenge_op_norm_prec, ShortChallenge, Transcript, DEFAULT_BOUND,
    DEFAULT_OP_NORM_BOUND, DEFAULT_WEIGHT,
};
use std::time::Instant;

const ROUND: usize = 256;
const REPS: usize = 9;
const CPU: usize = 3;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

fn best(mut f: impl FnMut() -> f64) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..REPS {
        let t = f();
        if t < b {
            b = t;
        }
    }
    b
}

const POWER_STEPS: usize = 2000;

fn dense_matrix(c: &ShortChallenge) -> Vec<f64> {
    let co = c.coeffs();
    let mut m = vec![0.0f64; N162 * N162];
    for j in 0..N162 {
        let mut t = vec![0.0f64; 2 * N162];
        for p in 0..N162 {
            t[p + j] += co[p] as f64;
        }
        for d in (N162..2 * N162).rev() {
            let v = t[d];
            if v != 0.0 {
                t[d] = 0.0;
                t[d - 81] -= v;
                t[d - N162] -= v;
            }
        }
        for r in 0..N162 {
            m[r * N162 + j] = t[r];
        }
    }
    m
}

fn dense_op_norm(c: &ShortChallenge) -> f64 {
    let m = dense_matrix(c);
    let mut a = vec![0.0f64; N162 * N162];
    for i in 0..N162 {
        for j in 0..N162 {
            let mut s = 0.0f64;
            for r in 0..N162 {
                s += m[r * N162 + i] * m[r * N162 + j];
            }
            a[i * N162 + j] = s;
        }
    }
    let mut x = vec![0.0f64; N162];
    for i in 0..N162 {
        x[i] = if i % 3 == 0 {
            1.0
        } else {
            -0.6 + 0.1 * i as f64
        };
    }
    let mut lam = 0.0f64;
    let mut y = vec![0.0f64; N162];
    for _ in 0..POWER_STEPS {
        for i in 0..N162 {
            let mut s = 0.0f64;
            for j in 0..N162 {
                s += a[i * N162 + j] * x[j];
            }
            y[i] = s;
        }
        let n = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n == 0.0 {
            break;
        }
        for i in 0..N162 {
            x[i] = y[i] / n;
        }
        lam = n;
    }
    lam.sqrt()
}

fn round(label: &str, single: bool) -> Vec<ShortChallenge> {
    let mut out = Vec::with_capacity(ROUND);
    let mut attempts = 0u64;
    let el = best(|| {
        let mut t = Transcript::new(b"bench");
        out.clear();
        attempts = 0;
        let start = Instant::now();
        for _ in 0..ROUND {
            let (c, a) = sample_short_challenge_op_norm_prec(
                &mut t,
                DEFAULT_WEIGHT,
                DEFAULT_OP_NORM_BOUND,
                single,
            );
            attempts += a;
            out.push(c);
        }
        start.elapsed().as_secs_f64()
    });
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    for c in &out {
        let n = op_norm_sq_f64(c).sqrt();
        sum += n;
        if n > max {
            max = n;
        }
    }
    println!(
        "{label:>10}: {:8.3} ms total, {:7.1} us/challenge, {:5.2} attempts, |M_c|_2 mean {:.4} max {:.4}, accept {:.1}%",
        el * 1e3,
        el * 1e6 / ROUND as f64,
        attempts as f64 / ROUND as f64,
        sum / ROUND as f64,
        max,
        100.0 * ROUND as f64 / attempts as f64,
    );
    out
}

fn main() {
    pin(CPU);
    let mut attempts = 0u64;
    let el = best(|| {
        let mut t = Transcript::new(b"bench");
        attempts = 0;
        let start = Instant::now();
        for _ in 0..ROUND {
            let (_, a) = sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND);
            attempts += a;
        }
        start.elapsed().as_secs_f64()
    });
    println!(
        "{:>10}: {:8.3} ms total, {:7.1} us/challenge, {:5.2} attempts",
        "canonical",
        el * 1e3,
        el * 1e6 / ROUND as f64,
        attempts as f64 / ROUND as f64
    );

    round("op-norm f64", false);
    round("op-norm f32", true);

    let cross: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(500);
    let mut t = Transcript::new(b"cross");
    let bound = DEFAULT_OP_NORM_BOUND;
    let (mut agree64, mut agree32, mut near) = (0usize, 0usize, 0usize);
    let (mut err64, mut err32) = (0.0f64, 0.0f64);
    let start = Instant::now();
    for _ in 0..cross {
        let c = sample_attempt(&mut t, DEFAULT_WEIGHT).signed();
        let d = dense_op_norm(&c);
        let n64 = op_norm_sq_f64(&c).sqrt();
        let n32 = op_norm_sq_f32(&c).sqrt();
        err64 = err64.max(((n64 - d) / d).abs());
        err32 = err32.max(((n32 - d) / d).abs());
        if (n64 <= bound) == (d <= bound) {
            agree64 += 1;
        }
        if (n32 <= bound) == (d <= bound) {
            agree32 += 1;
        }
        if ((d - bound) / bound).abs() < 1e-5 {
            near += 1;
        }
    }
    println!(
        "cross-check {cross} candidates in {:.1} s: f64 agrees {agree64}/{cross}, f32 agrees {agree32}/{cross}, borderline {near}",
        start.elapsed().as_secs_f64()
    );
    println!("max relative error vs dense: f64 {err64:.3e}, f32 {err32:.3e}");
}

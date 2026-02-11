//! Rust DCT benchmark for comparison with Sigil
//!
//! Compile and run: rustc -O bench_dct_rust.rs && ./bench_dct_rust

use std::f64::consts::PI;
use std::time::Instant;

fn dct_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let scale = (2.0 / n as f64).sqrt();
    let sqrt2 = 2.0_f64.sqrt();

    let mut output = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0_f64;
        for i in 0..n {
            let angle = PI * (k as f64) * ((i as f64) + 0.5) / (n as f64);
            sum += input[i] * angle.cos();
        }
        output[k] = sum * scale;
    }

    output[0] /= sqrt2;
    output
}

fn idct_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let scale_dc = (1.0 / n as f64).sqrt();
    let scale_ac = (2.0 / n as f64).sqrt();

    let mut output = vec![0.0; n];

    for i in 0..n {
        let mut sum = input[0] * scale_dc;

        for k in 1..n {
            let angle = PI * (k as f64) * ((i as f64) + 0.5) / (n as f64);
            sum += input[k] * scale_ac * angle.cos();
        }

        output[i] = sum;
    }

    output
}

fn generate_test_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).cos() * 0.5)
        .collect()
}

fn main() {
    println!("DCT/IDCT Benchmark - Rust (release)");
    println!("══════════════════════════════════════════════════");
    println!();

    let sizes = [16, 32, 64, 128, 256, 512];
    let iterations = [10000, 5000, 1000, 500, 100, 20];

    println!("{:>8} {:>12} {:>12} {:>15}", "Size", "DCT (μs)", "IDCT (μs)", "Roundtrip (μs)");
    println!("──────────────────────────────────────────────────");

    for (i, &n) in sizes.iter().enumerate() {
        let iters = iterations[i];
        let data = generate_test_data(n);

        // Warmup
        let _ = dct_1d(&data);

        // DCT benchmark
        let start = Instant::now();
        for _ in 0..iters {
            let _ = dct_1d(&data);
        }
        let dct_time = start.elapsed().as_micros() as f64 / iters as f64;

        // IDCT benchmark
        let coeffs = dct_1d(&data);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = idct_1d(&coeffs);
        }
        let idct_time = start.elapsed().as_micros() as f64 / iters as f64;

        // Roundtrip
        let start = Instant::now();
        for _ in 0..iters {
            let c = dct_1d(&data);
            let _ = idct_1d(&c);
        }
        let rt_time = start.elapsed().as_micros() as f64 / iters as f64;

        println!("{:>8} {:>12.2} {:>12.2} {:>15.2}", n, dct_time, idct_time, rt_time);
    }

    println!();
    println!("Algorithm: Naive O(n²) DCT-II");
}

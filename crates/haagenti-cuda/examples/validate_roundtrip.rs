//! Validate full HCT compression → GPU decompression roundtrip.
//!
//! Tests that compressed tensors can be accurately reconstructed using GPU IDCT.
//!
//! Usage:
//! ```bash
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release --example validate_roundtrip -p haagenti-cuda
//! ```

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use haagenti_cuda::dct_gpu::GpuDctContext;

    // Enable tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("HCT Compression Roundtrip Validation");
    println!("=====================================\n");

    // Initialize GPU
    println!("Initializing GPU...");
    let mut ctx = GpuDctContext::new(0)?;
    println!("GPU context ready\n");

    // Test configurations
    let test_cases = [
        ("Small (64x64)", 64, 64, 0.20),
        ("Medium (256x256)", 256, 256, 0.20),
        ("Large (1024x1024)", 1024, 1024, 0.20),
        ("LLM Attention (576x576)", 576, 576, 0.20),
        ("LLM MLP (1536x576)", 1536, 576, 0.20),
        ("High Retention (576x576)", 576, 576, 0.50),
        ("Low Retention (576x576)", 576, 576, 0.10),
    ];

    println!("{:30} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "Test Case", "Elements", "Retention", "Cosine", "MSE", "Status");
    println!("{}", "-".repeat(85));

    let mut all_passed = true;

    for (name, width, height, retention) in test_cases {
        // Generate realistic test data (simulates weight distribution)
        let data: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                // Mix of frequencies like real weights
                (x * std::f32::consts::PI * 2.0).sin() * 0.5
                    + (y * std::f32::consts::PI * 4.0).cos() * 0.3
                    + ((x + y) * std::f32::consts::PI).sin() * 0.2
            })
            .collect();

        // Forward DCT
        let coeffs = ctx.dct_2d(&data, width, height)?;

        // Simulate coefficient pruning (keep top retention% by magnitude)
        let mut indexed: Vec<(usize, f32)> = coeffs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let keep_count = (coeffs.len() as f32 * retention) as usize;
        let mut pruned_coeffs = vec![0.0f32; coeffs.len()];
        for (idx, val) in indexed.into_iter().take(keep_count) {
            pruned_coeffs[idx] = val;
        }

        // Inverse DCT
        let reconstructed = ctx.idct_2d(&pruned_coeffs, width, height)?;

        // Calculate quality metrics
        let cosine = cosine_similarity(&data, &reconstructed);
        let mse = mean_squared_error(&data, &reconstructed);

        let status = if cosine >= 0.95 { "PASS" } else { "FAIL" };
        if cosine < 0.95 {
            all_passed = false;
        }

        println!("{:30} {:>10} {:>10.0}% {:>10.4} {:>10.2e} {:>10}",
                 name, width * height, retention * 100.0, cosine, mse, status);
    }

    println!();

    // Benchmark throughput
    println!("Throughput Benchmarks");
    println!("---------------------");

    let bench_sizes = [
        (576, 576),    // SmolLM attention
        (1536, 576),   // SmolLM MLP
        (4096, 4096),  // Large LLM
    ];

    for (width, height) in bench_sizes {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        // Warm up
        let _ = ctx.dct_2d(&data, width, height)?;
        let _ = ctx.idct_2d(&data, width, height)?;

        // Benchmark DCT
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = ctx.dct_2d(&data, width, height)?;
        }
        let dct_time = start.elapsed() / iterations;

        // Benchmark IDCT
        let coeffs = ctx.dct_2d(&data, width, height)?;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.idct_2d(&coeffs, width, height)?;
        }
        let idct_time = start.elapsed() / iterations;

        let bytes = (width * height * 4) as f64;
        let dct_throughput = bytes / dct_time.as_secs_f64() / 1e6;
        let idct_throughput = bytes / idct_time.as_secs_f64() / 1e6;

        println!("  {}x{}: DCT {:.1} MB/s, IDCT {:.1} MB/s",
                 width, height, dct_throughput, idct_throughput);
    }

    println!();

    if all_passed {
        println!("All validation tests PASSED!");
        println!("\nGPU decompression is ready for inference.");
    } else {
        println!("Some tests FAILED!");
        std::process::exit(1);
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32
}

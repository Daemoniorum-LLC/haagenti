//! Test direct GPU DCT kernels for large tensors.
//!
//! This verifies the direct kernels (no shared memory limit) work correctly.
//!
//! Usage:
//! ```bash
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib RUST_LOG=debug cargo run --release --example test_direct_kernels
//! ```

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use haagenti_cuda::dct_gpu::GpuDctContext;

    // Enable tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Direct GPU DCT Kernel Test");
    println!("==========================\n");

    // Initialize GPU context
    println!("Initializing GPU context...");
    let mut ctx = GpuDctContext::new(0)?;
    println!("GPU context created successfully\n");

    // Test sizes: small (shared memory), large (direct kernels)
    let test_cases = [
        (64, 64, "Small (shared mem)"),
        (256, 256, "Medium (shared mem)"),
        (1024, 1024, "Boundary (shared mem)"),
        (1025, 1025, "Just over limit (direct)"),
        (2048, 2048, "Large (direct)"),
        (4096, 4096, "Very large (direct)"),
        // Typical LLM tensor shapes
        (3584, 4096, "LLM hidden projection"),
        (4096, 14336, "LLM MLP up projection"),
        (14336, 4096, "LLM MLP down projection"),
    ];

    for (width, height, desc) in test_cases {
        println!("Testing {}x{} - {}", width, height, desc);

        // Generate test data
        let data: Vec<f32> = (0..width * height)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();

        // Forward DCT
        let start = Instant::now();
        let coeffs = ctx.dct_2d(&data, width, height)?;
        let dct_time = start.elapsed();

        // Inverse DCT
        let start = Instant::now();
        let reconstructed = ctx.idct_2d(&coeffs, width, height)?;
        let idct_time = start.elapsed();

        // Calculate error
        let mse: f32 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let max_error: f32 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        // Calculate throughput
        let bytes = (width * height * 4) as f64;
        let dct_throughput = bytes / dct_time.as_secs_f64() / 1_000_000.0;
        let idct_throughput = bytes / idct_time.as_secs_f64() / 1_000_000.0;

        let pass = mse < 1e-3;
        println!(
            "  DCT:  {:>7.2} ms ({:>7.1} MB/s)",
            dct_time.as_secs_f64() * 1000.0,
            dct_throughput
        );
        println!(
            "  IDCT: {:>7.2} ms ({:>7.1} MB/s)",
            idct_time.as_secs_f64() * 1000.0,
            idct_throughput
        );
        println!(
            "  MSE: {:.2e}, Max Error: {:.2e} [{}]\n",
            mse,
            max_error,
            if pass { "PASS" } else { "FAIL" }
        );

        if !pass {
            eprintln!("ERROR: Reconstruction quality too low!");
            std::process::exit(1);
        }
    }

    println!("All tests passed!");

    Ok(())
}

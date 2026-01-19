//! HCT GPU Conformance Test
//!
//! Validates that GPU DCT/IDCT implementations produce outputs matching
//! the reference test vectors from the HCT specification.
//!
//! This is a **normative conformance test** - passing this test is required
//! for an implementation to claim HCT compliance.
//!
//! ## Usage
//!
//! ```bash
//! # On systems with NVIDIA GPU and CUDA:
//! cargo run --release --example conformance_test -p haagenti-cuda
//!
//! # With cuFFT support for large tensors:
//! cargo run --release --example conformance_test -p haagenti-cuda --features cufft
//!
//! # On WSL2 with GPU passthrough:
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release --example conformance_test -p haagenti-cuda
//! ```
//!
//! ## Test Vectors
//!
//! The test vectors are defined in the HCT Specification (docs/HCT-SPECIFICATION-DRAFT.md)
//! and implemented in `haagenti::hct_test_vectors`.
//!
//! Each test vector specifies:
//! - Input matrix
//! - Expected DCT coefficients
//! - Expected reconstruction quality (cosine similarity)
//! - Tolerance bounds
//!
//! ## Exit Codes
//!
//! - 0: All conformance tests passed
//! - 1: One or more tests failed
//! - 2: GPU initialization failed

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use haagenti::hct_test_vectors::all_test_vectors;
    use haagenti_cuda::dct_gpu::GpuDctContext;

    // Enable tracing for debugging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("haagenti_cuda=info".parse().unwrap()),
        )
        .init();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           HCT GPU CONFORMANCE TEST SUITE                      ║");
    println!("║   Reference: HCT-SPECIFICATION-DRAFT.md Section 7             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    println!("Initializing GPU context...");
    let start = Instant::now();
    let mut ctx = match GpuDctContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize GPU: {}", e);
            eprintln!();
            eprintln!("Possible causes:");
            eprintln!("  - No NVIDIA GPU available");
            eprintln!("  - CUDA drivers not installed");
            eprintln!("  - On WSL2: LD_LIBRARY_PATH=/usr/lib/wsl/lib not set");
            std::process::exit(2);
        }
    };
    println!("GPU initialized in {:?}", start.elapsed());
    println!();

    // Get all test vectors
    let vectors = all_test_vectors();
    let total_tests = vectors.len();
    let mut passed = 0;
    let mut failed = 0;

    println!("Running {} conformance tests...", total_tests);
    println!();
    println!("{:─<75}", "");
    println!(
        "{:25} {:>10} {:>12} {:>12} {:>8}",
        "Test Vector", "Shape", "GPU Cosine", "Expected", "Status"
    );
    println!("{:─<75}", "");

    for tv in &vectors {
        let result = run_conformance_test(&mut ctx, tv);

        match result {
            Ok(gpu_cosine) => {
                let tolerance = tv.cosine_tolerance;
                let expected = tv.expected_cosine_similarity;
                let diff = (gpu_cosine - expected).abs();

                let status = if diff <= tolerance {
                    passed += 1;
                    "PASS"
                } else {
                    failed += 1;
                    "FAIL"
                };

                let shape_str = format!("{}x{}", tv.shape[0], tv.shape[1]);
                println!(
                    "{:25} {:>10} {:>12.6} {:>12.6} {:>8}",
                    tv.name, shape_str, gpu_cosine, expected, status
                );

                if status == "FAIL" {
                    println!(
                        "    ^ Difference: {:.6} exceeds tolerance {:.6}",
                        diff, tolerance
                    );
                }
            }
            Err(e) => {
                failed += 1;
                let shape_str = format!("{}x{}", tv.shape[0], tv.shape[1]);
                println!(
                    "{:25} {:>10} {:>12} {:>12} {:>8}",
                    tv.name, shape_str, "ERROR", "-", "FAIL"
                );
                println!("    ^ Error: {}", e);
            }
        }
    }

    println!("{:─<75}", "");
    println!();

    // Detailed DCT coefficient comparison for sequential_4x4
    println!("Detailed DCT Coefficient Comparison (sequential_4x4)");
    println!("{:─<75}", "");

    if let Some(tv) = vectors.iter().find(|v| v.name == "sequential_4x4") {
        let rows = tv.shape[0];
        let cols = tv.shape[1];

        // Run GPU DCT
        let gpu_dct = ctx.dct_2d(&tv.input, cols, rows)?;

        println!(
            "{:>8} {:>15} {:>15} {:>12}",
            "Index", "GPU DCT", "Reference DCT", "Difference"
        );
        println!("{:─<55}", "");

        // Compare top coefficients by magnitude
        let mut indexed_gpu: Vec<(usize, f32)> =
            gpu_dct.iter().copied().enumerate().collect();
        indexed_gpu.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let mut indexed_ref: Vec<(usize, f32)> =
            tv.dct_coefficients.iter().copied().enumerate().collect();
        indexed_ref.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for i in 0..8.min(gpu_dct.len()) {
            let (idx, gpu_val) = indexed_gpu[i];
            let ref_val = tv.dct_coefficients[idx];
            let diff = (gpu_val - ref_val).abs();
            let row = idx / cols;
            let col = idx % cols;
            println!(
                "[{},{}]{:>4} {:>15.6} {:>15.6} {:>12.6}",
                row, col, "", gpu_val, ref_val, diff
            );
        }
    }

    println!();
    println!("{:═<75}", "");
    println!();

    // Summary
    println!("CONFORMANCE TEST SUMMARY");
    println!("========================");
    println!("  Total tests:  {}", total_tests);
    println!("  Passed:       {}", passed);
    println!("  Failed:       {}", failed);
    println!();

    if failed == 0 {
        println!("Result: ALL CONFORMANCE TESTS PASSED");
        println!();
        println!("This GPU implementation conforms to the HCT specification.");
        println!("Ready for production use.");
        Ok(())
    } else {
        println!("Result: CONFORMANCE TESTS FAILED");
        println!();
        println!("This GPU implementation does NOT conform to the HCT specification.");
        println!("Review the failures above and ensure DCT/IDCT matches the reference.");
        std::process::exit(1);
    }
}

/// Run conformance test for a single test vector.
///
/// Returns the cosine similarity achieved by GPU reconstruction.
fn run_conformance_test(
    ctx: &mut haagenti_cuda::dct_gpu::GpuDctContext,
    tv: &haagenti::hct_test_vectors::HctTestVector,
) -> Result<f32, Box<dyn std::error::Error>> {
    use haagenti::hct_test_vectors::cosine_similarity;

    let rows = tv.shape[0];
    let cols = tv.shape[1];
    let total = rows * cols;

    // Forward DCT on GPU
    let gpu_dct = ctx.dct_2d(&tv.input, cols, rows)?;

    // Apply coefficient truncation (same as reference)
    let retention = tv.retention;
    let keep_count = (total as f32 * retention).ceil() as usize;

    // Sort by magnitude and keep top coefficients
    let mut indexed: Vec<(usize, f32)> = gpu_dct.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    let mut truncated = vec![0.0f32; total];
    for (idx, val) in indexed.into_iter().take(keep_count) {
        truncated[idx] = val;
    }

    // Inverse DCT on GPU
    let reconstructed = ctx.idct_2d(&truncated, cols, rows)?;

    // Calculate cosine similarity
    let sim = cosine_similarity(&tv.input, &reconstructed);

    Ok(sim)
}

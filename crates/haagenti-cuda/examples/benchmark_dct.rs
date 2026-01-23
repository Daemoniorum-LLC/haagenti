//! Benchmark comparing GPU vs CPU DCT performance.
//!
//! Run with: LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release --example benchmark_dct

use haagenti_core::dct::{dct_2d, idct_2d};
use haagenti_cuda::dct_gpu::GpuDctContext;
use std::time::Instant;

fn generate_test_data(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.03).cos()) * 0.5)
        .collect()
}

fn benchmark_cpu_dct(data: &[f32], width: usize, height: usize, iterations: usize) -> f64 {
    let mut output = vec![0.0f32; width * height];

    let start = Instant::now();
    for _ in 0..iterations {
        dct_2d(data, &mut output, width, height);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_cpu_idct(data: &[f32], width: usize, height: usize, iterations: usize) -> f64 {
    let mut output = vec![0.0f32; width * height];

    let start = Instant::now();
    for _ in 0..iterations {
        idct_2d(data, &mut output, width, height);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_gpu_dct(
    ctx: &mut GpuDctContext,
    data: &[f32],
    width: usize,
    height: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Warmup
    let _ = ctx.dct_2d(data, width, height)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ctx.dct_2d(data, width, height)?;
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

fn benchmark_gpu_idct(
    ctx: &mut GpuDctContext,
    data: &[f32],
    width: usize,
    height: usize,
    iterations: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Warmup
    let _ = ctx.idct_2d(data, width, height)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ctx.idct_2d(data, width, height)?;
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DCT/IDCT Benchmark: GPU vs CPU");
    println!("==============================\n");

    // Try to create GPU context
    let mut gpu_ctx = match GpuDctContext::new(0) {
        Ok(ctx) => {
            println!("GPU: CUDA device initialized");
            Some(ctx)
        }
        Err(e) => {
            println!("GPU: Not available ({:?})", e);
            None
        }
    };

    // Test sizes from small to large
    let sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)];

    println!(
        "\n{:>12} {:>12} {:>12} {:>12} {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "Speedup", "Elements"
    );
    println!("{}", "-".repeat(60));

    for (width, height) in sizes {
        let data = generate_test_data(width, height);
        let elements = width * height;

        // Scale iterations inversely with size
        let iterations = match elements {
            n if n <= 4096 => 100,
            n if n <= 65536 => 50,
            n if n <= 262144 => 20,
            _ => 10,
        };

        // CPU benchmark
        let cpu_time_dct = benchmark_cpu_dct(&data, width, height, iterations);

        // GPU benchmark
        let (gpu_time_dct, speedup) = if let Some(ref mut ctx) = gpu_ctx {
            match benchmark_gpu_dct(ctx, &data, width, height, iterations) {
                Ok(t) => {
                    let s = cpu_time_dct / t;
                    (Some(t), Some(s))
                }
                Err(e) => {
                    eprintln!("GPU error at {}x{}: {:?}", width, height, e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        let gpu_str = gpu_time_dct
            .map(|t| format!("{:.3}", t * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());
        let speedup_str = speedup
            .map(|s| format!("{:.1}x", s))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:>5}x{:<5} {:>12.3} {:>12} {:>12} {:>10}",
            width,
            height,
            cpu_time_dct * 1000.0,
            gpu_str,
            speedup_str,
            elements
        );
    }

    // IDCT benchmark
    println!("\n\nIDCT Benchmark:");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>10}",
        "Size", "CPU (ms)", "GPU (ms)", "Speedup", "Elements"
    );
    println!("{}", "-".repeat(60));

    for (width, height) in sizes {
        let data = generate_test_data(width, height);
        let elements = width * height;

        let iterations = match elements {
            n if n <= 4096 => 100,
            n if n <= 65536 => 50,
            n if n <= 262144 => 20,
            _ => 10,
        };

        let cpu_time = benchmark_cpu_idct(&data, width, height, iterations);

        let (gpu_time, speedup) = if let Some(ref mut ctx) = gpu_ctx {
            match benchmark_gpu_idct(ctx, &data, width, height, iterations) {
                Ok(t) => (Some(t), Some(cpu_time / t)),
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        let gpu_str = gpu_time
            .map(|t| format!("{:.3}", t * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());
        let speedup_str = speedup
            .map(|s| format!("{:.1}x", s))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:>5}x{:<5} {:>12.3} {:>12} {:>12} {:>10}",
            width,
            height,
            cpu_time * 1000.0,
            gpu_str,
            speedup_str,
            elements
        );
    }

    // Batch benchmark
    if let Some(ref mut ctx) = gpu_ctx {
        println!("\n\nBatch DCT Benchmark (10 tensors at once):");
        println!(
            "{:>12} {:>12} {:>12} {:>12}",
            "Size", "CPU (ms)", "GPU (ms)", "Speedup"
        );
        println!("{}", "-".repeat(50));

        for (width, height) in &[(64, 64), (128, 128), (256, 256)] {
            let tensors: Vec<Vec<f32>> = (0..10)
                .map(|i| generate_test_data(*width, *height))
                .collect();
            let tensor_refs: Vec<&[f32]> = tensors.iter().map(|t| t.as_slice()).collect();

            // CPU (sequential)
            let start = Instant::now();
            let mut cpu_output = vec![0.0f32; width * height];
            for t in &tensors {
                dct_2d(t, &mut cpu_output, *width, *height);
            }
            let cpu_time = start.elapsed().as_secs_f64();

            // GPU (batch)
            let start = Instant::now();
            let _ = ctx.batch_dct_2d(&tensor_refs, *width, *height)?;
            let gpu_time = start.elapsed().as_secs_f64();

            let speedup = cpu_time / gpu_time;

            println!(
                "{:>5}x{:<5} {:>12.3} {:>12.3} {:>12.1}x",
                width,
                height,
                cpu_time * 1000.0,
                gpu_time * 1000.0,
                speedup
            );
        }
    }

    println!("\nBenchmark complete.");
    Ok(())
}

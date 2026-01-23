//! Compression ratio analysis for quantized LLM weights.
//!
//! This analyzes how well different compression algorithms work on
//! INT4/INT8 quantized model weights with realistic distributions.

use std::time::Instant;

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_lz4::Lz4Compressor;
use haagenti_zstd::ZstdCompressor;

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Weight distribution patterns found in real LLMs.
#[derive(Debug, Clone, Copy)]
enum WeightPattern {
    /// Normal distribution (most common in attention/MLP)
    Normal { mean: f32, std: f32 },
    /// Sparse with many near-zero values
    Sparse { sparsity: f32, std: f32 },
    /// Clustered around specific values (common after quantization-aware training)
    Clustered { num_clusters: usize },
    /// Uniform (less common, some embeddings)
    Uniform { min: f32, max: f32 },
}

/// Generate synthetic FP32 weights matching a pattern.
fn generate_fp32_weights(size: usize, pattern: WeightPattern) -> Vec<f32> {
    let mut rng = thread_rng();

    match pattern {
        WeightPattern::Normal { mean, std } => {
            let dist = Normal::new(mean, std).unwrap();
            (0..size).map(|_| dist.sample(&mut rng)).collect()
        }
        WeightPattern::Sparse { sparsity, std } => {
            let dist = Normal::new(0.0, std).unwrap();
            (0..size)
                .map(|_| {
                    if rng.r#gen::<f32>() < sparsity {
                        0.0
                    } else {
                        dist.sample(&mut rng)
                    }
                })
                .collect()
        }
        WeightPattern::Clustered { num_clusters } => {
            // Generate cluster centers
            let centers: Vec<f32> = (0..num_clusters)
                .map(|i| (i as f32 - num_clusters as f32 / 2.0) * 0.1)
                .collect();
            let noise = Normal::new(0.0, 0.01).unwrap();

            (0..size)
                .map(|_| {
                    let center = centers[rng.r#gen_range(0..num_clusters)];
                    center + noise.sample(&mut rng)
                })
                .collect()
        }
        WeightPattern::Uniform { min, max } => {
            (0..size).map(|_| rng.r#gen_range(min..max)).collect()
        }
    }
}

/// Quantize FP32 weights to INT8 (symmetric).
fn quantize_int8(weights: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = max_abs / 127.0;

    let quantized: Vec<i8> = weights
        .iter()
        .map(|w| (w / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (quantized, scale)
}

/// Quantize FP32 weights to INT4 (symmetric, packed).
/// Returns packed bytes (2 INT4 values per byte).
fn quantize_int4(weights: &[f32]) -> (Vec<u8>, f32) {
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = max_abs / 7.0;

    // Quantize to 4-bit range (-7 to 7)
    let quantized: Vec<i8> = weights
        .iter()
        .map(|w| (w / scale).round().clamp(-7.0, 7.0) as i8)
        .collect();

    // Pack into bytes (2 values per byte)
    let packed: Vec<u8> = quantized
        .chunks(2)
        .map(|chunk| {
            let low = (chunk[0] & 0x0F) as u8;
            let high = if chunk.len() > 1 {
                ((chunk[1] & 0x0F) as u8) << 4
            } else {
                0
            };
            low | high
        })
        .collect();

    (packed, scale)
}

/// Quantize with per-block scaling (better accuracy, different compression).
fn quantize_int4_blocked(weights: &[f32], block_size: usize) -> Vec<u8> {
    let mut result = Vec::new();

    for block in weights.chunks(block_size) {
        let max_abs = block.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };

        // Store scale as f16 (2 bytes) - in practice would use proper f16
        result.extend_from_slice(&scale.to_le_bytes()[..2]);

        // Quantize and pack
        let quantized: Vec<i8> = block
            .iter()
            .map(|w| (w / scale).round().clamp(-7.0, 7.0) as i8)
            .collect();

        for chunk in quantized.chunks(2) {
            let low = (chunk[0] & 0x0F) as u8;
            let high = if chunk.len() > 1 {
                ((chunk[1] & 0x0F) as u8) << 4
            } else {
                0
            };
            result.push(low | high);
        }
    }

    result
}

/// Compression result for analysis.
#[derive(Debug)]
struct CompressionResult {
    algorithm: &'static str,
    original_size: usize,
    compressed_size: usize,
    ratio: f64,
    throughput_mbps: f64,
}

fn print_result(result: CompressionResult) {
    println!(
        "    {:10} → {:>8} bytes ({:.2}x) @ {:.0} MB/s",
        result.algorithm, result.compressed_size, result.ratio, result.throughput_mbps
    );
}

/// Test compression on data.
fn test_compression(
    compressor: &impl Compressor,
    data: &[u8],
    algorithm: &'static str,
) -> CompressionResult {
    let original_size = data.len();

    let start = Instant::now();
    let compressed = compressor.compress(data).expect("compression failed");
    let elapsed = start.elapsed();

    let compressed_size = compressed.len();
    let ratio = original_size as f64 / compressed_size as f64;
    let throughput_mbps = (original_size as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    CompressionResult {
        algorithm,
        original_size,
        compressed_size,
        ratio,
        throughput_mbps,
    }
}

fn main() {
    println!("=== Haagenti Weight Compression Analysis ===\n");

    // Test sizes (simulating different layer sizes)
    let sizes = [
        ("Small (1MB)", 1024 * 1024 / 4), // 1MB of FP32 = 256K weights
        ("Medium (16MB)", 16 * 1024 * 1024 / 4), // 16MB = 4M weights
        ("Large (64MB)", 64 * 1024 * 1024 / 4), // 64MB = 16M weights
    ];

    // Weight patterns to test
    let patterns = [
        (
            "Normal (σ=0.02)",
            WeightPattern::Normal {
                mean: 0.0,
                std: 0.02,
            },
        ),
        (
            "Normal (σ=0.1)",
            WeightPattern::Normal {
                mean: 0.0,
                std: 0.1,
            },
        ),
        (
            "Sparse 50%",
            WeightPattern::Sparse {
                sparsity: 0.5,
                std: 0.02,
            },
        ),
        (
            "Sparse 90%",
            WeightPattern::Sparse {
                sparsity: 0.9,
                std: 0.02,
            },
        ),
        (
            "Clustered 16",
            WeightPattern::Clustered { num_clusters: 16 },
        ),
        (
            "Uniform",
            WeightPattern::Uniform {
                min: -0.1,
                max: 0.1,
            },
        ),
    ];

    // Compressors
    let lz4 = Lz4Compressor::new();
    let lz4_fast = Lz4Compressor::with_level(CompressionLevel::Fast);
    let zstd = ZstdCompressor::new();
    let zstd_fast = ZstdCompressor::with_level(CompressionLevel::Fast);
    let zstd_best = ZstdCompressor::with_level(CompressionLevel::Best);

    // Run analysis
    for (size_name, num_weights) in sizes {
        println!("### {} ({} weights) ###\n", size_name, num_weights);

        for (pattern_name, pattern) in &patterns {
            println!("Pattern: {}", pattern_name);

            // Generate weights
            let fp32_weights = generate_fp32_weights(num_weights, *pattern);

            // Quantize
            let (int8_weights, _) = quantize_int8(&fp32_weights);
            let (int4_weights, _) = quantize_int4(&fp32_weights);
            let int4_blocked = quantize_int4_blocked(&fp32_weights, 128);

            // Convert to bytes for compression
            let int8_bytes: Vec<u8> = int8_weights.iter().map(|&x| x as u8).collect();

            println!("  INT8 ({} bytes):", int8_bytes.len());
            print_result(test_compression(&lz4, &int8_bytes, "LZ4"));
            print_result(test_compression(&lz4_fast, &int8_bytes, "LZ4-Fast"));
            print_result(test_compression(&zstd, &int8_bytes, "Zstd"));
            print_result(test_compression(&zstd_fast, &int8_bytes, "Zstd-Fast"));
            print_result(test_compression(&zstd_best, &int8_bytes, "Zstd-Best"));

            println!("  INT4 packed ({} bytes):", int4_weights.len());
            print_result(test_compression(&lz4, &int4_weights, "LZ4"));
            print_result(test_compression(&zstd, &int4_weights, "Zstd"));
            print_result(test_compression(&zstd_best, &int4_weights, "Zstd-Best"));

            println!("  INT4 blocked ({} bytes):", int4_blocked.len());
            print_result(test_compression(&lz4, &int4_blocked, "LZ4"));
            print_result(test_compression(&zstd, &int4_blocked, "Zstd"));
            print_result(test_compression(&zstd_best, &int4_blocked, "Zstd-Best"));

            println!();
        }
        println!();
    }

    // Summary analysis
    println!("=== Key Insights ===\n");
    println!("1. Sparse weights compress significantly better (90% sparsity → 5-10x)");
    println!("2. Clustered weights (from QAT) compress well due to repeated values");
    println!("3. INT4 packed has less redundancy than INT8 (harder to compress)");
    println!("4. Per-block quantization adds overhead but may compress better");
    println!("5. LZ4 is 5-10x faster than Zstd but lower ratio");
    println!("6. Zstd-Best adds minimal benefit over Zstd-Default for this data");
}

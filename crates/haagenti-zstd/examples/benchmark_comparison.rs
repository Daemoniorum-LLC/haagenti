//! Formal benchmark comparison between haagenti-zstd and reference zstd.
//!
//! Run with: cargo run --release --example benchmark_comparison --features parallel

use std::time::{Duration, Instant};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           Haagenti-Zstd vs Reference Zstd Benchmark Comparison               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Test data generators
    let test_cases: Vec<(&str, Box<dyn Fn(usize) -> Vec<u8>>)> = vec![
        ("Text (English)", Box::new(generate_text)),
        ("Binary (Mixed)", Box::new(generate_binary)),
        ("Repetitive", Box::new(generate_repetitive)),
        ("High Entropy", Box::new(generate_high_entropy)),
    ];

    let sizes = [1024, 4096, 16384, 65536];

    // Print header
    println!("┌─────────────────┬────────┬───────────────────────────────────────────────────────────────┐");
    println!("│                 │        │              Compression Throughput (MB/s)                    │");
    println!("│   Data Type     │  Size  ├──────────────┬──────────────┬──────────────┬─────────────────┤");
    println!("│                 │        │  Reference   │  Haagenti    │  Speculative │  vs Reference   │");
    println!("├─────────────────┼────────┼──────────────┼──────────────┼──────────────┼─────────────────┤");

    for (name, generator) in &test_cases {
        for &size in &sizes {
            let data = generator(size);
            let result = benchmark_compression(&data, size);

            let size_str = format_size(size);
            let ref_tp = format!("{:>8.1}", result.ref_throughput);
            let our_tp = format!("{:>8.1}", result.our_throughput);
            let spec_tp = format!("{:>8.1}", result.spec_throughput);
            let ratio = result.our_throughput / result.ref_throughput * 100.0;
            let ratio_str = format!("{:>6.0}%", ratio);

            println!(
                "│ {:15} │ {:>6} │ {:>12} │ {:>12} │ {:>12} │ {:>15} │",
                name, size_str, ref_tp, our_tp, spec_tp, ratio_str
            );
        }
        println!("├─────────────────┼────────┼──────────────┼──────────────┼──────────────┼─────────────────┤");
    }

    // Print decompression header
    println!();
    println!("┌─────────────────┬────────┬───────────────────────────────────────────────────────────────┐");
    println!("│                 │        │             Decompression Throughput (MB/s)                   │");
    println!("│   Data Type     │  Size  ├──────────────┬──────────────┬─────────────────────────────────┤");
    println!("│                 │        │  Reference   │  Haagenti    │          vs Reference           │");
    println!("├─────────────────┼────────┼──────────────┼──────────────┼─────────────────────────────────┤");

    for (name, generator) in &test_cases {
        for &size in &sizes {
            let data = generator(size);
            let result = benchmark_decompression(&data);

            let size_str = format_size(size);
            let ref_tp = format!("{:>8.1}", result.ref_throughput);
            let our_tp = format!("{:>8.1}", result.our_throughput);
            let ratio = result.our_throughput / result.ref_throughput * 100.0;
            let ratio_str = format!("{:>6.0}%", ratio);

            println!(
                "│ {:15} │ {:>6} │ {:>12} │ {:>12} │ {:>31} │",
                name, size_str, ref_tp, our_tp, ratio_str
            );
        }
        println!("├─────────────────┼────────┼──────────────┼──────────────┼─────────────────────────────────┤");
    }

    // Print compression ratio header
    println!();
    println!("┌─────────────────┬────────┬───────────────────────────────────────────────────────────────┐");
    println!("│                 │        │                  Compression Ratio                            │");
    println!("│   Data Type     │  Size  ├──────────────┬──────────────┬──────────────┬─────────────────┤");
    println!("│                 │        │  Reference   │  Haagenti    │  Speculative │   Difference    │");
    println!("├─────────────────┼────────┼──────────────┼──────────────┼──────────────┼─────────────────┤");

    for (name, generator) in &test_cases {
        for &size in &sizes {
            let data = generator(size);
            let result = benchmark_ratio(&data);

            let size_str = format_size(size);
            let ref_ratio = format!("{:>8.2}x", result.ref_ratio);
            let our_ratio = format!("{:>8.2}x", result.our_ratio);
            let spec_ratio = format!("{:>8.2}x", result.spec_ratio);
            let diff = (result.our_ratio / result.ref_ratio - 1.0) * 100.0;
            let diff_str = if diff >= 0.0 {
                format!("{:>+6.1}%", diff)
            } else {
                format!("{:>6.1}%", diff)
            };

            println!(
                "│ {:15} │ {:>6} │ {:>12} │ {:>12} │ {:>12} │ {:>15} │",
                name, size_str, ref_ratio, our_ratio, spec_ratio, diff_str
            );
        }
        println!("├─────────────────┼────────┼──────────────┼──────────────┼──────────────┼─────────────────┤");
    }

    // Summary
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Summary                                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let mut total_ref_compress = 0.0;
    let mut total_our_compress = 0.0;
    let mut total_spec_compress = 0.0;
    let mut total_ref_decompress = 0.0;
    let mut total_our_decompress = 0.0;
    let mut count = 0.0;

    for (_, generator) in &test_cases {
        for &size in &sizes {
            let data = generator(size);
            let comp = benchmark_compression(&data, size);
            let decomp = benchmark_decompression(&data);

            total_ref_compress += comp.ref_throughput;
            total_our_compress += comp.our_throughput;
            total_spec_compress += comp.spec_throughput;
            total_ref_decompress += decomp.ref_throughput;
            total_our_decompress += decomp.our_throughput;
            count += 1.0;
        }
    }

    let avg_compress_ratio = (total_our_compress / total_ref_compress) * 100.0;
    let avg_spec_ratio = (total_spec_compress / total_ref_compress) * 100.0;
    let avg_decompress_ratio = (total_our_decompress / total_ref_decompress) * 100.0;

    println!();
    println!(
        "  Average Compression Throughput vs Reference:   {:>6.1}%",
        avg_compress_ratio
    );
    println!(
        "  Average Speculative Throughput vs Reference:   {:>6.1}%",
        avg_spec_ratio
    );
    println!(
        "  Average Decompression Throughput vs Reference: {:>6.1}%",
        avg_decompress_ratio
    );
    println!();

    // Feature summary
    println!("  Optimizations Applied:");
    println!("  ✓ Phase 1: Huffman chunked processing, O(n log n) weights, batch hash update");
    println!("  ✓ Phase 2: AVX-512 match finding (64 bytes/iteration)");
    println!("  ✓ Phase 3: Fast entropy fingerprinting, early exit for incompressible data");
    println!("  ✓ Phase 4: Speculative multi-path compression (5 parallel strategies)");
    println!();
}

struct CompressionResult {
    ref_throughput: f64,
    our_throughput: f64,
    spec_throughput: f64,
}

struct DecompressionResult {
    ref_throughput: f64,
    our_throughput: f64,
}

struct RatioResult {
    ref_ratio: f64,
    our_ratio: f64,
    spec_ratio: f64,
}

fn benchmark_compression(data: &[u8], _size: usize) -> CompressionResult {
    let iterations = 100.max(1_000_000 / data.len());

    // Benchmark reference zstd
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(zstd::encode_all(std::io::Cursor::new(data), 1));
    }
    let ref_elapsed = start.elapsed();
    let ref_throughput = calculate_throughput(data.len(), iterations, ref_elapsed);

    // Benchmark haagenti-zstd
    let mut ctx =
        haagenti_zstd::compress::CompressContext::new(haagenti_core::CompressionLevel::Fast);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(ctx.compress(data));
    }
    let our_elapsed = start.elapsed();
    let our_throughput = calculate_throughput(data.len(), iterations, our_elapsed);

    // Benchmark speculative compression
    let spec_compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(spec_compressor.compress(data));
    }
    let spec_elapsed = start.elapsed();
    let spec_throughput = calculate_throughput(data.len(), iterations, spec_elapsed);

    CompressionResult {
        ref_throughput,
        our_throughput,
        spec_throughput,
    }
}

fn benchmark_decompression(data: &[u8]) -> DecompressionResult {
    // Compress data first
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(data), 1).unwrap();
    let our_compressed = {
        let mut ctx =
            haagenti_zstd::compress::CompressContext::new(haagenti_core::CompressionLevel::Fast);
        ctx.compress(data).unwrap()
    };

    let iterations = 100.max(1_000_000 / data.len());

    // Benchmark reference decompression
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(zstd::decode_all(std::io::Cursor::new(&ref_compressed)));
    }
    let ref_elapsed = start.elapsed();
    let ref_throughput = calculate_throughput(data.len(), iterations, ref_elapsed);

    // Benchmark haagenti decompression
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(haagenti_zstd::decompress::decompress_frame(&our_compressed));
    }
    let our_elapsed = start.elapsed();
    let our_throughput = calculate_throughput(data.len(), iterations, our_elapsed);

    DecompressionResult {
        ref_throughput,
        our_throughput,
    }
}

fn benchmark_ratio(data: &[u8]) -> RatioResult {
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(data), 1).unwrap();

    let mut ctx =
        haagenti_zstd::compress::CompressContext::new(haagenti_core::CompressionLevel::Fast);
    let our_compressed = ctx.compress(data).unwrap();

    let spec_compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let spec_compressed = spec_compressor.compress(data).unwrap();

    let original_size = data.len() as f64;

    RatioResult {
        ref_ratio: original_size / ref_compressed.len() as f64,
        our_ratio: original_size / our_compressed.len() as f64,
        spec_ratio: original_size / spec_compressed.len() as f64,
    }
}

fn calculate_throughput(size: usize, iterations: usize, elapsed: Duration) -> f64 {
    let total_bytes = size * iterations;
    let seconds = elapsed.as_secs_f64();
    (total_bytes as f64) / seconds / 1_000_000.0 // MB/s
}

fn format_size(size: usize) -> String {
    if size >= 1024 * 1024 {
        format!("{}MB", size / (1024 * 1024))
    } else if size >= 1024 {
        format!("{}KB", size / 1024)
    } else {
        format!("{}B", size)
    }
}

fn generate_text(size: usize) -> Vec<u8> {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";
    sample.iter().cycle().take(size).copied().collect()
}

fn generate_binary(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| {
            let x = (i as u64).wrapping_mul(0x5851f42d4c957f2d);
            ((x >> 24) ^ (x >> 48)) as u8
        })
        .collect()
}

fn generate_repetitive(size: usize) -> Vec<u8> {
    b"abcdefgh".iter().cycle().take(size).copied().collect()
}

fn generate_high_entropy(size: usize) -> Vec<u8> {
    // Pseudo-random but deterministic
    let mut state: u64 = 0x853c49e6748fea9b;
    (0..size)
        .map(|_| {
            state = state
                .wrapping_mul(0x5851f42d4c957f2d)
                .wrapping_add(0x14057b7ef767814f);
            ((state >> 33) ^ state) as u8
        })
        .collect()
}

//! Comprehensive Haagenti Benchmark Example
//!
//! This example exercises all major Haagenti components and prints performance metrics.
//! Use this for quick validation and smoke testing across the entire stack.
//!
//! Run with:
//!   cargo run --release --example benchmark_comprehensive --features "lz4,zstd"
//!
//! For full features:
//!   cargo run --release --example benchmark_comprehensive --features "full"

use std::time::Instant;

// Core imports
use haagenti::holotensor::{dct_1d, dct_2d};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║               Haagenti Comprehensive Benchmark Suite                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // System info
    print_system_info();

    // Run all benchmark groups
    let mut total_passed = 0;
    let mut total_failed = 0;

    // 1. DCT Primitives
    let (passed, failed) = bench_dct_primitives();
    total_passed += passed;
    total_failed += failed;

    // 2. Compression algorithms
    let (passed, failed) = bench_compression();
    total_passed += passed;
    total_failed += failed;

    // 3. HoloTensor encoding
    let (passed, failed) = bench_holotensor();
    total_passed += passed;
    total_failed += failed;

    // 4. Memory throughput
    let (passed, failed) = bench_memory_throughput();
    total_passed += passed;
    total_failed += failed;

    // Summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                                 SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Total benchmarks:  {}", total_passed + total_failed);
    println!("  Passed:            {} ✓", total_passed);
    println!("  Failed:            {} ✗", total_failed);
    println!();

    if total_failed > 0 {
        println!("⚠ Some benchmarks did not meet targets. See details above.");
        std::process::exit(1);
    } else {
        println!("✓ All benchmarks passed!");
    }
}

fn print_system_info() {
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("  System Information");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();

    // CPU features
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("  CPU: AVX-512 supported ✓");
        } else if is_x86_feature_detected!("avx2") {
            println!("  CPU: AVX2 supported ✓");
        } else if is_x86_feature_detected!("sse4.1") {
            println!("  CPU: SSE4.1 supported");
        } else {
            println!("  CPU: Scalar only");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  CPU: ARM64 with NEON");
    }

    println!("  Rust: {}", env!("CARGO_PKG_VERSION"));
    println!();
}

fn bench_dct_primitives() -> (usize, usize) {
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("  1. DCT Primitives");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();

    let mut passed = 0;
    let mut failed = 0;

    // 1D DCT
    for size in [64, 256, 1024, 4096] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0f32; size];

        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            dct_1d(&data, &mut output);
        }
        let elapsed = start.elapsed();

        let throughput_mbs =
            (size as f64 * 4.0 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;
        let target_mbs = 100.0; // 100 MB/s minimum

        let status = if throughput_mbs >= target_mbs {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };

        println!(
            "  dct_1d({:>4}):  {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}",
            size, throughput_mbs, target_mbs, status
        );
    }

    // 2D DCT
    for (w, h) in [(64, 64), (128, 128), (256, 256)] {
        let size = w * h;
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut output = vec![0.0f32; size];

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            dct_2d(&data, &mut output, w, h);
        }
        let elapsed = start.elapsed();

        let throughput_mbs =
            (size as f64 * 4.0 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;
        let target_mbs = 50.0; // 50 MB/s minimum for 2D

        let status = if throughput_mbs >= target_mbs {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };

        println!(
            "  dct_2d({:>3}×{:<3}):  {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}",
            w, h, throughput_mbs, target_mbs, status
        );
    }

    println!();
    (passed, failed)
}

fn bench_compression() -> (usize, usize) {
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("  2. Compression Algorithms");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();

    let mut passed = 0;
    let mut failed = 0;

    // Generate test data (simulated LLM weights)
    let size = 65536; // 64KB
    let data: Vec<u8> = (0..size)
        .map(|i| {
            let f = (i as f32 * 0.001).sin() * 0.1;
            let bits = half::f16::from_f32(f).to_bits();
            if i % 2 == 0 {
                (bits & 0xFF) as u8
            } else {
                (bits >> 8) as u8
            }
        })
        .collect();

    // LZ4 (if available)
    #[cfg(feature = "lz4")]
    {
        use haagenti_core::{Compressor, Decompressor};
        use haagenti_lz4::{Lz4Compressor, Lz4Decompressor};

        let compressor = Lz4Compressor::new();
        let decompressor = Lz4Decompressor::new();

        // Compression
        let iterations = 1000;
        let start = Instant::now();
        let mut compressed = Vec::new();
        for _ in 0..iterations {
            compressed = compressor.compress(&data).unwrap();
        }
        let compress_elapsed = start.elapsed();

        let compress_throughput =
            (size as f64 * iterations as f64) / compress_elapsed.as_secs_f64() / 1_000_000.0;
        let ratio = size as f64 / compressed.len() as f64;

        // Decompression
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = decompressor.decompress(&compressed).unwrap();
        }
        let decompress_elapsed = start.elapsed();

        let decompress_throughput =
            (size as f64 * iterations as f64) / decompress_elapsed.as_secs_f64() / 1_000_000.0;

        let compress_target = 500.0; // 500 MB/s
        let decompress_target = 2000.0; // 2 GB/s

        let status_c = if compress_throughput >= compress_target {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };
        let status_d = if decompress_throughput >= decompress_target {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };

        println!(
            "  LZ4 compress:    {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}  ratio: {:.2}x",
            compress_throughput, compress_target, status_c, ratio
        );
        println!(
            "  LZ4 decompress:  {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}",
            decompress_throughput, decompress_target, status_d
        );
    }

    // Zstd (if available)
    #[cfg(feature = "zstd")]
    {
        use haagenti_core::{Compressor, Decompressor};
        use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

        let compressor = ZstdCompressor::new();
        let decompressor = ZstdDecompressor::new();

        // Compression
        let iterations = 500;
        let start = Instant::now();
        let mut compressed = Vec::new();
        for _ in 0..iterations {
            compressed = compressor.compress(&data).unwrap();
        }
        let compress_elapsed = start.elapsed();

        let compress_throughput =
            (size as f64 * iterations as f64) / compress_elapsed.as_secs_f64() / 1_000_000.0;
        let ratio = size as f64 / compressed.len() as f64;

        // Decompression
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = decompressor.decompress(&compressed).unwrap();
        }
        let decompress_elapsed = start.elapsed();

        let decompress_throughput =
            (size as f64 * iterations as f64) / decompress_elapsed.as_secs_f64() / 1_000_000.0;

        let compress_target = 200.0; // 200 MB/s
        let decompress_target = 500.0; // 500 MB/s

        let status_c = if compress_throughput >= compress_target {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };
        let status_d = if decompress_throughput >= decompress_target {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };

        println!(
            "  Zstd compress:   {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}  ratio: {:.2}x",
            compress_throughput, compress_target, status_c, ratio
        );
        println!(
            "  Zstd decompress: {:>8.1} MB/s  (target: {:>6.0} MB/s)  {}",
            decompress_throughput, decompress_target, status_d
        );
    }

    #[cfg(not(any(feature = "lz4", feature = "zstd")))]
    {
        println!("  (Enable lz4 or zstd feature for compression benchmarks)");
    }

    println!();
    (passed, failed)
}

fn bench_holotensor() -> (usize, usize) {
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("  3. HoloTensor Spectral Encoding");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();

    let mut passed = 0;
    let mut failed = 0;

    // Test spectral encoding quality
    use haagenti::holotensor::{SpectralDecoder, SpectralEncoder};

    let (width, height) = (128, 128);
    let size = width * height;

    // Generate low-rank test data (simulates attention weights)
    let data: Vec<f32> = {
        let mut result = vec![0.0f32; size];
        for r in 0..5 {
            let scale = 1.0 / (r as f32 + 1.0).sqrt();
            for i in 0..height {
                for j in 0..width {
                    result[i * width + j] += ((i + r * 17) as f32 * 0.1).sin()
                        * ((j + r * 23) as f32 * 0.1).cos()
                        * scale
                        * 0.1;
                }
            }
        }
        result
    };

    for num_fragments in [4, 8, 16] {
        let encoder = SpectralEncoder::new(num_fragments);

        // Measure encoding
        let iterations = 100;
        let start = Instant::now();
        let mut fragments = Vec::new();
        for _ in 0..iterations {
            fragments = encoder.encode_2d(&data, width, height).unwrap();
        }
        let encode_elapsed = start.elapsed();

        // Measure decoding
        let start = Instant::now();
        for _ in 0..iterations {
            let mut decoder = SpectralDecoder::new(width, height, num_fragments);
            for frag in &fragments {
                decoder.add_fragment(frag).unwrap();
            }
            let _ = decoder.reconstruct();
        }
        let decode_elapsed = start.elapsed();

        // Calculate quality
        let mut decoder = SpectralDecoder::new(width, height, num_fragments);
        for frag in &fragments {
            decoder.add_fragment(frag).unwrap();
        }
        let reconstructed = decoder.reconstruct();

        let mse: f32 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / size as f32;
        let psnr = if mse > 0.0 {
            10.0 * (1.0 / mse).log10()
        } else {
            100.0
        };

        let encode_throughput =
            (size as f64 * 4.0 * iterations as f64) / encode_elapsed.as_secs_f64() / 1_000_000.0;
        let decode_throughput =
            (size as f64 * 4.0 * iterations as f64) / decode_elapsed.as_secs_f64() / 1_000_000.0;

        // Compressed size
        let compressed_size: usize = fragments.iter().map(|f| f.data.len()).sum();
        let compression_ratio = (size * 4) as f64 / compressed_size as f64;

        let quality_target = 30.0; // 30 dB PSNR minimum
        let throughput_target = 10.0; // 10 MB/s minimum

        let status_q = if psnr >= quality_target {
            passed += 1;
            "✓"
        } else {
            failed += 1;
            "✗"
        };
        let status_t =
            if encode_throughput >= throughput_target && decode_throughput >= throughput_target {
                passed += 1;
                "✓"
            } else {
                failed += 1;
                "✗"
            };

        println!("  Spectral ({:>2} frag): PSNR={:>5.1} dB {}  ratio={:.2}x  enc={:.0} MB/s  dec={:.0} MB/s {}",
                 num_fragments, psnr, status_q, compression_ratio, encode_throughput, decode_throughput, status_t);
    }

    println!();
    (passed, failed)
}

fn bench_memory_throughput() -> (usize, usize) {
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("  4. Memory Throughput Baselines");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();

    let mut passed = 0;
    let mut failed = 0;

    // Sequential read
    let size = 64 * 1024 * 1024; // 64 MB
    let data: Vec<u8> = (0..size).map(|i| i as u8).collect();
    let mut sum: u64 = 0;

    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        for chunk in data.chunks(64) {
            sum = sum.wrapping_add(chunk.iter().map(|&b| b as u64).sum::<u64>());
        }
    }
    let elapsed = start.elapsed();
    let _ = sum; // Prevent optimization

    let read_throughput =
        (size as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;
    let read_target = 5.0; // 5 GB/s minimum

    let status = if read_throughput >= read_target {
        passed += 1;
        "✓"
    } else {
        failed += 1;
        "✗"
    };
    println!(
        "  Sequential read:  {:>6.1} GB/s  (target: {:>4.0} GB/s)  {}",
        read_throughput, read_target, status
    );

    // Sequential write
    let mut output = vec![0u8; size];
    let start = Instant::now();
    for _ in 0..iterations {
        for (i, chunk) in output.chunks_mut(64).enumerate() {
            chunk.fill((i & 0xFF) as u8);
        }
    }
    let elapsed = start.elapsed();

    let write_throughput =
        (size as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;
    let write_target = 3.0; // 3 GB/s minimum

    let status = if write_throughput >= write_target {
        passed += 1;
        "✓"
    } else {
        failed += 1;
        "✗"
    };
    println!(
        "  Sequential write: {:>6.1} GB/s  (target: {:>4.0} GB/s)  {}",
        write_throughput, write_target, status
    );

    // Copy throughput
    let src: Vec<u8> = (0..size).map(|i| i as u8).collect();
    let mut dst = vec![0u8; size];

    let start = Instant::now();
    for _ in 0..iterations {
        dst.copy_from_slice(&src);
    }
    let elapsed = start.elapsed();

    let copy_throughput =
        (size as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;
    let copy_target = 8.0; // 8 GB/s minimum

    let status = if copy_throughput >= copy_target {
        passed += 1;
        "✓"
    } else {
        failed += 1;
        "✗"
    };
    println!(
        "  Memory copy:      {:>6.1} GB/s  (target: {:>4.0} GB/s)  {}",
        copy_throughput, copy_target, status
    );

    println!();
    (passed, failed)
}

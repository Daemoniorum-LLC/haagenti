//! Performance regression tests for haagenti-zstd.
//!
//! These tests enforce minimum throughput thresholds to prevent performance regressions.
//! Run with: `cargo test -p haagenti-zstd --release perf_tests`
//!
//! Note: These tests are ignored in debug builds because performance is much lower.
//! The thresholds are calibrated for release builds.

#![cfg(not(debug_assertions))]

use crate::{ZstdCompressor, ZstdDecompressor};
use haagenti_core::{Compressor, Decompressor};
use std::time::{Duration, Instant};

/// Minimum acceptable throughput in MiB/s for compression
/// Current baseline - conservative for CI environments
/// Will increase as optimizations are implemented
const MIN_COMPRESS_THROUGHPUT_1KB: f64 = 10.0; // Target: 50.0
const MIN_COMPRESS_THROUGHPUT_4KB: f64 = 15.0; // Target: 150.0
const MIN_COMPRESS_THROUGHPUT_16KB: f64 = 20.0; // Target: 500.0
const MIN_COMPRESS_THROUGHPUT_64KB: f64 = 15.0; // Target: 600.0

/// Minimum acceptable throughput in MiB/s for decompression
/// Current baseline - will increase as optimizations are implemented
const MIN_DECOMPRESS_THROUGHPUT_1KB: f64 = 30.0; // Target: 200.0
const MIN_DECOMPRESS_THROUGHPUT_16KB: f64 = 40.0; // Target: 500.0
const MIN_DECOMPRESS_THROUGHPUT_64KB: f64 = 50.0; // Target: 600.0

fn generate_text_data(size: usize) -> Vec<u8> {
    let words = [
        "the ",
        "quick ",
        "brown ",
        "fox ",
        "jumps ",
        "over ",
        "lazy ",
        "dog ",
        "compression ",
        "algorithm ",
        "performance ",
        "benchmark ",
        "testing ",
    ];
    let mut data = Vec::with_capacity(size);
    let mut i = 0;
    while data.len() < size {
        let word = words[i % words.len()];
        data.extend_from_slice(word.as_bytes());
        i += 1;
    }
    data.truncate(size);
    data
}

fn generate_binary_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut val: u32 = 0x12345678;
    while data.len() < size {
        data.extend_from_slice(&val.to_le_bytes());
        val = val.wrapping_mul(1103515245).wrapping_add(12345);
    }
    data.truncate(size);
    data
}

fn measure_throughput<F>(iterations: usize, data_size: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..5 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();

    let total_bytes = data_size * iterations;
    let throughput_mib = total_bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    throughput_mib
}

// ========== COMPRESSION THROUGHPUT TESTS ==========

#[test]
fn test_compress_throughput_1kb_text() {
    let compressor = ZstdCompressor::new();
    let data = generate_text_data(1024);

    let throughput = measure_throughput(1000, data.len(), || {
        let _ = compressor.compress(&data);
    });

    assert!(
        throughput >= MIN_COMPRESS_THROUGHPUT_1KB,
        "Compression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 1KB text",
        throughput,
        MIN_COMPRESS_THROUGHPUT_1KB
    );
}

#[test]
fn test_compress_throughput_4kb_text() {
    let compressor = ZstdCompressor::new();
    let data = generate_text_data(4096);

    let throughput = measure_throughput(500, data.len(), || {
        let _ = compressor.compress(&data);
    });

    assert!(
        throughput >= MIN_COMPRESS_THROUGHPUT_4KB,
        "Compression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 4KB text",
        throughput,
        MIN_COMPRESS_THROUGHPUT_4KB
    );
}

#[test]
fn test_compress_throughput_16kb_text() {
    let compressor = ZstdCompressor::new();
    let data = generate_text_data(16384);

    let throughput = measure_throughput(200, data.len(), || {
        let _ = compressor.compress(&data);
    });

    assert!(
        throughput >= MIN_COMPRESS_THROUGHPUT_16KB,
        "Compression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 16KB text",
        throughput,
        MIN_COMPRESS_THROUGHPUT_16KB
    );
}

#[test]
fn test_compress_throughput_64kb_binary() {
    let compressor = ZstdCompressor::new();
    let data = generate_binary_data(65536);

    let throughput = measure_throughput(100, data.len(), || {
        let _ = compressor.compress(&data);
    });

    assert!(
        throughput >= MIN_COMPRESS_THROUGHPUT_64KB,
        "Compression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 64KB binary",
        throughput,
        MIN_COMPRESS_THROUGHPUT_64KB
    );
}

// ========== DECOMPRESSION THROUGHPUT TESTS ==========

#[test]
fn test_decompress_throughput_1kb() {
    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();
    let data = generate_text_data(1024);
    let compressed = compressor.compress(&data).unwrap();

    let throughput = measure_throughput(2000, data.len(), || {
        let _ = decompressor.decompress(&compressed);
    });

    assert!(
        throughput >= MIN_DECOMPRESS_THROUGHPUT_1KB,
        "Decompression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 1KB",
        throughput,
        MIN_DECOMPRESS_THROUGHPUT_1KB
    );
}

#[test]
fn test_decompress_throughput_16kb() {
    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();
    let data = generate_text_data(16384);
    let compressed = compressor.compress(&data).unwrap();

    let throughput = measure_throughput(500, data.len(), || {
        let _ = decompressor.decompress(&compressed);
    });

    assert!(
        throughput >= MIN_DECOMPRESS_THROUGHPUT_16KB,
        "Decompression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 16KB",
        throughput,
        MIN_DECOMPRESS_THROUGHPUT_16KB
    );
}

#[test]
fn test_decompress_throughput_64kb() {
    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();
    let data = generate_binary_data(65536);
    let compressed = compressor.compress(&data).unwrap();

    let throughput = measure_throughput(200, data.len(), || {
        let _ = decompressor.decompress(&compressed);
    });

    assert!(
        throughput >= MIN_DECOMPRESS_THROUGHPUT_64KB,
        "Decompression throughput {:.1} MiB/s below minimum {:.1} MiB/s for 64KB",
        throughput,
        MIN_DECOMPRESS_THROUGHPUT_64KB
    );
}

// ========== COMPONENT-LEVEL PERFORMANCE TESTS ==========

#[test]
fn test_huffman_encode_throughput() {
    use crate::huffman::HuffmanEncoder;

    // Build encoder with typical text distribution
    let sample = generate_text_data(4096);
    let encoder = HuffmanEncoder::build(&sample).unwrap();

    let data = generate_text_data(65536);
    let throughput = measure_throughput(100, data.len(), || {
        let _ = encoder.encode(&data);
    });

    // Huffman encoding should be at least 30 MiB/s (conservative for CI)
    assert!(
        throughput >= 30.0,
        "Huffman encode throughput {:.1} MiB/s below minimum 30 MiB/s",
        throughput
    );
}

#[test]
fn test_match_finder_throughput() {
    use crate::compress::MatchFinder;

    let data = generate_text_data(65536);
    let mut finder = MatchFinder::new(6);

    let throughput = measure_throughput(50, data.len(), || {
        // find_matches internally resets the finder
        let _ = finder.find_matches(&data);
    });

    // Match finder should be at least 20 MiB/s (conservative for CI)
    assert!(
        throughput >= 20.0,
        "Match finder throughput {:.1} MiB/s below minimum 20 MiB/s",
        throughput
    );
}

// ========== REGRESSION PREVENTION TESTS ==========

#[test]
fn test_compression_ratio_text() {
    let compressor = ZstdCompressor::new();
    // Use smaller data that fits in single block with good compression
    let data = generate_text_data(8192);
    let compressed = compressor.compress(&data).unwrap();

    let ratio = data.len() as f64 / compressed.len() as f64;

    // Should achieve at least 1.5x compression on text (conservative baseline)
    // Target: 3.0x+ as Huffman encoding improves
    assert!(
        ratio >= 1.5,
        "Compression ratio {:.2}x below minimum 1.5x for text data",
        ratio
    );
}

#[test]
fn test_no_compression_regression_binary() {
    let compressor = ZstdCompressor::new();
    let data = generate_binary_data(65536);
    let compressed = compressor.compress(&data).unwrap();

    // Binary data should not expand significantly
    assert!(
        compressed.len() <= data.len() + 100,
        "Binary data expanded too much: {} -> {} bytes",
        data.len(),
        compressed.len()
    );
}

// ========== LATENCY TESTS ==========

#[test]
fn test_small_input_latency() {
    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();
    let data = b"Hello, World!";

    let start = Instant::now();
    for _ in 0..10000 {
        let compressed = compressor.compress(data).unwrap();
        let _ = decompressor.decompress(&compressed).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_latency = elapsed / 10000;

    // Small input roundtrip should be < 200µs (conservative for CI)
    assert!(
        avg_latency < Duration::from_micros(200),
        "Small input latency {:?} exceeds 200µs threshold",
        avg_latency
    );
}

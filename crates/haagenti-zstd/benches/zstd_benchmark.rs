//! Comprehensive Zstd Benchmarks
//!
//! Compares haagenti-zstd native Rust implementation against
//! the reference zstd C library.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use haagenti_core::{CompressionLevel, Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

// ============================================================================
// Test Data Generators
// ============================================================================

fn generate_text_data(size: usize) -> Vec<u8> {
    // Generate repeating text pattern (known to work with haagenti roundtrip)
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut result = Vec::with_capacity(size);
    while result.len() < size {
        result.extend_from_slice(pattern);
    }
    result.truncate(size);
    result
}

fn generate_binary_data(size: usize) -> Vec<u8> {
    // Generate repeating binary pattern
    let pattern: Vec<u8> = (0..=255).collect();
    let mut result = Vec::with_capacity(size);
    while result.len() < size {
        result.extend_from_slice(&pattern);
    }
    result.truncate(size);
    result
}

fn generate_llm_weights(size: usize) -> Vec<u8> {
    // Simulated LLM weight patterns (mostly near-zero f16 values)
    // Use a deterministic pattern that works with haagenti
    let mut data = Vec::with_capacity(size);

    // Generate a repeating pattern of f16 values near zero
    let values: [f32; 16] = [
        0.0, 0.001, -0.001, 0.01, -0.01, 0.1, -0.1, 0.5, -0.5, 0.001, 0.002, -0.002, 0.0, 0.0,
        0.003, -0.003,
    ];

    while data.len() < size {
        for &v in &values {
            let f16_bits = half::f16::from_f32(v).to_bits();
            data.extend_from_slice(&f16_bits.to_le_bytes());
            if data.len() >= size {
                break;
            }
        }
    }
    data.truncate(size);
    data
}

fn generate_highly_compressible(size: usize) -> Vec<u8> {
    // Highly repetitive data
    let pattern = b"AAAAAAAAAAAAAAAA";
    pattern.iter().cycle().take(size).cloned().collect()
}

// ============================================================================
// Compression Benchmarks
// ============================================================================

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_compression");

    let sizes = [1024, 4096, 16384, 65536, 262144]; // 1KB to 256KB

    for size in sizes {
        let text_data = generate_text_data(size);
        let binary_data = generate_binary_data(size);
        let llm_data = generate_llm_weights(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Haagenti compression
        group.bench_with_input(
            BenchmarkId::new("haagenti/text", size),
            &text_data,
            |b, data| {
                let compressor = ZstdCompressor::new();
                b.iter(|| compressor.compress(black_box(data)).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("haagenti/binary", size),
            &binary_data,
            |b, data| {
                let compressor = ZstdCompressor::new();
                b.iter(|| compressor.compress(black_box(data)).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("haagenti/llm_weights", size),
            &llm_data,
            |b, data| {
                let compressor = ZstdCompressor::new();
                b.iter(|| compressor.compress(black_box(data)).unwrap())
            },
        );

        // Reference zstd (C library)
        group.bench_with_input(
            BenchmarkId::new("zstd_ref/text", size),
            &text_data,
            |b, data| b.iter(|| zstd::encode_all(black_box(data.as_slice()), 3).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("zstd_ref/binary", size),
            &binary_data,
            |b, data| b.iter(|| zstd::encode_all(black_box(data.as_slice()), 3).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("zstd_ref/llm_weights", size),
            &llm_data,
            |b, data| b.iter(|| zstd::encode_all(black_box(data.as_slice()), 3).unwrap()),
        );
    }

    group.finish();
}

fn bench_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_decompression");

    let sizes = [1024, 4096, 16384, 65536, 262144];

    for size in sizes {
        let text_data = generate_text_data(size);
        let llm_data = generate_llm_weights(size);

        // Compress with haagenti (for haagenti decompression benchmark)
        let haagenti_compressor = ZstdCompressor::new();
        let haagenti_text = haagenti_compressor.compress(&text_data).unwrap();
        let haagenti_llm = haagenti_compressor.compress(&llm_data).unwrap();

        // Compress with reference zstd (for zstd decompression benchmark)
        let zstd_text = zstd::encode_all(text_data.as_slice(), 3).unwrap();
        let zstd_llm = zstd::encode_all(llm_data.as_slice(), 3).unwrap();

        group.throughput(Throughput::Bytes(size as u64));

        // Haagenti decompression (of haagenti-compressed data)
        group.bench_with_input(
            BenchmarkId::new("haagenti/text", size),
            &haagenti_text,
            |b, data| {
                let decompressor = ZstdDecompressor::new();
                b.iter(|| decompressor.decompress(black_box(data)).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("haagenti/llm_weights", size),
            &haagenti_llm,
            |b, data| {
                let decompressor = ZstdDecompressor::new();
                b.iter(|| decompressor.decompress(black_box(data)).unwrap())
            },
        );

        // Reference zstd decompression (of zstd-compressed data)
        group.bench_with_input(
            BenchmarkId::new("zstd_ref/text", size),
            &zstd_text,
            |b, data| b.iter(|| zstd::decode_all(black_box(data.as_slice())).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("zstd_ref/llm_weights", size),
            &zstd_llm,
            |b, data| b.iter(|| zstd::decode_all(black_box(data.as_slice())).unwrap()),
        );
    }

    group.finish();
}

// ============================================================================
// Compression Ratio Benchmarks
// ============================================================================

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_ratio");
    group.sample_size(10); // Fewer samples for ratio test

    let size = 65536; // 64KB for ratio tests

    let text_data = generate_text_data(size);
    let binary_data = generate_binary_data(size);
    let llm_data = generate_llm_weights(size);
    let highly_compressible = generate_highly_compressible(size);

    let datasets = [
        ("text", text_data),
        ("binary", binary_data),
        ("llm_weights", llm_data),
        ("highly_repetitive", highly_compressible),
    ];

    for (name, data) in &datasets {
        let compressor = ZstdCompressor::new();

        group.bench_function(BenchmarkId::new("haagenti", *name), |b| {
            b.iter(|| {
                let compressed = compressor.compress(black_box(data)).unwrap();
                let ratio = data.len() as f64 / compressed.len() as f64;
                black_box(ratio)
            })
        });

        group.bench_function(BenchmarkId::new("zstd_ref", *name), |b| {
            b.iter(|| {
                let compressed = zstd::encode_all(black_box(data.as_slice()), 3).unwrap();
                let ratio = data.len() as f64 / compressed.len() as f64;
                black_box(ratio)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Roundtrip Benchmarks
// ============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_roundtrip");

    let sizes = [4096, 65536, 262144]; // 4KB, 64KB, 256KB

    for size in sizes {
        let data = generate_llm_weights(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Haagenti roundtrip
        group.bench_with_input(BenchmarkId::new("haagenti", size), &data, |b, data| {
            let compressor = ZstdCompressor::new();
            let decompressor = ZstdDecompressor::new();
            b.iter(|| {
                let compressed = compressor.compress(black_box(data)).unwrap();
                decompressor.decompress(&compressed).unwrap()
            })
        });

        // Reference zstd roundtrip
        group.bench_with_input(BenchmarkId::new("zstd_ref", size), &data, |b, data| {
            b.iter(|| {
                let compressed = zstd::encode_all(black_box(data.as_slice()), 3).unwrap();
                zstd::decode_all(compressed.as_slice()).unwrap()
            })
        });
    }

    group.finish();
}

// ============================================================================
// Compression Level Benchmarks
// ============================================================================

fn bench_compression_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_levels");

    let data = generate_llm_weights(65536); // 64KB

    let levels = [
        ("fast", CompressionLevel::Fast),
        ("default", CompressionLevel::Default),
        ("best", CompressionLevel::Best),
    ];

    for (name, level) in &levels {
        group.bench_function(BenchmarkId::new("haagenti", *name), |b| {
            let compressor = ZstdCompressor::with_level(*level);
            b.iter(|| compressor.compress(black_box(&data)).unwrap())
        });
    }

    // Reference zstd levels
    for (name, zstd_level) in [("fast", 1), ("default", 3), ("best", 19)] {
        group.bench_function(BenchmarkId::new("zstd_ref", name), |b| {
            b.iter(|| zstd::encode_all(black_box(data.as_slice()), zstd_level).unwrap())
        });
    }

    group.finish();
}

// ============================================================================
// Dictionary Compression Benchmarks
// ============================================================================

fn bench_dictionary_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_dictionary");

    // Generate similar samples for dictionary training
    let samples: Vec<Vec<u8>> = (0..10)
        .map(|i| {
            let mut data = Vec::with_capacity(1024);
            data.extend_from_slice(b"common_prefix_");
            data.extend_from_slice(format!("sample_{}_data_", i).as_bytes());
            data.extend_from_slice(&generate_text_data(1000));
            data.extend_from_slice(b"common_suffix");
            data
        })
        .collect();

    let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();

    group.bench_function("train_dict_1kb", |b| {
        b.iter(|| haagenti_zstd::ZstdDictionary::train(black_box(&sample_refs), 1024).unwrap())
    });

    group.bench_function("train_dict_16kb", |b| {
        b.iter(|| haagenti_zstd::ZstdDictionary::train(black_box(&sample_refs), 16384).unwrap())
    });

    // Test dictionary compression vs regular
    let dict = haagenti_zstd::ZstdDictionary::train(&sample_refs, 8192).unwrap();
    let test_data = samples[0].clone();

    group.bench_function("compress_with_dict", |b| {
        let compressor = haagenti_zstd::ZstdDictCompressor::new(dict.clone());
        b.iter(|| compressor.compress(black_box(&test_data)).unwrap())
    });

    group.bench_function("compress_without_dict", |b| {
        let compressor = ZstdCompressor::new();
        b.iter(|| compressor.compress(black_box(&test_data)).unwrap())
    });

    group.finish();
}

fn bench_dictionary_match_finding(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_match_finding");

    let dict_content: Vec<u8> = (0..32768)
        .map(|i| ((i % 256) as u8).wrapping_add((i / 256) as u8))
        .collect();
    let dict = haagenti_zstd::ZstdDictionary::from_content(dict_content).unwrap();

    let test_input = generate_text_data(4096);

    group.bench_function("find_matches_4kb", |b| {
        b.iter(|| {
            let mut matches = 0;
            for pos in 0..test_input.len().saturating_sub(4) {
                if dict.find_match(black_box(&test_input), pos).is_some() {
                    matches += 1;
                }
            }
            matches
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compression,
    bench_decompression,
    bench_compression_ratio,
    bench_roundtrip,
    bench_compression_levels,
    bench_dictionary_training,
    bench_dictionary_match_finding,
);

criterion_main!(benches);

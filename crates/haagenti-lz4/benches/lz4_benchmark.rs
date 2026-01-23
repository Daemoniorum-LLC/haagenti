//! Benchmarks for LZ4 and LZ4-HC compression.
//!
//! Run with: `cargo bench -p haagenti-lz4 --features hc`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use haagenti_lz4::block::{compress_block, decompress_block, max_compressed_size};
use haagenti_lz4::Lz4Codec;

#[cfg(feature = "hc")]
use haagenti_lz4::hc::{compress_hc, Lz4HcCompressor};

/// Generate test data with varying compressibility.
fn generate_test_data(size: usize, compressibility: f64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(size);

    if compressibility > 0.9 {
        // Highly repetitive data
        let pattern = b"ABCDEFGHIJKLMNOP";
        while data.len() < size {
            data.extend_from_slice(pattern);
        }
        data.truncate(size);
    } else if compressibility > 0.5 {
        // Mixed data - some patterns, some randomness
        let phrases: &[&[u8]] = &[
            b"The quick brown fox jumps over the lazy dog. ",
            b"Pack my box with five dozen liquor jugs! ",
            b"How vexingly quick daft zebras jump!! ",
        ];
        while data.len() < size {
            if rng.gen_bool(compressibility) {
                let phrase = phrases[rng.gen_range(0..phrases.len())];
                data.extend_from_slice(phrase);
            } else {
                data.push(rng.r#gen::<u8>());
            }
        }
        data.truncate(size);
    } else {
        // Random/incompressible data
        data.resize(size, 0);
        rng.fill(&mut data[..]);
    }

    data
}

/// Generate data that simulates neural network weights (near-zero distribution).
#[allow(dead_code)]
fn generate_weight_like_data(size: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(size);

    // Simulate bf16 weights: mostly small values, occasional larger ones
    while data.len() < size {
        let value: f32 = if rng.gen_bool(0.7) {
            // 70% near-zero
            rng.gen_range(-0.1..0.1)
        } else if rng.gen_bool(0.8) {
            // 24% moderate
            rng.gen_range(-1.0..1.0)
        } else {
            // 6% larger
            rng.gen_range(-5.0..5.0)
        };
        data.extend_from_slice(&value.to_le_bytes());
    }
    data.truncate(size);
    data
}

fn bench_standard_lz4_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_standard_compress");

    for size in [4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024] {
        let data = generate_test_data(size, 0.7);
        let mut output = vec![0u8; max_compressed_size(size)];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &data,
            |b, data| b.iter(|| compress_block(black_box(data), black_box(&mut output)).unwrap()),
        );
    }

    group.finish();
}

fn bench_standard_lz4_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_standard_decompress");

    for size in [4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024] {
        let data = generate_test_data(size, 0.7);
        let mut compressed = vec![0u8; max_compressed_size(size)];
        let compressed_len = compress_block(&data, &mut compressed).unwrap();
        compressed.truncate(compressed_len);

        let mut decompressed = vec![0u8; size];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    decompress_block(black_box(compressed), black_box(&mut decompressed), size)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "hc")]
fn bench_hc_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_hc_compress");

    let size = 64 * 1024; // 64KB
    let data = generate_test_data(size, 0.7);

    for level in [1, 4, 7, 9] {
        let mut output = vec![0u8; max_compressed_size(size)];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("level_{}", level)),
            &data,
            |b, data| {
                b.iter(|| compress_hc(black_box(data), black_box(&mut output), level).unwrap())
            },
        );
    }

    group.finish();
}

#[cfg(feature = "hc")]
fn bench_hc_vs_standard_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_hc_vs_standard");

    let sizes = [16 * 1024, 64 * 1024, 256 * 1024];

    for size in sizes {
        let data = generate_test_data(size, 0.8);

        // Standard LZ4
        let mut std_output = vec![0u8; max_compressed_size(size)];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("standard", format!("{}KB", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| compress_block(black_box(data), black_box(&mut std_output)).unwrap())
            },
        );

        // HC level 9
        let mut hc_output = vec![0u8; max_compressed_size(size)];
        group.bench_with_input(
            BenchmarkId::new("hc_level9", format!("{}KB", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| compress_hc(black_box(data), black_box(&mut hc_output), 9).unwrap())
            },
        );
    }

    group.finish();
}

#[cfg(feature = "hc")]
fn bench_hc_weights_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_hc_weights");

    let size = 256 * 1024; // 256KB of simulated weight data
    let data = generate_weight_like_data(size);

    for level in [1, 4, 9] {
        let mut output = vec![0u8; max_compressed_size(size)];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("level_{}", level)),
            &data,
            |b, data| {
                b.iter(|| compress_hc(black_box(data), black_box(&mut output), level).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_vs_lz4_flex(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_vs_lz4flex");

    let size = 64 * 1024;
    let data = generate_test_data(size, 0.7);

    // Our standard LZ4
    let mut our_output = vec![0u8; max_compressed_size(size)];
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_with_input(
        BenchmarkId::new("haagenti", "compress"),
        &data,
        |b, data| b.iter(|| compress_block(black_box(data), black_box(&mut our_output)).unwrap()),
    );

    // lz4_flex
    group.bench_with_input(
        BenchmarkId::new("lz4_flex", "compress"),
        &data,
        |b, data| b.iter(|| lz4_flex::compress(black_box(data))),
    );

    // Decompress comparison
    let compressed = lz4_flex::compress(&data);
    let mut decompressed = vec![0u8; size];

    group.bench_with_input(
        BenchmarkId::new("haagenti", "decompress"),
        &compressed,
        |b, compressed| {
            b.iter(|| {
                decompress_block(black_box(compressed), black_box(&mut decompressed), size).unwrap()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("lz4_flex", "decompress"),
        &compressed,
        |b, compressed| b.iter(|| lz4_flex::decompress(black_box(compressed), size).unwrap()),
    );

    group.finish();
}

fn bench_codec_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_codec_roundtrip");

    let codec = Lz4Codec::new();

    for size in [4 * 1024, 64 * 1024, 256 * 1024] {
        let data = generate_test_data(size, 0.7);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    let (compressed, original_size) =
                        codec.compress_with_size(black_box(data)).unwrap();
                    codec
                        .decompress_sized(black_box(&compressed), original_size)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "hc")]
fn bench_hc_compressor_api(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_hc_compressor_api");

    let size = 64 * 1024;
    let data = generate_test_data(size, 0.7);

    for level in [1, 4, 9] {
        let compressor = Lz4HcCompressor::new(level);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("level_{}", level)),
            &data,
            |b, data| b.iter(|| compressor.compress(black_box(data)).unwrap()),
        );
    }

    group.finish();
}

#[cfg(feature = "hc")]
criterion_group!(
    benches,
    bench_standard_lz4_compress,
    bench_standard_lz4_decompress,
    bench_hc_compress,
    bench_hc_vs_standard_ratio,
    bench_hc_weights_data,
    bench_vs_lz4_flex,
    bench_codec_roundtrip,
    bench_hc_compressor_api,
);

#[cfg(not(feature = "hc"))]
criterion_group!(
    benches,
    bench_standard_lz4_compress,
    bench_standard_lz4_decompress,
    bench_vs_lz4_flex,
    bench_codec_roundtrip,
);

criterion_main!(benches);

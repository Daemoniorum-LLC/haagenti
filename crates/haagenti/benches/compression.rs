//! Comprehensive Haagenti Compression Benchmarks
//!
//! Benchmarks cover:
//! - DCT primitives (1D, 2D)
//! - Spectral/Holographic encoding
//! - Compressive spectral (retention-based)
//! - Adaptive spectral (energy-based)
//! - Mixed precision (FP16+INT4)
//! - Importance-guided compression
//! - Memory usage and throughput

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use haagenti::adaptive::AdaptiveSpectralEncoder;
use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::holotensor::{dct_1d, dct_2d, idct_1d, idct_2d, SpectralDecoder, SpectralEncoder};
use haagenti::importance::{ImportanceGuidedEncoder, ImportanceMap};
use haagenti::mixed_precision::{MixedPrecisionDecoder, MixedPrecisionEncoder};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn generate_llm_weights_f32(size: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();
    (0..size).map(|_| normal.sample(&mut rng) as f32).collect()
}

fn generate_attention_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut result = vec![0.0f32; rows * cols];
    let rank = 10;
    let scale_factor = 0.5_f32;
    for r in 0..rank {
        let row_vec: Vec<f32> = (0..rows)
            .map(|_| Rng::r#gen::<f32>(&mut rng) - scale_factor)
            .collect();
        let col_vec: Vec<f32> = (0..cols)
            .map(|_| Rng::r#gen::<f32>(&mut rng) - scale_factor)
            .collect();
        let scale = 1.0 / (r as f32 + 1.0).sqrt();
        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] += row_vec[i] * col_vec[j] * scale;
            }
        }
    }
    result
}

fn bench_dct_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_primitives");
    let sizes = [64, 256, 1024, 4096];
    for size in sizes {
        let data = generate_llm_weights_f32(size);
        let mut output = vec![0.0f32; size];
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dct_1d", size), &data, |b, data| {
            b.iter(|| dct_1d(black_box(data), black_box(&mut output)))
        });

        group.bench_with_input(BenchmarkId::new("idct_1d", size), &data, |b, data| {
            b.iter(|| idct_1d(black_box(data), black_box(&mut output)))
        });
    }

    let dims = [(64, 64), (128, 128), (256, 256)];
    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        let mut output = vec![0.0f32; size];
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("dct_2d", format!("{}x{}", width, height)),
            &data,
            |b, data| b.iter(|| dct_2d(black_box(data), black_box(&mut output), width, height)),
        );

        group.bench_with_input(
            BenchmarkId::new("idct_2d", format!("{}x{}", width, height)),
            &data,
            |b, data| b.iter(|| idct_2d(black_box(data), black_box(&mut output), width, height)),
        );
    }
    group.finish();
}

fn bench_spectral_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_holographic");
    let dims = [(64, 64), (128, 128), (256, 256)];
    let fragment_counts = [4, 8, 16];

    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        group.throughput(Throughput::Elements(size as u64));

        for num_fragments in fragment_counts {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("encode/{}_fragments", num_fragments),
                    format!("{}x{}", width, height),
                ),
                &data,
                |b, data| {
                    let encoder = SpectralEncoder::new(num_fragments);
                    b.iter(|| encoder.encode_2d(black_box(data), width, height).unwrap())
                },
            );
        }
    }

    let (width, height) = (128, 128);
    let data = generate_attention_matrix(height, width);
    for num_fragments in fragment_counts {
        let encoder = SpectralEncoder::new(num_fragments);
        let fragments = encoder.encode_2d(&data, width, height).unwrap();

        group.bench_function(
            BenchmarkId::new(
                format!("decode_full/{}_fragments", num_fragments),
                format!("{}x{}", width, height),
            ),
            |b| {
                b.iter(|| {
                    let mut decoder = SpectralDecoder::new(width, height, num_fragments);
                    for frag in &fragments {
                        decoder.add_fragment(black_box(frag)).unwrap();
                    }
                    decoder.reconstruct()
                })
            },
        );
    }
    group.finish();
}

fn bench_reconstruction_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("holotensor_quality");
    group.sample_size(10);
    let data = generate_attention_matrix(128, 128);

    for num_fragments in [2, 4, 8, 16] {
        let encoder = SpectralEncoder::new(num_fragments);
        let fragments = encoder.encode_2d(&data, 128, 128).unwrap();

        group.bench_function(BenchmarkId::new("spectral_mse", num_fragments), |b| {
            b.iter(|| {
                let mut decoder = SpectralDecoder::new(128, 128, num_fragments);
                for frag in &fragments {
                    decoder.add_fragment(frag).unwrap();
                }
                let reconstructed = decoder.reconstruct();
                let mse: f32 = data
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / data.len() as f32;
                black_box(mse)
            })
        });
    }
    group.finish();
}

/// Benchmark compressive spectral encoder at various retention levels
fn bench_compressive_spectral(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressive_spectral");
    let dims = [(64, 64), (128, 128), (256, 256)];
    let retentions = [0.30, 0.50, 0.70, 0.90];

    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        group.throughput(Throughput::Elements(size as u64));

        for retention in retentions {
            let label = format!("{}x{}/{}%", width, height, (retention * 100.0) as i32);

            group.bench_function(BenchmarkId::new("encode", &label), |b| {
                let encoder = CompressiveSpectralEncoder::new(8, retention);
                b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
            });

            // Also benchmark decode
            let encoder = CompressiveSpectralEncoder::new(8, retention);
            let fragments = encoder.encode_2d(&data, width, height).unwrap();

            group.bench_function(BenchmarkId::new("decode", &label), |b| {
                b.iter(|| {
                    let mut decoder = CompressiveSpectralDecoder::new();
                    decoder.add_essentials(black_box(&fragments[0])).unwrap();
                    for frag in &fragments[1..] {
                        decoder.add_detail(black_box(frag)).unwrap();
                    }
                    decoder.reconstruct().unwrap()
                })
            });
        }
    }
    group.finish();
}

/// Benchmark adaptive spectral encoder
fn bench_adaptive_spectral(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_spectral");
    let dims = [(64, 64), (128, 128)];
    let target_qualities = [0.85, 0.90, 0.95];

    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        group.throughput(Throughput::Elements(size as u64));

        for target in target_qualities {
            let label = format!("{}x{}/q{}", width, height, (target * 100.0) as i32);

            group.bench_function(BenchmarkId::new("encode", &label), |b| {
                let encoder = AdaptiveSpectralEncoder::new(target, 8);
                b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
            });
        }
    }
    group.finish();
}

/// Benchmark mixed precision encoder (FP16 essentials + INT4 details)
fn bench_mixed_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision");
    // Use smaller sizes for mixed precision due to O(n^2) DCT
    let dims = [(32, 32), (64, 64), (128, 128)];
    let fp16_ratios = [0.10, 0.20, 0.30];

    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        group.throughput(Throughput::Elements(size as u64));

        for fp16_ratio in fp16_ratios {
            let label = format!("{}x{}/fp16_{}", width, height, (fp16_ratio * 100.0) as i32);

            group.bench_function(BenchmarkId::new("encode", &label), |b| {
                let encoder = MixedPrecisionEncoder::new(0.70, fp16_ratio);
                b.iter(|| encoder.encode(black_box(&data), width, height).unwrap())
            });

            // Benchmark decode
            let encoder = MixedPrecisionEncoder::new(0.70, fp16_ratio);
            let compressed = encoder.encode(&data, width, height).unwrap();

            group.bench_function(BenchmarkId::new("decode", &label), |b| {
                let decoder = MixedPrecisionDecoder::new();
                b.iter(|| decoder.decode(black_box(&compressed)).unwrap())
            });
        }
    }
    group.finish();
}

/// Benchmark importance-guided compression
fn bench_importance_guided(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance_guided");
    let dims = [(64, 64), (128, 128)];
    // Use unique short names for benchmark IDs
    let tensor_configs = [
        ("model.layers.0.mlp.gate_proj.weight", "mlp_gate"),
        ("model.layers.0.self_attn.q_proj.weight", "attn_q"),
        ("model.layers.0.self_attn.v_proj.weight", "attn_v"),
        ("model.embed_tokens.weight", "embed"),
    ];

    let importance_map = ImportanceMap::heuristic_only();
    let encoder = ImportanceGuidedEncoder::new(0.50, importance_map);

    for (width, height) in dims {
        let size = width * height;
        let data = generate_attention_matrix(height, width);
        group.throughput(Throughput::Elements(size as u64));

        for (tensor_name, short_name) in tensor_configs {
            let label = format!("{}x{}/{}", width, height, short_name);

            group.bench_function(BenchmarkId::new("encode", &label), |b| {
                b.iter(|| {
                    encoder
                        .encode(black_box(&data), width, height, tensor_name)
                        .unwrap()
                })
            });
        }
    }
    group.finish();
}

/// Benchmark throughput at various tensor sizes (simulating real model weights)
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(10); // Fewer samples for large tensors

    // Sizes representative of real model tensors
    let sizes = [
        (256, 256),   // 64K - small attention
        (512, 512),   // 256K - medium
        (1024, 512),  // 512K - typical MLP
        (1024, 1024), // 1M - large attention
    ];

    for (width, height) in sizes {
        let size = width * height;
        let data = generate_llm_weights_f32(size);
        group.throughput(Throughput::Bytes((size * 4) as u64)); // F32 = 4 bytes

        let label = format!("{}x{}", width, height);

        // Compressive at 70% retention (good balance)
        group.bench_function(BenchmarkId::new("compressive_70", &label), |b| {
            let encoder = CompressiveSpectralEncoder::new(8, 0.70);
            b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
        });
    }
    group.finish();
}

/// Benchmark encoder comparison at fixed size
fn bench_encoder_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_comparison");
    let (width, height) = (128, 128);
    let size = width * height;
    let data = generate_attention_matrix(height, width);
    group.throughput(Throughput::Elements(size as u64));

    // Spectral (holographic)
    group.bench_function("spectral_8frag", |b| {
        let encoder = SpectralEncoder::new(8);
        b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
    });

    // Compressive at 70%
    group.bench_function("compressive_70", |b| {
        let encoder = CompressiveSpectralEncoder::new(8, 0.70);
        b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
    });

    // Adaptive at 90% quality
    group.bench_function("adaptive_q90", |b| {
        let encoder = AdaptiveSpectralEncoder::new(0.90, 8);
        b.iter(|| encoder.encode_2d(black_box(&data), width, height).unwrap())
    });

    // Mixed precision
    group.bench_function("mixed_fp16_20", |b| {
        let encoder = MixedPrecisionEncoder::new(0.70, 0.20);
        b.iter(|| encoder.encode(black_box(&data), width, height).unwrap())
    });

    // Importance-guided
    group.bench_function("importance_heuristic", |b| {
        let importance_map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, importance_map);
        b.iter(|| {
            encoder
                .encode(
                    black_box(&data),
                    width,
                    height,
                    "model.layers.0.mlp.gate_proj.weight",
                )
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dct_primitives,
    bench_spectral_encoding,
    bench_reconstruction_quality,
    bench_compressive_spectral,
    bench_adaptive_spectral,
    bench_mixed_precision,
    bench_importance_guided,
    bench_throughput_scaling,
    bench_encoder_comparison,
);
criterion_main!(benches);

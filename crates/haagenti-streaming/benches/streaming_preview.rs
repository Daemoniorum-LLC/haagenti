//! Benchmarks for streaming preview generation
//!
//! Tests preview generation performance across different quality levels
//! and buffer configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use haagenti_streaming::{PreviewBuffer, PreviewConfig, PreviewData, PreviewFrame, PreviewQuality};
use std::time::Instant;

/// Generate a synthetic preview frame for benchmarking
fn create_test_frame(step: u32, total_steps: u32, quality: PreviewQuality) -> PreviewFrame {
    let (width, height) = quality.resolution();
    let pixel_count = (width * height) as usize;

    // Generate synthetic RGBA data
    let data = vec![128u8; pixel_count * 4];

    PreviewFrame {
        step,
        total_steps,
        data: PreviewData::Rgba(data),
        width,
        height,
        timestamp: Instant::now(),
        progress: step as f32 / total_steps as f32,
        is_final: step == total_steps,
        quality,
        decode_time_ms: quality.decode_time_ms(),
    }
}

/// Benchmark preview frame creation at different quality levels
fn bench_preview_frame_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("preview_frame_creation");

    for quality in [
        PreviewQuality::Thumbnail,
        PreviewQuality::Low,
        PreviewQuality::Medium,
        PreviewQuality::Full,
    ] {
        let (w, h) = quality.resolution();
        group.bench_with_input(
            BenchmarkId::new("quality", format!("{}x{}", w, h)),
            &quality,
            |b, quality| {
                b.iter(|| create_test_frame(black_box(10), black_box(25), *quality));
            },
        );
    }

    group.finish();
}

/// Benchmark preview buffer operations
fn bench_preview_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("preview_buffer");

    // Benchmark adding frames to buffer
    group.bench_function("add_frame", |b| {
        let mut buffer = PreviewBuffer::new(16);
        let frame = create_test_frame(5, 25, PreviewQuality::Medium);
        b.iter(|| {
            buffer.add(black_box(frame.clone()));
        });
    });

    // Benchmark buffer iteration
    group.bench_function("iterate_8_frames", |b| {
        let mut buffer = PreviewBuffer::new(16);
        for i in 0..8 {
            buffer.add(create_test_frame(i, 25, PreviewQuality::Medium));
        }
        b.iter(|| {
            let count: usize = buffer.iter().count();
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark memory allocation for different quality levels
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for quality in [
        PreviewQuality::Thumbnail,
        PreviewQuality::Low,
        PreviewQuality::Medium,
        PreviewQuality::Full,
    ] {
        let (w, h) = quality.resolution();
        let size = quality.memory_size();

        group.bench_with_input(
            BenchmarkId::new("alloc_rgba", format!("{}x{}_{}KB", w, h, size / 1024)),
            &size,
            |b, size| {
                b.iter(|| {
                    let data: Vec<u8> = vec![0u8; *size];
                    black_box(data)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark quality score calculation
fn bench_quality_score(c: &mut Criterion) {
    c.bench_function("quality_score_calculation", |b| {
        let frame = create_test_frame(15, 25, PreviewQuality::Medium);
        b.iter(|| {
            // Quality score based on step progress and quality level
            let progress_factor = frame.progress;
            let quality_factor = match frame.quality {
                PreviewQuality::Thumbnail => 0.25,
                PreviewQuality::Low => 0.5,
                PreviewQuality::Medium => 0.75,
                PreviewQuality::Full => 1.0,
            };
            black_box(progress_factor * quality_factor)
        });
    });
}

/// Benchmark preview config creation
fn bench_preview_config(c: &mut Criterion) {
    c.bench_function("preview_config_default", |b| {
        b.iter(|| {
            let config = PreviewConfig::default();
            black_box(config)
        });
    });
}

criterion_group!(
    benches,
    bench_preview_frame_creation,
    bench_preview_buffer,
    bench_memory_allocation,
    bench_quality_score,
    bench_preview_config,
);

criterion_main!(benches);

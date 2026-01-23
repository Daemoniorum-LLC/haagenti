//! Benchmarks for streaming throughput
//!
//! Tests the throughput of streaming operations including
//! scheduler modes and protocol message handling.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use haagenti_streaming::{
    MessageType, PreviewConfig, PreviewQuality, PreviewScheduler, ScheduleMode, ScheduleStats,
    StreamMessage, StreamProtocol,
};

/// Benchmark scheduler performance with different modes
fn bench_scheduler_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_modes");

    // Every step mode
    group.bench_function("every_step_25", |b| {
        b.iter(|| {
            let scheduler = PreviewScheduler::new(ScheduleMode::EveryStep, 25);
            let mut previews = 0;
            for step in 0..25 {
                if scheduler.should_preview(black_box(step)) {
                    previews += 1;
                }
            }
            black_box(previews)
        });
    });

    // Interval mode
    for interval in [2, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("interval", interval),
            &interval,
            |b, interval| {
                b.iter(|| {
                    let scheduler =
                        PreviewScheduler::new(ScheduleMode::Interval { steps: *interval }, 25);
                    let mut previews = 0;
                    for step in 0..25 {
                        if scheduler.should_preview(black_box(step)) {
                            previews += 1;
                        }
                    }
                    black_box(previews)
                });
            },
        );
    }

    // Quality-based mode
    group.bench_function("quality_adaptive", |b| {
        b.iter(|| {
            let scheduler = PreviewScheduler::new(
                ScheduleMode::Quality {
                    min_quality: 0.3,
                    max_quality: 0.9,
                },
                25,
            );
            let mut previews = 0;
            for step in 0..25 {
                if scheduler.should_preview(black_box(step)) {
                    previews += 1;
                }
            }
            black_box(previews)
        });
    });

    group.finish();
}

/// Benchmark schedule stats calculation
fn bench_schedule_stats(c: &mut Criterion) {
    c.bench_function("schedule_stats_calculation", |b| {
        let scheduler = PreviewScheduler::new(ScheduleMode::Interval { steps: 5 }, 100);
        b.iter(|| {
            let stats = scheduler.stats();
            black_box(stats)
        });
    });
}

/// Benchmark protocol message serialization
fn bench_protocol_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_serialization");

    // JSON serialization
    group.bench_function("json_serialize", |b| {
        let msg = StreamMessage {
            msg_type: MessageType::Preview,
            step: 10,
            total_steps: 25,
            progress: 0.4,
            payload: Some(vec![0u8; 1024]),
            timestamp_ms: 12345678,
        };
        b.iter(|| {
            let json = serde_json::to_string(black_box(&msg)).unwrap();
            black_box(json)
        });
    });

    // JSON deserialization
    group.bench_function("json_deserialize", |b| {
        let msg = StreamMessage {
            msg_type: MessageType::Preview,
            step: 10,
            total_steps: 25,
            progress: 0.4,
            payload: Some(vec![0u8; 1024]),
            timestamp_ms: 12345678,
        };
        let json = serde_json::to_string(&msg).unwrap();
        b.iter(|| {
            let parsed: StreamMessage = serde_json::from_str(black_box(&json)).unwrap();
            black_box(parsed)
        });
    });

    // Binary serialization (if available)
    group.bench_function("binary_serialize", |b| {
        let msg = StreamMessage {
            msg_type: MessageType::Preview,
            step: 10,
            total_steps: 25,
            progress: 0.4,
            payload: Some(vec![0u8; 1024]),
            timestamp_ms: 12345678,
        };
        b.iter(|| {
            let binary = bincode::serialize(black_box(&msg)).unwrap();
            black_box(binary)
        });
    });

    group.finish();
}

/// Benchmark message throughput simulation
fn bench_message_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_throughput");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("process_1000_messages", |b| {
        let messages: Vec<StreamMessage> = (0..1000)
            .map(|i| StreamMessage {
                msg_type: MessageType::Preview,
                step: i % 25,
                total_steps: 25,
                progress: (i % 25) as f32 / 25.0,
                payload: None,
                timestamp_ms: i as u64,
            })
            .collect();

        b.iter(|| {
            let mut processed = 0;
            for msg in &messages {
                // Simulate message processing
                if msg.msg_type == MessageType::Preview {
                    processed += 1;
                }
            }
            black_box(processed)
        });
    });

    group.finish();
}

/// Benchmark quality resolution lookup
fn bench_quality_resolution(c: &mut Criterion) {
    c.bench_function("quality_resolution_lookup", |b| {
        let qualities = [
            PreviewQuality::Thumbnail,
            PreviewQuality::Low,
            PreviewQuality::Medium,
            PreviewQuality::Full,
        ];
        b.iter(|| {
            let mut total_pixels = 0u64;
            for quality in &qualities {
                let (w, h) = quality.resolution();
                total_pixels += (w * h) as u64;
            }
            black_box(total_pixels)
        });
    });
}

criterion_group!(
    benches,
    bench_scheduler_modes,
    bench_schedule_stats,
    bench_protocol_serialization,
    bench_message_throughput,
    bench_quality_resolution,
);

criterion_main!(benches);

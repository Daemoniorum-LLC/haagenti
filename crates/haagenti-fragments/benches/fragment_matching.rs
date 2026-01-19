//! Fragment Matching Benchmarks
//!
//! Benchmarks for locality-sensitive hashing (LSH) and
//! fragment similarity search operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use haagenti_fragments::{FragmentSignature, SignatureConfig, FragmentId};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_tensor_data(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let value: f32 = if rng.gen_ratio(7, 10) {
            rng.gen_range(-0.1..0.1)
        } else {
            rng.gen_range(-1.0..1.0)
        };
        let f16_bits = half::f16::from_f32(value).to_bits();
        data.extend_from_slice(&f16_bits.to_le_bytes());
    }
    data.truncate(size);
    data
}

fn generate_similar_tensor(base: &[u8], noise_level: f32, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = base.to_vec();
    let num_changes = (result.len() as f32 * noise_level / 2.0) as usize;
    for _ in 0..num_changes {
        let idx = rng.gen_range(0..result.len() / 2) * 2;
        if idx + 1 < result.len() {
            let bits = u16::from_le_bytes([result[idx], result[idx + 1]]);
            let value = half::f16::from_bits(bits).to_f32();
            let noise: f32 = rng.gen_range(-0.1..0.1);
            let new_bits = half::f16::from_f32(value + noise).to_bits();
            let new_bytes = new_bits.to_le_bytes();
            result[idx] = new_bytes[0];
            result[idx + 1] = new_bytes[1];
        }
    }
    result
}

fn bench_signature_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature_compute");
    let sizes = [1024, 4096, 16384, 65536, 262144];
    for size in sizes {
        let data = generate_tensor_data(size, 42);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("default", size), &data, |b, data| {
            let config = SignatureConfig::default();
            b.iter(|| FragmentSignature::compute(black_box(data), &config))
        });
        group.bench_with_input(BenchmarkId::new("fast", size), &data, |b, data| {
            let config = SignatureConfig { num_hashes: 128, shingle_size: 16, num_bands: 16 };
            b.iter(|| FragmentSignature::compute(black_box(data), &config))
        });
    }
    group.finish();
}

fn bench_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_compare");
    let size = 16384;
    let config = SignatureConfig::default();
    let base_data = generate_tensor_data(size, 42);
    let similar_data = generate_similar_tensor(&base_data, 0.1, 100);
    let different_data = generate_tensor_data(size, 999);
    let sig_base = FragmentSignature::compute(&base_data, &config);
    let sig_similar = FragmentSignature::compute(&similar_data, &config);
    let sig_different = FragmentSignature::compute(&different_data, &config);

    group.bench_function("simhash_distance", |b| {
        b.iter(|| sig_base.simhash_distance(black_box(&sig_similar)))
    });
    group.bench_function("minhash_similarity", |b| {
        b.iter(|| sig_base.minhash_similarity(black_box(&sig_similar)))
    });
    group.bench_function("combined_similarity/similar", |b| {
        b.iter(|| sig_base.similarity(black_box(&sig_similar)))
    });
    group.bench_function("combined_similarity/different", |b| {
        b.iter(|| sig_base.similarity(black_box(&sig_different)))
    });
    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    let fragment_size = 8192;
    let config = SignatureConfig::default();
    let batch_sizes = [10, 50, 100];
    for batch_size in batch_sizes {
        let fragments: Vec<_> = (0..batch_size).map(|i| generate_tensor_data(fragment_size, i as u64)).collect();
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(BenchmarkId::new("compute_signatures", batch_size), &fragments, |b, fragments| {
            b.iter(|| fragments.iter().map(|data| FragmentSignature::compute(black_box(data), &config)).collect::<Vec<_>>())
        });
    }
    let pair_counts = [10, 50, 100];
    for n in pair_counts {
        let fragments: Vec<_> = (0..n).map(|i| {
            let data = generate_tensor_data(fragment_size, i as u64);
            FragmentSignature::compute(&data, &config)
        }).collect();
        group.throughput(Throughput::Elements((n * (n - 1) / 2) as u64));
        group.bench_function(BenchmarkId::new("all_pairs_similarity", n), |b| {
            b.iter(|| {
                let mut sims = Vec::new();
                for i in 0..fragments.len() {
                    for j in (i+1)..fragments.len() {
                        sims.push(fragments[i].similarity(&fragments[j]));
                    }
                }
                black_box(sims)
            })
        });
    }
    group.finish();
}

fn bench_fragment_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragment_id");
    let sizes = [1024, 16384, 262144];
    for size in sizes {
        let data = generate_tensor_data(size, 42);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("from_content", size), &data, |b, data| {
            b.iter(|| FragmentId::from_content(black_box(data)))
        });
    }
    let id = FragmentId::from_content(&generate_tensor_data(1024, 42));
    group.bench_function("to_hex", |b| b.iter(|| black_box(&id).to_hex()));
    let hex = id.to_hex();
    group.bench_function("from_hex", |b| b.iter(|| FragmentId::from_hex(black_box(&hex))));
    group.finish();
}

criterion_group!(benches, bench_signature_computation, bench_similarity_comparison, bench_batch_operations, bench_fragment_id);
criterion_main!(benches);

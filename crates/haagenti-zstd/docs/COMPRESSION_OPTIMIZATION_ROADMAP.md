# Compression Optimization Roadmap

## Current State (Baseline)

| Metric | Small Data (1-4KB) | Large Data (64KB+) |
|--------|-------------------|-------------------|
| Compression vs zstd | **+8-116% faster** | -117-164% slower |
| Decompression vs zstd | **+170-180% faster** | **+150%+ faster** |

**Goal**: Close the large-data compression gap while maintaining small-data wins.

---

## Phase 1: Low-Hanging Fruit (1-2 weeks)

### 1.1 Adaptive Search Depth

**Hypothesis**: Reducing search depth for large inputs improves throughput with minimal ratio loss.

```rust
// TDD: Write test first
#[test]
fn test_adaptive_depth_large_input() {
    let data = generate_text_data(65536);
    let mut finder = MatchFinder::with_adaptive_depth(16);

    let start = Instant::now();
    let matches = finder.find_matches(&data);
    let elapsed = start.elapsed();

    // Must be at least 80% of small-data throughput
    let throughput = data.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    assert!(throughput >= 100.0, "Large input throughput {:.1} MB/s too low", throughput);

    // Compression ratio must not degrade more than 5%
    let ratio = matches.iter().map(|m| m.length).sum::<usize>() as f64 / data.len() as f64;
    assert!(ratio >= 0.60, "Match coverage {:.1}% too low", ratio * 100.0);
}

#[test]
fn test_adaptive_depth_scales_with_size() {
    let finder = MatchFinder::with_adaptive_depth(16);

    assert_eq!(finder.effective_depth(1024), 16);    // Small: full depth
    assert_eq!(finder.effective_depth(16384), 12);   // Medium: reduced
    assert_eq!(finder.effective_depth(65536), 8);    // Large: minimal
    assert_eq!(finder.effective_depth(262144), 6);   // Very large: aggressive
}
```

**Implementation**:
```rust
impl MatchFinder {
    fn effective_depth(&self, input_len: usize) -> usize {
        match input_len {
            0..=4096 => self.search_depth,
            4097..=16384 => self.search_depth * 3 / 4,
            16385..=65536 => self.search_depth / 2,
            _ => self.search_depth / 3,
        }
    }
}
```

**Expected Impact**: +15-25% throughput on large data, <3% ratio loss

---

### 1.2 Smarter Lazy Threshold Scaling

**Hypothesis**: Larger inputs benefit from higher lazy thresholds (commit earlier).

```rust
#[test]
fn test_lazy_threshold_scaling() {
    // Small input: lazy helps
    let small = generate_text_data(1024);
    let mut finder_small = LazyMatchFinder::with_adaptive_threshold(16);
    finder_small.configure_for_size(small.len());
    assert_eq!(finder_small.lazy_threshold, 24);

    // Large input: commit earlier
    let large = generate_text_data(65536);
    let mut finder_large = LazyMatchFinder::with_adaptive_threshold(16);
    finder_large.configure_for_size(large.len());
    assert_eq!(finder_large.lazy_threshold, 12); // Lower = commit sooner
}

#[test]
fn test_lazy_scaling_improves_large_throughput() {
    let data = generate_text_data(65536);

    let mut fixed = LazyMatchFinder::new(16);
    let mut adaptive = LazyMatchFinder::with_adaptive_threshold(16);
    adaptive.configure_for_size(data.len());

    let fixed_time = benchmark(|| fixed.find_matches(&data));
    let adaptive_time = benchmark(|| adaptive.find_matches(&data));

    // Adaptive should be at least 10% faster
    assert!(adaptive_time < fixed_time * 0.9);
}
```

**Expected Impact**: +10-15% throughput on large data

---

### 1.3 Early Exit on Excellent Matches

**Hypothesis**: Stop searching when we find a "good enough" match relative to position.

```rust
#[test]
fn test_early_exit_excellent_match() {
    let mut finder = MatchFinder::new(16);

    // When we find a 32+ byte match, stop searching immediately
    let data = b"abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789";
    let matches = finder.find_matches(data);

    // Should find one excellent match and not waste time searching further
    assert_eq!(matches.len(), 1);
    assert!(matches[0].length >= 32);
}

#[test]
fn test_early_exit_threshold_by_position() {
    let finder = MatchFinder::new(16);

    // Early in file: need longer match to exit early
    assert_eq!(finder.early_exit_threshold(100), 32);

    // Later in file: shorter match is acceptable
    assert_eq!(finder.early_exit_threshold(10000), 24);
    assert_eq!(finder.early_exit_threshold(50000), 16);
}
```

**Expected Impact**: +5-10% throughput on repetitive data

---

## Phase 2: Structural Optimizations (2-3 weeks)

### 2.1 Block Chunking with Independent Hash Tables

**Hypothesis**: Processing in 16KB chunks improves cache locality.

```rust
#[test]
fn test_chunked_compression_correctness() {
    let data = generate_text_data(65536);

    let standard = compress_standard(&data);
    let chunked = compress_chunked(&data, 16384);

    // Both must decompress to same data
    assert_eq!(decompress(&standard), data);
    assert_eq!(decompress(&chunked), data);
}

#[test]
fn test_chunked_compression_throughput() {
    let data = generate_text_data(262144);

    let standard_time = benchmark(|| compress_standard(&data));
    let chunked_time = benchmark(|| compress_chunked(&data, 16384));

    // Chunked should be at least 20% faster on large data
    assert!(chunked_time < standard_time * 0.8);
}

#[test]
fn test_chunk_boundaries_preserve_matches() {
    // Ensure matches spanning chunk boundaries are handled correctly
    let data = b"AAAA".repeat(5000); // Pattern spans multiple chunks
    let compressed = compress_chunked(&data, 1024);
    let decompressed = decompress(&compressed);

    assert_eq!(decompressed, data);
}
```

**Implementation Sketch**:
```rust
fn compress_chunked(input: &[u8], chunk_size: usize) -> Vec<u8> {
    let chunks: Vec<_> = input.chunks(chunk_size).collect();

    // Process chunks (potentially in parallel)
    let compressed_chunks: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            let mut finder = MatchFinder::new(16); // Fresh hash table per chunk
            let matches = finder.find_matches(chunk);
            encode_block(chunk, &matches)
        })
        .collect();

    // Combine into multi-block frame
    build_multiblock_frame(&compressed_chunks)
}
```

**Expected Impact**: +20-30% throughput on 64KB+ data

---

### 2.2 Two-Tier Hash Table (Short + Long Matches)

**Hypothesis**: Separate hash tables for 4-byte and 8-byte prefixes improve hit rates.

```rust
#[test]
fn test_two_tier_hash_finds_long_matches_faster() {
    let data = generate_text_data(16384);

    let mut single = MatchFinder::new(16);
    let mut two_tier = TwoTierMatchFinder::new(16);

    let single_matches = single.find_matches(&data);
    let two_tier_matches = two_tier.find_matches(&data);

    // Two-tier should find same or better matches
    let single_coverage: usize = single_matches.iter().map(|m| m.length).sum();
    let two_tier_coverage: usize = two_tier_matches.iter().map(|m| m.length).sum();

    assert!(two_tier_coverage >= single_coverage * 95 / 100);
}

#[test]
fn test_two_tier_hash_throughput() {
    let data = generate_text_data(65536);

    let single_time = benchmark(|| {
        let mut finder = MatchFinder::new(16);
        finder.find_matches(&data)
    });

    let two_tier_time = benchmark(|| {
        let mut finder = TwoTierMatchFinder::new(16);
        finder.find_matches(&data)
    });

    // Two-tier should not be slower (goal: 10% faster)
    assert!(two_tier_time <= single_time);
}
```

**Expected Impact**: +10-15% throughput, +2-3% better ratio

---

### 2.3 Streaming Entropy Encoding Pipeline

**Hypothesis**: Overlap match finding with entropy encoding to hide latency.

```rust
#[test]
fn test_pipelined_compression_correctness() {
    let data = generate_text_data(65536);

    let standard = compress_standard(&data);
    let pipelined = compress_pipelined(&data);

    // Must produce identical output
    assert_eq!(standard, pipelined);
}

#[test]
fn test_pipeline_stages_overlap() {
    let data = generate_text_data(65536);

    // Measure individual stages
    let match_time = benchmark(|| find_all_matches(&data));
    let encode_time = benchmark(|| encode_all_matches(&data, &matches));
    let sequential_total = match_time + encode_time;

    // Pipelined should be faster than sequential sum
    let pipelined_time = benchmark(|| compress_pipelined(&data));

    assert!(pipelined_time < sequential_total * 0.85);
}
```

**Expected Impact**: +15-20% throughput on large data (hides encoding latency)

---

## Phase 3: Novel Approaches (3-4 weeks)

### 3.1 Learned Hash Function

**Hypothesis**: Data-adaptive hash function reduces collisions for specific data types.

```rust
#[test]
fn test_learned_hash_reduces_collisions() {
    let text_data = generate_text_data(65536);
    let binary_data = generate_binary_data(65536);

    // Standard hash collision rate
    let std_text_collisions = measure_collision_rate(&text_data, standard_hash);
    let std_binary_collisions = measure_collision_rate(&binary_data, standard_hash);

    // Learn hash from sample
    let text_hash = LearnedHash::train(&text_data[..4096]);
    let binary_hash = LearnedHash::train(&binary_data[..4096]);

    let learned_text_collisions = measure_collision_rate(&text_data, text_hash);
    let learned_binary_collisions = measure_collision_rate(&binary_data, binary_hash);

    // Learned hash should have fewer collisions
    assert!(learned_text_collisions < std_text_collisions * 0.8);
    assert!(learned_binary_collisions < std_binary_collisions * 0.8);
}

#[test]
fn test_learned_hash_improves_compression() {
    let data = generate_text_data(65536);

    let standard_ratio = compress_ratio_with_hash(&data, standard_hash);

    let learned = LearnedHash::train(&data[..4096]);
    let learned_ratio = compress_ratio_with_hash(&data, learned);

    // Learned should achieve better compression
    assert!(learned_ratio > standard_ratio);
}
```

**Implementation Concept**:
```rust
struct LearnedHash {
    byte_weights: [u32; 256],
    mix_constant: u32,
}

impl LearnedHash {
    fn train(sample: &[u8]) -> Self {
        // Analyze byte frequency distribution
        let mut freq = [0u32; 256];
        for &b in sample {
            freq[b as usize] += 1;
        }

        // Weight rare bytes higher (more discriminative)
        let total = sample.len() as f32;
        let weights: [u32; 256] = freq.map(|f| {
            let p = f as f32 / total;
            ((1.0 / (p + 0.001)) * 1000.0) as u32
        });

        Self {
            byte_weights: weights,
            mix_constant: Self::find_optimal_mix(sample, &weights),
        }
    }

    fn hash(&self, bytes: &[u8; 4]) -> u32 {
        let mut h = 0u32;
        for &b in bytes {
            h = h.wrapping_mul(self.mix_constant)
                 .wrapping_add(self.byte_weights[b as usize]);
        }
        h
    }
}
```

**Expected Impact**: +5-10% ratio, +5% throughput (fewer collisions = less chain traversal)

---

### 3.2 Speculative Parallel Match Finding

**Hypothesis**: Speculatively process multiple positions, discard invalid results.

```rust
#[test]
fn test_speculative_parallel_correctness() {
    let data = generate_text_data(65536);

    let sequential = find_matches_sequential(&data);
    let speculative = find_matches_speculative(&data);

    // Must find equivalent matches
    assert_equivalent_matches(&sequential, &speculative);
}

#[test]
fn test_speculative_parallel_throughput() {
    let data = generate_text_data(262144);

    let seq_time = benchmark(|| find_matches_sequential(&data));
    let spec_time = benchmark(|| find_matches_speculative(&data));

    // Speculative should be significantly faster on large data
    assert!(spec_time < seq_time * 0.7);
}

#[test]
fn test_speculative_handles_dependencies() {
    // Ensure speculative results are correctly filtered
    let data = b"ABCABCABCABC"; // Overlapping matches

    let matches = find_matches_speculative(data);

    // No overlapping matches in output
    for i in 1..matches.len() {
        assert!(matches[i].position >= matches[i-1].position + matches[i-1].length);
    }
}
```

**Implementation Concept**:
```rust
fn find_matches_speculative(input: &[u8]) -> Vec<Match> {
    const LOOKAHEAD: usize = 4;

    let mut results = Vec::new();
    let mut pos = 0;

    while pos < input.len() - MIN_MATCH_LENGTH {
        // Speculatively compute hashes for next LOOKAHEAD positions
        let hashes: [u32; LOOKAHEAD] = std::array::from_fn(|i| {
            if pos + i + 4 <= input.len() {
                hash4(input, pos + i)
            } else {
                0
            }
        });

        // Find best match among speculative positions
        let candidates: Vec<_> = hashes.iter()
            .enumerate()
            .filter_map(|(i, &h)| find_match_at(input, pos + i, h))
            .collect();

        if let Some(best) = candidates.into_iter().max_by_key(|m| m.length) {
            // Emit literals for skipped positions
            results.push(best);
            pos = best.position + best.length;
        } else {
            pos += 1;
        }
    }

    results
}
```

**Expected Impact**: +25-35% throughput on large data (parallelizable)

---

### 3.3 Context-Adaptive Entropy Selection

**Hypothesis**: Different entropy encoders for different data regions.

```rust
#[test]
fn test_context_detection() {
    let mixed = concat(&[
        &generate_text_data(1024),    // Text region
        &generate_binary_data(1024),  // Binary region
        &vec![0u8; 1024],             // Zero region
    ]);

    let contexts = detect_contexts(&mixed);

    assert_eq!(contexts[0].region_type, RegionType::Text);
    assert_eq!(contexts[1].region_type, RegionType::Binary);
    assert_eq!(contexts[2].region_type, RegionType::Rle);
}

#[test]
fn test_context_adaptive_improves_ratio() {
    let mixed = generate_mixed_data(65536);

    let uniform_ratio = compress_uniform(&mixed);
    let adaptive_ratio = compress_context_adaptive(&mixed);

    // Adaptive should compress better
    assert!(adaptive_ratio > uniform_ratio * 1.05);
}

#[test]
fn test_context_adaptive_encoding_selection() {
    let text = generate_text_data(4096);
    let binary = generate_binary_data(4096);

    let text_encoder = select_encoder_for_context(&text);
    let binary_encoder = select_encoder_for_context(&binary);

    // Different contexts should select different encoders
    assert_ne!(text_encoder, binary_encoder);
}
```

**Expected Impact**: +3-8% ratio on mixed-content data

---

### 3.4 Match Cache with Bloom Filter

**Hypothesis**: Cache recently successful match patterns for O(1) lookup.

```rust
#[test]
fn test_match_cache_hit_rate() {
    let data = generate_text_data(65536);
    let mut cache = MatchCache::new(1024);

    // First pass: populate cache
    let mut finder = MatchFinder::new(16);
    let matches = finder.find_matches(&data);
    for m in &matches {
        cache.insert(&data[m.position..m.position+4], m.offset);
    }

    // Measure hit rate on same data
    let mut hits = 0;
    let mut total = 0;
    for i in 0..data.len()-4 {
        total += 1;
        if cache.lookup(&data[i..i+4]).is_some() {
            hits += 1;
        }
    }

    let hit_rate = hits as f64 / total as f64;
    assert!(hit_rate > 0.3, "Hit rate {:.1}% too low", hit_rate * 100.0);
}

#[test]
fn test_bloom_filter_rejects_misses_fast() {
    let cache = MatchCache::new(1024);

    // Bloom filter should reject non-existent patterns in O(1)
    let miss_time = benchmark(|| {
        for _ in 0..10000 {
            cache.lookup(b"ZZZZ"); // Definitely not in cache
        }
    });

    // Should be very fast (< 1ns per lookup)
    let per_lookup = miss_time.as_nanos() / 10000;
    assert!(per_lookup < 10, "Bloom filter too slow: {}ns", per_lookup);
}
```

**Expected Impact**: +10-15% throughput on repetitive data

---

### 3.5 Vectorized Hash Computation (AVX2/AVX-512)

**Hypothesis**: Compute multiple hashes in parallel using SIMD.

```rust
#[test]
fn test_simd_hash_correctness() {
    let data = generate_text_data(1024);

    for i in 0..data.len()-32 {
        let scalar_hashes: [u32; 8] = std::array::from_fn(|j| {
            scalar_hash4(&data, i + j * 4)
        });

        let simd_hashes = simd_hash4x8(&data[i..]);

        assert_eq!(scalar_hashes, simd_hashes);
    }
}

#[test]
fn test_simd_hash_throughput() {
    let data = generate_text_data(65536);

    let scalar_time = benchmark(|| {
        for i in (0..data.len()-4).step_by(32) {
            for j in 0..8 {
                black_box(scalar_hash4(&data, i + j * 4));
            }
        }
    });

    let simd_time = benchmark(|| {
        for i in (0..data.len()-32).step_by(32) {
            black_box(simd_hash4x8(&data[i..]));
        }
    });

    // SIMD should be at least 4x faster
    assert!(simd_time < scalar_time / 4);
}
```

**Implementation (AVX2)**:
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_hash4x8(data: &[u8]) -> [u32; 8] {
    use std::arch::x86_64::*;

    // Load 32 bytes (8 overlapping 4-byte windows)
    let input = _mm256_loadu_si256(data.as_ptr() as *const __m256i);

    // Multiply by hash constant
    let prime = _mm256_set1_epi32(0x9E3779B9u32 as i32);
    let h = _mm256_mullo_epi32(input, prime);

    // XOR-shift mix
    let h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 15));
    let h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x85EBCA6Bu32 as i32));

    // Extract results
    let mut result = [0u32; 8];
    _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, h);
    result
}
```

**Expected Impact**: +15-25% match finding throughput

---

## Phase 4: Integration & Validation (2 weeks)

### 4.1 Comprehensive Benchmark Suite

```rust
#[test]
fn benchmark_matrix() {
    let sizes = [1024, 4096, 16384, 65536, 262144];
    let types = ["text", "binary", "mixed", "llm_weights"];

    for &size in &sizes {
        for &dtype in &types {
            let data = generate_data(dtype, size);

            let haagenti_time = benchmark(|| compress_haagenti(&data));
            let zstd_time = benchmark(|| compress_zstd(&data));

            let ratio = haagenti_time.as_nanos() as f64 / zstd_time.as_nanos() as f64;

            println!("{:>6} {:>12}: haagenti/zstd = {:.2}x", size, dtype, ratio);

            // All cases should be within 2x of reference
            assert!(ratio < 2.0, "{}B {} too slow: {:.2}x", size, dtype, ratio);
        }
    }
}
```

### 4.2 Regression Test Suite

```rust
#[test]
fn no_compression_regression() {
    let test_cases = load_regression_corpus();

    for (name, data, expected_ratio) in test_cases {
        let compressed = compress(&data);
        let actual_ratio = compressed.len() as f64 / data.len() as f64;

        assert!(
            actual_ratio <= expected_ratio * 1.02,
            "{}: ratio {:.3} exceeds expected {:.3}",
            name, actual_ratio, expected_ratio
        );
    }
}

#[test]
fn no_throughput_regression() {
    let baselines = load_throughput_baselines();

    for (name, data, min_throughput) in baselines {
        let throughput = measure_throughput(|| compress(&data));

        assert!(
            throughput >= min_throughput * 0.95,
            "{}: throughput {:.1} MB/s below baseline {:.1} MB/s",
            name, throughput, min_throughput
        );
    }
}
```

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| 64KB text compression | 2.2x slower | 1.3x slower | **Equal** |
| 256KB text compression | 2.6x slower | 1.5x slower | 1.2x slower |
| 1KB compression | 8% faster | 15% faster | 25% faster |
| Decompression | 2.8x faster | Maintain | 3x faster |

---

## Priority Order

1. **Adaptive Search Depth** (Phase 1.1) - Immediate win, low risk
2. **Block Chunking** (Phase 2.1) - Best ROI for large data
3. **Vectorized Hash** (Phase 3.5) - Significant throughput gain
4. **Speculative Parallel** (Phase 3.2) - Novel differentiator
5. **Match Cache** (Phase 3.4) - Good for repetitive data
6. **Learned Hash** (Phase 3.1) - Research/differentiation value

---

## Implementation Notes

- All optimizations must maintain bit-exact decompression compatibility
- Each phase should be independently deployable
- Feature flags for experimental optimizations
- Continuous benchmarking in CI to catch regressions

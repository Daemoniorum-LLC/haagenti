# Haagenti-Zstd Performance Optimization Roadmap

## Current Status

**Throughput vs Reference Zstd:**
| Size | Text | Binary | LLM Weights |
|------|------|--------|-------------|
| 1 KB | 51% | 59% | 53% |
| 4 KB | 32% | 48% | 35% |
| 16 KB | 15% | 28% | 17% |

**Target:** Achieve 60-80% of reference zstd performance across all sizes.

---

## TDD Approach

Each optimization follows the Red-Green-Refactor cycle:
1. **RED:** Write failing performance test with target threshold
2. **GREEN:** Implement optimization to pass the test
3. **REFACTOR:** Clean up while maintaining performance

---

## Phase 1: Quick Wins (Expected: +30-50% throughput) ✅ COMPLETED

### 1.1 Hash Table Reset Optimization ✅
**File:** `compress/match_finder.rs:86-97`
**Issue:** Manual loop zeroing instead of optimized memset
**Status:** COMPLETED - Uses `fill()` for optimized memset

```rust
// BEFORE (slow)
for entry in &mut self.hash_table { *entry = 0; }

// AFTER (fast)
self.hash_table.fill(0);
```

**TDD Test:**
```rust
#[test]
fn test_match_finder_reset_performance() {
    let mut finder = MatchFinder::new(6);
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        finder.reset();
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 50, "Reset too slow: {:?}", elapsed);
}
```

**Expected Gain:** 15-30% faster for large inputs

---

### 1.2 Pre-allocated Output Buffers
**Files:** `compress/block.rs`, `huffman/encoder.rs`
**Issue:** Vec reallocations in hot paths

**Changes:**
- Pre-calculate output size estimates
- Use `Vec::with_capacity()` with accurate estimates
- Add `shrink_to_fit()` only at frame boundaries

**TDD Test:**
```rust
#[test]
fn test_no_reallocation_during_compression() {
    let compressor = ZstdCompressor::new();
    let data = vec![0u8; 65536];

    // Use custom allocator to track allocations
    let alloc_count_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let _ = compressor.compress(&data);
    let alloc_count_after = ALLOC_COUNT.load(Ordering::SeqCst);

    // Should have minimal allocations (< 10)
    assert!(alloc_count_after - alloc_count_before < 10);
}
```

**Expected Gain:** 10-15% for medium/large inputs

---

### 1.3 Huffman Encode Chunked Processing with Prefetching ✅
**File:** `huffman/encoder.rs:422-549`
**Issue:** Reverse iteration breaks CPU prefetcher
**Status:** COMPLETED - Chunked reverse processing with software prefetching

**Implementation:**
- Process in 64-byte cache-line chunks (reverse chunk order, forward within chunk)
- Software prefetching brings next chunk into L1 cache ahead of time
- Unrolled inner loop (4 symbols at a time) for better ILP
- Maintains correct reverse encoding order required by Zstd format

```rust
// Chunked processing with prefetch
const CHUNK_SIZE: usize = 64;
while pos > 0 {
    let chunk_start = pos.saturating_sub(CHUNK_SIZE);
    // Prefetch next chunk while processing current
    _mm_prefetch(literals.as_ptr().add(chunk_start - CHUNK_SIZE), _MM_HINT_T0);
    // Process chunk in reverse with 4x unrolling
    ...
}
```

**Expected Gain:** 20-40% faster encoding

---

### 1.4 O(n log n) Weight Normalization Algorithm ✅
**File:** `huffman/encoder.rs:145-261`
**Issue:** O(n²) nested loops that could iterate 1000+ times for large symbol sets
**Status:** COMPLETED - Single-pass greedy Kraft-sum algorithm

**Implementation:**
- Sort symbols by frequency once: O(n log n)
- Calculate initial weights based on log2 frequency ratios: O(n)
- Adjust weights to satisfy Kraft inequality using sorted order: O(n log n)
- Final pass to fill remaining capacity: O(n)

**Expected Gain:** 30-50% faster for high-entropy data (200+ unique symbols)

---

### 1.5 Match Finder Batch Hash Update with Prefetching ✅
**File:** `compress/match_finder.rs:296-358`
**Issue:** step_by(2) loop recomputes hashes inefficiently
**Status:** COMPLETED - Batch processing with software prefetching

**Implementation:**
- Process hash updates in batches of 8 positions
- Software prefetch next batch of input data
- Maintains step_by(2) behavior for speed/quality tradeoff

**Expected Gain:** 15-25% faster match finding for repetitive data

---

## Phase 2: SIMD Optimizations (Expected: +50-100% throughput) ✅ COMPLETED

### 2.1 SIMD Match Length Comparison ✅
**File:** `compress/match_finder.rs:276-307`
**Issue:** Byte-by-byte fallback after u64 comparison
**Status:** COMPLETED - Uses `haagenti_simd::find_match_length()` with AVX2

**Implementation:**
```rust
#[cfg(target_arch = "x86_64")]
fn match_length_simd(a: &[u8], b: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let mut len = 0;
    let max_len = a.len().min(b.len());

    // Process 32 bytes at a time with AVX2
    while len + 32 <= max_len {
        unsafe {
            let va = _mm256_loadu_si256(a[len..].as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b[len..].as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(va, vb);
            let mask = _mm256_movemask_epi8(cmp) as u32;

            if mask != 0xFFFFFFFF {
                return len + mask.trailing_ones() as usize;
            }
            len += 32;
        }
    }

    // Fallback for remainder
    while len < max_len && a[len] == b[len] {
        len += 1;
    }
    len
}
```

**TDD Test:**
```rust
#[test]
fn test_simd_match_length_correctness() {
    let a = vec![0u8; 1000];
    let mut b = vec![0u8; 1000];
    b[500] = 1;

    assert_eq!(match_length_simd(&a, &b), 500);
    assert_eq!(match_length_simd(&a, &a), 1000);
}

#[test]
fn test_simd_match_length_performance() {
    let a = vec![0u8; 1_000_000];
    let b = vec![0u8; 1_000_000];

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = match_length_simd(&a, &b);
    }
    let elapsed = start.elapsed();
    let throughput = (1_000_000 * 100) as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

    assert!(throughput > 10.0, "SIMD match too slow: {} GB/s", throughput);
}
```

**Expected Gain:** 3-5x faster match finding

---

### 2.2 SIMD Frequency Histogram
**File:** `huffman/encoder.rs:105-130`
**Issue:** Sequential frequency counting

**Implementation:**
```rust
#[cfg(target_arch = "x86_64")]
fn count_frequencies_simd(data: &[u8]) -> [u64; 256] {
    use std::arch::x86_64::*;

    // Use 4 parallel histograms to avoid conflicts
    let mut hist = [[0u64; 256]; 4];

    for (i, &byte) in data.iter().enumerate() {
        hist[i & 3][byte as usize] += 1;
    }

    // Merge histograms
    let mut result = [0u64; 256];
    for i in 0..256 {
        result[i] = hist[0][i] + hist[1][i] + hist[2][i] + hist[3][i];
    }
    result
}
```

**TDD Test:**
```rust
#[test]
fn test_simd_histogram_correctness() {
    let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let freq = count_frequencies_simd(&data);

    for i in 0..256 {
        assert_eq!(freq[i], 4, "Symbol {} has wrong count", i);
    }
}
```

**Expected Gain:** 2-4x faster frequency counting

---

### 2.3 SIMD Overlapping Copy
**File:** `decompress.rs:368-437`
**Issue:** Branch-heavy pattern copy

**Implementation:**
```rust
#[inline(always)]
fn copy_match_simd(output: &mut Vec<u8>, offset: usize, length: usize) {
    let dst_start = output.len();
    output.reserve(length);

    unsafe {
        let ptr = output.as_mut_ptr().add(dst_start);
        let src = ptr.sub(offset);

        if offset >= 16 {
            // Fast path: no overlap, use memcpy
            std::ptr::copy_nonoverlapping(src, ptr, length);
        } else {
            // Overlap: copy in offset-sized chunks
            let mut copied = 0;
            while copied < length {
                let chunk = (length - copied).min(offset);
                std::ptr::copy_nonoverlapping(src, ptr.add(copied), chunk);
                copied += chunk;
            }
        }
        output.set_len(dst_start + length);
    }
}
```

**Expected Gain:** 15-25% faster decompression

---

## Phase 3: Algorithmic Improvements (Expected: +20-40% throughput) ✅ COMPLETED

### 3.1 FSE-Compressed Huffman Weights ✅
**File:** `huffman/encoder.rs:609-831`
**Issue:** Falls back to raw literals for >128 symbols
**Status:** COMPLETED - FSE encoding infrastructure implemented

**Implementation:**
- Removed 127 symbol limit in HuffmanEncoder::build()
- serialize_weights() now calls serialize_weights_fse() for >128 symbols
- Simulation-based FSE encoding: finds valid decoder state sequence
- Proper bitstream reversal for FSE decoder compatibility
- Graceful fallback to raw block if FSE encoding fails (complex symbol distributions)

**TDD Test:**
```rust
#[test]
fn test_fse_weights_roundtrip() {
    // Data with all 256 symbols
    let data: Vec<u8> = (0..=255).cycle().take(4096).collect();

    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();

    let compressed = compressor.compress(&data).unwrap();
    let decompressed = decompressor.decompress(&compressed).unwrap();

    assert_eq!(decompressed, data);
    // Should achieve some compression (not raw)
    assert!(compressed.len() < data.len());
}
```

**Expected Gain:** Better compression ratio for high-entropy data

---

### 3.2 Lazy Match Finding ✅
**File:** `compress/match_finder.rs`
**Issue:** Greedy matching misses better opportunities
**Status:** COMPLETED - LazyMatchFinder integrated for Default+ compression levels

**Implementation:**
- Look ahead 1 position for better matches
- Only emit match if it's better than next position's match + 1
- Uses MatchFinderVariant enum in CompressContext to switch between greedy/lazy

**TDD Test:**
```rust
#[test]
fn test_lazy_matching_improves_ratio() {
    let compressor_greedy = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressor_lazy = ZstdCompressor::with_level(CompressionLevel::Default);

    let data = include_bytes!("../testdata/enwik8.100k");

    let greedy = compressor_greedy.compress(data).unwrap();
    let lazy = compressor_lazy.compress(data).unwrap();

    assert!(lazy.len() < greedy.len(), "Lazy should compress better");
}
```

**Expected Gain:** 5-15% better compression ratio

---

### 3.3 Parallel 4-Stream Encoding ✅
**File:** `compress/block.rs`
**Issue:** Sequential stream compression
**Status:** COMPLETED - Parallel encoding with rayon feature flag

**Implementation:**
- Added `parallel` feature in Cargo.toml with rayon dependency
- `encode_huffman_4stream()` uses `par_iter()` when parallel feature is enabled
- Falls back to sequential encoding when parallel feature is disabled
- Feature-gated with `#[cfg(feature = "parallel")]`

**Expected Gain:** 2-3x faster for large literals (requires `rayon` dependency)

---

## Phase 4: Memory Layout Optimizations (Expected: +10-20% throughput) ✅ COMPLETED

### 4.1 Cache-Aligned Structures ✅
**File:** `compress/match_finder.rs`
**Issue:** Hash tables may straddle cache lines
**Status:** COMPLETED - AlignedHashTable with 64-byte cache line alignment

**Implementation:**
- Created `AlignedHashTable` struct with `#[repr(C, align(64))]`
- Uses `alloc_zeroed` for efficient initialization
- MatchFinder now uses `Box<AlignedHashTable>` for heap allocation
- Added tests verifying 64-byte alignment of data pointer

```rust
#[repr(C, align(64))]  // Cache line alignment
struct AlignedHashTable {
    data: [u32; HASH_SIZE],
}

impl AlignedHashTable {
    fn new_boxed() -> Box<Self> {
        // Zeroed allocation for efficiency
        unsafe {
            let layout = core::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            Box::from_raw(ptr)
        }
    }
}
```

### 4.2 Arena Allocation ✅
**File:** `compress/arena.rs`
**Issue:** Many small allocations during compression
**Status:** COMPLETED - Arena allocator with per-frame reset

**Implementation:**
- Created `Arena` bump allocator for temporary allocations
- 64KB default size (covers most per-frame needs)
- O(1) reset between frames via `arena.reset()`
- `ArenaVec` for growable arena-backed buffers
- Peak usage tracking for diagnostics and tuning
- Integrated into `CompressContext` with automatic reset per frame

```rust
pub struct Arena {
    buffer: Vec<u8>,
    pos: Cell<usize>,      // Bump pointer
    peak_usage: Cell<usize>, // Diagnostic tracking
}

impl Arena {
    fn alloc_slice(&self, len: usize) -> Option<&mut [u8]> {
        let pos = self.pos.get();
        let new_pos = pos.checked_add(len)?;
        if new_pos > self.buffer.len() { return None; }
        self.pos.set(new_pos);
        // Return slice from buffer[pos..new_pos]
    }

    fn reset(&self) {
        self.pos.set(0);  // O(1) reset
    }
}
```

---

## Implementation Priority

| Phase | Item | Impact | Effort | Priority |
|-------|------|--------|--------|----------|
| 1.1 | Hash reset | +15% | 5 min | P0 |
| 1.3 | Huffman forward | +25% | 2 hrs | P0 |
| 1.2 | Pre-alloc buffers | +10% | 1 hr | P1 |
| 2.1 | SIMD match len | +30% | 4 hrs | P1 |
| 2.3 | SIMD overlap copy | +20% | 3 hrs | P1 |
| 2.2 | SIMD histogram | +15% | 3 hrs | P2 |
| 3.1 | FSE weights | +10% ratio | 8 hrs | P2 |
| 3.2 | Lazy matching | +10% ratio | 6 hrs | P2 |
| 3.3 | Parallel streams | +100% | 4 hrs | P3 |
| 4.x | Memory layout | +10% | 4 hrs | P3 |

---

## Benchmark Targets

After all optimizations:

| Size | Target Throughput | vs Reference |
|------|-------------------|--------------|
| 1 KB | 50 MiB/s | 80% |
| 4 KB | 150 MiB/s | 65% |
| 16 KB | 500 MiB/s | 60% |
| 64 KB | 600 MiB/s | 55% |

---

## Test Commands

```bash
# Run all tests
cargo test -p haagenti-zstd

# Run benchmarks
cargo bench -p haagenti-zstd

# Run with specific features
cargo test -p haagenti-zstd --features simd
cargo bench -p haagenti-zstd --features parallel

# Profile hot paths
cargo flamegraph --bench zstd_benchmark -- --bench
```

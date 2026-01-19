# Haagenti Agent Quickstart Guide

**Purpose:** This document helps AI agents quickly understand how to run benchmarks, find key files, and replicate performance testing for the Haagenti compression library.

**Last Updated:** 2025-01-03

---

## TL;DR - Run This

```bash
cd /home/crook/dev2/workspace/nyx/haagenti

# Full benchmark comparison (Haagenti vs reference zstd)
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native" \
  cargo run --release -p haagenti-zstd \
  --example benchmark_comparison --features parallel
```

Expected output shows:
- **Decompression:** 7-29x faster than reference
- **Compression (small data):** 1.4-2.9x faster
- **Compression (large data):** 0.5-0.9x (optimization target)

---

## Key Locations

### Workspace Root
```
/home/crook/dev2/workspace/nyx/haagenti/
```

### Core Crates
```
crates/haagenti-core/     # Traits, types, streaming API
crates/haagenti-zstd/     # Zstd implementation (THE BIG ONE)
crates/haagenti-simd/     # SIMD primitives, histograms
crates/haagenti-lz4/      # LZ4 implementation
crates/haagenti-brotli/   # Brotli implementation
crates/haagenti-deflate/  # Deflate/Gzip/Zlib
```

### Zstd Implementation (Key Files)
```
crates/haagenti-zstd/src/
├── lib.rs                    # Public API
├── compress/
│   ├── mod.rs                # CompressContext, frame encoding
│   ├── match_finder.rs       # LZ77 match finding (AVX-512!)
│   ├── speculative.rs        # Multi-path parallel compression
│   ├── analysis.rs           # Entropy fingerprinting
│   └── block.rs              # Block encoding
├── decompress/
│   └── mod.rs                # Decompression pipeline
├── fse/                      # Finite State Entropy coding
├── huffman/                  # Huffman coding
├── frame/                    # Zstd frame format
└── block/                    # Block parsing
```

### Benchmark Files
```
crates/haagenti-zstd/examples/
├── benchmark_comparison.rs   # Full comparison vs reference
└── quick_bench.rs            # Quick sanity check

crates/haagenti-zstd/benches/
└── zstd_benchmark.rs         # Criterion benchmarks
```

### Documentation
```
docs/AGENT_QUICKSTART.md                  # This file
docs/ARCANUM_INTEGRATION_BENCHMARKS.md    # Crypto integration
docs/ZSTD-ROADMAP.md                      # Implementation checklist

crates/haagenti-zstd/docs/
├── COMPRESSION_OPTIMIZATION_ROADMAP.md   # Optimization plan
├── PERFORMANCE-DOMINATION-ROADMAP.md     # Advanced opts
└── PERFORMANCE-ROADMAP.md                # Original roadmap

BENCHMARK_REPORT.md                       # Summary for humans
```

---

## Required Flags

### RUSTFLAGS (Critical!)
```bash
RUSTFLAGS="-C target-cpu=native"
```
Enables AVX-512 and optimal SIMD codegen.

### CARGO_INCREMENTAL
```bash
CARGO_INCREMENTAL=0
```
Required because sccache conflicts with incremental compilation.

### Feature Flags
```bash
--features parallel      # Enable rayon for speculative compression
```

---

## Running Benchmarks

### Full Comparison (Recommended)
```bash
cd /home/crook/dev2/workspace/nyx/haagenti

CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native" \
  cargo run --release -p haagenti-zstd \
  --example benchmark_comparison --features parallel
```

### Quick Benchmark
```bash
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native" \
  cargo run --release -p haagenti-zstd \
  --example quick_bench
```

### Criterion Benchmarks
```bash
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p haagenti-zstd
```

Results appear in `target/criterion/*/report/index.html`.

---

## Key APIs

### Compression
```rust
use haagenti_core::{Compressor, CompressionLevel};
use haagenti_zstd::ZstdCompressor;

// Simple API
let compressor = ZstdCompressor::new();
let compressed = compressor.compress(&data)?;

// With level
let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
let compressed = compressor.compress(&data)?;

// Speculative (tries 5 strategies in parallel, picks best)
use haagenti_zstd::compress::SpeculativeCompressor;
let compressor = SpeculativeCompressor::new();
let compressed = compressor.compress(&data)?;
```

### Decompression
```rust
use haagenti_core::Decompressor;
use haagenti_zstd::ZstdDecompressor;

let decompressor = ZstdDecompressor::new();
let original = decompressor.decompress(&compressed)?;
```

### Low-Level
```rust
use haagenti_zstd::compress::CompressContext;
use haagenti_zstd::decompress::decompress_frame;

// Compression context (reusable)
let mut ctx = CompressContext::new(CompressionLevel::Default);
let compressed = ctx.compress(&data)?;

// Direct frame decompression
let original = decompress_frame(&compressed)?;
```

---

## Performance Results Summary

### Decompression (THE BIG WIN)
```
Binary 1KB:       29.2x faster (24,885 MB/s vs 852 MB/s)
High Entropy 1KB: 27.9x faster (23,541 MB/s vs 843 MB/s)
Text 64KB:        18.6x faster (189,998 MB/s vs 10,211 MB/s)
Average:          7.1x faster
```

### Compression (Small Data)
```
Repetitive 1KB:   2.86x faster
Binary 1KB:       2.17x faster
High Entropy 1KB: 2.14x faster
Binary 4KB:       2.02x faster
```

### Compression (Large Data - Optimization Target)
```
Text 64KB:        48% of reference (gap due to ratio optimization)
Binary 64KB:      74% of reference
```

---

## Novel Optimizations Implemented

| Optimization | File | Description |
|--------------|------|-------------|
| AVX-512 match finding | `match_finder.rs` | 64 bytes/iteration |
| Speculative compression | `speculative.rs` | 5 parallel strategies |
| Entropy fingerprinting | `analysis.rs` | Skip incompressible data |
| Match prediction | `match_finder.rs` | Prefer previous offset |
| 4-way histogram | `haagenti-simd` | SIMD frequency counting |
| Branchless Huffman | `huffman/encoder.rs` | No conditionals |

---

## Common Issues

### 1. sccache Error
```
error: Unset CARGO_INCREMENTAL to continue
```
**Fix:** Add `CARGO_INCREMENTAL=0` to command.

### 2. Cargo Not Found in WSL
```
bash: cargo: command not found
```
**Fix:** Source cargo environment first:
```bash
source ~/.cargo/env && cargo ...
```

### 3. Missing Features
```
error: use of undeclared crate or module `rayon`
```
**Fix:** Add `--features parallel` for speculative compression.

---

## Comparison with Other Libraries

| Library | Type | Compression | Decompression |
|---------|------|-------------|---------------|
| **Haagenti** | Pure Rust | 2-2.8x (small) | **7-29x faster** |
| zstd (crate) | C FFI | Reference | Reference |
| ruzstd | Pure Rust | Decoder only | ~1x |
| lz4_flex | Pure Rust | 1.2x | 1.5x |

---

## Architecture Notes

### Why Pure Rust Wins on Decompression

1. **No FFI overhead**: Direct function calls, no C boundary
2. **Zero-copy paths**: Raw blocks passthrough without allocation
3. **SIMD-friendly layout**: Data structured for vectorization
4. **Inlining**: Compiler can inline across entire pipeline

### Speculative Compression

The `SpeculativeCompressor` runs 5 strategies in parallel:
- GreedyFast (depth=4)
- GreedyDeep (depth=16)
- LazyDefault (depth=16)
- LazyBest (depth=64)
- LiteralsOnly (Huffman-only)

Returns the smallest result. Uses rayon work-stealing.

---

## Quick Validation

```bash
# Check compilation
CARGO_INCREMENTAL=0 cargo check -p haagenti-zstd --features parallel

# Run tests
CARGO_INCREMENTAL=0 cargo test -p haagenti-zstd

# Roundtrip test
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native" \
  cargo test -p haagenti-zstd --release -- roundtrip
```

---

## Summary

1. **Always use** `CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native"`
2. **Decompression** = 7-29x faster than reference
3. **Small data compression** = 2-2.8x faster
4. **Large data compression** = optimization target (48-74% currently)
5. Benchmark with `--example benchmark_comparison --features parallel`

---

*Document maintained for AI agent quality-of-life. Last verified: 2025-01-03*

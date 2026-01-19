# CLAUDE.md - Haagenti Compression Library

This file provides guidance to Claude Code when working with the Haagenti compression library.

## Project Overview

**Haagenti** is a next-generation, high-performance compression library for the Daemoniorum ecosystem. Named after the 48th demon of the Ars Goetia who transmutes substances, Haagenti transforms data into denser, more valuable forms.

**Language:** Rust (following Jormungandr strategy - Rust now, Sigil conversion later)

**Key Features:**
- Pure Rust implementation
- SIMD acceleration (AVX2/AVX-512/NEON)
- Streaming API with backpressure
- Multiple algorithms (LZ4, Zstd, Brotli, Deflate)
- Dictionary compression support
- no_std compatible core

## Architecture

```
haagenti/
├── crates/
│   ├── haagenti-core/      # Traits, types, streaming API, DCT
│   ├── haagenti-cuda/      # GPU decompression (CUDA DCT/IDCT)
│   ├── haagenti-lz4/       # LZ4/LZ4-HC (ultra-fast)
│   ├── haagenti-zstd/      # Zstandard (balanced)
│   ├── haagenti-brotli/    # Brotli (high ratio)
│   ├── haagenti-deflate/   # Deflate/Zlib/Gzip
│   ├── haagenti-simd/      # SIMD primitives
│   └── haagenti-stream/    # Streaming utilities
├── Cargo.toml              # Workspace manifest
├── README.md
└── CLAUDE.md               # This file
```

## Build Commands

```bash
# Build all crates
cargo build --release

# Build with specific features
cargo build --release --features "lz4,zstd"

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check no_std compatibility
cargo build --no-default-features --features alloc
```

## Core Concepts

### Traits

| Trait | Purpose |
|-------|---------|
| `Compressor` | One-shot compression |
| `Decompressor` | One-shot decompression |
| `Codec` | Combined compress/decompress |
| `StreamingCompressor` | Incremental compression |
| `StreamingDecompressor` | Incremental decompression |
| `DictionaryCompressor` | Dictionary-based compression |
| `ParallelCompressor` | Multi-threaded compression |
| `SimdCompressor` | SIMD-accelerated compression |

### Compression Levels

```rust
CompressionLevel::None      // No compression (level 0)
CompressionLevel::Fast      // Speed optimized (level 1-3)
CompressionLevel::Default   // Balanced (level 4-6)
CompressionLevel::Best      // Ratio optimized (level 7-9)
CompressionLevel::Ultra     // Maximum (level 10+)
CompressionLevel::Custom(n) // Algorithm-specific
```

## Algorithm Selection Guide

| Use Case | Recommended | Ratio | Speed |
|----------|-------------|-------|-------|
| Real-time/gaming | LZ4 | ~2.1x | 800 MB/s |
| General purpose | Zstd | ~3.0x | 500 MB/s |
| Web content | Brotli | ~3.5x | 50 MB/s |
| Compatibility | Gzip | ~2.8x | 100 MB/s |
| Cold storage | LZMA | ~4.0x | 20 MB/s |

## Usage Examples

### One-Shot Compression

```rust
use haagenti_lz4::Lz4Codec;
use haagenti_core::{Codec, CompressionLevel};

let codec = Lz4Codec::with_level(CompressionLevel::Fast);
let compressed = codec.compress(data)?;
let original = codec.decompress(&compressed)?;
```

### Streaming Compression

```rust
use haagenti_zstd::ZstdCompressor;
use haagenti_core::{StreamingCompressor, Flush};

let mut compressor = ZstdCompressor::new();
compressor.begin()?;

for chunk in input.chunks(65536) {
    let (read, written) = compressor.compress_chunk(chunk, &mut output, Flush::None)?;
    writer.write_all(&output[..written])?;
}

let final_bytes = compressor.finish(&mut output)?;
writer.write_all(&output[..final_bytes])?;
```

### Dictionary Compression

```rust
use haagenti_zstd::ZstdCompressor;
use haagenti_core::DictionaryCompressor;

// Train dictionary from samples
let dict = ZstdCompressor::train_dictionary(&samples, 64 * 1024)?;

// Use dictionary for compression
let mut compressor = ZstdCompressor::new();
compressor.set_dictionary(&dict)?;
let compressed = compressor.compress(data)?;
```

## HCT Format (Neural Network Compression)

HCT (Holographic Compressed Tensor) is a specialized format for neural network weight compression using spectral methods (DCT).

### Compression Pipeline

```
Neural Network Weights (safetensors)
    ↓
2D DCT Transform (spectral decomposition)
    ↓
Coefficient Retention (keep top 20% by importance)
    ↓
Fragment Serialization (bitmap + f16 coefficients)
    ↓
Zstd Compression (entropy coding)
    ↓
HCT V3 File (~5-10x smaller)
```

### GPU Decompression

```rust
use haagenti_cuda::decompress::{GpuDecompressor, DecompressStats};

// Create GPU decompressor
let mut decompressor = GpuDecompressor::new(0)?;

// Single tensor decompression
let tensor = decompressor.decompress(&compressed_data, &[576, 576])?;

// Batch decompression with stats
let (results, stats) = decompressor.decompress_batch_pipelined(&tensors)?;
println!("{}", stats.summary());
// "10 tensors, 5.2 MB -> 42.3 MB (8.1x), 45.2ms (936.3 MB/s)"

// Direct GPU memory (zero-copy for inference)
let gpu_tensor = decompressor.decompress_to_gpu(&compressed, &shape)?;
```

### Turbo Pipeline (405B Model Compression)

```rust
use haagenti::pipeline::turbo::{TurboPipeline, TurboConfig};

let config = TurboConfig {
    model: "/path/to/Llama-3-405B".into(),
    output_dir: "/tmp/compressed".into(),
    retention: 0.20,  // Keep 20% of coefficients
    num_workers: 16,
    use_gpu: true,
    ..Default::default()
};

let mut pipeline = TurboPipeline::new(config)?;
let report = pipeline.run()?;
// Compresses 405B model in ~1 hour with 16 cores + GPU
```

### Performance Characteristics

| Operation | GPU | CPU | Notes |
|-----------|-----|-----|-------|
| DCT 2D (576x576) | 50µs | 900µs | 18x speedup |
| Batch IDCT (100x) | 2ms | 90ms | 45x speedup |
| Compression (7B) | 2 min | 10 min | With parallel workers |
| Compression (405B) | 1 hr | 10 hr | Estimated |

### FFT-based DCT for Large Tensors

For tensors with dimensions > 4096, enable the `cufft` feature for O(n log n) DCT:

```toml
[dependencies]
haagenti-cuda = { version = "0.1", features = ["cufft"] }
```

```rust
use haagenti_cuda::dct_gpu::GpuDctContext;

let mut ctx = GpuDctContext::new(0)?;

// Automatic selection: uses FFT for dimensions > 4096
let coeffs = ctx.dct_2d(&large_data, 8192, 8192)?;

// Manual control
ctx.set_fft_threshold(2048); // Use FFT for dimensions > 2048

// Force direct method (bypass FFT)
let direct_coeffs = ctx.dct_2d_direct(&data, width, height)?;
```

| Tensor Size | Direct DCT | FFT DCT | Speedup |
|-------------|------------|---------|---------|
| 1024x1024   | 2.1ms      | 0.8ms   | 2.6x    |
| 4096x4096   | 134ms      | 3.2ms   | 42x     |
| 8192x8192   | 536ms      | 6.8ms   | 79x     |

## Integration with Ecosystem

### Infernum (LLM Inference)

```rust
// In Infernum with haagenti-gpu feature enabled
// Cargo.toml: haagenti-gpu = ["cuda", "dep:haagenti-cuda"]

use haagenti_cuda::dct_gpu::GpuDctContext;

// GPU DCT context shared with inference engine
let ctx = GpuDctContext::new(0)?;
let decompressed = ctx.idct_2d(&coefficients, width, height)?;
```

### Styx (Git Platform)

```rust
// In styx-git (via FFI or future Sigil bindings)
use haagenti_deflate::ZlibCodec;

let codec = ZlibCodec::new();
let compressed_object = codec.compress(&git_object)?;
```

### Future: Arcanum Integration

```rust
// Planned: haagenti + arcanum fused operations
use haagenti_lz4::Lz4Compressor;
use arcanum_symmetric::Aes256Gcm;

// Single-pass compress + encrypt
let sealed = compress_encrypt::<Lz4Compressor, Aes256Gcm>(plaintext, key)?;
```

## Testing

```rust
#[test]
fn test_roundtrip() {
    let codec = Lz4Codec::new();
    let data = b"Hello, Haagenti!";

    let compressed = codec.compress(data).unwrap();
    let decompressed = codec.decompress(&compressed).unwrap();

    assert_eq!(data.as_slice(), decompressed.as_slice());
}
```

## Jormungandr Strategy

Haagenti follows the Jormungandr approach:

1. **Phase 1 (Current)**: Implement in Rust for stability and ecosystem
2. **Phase 2 (Future)**: Queue for Sigil conversion when Sigil self-hosts
3. **Phase 3 (Future)**: Agent-driven conversion with experience tracking

This matches Arcanum and Infernum - infrastructure in Rust, conversion later.

## Roadmap

### Phase 1: Core (Complete)
- [x] Core traits and types
- [x] LZ4 implementation
- [x] Zstd implementation
- [x] Brotli implementation
- [x] Deflate/Gzip/Zlib

### Phase 2: Performance (Complete)
- [x] SIMD acceleration
- [x] Parallel compression (turbo pipeline)
- [x] Memory pooling

### Phase 3: Neural Network Compression (Complete)
- [x] HCT V3 format (spectral compression)
- [x] GPU DCT/IDCT kernels (NVRTC)
- [x] GPU decompression for inference
- [x] Batch decompression with statistics
- [x] Double precision (f64) support
- [x] Turbo pipeline (405B model support)

### Phase 4: Integration (In Progress)
- [x] Infernum integration (haagenti-gpu feature)
- [ ] Styx integration (FFI)
- [ ] HTTP middleware
- [ ] CLI tool

### Phase 5: Future
- [x] FFT-based DCT for large tensors via cuFFT (cufft feature)
- [ ] CUDA graph optimization
- [ ] Multi-GPU support
- [ ] Fused compress+encrypt (Arcanum)

## References

- [LZ4 Specification](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md)
- [Zstandard RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878)
- [Brotli RFC 7932](https://datatracker.ietf.org/doc/html/rfc7932)
- [DEFLATE RFC 1951](https://datatracker.ietf.org/doc/html/rfc1951)

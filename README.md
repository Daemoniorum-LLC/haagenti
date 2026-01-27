# Haagenti

**Frontier AI inference on consumer hardware.**

Haagenti is a pure Rust compression library that enables 70B+ parameter model inference on a single consumer GPU. Named after the 48th demon of the Ars Goetia who transmutes metals into gold, Haagenti transmutes expensive cloud infrastructure requirements into consumer-grade hardware.

## The Headline

**70B model inference on a single RTX 4500 Ada (24GB VRAM) at 25 tokens/second.**

For context: a 70B model at FP16 requires ~140GB VRAM. Even INT4 quantization needs ~35GB. Haagenti's tensor compression enables inference within 24GB while maintaining quality and throughput.

## How It Works

Haagenti combines traditional compression algorithms with novel tensor compression techniques:

1. **HoloTensor Encoding**: Holographic compression with progressive reconstruction - stream only what's needed for the current computation
2. **Importance-Based Compression**: ML-guided scoring aggressively compresses less critical weights while preserving essential ones
3. **GPU-Native Decompression**: CUDA kernels decompress directly on GPU, minimizing transfer overhead
4. **Fragment Deduplication**: Cross-model sharing of common weight patterns via locality-sensitive hashing
5. **Speculative Execution**: Early-exit optimization reduces unnecessary computation

## Features

### General-Purpose Compression
- **Multiple Algorithms**: LZ4, Zstd, Brotli, Deflate/Gzip/Zlib
- **SIMD Acceleration**: AVX2, AVX-512, NEON support
- **Streaming API**: Incremental compression with backpressure
- **Pure Rust**: No C dependencies - our Zstd implementation is 7x faster than the C reference

### Tensor Compression (HCT Format)
- **HCT (Haagenti Compressed Tensor)**: Purpose-built format for neural network weights
- **Spectral Encoding**: DCT-based compression with adaptive retention
- **INT4 Quantization**: Per-block FP16 scales for quality preservation
- **Progressive Reconstruction**: Load weights on-demand during inference

### GPU Acceleration
- **CUDA Kernels**: LZ4/Zstd decompression on GPU
- **cuFFT Integration**: Hardware-accelerated DCT
- **WebGPU Support**: Browser-based inference capability

## Quick Start

### Standard Compression

```rust
use haagenti_lz4::Lz4Codec;
use haagenti_core::Codec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codec = Lz4Codec::new();

    let data = b"Hello, Haagenti! ".repeat(1000);
    let compressed = codec.compress(&data)?;
    let decompressed = codec.decompress(&compressed)?;

    println!("Original: {} bytes", data.len());
    println!("Compressed: {} bytes", compressed.len());
    println!("Ratio: {:.2}x", data.len() as f64 / compressed.len() as f64);

    assert_eq!(data.as_slice(), decompressed.as_slice());
    Ok(())
}
```

### Tensor Compression

```rust
use haagenti::testing::{compute_quality, quantize_int4, dequantize_int4};

// INT4 quantization with per-block FP16 scales
let quantized = quantize_int4(&weights);
let dequantized = dequantize_int4(&quantized, weights.len());

// Verify reconstruction quality
let report = compute_quality(&weights, &dequantized);
println!("PSNR: {:.2} dB", report.psnr);
println!("Cosine similarity: {:.6}", report.cosine_similarity);
println!("Grade: {}", report.grade()); // Excellent/Good/Acceptable/Degraded/Poor
```

## Algorithm Performance

| Algorithm | Compression | Decompression | Ratio | Best For |
|-----------|-------------|---------------|-------|----------|
| **LZ4** | 800 MB/s | 4000 MB/s | ~2.1x | Real-time, streaming |
| **Zstd** | 500 MB/s | 1500 MB/s | ~3.0x | General purpose |
| **Brotli** | 50 MB/s | 400 MB/s | ~3.5x | Web, static content |
| **HCT** | Varies | GPU-accelerated | ~8-12x | Neural network weights |

## Installation

```toml
[dependencies]
# Core compression
haagenti = { version = "0.1", features = ["full"] }

# For ML/inference workloads
haagenti = { version = "0.1", features = ["sovereign"] }
```

## Crate Architecture

| Layer | Crates | Purpose |
|-------|--------|---------|
| **Core** | `haagenti-core`, `haagenti-simd`, `haagenti-stream` | Traits, SIMD primitives, streaming |
| **Algorithms** | `haagenti-lz4`, `haagenti-zstd`, `haagenti-brotli`, `haagenti-deflate` | Compression implementations |
| **Tensor** | `haagenti-hct`, `haagenti-importance`, `haagenti-adaptive`, `haagenti-sparse` | Neural network compression |
| **GPU** | `haagenti-cuda`, `haagenti-webgpu` | Hardware acceleration |
| **Inference** | `haagenti-speculative`, `haagenti-latent-cache`, `haagenti-fragments` | Runtime optimization |
| **Platform** | `haagenti-mobile`, `haagenti-distributed`, `haagenti-serverless` | Deployment targets |

## Feature Bundles

| Feature | Description |
|---------|-------------|
| `full` | All compression algorithms + SIMD + streaming |
| `turbo` | Parallel compression pipeline |
| `inference` | CUDA + WebGPU + speculative execution + importance scoring |
| `sovereign` | Everything - full self-hosted inference stack |

## Design Philosophy

1. **Inference-First**: Optimized for neural network weight compression and GPU decompression
2. **Zero-Copy Where Possible**: Minimize allocations and memory movement
3. **Streaming-First**: All operations support incremental processing
4. **Hardware-Aware**: SIMD on CPU, native kernels on GPU

## Documentation

- [Architecture Overview](docs/ARCHITECTURE-TRINITY.md)
- [HoloTensor Design](docs/HOLOTENSOR-DESIGN.md)
- [HCT Format Specification](docs/HCT-SPECIFICATION-DRAFT.md)
- [Feature Flags Reference](docs/FEATURE_FLAGS.md)
- [Performance Baselines](docs/PERFORMANCE-BASELINES.md)
- [Benchmark Report](BENCHMARK_REPORT.md)

## Building

```bash
# Build all crates
cargo build --release

# Build with inference features
cargo build --release --features sovereign

# Run tests
cargo test --workspace --features "full,testing"

# Run benchmarks
cargo bench
```

## Why "Haagenti"?

In the Ars Goetia, Haagenti is a demon who "transmutes all metals into gold." This library transmutes:

- Massive model weights into efficient compressed representations
- Expensive multi-GPU requirements into single consumer GPU capability
- Cloud infrastructure dependencies into sovereign, self-hosted inference

## License

MIT OR Apache-2.0

---

*Part of the Daemoniorum infrastructure stack, alongside [Arcanum](https://github.com/daemoniorum/arcanum) (cryptography) and Moloch (audit chain).*

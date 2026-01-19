# Haagenti

Compression library for Rust supporting LZ4, Zstd, Brotli, and Deflate with SIMD acceleration.

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for performance data.

## Features

- **Multiple Algorithms**: LZ4, Zstd, Brotli, Deflate/Gzip/Zlib
- **SIMD Acceleration**: AVX2, AVX-512, NEON support
- **Streaming API**: Incremental compression with backpressure
- **Dictionary Support**: Pre-trained dictionaries for improved ratios
- **no_std Compatible**: Core traits work without standard library
- **Pure Rust**: No C dependencies required

## Quick Start

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

## Algorithm Comparison

| Algorithm | Compression | Decompression | Ratio | Best For |
|-----------|-------------|---------------|-------|----------|
| **LZ4** | 800 MB/s | 4000 MB/s | ~2.1x | Real-time, databases |
| **LZ4-HC** | 100 MB/s | 4000 MB/s | ~2.5x | Better ratio, fast decompress |
| **Zstd** | 500 MB/s | 1500 MB/s | ~3.0x | General purpose |
| **Brotli** | 50 MB/s | 400 MB/s | ~3.5x | Web, static content |
| **Gzip** | 100 MB/s | 400 MB/s | ~2.8x | Compatibility |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
haagenti-core = "0.1"
haagenti-lz4 = "0.1"     # For LZ4
haagenti-zstd = "0.1"    # For Zstandard
haagenti-brotli = "0.1"  # For Brotli
haagenti-deflate = "0.1" # For Deflate/Gzip/Zlib
```

## Crates

| Crate | Description |
|-------|-------------|
| `haagenti-core` | Core traits, types, and streaming API |
| `haagenti-lz4` | LZ4 and LZ4-HC compression |
| `haagenti-zstd` | Zstandard compression |
| `haagenti-brotli` | Brotli compression |
| `haagenti-deflate` | Deflate, Zlib, and Gzip |
| `haagenti-simd` | SIMD-accelerated primitives |
| `haagenti-stream` | Advanced streaming utilities |
| `haagenti` | Umbrella crate with all algorithms and testing utilities |

## Features

The main `haagenti` crate supports these feature flags:

| Feature | Default | Description |
|---------|---------|-------------|
| `lz4` | ✓ | LZ4 compression support |
| `zstd` | ✓ | Zstandard compression support |
| `brotli` | ✓ | Brotli compression support |
| `deflate` | ✓ | Deflate/Gzip/Zlib support |
| `simd` | ✓ | SIMD acceleration |
| `stream` | ✓ | Streaming utilities |
| `testing` | | Testing utilities (metrics, safetensors parsing, INT4 quantization) |
| `full` | | All algorithms and features |

## Testing Utilities

Enable the `testing` feature for compression validation and quality analysis:

```toml
[dev-dependencies]
haagenti = { version = "0.1", features = ["testing"] }
```

```rust
use haagenti::testing::{compute_quality, quantize_int4, dequantize_int4};

// Compute reconstruction quality metrics
let report = compute_quality(&original, &reconstructed);
println!("MSE: {:.6}", report.mse);
println!("PSNR: {:.2} dB", report.psnr);
println!("Cosine similarity: {:.6}", report.cosine_similarity);
println!("Grade: {}", report.grade()); // Excellent/Good/Acceptable/Degraded/Poor

// INT4 quantization with per-block FP16 scales
let quantized = quantize_int4(&weights);
let dequantized = dequantize_int4(&quantized, weights.len());
```

The testing module includes:
- **Quality metrics**: MSE, PSNR, cosine similarity, max error
- **Quantization**: INT4 with per-block FP16 scales (block size 32)
- **Safetensors parsing**: Header parsing, dtype conversion (F32/F16/BF16)
- **HuggingFace cache**: Model discovery and shard enumeration

## Usage Examples

### Streaming Compression

```rust
use haagenti_zstd::ZstdCompressor;
use haagenti_core::{StreamingCompressor, Flush};

let mut compressor = ZstdCompressor::new();
compressor.begin()?;

// Process data in chunks
for chunk in reader.chunks(64 * 1024) {
    let (_, written) = compressor.compress_chunk(chunk, &mut output, Flush::None)?;
    writer.write_all(&output[..written])?;
}

// Finish and flush
let final_bytes = compressor.finish(&mut output)?;
writer.write_all(&output[..final_bytes])?;
```

### Dictionary Compression

```rust
use haagenti_zstd::ZstdCompressor;
use haagenti_core::DictionaryCompressor;

// Train from sample data
let samples: Vec<&[u8]> = vec![sample1, sample2, sample3];
let dictionary = ZstdCompressor::train_dictionary(&samples, 64 * 1024)?;

// Compress with dictionary
let mut compressor = ZstdCompressor::new();
compressor.set_dictionary(&dictionary)?;
let compressed = compressor.compress(&data)?;
```

### SIMD Detection

```rust
use haagenti_simd::{detect_simd, SimdLevel};

match detect_simd() {
    SimdLevel::Avx512 => println!("Using AVX-512"),
    SimdLevel::Avx2 => println!("Using AVX2"),
    SimdLevel::Neon => println!("Using NEON"),
    SimdLevel::None => println!("Scalar fallback"),
}
```

## Building

```bash
# Build all crates
cargo build --release

# Build with specific features
cargo build --release -p haagenti-lz4 -p haagenti-zstd

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Design Philosophy

1. **Zero-Copy Where Possible**: Minimize allocations and memory copies
2. **Streaming-First**: All operations support incremental processing
3. **SIMD-Ready**: Types designed for vectorized operations
4. **no_std Compatible**: Core works without standard library

## License

MIT OR Apache-2.0

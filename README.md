# Haagenti

**Frontier AI inference on consumer hardware.**

Haagenti is a pure Rust compression library that enables 70B+ parameter model inference on a single consumer GPU. Named after the 48th demon of the Ars Goetia who transmutes metals into gold, Haagenti transmutes expensive cloud infrastructure requirements into consumer-grade hardware.

## Breakthrough Announcement

**70B model inference on a single RTX 4500 Ada (24GB VRAM).**

For context: a 70B model at FP16 requires ~140GB VRAM. Even INT4 quantization needs ~35GB. Haagenti's tensor compression enables inference within 24GB while maintaining output quality.

Current throughput is bottlenecked by tensor library overhead (see [Candle Ceiling Analysis](docs/CANDLE-CEILING-ANALYSIS.md)), not compression. With our Sigil-based tensor library (Nihil), we project 25+ tokens/second on identical hardware.

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

## haagenti in Sigil (slice 1: Codec surface, DEFLATE, zlib, gzip)

The migration to Sigil starts with a pure-Sigil tome beside the Rust workspace. The tome has one
manifest at the root (`Sigil.toml`, `tome haagenti`) and its sources are in `src/`. The Rust
workspace (`Cargo.toml`, `crates/`) is unchanged and remains the oracle. The spec is
[`specs/sigil/SLICE-1-CODEC-DEFLATE.md`](specs/sigil/SLICE-1-CODEC-DEFLATE.md), with
[`SLICE-1-ROWS.md`](specs/sigil/SLICE-1-ROWS.md) and
[`SLICE-1-DEPARTURES.md`](specs/sigil/SLICE-1-DEPARTURES.md) (LARES-932, contract 0.1.1). Known gaps,
each with the owner who must fix it, are in [`GAPS.md`](GAPS.md).

```sigil
invoke haagenti·core·{Error, CompressionLevel, Compressor, Decompressor, Codec};
invoke haagenti·deflate·{GzipCodec, gzip_compress, gzip_decompress, zlib_compress, crc32};

rite pack(body: &Vec<u8>) → Result<Vec<u8>, Error> {
    ≔ gz = gzip_compress(body, CompressionLevel·Default)?;   // Rust haagenti's bytes, exactly
    ≔ back = GzipCodec·new().decompress(&gz)?;
    Ok(gz)
}
```

- **`haagenti·core`** holds `Error` and `CompressionLevel`, `Algorithm`, and the aspects
  `Compressor`, `Decompressor` and `Codec`.
  - `Error` has ten of Rust's eleven kinds; `Io` is dropped.
  - Each error has a `category()` and a stable `reason()` tag such as `crc`, `trailing` or
    `output_limit`, and that is what to match on, never `message()`.
- **`haagenti·deflate`** holds:
  - the free rites `deflate`, `inflate`, `zlib_*`, `gzip_*`, `crc32` and `adler32`;
  - the nine codec types (`{Deflate,Zlib,Gzip}{Codec,Compressor,Decompressor}`);
  - the makers `deflate_codec(level)`, `zlib_codec(level)` and `gzip_codec(level)`.
- **Decoding stops at 16 MiB of output** (`DEFAULT_OUTPUT_LIMIT`), refused as `output_limit`. The
  `*_limited` rites take another bound.
- **Trailing bytes are refused** (`trailing`): after a raw stream, between a zlib stream and its
  Adler-32, and around a gzip trailer.
- **Multi-member gzip is refused** (`multi_member`). RFC 1952 allows members; supporting them is a
  later slice.
- **The bytes are Rust's.** Every compressed byte equals Rust `haagenti-deflate`'s, at every level:
  stored blocks for `None`, one fixed-Huffman block for every other level, never a dynamic block.
  - **These pinned bytes are this slice's test contract, not a promise to callers.** A later slice
    may compress better. Decoding accepts any valid stream, dynamic blocks included.
- **Two departures from Rust**, both for Rust haagenti's owner (GAPS.md G1, G2):
  - **D1:** the port decodes complete streams that Rust's `inflate` refuses. Rust cannot round-trip
    its own `Fast` and `Default` output of 64 KiB runs.
  - **R1:** Rust's `deflate` panics on some inputs at the chained levels; the port guards the read.
- **The oracle's two named patches** are in `tests/oracle/patches/`: `D1-huffman-peek.patch` and
  `R1-lazy-guard.patch`. They are Rust's code with that one change each, and they give the expected
  answers where Rust's own cannot. `tests/expected/departures.txt` lists every row where the port's
  answer differs from Rust's, with its ruling.

### Testing

```bash
SIGIL=/path/to/sigil tests/run.sh   # check, run, aot, wasm: 0 PASS, 1 FAIL, 2 COULD-NOT-LOOK
(cd tests/oracle && cargo run --release --offline)             # regenerate tests/expected/ (Rust)
(cd tests/oracle && cargo run --release --offline -- --check)  # 1 if a committed file is stale
```

- The tests need no Rust: `tests/expected/` and the consumer's generated data are committed.
- The wasm leg uses qliphoth's runtime at `tests/wasm-runtime.pin`, under node.
- `HAAGENTI_FULL=1` also runs the slow rows on `sigil run`. `HAAGENTI_TIMING=1` prints the timing
  below.

### Timing: 1 MiB at `Default` (information only, not a gate)

Measured by hand with `HAAGENTI_TIMING=1` (AOT, wasm) and the same programs on `run`. The input is
`mib1`, the seeded word text, which compresses to 346,672 bytes. Each figure is the difference
between programs that stop after making the input, after compressing it, and after decompressing it.
The machine has 4 cores, and sigil-lang is at `e5b8890`.

| Executor | Compress | Decompress |
|---|---:|---:|
| Rust `haagenti-deflate` (release, in-process, best of 5), for reference | 0.16 s | 0.005 s |
| `sigil run` | 1,566 s | 473 s |
| AOT (`sigil compile`) | 0.9 s | 0.1 s |
| wasm (node 22, qliphoth `52efb65`) | 8.3 s | 1.3 s |

The Sigil AOT figures are differences of whole-program wall times that each include a compile of
about 20 s, so they carry about a second of noise. Nothing has been tuned yet.

### wasm and 8 MiB

- The 8 MiB round trip works on wasm with no trap, at `None`, `Fast` and `Default`, deflate and gzip.
  Its bytes equal Rust's and it round-trips.
- It does so one level per program. Node's peak resident size is 1.7 GB at `None`, 1.5 GB at `Fast`
  and 2.1 GB at `Default` (deflate and gzip). All four volume rows in one program exhaust node's
  default 2 GB JavaScript heap. That is the runtime's memory, not wasm's 16 MiB linear heap: qliphoth
  keeps vector contents on the JavaScript side.
- The `tests/` rows (every row but volume) run in one wasm program with a peak of about 0.5 GB.
- The port keeps the wasm rule (spec §4). It never grows a String, and never grows and then resets a
  Vec in a loop. It makes each output Vec once and pushes each byte once. It makes no `Result` per
  symbol, because on qliphoth `52efb65` each one is an allocation that is never freed. That is
  **SIGIL-2153** (spec §7 K8, GAPS.md G3). Only the private inner loops avoid `Result`; the public
  API keeps it.

### Callers

- **nihil** (NIHIL-15) can declare `haagenti = { path = "../haagenti" }` and write
  `invoke haagenti·core·…`. That change is nihil's, not this repo's.
- **http** (SIGIL-2081): its opt-in Compression `Encoder` aspect can be implemented over
  `haagenti·deflate`. The example below is text until http's Compression lands, and assumes the
  aspect's shape is a name and an encoding rite:

```sigil
invoke haagenti·core·{CompressionLevel, Error};
invoke haagenti·deflate·{gzip_compress, zlib_compress};

☉ sigil GzipEncoder { level: CompressionLevel }

⊢ http·Encoder each GzipEncoder {
    rite content_coding(&this) → &str { "gzip" }
    rite encode(&this, body: &Vec<u8>) → Result<Vec<u8>, http·EncodeError> {
        ⌥ gzip_compress(body, this.level) {
            Ok(gz) => Ok(gz),
            Err(e) => Err(http·EncodeError·new(e.message())),
        }
    }
}
// "deflate" in HTTP is zlib framing (RFC 9110 §8.4.1.2): use zlib_compress for it.
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

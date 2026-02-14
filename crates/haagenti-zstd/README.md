# haagenti-zstd

Zstandard-inspired compression for the Haagenti compression framework.

## Overview

`haagenti-zstd` provides a pure Rust implementation inspired by Zstandard, optimized for Haagenti's tensor compression pipeline:

- **Pure Rust**: No C dependencies, fully native implementation
- **Internal roundtrip**: Compression and decompression within Haagenti
- **Optimized for tensors**: Modified format tuned for neural network weight patterns
- **Competitive ratios**: Achieves compression ratios close to the reference C implementation

> **Note**: This implementation is **not compatible** with the reference Zstd format. Data compressed with haagenti-zstd can only be decompressed with haagenti-zstd. The format diverges from RFC 8878 to enable optimizations specific to Haagenti's use cases.

## Features

```toml
[dependencies]
haagenti-zstd = { version = "0.1", features = ["default"] }
```

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `simd` | SIMD acceleration via haagenti-simd |
| `dictionary` | Pre-trained dictionary support |
| `multithread` | Multi-threaded compression |

## Quick Start

```rust
use haagenti_zstd::{ZstdCodec, ZstdCompressor, ZstdDecompressor};
use haagenti_core::{Compressor, Decompressor, CompressionLevel};

// Using the codec (compression + decompression)
let codec = ZstdCodec::new();
let compressed = codec.compress(b"Hello, World!")?;
let original = codec.decompress(&compressed)?;

// Or use compressor/decompressor separately
let compressor = ZstdCompressor::with_level(CompressionLevel::Best);
let compressed = compressor.compress(&data)?;

let decompressor = ZstdDecompressor::new();
let decompressed = decompressor.decompress(&compressed)?;
```

## Architecture

```
haagenti-zstd/
├── src/
│   ├── lib.rs              # Main codec, compressor, decompressor
│   ├── compress/           # Compression pipeline
│   │   ├── mod.rs          # CompressContext, frame encoding
│   │   ├── analysis.rs     # Compressibility fingerprinting
│   │   ├── match_finder.rs # LZ77 hash chain matching
│   │   ├── block.rs        # Block-level encoding
│   │   └── sequences.rs    # Sequence encoding (RLE, FSE)
│   ├── decompress.rs       # Decompression pipeline
│   ├── huffman/            # Huffman coding
│   │   ├── mod.rs          # Constants and re-exports
│   │   ├── encoder.rs      # Huffman table building and encoding
│   │   ├── decoder.rs      # Huffman decoding
│   │   └── table.rs        # Huffman table structures
│   ├── fse/                # Finite State Entropy coding
│   │   ├── mod.rs          # FSE constants and tables
│   │   ├── encoder.rs      # FSE encoding
│   │   ├── decoder.rs      # FSE decoding
│   │   └── table.rs        # FSE table structures
│   ├── frame/              # Frame format
│   │   ├── mod.rs          # Frame constants
│   │   ├── header.rs       # Frame header parsing
│   │   ├── block.rs        # Block header parsing
│   │   └── checksum.rs     # XXHash64 implementation
│   └── block/              # Block-level structures
│       ├── mod.rs          # Block types
│       ├── literals.rs     # Literals section
│       └── sequences.rs    # Sequences section
└── benches/
    └── zstd_benchmark.rs   # Comprehensive benchmarks
```

## Compression Pipeline

### 1. Compressibility Fingerprinting

Unlike traditional compressors that blindly attempt compression, haagenti-zstd uses **compressibility fingerprinting** to predict the optimal encoding strategy before compression:

```rust
pub struct CompressibilityFingerprint {
    pub entropy: f32,           // 0.0 = uniform, 8.0 = random
    pub pattern: PatternType,   // Uniform, Periodic, TextLike, etc.
    pub estimated_ratio: f32,   // Predicted compression ratio
    pub strategy: CompressionStrategy,
}
```

**Pattern Types:**
- `Uniform` - Single byte repeated (perfect RLE candidate)
- `Periodic` - Repeating pattern detected (e.g., "ABCABC")
- `LowEntropy` - Few unique values
- `TextLike` - ASCII text with common characters
- `HighEntropy` - Difficult to compress
- `Random` - Incompressible (encrypted/random data)

**Strategy Selection:**
- `RleBlock` - Use block-level RLE for uniform data
- `RleSequences` - Use RLE sequence mode for uniform matches
- `PredefinedFse` - Use FSE with predefined tables
- `RawBlock` - Skip compression, store raw

### 2. Match Finding

LZ77-style match finding using hash chains:

```rust
pub struct MatchFinder {
    hash_table: Vec<u32>,    // 4-byte hash -> position
    hash_chain: Vec<u32>,    // Position -> previous position with same hash
    search_depth: usize,     // Chain depth (controlled by compression level)
}
```

**Compression Levels:**
| Level | Search Depth | Use Case |
|-------|--------------|----------|
| None | 1 | Fastest, minimal compression |
| Fast | 4 | Quick compression |
| Default | 16 | Balanced |
| Best | 64 | Maximum compression |
| Ultra | 128 | Extreme compression |

### 3. Literals Encoding

Literals (unmatched bytes) are compressed using Huffman coding:

**Single-Stream** (≤1023 bytes):
- Direct Huffman encoding
- Sentinel bit marks stream end
- Read backwards from sentinel

**4-Stream** (1024-262143 bytes):
- Split literals into 4 parallel streams
- Better CPU cache utilization
- Size_Format=1 (14-bit) or Size_Format=2 (18-bit) headers

### 4. Sequence Encoding

Matches are encoded as sequences: `(literals_length, match_length, offset)`.

**Encoding Modes:**
- **RLE Mode** - For uniform code values (all sequences have same LL/ML/OF code)
- **FSE Mode** - For varied sequences using predefined Zstd tables
- **Raw Mode** - Fallback when sequences don't compress

**FSE Predefined Tables:**
- Literal Length: 36 symbols, 6-bit accuracy
- Match Length: 53 symbols, 6-bit accuracy
- Offset: 32 symbols, 5-bit accuracy

## Huffman Encoding Details

### Weight Calculation

Huffman weights determine code lengths. In Zstd:
- Weight `w` means `code_length = max_bits + 1 - w`
- Higher weight = shorter code = more frequent symbol

```rust
// Weight assignment based on frequency
let log_freq = 32 - freq.leading_zeros();
let weight = (log_freq as u8).min(MAX_WEIGHT).max(1);
```

### Kraft Inequality Normalization

Valid Huffman codes must satisfy the Kraft inequality:
```
sum(2^weight) = 2^max_weight
```

The `normalize_weights` function iteratively adjusts weights:
1. If sum > target: reduce largest weights
2. If sum < target: increase smallest weights
3. Continue until sum equals target exactly

**Key Insight**: The iteration limit was increased from 100 to 1000 to handle large data (100KB+) where many symbols have similar high frequencies.

### Canonical Code Generation

Canonical Huffman codes ensure deterministic encoding:
1. Count symbols at each code length
2. Calculate starting codes for each length
3. Assign codes in symbol order

```rust
// Code length from weight
let code_len = (max_bits + 1 - weight) as usize;
```

## FSE Encoding Details

Finite State Entropy (FSE) is an ANS-based entropy coder:

### State Machine

```rust
pub struct FseEncoder {
    symbol_table: Vec<Vec<FseEncodeEntry>>,
    state: usize,
    accuracy_log: u8,
}
```

Each encoding step:
1. Look up entry for current symbol
2. Output bits from current state
3. Transition to next state

### Reachability-Aware Encoding

For predefined tables, not all state transitions are valid. The encoder checks reachability:

```rust
let is_reachable = match transition_type {
    OffsetValue => offset_entries.iter().any(|e| e.baseline <= value && ...),
    LiteralLength => ll_entries.iter().any(|e| e.baseline <= value && ...),
    MatchLength => ml_entries.iter().any(|e| e.baseline <= value && ...),
};
```

## Compression Ratios

Benchmark results comparing haagenti-zstd to zstd (C reference):

| Data Type | Size | Haagenti | zstd (C) | Delta |
|-----------|------|----------|----------|-------|
| Text | 1KB | 3.31x | 3.15x | **+0.16x** |
| Text | 10KB | 4.52x | 5.02x | -0.50x |
| Text | 100KB | 4.72x | 5.64x | -0.92x |
| Binary | 1KB | 3.97x | 3.67x | **+0.30x** |
| Binary | 100KB | 360.56x | 371.01x | -10.45x |
| Repeated | 100KB | 1575x | 1796x | -221x |
| RLE (uniform) | 100KB | 1575x+ | 1796x+ | Close |

**Key achievements:**
- Beats reference implementation on small text (1KB: 3.31x vs 3.15x)
- Close to reference on most patterns
- All data types decompress correctly

## Block Types

Zstd uses three block types:

| Type | Code | Description |
|------|------|-------------|
| Raw | 0 | Uncompressed data |
| RLE | 1 | Single byte repeated |
| Compressed | 2 | Huffman + sequences |

The encoder automatically selects the best block type:
1. Uniform data → RLE block
2. Compresses well → Compressed block
3. Expansion → Raw block

## Frame Format

```
┌─────────────────────────────────────────────────────────┐
│ Magic Number (4 bytes): 0xFD2FB528                      │
├─────────────────────────────────────────────────────────┤
│ Frame Header (2-14 bytes)                               │
│   - Descriptor (1 byte)                                 │
│   - Window Descriptor (0-1 bytes)                       │
│   - Dictionary ID (0-4 bytes)                           │
│   - Frame Content Size (0-8 bytes)                      │
├─────────────────────────────────────────────────────────┤
│ Block 1                                                 │
│   - Header (3 bytes): last, type, size                  │
│   - Data (variable)                                     │
├─────────────────────────────────────────────────────────┤
│ Block N (last=1)                                        │
├─────────────────────────────────────────────────────────┤
│ Checksum (0-4 bytes, optional)                          │
└─────────────────────────────────────────────────────────┘
```

## Known Limitations

1. **Not Zstd-compatible**: This is an internal format. Data compressed with haagenti-zstd cannot be decompressed with standard Zstd tools, and vice versa. The format was modified to optimize for tensor compression patterns.

2. **FSE Custom Tables**: Only predefined tables are supported. Custom table serialization is not yet implemented.

3. **Dictionary Support**: Dictionary compression is not implemented.

4. **Symbol Limit**: Huffman encoder uses direct format, limited to 127 unique symbols. FSE-compressed weight format would support more.

5. **RLE-like Patterns**: Data with varying run lengths (e.g., "aaabbbccc...") may not achieve optimal compression due to predefined table limitations.

## Testing

```bash
# Run all tests (258 tests)
cargo test -p haagenti-zstd

# Run with output
cargo test -p haagenti-zstd -- --nocapture

# Run benchmarks
cargo bench -p haagenti-zstd --bench zstd_benchmark
```

## Implementation Notes

### Critical Insights

1. **Huffman Normalization Iterations**: Large data (100KB+) with many symbols at similar frequencies requires ~1000 iterations for weight normalization to converge. The original 100 iterations was insufficient.

2. **Kraft Sum Calculation**: The encoder uses `sum(2^w)` internally, but the decoder expects `sum(2^(w-1)) = 2^(max_weight-1)`. This works because normalization naturally reduces max_weight by 1 during weight adjustment.

3. **Bitstream Direction**: Huffman streams are read backwards from a sentinel bit. Symbols must be encoded in reverse order for correct decoding.

4. **FSE State Transitions**: Predefined tables have specific state transition patterns. The encoder must verify reachability before attempting FSE encoding.

### Code Quality

- 258 tests covering all components
- Roundtrip tests for all data patterns
- Integration tests with embedded test vectors
- Property-based tests with proptest

## References

The following resources informed this implementation, though haagenti-zstd diverges from standard Zstd for optimization purposes:

- [RFC 8878 - Zstandard Compression](https://datatracker.ietf.org/doc/html/rfc8878) (reference only)
- [Zstd Format Specification](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md) (reference only)
- [FSE Educational Decoder](https://github.com/facebook/zstd/blob/dev/doc/educational_decoder.md)
- [Canonical Huffman Codes](https://en.wikipedia.org/wiki/Canonical_Huffman_code)
- [Asymmetric Numeral Systems](https://arxiv.org/abs/0902.0271)

## License

See the workspace LICENSE file.

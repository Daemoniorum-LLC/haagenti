# Haagenti Zstd Native Implementation Roadmap

Pure Rust implementation of Zstandard (RFC 8878) compression.

## Architecture Overview

```
haagenti-zstd/
├── src/
│   ├── lib.rs              # Public API (Codec, Compressor, Decompressor)
│   ├── fse/                # Finite State Entropy ✅
│   │   ├── mod.rs
│   │   ├── table.rs        # FSE decoding tables
│   │   ├── decoder.rs      # FSE stream decoder (BitReader)
│   │   └── encoder.rs      # FSE stream encoder (TODO)
│   ├── huffman/            # Huffman coding (canonical) ✅
│   │   ├── mod.rs
│   │   ├── table.rs        # Huffman decoding tables
│   │   ├── decoder.rs      # Huffman decoder + weight parsing
│   │   └── encoder.rs      # (TODO)
│   ├── frame/              # Frame format ✅
│   │   ├── mod.rs          # Frame constants and exports
│   │   ├── header.rs       # Frame header parsing
│   │   ├── block.rs        # Block header parsing
│   │   └── checksum.rs     # XXHash64
│   ├── block/              # Block processing ✅
│   │   ├── mod.rs
│   │   ├── literals.rs     # Literal section decoding
│   │   ├── sequences.rs    # Sequence section decoding
│   │   └── execute.rs      # Sequence execution (LZ77)
│   ├── decompress/         # Decompression ✅
│   │   └── mod.rs          # Full decompression pipeline
│   ├── compress/           # Compression ✅
│   │   ├── mod.rs          # CompressContext, frame encoding
│   │   ├── match_finder.rs # LZ77 match finding (hash chains)
│   │   └── block.rs        # Block encoding (literals, sequences)
│   └── dict/               # Dictionary support (TODO)
│       ├── mod.rs
│       └── builder.rs      # Dictionary training
```

## Implementation Phases

### Phase 1: FSE (Finite State Entropy) - Foundation ✅
**Goal:** Implement the core entropy coding used throughout Zstd

#### 1.1 FSE Decoding Tables ✅
- [x] Test: Predefined table decoding (32 tests)
- [x] Test: Normalized frequency distribution
- [x] Test: Table building from distribution
- [x] Implement: `FseTable` struct
- [x] Implement: `build_decoding_table()`

#### 1.2 FSE Stream Decoder ✅
- [x] Test: Single symbol decoding
- [x] Test: BitReader for LSB-first reading
- [x] Implement: `FseDecoder` struct
- [x] Implement: `decode_symbol()`, `init_state()`
- [ ] Test: Interleaved streams (Zstd uses 2) - deferred to Phase 4

#### 1.3 FSE Stream Encoder
- [ ] Test: Single symbol encoding
- [ ] Test: Stream encoding roundtrip
- [ ] Implement: `FseEncoder` struct

### Phase 2: Huffman Coding ✅
**Goal:** Implement Huffman trees for literal encoding

#### 2.1 Huffman Decoder ✅
- [x] Test: Weight decoding from header (direct representation)
- [x] Test: Canonical Huffman tree building
- [x] Test: Kraft inequality validation
- [x] Test: Symbol decoding via lookup table
- [x] Implement: `HuffmanTable`, `HuffmanDecoder`
- [x] Implement: `parse_huffman_weights()`, `build_table_from_weights()`
- [ ] Test: FSE-compressed weights - deferred (uncommon case)

#### 2.2 Huffman Encoder
- [ ] Test: Frequency counting
- [ ] Test: Tree building
- [ ] Test: Encoding roundtrip
- [ ] Implement: `HuffmanEncoder`

### Phase 3: Frame Format ✅
**Goal:** Parse and generate Zstd frame structure

#### 3.1 Frame Header ✅
- [x] Test: Magic number validation (0xFD2FB528)
- [x] Test: Frame header descriptor parsing (all flags)
- [x] Test: Window size calculation (exponent + mantissa)
- [x] Test: Dictionary ID handling (0, 1, 2, 4 bytes)
- [x] Test: Frame content size parsing (0, 1, 2, 4, 8 bytes)
- [x] Implement: `FrameDescriptor`, `FrameHeader::parse()`

#### 3.2 Block Header ✅
- [x] Test: Block type detection (Raw, RLE, Compressed, Reserved error)
- [x] Test: Block size parsing (21-bit)
- [x] Test: Last block flag
- [x] Test: Maximum block size validation (128KB - 1)
- [x] Implement: `BlockType`, `BlockHeader::parse()`

#### 3.3 Checksum (XXHash64) ✅
- [x] Test: Hash consistency for various input sizes
- [x] Test: Different seeds produce different hashes
- [x] Test: Different inputs produce different hashes
- [x] Implement: `xxhash64()` with block processing

### Phase 4: Block Decoding ✅
**Goal:** Decode compressed blocks

#### 4.1 Literals Section ✅
- [x] Test: Raw literals (5-bit, 12-bit, 20-bit sizes)
- [x] Test: RLE literals
- [x] Test: Huffman compressed literals (single stream, 4-stream)
- [ ] Test: Treeless Huffman (repeat previous)
- [x] Implement: `LiteralsSection`, `LiteralsBlockType`
- [x] Implement: `parse()` for Raw/RLE/Huffman modes
- [x] Implement: `decode_huffman_literals()`, 4-stream support

#### 4.2 Sequences Section ✅
- [x] Test: Sequence count parsing (1, 2, 3 byte formats)
- [x] Test: Symbol mode detection
- [x] Implement: `Sequence`, `SequencesSection`, `SymbolMode`
- [x] Implement: Baseline tables for literal/match lengths
- [x] Implement: Predefined FSE distributions (RFC 8878)
- [x] Implement: `build_predefined_tables()` for LL/OF/ML
- [x] Implement: `decode_fse_sequences()`, `decode_rle_sequences()`
- [x] Implement: Custom FSE table parsing in sequences

#### 4.3 Sequence Execution ✅
- [x] Test: Literal copy
- [x] Test: Match copy (non-overlapping)
- [x] Test: Match copy (overlapping/RLE pattern)
- [x] Implement: `execute_sequences()`
- [x] Test: Offset codes (1-3 repeat offsets)

### Phase 5: Full Decompressor ✅
**Goal:** Complete decompression pipeline

- [x] Test: Magic number and header validation
- [x] Test: Multi-block frames (Raw, RLE)
- [x] Test: Content size verification
- [x] Test: XXHash64 checksum verification
- [x] Implement: `DecompressContext` with repeat offsets
- [x] Implement: `decompress_frame()` pipeline
- [x] Implement: `ZstdDecompressor` trait integration
- [x] Implement: Huffman-compressed literals (single + 4-stream)
- [x] Implement: FSE-based sequence decoding (predefined + RLE modes)
- [x] Test: Compressed blocks with literals-only (Raw, RLE literals)
- [x] Test: Integration tests with embedded test vectors
- [x] Test: Multi-block frames, mixed Raw/RLE blocks
- [x] Test: 2-byte FCS, binary data, checksum verification
- [x] Test: Error rejection (invalid checksum, content size mismatch)
- [ ] Test: Large files (>128KB window)
- [ ] Test: Interop with `zstd` CLI output (requires zstd binary)

### Phase 6: Compression ✅ (Basic)
**Goal:** Implement compression pipeline

#### 6.1 Match Finder ✅
- [x] Test: Hash chain match finding
- [x] Test: Match length calculation
- [x] Test: Simple repeat patterns
- [x] Test: Overlapping matches (RLE patterns)
- [x] Implement: `MatchFinder` with hash chains
- [ ] Implement: Binary tree match finding (optimization)
- [ ] Implement: Match selection heuristics (optimization)

#### 6.2 Block Encoder ✅
- [x] Test: Raw block encoding
- [x] Test: RLE block encoding
- [x] Test: Literals section encoding (Raw, RLE modes)
- [x] Implement: `encode_literals()`, `encode_rle_literals()`
- [x] Implement: `encode_sequences()` (empty/count encoding)
- [x] Implement: Block type selection (Raw vs RLE vs Compressed)
- [x] Implement: RLE sequence encoding for uniform patterns (compressibility fingerprinting)
- [ ] Implement: FSE sequence encoding with extra bits
- [ ] Implement: Huffman literal encoding

#### 6.3 Full Compressor ✅
- [x] Test: Compression roundtrip (empty, small, RLE, binary, patterns)
- [x] Test: Compression levels (None, Fast, Default, Best)
- [x] Test: Codec roundtrip integration
- [x] Implement: `CompressContext` with level-based search depth
- [x] Implement: `ZstdCompressor` trait integration
- [x] Implement: Frame header encoding (magic, FCS, window size)
- [x] Implement: XXHash64 checksum generation
- [ ] Test: Interop with `zstd` CLI (requires zstd binary)

### Phase 7: Dictionary Support
- [ ] Test: Dictionary loading
- [ ] Test: Compression with dictionary
- [ ] Test: Decompression with dictionary
- [ ] Implement: Dictionary integration

## Test Strategy

### Unit Tests
Each module has isolated tests for its specific functionality.

### Integration Tests
- Roundtrip tests (compress → decompress → verify)
- Interoperability tests with reference `zstd` implementation
- Property-based tests with proptest

### Test Vectors
Use official Zstd test vectors from:
- https://github.com/facebook/zstd/tree/dev/tests

### Performance Benchmarks
- Compare against `zstd` crate (C wrapper)
- Track compression ratio vs speed tradeoffs

## References

- [RFC 8878 - Zstandard Compression](https://datatracker.ietf.org/doc/html/rfc8878)
- [Zstd Format Specification](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md)
- [FSE Explanation](https://github.com/facebook/zstd/blob/dev/doc/educational_decoder.md)
- [ruzstd](https://github.com/KillingSpark/zstd-rs) - Reference pure-Rust decoder

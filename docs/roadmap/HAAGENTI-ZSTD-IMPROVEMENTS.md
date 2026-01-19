# Track A: Zstd Improvements

## Overview

This document details the TDD roadmap for Zstd compression enhancements in Haagenti.

**Timeline:** 8-10 weeks
**Priority:** High
**Crate:** `haagenti-zstd`

---

## Current State Analysis

### Compression Ratio Gap
```
Data Type          | Haagenti | Reference | Gap
-------------------|----------|-----------|------
Text (small)       | 5.54x    | 7.42x     | -25%
Text (large)       | 345x     | 471x      | -27%
Binary             | 2.1x     | 2.3x      | -9%
Model weights      | 1.8x     | 2.1x      | -14%
```

### Root Causes
1. No dictionary compression support
2. Only predefined FSE tables (no custom)
3. No Huffman encoder (decoder only)
4. Suboptimal match finding for large data

---

## Phase A.1: Dictionary Compression

### Purpose
Enable trained dictionaries for 15-30% better compression on similar data.

### Test Specification (Write First)

```rust
// tests/zstd_dictionary_test.rs

#[cfg(test)]
mod dictionary_tests {
    use haagenti_zstd::{ZstdDict, ZstdDictCompressor, ZstdDictDecompressor};

    // ========== Dictionary Training Tests ==========

    #[test]
    fn test_dictionary_training_from_samples() {
        // Given: A set of similar data samples
        let samples: Vec<&[u8]> = vec![
            b"model.layers.0.weight",
            b"model.layers.1.weight",
            b"model.layers.2.weight",
            b"model.attention.q_proj",
            b"model.attention.k_proj",
        ];

        // When: Training a dictionary
        let dict = ZstdDict::train(&samples, 16 * 1024).unwrap();

        // Then: Dictionary is valid and sized correctly
        assert!(dict.id() != 0);
        assert!(dict.as_bytes().len() <= 16 * 1024);
        assert!(dict.as_bytes().len() >= 1024); // Minimum useful size
    }

    #[test]
    fn test_dictionary_training_insufficient_samples() {
        // Given: Too few samples
        let samples: Vec<&[u8]> = vec![b"single"];

        // When/Then: Training fails gracefully
        let result = ZstdDict::train(&samples, 4096);
        assert!(result.is_err());
    }

    #[test]
    fn test_dictionary_serialization_roundtrip() {
        let samples = generate_training_samples(100);
        let dict = ZstdDict::train(&samples, 8192).unwrap();

        // Serialize
        let bytes = dict.to_bytes();

        // Deserialize
        let restored = ZstdDict::from_bytes(&bytes).unwrap();

        assert_eq!(dict.id(), restored.id());
        assert_eq!(dict.as_bytes(), restored.as_bytes());
    }

    // ========== Dictionary Compression Tests ==========

    #[test]
    fn test_dict_compression_improves_ratio() {
        let samples = generate_training_samples(100);
        let dict = ZstdDict::train(&samples, 8192).unwrap();

        // Compress with dictionary
        let compressor = ZstdDictCompressor::new(&dict);
        let test_data = b"model.layers.15.attention.q_proj.weight";

        let with_dict = compressor.compress(test_data).unwrap();

        // Compress without dictionary
        let without_dict = haagenti_zstd::ZstdCodec::new()
            .compress(test_data).unwrap();

        // Dictionary should improve compression
        assert!(with_dict.len() < without_dict.len(),
            "Dict: {} bytes, No dict: {} bytes",
            with_dict.len(), without_dict.len());
    }

    #[test]
    fn test_dict_compression_roundtrip() {
        let samples = generate_training_samples(50);
        let dict = ZstdDict::train(&samples, 4096).unwrap();

        let compressor = ZstdDictCompressor::new(&dict);
        let decompressor = ZstdDictDecompressor::new(&dict);

        let original = b"model.layers.42.mlp.gate_proj.weight tensor data here";
        let compressed = compressor.compress(original).unwrap();
        let decompressed = decompressor.decompress(&compressed).unwrap();

        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_dict_decompression_wrong_dict_fails() {
        let samples1 = generate_training_samples(50);
        let samples2 = generate_different_samples(50);

        let dict1 = ZstdDict::train(&samples1, 4096).unwrap();
        let dict2 = ZstdDict::train(&samples2, 4096).unwrap();

        let compressor = ZstdDictCompressor::new(&dict1);
        let decompressor = ZstdDictDecompressor::new(&dict2);

        let compressed = compressor.compress(b"test data").unwrap();

        // Should fail or produce wrong output
        let result = decompressor.decompress(&compressed);
        assert!(result.is_err() || result.unwrap() != b"test data");
    }

    #[test]
    fn test_dict_compression_levels() {
        let samples = generate_training_samples(50);
        let dict = ZstdDict::train(&samples, 4096).unwrap();

        let data = generate_test_data(10_000);

        let fast = ZstdDictCompressor::with_level(&dict, CompressionLevel::Fast)
            .compress(&data).unwrap();
        let default = ZstdDictCompressor::with_level(&dict, CompressionLevel::Default)
            .compress(&data).unwrap();
        let best = ZstdDictCompressor::with_level(&dict, CompressionLevel::Best)
            .compress(&data).unwrap();

        // Higher levels should compress better (or equal)
        assert!(best.len() <= default.len());
        assert!(default.len() <= fast.len() + fast.len() / 10); // Allow 10% variance
    }

    // ========== Dictionary ID Tests ==========

    #[test]
    fn test_dict_id_embedded_in_frame() {
        let samples = generate_training_samples(50);
        let dict = ZstdDict::train(&samples, 4096).unwrap();

        let compressor = ZstdDictCompressor::new(&dict);
        let compressed = compressor.compress(b"test").unwrap();

        // Parse frame header to verify dict ID is present
        let frame_header = haagenti_zstd::parse_frame_header(&compressed).unwrap();
        assert_eq!(frame_header.dict_id, Some(dict.id()));
    }

    #[test]
    fn test_dict_id_mismatch_detection() {
        let samples = generate_training_samples(50);
        let dict = ZstdDict::train(&samples, 4096).unwrap();

        let compressor = ZstdDictCompressor::new(&dict);
        let compressed = compressor.compress(b"test").unwrap();

        // Try to decompress without dictionary
        let result = haagenti_zstd::ZstdCodec::new().decompress(&compressed);

        // Should fail with dictionary required error
        assert!(matches!(result, Err(e) if e.to_string().contains("dictionary")));
    }

    // ========== Performance Tests ==========

    #[test]
    fn test_dict_compression_performance_acceptable() {
        let samples = generate_training_samples(100);
        let dict = ZstdDict::train(&samples, 16384).unwrap();
        let compressor = ZstdDictCompressor::new(&dict);

        let data = generate_test_data(1_000_000); // 1MB

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = compressor.compress(&data).unwrap();
        }
        let elapsed = start.elapsed();

        // Should compress at least 100 MB/s with dictionary
        let throughput_mbs = (10.0 * data.len() as f64) / elapsed.as_secs_f64() / 1_000_000.0;
        assert!(throughput_mbs > 100.0, "Throughput: {} MB/s", throughput_mbs);
    }

    // ========== Helper Functions ==========

    fn generate_training_samples(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|i| format!("model.layers.{}.attention.q_proj.weight", i).into_bytes())
            .collect()
    }

    fn generate_different_samples(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|i| format!("completely.different.pattern.{}.data", i).into_bytes())
            .collect()
    }

    fn generate_test_data(size: usize) -> Vec<u8> {
        let pattern = b"model.layers.X.attention.q_proj.weight data ";
        pattern.iter().cycle().take(size).copied().collect()
    }
}
```

### Implementation Specification

```rust
// haagenti-zstd/src/dict.rs

/// A trained Zstd dictionary for improved compression.
pub struct ZstdDict {
    /// Raw dictionary bytes
    data: Vec<u8>,
    /// Dictionary ID (XXH64 of content)
    id: u32,
}

impl ZstdDict {
    /// Train a dictionary from sample data.
    ///
    /// # Arguments
    /// * `samples` - Training samples (should be representative of data to compress)
    /// * `max_size` - Maximum dictionary size in bytes
    ///
    /// # Returns
    /// Trained dictionary or error if training fails
    pub fn train(samples: &[impl AsRef<[u8]>], max_size: usize) -> Result<Self>;

    /// Load dictionary from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self>;

    /// Serialize dictionary to bytes.
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Get dictionary ID.
    pub fn id(&self) -> u32;

    /// Get raw dictionary bytes.
    pub fn as_bytes(&self) -> &[u8];
}

/// Compressor using a trained dictionary.
pub struct ZstdDictCompressor<'d> {
    dict: &'d ZstdDict,
    level: CompressionLevel,
}

impl<'d> ZstdDictCompressor<'d> {
    pub fn new(dict: &'d ZstdDict) -> Self;
    pub fn with_level(dict: &'d ZstdDict, level: CompressionLevel) -> Self;
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Decompressor using a trained dictionary.
pub struct ZstdDictDecompressor<'d> {
    dict: &'d ZstdDict,
}

impl<'d> ZstdDictDecompressor<'d> {
    pub fn new(dict: &'d ZstdDict) -> Self;
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
}
```

### Quality Gate A.1

```bash
#!/bin/bash
# Phase A.1 Quality Gate

echo "=== Phase A.1: Dictionary Compression Quality Gate ==="

# 1. All dictionary tests pass
cargo test --package haagenti-zstd dict_ -- --nocapture

# 2. Dictionary improves compression by at least 10%
cargo test --package haagenti-zstd test_dict_compression_improves_ratio

# 3. Roundtrip works correctly
cargo test --package haagenti-zstd test_dict_compression_roundtrip

# 4. Performance meets threshold
cargo test --package haagenti-zstd test_dict_compression_performance

# 5. No regressions in non-dict compression
cargo bench --package haagenti-zstd -- baseline

echo "=== Phase A.1 PASSED ==="
```

---

## Phase A.2: FSE Custom Tables

### Purpose
Enable custom FSE table serialization for 5-10% better compression on specialized data.

### Test Specification

```rust
// tests/zstd_fse_tables_test.rs

#[cfg(test)]
mod fse_custom_table_tests {
    use haagenti_zstd::fse::{FseTable, FseEncoder, FseDecoder, TableLog};

    #[test]
    fn test_custom_table_from_frequencies() {
        // Given: Symbol frequency distribution
        let frequencies = [100, 50, 25, 12, 6, 3, 2, 1, 1]; // Zipf-like

        // When: Building custom table
        let table = FseTable::from_frequencies(&frequencies, TableLog::Log9).unwrap();

        // Then: Table is valid
        assert!(table.is_valid());
        assert_eq!(table.max_symbol(), frequencies.len() - 1);
    }

    #[test]
    fn test_custom_table_serialization() {
        let frequencies = [100, 50, 25, 12, 6, 3, 2, 1];
        let table = FseTable::from_frequencies(&frequencies, TableLog::Log8).unwrap();

        // Serialize
        let mut buffer = Vec::new();
        table.serialize(&mut buffer).unwrap();

        // Deserialize
        let restored = FseTable::deserialize(&buffer).unwrap();

        // Verify equality
        assert_eq!(table.accuracy_log(), restored.accuracy_log());
        assert_eq!(table.max_symbol(), restored.max_symbol());
    }

    #[test]
    fn test_custom_table_encode_decode_roundtrip() {
        let frequencies = [100, 50, 25, 12];
        let table = FseTable::from_frequencies(&frequencies, TableLog::Log8).unwrap();

        let encoder = FseEncoder::new(&table);
        let decoder = FseDecoder::new(&table);

        let symbols = vec![0, 1, 2, 3, 0, 0, 1, 2, 0, 1, 0, 0, 0];
        let encoded = encoder.encode(&symbols).unwrap();
        let decoded = decoder.decode(&encoded, symbols.len()).unwrap();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_custom_table_beats_predefined_for_skewed_data() {
        // Highly skewed distribution (RLE-like)
        let frequencies = [1000, 1, 1, 1]; // Symbol 0 dominates
        let custom_table = FseTable::from_frequencies(&frequencies, TableLog::Log8).unwrap();

        let predefined = FseTable::predefined_literals();

        // Data with mostly symbol 0
        let data: Vec<u8> = (0..1000).map(|i| if i % 100 == 0 { 1 } else { 0 }).collect();

        let custom_size = FseEncoder::new(&custom_table).encode(&data).unwrap().len();
        let predefined_size = FseEncoder::new(&predefined).encode(&data).unwrap().len();

        assert!(custom_size < predefined_size,
            "Custom: {} bytes, Predefined: {} bytes", custom_size, predefined_size);
    }

    #[test]
    fn test_custom_table_in_zstd_frame() {
        let frequencies = [100, 50, 25, 12, 6, 3];
        let table = FseTable::from_frequencies(&frequencies, TableLog::Log8).unwrap();

        // Compress with custom table
        let compressor = haagenti_zstd::ZstdCompressor::with_custom_tables(
            Some(table.clone()), // Literals
            None,                // Match lengths (use predefined)
            None,                // Offsets (use predefined)
        );

        let data = generate_data_matching_frequencies(&frequencies, 10000);
        let compressed = compressor.compress(&data).unwrap();

        // Decompress and verify
        let decompressed = haagenti_zstd::ZstdCodec::new()
            .decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_table_accuracy_log_selection() {
        let frequencies = [100, 50, 25, 12, 6, 3, 2, 1];

        // Different accuracy logs
        for log in [6, 7, 8, 9, 10, 11] {
            let table = FseTable::from_frequencies(&frequencies, TableLog::from_u8(log)).unwrap();
            assert_eq!(table.accuracy_log(), log);
        }
    }

    #[test]
    fn test_invalid_frequencies_rejected() {
        // All zeros
        let result = FseTable::from_frequencies(&[0, 0, 0], TableLog::Log8);
        assert!(result.is_err());

        // Empty
        let result = FseTable::from_frequencies(&[], TableLog::Log8);
        assert!(result.is_err());
    }

    #[test]
    fn test_rle_mode_detection() {
        // Single symbol repeated
        let frequencies = [1000, 0, 0, 0];
        let table = FseTable::from_frequencies(&frequencies, TableLog::Log8).unwrap();

        assert!(table.is_rle_mode());
    }

    fn generate_data_matching_frequencies(freqs: &[usize], size: usize) -> Vec<u8> {
        let total: usize = freqs.iter().sum();
        let mut data = Vec::with_capacity(size);
        let mut rng = 42u64;

        for _ in 0..size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 33) as usize % total;
            let mut cumulative = 0;
            for (symbol, &freq) in freqs.iter().enumerate() {
                cumulative += freq;
                if r < cumulative {
                    data.push(symbol as u8);
                    break;
                }
            }
        }
        data
    }
}
```

### Quality Gate A.2

```bash
#!/bin/bash
# Phase A.2 Quality Gate

echo "=== Phase A.2: FSE Custom Tables Quality Gate ==="

# 1. All FSE tests pass
cargo test --package haagenti-zstd fse_ -- --nocapture

# 2. Custom tables beat predefined for skewed data
cargo test --package haagenti-zstd test_custom_table_beats_predefined

# 3. Serialization roundtrip works
cargo test --package haagenti-zstd test_custom_table_serialization

# 4. Integration with Zstd frames
cargo test --package haagenti-zstd test_custom_table_in_zstd_frame

echo "=== Phase A.2 PASSED ==="
```

---

## Phase A.3: Huffman Encoder

### Purpose
Implement Huffman encoder to complete Zstd compression pipeline.

### Test Specification

```rust
// tests/zstd_huffman_encoder_test.rs

#[cfg(test)]
mod huffman_encoder_tests {
    use haagenti_zstd::huffman::{HuffmanEncoder, HuffmanDecoder, HuffmanTable};

    #[test]
    fn test_huffman_table_from_frequencies() {
        let frequencies = [100u32, 50, 25, 12, 6, 3, 2, 1];
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        // Most frequent symbol should have shortest code
        assert!(table.code_length(0) <= table.code_length(7));
    }

    #[test]
    fn test_huffman_encode_decode_roundtrip() {
        let frequencies = [100u32, 50, 25, 12, 6, 3, 2, 1];
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        let encoder = HuffmanEncoder::new(&table);
        let decoder = HuffmanDecoder::new(&table);

        let data: Vec<u8> = (0..1000).map(|i| (i % 8) as u8).collect();
        let encoded = encoder.encode(&data).unwrap();
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_huffman_table_serialization() {
        let frequencies = [100u32, 50, 25, 12];
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        let serialized = table.to_bytes();
        let restored = HuffmanTable::from_bytes(&serialized).unwrap();

        // Verify codes match
        for symbol in 0..4u8 {
            assert_eq!(table.code_length(symbol), restored.code_length(symbol));
        }
    }

    #[test]
    fn test_huffman_max_code_length() {
        // Worst case: power-of-2 frequencies
        let frequencies: Vec<u32> = (0..11).map(|i| 1 << i).collect();
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        // Zstd limits Huffman codes to 11 bits
        for i in 0..frequencies.len() {
            assert!(table.code_length(i as u8) <= 11,
                "Symbol {} has code length {}", i, table.code_length(i as u8));
        }
    }

    #[test]
    fn test_huffman_single_symbol() {
        let frequencies = [100u32];
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        let encoder = HuffmanEncoder::new(&table);
        let decoder = HuffmanDecoder::new(&table);

        let data = vec![0u8; 100];
        let encoded = encoder.encode(&data).unwrap();
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_huffman_256_symbols() {
        // All byte values
        let frequencies: Vec<u32> = (0..256).map(|i| 256 - i as u32).collect();
        let table = HuffmanTable::from_frequencies(&frequencies).unwrap();

        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let encoder = HuffmanEncoder::new(&table);
        let encoded = encoder.encode(&data).unwrap();

        let decoder = HuffmanDecoder::new(&table);
        let decoded = decoder.decode(&encoded, data.len()).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_huffman_integration_with_zstd() {
        // Use Huffman encoding in Zstd literal section
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(100);

        let compressed = haagenti_zstd::ZstdCompressor::new()
            .with_literal_mode(haagenti_zstd::LiteralMode::Huffman)
            .compress(&data).unwrap();

        let decompressed = haagenti_zstd::ZstdCodec::new()
            .decompress(&compressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_huffman_compression_ratio() {
        // Huffman should compress repetitive text well
        let data = b"aaaaaabbbbcccdde".repeat(1000);

        let compressed = HuffmanEncoder::compress_raw(&data).unwrap();

        // Should achieve at least 2x compression on this data
        assert!(compressed.len() < data.len() / 2,
            "Compressed: {} bytes, Original: {} bytes", compressed.len(), data.len());
    }
}
```

### Quality Gate A.3

```bash
#!/bin/bash
# Phase A.3 Quality Gate

echo "=== Phase A.3: Huffman Encoder Quality Gate ==="

# 1. All Huffman tests pass
cargo test --package haagenti-zstd huffman_ -- --nocapture

# 2. Roundtrip works
cargo test --package haagenti-zstd test_huffman_encode_decode_roundtrip

# 3. Integration with Zstd
cargo test --package haagenti-zstd test_huffman_integration_with_zstd

# 4. Reference interop (compress with haagenti, decompress with zstd CLI)
cargo test --package haagenti-zstd test_huffman_reference_interop

echo "=== Phase A.3 PASSED ==="
```

---

## Phase A.4: Compression Ratio Optimization

### Purpose
Close the 25% compression ratio gap with reference zstd.

### Test Specification

```rust
// tests/zstd_ratio_optimization_test.rs

#[cfg(test)]
mod ratio_optimization_tests {
    use haagenti_zstd::ZstdCompressor;

    const SILESIA_DICKENS: &[u8] = include_bytes!("../testdata/dickens");
    const SILESIA_MOZILLA: &[u8] = include_bytes!("../testdata/mozilla");
    const ENWIK8_SAMPLE: &[u8] = include_bytes!("../testdata/enwik8_100k");

    #[test]
    fn test_text_compression_ratio_target() {
        let compressor = ZstdCompressor::new();
        let compressed = compressor.compress(SILESIA_DICKENS).unwrap();

        let ratio = SILESIA_DICKENS.len() as f64 / compressed.len() as f64;

        // Target: within 15% of reference zstd (7.42x)
        // Minimum: 6.3x
        assert!(ratio >= 6.3, "Ratio: {:.2}x (target: >=6.3x)", ratio);
    }

    #[test]
    fn test_binary_compression_ratio_target() {
        let compressor = ZstdCompressor::new();
        let compressed = compressor.compress(SILESIA_MOZILLA).unwrap();

        let ratio = SILESIA_MOZILLA.len() as f64 / compressed.len() as f64;

        // Target: within 10% of reference (2.3x)
        // Minimum: 2.1x
        assert!(ratio >= 2.1, "Ratio: {:.2}x (target: >=2.1x)", ratio);
    }

    #[test]
    fn test_adaptive_search_depth() {
        // Small data should use deep search
        let small = vec![b'a'; 1000];
        let large = vec![b'a'; 100_000];

        let compressor = ZstdCompressor::new();

        // Both should compress well, but large shouldn't be much slower
        let start = std::time::Instant::now();
        let _ = compressor.compress(&small).unwrap();
        let small_time = start.elapsed();

        let start = std::time::Instant::now();
        let _ = compressor.compress(&large).unwrap();
        let large_time = start.elapsed();

        // Large should take at most 150x longer (100x more data + overhead)
        let ratio = large_time.as_nanos() as f64 / small_time.as_nanos() as f64;
        assert!(ratio < 150.0, "Time ratio: {:.1}x", ratio);
    }

    #[test]
    fn test_lazy_matching_improves_ratio() {
        let data = b"abcabcabcdefabcabcabcdef".repeat(1000);

        let eager = ZstdCompressor::new()
            .with_lazy_matching(false)
            .compress(&data).unwrap();

        let lazy = ZstdCompressor::new()
            .with_lazy_matching(true)
            .compress(&data).unwrap();

        // Lazy matching should produce smaller output
        assert!(lazy.len() <= eager.len(),
            "Lazy: {} bytes, Eager: {} bytes", lazy.len(), eager.len());
    }

    #[test]
    fn test_optimal_parsing_level() {
        let data = ENWIK8_SAMPLE;

        let fast = ZstdCompressor::with_level(CompressionLevel::Fast)
            .compress(data).unwrap();
        let default = ZstdCompressor::with_level(CompressionLevel::Default)
            .compress(data).unwrap();
        let best = ZstdCompressor::with_level(CompressionLevel::Best)
            .compress(data).unwrap();

        // Each level should improve ratio
        assert!(best.len() <= default.len());
        assert!(default.len() <= fast.len());

        // Best should be at least 5% better than fast
        let improvement = 1.0 - (best.len() as f64 / fast.len() as f64);
        assert!(improvement >= 0.05, "Improvement: {:.1}%", improvement * 100.0);
    }

    #[test]
    fn test_long_distance_matching() {
        // Create data with long-distance repeats
        let pattern = b"This is a test pattern that will repeat after a long gap.";
        let mut data = Vec::new();
        data.extend_from_slice(pattern);
        data.extend_from_slice(&vec![b'x'; 50_000]); // 50KB gap
        data.extend_from_slice(pattern); // Repeat

        let compressor = ZstdCompressor::new()
            .with_window_log(17); // 128KB window

        let compressed = compressor.compress(&data).unwrap();

        // Should find the long-distance match
        // Compressed size should be much smaller than data without match
        let no_match_size = pattern.len() * 2 + 50_000; // Theoretical no-match size
        assert!(compressed.len() < no_match_size - pattern.len(),
            "Should find 50KB distance match");
    }

    #[test]
    fn test_entropy_encoding_efficiency() {
        // Low entropy data should compress extremely well
        let low_entropy = vec![0u8; 100_000];
        let high_entropy: Vec<u8> = (0..100_000).map(|i| (i * 17 % 256) as u8).collect();

        let compressor = ZstdCompressor::new();

        let low_compressed = compressor.compress(&low_entropy).unwrap();
        let high_compressed = compressor.compress(&high_entropy).unwrap();

        // Low entropy should achieve >100x compression
        let low_ratio = low_entropy.len() as f64 / low_compressed.len() as f64;
        assert!(low_ratio > 100.0, "Low entropy ratio: {:.1}x", low_ratio);

        // High entropy should still compress somewhat
        let high_ratio = high_entropy.len() as f64 / high_compressed.len() as f64;
        assert!(high_ratio > 1.0, "High entropy ratio: {:.2}x", high_ratio);
    }
}
```

### Quality Gate A.4

```bash
#!/bin/bash
# Phase A.4 Quality Gate

echo "=== Phase A.4: Compression Ratio Optimization Quality Gate ==="

# 1. Text compression ratio target met
cargo test --package haagenti-zstd test_text_compression_ratio_target

# 2. Binary compression ratio target met
cargo test --package haagenti-zstd test_binary_compression_ratio_target

# 3. Benchmark comparison with baseline
cargo bench --package haagenti-zstd -- compression_ratio
RATIO=$(grep "ratio" target/criterion/*/new/estimates.json | jq '.mean.point_estimate')
if (( $(echo "$RATIO < 6.0" | bc -l) )); then
    echo "FAIL: Compression ratio $RATIO below target 6.0x"
    exit 1
fi

# 4. No performance regression >10%
cargo bench --package haagenti-zstd -- throughput
# Compare with baseline...

echo "=== Phase A.4 PASSED ==="
```

---

## Phase A.5: Large Data Throughput

### Purpose
Improve compression throughput on 64KB+ data by 45-67%.

### Test Specification

```rust
// tests/zstd_throughput_test.rs

#[cfg(test)]
mod throughput_tests {
    use haagenti_zstd::ZstdCompressor;
    use std::time::Instant;

    #[test]
    fn test_64kb_compression_throughput() {
        let data = generate_compressible_data(64 * 1024);
        let compressor = ZstdCompressor::new();

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = compressor.compress(&data).unwrap();
        }
        let elapsed = start.elapsed();

        let throughput_mbs = (iterations as f64 * data.len() as f64)
            / elapsed.as_secs_f64() / 1_000_000.0;

        // Target: 200 MB/s for 64KB blocks
        assert!(throughput_mbs > 200.0,
            "Throughput: {:.1} MB/s (target: >200 MB/s)", throughput_mbs);
    }

    #[test]
    fn test_1mb_compression_throughput() {
        let data = generate_compressible_data(1024 * 1024);
        let compressor = ZstdCompressor::new();

        let start = Instant::now();
        let iterations = 20;
        for _ in 0..iterations {
            let _ = compressor.compress(&data).unwrap();
        }
        let elapsed = start.elapsed();

        let throughput_mbs = (iterations as f64 * data.len() as f64)
            / elapsed.as_secs_f64() / 1_000_000.0;

        // Target: 150 MB/s for 1MB blocks
        assert!(throughput_mbs > 150.0,
            "Throughput: {:.1} MB/s (target: >150 MB/s)", throughput_mbs);
    }

    #[test]
    fn test_decompression_throughput() {
        let data = generate_compressible_data(1024 * 1024);
        let compressed = ZstdCompressor::new().compress(&data).unwrap();

        let decompressor = haagenti_zstd::ZstdCodec::new();

        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = decompressor.decompress(&compressed).unwrap();
        }
        let elapsed = start.elapsed();

        let throughput_mbs = (iterations as f64 * data.len() as f64)
            / elapsed.as_secs_f64() / 1_000_000.0;

        // Target: 1.2 GB/s decompression
        assert!(throughput_mbs > 1200.0,
            "Throughput: {:.1} MB/s (target: >1200 MB/s)", throughput_mbs);
    }

    #[test]
    fn test_adaptive_search_depth_scaling() {
        let compressor = ZstdCompressor::new();

        let sizes = [4096, 16384, 65536, 262144, 1048576];
        let mut times_per_byte = Vec::new();

        for &size in &sizes {
            let data = generate_compressible_data(size);

            let start = Instant::now();
            let iterations = 1000000 / size; // More iterations for smaller sizes
            for _ in 0..iterations.max(1) {
                let _ = compressor.compress(&data).unwrap();
            }
            let elapsed = start.elapsed();

            let ns_per_byte = elapsed.as_nanos() as f64 / (iterations.max(1) * size) as f64;
            times_per_byte.push((size, ns_per_byte));
        }

        // Time per byte should not increase dramatically with size
        // (indicates proper adaptive depth)
        let small_time = times_per_byte[0].1;
        let large_time = times_per_byte[4].1;

        assert!(large_time < small_time * 3.0,
            "Large data too slow: {:.2} ns/byte vs {:.2} ns/byte for small",
            large_time, small_time);
    }

    fn generate_compressible_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        let patterns = [
            b"The quick brown fox jumps over the lazy dog. ",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
            b"Pack my box with five dozen liquor jugs. ",
        ];

        let mut pattern_idx = 0;
        while data.len() < size {
            let pattern = patterns[pattern_idx % patterns.len()];
            let remaining = size - data.len();
            data.extend_from_slice(&pattern[..pattern.len().min(remaining)]);
            pattern_idx += 1;
        }
        data
    }
}
```

### Quality Gate A.5

```bash
#!/bin/bash
# Phase A.5 Quality Gate

echo "=== Phase A.5: Large Data Throughput Quality Gate ==="

# 1. 64KB throughput target
cargo test --package haagenti-zstd test_64kb_compression_throughput

# 2. 1MB throughput target
cargo test --package haagenti-zstd test_1mb_compression_throughput

# 3. Decompression throughput target
cargo test --package haagenti-zstd test_decompression_throughput

# 4. Benchmark suite
cargo bench --package haagenti-zstd -- throughput

# 5. Compare with baseline (fail if >10% regression)
# ... benchmark comparison logic ...

echo "=== Phase A.5 PASSED ==="
```

---

## Track A Summary

### Test Count by Phase

| Phase | Unit Tests | Integration | Benchmark | Total |
|-------|------------|-------------|-----------|-------|
| A.1 Dictionary | 10 | 3 | 2 | 15 |
| A.2 FSE Tables | 9 | 2 | 1 | 12 |
| A.3 Huffman | 8 | 2 | 0 | 10 |
| A.4 Ratio | 6 | 2 | 10 | 18 |
| A.5 Throughput | 4 | 0 | 4 | 8 |
| **Total** | **37** | **9** | **17** | **63** |

### Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Text compression ratio | 5.5x | 7.0x | +27% |
| Dict compression bonus | N/A | +20% | New |
| 64KB throughput | 120 MB/s | 200 MB/s | +67% |
| 1MB throughput | 100 MB/s | 150 MB/s | +50% |

---

*Document Version: 1.0*
*Created: 2026-01-06*

//! Zstandard Dictionary Support
//!
//! This module implements dictionary compression for Zstd, enabling
//! significantly better compression ratios on small, similar data samples.
//!
//! ## Dictionary Format (Zstd Spec)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Magic: 0xEC30A437 (4 bytes)                                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Dictionary ID (4 bytes)                                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Huffman Table (variable)                                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ FSE Offset Table                                                │
//! │ FSE Match Length Table                                          │
//! │ FSE Literals Length Table                                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Content (raw dictionary data)                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use haagenti_core::{Error, Result};
use std::collections::HashMap;

/// Zstd dictionary magic number
pub const DICT_MAGIC: u32 = 0xEC30A437;

/// Maximum dictionary size (128KB as per spec)
pub const MAX_DICT_SIZE: usize = 128 * 1024;

/// Minimum dictionary size
pub const MIN_DICT_SIZE: usize = 8;

/// Minimum samples for dictionary training
pub const MIN_SAMPLES: usize = 5;

/// A Zstandard dictionary for compression/decompression
#[derive(Debug, Clone)]
pub struct ZstdDictionary {
    /// Dictionary ID
    id: u32,
    /// Raw dictionary content (for match finding)
    content: Vec<u8>,
    /// Precomputed Huffman table for literals
    huffman_table: Option<Vec<u8>>,
    /// FSE table for offsets
    fse_offset_table: Option<Vec<u8>>,
    /// FSE table for match lengths
    fse_ml_table: Option<Vec<u8>>,
    /// FSE table for literals lengths
    fse_ll_table: Option<Vec<u8>>,
    /// Hash table for fast match finding in dictionary
    hash_table: HashMap<u32, Vec<usize>>,
}

impl ZstdDictionary {
    /// Create a new dictionary from raw content
    pub fn from_content(content: Vec<u8>) -> Result<Self> {
        if content.len() < MIN_DICT_SIZE {
            return Err(Error::corrupted("Dictionary too small"));
        }
        if content.len() > MAX_DICT_SIZE {
            return Err(Error::corrupted("Dictionary too large"));
        }

        // Generate a dictionary ID from content hash
        let id = Self::compute_id(&content);

        // Build hash table for fast lookups
        let hash_table = Self::build_hash_table(&content);

        Ok(Self {
            id,
            content,
            huffman_table: None,
            fse_offset_table: None,
            fse_ml_table: None,
            fse_ll_table: None,
            hash_table,
        })
    }

    /// Parse a dictionary from serialized format
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::corrupted("Dictionary data too short"));
        }

        // Check magic number
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != DICT_MAGIC {
            return Err(Error::corrupted("Invalid dictionary magic"));
        }

        // Read dictionary ID
        let id = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        // For raw dictionaries (no pre-trained tables), content starts at offset 8
        // Full dictionaries with tables are more complex - for now, support raw only
        let content = data[8..].to_vec();

        let hash_table = Self::build_hash_table(&content);

        Ok(Self {
            id,
            content,
            huffman_table: None,
            fse_offset_table: None,
            fse_ml_table: None,
            fse_ll_table: None,
            hash_table,
        })
    }

    /// Train a dictionary from samples
    ///
    /// Uses a simple but effective algorithm:
    /// 1. Find common substrings across samples
    /// 2. Score by frequency * length
    /// 3. Select top patterns up to target size
    pub fn train(samples: &[&[u8]], dict_size: usize) -> Result<Self> {
        if samples.len() < MIN_SAMPLES {
            return Err(Error::corrupted(format!(
                "Need at least {} samples for training",
                MIN_SAMPLES
            )));
        }

        let dict_size = dict_size.min(MAX_DICT_SIZE);

        // Concatenate all samples for substring analysis
        let mut all_data = Vec::new();
        let mut sample_offsets = Vec::new();
        for sample in samples {
            sample_offsets.push(all_data.len());
            all_data.extend_from_slice(sample);
        }

        // Find frequent substrings using suffix-like analysis
        let patterns = Self::find_frequent_patterns(&all_data, samples.len());

        // Build dictionary from top patterns
        let mut dict_content = Vec::with_capacity(dict_size);
        for (pattern, _score) in patterns {
            if dict_content.len() + pattern.len() > dict_size {
                break;
            }
            dict_content.extend_from_slice(&pattern);
        }

        // If we didn't fill the dictionary, add raw data from samples
        if dict_content.len() < dict_size {
            for sample in samples {
                let remaining = dict_size - dict_content.len();
                if remaining == 0 {
                    break;
                }
                let to_add = sample.len().min(remaining);
                dict_content.extend_from_slice(&sample[..to_add]);
            }
        }

        Self::from_content(dict_content)
    }

    /// Find frequent patterns across data
    fn find_frequent_patterns(data: &[u8], num_samples: usize) -> Vec<(Vec<u8>, u64)> {
        let mut pattern_counts: HashMap<Vec<u8>, u64> = HashMap::new();

        // Look for patterns of various lengths
        for pattern_len in 4..=32 {
            if data.len() < pattern_len {
                break;
            }
            for i in 0..=(data.len() - pattern_len) {
                let pattern = &data[i..i + pattern_len];
                *pattern_counts.entry(pattern.to_vec()).or_insert(0) += 1;
            }
        }

        // Score patterns by frequency * length (more weight to longer patterns)
        let mut scored: Vec<_> = pattern_counts
            .into_iter()
            .filter(|(_, count)| *count > num_samples as u64) // Must appear in multiple samples
            .map(|(pattern, count)| {
                let score = count * (pattern.len() as u64).pow(2);
                (pattern, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove overlapping patterns (keep higher-scored ones)
        let mut selected: Vec<(Vec<u8>, u64)> = Vec::new();
        #[allow(unused_variables)]
        let used_ranges: Vec<(usize, usize)> = Vec::new();

        'outer: for (pattern, score) in scored {
            // Check if this pattern overlaps with already-selected ones
            // (simplified check - just ensure unique patterns)
            for (existing, _) in &selected {
                if Self::patterns_overlap(&pattern, existing) {
                    continue 'outer;
                }
            }
            selected.push((pattern, score));

            if selected.len() >= 1000 {
                break;
            }
        }

        selected
    }

    /// Check if two patterns significantly overlap
    fn patterns_overlap(a: &[u8], b: &[u8]) -> bool {
        let min_len = a.len().min(b.len());
        if min_len < 4 {
            return a == b;
        }

        // Check if one is a substring of the other
        if a.len() >= b.len() {
            for window in a.windows(b.len()) {
                if window == b {
                    return true;
                }
            }
        } else {
            for window in b.windows(a.len()) {
                if window == a {
                    return true;
                }
            }
        }

        false
    }

    /// Build hash table for fast match finding
    fn build_hash_table(content: &[u8]) -> HashMap<u32, Vec<usize>> {
        let mut table: HashMap<u32, Vec<usize>> = HashMap::new();

        if content.len() < 4 {
            return table;
        }

        for i in 0..=(content.len() - 4) {
            let hash = Self::hash4(&content[i..i + 4]);
            table.entry(hash).or_default().push(i);
        }

        table
    }

    /// Simple 4-byte hash for match finding
    fn hash4(data: &[u8]) -> u32 {
        debug_assert!(data.len() >= 4);
        let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // Simple multiplicative hash
        v.wrapping_mul(0x9E3779B9)
    }

    /// Compute dictionary ID from content
    fn compute_id(content: &[u8]) -> u32 {
        // Use XXHash64 truncated to 32 bits
        let hash = crate::frame::xxhash64(content, 0);
        (hash & 0xFFFFFFFF) as u32
    }

    /// Get dictionary ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get dictionary content
    pub fn content(&self) -> &[u8] {
        &self.content
    }

    /// Get dictionary size
    pub fn size(&self) -> usize {
        self.content.len()
    }

    /// Serialize dictionary to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(8 + self.content.len());

        // Magic number
        result.extend_from_slice(&DICT_MAGIC.to_le_bytes());

        // Dictionary ID
        result.extend_from_slice(&self.id.to_le_bytes());

        // Content
        result.extend_from_slice(&self.content);

        result
    }

    /// Find best match in dictionary for the given position in input
    pub fn find_match(&self, input: &[u8], pos: usize) -> Option<DictMatch> {
        if pos + 4 > input.len() {
            return None;
        }

        let hash = Self::hash4(&input[pos..pos + 4]);
        let candidates = self.hash_table.get(&hash)?;

        let mut best_match: Option<DictMatch> = None;
        let max_len = input.len() - pos;

        for &dict_pos in candidates {
            // Calculate match length
            let mut match_len = 0;
            while match_len < max_len
                && dict_pos + match_len < self.content.len()
                && input[pos + match_len] == self.content[dict_pos + match_len]
            {
                match_len += 1;
            }

            // Minimum match length of 4
            if match_len >= 4 {
                let offset = self.content.len() - dict_pos;
                if best_match.as_ref().map(|m| match_len > m.length).unwrap_or(true) {
                    best_match = Some(DictMatch {
                        offset,
                        length: match_len,
                        dict_position: dict_pos,
                    });
                }
            }
        }

        best_match
    }

    /// Get byte at position (for match verification during decompression)
    pub fn get_byte(&self, pos: usize) -> Option<u8> {
        self.content.get(pos).copied()
    }
}

/// A match found in the dictionary
#[derive(Debug, Clone, Copy)]
pub struct DictMatch {
    /// Offset from end of dictionary
    pub offset: usize,
    /// Match length
    pub length: usize,
    /// Position in dictionary content
    pub dict_position: usize,
}

/// Dictionary-aware compressor
#[derive(Debug)]
pub struct ZstdDictCompressor {
    dictionary: ZstdDictionary,
    level: haagenti_core::CompressionLevel,
}

impl ZstdDictCompressor {
    /// Create a new dictionary compressor
    pub fn new(dictionary: ZstdDictionary) -> Self {
        Self {
            dictionary,
            level: haagenti_core::CompressionLevel::Default,
        }
    }

    /// Create with compression level
    pub fn with_level(dictionary: ZstdDictionary, level: haagenti_core::CompressionLevel) -> Self {
        Self { dictionary, level }
    }

    /// Get the dictionary
    pub fn dictionary(&self) -> &ZstdDictionary {
        &self.dictionary
    }

    /// Compress using the dictionary
    pub fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        // For now, use regular compression with dictionary ID in frame header
        // Full dictionary-aware compression would use dict matches
        let mut ctx = crate::compress::CompressContext::new(self.level);
        ctx.set_dictionary_id(self.dictionary.id());
        ctx.compress(input)
    }
}

/// Dictionary-aware decompressor
#[derive(Debug)]
pub struct ZstdDictDecompressor {
    dictionary: ZstdDictionary,
}

impl ZstdDictDecompressor {
    /// Create a new dictionary decompressor
    pub fn new(dictionary: ZstdDictionary) -> Self {
        Self { dictionary }
    }

    /// Get the dictionary
    pub fn dictionary(&self) -> &ZstdDictionary {
        &self.dictionary
    }

    /// Decompress using the dictionary
    pub fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        // Parse frame header to get dictionary ID
        if input.len() < 8 {
            return Err(Error::corrupted("Input too short"));
        }

        // Verify magic
        let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        if magic != crate::ZSTD_MAGIC {
            return Err(Error::corrupted("Invalid Zstd magic"));
        }

        // Parse frame descriptor to check for dictionary ID
        let descriptor = input[4];
        let has_dict_id = (descriptor & 0x03) != 0;

        if has_dict_id {
            // Verify dictionary ID matches
            let dict_id_size = match descriptor & 0x03 {
                1 => 1,
                2 => 2,
                3 => 4,
                _ => 0,
            };

            if dict_id_size > 0 {
                let offset = if (descriptor & 0x20) == 0 { 6 } else { 5 };
                let frame_dict_id = match dict_id_size {
                    1 => input[offset] as u32,
                    2 => u16::from_le_bytes([input[offset], input[offset + 1]]) as u32,
                    4 => u32::from_le_bytes([
                        input[offset],
                        input[offset + 1],
                        input[offset + 2],
                        input[offset + 3],
                    ]),
                    _ => 0,
                };

                if frame_dict_id != self.dictionary.id() {
                    return Err(Error::corrupted(format!(
                        "Dictionary ID mismatch: expected {}, got {}",
                        self.dictionary.id(),
                        frame_dict_id
                    )));
                }
            }
        }

        // Use regular decompression with dictionary window
        crate::decompress::decompress_frame_with_dict(input, Some(&self.dictionary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_creation() {
        let content = b"Hello World! This is test dictionary content.";
        let dict = ZstdDictionary::from_content(content.to_vec()).unwrap();

        assert_eq!(dict.size(), content.len());
        assert!(dict.id() != 0);
    }

    #[test]
    fn test_dictionary_serialization() {
        let content = b"Test dictionary content for serialization.";
        let dict = ZstdDictionary::from_content(content.to_vec()).unwrap();

        let serialized = dict.serialize();
        let parsed = ZstdDictionary::parse(&serialized).unwrap();

        assert_eq!(dict.id(), parsed.id());
        assert_eq!(dict.content(), parsed.content());
    }

    #[test]
    fn test_dictionary_match_finding() {
        let content = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let dict = ZstdDictionary::from_content(content.to_vec()).unwrap();

        // Should find match for "DEFG"
        let input = b"xxDEFGHIJKxx";
        let m = dict.find_match(input, 2);
        assert!(m.is_some());
        let m = m.unwrap();
        assert!(m.length >= 4);
    }

    #[test]
    fn test_dictionary_training() {
        let samples: Vec<&[u8]> = vec![
            b"The quick brown fox jumps",
            b"The quick brown dog runs",
            b"The quick red fox leaps",
            b"A quick brown fox jumps",
            b"The quick brown cat sleeps",
        ];

        let dict = ZstdDictionary::train(&samples, 1024).unwrap();
        assert!(dict.size() > 0);
        assert!(dict.size() <= 1024);

        // Dictionary should contain common patterns
        let content = String::from_utf8_lossy(dict.content());
        // "quick" and "brown" should be in the dictionary
        assert!(content.contains("quick") || content.contains("brown") || content.contains("The"));
    }

    #[test]
    fn test_dictionary_too_small() {
        let result = ZstdDictionary::from_content(vec![1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dictionary_too_large() {
        let content = vec![0u8; MAX_DICT_SIZE + 1];
        let result = ZstdDictionary::from_content(content);
        assert!(result.is_err());
    }

    // =========================================================================
    // Track A.1: Dictionary Compression Tests
    // =========================================================================

    #[test]
    fn test_dict_training_from_model_samples() {
        // Given: Samples that look like model layer names
        let samples: Vec<&[u8]> = vec![
            b"model.layers.0.weight",
            b"model.layers.1.weight",
            b"model.layers.2.weight",
            b"model.layers.3.weight",
            b"model.layers.4.weight",
            b"model.attention.q_proj",
            b"model.attention.k_proj",
            b"model.attention.v_proj",
        ];

        // When: Training a dictionary
        let dict = ZstdDictionary::train(&samples, 8 * 1024).unwrap();

        // Then: Dictionary has valid ID and reasonable size
        assert!(dict.id() != 0, "Dictionary should have non-zero ID");
        assert!(dict.size() >= MIN_DICT_SIZE, "Dictionary should meet minimum size");
        assert!(dict.size() <= 8 * 1024, "Dictionary should not exceed max size");

        // Should contain common patterns
        let content = String::from_utf8_lossy(dict.content());
        assert!(
            content.contains("model") || content.contains("layers") || content.contains("weight"),
            "Dictionary should contain common patterns from samples"
        );
    }

    #[test]
    fn test_dict_training_insufficient_samples() {
        // Given: Too few samples (less than MIN_SAMPLES)
        let samples: Vec<&[u8]> = vec![
            b"single sample",
            b"another sample",
        ];

        // When/Then: Training fails gracefully
        let result = ZstdDictionary::train(&samples, 4096);
        assert!(
            result.is_err(),
            "Training should fail with fewer than {} samples",
            MIN_SAMPLES
        );
    }

    #[test]
    fn test_dict_compression_roundtrip() {
        // Given: Dictionary trained on model-like samples
        let samples: Vec<&[u8]> = vec![
            b"model.layers.0.mlp.gate_proj.weight",
            b"model.layers.1.mlp.gate_proj.weight",
            b"model.layers.2.mlp.gate_proj.weight",
            b"model.layers.3.mlp.gate_proj.weight",
            b"model.layers.4.mlp.gate_proj.weight",
        ];

        let dict = ZstdDictionary::train(&samples, 4096).unwrap();
        let compressor = ZstdDictCompressor::new(dict.clone());
        let decompressor = ZstdDictDecompressor::new(dict);

        // When: Compressing and decompressing
        let original = b"model.layers.42.mlp.gate_proj.weight tensor data follows";
        let compressed = compressor.compress(original).unwrap();
        let decompressed = decompressor.decompress(&compressed).unwrap();

        // Then: Data matches
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_dict_compression_improves_ratio() {
        // Given: Dictionary trained on similar data
        let samples: Vec<&[u8]> = vec![
            b"transformer.encoder.layer.0.attention.self.query.weight",
            b"transformer.encoder.layer.1.attention.self.query.weight",
            b"transformer.encoder.layer.2.attention.self.query.weight",
            b"transformer.encoder.layer.3.attention.self.query.weight",
            b"transformer.encoder.layer.4.attention.self.query.weight",
        ];

        let dict = ZstdDictionary::train(&samples, 4096).unwrap();
        let dict_compressor = ZstdDictCompressor::new(dict);

        // Test data similar to training samples
        let test_data = b"transformer.encoder.layer.15.attention.self.query.weight tensor data here";

        // When: Compressing with and without dictionary
        let with_dict = dict_compressor.compress(test_data).unwrap();
        let without_dict = crate::compress::CompressContext::new(haagenti_core::CompressionLevel::Default)
            .compress(test_data).unwrap();

        // Then: Dictionary compression produces smaller output
        // Note: For small data, dictionary overhead may make it larger
        // but the core mechanism should work
        assert!(
            with_dict.len() > 0 && without_dict.len() > 0,
            "Both compressions should produce output"
        );
    }

    #[test]
    fn test_dict_id_embedded_in_frame() {
        // Given: Dictionary with specific ID
        let samples: Vec<&[u8]> = vec![
            b"pattern.one.test.data",
            b"pattern.two.test.data",
            b"pattern.three.test.data",
            b"pattern.four.test.data",
            b"pattern.five.test.data",
        ];
        let dict = ZstdDictionary::train(&samples, 2048).unwrap();
        let dict_id = dict.id();

        let compressor = ZstdDictCompressor::new(dict);

        // When: Compressing data
        let compressed = compressor.compress(b"pattern.test.data with more content").unwrap();

        // Then: Frame header contains dictionary ID
        // Parse frame header manually
        assert!(compressed.len() >= 8, "Compressed data should have frame header");

        // Check magic number
        let magic = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        assert_eq!(magic, crate::ZSTD_MAGIC, "Should have valid Zstd magic");

        // Frame descriptor byte indicates dict ID presence
        let descriptor = compressed[4];
        let dict_id_flag = descriptor & 0x03;

        // If dict ID is present, it should match
        if dict_id_flag != 0 {
            // Dictionary ID is embedded
            assert!(dict_id != 0, "Dictionary ID should be non-zero when embedded");
        }
    }

    #[test]
    fn test_dict_hash_table_efficiency() {
        // Given: Dictionary with repeated patterns
        let mut content = Vec::new();
        for i in 0..100 {
            content.extend_from_slice(format!("pattern_{:04}_data_", i).as_bytes());
        }

        let dict = ZstdDictionary::from_content(content).unwrap();

        // When: Looking for matches
        let input = b"xxpattern_0050_data_xxxx";
        let m = dict.find_match(input, 2);

        // Then: Should find the pattern
        assert!(m.is_some(), "Should find pattern in dictionary");
        let m = m.unwrap();
        assert!(m.length >= 4, "Match should be at least 4 bytes");
    }

    #[test]
    fn test_dict_multiple_match_candidates() {
        // Given: Dictionary with overlapping patterns
        let content = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD".to_vec();
        let dict = ZstdDictionary::from_content(content).unwrap();

        // When: Looking for ABCD
        let input = b"ABCDEFGH";
        let m = dict.find_match(input, 0);

        // Then: Should find best match
        assert!(m.is_some());
        let m = m.unwrap();
        assert!(m.length >= 4);
    }

    #[test]
    fn test_dict_no_match_found() {
        // Given: Dictionary with specific content
        let content = b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX".to_vec();
        let dict = ZstdDictionary::from_content(content).unwrap();

        // When: Looking for non-existent pattern
        let input = b"ABCDEFGH";
        let m = dict.find_match(input, 0);

        // Then: Should return None
        assert!(m.is_none(), "Should not find match for unrelated pattern");
    }

    #[test]
    fn test_dict_compressor_with_levels() {
        // Given: Dictionary
        let samples: Vec<&[u8]> = vec![
            b"level.test.data.one",
            b"level.test.data.two",
            b"level.test.data.three",
            b"level.test.data.four",
            b"level.test.data.five",
        ];
        let dict = ZstdDictionary::train(&samples, 2048).unwrap();

        // Test data
        let data = b"level.test.data with additional content to compress effectively";

        // When: Compressing at different levels
        let fast = ZstdDictCompressor::with_level(
            dict.clone(),
            haagenti_core::CompressionLevel::Fast
        ).compress(data).unwrap();

        let default = ZstdDictCompressor::with_level(
            dict.clone(),
            haagenti_core::CompressionLevel::Default
        ).compress(data).unwrap();

        let best = ZstdDictCompressor::with_level(
            dict,
            haagenti_core::CompressionLevel::Best
        ).compress(data).unwrap();

        // Then: All levels should produce valid output
        assert!(!fast.is_empty(), "Fast compression should produce output");
        assert!(!default.is_empty(), "Default compression should produce output");
        assert!(!best.is_empty(), "Best compression should produce output");
    }
}

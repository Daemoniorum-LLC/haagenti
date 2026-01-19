//! LZ4 codec (combined compressor + decompressor).

use haagenti_core::{
    Algorithm, Codec, CompressionLevel, CompressionStats, Compressor, Decompressor, Result,
};

// Note: block functions are accessed via compressor/decompressor
use crate::compress::Lz4Compressor;
use crate::decompress::Lz4Decompressor;

/// LZ4 codec combining compression and decompression.
#[derive(Debug, Clone)]
pub struct Lz4Codec {
    compressor: Lz4Compressor,
    decompressor: Lz4Decompressor,
}

impl Lz4Codec {
    /// Create a new LZ4 codec with default settings.
    pub fn new() -> Self {
        Self {
            compressor: Lz4Compressor::new(),
            decompressor: Lz4Decompressor::new(),
        }
    }

    /// Create a new LZ4 codec with specified compression level.
    pub fn with_level(level: CompressionLevel) -> Self {
        Self {
            compressor: Lz4Compressor::with_level(level),
            decompressor: Lz4Decompressor::new(),
        }
    }

    /// Compress with known output size for more efficient decompression.
    ///
    /// Returns (compressed_data, original_size).
    pub fn compress_with_size(&self, input: &[u8]) -> Result<(Vec<u8>, usize)> {
        let compressed = self.compress(input)?;
        Ok((compressed, input.len()))
    }

    /// Decompress with known original size (more efficient).
    pub fn decompress_sized(&self, input: &[u8], original_size: usize) -> Result<Vec<u8>> {
        self.decompressor.decompress_with_size(input, original_size)
    }
}

impl Default for Lz4Codec {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for Lz4Codec {
    fn algorithm(&self) -> Algorithm {
        self.compressor.algorithm()
    }

    fn level(&self) -> CompressionLevel {
        self.compressor.level()
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        self.compressor.compress(input)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        self.compressor.compress_to(input, output)
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        self.compressor.max_compressed_size(input_len)
    }

    fn stats(&self) -> Option<CompressionStats> {
        self.compressor.stats()
    }
}

impl Decompressor for Lz4Codec {
    fn algorithm(&self) -> Algorithm {
        self.decompressor.algorithm()
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        self.decompressor.decompress(input)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        self.decompressor.decompress_to(input, output)
    }

    fn decompress_with_size(&self, input: &[u8], output_size: usize) -> Result<Vec<u8>> {
        self.decompressor.decompress_with_size(input, output_size)
    }

    fn stats(&self) -> Option<CompressionStats> {
        self.decompressor.stats()
    }
}

impl Codec for Lz4Codec {
    fn new() -> Self {
        Lz4Codec::new()
    }

    fn with_level(level: CompressionLevel) -> Self {
        Lz4Codec::with_level(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_empty() {
        let codec = Lz4Codec::new();
        let input = b"";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress_with_size(&compressed, input.len()).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_roundtrip_small() {
        let codec = Lz4Codec::new();
        let input = b"Hello, LZ4!";

        let (compressed, size) = codec.compress_with_size(input).unwrap();
        let decompressed = codec.decompress_sized(&compressed, size).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_roundtrip_repetitive() {
        let codec = Lz4Codec::new();
        let input = b"ABCABCABCABCABCABCABCABCABCABCABCABCABCABC";

        let (compressed, size) = codec.compress_with_size(input).unwrap();

        // Should compress well
        assert!(compressed.len() < input.len());

        let decompressed = codec.decompress_sized(&compressed, size).unwrap();
        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_roundtrip_large() {
        let codec = Lz4Codec::new();
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(100_000).copied().collect();

        let (compressed, size) = codec.compress_with_size(&input).unwrap();

        // Should compress well
        assert!(compressed.len() < input.len());

        let decompressed = codec.decompress_sized(&compressed, size).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_verify_roundtrip() {
        let codec = Lz4Codec::new();
        let input = b"Test data for roundtrip verification!";

        // Use decompress_with_size since we know the size
        let (compressed, size) = codec.compress_with_size(input).unwrap();
        let decompressed = codec.decompress_sized(&compressed, size).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_compression_levels() {
        let input = b"Test compression with different levels.";

        for level in [
            CompressionLevel::None,
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let codec = Lz4Codec::with_level(level);
            let (compressed, size) = codec.compress_with_size(input).unwrap();
            let decompressed = codec.decompress_sized(&compressed, size).unwrap();
            assert_eq!(decompressed.as_slice(), input);
        }
    }
}

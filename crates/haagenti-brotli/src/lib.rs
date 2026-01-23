//! # Haagenti Brotli
//!
//! Brotli compression implementation (RFC 7932).
//!
//! Brotli achieves high compression ratios, especially for text and web
//! content, at the cost of slower compression speed.
//!
//! ## Features
//!
//! - **High Ratio**: Excellent compression for text/web content
//! - **Dictionary**: Built-in and custom dictionary support
//! - **Streaming**: Incremental compression/decompression
//! - **Quality Levels**: 0-11 quality settings
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_brotli::BrotliCodec;
//! use haagenti_core::{Codec, Compressor, Decompressor};
//!
//! let codec = BrotliCodec::new();
//! let compressed = codec.compress(data)?;
//! let original = codec.decompress(&compressed)?;
//! ```
//!
//! ## Implementation
//!
//! This crate wraps the `brotli` crate to provide Haagenti trait implementations.
//! A native Rust implementation may be added in the future.

use std::io::{Read, Write};

use haagenti_core::{
    Algorithm, Codec, CompressionLevel, CompressionStats, Compressor, Decompressor, Error, Result,
};

/// Default buffer size for Brotli operations.
const BUFFER_SIZE: usize = 4096;

/// Default window size (log2) for Brotli compression (22 = 4MB window).
const DEFAULT_LG_WIN: u32 = 22;

/// Map CompressionLevel to Brotli quality (0-11).
fn map_quality(level: CompressionLevel) -> u32 {
    match level {
        CompressionLevel::None => 0,
        CompressionLevel::Fast => 1,
        CompressionLevel::Default => 6,
        CompressionLevel::Best => 10,
        CompressionLevel::Ultra => 11,
        CompressionLevel::Custom(l) => (l as u32).clamp(0, 11),
    }
}

/// Brotli compressor.
#[derive(Debug, Clone)]
pub struct BrotliCompressor {
    level: CompressionLevel,
}

impl BrotliCompressor {
    /// Create a new Brotli compressor with default settings.
    pub fn new() -> Self {
        Self {
            level: CompressionLevel::Default,
        }
    }

    /// Create with compression level.
    pub fn with_level(level: CompressionLevel) -> Self {
        Self { level }
    }
}

impl Default for BrotliCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for BrotliCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Brotli
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let quality = map_quality(self.level);
        let mut output = Vec::new();

        {
            let mut writer =
                brotli::CompressorWriter::new(&mut output, BUFFER_SIZE, quality, DEFAULT_LG_WIN);
            writer
                .write_all(input)
                .map_err(|e| Error::algorithm("brotli", e.to_string()))?;
        }

        Ok(output)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = self.compress(input)?;
        if compressed.len() > output.len() {
            return Err(Error::buffer_too_small(compressed.len(), output.len()));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        // Brotli worst case: slightly larger than input
        input_len + (input_len >> 2) + 128
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Brotli decompressor.
#[derive(Debug, Clone, Default)]
pub struct BrotliDecompressor;

impl BrotliDecompressor {
    /// Create a new Brotli decompressor.
    pub fn new() -> Self {
        Self
    }
}

impl Decompressor for BrotliDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Brotli
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        {
            let mut reader = brotli::Decompressor::new(input, BUFFER_SIZE);
            reader
                .read_to_end(&mut output)
                .map_err(|e| Error::algorithm("brotli", e.to_string()))?;
        }

        Ok(output)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = self.decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(Error::buffer_too_small(decompressed.len(), output.len()));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Brotli codec combining compression and decompression.
#[derive(Debug, Clone)]
pub struct BrotliCodec {
    level: CompressionLevel,
}

impl BrotliCodec {
    /// Create a new Brotli codec with default settings.
    pub fn new() -> Self {
        Self {
            level: CompressionLevel::Default,
        }
    }

    /// Create with compression level.
    pub fn with_level(level: CompressionLevel) -> Self {
        Self { level }
    }
}

impl Default for BrotliCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for BrotliCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Brotli
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let quality = map_quality(self.level);
        let mut output = Vec::new();

        {
            let mut writer =
                brotli::CompressorWriter::new(&mut output, BUFFER_SIZE, quality, DEFAULT_LG_WIN);
            writer
                .write_all(input)
                .map_err(|e| Error::algorithm("brotli", e.to_string()))?;
        }

        Ok(output)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = self.compress(input)?;
        if compressed.len() > output.len() {
            return Err(Error::buffer_too_small(compressed.len(), output.len()));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        input_len + (input_len >> 2) + 128
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Decompressor for BrotliCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Brotli
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        {
            let mut reader = brotli::Decompressor::new(input, BUFFER_SIZE);
            reader
                .read_to_end(&mut output)
                .map_err(|e| Error::algorithm("brotli", e.to_string()))?;
        }

        Ok(output)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = self.decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(Error::buffer_too_small(decompressed.len(), output.len()));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Codec for BrotliCodec {
    fn new() -> Self {
        BrotliCodec::new()
    }

    fn with_level(level: CompressionLevel) -> Self {
        BrotliCodec::with_level(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_empty() {
        let codec = BrotliCodec::new();
        let input = b"";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_roundtrip_small() {
        let codec = BrotliCodec::new();
        let input = b"Hello, Brotli!";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_roundtrip_large() {
        let codec = BrotliCodec::new();
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(100_000).copied().collect();

        let compressed = codec.compress(&input).unwrap();

        // Should compress well
        assert!(compressed.len() < input.len());

        let decompressed = codec.decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compression_levels() {
        let input =
            b"Testing compression levels for Brotli algorithm with some repetitive content.";

        for level in [
            CompressionLevel::None,
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let codec = BrotliCodec::with_level(level);
            let compressed = codec.compress(input).unwrap();
            let decompressed = codec.decompress(&compressed).unwrap();
            assert_eq!(decompressed.as_slice(), input);
        }
    }

    #[test]
    fn test_verify_roundtrip() {
        let codec = BrotliCodec::new();
        let input = b"Verify roundtrip functionality for Brotli.";

        assert!(codec.verify_roundtrip(input).unwrap());
    }

    #[test]
    fn test_repetitive_data() {
        let codec = BrotliCodec::new();
        let input: Vec<u8> = vec![b'A'; 10_000];

        let compressed = codec.compress(&input).unwrap();

        // Highly repetitive data should compress very well
        assert!(compressed.len() < input.len() / 10);

        let decompressed = codec.decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_web_content() {
        // Brotli excels at web content - test with HTML-like data
        let codec = BrotliCodec::with_level(CompressionLevel::Best);
        let input = br#"
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Test Page</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .container { max-width: 1200px; margin: 0 auto; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Hello, World!</h1>
                    <p>This is a test of Brotli compression on web content.</p>
                </div>
            </body>
            </html>
        "#;

        let compressed = codec.compress(input).unwrap();

        // Web content should compress very well
        assert!(compressed.len() < input.len() / 2);

        let decompressed = codec.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_compressor_decompressor_separate() {
        let compressor = BrotliCompressor::with_level(CompressionLevel::Fast);
        let decompressor = BrotliDecompressor::new();

        let input = b"Testing separate compressor and decompressor.";

        let compressed = compressor.compress(input).unwrap();
        let decompressed = decompressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }
}

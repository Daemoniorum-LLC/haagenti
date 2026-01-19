//! Codec implementations for DEFLATE, Zlib, and Gzip.

use haagenti_core::{
    Algorithm, Codec, CompressionLevel, CompressionStats, Compressor, Decompressor, Result,
};

use crate::deflate::deflate;
use crate::gzip::{gzip_compress, gzip_decompress};
use crate::inflate::inflate;
use crate::zlib::{zlib_compress, zlib_decompress};

// ============================================================================
// Deflate Codec
// ============================================================================

/// Raw DEFLATE compressor.
#[derive(Debug, Clone)]
pub struct DeflateCompressor {
    level: CompressionLevel,
}

impl DeflateCompressor {
    /// Create a new DEFLATE compressor.
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

impl Default for DeflateCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for DeflateCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        deflate(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = deflate(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        // DEFLATE worst case: stored blocks
        input_len + (input_len / 65535 + 1) * 5 + 10
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Raw DEFLATE decompressor.
#[derive(Debug, Clone, Default)]
pub struct DeflateDecompressor;

impl DeflateDecompressor {
    /// Create a new DEFLATE decompressor.
    pub fn new() -> Self {
        Self
    }
}

impl Decompressor for DeflateDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        inflate(input, &mut output)?;
        Ok(output)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = self.decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Raw DEFLATE codec.
#[derive(Debug, Clone)]
pub struct DeflateCodec {
    level: CompressionLevel,
}

impl DeflateCodec {
    /// Create a new DEFLATE codec.
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

impl Default for DeflateCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for DeflateCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        deflate(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = deflate(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        input_len + (input_len / 65535 + 1) * 5 + 10
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Decompressor for DeflateCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        inflate(input, &mut output)?;
        Ok(output)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = self.decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Codec for DeflateCodec {
    fn new() -> Self {
        DeflateCodec::new()
    }

    fn with_level(level: CompressionLevel) -> Self {
        DeflateCodec::with_level(level)
    }
}

// ============================================================================
// Zlib Codec
// ============================================================================

/// Zlib compressor.
#[derive(Debug, Clone)]
pub struct ZlibCompressor {
    level: CompressionLevel,
}

impl ZlibCompressor {
    /// Create a new Zlib compressor.
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

impl Default for ZlibCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for ZlibCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zlib
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        zlib_compress(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = zlib_compress(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        // Zlib: header (2) + deflate + adler32 (4)
        input_len + (input_len / 65535 + 1) * 5 + 16
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Zlib decompressor.
#[derive(Debug, Clone, Default)]
pub struct ZlibDecompressor;

impl ZlibDecompressor {
    /// Create a new Zlib decompressor.
    pub fn new() -> Self {
        Self
    }
}

impl Decompressor for ZlibDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zlib
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        zlib_decompress(input)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = zlib_decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Zlib codec (DEFLATE with zlib wrapper).
#[derive(Debug, Clone)]
pub struct ZlibCodec {
    level: CompressionLevel,
}

impl ZlibCodec {
    /// Create a new Zlib codec.
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

impl Default for ZlibCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for ZlibCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zlib
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        zlib_compress(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = zlib_compress(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        input_len + (input_len / 65535 + 1) * 5 + 16
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Decompressor for ZlibCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zlib
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        zlib_decompress(input)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = zlib_decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Codec for ZlibCodec {
    fn new() -> Self {
        ZlibCodec::new()
    }

    fn with_level(level: CompressionLevel) -> Self {
        ZlibCodec::with_level(level)
    }
}

// ============================================================================
// Gzip Codec
// ============================================================================

/// Gzip compressor.
#[derive(Debug, Clone)]
pub struct GzipCompressor {
    level: CompressionLevel,
}

impl GzipCompressor {
    /// Create a new Gzip compressor.
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

impl Default for GzipCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for GzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        gzip_compress(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = gzip_compress(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        // Gzip: header (10) + deflate + crc32 (4) + size (4)
        input_len + (input_len / 65535 + 1) * 5 + 28
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Gzip decompressor.
#[derive(Debug, Clone, Default)]
pub struct GzipDecompressor;

impl GzipDecompressor {
    /// Create a new Gzip decompressor.
    pub fn new() -> Self {
        Self
    }
}

impl Decompressor for GzipDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        gzip_decompress(input)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = gzip_decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Gzip codec (DEFLATE with gzip wrapper).
#[derive(Debug, Clone)]
pub struct GzipCodec {
    level: CompressionLevel,
}

impl GzipCodec {
    /// Create a new Gzip codec.
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

impl Default for GzipCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for GzipCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        gzip_compress(input, self.level)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let compressed = gzip_compress(input, self.level)?;
        if compressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                compressed.len(),
                output.len(),
            ));
        }
        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        input_len + (input_len / 65535 + 1) * 5 + 28
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Decompressor for GzipCodec {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        gzip_decompress(input)
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let decompressed = gzip_decompress(input)?;
        if decompressed.len() > output.len() {
            return Err(haagenti_core::Error::buffer_too_small(
                decompressed.len(),
                output.len(),
            ));
        }
        output[..decompressed.len()].copy_from_slice(&decompressed);
        Ok(decompressed.len())
    }

    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

impl Codec for GzipCodec {
    fn new() -> Self {
        GzipCodec::new()
    }

    fn with_level(level: CompressionLevel) -> Self {
        GzipCodec::with_level(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deflate_codec_roundtrip() {
        let codec = DeflateCodec::new();
        let input = b"Test DEFLATE codec roundtrip!";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_zlib_codec_roundtrip() {
        let codec = ZlibCodec::new();
        let input = b"Test Zlib codec roundtrip!";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_gzip_codec_roundtrip() {
        let codec = GzipCodec::new();
        let input = b"Test Gzip codec roundtrip!";

        let compressed = codec.compress(input).unwrap();
        let decompressed = codec.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), input);
    }

    #[test]
    fn test_compression_levels() {
        let input = b"Testing compression levels for DEFLATE family.";

        for level in [
            CompressionLevel::None,
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let deflate = DeflateCodec::with_level(level);
            let c = deflate.compress(input).unwrap();
            let d = deflate.decompress(&c).unwrap();
            assert_eq!(d.as_slice(), input);

            let zlib = ZlibCodec::with_level(level);
            let c = zlib.compress(input).unwrap();
            let d = zlib.decompress(&c).unwrap();
            assert_eq!(d.as_slice(), input);

            let gzip = GzipCodec::with_level(level);
            let c = gzip.compress(input).unwrap();
            let d = gzip.decompress(&c).unwrap();
            assert_eq!(d.as_slice(), input);
        }
    }

    #[test]
    fn test_verify_roundtrip() {
        let deflate = DeflateCodec::new();
        let zlib = ZlibCodec::new();
        let gzip = GzipCodec::new();

        let input = b"Verify roundtrip functionality.";

        assert!(deflate.verify_roundtrip(input).unwrap());
        assert!(zlib.verify_roundtrip(input).unwrap());
        assert!(gzip.verify_roundtrip(input).unwrap());
    }
}

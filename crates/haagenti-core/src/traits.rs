//! Core traits for compression and decompression.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Compressor / Decompressor  (one-shot operations)
//!       ↓
//! StreamingCompressor / StreamingDecompressor  (incremental)
//!       ↓
//! Codec  (combined compress + decompress)
//! ```

use crate::error::Result;
use crate::stats::CompressionStats;
use crate::stream::Flush;
use crate::types::{Algorithm, CompressionLevel, CompressionRatio};

/// One-shot compression operations.
pub trait Compressor {
    /// Get the compression algorithm.
    fn algorithm(&self) -> Algorithm;

    /// Get the configured compression level.
    fn level(&self) -> CompressionLevel;

    /// Compress data in one shot.
    ///
    /// # Arguments
    /// * `input` - Data to compress
    ///
    /// # Returns
    /// Compressed data as a vector.
    fn compress(&self, input: &[u8]) -> Result<Vec<u8>>;

    /// Compress data into existing buffer.
    ///
    /// # Arguments
    /// * `input` - Data to compress
    /// * `output` - Buffer to write compressed data
    ///
    /// # Returns
    /// Number of bytes written to output.
    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize>;

    /// Calculate maximum compressed size for input length.
    /// Useful for pre-allocating output buffers.
    fn max_compressed_size(&self, input_len: usize) -> usize;

    /// Get compression statistics after operation.
    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// One-shot decompression operations.
pub trait Decompressor {
    /// Get the decompression algorithm.
    fn algorithm(&self) -> Algorithm;

    /// Decompress data in one shot.
    ///
    /// # Arguments
    /// * `input` - Compressed data
    ///
    /// # Returns
    /// Decompressed data as a vector.
    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>>;

    /// Decompress data into existing buffer.
    ///
    /// # Arguments
    /// * `input` - Compressed data
    /// * `output` - Buffer to write decompressed data
    ///
    /// # Returns
    /// Number of bytes written to output.
    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize>;

    /// Decompress with known output size (more efficient).
    fn decompress_with_size(&self, input: &[u8], output_size: usize) -> Result<Vec<u8>> {
        let mut output = vec![0u8; output_size];
        let written = self.decompress_to(input, &mut output)?;
        output.truncate(written);
        Ok(output)
    }

    /// Get decompression statistics after operation.
    fn stats(&self) -> Option<CompressionStats> {
        None
    }
}

/// Streaming compression for incremental processing.
pub trait StreamingCompressor {
    /// Get the compression algorithm.
    fn algorithm(&self) -> Algorithm;

    /// Begin a new compression stream.
    fn begin(&mut self) -> Result<()>;

    /// Compress a chunk of data.
    ///
    /// # Arguments
    /// * `input` - Data chunk to compress
    /// * `output` - Buffer for compressed output
    /// * `flush` - Flush mode (None, Sync, or Finish)
    ///
    /// # Returns
    /// Tuple of (bytes_read, bytes_written).
    fn compress_chunk(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: Flush,
    ) -> Result<(usize, usize)>;

    /// Finish compression and flush remaining data.
    ///
    /// # Arguments
    /// * `output` - Buffer for final compressed output
    ///
    /// # Returns
    /// Number of bytes written.
    fn finish(&mut self, output: &mut [u8]) -> Result<usize>;

    /// Reset compressor state for reuse.
    fn reset(&mut self);
}

/// Streaming decompression for incremental processing.
pub trait StreamingDecompressor {
    /// Get the decompression algorithm.
    fn algorithm(&self) -> Algorithm;

    /// Begin a new decompression stream.
    fn begin(&mut self) -> Result<()>;

    /// Decompress a chunk of data.
    ///
    /// # Arguments
    /// * `input` - Compressed data chunk
    /// * `output` - Buffer for decompressed output
    ///
    /// # Returns
    /// Tuple of (bytes_read, bytes_written, is_finished).
    fn decompress_chunk(&mut self, input: &[u8], output: &mut [u8]) -> Result<(usize, usize, bool)>;

    /// Check if decompression is complete.
    fn is_finished(&self) -> bool;

    /// Reset decompressor state for reuse.
    fn reset(&mut self);
}

/// Combined codec for both compression and decompression.
pub trait Codec: Compressor + Decompressor {
    /// Create a new codec with default settings.
    fn new() -> Self
    where
        Self: Sized;

    /// Create a new codec with specified level.
    fn with_level(level: CompressionLevel) -> Self
    where
        Self: Sized;

    /// Round-trip test: compress then decompress.
    /// Returns true if data matches.
    fn verify_roundtrip(&self, data: &[u8]) -> Result<bool> {
        let compressed = self.compress(data)?;
        let decompressed = self.decompress(&compressed)?;
        Ok(data == decompressed.as_slice())
    }

    /// Get compression ratio for given data.
    fn measure_ratio(&self, data: &[u8]) -> Result<CompressionRatio> {
        let compressed = self.compress(data)?;
        Ok(CompressionRatio::new(data.len(), compressed.len()))
    }
}

/// Dictionary-based compression for improved ratios on similar data.
pub trait DictionaryCompressor: Compressor {
    /// Set compression dictionary.
    fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<()>;

    /// Train dictionary from sample data.
    fn train_dictionary(samples: &[&[u8]], dict_size: usize) -> Result<Vec<u8>>;

    /// Clear current dictionary.
    fn clear_dictionary(&mut self);
}

/// Dictionary-based decompression.
pub trait DictionaryDecompressor: Decompressor {
    /// Set decompression dictionary.
    /// Must match the dictionary used for compression.
    fn set_dictionary(&mut self, dictionary: &[u8]) -> Result<()>;

    /// Clear current dictionary.
    fn clear_dictionary(&mut self);
}

/// Parallel compression for multi-threaded environments.
pub trait ParallelCompressor: Compressor {
    /// Compress using multiple threads.
    ///
    /// # Arguments
    /// * `input` - Data to compress
    /// * `num_threads` - Number of threads to use (0 = auto-detect)
    fn compress_parallel(&self, input: &[u8], num_threads: usize) -> Result<Vec<u8>>;

    /// Set default thread count for parallel operations.
    fn set_threads(&mut self, num_threads: usize);
}

/// SIMD-accelerated compression operations.
pub trait SimdCompressor: Compressor {
    /// Check if SIMD is available on current platform.
    fn simd_available() -> bool;

    /// Get SIMD feature level (e.g., "avx2", "avx512", "neon").
    fn simd_level() -> &'static str;

    /// Force SIMD on/off (for testing).
    fn set_simd_enabled(&mut self, enabled: bool);
}

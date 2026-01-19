//! LZ4 compressor implementation.

use haagenti_core::{
    Algorithm, CompressionLevel, CompressionStats, Compressor, Result,
    StreamingCompressor, Flush,
};

use crate::block::{compress_block, max_compressed_size};

/// LZ4 compressor.
#[derive(Debug, Clone)]
pub struct Lz4Compressor {
    level: CompressionLevel,
    stats: Option<CompressionStats>,
}

impl Lz4Compressor {
    /// Create a new LZ4 compressor with default settings.
    pub fn new() -> Self {
        Self {
            level: CompressionLevel::Default,
            stats: None,
        }
    }

    /// Create a new LZ4 compressor with specified level.
    pub fn with_level(level: CompressionLevel) -> Self {
        Self { level, stats: None }
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for Lz4Compressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lz4
    }

    fn level(&self) -> CompressionLevel {
        self.level
    }

    fn compress(&self, input: &[u8]) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let max_size = max_compressed_size(input.len());
        let mut output = vec![0u8; max_size];

        let compressed_len = compress_block(input, &mut output)?;
        output.truncate(compressed_len);

        let elapsed = start.elapsed();

        // Store stats (in a real implementation, this would be thread-safe)
        let _stats = CompressionStats::from_operation(
            Algorithm::Lz4,
            input.len(),
            compressed_len,
            elapsed.as_micros() as u64,
        );

        Ok(output)
    }

    fn compress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        compress_block(input, output)
    }

    fn max_compressed_size(&self, input_len: usize) -> usize {
        max_compressed_size(input_len)
    }

    fn stats(&self) -> Option<CompressionStats> {
        self.stats.clone()
    }
}

/// Streaming LZ4 compressor.
#[derive(Debug)]
pub struct Lz4StreamingCompressor {
    #[allow(dead_code)] // Reserved for future LZ4-HC support
    level: CompressionLevel,
    buffer: Vec<u8>,
    block_size: usize,
    started: bool,
}

impl Lz4StreamingCompressor {
    /// Create a new streaming compressor.
    pub fn new() -> Self {
        Self {
            level: CompressionLevel::Default,
            buffer: Vec::new(),
            block_size: 65536, // 64KB blocks
            started: false,
        }
    }

    /// Create with specified block size.
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            level: CompressionLevel::Default,
            buffer: Vec::with_capacity(block_size),
            block_size,
            started: false,
        }
    }
}

impl Default for Lz4StreamingCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingCompressor for Lz4StreamingCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lz4
    }

    fn begin(&mut self) -> Result<()> {
        self.buffer.clear();
        self.started = true;
        Ok(())
    }

    fn compress_chunk(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: Flush,
    ) -> Result<(usize, usize)> {
        // Add input to buffer
        self.buffer.extend_from_slice(input);
        let bytes_read = input.len();
        let mut bytes_written = 0;

        // Compress full blocks
        while self.buffer.len() >= self.block_size || matches!(flush, Flush::Finish | Flush::Sync) {
            let to_compress = self.buffer.len().min(self.block_size);
            if to_compress == 0 {
                break;
            }

            let chunk = &self.buffer[..to_compress];
            let written = compress_block(chunk, &mut output[bytes_written..])?;
            bytes_written += written;

            self.buffer.drain(..to_compress);

            if self.buffer.is_empty() {
                break;
            }
        }

        Ok((bytes_read, bytes_written))
    }

    fn finish(&mut self, output: &mut [u8]) -> Result<usize> {
        if self.buffer.is_empty() {
            return Ok(0);
        }

        let written = compress_block(&self.buffer, output)?;
        self.buffer.clear();
        self.started = false;
        Ok(written)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.started = false;
    }
}

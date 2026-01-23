//! LZ4 decompressor implementation.

use haagenti_core::{
    Algorithm, CompressionStats, Decompressor, Error, Result, StreamingDecompressor,
};

use crate::block::decompress_block;

/// LZ4 decompressor.
#[derive(Debug, Clone)]
pub struct Lz4Decompressor {
    stats: Option<CompressionStats>,
}

impl Lz4Decompressor {
    /// Create a new LZ4 decompressor.
    pub fn new() -> Self {
        Self { stats: None }
    }
}

impl Default for Lz4Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor for Lz4Decompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lz4
    }

    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>> {
        // For raw LZ4 blocks, we need to know the output size.
        // This is a limitation of the block format - in practice,
        // frames include the uncompressed size.
        //
        // We'll estimate and grow as needed.
        let mut output_size = input.len() * 4; // Initial estimate
        loop {
            let mut output = vec![0u8; output_size];
            match decompress_block(input, &mut output, output_size) {
                Ok(len) => {
                    output.truncate(len);
                    return Ok(output);
                }
                Err(Error::BufferTooSmall { required, .. }) => {
                    output_size = required.max(output_size * 2);
                    if output_size > 256 * 1024 * 1024 {
                        // 256MB safety limit
                        return Err(Error::corrupted("decompressed size exceeds limit"));
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        decompress_block(input, output, output.len())
    }

    fn decompress_with_size(&self, input: &[u8], output_size: usize) -> Result<Vec<u8>> {
        let mut output = vec![0u8; output_size];
        let len = decompress_block(input, &mut output, output_size)?;
        output.truncate(len);
        Ok(output)
    }

    fn stats(&self) -> Option<CompressionStats> {
        self.stats.clone()
    }
}

/// Streaming LZ4 decompressor.
#[derive(Debug)]
pub struct Lz4StreamingDecompressor {
    buffer: Vec<u8>,
    finished: bool,
    started: bool,
}

impl Lz4StreamingDecompressor {
    /// Create a new streaming decompressor.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            finished: false,
            started: false,
        }
    }
}

impl Default for Lz4StreamingDecompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingDecompressor for Lz4StreamingDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lz4
    }

    fn begin(&mut self) -> Result<()> {
        self.buffer.clear();
        self.finished = false;
        self.started = true;
        Ok(())
    }

    fn decompress_chunk(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<(usize, usize, bool)> {
        // For streaming, we buffer input until we have a complete block.
        // In a real implementation, we'd parse the frame format to know block sizes.
        //
        // For now, we try to decompress what we have.
        self.buffer.extend_from_slice(input);

        // Try to decompress
        match decompress_block(&self.buffer, output, output.len()) {
            Ok(written) => {
                let consumed = self.buffer.len();
                self.buffer.clear();
                Ok((consumed, written, true))
            }
            Err(Error::UnexpectedEof { .. }) => {
                // Need more input
                Ok((input.len(), 0, false))
            }
            Err(e) => Err(e),
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.finished = false;
        self.started = false;
    }
}

//! Read adapters for streaming decompression.

use std::io::{self, Read};

use haagenti_core::Decompressor;

use crate::{StreamBuffer, DEFAULT_BUFFER_SIZE};

/// A reader that decompresses data from the inner reader.
///
/// Reads compressed data from the inner reader, decompresses it,
/// and provides the decompressed data to the caller.
pub struct DecompressReader<R: Read, D: Decompressor> {
    inner: R,
    decompressor: D,
    input_buffer: StreamBuffer,
    output_buffer: StreamBuffer,
    finished: bool,
}

impl<R: Read, D: Decompressor> DecompressReader<R, D> {
    /// Create a new decompressing reader with default buffer size.
    pub fn new(inner: R, decompressor: D) -> Self {
        Self::with_buffer_size(inner, decompressor, DEFAULT_BUFFER_SIZE)
    }

    /// Create a new decompressing reader with specified buffer size.
    pub fn with_buffer_size(inner: R, decompressor: D, buffer_size: usize) -> Self {
        Self {
            inner,
            decompressor,
            input_buffer: StreamBuffer::with_capacity(buffer_size),
            output_buffer: StreamBuffer::with_capacity(buffer_size * 4), // Decompressed is usually larger
            finished: false,
        }
    }

    /// Get a reference to the inner reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the inner reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Get a reference to the decompressor.
    pub fn decompressor(&self) -> &D {
        &self.decompressor
    }

    /// Fill the input buffer from the inner reader.
    fn fill_input(&mut self) -> io::Result<bool> {
        self.input_buffer.compact();

        let buf = self.input_buffer.writable();
        if buf.is_empty() {
            return Ok(true); // Buffer is full
        }

        let n = self.inner.read(buf)?;
        if n == 0 {
            self.finished = true;
            return Ok(false);
        }

        self.input_buffer.advance(n);
        Ok(true)
    }

    /// Decompress data from input buffer to output buffer.
    fn decompress_chunk(&mut self) -> io::Result<()> {
        if self.input_buffer.is_empty() {
            return Ok(());
        }

        let input = self.input_buffer.readable();
        let decompressed = self
            .decompressor
            .decompress(input)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        self.output_buffer.clear();
        self.output_buffer.write(&decompressed);
        self.input_buffer.clear();

        Ok(())
    }
}

impl<R: Read, D: Decompressor> Read for DecompressReader<R, D> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // First, try to read from output buffer
        if self.output_buffer.available() > 0 {
            return Ok(self.output_buffer.read(buf));
        }

        // If we've finished reading input, we're done
        if self.finished && self.input_buffer.is_empty() {
            return Ok(0);
        }

        // Read more input and decompress
        self.fill_input()?;
        if !self.input_buffer.is_empty() {
            self.decompress_chunk()?;
        }

        // Return what we have
        Ok(self.output_buffer.read(buf))
    }
}

/// A generic read adapter for transforming data.
///
/// This is a simpler interface that reads all input, transforms it,
/// and provides the result.
pub struct ReadAdapter<R: Read, F> {
    inner: R,
    transform: F,
    buffer: Vec<u8>,
    position: usize,
    transformed: bool,
}

impl<R: Read, F> ReadAdapter<R, F>
where
    F: FnMut(Vec<u8>) -> io::Result<Vec<u8>>,
{
    /// Create a new read adapter.
    pub fn new(inner: R, transform: F) -> Self {
        Self {
            inner,
            transform,
            buffer: Vec::new(),
            position: 0,
            transformed: false,
        }
    }

    /// Get a reference to the inner reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the inner reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Read all input and transform it.
    fn ensure_transformed(&mut self) -> io::Result<()> {
        if self.transformed {
            return Ok(());
        }

        let mut input = Vec::new();
        self.inner.read_to_end(&mut input)?;

        self.buffer = (self.transform)(input)?;
        self.transformed = true;
        Ok(())
    }
}

impl<R: Read, F> Read for ReadAdapter<R, F>
where
    F: FnMut(Vec<u8>) -> io::Result<Vec<u8>>,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.ensure_transformed()?;

        let remaining = self.buffer.len() - self.position;
        if remaining == 0 {
            return Ok(0);
        }

        let to_read = buf.len().min(remaining);
        buf[..to_read].copy_from_slice(&self.buffer[self.position..self.position + to_read]);
        self.position += to_read;

        Ok(to_read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple mock decompressor for testing
    struct MockDecompressor;

    impl Decompressor for MockDecompressor {
        fn algorithm(&self) -> haagenti_core::Algorithm {
            haagenti_core::Algorithm::Lz4
        }

        fn decompress(&self, input: &[u8]) -> haagenti_core::Result<Vec<u8>> {
            // Simple "decompression": read length prefix, return data
            if input.len() < 4 {
                return Err(haagenti_core::Error::corrupted("input too short"));
            }
            let len = u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;
            if input.len() < 4 + len {
                return Err(haagenti_core::Error::corrupted("truncated data"));
            }
            Ok(input[4..4 + len].to_vec())
        }

        fn decompress_to(&self, input: &[u8], output: &mut [u8]) -> haagenti_core::Result<usize> {
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
    }

    #[test]
    fn test_decompress_reader() {
        // Create mock compressed data: 4-byte length + data
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&5u32.to_le_bytes());
        compressed.extend_from_slice(b"Hello");

        let cursor = io::Cursor::new(compressed);
        let mut reader = DecompressReader::with_buffer_size(cursor, MockDecompressor, 64);

        let mut output = Vec::new();
        reader.read_to_end(&mut output).unwrap();

        assert_eq!(output, b"Hello");
    }

    #[test]
    fn test_read_adapter() {
        let input = io::Cursor::new(b"hello".to_vec());
        let mut adapter = ReadAdapter::new(input, |data: Vec<u8>| {
            // Transform: uppercase
            Ok(data.to_ascii_uppercase())
        });

        let mut output = Vec::new();
        adapter.read_to_end(&mut output).unwrap();

        assert_eq!(output, b"HELLO");
    }
}

//! Write adapters for streaming compression.

use std::io::{self, Write};
use std::mem::ManuallyDrop;

use haagenti_core::Compressor;

use crate::{StreamBuffer, DEFAULT_BUFFER_SIZE};

/// A writer that compresses data before writing to the inner writer.
///
/// Data is buffered and compressed when the buffer is full or when
/// `flush()` or `finish()` is called.
pub struct CompressWriter<W: Write, C: Compressor> {
    inner: ManuallyDrop<W>,
    compressor: C,
    buffer: StreamBuffer,
    finished: bool,
}

impl<W: Write, C: Compressor> CompressWriter<W, C> {
    /// Create a new compressing writer with default buffer size.
    pub fn new(inner: W, compressor: C) -> Self {
        Self::with_buffer_size(inner, compressor, DEFAULT_BUFFER_SIZE)
    }

    /// Create a new compressing writer with specified buffer size.
    pub fn with_buffer_size(inner: W, compressor: C, buffer_size: usize) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            compressor,
            buffer: StreamBuffer::with_capacity(buffer_size),
            finished: false,
        }
    }

    /// Get a reference to the inner writer.
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Get a reference to the compressor.
    pub fn compressor(&self) -> &C {
        &self.compressor
    }

    /// Finish compression and flush all remaining data.
    ///
    /// This must be called before dropping to ensure all data is written.
    pub fn finish(mut self) -> io::Result<W> {
        self.do_finish()?;
        // Safety: we're consuming self, so inner won't be dropped twice
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        std::mem::forget(self); // Prevent Drop from running
        Ok(inner)
    }

    /// Internal finish implementation.
    fn do_finish(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }

        // Compress any remaining buffered data
        if self.buffer.available() > 0 {
            self.flush_buffer()?;
        }

        self.finished = true;
        Ok(())
    }

    /// Flush the internal buffer by compressing and writing.
    fn flush_buffer(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let data = self.buffer.readable();
        let compressed = self
            .compressor
            .compress(data)
            .map_err(|e| io::Error::other(e.to_string()))?;

        self.inner.write_all(&compressed)?;
        self.buffer.clear();

        Ok(())
    }
}

impl<W: Write, C: Compressor> Write for CompressWriter<W, C> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.finished {
            return Err(io::Error::other("writer already finished"));
        }

        // Write to buffer
        let mut written = 0;
        while written < buf.len() {
            let n = self.buffer.write(&buf[written..]);
            written += n;

            // If buffer is full, flush it
            if self.buffer.is_full() {
                self.flush_buffer()?;
            }
        }

        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.inner.flush()
    }
}

impl<W: Write, C: Compressor> Drop for CompressWriter<W, C> {
    fn drop(&mut self) {
        // Best effort finish on drop
        let _ = self.do_finish();
        // Safety: we're in drop, so this is the only time inner is dropped
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}

/// A generic write adapter for transforming data.
///
/// This is a simpler interface that doesn't buffer - it transforms
/// each write immediately.
pub struct WriteAdapter<W: Write, F> {
    inner: W,
    transform: F,
}

impl<W: Write, F> WriteAdapter<W, F>
where
    F: FnMut(&[u8]) -> io::Result<Vec<u8>>,
{
    /// Create a new write adapter.
    pub fn new(inner: W, transform: F) -> Self {
        Self { inner, transform }
    }

    /// Get a reference to the inner writer.
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the inner writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Consume the adapter and return the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write, F> Write for WriteAdapter<W, F>
where
    F: FnMut(&[u8]) -> io::Result<Vec<u8>>,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let transformed = (self.transform)(buf)?;
        self.inner.write_all(&transformed)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple mock compressor for testing
    struct MockCompressor;

    impl Compressor for MockCompressor {
        fn algorithm(&self) -> haagenti_core::Algorithm {
            haagenti_core::Algorithm::Lz4
        }

        fn level(&self) -> haagenti_core::CompressionLevel {
            haagenti_core::CompressionLevel::Default
        }

        fn compress(&self, input: &[u8]) -> haagenti_core::Result<Vec<u8>> {
            // Simple "compression": prefix with length
            let mut result = Vec::with_capacity(4 + input.len());
            result.extend_from_slice(&(input.len() as u32).to_le_bytes());
            result.extend_from_slice(input);
            Ok(result)
        }

        fn compress_to(&self, input: &[u8], output: &mut [u8]) -> haagenti_core::Result<usize> {
            let compressed = self.compress(input)?;
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
            input_len + 4
        }
    }

    #[test]
    fn test_compress_writer() {
        let mut output = Vec::new();
        {
            let mut writer = CompressWriter::with_buffer_size(&mut output, MockCompressor, 16);
            writer.write_all(b"Hello").unwrap();
            writer.finish().unwrap();
        }

        // Verify output contains length prefix + data
        assert_eq!(output.len(), 4 + 5);
        let len = u32::from_le_bytes(output[..4].try_into().unwrap());
        assert_eq!(len, 5);
        assert_eq!(&output[4..], b"Hello");
    }

    #[test]
    fn test_compress_writer_multiple_flushes() {
        let mut output = Vec::new();
        {
            let mut writer = CompressWriter::with_buffer_size(&mut output, MockCompressor, 8);

            // This should cause multiple buffer flushes
            writer.write_all(b"Hello, World! This is a test.").unwrap();
            writer.finish().unwrap();
        }

        // Output should contain multiple compressed blocks
        assert!(output.len() > 4);
    }

    #[test]
    fn test_write_adapter() {
        let mut output = Vec::new();
        {
            let mut adapter = WriteAdapter::new(&mut output, |data: &[u8]| {
                // Transform: uppercase
                Ok(data.to_ascii_uppercase())
            });
            adapter.write_all(b"hello").unwrap();
        }

        assert_eq!(output, b"HELLO");
    }
}

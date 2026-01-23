//! Async I/O adapters for streaming compression.
//!
//! This module provides async versions of the compress/decompress adapters
//! using tokio's AsyncRead/AsyncWrite traits.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::ready;
use pin_project_lite::pin_project;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use haagenti_core::{Compressor, Decompressor};

use crate::{StreamBuffer, DEFAULT_BUFFER_SIZE};

pin_project! {
    /// An async writer that compresses data before writing to the inner writer.
    pub struct AsyncCompressWriter<W, C> {
        #[pin]
        inner: W,
        compressor: C,
        buffer: StreamBuffer,
        compressed_buffer: Vec<u8>,
        write_pos: usize,
        finished: bool,
    }
}

impl<W, C> AsyncCompressWriter<W, C>
where
    W: AsyncWrite,
    C: Compressor,
{
    /// Create a new async compressing writer with default buffer size.
    pub fn new(inner: W, compressor: C) -> Self {
        Self::with_buffer_size(inner, compressor, DEFAULT_BUFFER_SIZE)
    }

    /// Create a new async compressing writer with specified buffer size.
    pub fn with_buffer_size(inner: W, compressor: C, buffer_size: usize) -> Self {
        Self {
            inner,
            compressor,
            buffer: StreamBuffer::with_capacity(buffer_size),
            compressed_buffer: Vec::new(),
            write_pos: 0,
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
}

impl<W, C> AsyncWrite for AsyncCompressWriter<W, C>
where
    W: AsyncWrite + Unpin,
    C: Compressor + Unpin,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let mut this = self.project();

        if *this.finished {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Other,
                "writer already finished",
            )));
        }

        // First, flush any pending compressed data
        while *this.write_pos < this.compressed_buffer.len() {
            let n = ready!(this
                .inner
                .as_mut()
                .poll_write(cx, &this.compressed_buffer[*this.write_pos..]))?;
            *this.write_pos += n;
        }

        // Clear compressed buffer after fully written
        if *this.write_pos >= this.compressed_buffer.len() {
            this.compressed_buffer.clear();
            *this.write_pos = 0;
        }

        // Write to input buffer
        let written = this.buffer.write(buf);

        // If buffer is full, compress it
        if this.buffer.is_full() {
            let data = this.buffer.readable();
            let compressed = this
                .compressor
                .compress(data)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            *this.compressed_buffer = compressed;
            this.buffer.clear();
        }

        Poll::Ready(Ok(written))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut this = self.project();

        // Compress any remaining buffered data
        if this.buffer.available() > 0 && this.compressed_buffer.is_empty() {
            let data = this.buffer.readable();
            let compressed = this
                .compressor
                .compress(data)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            *this.compressed_buffer = compressed;
            this.buffer.clear();
        }

        // Flush compressed data to inner writer
        while *this.write_pos < this.compressed_buffer.len() {
            let n = ready!(this
                .inner
                .as_mut()
                .poll_write(cx, &this.compressed_buffer[*this.write_pos..]))?;
            *this.write_pos += n;
        }

        // Clear after fully written
        this.compressed_buffer.clear();
        *this.write_pos = 0;

        this.inner.poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Flush first
        ready!(self.as_mut().poll_flush(cx))?;

        let this = self.project();
        *this.finished = true;
        this.inner.poll_shutdown(cx)
    }
}

pin_project! {
    /// An async reader that decompresses data from the inner reader.
    pub struct AsyncDecompressReader<R, D> {
        #[pin]
        inner: R,
        decompressor: D,
        input_buffer: Vec<u8>,
        output_buffer: Vec<u8>,
        output_pos: usize,
        finished: bool,
    }
}

impl<R, D> AsyncDecompressReader<R, D>
where
    R: AsyncRead,
    D: Decompressor,
{
    /// Create a new async decompressing reader with default buffer size.
    pub fn new(inner: R, decompressor: D) -> Self {
        Self::with_buffer_size(inner, decompressor, DEFAULT_BUFFER_SIZE)
    }

    /// Create a new async decompressing reader with specified buffer size.
    pub fn with_buffer_size(inner: R, decompressor: D, buffer_size: usize) -> Self {
        Self {
            inner,
            decompressor,
            input_buffer: Vec::with_capacity(buffer_size),
            output_buffer: Vec::new(),
            output_pos: 0,
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
}

impl<R, D> AsyncRead for AsyncDecompressReader<R, D>
where
    R: AsyncRead + Unpin,
    D: Decompressor + Unpin,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let mut this = self.project();

        // First, return any buffered decompressed data
        if *this.output_pos < this.output_buffer.len() {
            let available = &this.output_buffer[*this.output_pos..];
            let to_copy = available.len().min(buf.remaining());
            buf.put_slice(&available[..to_copy]);
            *this.output_pos += to_copy;

            // Clear buffer if fully consumed
            if *this.output_pos >= this.output_buffer.len() {
                this.output_buffer.clear();
                *this.output_pos = 0;
            }

            return Poll::Ready(Ok(()));
        }

        // If finished, we're done
        if *this.finished {
            return Poll::Ready(Ok(()));
        }

        // Read more input
        let mut temp_buf = [0u8; 4096];
        let mut read_buf = ReadBuf::new(&mut temp_buf);

        match this.inner.as_mut().poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => {
                let filled = read_buf.filled();
                if filled.is_empty() {
                    // EOF - decompress remaining input if any
                    if !this.input_buffer.is_empty() {
                        let decompressed = this
                            .decompressor
                            .decompress(&this.input_buffer)
                            .map_err(|e| {
                                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
                            })?;

                        *this.output_buffer = decompressed;
                        *this.output_pos = 0;
                        this.input_buffer.clear();

                        // Return data if available
                        if !this.output_buffer.is_empty() {
                            let to_copy = this.output_buffer.len().min(buf.remaining());
                            buf.put_slice(&this.output_buffer[..to_copy]);
                            *this.output_pos = to_copy;
                        }
                    }
                    *this.finished = true;
                    Poll::Ready(Ok(()))
                } else {
                    // Add to input buffer
                    this.input_buffer.extend_from_slice(filled);

                    // Try to decompress (for streaming formats, we'd decompress incrementally)
                    // For block formats, we need all the input first
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Mock compressor/decompressor for testing
    struct MockCompressor;
    struct MockDecompressor;

    impl Compressor for MockCompressor {
        fn algorithm(&self) -> haagenti_core::Algorithm {
            haagenti_core::Algorithm::Lz4
        }

        fn level(&self) -> haagenti_core::CompressionLevel {
            haagenti_core::CompressionLevel::Default
        }

        fn compress(&self, input: &[u8]) -> haagenti_core::Result<Vec<u8>> {
            // Simple mock: prefix with length
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

    impl Decompressor for MockDecompressor {
        fn algorithm(&self) -> haagenti_core::Algorithm {
            haagenti_core::Algorithm::Lz4
        }

        fn decompress(&self, input: &[u8]) -> haagenti_core::Result<Vec<u8>> {
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

    #[tokio::test]
    async fn test_async_compress_writer() {
        let mut output = Vec::new();

        {
            let mut writer = AsyncCompressWriter::with_buffer_size(&mut output, MockCompressor, 16);
            writer.write_all(b"Hello").await.unwrap();
            writer.shutdown().await.unwrap();
        }

        // Verify output contains length prefix + data
        assert_eq!(output.len(), 4 + 5);
        let len = u32::from_le_bytes(output[..4].try_into().unwrap());
        assert_eq!(len, 5);
        assert_eq!(&output[4..], b"Hello");
    }

    #[tokio::test]
    async fn test_async_decompress_reader() {
        // Create mock compressed data
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&5u32.to_le_bytes());
        compressed.extend_from_slice(b"Hello");

        let cursor = std::io::Cursor::new(compressed);
        let mut reader = AsyncDecompressReader::with_buffer_size(
            tokio::io::BufReader::new(cursor),
            MockDecompressor,
            64,
        );

        let mut output = Vec::new();
        reader.read_to_end(&mut output).await.unwrap();

        assert_eq!(output, b"Hello");
    }
}

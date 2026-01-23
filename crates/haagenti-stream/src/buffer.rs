//! Buffer management for streaming operations.

use crate::DEFAULT_BUFFER_SIZE;

/// A reusable buffer for streaming operations.
///
/// Manages input and output buffers with efficient memory reuse.
#[derive(Debug)]
pub struct StreamBuffer {
    /// Internal buffer storage.
    data: Vec<u8>,
    /// Current read position.
    read_pos: usize,
    /// Current write position (end of valid data).
    write_pos: usize,
}

impl StreamBuffer {
    /// Create a new buffer with default size.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUFFER_SIZE)
    }

    /// Create a new buffer with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity],
            read_pos: 0,
            write_pos: 0,
        }
    }

    /// Get the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Get the number of bytes available to read.
    #[inline]
    pub fn available(&self) -> usize {
        self.write_pos - self.read_pos
    }

    /// Get the number of bytes that can be written.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len() - self.write_pos
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.read_pos >= self.write_pos
    }

    /// Check if buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.write_pos >= self.data.len()
    }

    /// Get a slice of readable data.
    #[inline]
    pub fn readable(&self) -> &[u8] {
        &self.data[self.read_pos..self.write_pos]
    }

    /// Get a mutable slice for writing.
    #[inline]
    pub fn writable(&mut self) -> &mut [u8] {
        &mut self.data[self.write_pos..]
    }

    /// Consume `n` bytes from the read position.
    #[inline]
    pub fn consume(&mut self, n: usize) {
        self.read_pos = (self.read_pos + n).min(self.write_pos);
    }

    /// Advance the write position by `n` bytes.
    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.write_pos = (self.write_pos + n).min(self.data.len());
    }

    /// Reset buffer to empty state.
    #[inline]
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
    }

    /// Compact the buffer by moving unread data to the beginning.
    pub fn compact(&mut self) {
        if self.read_pos > 0 {
            let available = self.available();
            if available > 0 {
                self.data.copy_within(self.read_pos..self.write_pos, 0);
            }
            self.read_pos = 0;
            self.write_pos = available;
        }
    }

    /// Write data into the buffer.
    ///
    /// Returns number of bytes written.
    pub fn write(&mut self, data: &[u8]) -> usize {
        let space = self.remaining();
        let to_write = data.len().min(space);

        if to_write > 0 {
            self.data[self.write_pos..self.write_pos + to_write].copy_from_slice(&data[..to_write]);
            self.write_pos += to_write;
        }

        to_write
    }

    /// Read data from the buffer.
    ///
    /// Returns number of bytes read.
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.available();
        let to_read = buf.len().min(available);

        if to_read > 0 {
            buf[..to_read].copy_from_slice(&self.data[self.read_pos..self.read_pos + to_read]);
            self.read_pos += to_read;
        }

        to_read
    }

    /// Get the underlying buffer for direct access.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.write_pos]
    }

    /// Get mutable access to the entire buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = StreamBuffer::new();
        assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
        assert!(buf.is_empty());
        assert!(!buf.is_full());
    }

    #[test]
    fn test_write_and_read() {
        let mut buf = StreamBuffer::with_capacity(64);

        let written = buf.write(b"Hello, World!");
        assert_eq!(written, 13);
        assert_eq!(buf.available(), 13);

        let mut out = [0u8; 64];
        let read = buf.read(&mut out);
        assert_eq!(read, 13);
        assert_eq!(&out[..13], b"Hello, World!");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_compact() {
        let mut buf = StreamBuffer::with_capacity(32);

        buf.write(b"Hello, World!");
        buf.consume(7); // consume "Hello, "

        assert_eq!(buf.readable(), b"World!");
        assert_eq!(buf.read_pos, 7);

        buf.compact();
        assert_eq!(buf.read_pos, 0);
        assert_eq!(buf.write_pos, 6);
        assert_eq!(buf.readable(), b"World!");
    }

    #[test]
    fn test_full_buffer() {
        let mut buf = StreamBuffer::with_capacity(8);

        let written = buf.write(b"12345678");
        assert_eq!(written, 8);
        assert!(buf.is_full());

        // Try to write more
        let written = buf.write(b"more");
        assert_eq!(written, 0);
    }

    #[test]
    fn test_clear() {
        let mut buf = StreamBuffer::with_capacity(32);
        buf.write(b"Hello");
        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.available(), 0);
    }
}

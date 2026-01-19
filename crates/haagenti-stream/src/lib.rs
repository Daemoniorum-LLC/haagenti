//! # Haagenti Stream
//!
//! Advanced streaming compression utilities.
//!
//! Provides buffered streams, adapters, and I/O integration for
//! efficient streaming compression operations.
//!
//! ## Features
//!
//! - **Buffered Streams**: Memory-efficient buffering
//! - **I/O Integration**: Read/Write trait implementations
//! - **Backpressure**: Flow control for slow consumers
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_stream::CompressWriter;
//! use haagenti_lz4::Lz4Compressor;
//! use std::io::Write;
//!
//! let file = File::create("output.lz4")?;
//! let mut writer = CompressWriter::new(file, Lz4Compressor::new());
//!
//! writer.write_all(b"Hello, compression!")?;
//! writer.finish()?;
//! ```

mod buffer;
mod reader;
mod writer;

#[cfg(feature = "async")]
mod async_io;

pub use buffer::StreamBuffer;
pub use reader::{DecompressReader, ReadAdapter};
pub use writer::{CompressWriter, WriteAdapter};

#[cfg(feature = "async")]
pub use async_io::{AsyncCompressWriter, AsyncDecompressReader};

/// Default buffer size for streaming operations (64 KB).
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Minimum buffer size allowed.
pub const MIN_BUFFER_SIZE: usize = 4 * 1024;

/// Maximum buffer size allowed (1 MB).
pub const MAX_BUFFER_SIZE: usize = 1024 * 1024;

/// Clamp buffer size to valid range.
#[inline]
pub fn clamp_buffer_size(size: usize) -> usize {
    size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_buffer_size() {
        assert_eq!(clamp_buffer_size(1024), MIN_BUFFER_SIZE);
        assert_eq!(clamp_buffer_size(DEFAULT_BUFFER_SIZE), DEFAULT_BUFFER_SIZE);
        assert_eq!(clamp_buffer_size(10 * 1024 * 1024), MAX_BUFFER_SIZE);
    }
}

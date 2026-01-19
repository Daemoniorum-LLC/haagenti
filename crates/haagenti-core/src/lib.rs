//! # Haagenti Core
//!
//! Core traits, types, and streaming API for the Haagenti compression library.
//!
//! Haagenti is named after the 48th demon of the Ars Goetia, who transmutes
//! substances into more valuable forms - just as compression transforms data
//! into denser representations.
//!
//! ## Design Philosophy
//!
//! - **Zero-copy where possible**: Minimize allocations and memory copies
//! - **Streaming-first**: All operations support incremental processing
//! - **SIMD-ready**: Types designed for vectorized operations
//! - **No-std compatible**: Core traits work without standard library
//!
//! ## Core Traits
//!
//! - [`Compressor`] - One-shot compression operations
//! - [`Decompressor`] - One-shot decompression operations
//! - [`Codec`] - Combined compress/decompress capability
//! - [`StreamingCompressor`] - Incremental compression
//! - [`StreamingDecompressor`] - Incremental decompression
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_core::{Codec, CompressionLevel};
//! use haagenti_lz4::Lz4Codec;
//!
//! let codec = Lz4Codec::with_level(CompressionLevel::Fast);
//! let compressed = codec.compress(data)?;
//! let original = codec.decompress(&compressed)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod types;
pub mod traits;
pub mod stream;
pub mod stats;

#[cfg(feature = "dct")]
pub mod dct;

pub use error::{Error, Result};
pub use types::{Algorithm, CompressionLevel, CompressionRatio, Checksum, WindowSize};
pub use traits::{
    Compressor, Decompressor, Codec, StreamingCompressor, StreamingDecompressor,
    DictionaryCompressor, DictionaryDecompressor, ParallelCompressor, SimdCompressor,
};
pub use stream::{Flush, StreamConfig, StreamState};
pub use stats::{CompressionStats, Metrics};

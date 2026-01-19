//! # Haagenti Deflate
//!
//! Deflate, Zlib, and Gzip compression implementations.
//!
//! These formats are widely used for compatibility with existing systems
//! (ZIP files, HTTP compression, PNG, git objects).
//!
//! ## Formats
//!
//! - **Deflate**: Raw DEFLATE (RFC 1951)
//! - **Zlib**: DEFLATE with Zlib wrapper (RFC 1950)
//! - **Gzip**: DEFLATE with Gzip wrapper (RFC 1952)
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_deflate::{DeflateCodec, ZlibCodec, GzipCodec};
//! use haagenti_core::Codec;
//!
//! // Raw deflate
//! let deflate = DeflateCodec::new();
//!
//! // Zlib (for PNG, git objects)
//! let zlib = ZlibCodec::new();
//!
//! // Gzip (for HTTP, files)
//! let gzip = GzipCodec::new();
//! ```

pub mod codec;
pub mod deflate;
pub mod gzip;
pub mod huffman;
pub mod inflate;
pub mod zlib;

// Re-export main types
pub use codec::{
    DeflateCodec, DeflateCompressor, DeflateDecompressor, GzipCodec, GzipCompressor,
    GzipDecompressor, ZlibCodec, ZlibCompressor, ZlibDecompressor,
};

// Re-export raw functions for advanced use
pub use deflate::deflate;
pub use gzip::{gzip_compress, gzip_decompress};
pub use inflate::inflate;
pub use zlib::{zlib_compress, zlib_decompress};

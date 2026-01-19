//! # Haagenti LZ4
//!
//! LZ4 and LZ4-HC compression implementation.
//!
//! LZ4 is an extremely fast compression algorithm, prioritizing speed over
//! compression ratio. It's ideal for real-time applications, databases,
//! and caching.
//!
//! ## Features
//!
//! - **LZ4**: Ultra-fast compression (~800 MB/s)
//! - **LZ4-HC**: High compression mode (~100 MB/s, better ratio)
//! - **Streaming**: Incremental compression/decompression
//! - **SIMD**: AVX2/AVX-512/NEON acceleration (optional)
//!
//! ## Example
//!
//! ```ignore
//! use haagenti_lz4::Lz4Codec;
//! use haagenti_core::{Codec, Compressor, Decompressor};
//!
//! let codec = Lz4Codec::new();
//! let (compressed, size) = codec.compress_with_size(data)?;
//! let original = codec.decompress_sized(&compressed, size)?;
//! ```

pub mod block;
pub mod codec;
pub mod compress;
pub mod decompress;

// Re-export main types
pub use codec::Lz4Codec;
pub use compress::{Lz4Compressor, Lz4StreamingCompressor};
pub use decompress::{Lz4Decompressor, Lz4StreamingDecompressor};

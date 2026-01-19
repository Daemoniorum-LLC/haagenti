//! Core type definitions for compression operations.

/// Compression level presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionLevel {
    /// No compression, just framing (fastest).
    None,

    /// Optimized for speed over ratio (level 1-3).
    Fast,

    /// Balanced speed and ratio (level 4-6, default).
    #[default]
    Default,

    /// Optimized for ratio over speed (level 7-9).
    Best,

    /// Maximum compression, slowest (level 10+).
    Ultra,

    /// Custom level (algorithm-specific range).
    Custom(i32),
}

impl CompressionLevel {
    /// Convert to numeric level for algorithms.
    pub fn to_level(self) -> i32 {
        match self {
            CompressionLevel::None => 0,
            CompressionLevel::Fast => 1,
            CompressionLevel::Default => 6,
            CompressionLevel::Best => 9,
            CompressionLevel::Ultra => 12,
            CompressionLevel::Custom(level) => level,
        }
    }

    /// Create from numeric level.
    pub fn from_level(level: i32) -> Self {
        match level {
            0 => CompressionLevel::None,
            1..=3 => CompressionLevel::Fast,
            4..=6 => CompressionLevel::Default,
            7..=9 => CompressionLevel::Best,
            10.. => CompressionLevel::Ultra,
            _ => CompressionLevel::Custom(level),
        }
    }
}

/// Supported compression algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// LZ4 - Ultra-fast compression.
    Lz4,
    /// LZ4-HC - High-compression variant of LZ4.
    Lz4Hc,
    /// Zstandard - Balanced speed and ratio.
    Zstd,
    /// Brotli - High compression ratio.
    Brotli,
    /// Deflate - Widely compatible (RFC 1951).
    Deflate,
    /// Gzip - Deflate with headers/checksums (RFC 1952).
    Gzip,
    /// Zlib - Deflate with different framing (RFC 1950).
    Zlib,
    /// Snappy - Google's fast codec.
    Snappy,
    /// LZMA/XZ - Maximum compression.
    Lzma,
}

impl Algorithm {
    /// Get algorithm name as string.
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Lz4 => "lz4",
            Algorithm::Lz4Hc => "lz4hc",
            Algorithm::Zstd => "zstd",
            Algorithm::Brotli => "brotli",
            Algorithm::Deflate => "deflate",
            Algorithm::Gzip => "gzip",
            Algorithm::Zlib => "zlib",
            Algorithm::Snappy => "snappy",
            Algorithm::Lzma => "lzma",
        }
    }

    /// Get typical compression ratio for text data.
    pub fn typical_ratio(self) -> f32 {
        match self {
            Algorithm::Lz4 => 2.1,
            Algorithm::Lz4Hc => 2.5,
            Algorithm::Zstd => 3.0,
            Algorithm::Brotli => 3.5,
            Algorithm::Deflate => 2.8,
            Algorithm::Gzip => 2.8,
            Algorithm::Zlib => 2.8,
            Algorithm::Snappy => 1.8,
            Algorithm::Lzma => 4.0,
        }
    }

    /// Get typical compression speed (MB/s on modern CPU).
    pub fn typical_speed_mbs(self) -> u32 {
        match self {
            Algorithm::Lz4 => 800,
            Algorithm::Lz4Hc => 100,
            Algorithm::Zstd => 500,
            Algorithm::Brotli => 50,
            Algorithm::Deflate => 100,
            Algorithm::Gzip => 100,
            Algorithm::Zlib => 100,
            Algorithm::Snappy => 500,
            Algorithm::Lzma => 20,
        }
    }

    /// Check if algorithm supports dictionaries.
    pub fn supports_dictionary(self) -> bool {
        matches!(
            self,
            Algorithm::Lz4
                | Algorithm::Lz4Hc
                | Algorithm::Zstd
                | Algorithm::Brotli
                | Algorithm::Deflate
                | Algorithm::Zlib
                | Algorithm::Lzma
        )
    }
}

/// Compression ratio metrics.
#[derive(Debug, Clone, Copy)]
pub struct CompressionRatio {
    /// Original uncompressed size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
}

impl CompressionRatio {
    /// Create new ratio from sizes.
    pub fn new(original: usize, compressed: usize) -> Self {
        CompressionRatio {
            original_size: original,
            compressed_size: compressed,
        }
    }

    /// Calculate ratio (original / compressed).
    /// Higher is better (more compression).
    pub fn ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            return 0.0;
        }
        self.original_size as f64 / self.compressed_size as f64
    }

    /// Calculate space savings as percentage (0-100).
    pub fn savings_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - (self.compressed_size as f64 / self.original_size as f64)) * 100.0
    }

    /// Calculate bytes saved.
    pub fn bytes_saved(&self) -> isize {
        self.original_size as isize - self.compressed_size as isize
    }

    /// Check if compression was effective (saved space).
    pub fn is_effective(&self) -> bool {
        self.compressed_size < self.original_size
    }
}

/// Window size for LZ-family algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowSize {
    /// 1 KB window.
    W1K,
    /// 4 KB window.
    W4K,
    /// 16 KB window.
    W16K,
    /// 64 KB window (LZ4 default).
    #[default]
    W64K,
    /// 256 KB window.
    W256K,
    /// 1 MB window.
    W1M,
    /// 4 MB window.
    W4M,
    /// 8 MB window (Zstd max).
    W8M,
    /// Custom size in bytes.
    Custom(usize),
}

impl WindowSize {
    /// Convert to bytes.
    pub fn to_bytes(self) -> usize {
        match self {
            WindowSize::W1K => 1024,
            WindowSize::W4K => 4096,
            WindowSize::W16K => 16384,
            WindowSize::W64K => 65536,
            WindowSize::W256K => 262144,
            WindowSize::W1M => 1048576,
            WindowSize::W4M => 4194304,
            WindowSize::W8M => 8388608,
            WindowSize::Custom(size) => size,
        }
    }
}

/// Checksum algorithm for data integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Checksum {
    /// No checksum (fastest, no integrity check).
    None,
    /// CRC32 (fast, good for detecting errors).
    Crc32,
    /// CRC32C (hardware-accelerated on modern CPUs).
    Crc32c,
    /// Adler-32 (faster than CRC32, used by Zlib).
    Adler32,
    /// xxHash32 (very fast, good distribution).
    #[default]
    XxHash32,
    /// xxHash64 (fast, better for large data).
    XxHash64,
    /// xxHash3 (newest, fastest).
    XxHash3,
}

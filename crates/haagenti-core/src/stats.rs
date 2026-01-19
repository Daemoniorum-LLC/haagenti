//! Statistics and metrics for compression operations.

use crate::types::{Algorithm, CompressionRatio};

/// Statistics from a compression/decompression operation.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Algorithm used.
    pub algorithm: Option<Algorithm>,

    /// Original (uncompressed) size in bytes.
    pub original_size: usize,

    /// Compressed size in bytes.
    pub compressed_size: usize,

    /// Time taken in microseconds.
    pub time_us: u64,

    /// Number of blocks processed.
    pub blocks_processed: usize,

    /// Peak memory usage in bytes.
    pub peak_memory: Option<usize>,

    /// Whether dictionary was used.
    pub dictionary_used: bool,

    /// Whether SIMD was used.
    pub simd_used: bool,

    /// Checksum (if computed).
    pub checksum: Option<u64>,
}

impl CompressionStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create stats from a completed operation.
    pub fn from_operation(
        algorithm: Algorithm,
        original_size: usize,
        compressed_size: usize,
        time_us: u64,
    ) -> Self {
        CompressionStats {
            algorithm: Some(algorithm),
            original_size,
            compressed_size,
            time_us,
            blocks_processed: 1,
            ..Default::default()
        }
    }

    /// Get compression ratio.
    pub fn ratio(&self) -> CompressionRatio {
        CompressionRatio::new(self.original_size, self.compressed_size)
    }

    /// Get throughput in bytes per second.
    pub fn throughput_bps(&self) -> f64 {
        if self.time_us == 0 {
            return 0.0;
        }
        self.original_size as f64 * 1_000_000.0 / self.time_us as f64
    }

    /// Get throughput in MB/s.
    pub fn throughput_mbs(&self) -> f64 {
        self.throughput_bps() / 1_000_000.0
    }

    /// Get space savings as percentage.
    pub fn savings_percent(&self) -> f64 {
        self.ratio().savings_percent()
    }

    /// Merge stats from multiple operations.
    pub fn merge(&mut self, other: &CompressionStats) {
        self.original_size += other.original_size;
        self.compressed_size += other.compressed_size;
        self.time_us += other.time_us;
        self.blocks_processed += other.blocks_processed;

        // Peak memory is max of both
        match (&self.peak_memory, &other.peak_memory) {
            (Some(a), Some(b)) => self.peak_memory = Some((*a).max(*b)),
            (None, Some(b)) => self.peak_memory = Some(*b),
            _ => {}
        }
    }
}

/// Metrics collector for aggregate statistics.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Total operations performed.
    pub total_operations: u64,

    /// Total bytes compressed.
    pub total_bytes_in: u64,

    /// Total bytes produced.
    pub total_bytes_out: u64,

    /// Total time spent in microseconds.
    pub total_time_us: u64,

    /// Number of errors encountered.
    pub error_count: u64,
}

impl Metrics {
    /// Create new metrics collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed operation.
    pub fn record(&mut self, stats: &CompressionStats) {
        self.total_operations += 1;
        self.total_bytes_in += stats.original_size as u64;
        self.total_bytes_out += stats.compressed_size as u64;
        self.total_time_us += stats.time_us;
    }

    /// Record an error.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Get average compression ratio.
    pub fn average_ratio(&self) -> f64 {
        if self.total_bytes_out == 0 {
            return 1.0;
        }
        self.total_bytes_in as f64 / self.total_bytes_out as f64
    }

    /// Get average throughput in MB/s.
    pub fn average_throughput_mbs(&self) -> f64 {
        if self.total_time_us == 0 {
            return 0.0;
        }
        self.total_bytes_in as f64 / self.total_time_us as f64
    }

    /// Get error rate (0.0 to 1.0).
    pub fn error_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        self.error_count as f64 / self.total_operations as f64
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get metrics summary as string.
    pub fn summary(&self) -> String {
        format!(
            "Operations: {}, Bytes: {} -> {} (ratio: {:.2}x), Throughput: {:.1} MB/s, Errors: {}",
            self.total_operations,
            self.total_bytes_in,
            self.total_bytes_out,
            self.average_ratio(),
            self.average_throughput_mbs(),
            self.error_count,
        )
    }
}

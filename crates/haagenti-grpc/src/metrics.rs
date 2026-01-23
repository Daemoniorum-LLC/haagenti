//! Prometheus metrics for the compression service.

use metrics::{counter, gauge, histogram};

/// Record a compression operation
pub fn record_compression(original_size: u64, compressed_size: u64, duration_ns: u64) {
    counter!("haagenti_compression_total").increment(1);
    counter!("haagenti_bytes_compressed_total").increment(original_size);
    counter!("haagenti_bytes_output_total").increment(compressed_size);
    histogram!("haagenti_compression_duration_seconds")
        .record(duration_ns as f64 / 1_000_000_000.0);

    if original_size > 0 {
        let ratio = original_size as f64 / compressed_size as f64;
        gauge!("haagenti_compression_ratio").set(ratio);
    }
}

/// Record a decompression operation
pub fn record_decompression(_compressed_size: u64, original_size: u64, duration_ns: u64) {
    counter!("haagenti_decompression_total").increment(1);
    counter!("haagenti_bytes_decompressed_total").increment(original_size);
    histogram!("haagenti_decompression_duration_seconds")
        .record(duration_ns as f64 / 1_000_000_000.0);
}

/// Record an error
pub fn record_error(operation: &str) {
    counter!("haagenti_errors_total", "operation" => operation.to_string()).increment(1);
}

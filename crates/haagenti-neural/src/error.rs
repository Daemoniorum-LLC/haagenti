//! Error types for neural compression

use thiserror::Error;

/// Errors that can occur during neural compression
#[derive(Debug, Error)]
pub enum NeuralError {
    /// Invalid codebook configuration
    #[error("Invalid codebook: {0}")]
    InvalidCodebook(String),

    /// Codebook not found
    #[error("Codebook not found for layer type: {0}")]
    CodebookNotFound(String),

    /// Encoding failed
    #[error("Encoding failed: {0}")]
    EncodingError(String),

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingError(String),

    /// Invalid file format
    #[error("Invalid NCT file: {0}")]
    InvalidFormat(String),

    /// File I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Training error
    #[error("Training failed: {0}")]
    TrainingError(String),

    /// Quality threshold not met
    #[error("Quality {actual:.4} below threshold {threshold:.4}")]
    QualityThreshold { actual: f32, threshold: f32 },

    /// Compression ratio not achieved
    #[error("Compression ratio {actual:.1}x below target {target:.1}x")]
    CompressionRatio { actual: f32, target: f32 },
}

/// Result type for neural compression operations
pub type Result<T> = std::result::Result<T, NeuralError>;

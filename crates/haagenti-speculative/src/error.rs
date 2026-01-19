//! Error types for speculative loading

use thiserror::Error;

/// Result type for speculative operations
pub type Result<T> = std::result::Result<T, SpeculativeError>;

/// Errors that can occur during speculative loading
#[derive(Error, Debug)]
pub enum SpeculativeError {
    /// Intent prediction failed
    #[error("Intent prediction failed: {0}")]
    PredictionFailed(String),

    /// Buffer is full
    #[error("Speculation buffer is full")]
    BufferFull,

    /// Fragment not found
    #[error("Fragment not found: {0}")]
    FragmentNotFound(String),

    /// Session history corrupted
    #[error("Session history corrupted: {0}")]
    HistoryCorrupted(String),

    /// Cancellation failed
    #[error("Failed to cancel speculation: {0}")]
    CancellationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),
}

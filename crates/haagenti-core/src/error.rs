//! Error types for compression operations.

use thiserror::Error;

/// Result type alias for compression operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Compression error types.
#[derive(Debug, Error)]
pub enum Error {
    /// Input data is corrupted or invalid.
    #[error("corrupted data: {message}")]
    CorruptedData {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Buffer too small for output.
    #[error("buffer too small: need {required} bytes, got {provided}")]
    BufferTooSmall { required: usize, provided: usize },

    /// Invalid compression level specified.
    #[error("invalid compression level {level}: must be in range [{min}, {max}]")]
    InvalidLevel { level: i32, min: i32, max: i32 },

    /// Dictionary not found or invalid.
    #[error("invalid dictionary: {0}")]
    InvalidDictionary(String),

    /// Checksum verification failed.
    #[error("checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Unexpected end of input stream.
    #[error("unexpected EOF after {bytes_read} bytes")]
    UnexpectedEof { bytes_read: usize },

    /// I/O error from underlying stream.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Memory allocation failed.
    #[error("allocation failed: could not allocate {requested_bytes} bytes")]
    AllocationFailed { requested_bytes: usize },

    /// Algorithm-specific error.
    #[error("{algorithm} error: {message}")]
    Algorithm {
        algorithm: &'static str,
        message: String,
    },

    /// Stream state error.
    #[error("invalid state: expected {expected}, got {actual}")]
    InvalidState {
        expected: &'static str,
        actual: &'static str,
    },

    /// Unsupported feature or format.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl Error {
    /// Create a corrupted data error.
    pub fn corrupted(message: impl Into<String>) -> Self {
        Error::CorruptedData {
            message: message.into(),
            source: None,
        }
    }

    /// Create a corrupted data error with offset context.
    pub fn corrupted_at(message: impl Into<String>, offset: usize) -> Self {
        Error::CorruptedData {
            message: format!("{} at offset {}", message.into(), offset),
            source: None,
        }
    }

    /// Create a buffer too small error.
    pub fn buffer_too_small(required: usize, provided: usize) -> Self {
        Error::BufferTooSmall { required, provided }
    }

    /// Create a checksum mismatch error.
    pub fn checksum_mismatch(expected: u32, actual: u32) -> Self {
        Error::ChecksumMismatch { expected, actual }
    }

    /// Create an unexpected EOF error.
    pub fn unexpected_eof(bytes_read: usize) -> Self {
        Error::UnexpectedEof { bytes_read }
    }

    /// Create an I/O error with a custom message.
    pub fn io(message: impl Into<String>) -> Self {
        Error::Io(std::io::Error::other(message.into()))
    }

    /// Create an algorithm-specific error.
    pub fn algorithm(algorithm: &'static str, message: impl Into<String>) -> Self {
        Error::Algorithm {
            algorithm,
            message: message.into(),
        }
    }

    /// Check if error is recoverable (can retry with different parameters).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::UnexpectedEof { .. } | Error::BufferTooSmall { .. }
        )
    }

    /// Get error category for metrics.
    pub fn category(&self) -> &'static str {
        match self {
            Error::CorruptedData { .. } => "corrupted_data",
            Error::BufferTooSmall { .. } => "buffer_too_small",
            Error::InvalidLevel { .. } => "invalid_level",
            Error::InvalidDictionary(_) => "invalid_dictionary",
            Error::ChecksumMismatch { .. } => "checksum_mismatch",
            Error::UnexpectedEof { .. } => "unexpected_eof",
            Error::Io(_) => "io_error",
            Error::AllocationFailed { .. } => "allocation_failed",
            Error::Algorithm { .. } => "algorithm_error",
            Error::InvalidState { .. } => "invalid_state",
            Error::Unsupported(_) => "unsupported",
        }
    }
}

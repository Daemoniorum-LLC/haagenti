//! Error types for network operations

use thiserror::Error;

/// Result type for network operations
pub type Result<T> = std::result::Result<T, NetworkError>;

/// Errors that can occur during network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    /// HTTP request failed
    #[error("HTTP error: {status} - {message}")]
    Http {
        status: u16,
        message: String,
    },

    /// Network connection failed
    #[error("Connection failed: {0}")]
    Connection(String),

    /// Request timeout
    #[error("Request timeout after {0}ms")]
    Timeout(u64),

    /// Invalid URL
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// Fragment not found on CDN
    #[error("Fragment not found: {0}")]
    NotFound(String),

    /// Checksum mismatch
    #[error("Checksum mismatch for {fragment_id}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        fragment_id: String,
        expected: String,
        actual: String,
    },

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// All retries exhausted
    #[error("All retries exhausted: {0}")]
    RetriesExhausted(String),

    /// Rate limited
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        retry_after_ms: u64,
    },

    /// CDN configuration error
    #[error("CDN configuration error: {0}")]
    Configuration(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Cancelled
    #[error("Request cancelled")]
    Cancelled,
}

impl NetworkError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            NetworkError::Connection(_) => true,
            NetworkError::Timeout(_) => true,
            NetworkError::RateLimited { .. } => true,
            NetworkError::Http { status, .. } => *status >= 500,
            _ => false,
        }
    }

    /// Get retry delay if rate limited
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        if let NetworkError::RateLimited { retry_after_ms } = self {
            Some(std::time::Duration::from_millis(*retry_after_ms))
        } else {
            None
        }
    }
}

impl From<reqwest::Error> for NetworkError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            NetworkError::Timeout(30000)
        } else if e.is_connect() {
            NetworkError::Connection(e.to_string())
        } else if let Some(status) = e.status() {
            NetworkError::Http {
                status: status.as_u16(),
                message: e.to_string(),
            }
        } else {
            NetworkError::Connection(e.to_string())
        }
    }
}

impl From<url::ParseError> for NetworkError {
    fn from(e: url::ParseError) -> Self {
        NetworkError::InvalidUrl(e.to_string())
    }
}

//! Error types for serverless deployment

use thiserror::Error;

/// Serverless deployment errors
#[derive(Debug, Error)]
pub enum ServerlessError {
    /// Cold start timeout
    #[error("Cold start timeout: {0}ms exceeded")]
    ColdStartTimeout(u64),

    /// Fragment pool error
    #[error("Fragment pool error: {0}")]
    PoolError(String),

    /// Snapshot error
    #[error("Snapshot error: {0}")]
    SnapshotError(String),

    /// State serialization error
    #[error("State serialization error: {0}")]
    SerializationError(String),

    /// State deserialization error
    #[error("State deserialization error: {0}")]
    DeserializationError(String),

    /// Provider error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Memory limit exceeded
    #[error("Memory limit exceeded: {used_mb}MB / {limit_mb}MB")]
    MemoryLimitExceeded { used_mb: u64, limit_mb: u64 },

    /// GPU not available
    #[error("GPU not available in this environment")]
    GpuNotAvailable,

    /// Warmup failed
    #[error("Warmup failed: {0}")]
    WarmupFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for serverless operations
pub type Result<T> = std::result::Result<T, ServerlessError>;

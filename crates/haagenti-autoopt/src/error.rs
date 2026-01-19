//! Error types for auto-optimization

use thiserror::Error;

/// Optimization errors
#[derive(Debug, Error)]
pub enum OptError {
    /// Invalid parameter range
    #[error("Invalid parameter range: {0}")]
    InvalidRange(String),

    /// Optimization failed
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    /// Profiling error
    #[error("Profiling error: {0}")]
    ProfilingError(String),

    /// Hardware detection error
    #[error("Hardware detection error: {0}")]
    HardwareError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout
    #[error("Optimization timed out after {timeout_secs}s")]
    Timeout { timeout_secs: u64 },

    /// No improvement
    #[error("No improvement after {trials} trials")]
    NoImprovement { trials: usize },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for optimization operations
pub type Result<T> = std::result::Result<T, OptError>;

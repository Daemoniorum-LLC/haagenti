//! Error types for mobile deployment

use thiserror::Error;

/// Mobile deployment errors
#[derive(Debug, Error)]
pub enum MobileError {
    /// Platform not supported
    #[error("Platform not supported: {0}")]
    UnsupportedPlatform(String),

    /// CoreML error
    #[error("CoreML error: {0}")]
    CoreMLError(String),

    /// NNAPI error
    #[error("NNAPI error: {0}")]
    NnapiError(String),

    /// Model loading error
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Quantization error
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Thermal throttling
    #[error("Thermal throttling active: {state:?}, temperature: {temp_celsius}°C")]
    ThermalThrottling { state: String, temp_celsius: f32 },

    /// Out of memory
    #[error("Out of memory: requested {requested_mb}MB, available {available_mb}MB")]
    OutOfMemory {
        requested_mb: u64,
        available_mb: u64,
    },

    /// Battery too low
    #[error("Battery level too low: {level}%")]
    BatteryLow { level: u8 },

    /// Timeout
    #[error("Operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for mobile operations
pub type Result<T> = std::result::Result<T, MobileError>;

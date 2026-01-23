//! Error types for WebGPU operations

use thiserror::Error;

/// Errors that can occur during WebGPU operations
#[derive(Debug, Error)]
pub enum WebGpuError {
    /// WebGPU not available in browser
    #[error("WebGPU not available: {0}")]
    NotAvailable(String),

    /// Failed to get GPU adapter
    #[error("Failed to get GPU adapter: {0}")]
    AdapterError(String),

    /// Failed to get GPU device
    #[error("Failed to get GPU device: {0}")]
    DeviceError(String),

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),

    /// Pipeline creation failed
    #[error("Pipeline creation failed: {0}")]
    PipelineError(String),

    /// Buffer operation failed
    #[error("Buffer error: {0}")]
    BufferError(String),

    /// Out of GPU memory
    #[error("Out of GPU memory: requested {requested_mb}MB, available {available_mb}MB")]
    OutOfMemory {
        requested_mb: u64,
        available_mb: u64,
    },

    /// Cache operation failed
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Network fetch failed
    #[error("Fetch failed: {0}")]
    FetchError(String),

    /// Timeout
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Feature not supported
    #[error("Feature not supported: {0}")]
    Unsupported(String),
}

/// Result type for WebGPU operations
pub type Result<T> = std::result::Result<T, WebGpuError>;

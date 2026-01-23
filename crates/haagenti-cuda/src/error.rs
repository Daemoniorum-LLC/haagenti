//! Error types for CUDA GPU decompression.

use thiserror::Error;

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;

/// Errors that can occur during GPU decompression.
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA driver error.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// Memory allocation failed.
    #[error("GPU memory allocation failed: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },

    /// Memory pool exhausted.
    #[error("Memory pool exhausted: {0}")]
    PoolExhausted(String),

    /// Invalid compressed data.
    #[error("Invalid compressed data: {0}")]
    InvalidData(String),

    /// Decompression failed.
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    /// Unsupported compression algorithm.
    #[error("Unsupported compression algorithm for GPU decompression")]
    UnsupportedAlgorithm,

    /// Kernel launch failed.
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunch(String),

    /// Kernel loading/compilation failed.
    #[error("CUDA kernel load failed: {0}")]
    KernelLoad(String),

    /// Stream synchronization failed.
    #[error("CUDA stream synchronization failed: {0}")]
    StreamSync(String),

    /// Device not found.
    #[error("CUDA device {0} not found")]
    DeviceNotFound(usize),

    /// Compute capability too low.
    #[error("GPU compute capability {0}.{1} too low, requires {2}.{3}")]
    InsufficientComputeCapability(usize, usize, usize, usize),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl CudaError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            CudaError::OutOfMemory { .. } | CudaError::PoolExhausted(_) | CudaError::StreamSync(_)
        )
    }

    /// Check if this error indicates the GPU is unavailable.
    pub fn is_device_error(&self) -> bool {
        matches!(
            self,
            CudaError::DeviceNotFound(_)
                | CudaError::InsufficientComputeCapability(..)
                | CudaError::Driver(_)
        )
    }
}

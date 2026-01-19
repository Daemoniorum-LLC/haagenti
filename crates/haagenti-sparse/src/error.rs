//! Error types for sparse attention

use thiserror::Error;

/// Errors that can occur during sparse attention operations
#[derive(Debug, Error)]
pub enum SparseError {
    /// Invalid mask dimensions
    #[error("Invalid mask dimensions: expected {expected_heads} heads, {expected_layers} layers, got {actual_heads}x{actual_layers}")]
    InvalidDimensions {
        expected_heads: usize,
        expected_layers: usize,
        actual_heads: usize,
        actual_layers: usize,
    },

    /// Head index out of range
    #[error("Head index {index} out of range (max: {max})")]
    HeadIndexOutOfRange { index: usize, max: usize },

    /// Layer index out of range
    #[error("Layer index {index} out of range (max: {max})")]
    LayerIndexOutOfRange { index: usize, max: usize },

    /// Sparsity constraint violation
    #[error("Sparsity {actual:.2} below minimum required {minimum:.2}")]
    SparsityTooLow { actual: f32, minimum: f32 },

    /// Quality constraint violation
    #[error("Quality loss {actual:.3} exceeds threshold {threshold:.3}")]
    QualityLoss { actual: f32, threshold: f32 },

    /// Kernel execution error
    #[error("Kernel execution failed: {0}")]
    KernelError(String),

    /// Prediction model error
    #[error("Prediction failed: {0}")]
    PredictionError(String),

    /// Analysis error
    #[error("Analysis failed: {0}")]
    AnalysisError(String),
}

/// Result type for sparse attention operations
pub type Result<T> = std::result::Result<T, SparseError>;

//! Error types for model merging

use thiserror::Error;

/// Model merging errors
#[derive(Debug, Error)]
pub enum MergeError {
    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Model incompatible
    #[error("Models are incompatible: {0}")]
    IncompatibleModels(String),

    /// Missing layer
    #[error("Missing layer: {0}")]
    MissingLayer(String),

    /// Invalid weights
    #[error("Invalid weights: {0}")]
    InvalidWeights(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Evolution failed
    #[error("Evolution failed: {0}")]
    EvolutionFailed(String),
}

/// Result type for merge operations
pub type Result<T> = std::result::Result<T, MergeError>;

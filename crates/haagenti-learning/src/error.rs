//! Error types for continuous learning

use thiserror::Error;

/// Learning errors
#[derive(Debug, Error)]
pub enum LearningError {
    /// Adapter error
    #[error("Adapter error: {0}")]
    AdapterError(String),

    /// Buffer error
    #[error("Buffer error: {0}")]
    BufferError(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Gradient error
    #[error("Gradient error: {0}")]
    GradientError(String),

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for learning operations
pub type Result<T> = std::result::Result<T, LearningError>;

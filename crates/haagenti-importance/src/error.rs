//! Error types for importance scoring

use thiserror::Error;

/// Result type for importance operations
pub type Result<T> = std::result::Result<T, ImportanceError>;

/// Errors that can occur during importance scoring
#[derive(Error, Debug)]
pub enum ImportanceError {
    /// Invalid prompt
    #[error("Invalid prompt: {0}")]
    InvalidPrompt(String),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// History data corrupted
    #[error("History data corrupted: {0}")]
    HistoryCorrupted(String),

    /// Prediction failed
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<bincode::Error> for ImportanceError {
    fn from(e: bincode::Error) -> Self {
        ImportanceError::Serialization(e.to_string())
    }
}

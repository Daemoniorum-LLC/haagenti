//! Error types for fragment operations

use thiserror::Error;

/// Result type for fragment operations
pub type Result<T> = std::result::Result<T, FragmentError>;

/// Errors that can occur during fragment operations
#[derive(Error, Debug)]
pub enum FragmentError {
    /// Fragment not found in library
    #[error("Fragment not found: {0}")]
    NotFound(String),

    /// Duplicate fragment ID
    #[error("Duplicate fragment ID: {0}")]
    DuplicateId(String),

    /// Invalid fragment data
    #[error("Invalid fragment data: {0}")]
    InvalidData(String),

    /// Signature computation failed
    #[error("Signature computation failed: {0}")]
    SignatureError(String),

    /// Similarity index error
    #[error("Similarity index error: {0}")]
    SimilarityError(String),

    /// Library corruption detected
    #[error("Library corruption: {0}")]
    Corruption(String),

    /// Model manifest error
    #[error("Manifest error: {0}")]
    ManifestError(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Library is locked
    #[error("Library is locked by another process")]
    Locked,

    /// Version mismatch
    #[error("Library version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },
}

impl From<bincode::Error> for FragmentError {
    fn from(e: bincode::Error) -> Self {
        FragmentError::Serialization(e.to_string())
    }
}

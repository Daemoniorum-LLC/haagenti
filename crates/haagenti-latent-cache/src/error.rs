//! Error types for latent caching

use thiserror::Error;

/// Result type for cache operations
pub type Result<T> = std::result::Result<T, CacheError>;

/// Errors that can occur during caching
#[derive(Error, Debug)]
pub enum CacheError {
    /// Embedding computation failed
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),

    /// Similarity search failed
    #[error("Similarity search failed: {0}")]
    SearchFailed(String),

    /// Latent storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Divergence prediction failed
    #[error("Divergence prediction failed: {0}")]
    DivergenceFailed(String),

    /// Cache is full
    #[error("Cache is full")]
    CacheFull,

    /// Entry not found
    #[error("Entry not found: {0}")]
    NotFound(String),

    /// Invalid latent format
    #[error("Invalid latent format: {0}")]
    InvalidFormat(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

//! Error types for distributed inference

use thiserror::Error;

/// Distributed inference errors
#[derive(Debug, Error)]
pub enum DistributedError {
    /// Node connection failed
    #[error("Failed to connect to node {node_id}: {reason}")]
    ConnectionFailed { node_id: String, reason: String },

    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Node timeout
    #[error("Node {node_id} timed out after {timeout_ms}ms")]
    NodeTimeout { node_id: String, timeout_ms: u64 },

    /// Partition error
    #[error("Partition error: {0}")]
    PartitionError(String),

    /// Synchronization error
    #[error("Synchronization error: {0}")]
    SyncError(String),

    /// Communication error
    #[error("Communication error: {0}")]
    CommError(String),

    /// Topology error
    #[error("Topology error: {0}")]
    TopologyError(String),

    /// Insufficient nodes
    #[error("Insufficient nodes: need {required}, have {available}")]
    InsufficientNodes { required: usize, available: usize },

    /// Job failed
    #[error("Job {job_id} failed: {reason}")]
    JobFailed { job_id: String, reason: String },

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for distributed operations
pub type Result<T> = std::result::Result<T, DistributedError>;

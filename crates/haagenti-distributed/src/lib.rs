//! Distributed inference across multiple nodes
//!
//! This crate provides distributed model inference with:
//! - Tensor parallelism for large layers
//! - Pipeline parallelism for model stages
//! - Expert parallelism for MoE models
//! - Ring all-reduce for gradient synchronization
//! - Fault-tolerant execution with node failover

mod coordinator;
mod error;
mod node;
mod partition;
mod protocol;
mod topology;

pub use coordinator::{Coordinator, CoordinatorConfig, JobStatus};
pub use error::{DistributedError, Result};
pub use node::{Node, NodeConfig, NodeRole, NodeStatus};
pub use partition::{ModelPartition, PartitionStrategy, TensorPartition};
pub use protocol::{Message, MessageType, Protocol};
pub use topology::{Mesh, Ring, Topology, TopologyConfig};

/// Parallelism strategies
pub mod parallelism {
    pub use super::partition::ExpertParallel;
    pub use super::partition::PipelineParallel;
    pub use super::partition::TensorParallel;
}

/// Communication primitives
pub mod comm {
    pub use super::protocol::AllReduce;
    pub use super::protocol::Broadcast;
    pub use super::protocol::Gather;
    pub use super::protocol::Scatter;
}

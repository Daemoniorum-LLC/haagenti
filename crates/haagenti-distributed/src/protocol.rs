//! Communication protocol for distributed inference

use crate::{DistributedError, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Message type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Heartbeat
    Heartbeat,
    /// Heartbeat acknowledgment
    HeartbeatAck,
    /// Join request
    JoinRequest,
    /// Join response
    JoinResponse,
    /// Leave notification
    Leave,
    /// Tensor data
    TensorData,
    /// Tensor request
    TensorRequest,
    /// All-reduce operation
    AllReduce,
    /// Broadcast operation
    Broadcast,
    /// Scatter operation
    Scatter,
    /// Gather operation
    Gather,
    /// Barrier synchronization
    Barrier,
    /// Error
    Error,
}

/// Protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message ID
    pub id: u64,
    /// Source node
    pub source: String,
    /// Destination node (empty for broadcast)
    pub destination: String,
    /// Message type
    pub msg_type: MessageType,
    /// Payload
    pub payload: Vec<u8>,
    /// Timestamp (unix ms)
    pub timestamp: u64,
    /// Sequence number for ordering
    pub sequence: u64,
}

impl Message {
    /// Create new message
    pub fn new(
        source: impl Into<String>,
        destination: impl Into<String>,
        msg_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        static NEXT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            source: source.into(),
            destination: destination.into(),
            msg_type,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            sequence: NEXT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Create heartbeat message
    pub fn heartbeat(source: impl Into<String>) -> Self {
        Self::new(source, "", MessageType::Heartbeat, Vec::new())
    }

    /// Create heartbeat ack
    pub fn heartbeat_ack(source: impl Into<String>, destination: impl Into<String>) -> Self {
        Self::new(source, destination, MessageType::HeartbeatAck, Vec::new())
    }

    /// Create error message
    pub fn error(source: impl Into<String>, destination: impl Into<String>, error: &str) -> Self {
        Self::new(source, destination, MessageType::Error, error.as_bytes().to_vec())
    }

    /// Serialize message
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| DistributedError::SerializationError(e.to_string()))
    }

    /// Deserialize message
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| DistributedError::SerializationError(e.to_string()))
    }

    /// Get payload as string
    pub fn payload_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.payload).ok()
    }
}

/// Communication protocol
#[derive(Debug)]
pub struct Protocol {
    /// Node ID
    node_id: String,
    /// Message handlers
    pending_responses: std::collections::HashMap<u64, tokio::sync::oneshot::Sender<Message>>,
    /// Request timeout
    timeout: Duration,
}

impl Protocol {
    /// Create new protocol instance
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            pending_responses: std::collections::HashMap::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Set timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
}

/// All-reduce operation for gradient synchronization
#[derive(Debug)]
pub struct AllReduce {
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
    /// Reduction operation
    op: ReduceOp,
}

/// Reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum
    Sum,
    /// Average
    Avg,
    /// Maximum
    Max,
    /// Minimum
    Min,
}

impl AllReduce {
    /// Create new all-reduce
    pub fn new(world_size: usize, rank: usize, op: ReduceOp) -> Self {
        Self {
            world_size,
            rank,
            op,
        }
    }

    /// Ring all-reduce algorithm
    pub fn ring_allreduce(&self, local_data: &mut [f32]) -> Result<()> {
        let n = local_data.len();
        let chunk_size = (n + self.world_size - 1) / self.world_size;

        // Phase 1: Reduce-scatter
        for step in 0..self.world_size - 1 {
            let send_chunk = (self.rank + self.world_size - step - 1) % self.world_size;
            let recv_chunk = (self.rank + self.world_size - step) % self.world_size;

            let send_start = send_chunk * chunk_size;
            let recv_start = recv_chunk * chunk_size;

            // In real implementation, send/receive over network
            // For now, simulate the reduction
            let _send_data = &local_data[send_start..send_start.min(n).max(send_start + chunk_size).min(n)];
            let recv_slice = &mut local_data[recv_start..recv_start.min(n).max(recv_start + chunk_size).min(n)];

            // Apply reduction (simulated - would receive from neighbor)
            for val in recv_slice.iter_mut() {
                match self.op {
                    ReduceOp::Sum => *val *= 2.0, // Simulated sum from 2 nodes
                    ReduceOp::Avg => {}, // Average divides after all sums
                    ReduceOp::Max => {},
                    ReduceOp::Min => {},
                }
            }
        }

        // Phase 2: All-gather
        for _step in 0..self.world_size - 1 {
            // Similar to reduce-scatter but without reduction
        }

        // Apply final operation
        if self.op == ReduceOp::Avg {
            for val in local_data.iter_mut() {
                *val /= self.world_size as f32;
            }
        }

        Ok(())
    }

    /// Bandwidth-optimal recursive halving-doubling
    pub fn recursive_halving_doubling(&self, local_data: &mut [f32]) -> Result<()> {
        // More efficient for small messages or non-power-of-2 world sizes
        let mut step = 1;

        // Reduce-scatter phase
        while step < self.world_size {
            let partner = self.rank ^ step;
            if partner < self.world_size {
                // Exchange and reduce with partner
                // Simulated here
            }
            step *= 2;
        }

        // All-gather phase
        step = self.world_size / 2;
        while step >= 1 {
            let partner = self.rank ^ step;
            if partner < self.world_size {
                // Exchange with partner
            }
            step /= 2;
        }

        Ok(())
    }
}

/// Broadcast operation
#[derive(Debug)]
pub struct Broadcast {
    /// Root rank
    root: usize,
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
}

impl Broadcast {
    /// Create new broadcast
    pub fn new(root: usize, world_size: usize, rank: usize) -> Self {
        Self {
            root,
            world_size,
            rank,
        }
    }

    /// Binary tree broadcast
    pub fn tree_broadcast<T: Clone>(&self, data: &mut Option<T>) -> Result<()> {
        if self.rank == self.root {
            // Root sends to children
            if data.is_none() {
                return Err(DistributedError::CommError("Root has no data".into()));
            }
        } else {
            // Non-root receives from parent
            // Simulated - would receive over network
        }
        Ok(())
    }

    /// Is this rank the root
    pub fn is_root(&self) -> bool {
        self.rank == self.root
    }
}

/// Scatter operation
#[derive(Debug)]
pub struct Scatter {
    /// Root rank
    root: usize,
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
}

impl Scatter {
    /// Create new scatter
    pub fn new(root: usize, world_size: usize, rank: usize) -> Self {
        Self {
            root,
            world_size,
            rank,
        }
    }

    /// Scatter data from root to all ranks
    pub fn scatter<T: Clone>(&self, send_data: Option<&[T]>, recv_buf: &mut [T]) -> Result<()> {
        if self.rank == self.root {
            let data = send_data.ok_or_else(|| {
                DistributedError::CommError("Root must provide send data".into())
            })?;

            // Calculate chunk for each rank
            let chunk_size = data.len() / self.world_size;

            // Root keeps its chunk
            let my_chunk = &data[self.rank * chunk_size..(self.rank + 1) * chunk_size];
            recv_buf[..chunk_size].clone_from_slice(my_chunk);

            // Send to other ranks (simulated)
        } else {
            // Receive from root (simulated)
        }

        Ok(())
    }
}

/// Gather operation
#[derive(Debug)]
pub struct Gather {
    /// Root rank
    root: usize,
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
}

impl Gather {
    /// Create new gather
    pub fn new(root: usize, world_size: usize, rank: usize) -> Self {
        Self {
            root,
            world_size,
            rank,
        }
    }

    /// Gather data from all ranks to root
    pub fn gather<T: Clone>(&self, send_data: &[T], recv_buf: Option<&mut [T]>) -> Result<()> {
        if self.rank == self.root {
            let buf = recv_buf.ok_or_else(|| {
                DistributedError::CommError("Root must provide receive buffer".into())
            })?;

            let chunk_size = send_data.len();

            // Root copies its own data
            buf[self.rank * chunk_size..(self.rank + 1) * chunk_size]
                .clone_from_slice(send_data);

            // Receive from other ranks (simulated)
        } else {
            // Send to root (simulated)
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new("node-1", "node-2", MessageType::TensorData, vec![1, 2, 3]);

        assert_eq!(msg.source, "node-1");
        assert_eq!(msg.destination, "node-2");
        assert_eq!(msg.msg_type, MessageType::TensorData);
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::heartbeat("node-1");
        let data = msg.serialize().unwrap();
        let restored = Message::deserialize(&data).unwrap();

        assert_eq!(msg.id, restored.id);
        assert_eq!(msg.source, restored.source);
    }

    #[test]
    fn test_allreduce_creation() {
        let allreduce = AllReduce::new(4, 0, ReduceOp::Sum);
        assert_eq!(allreduce.world_size, 4);
        assert_eq!(allreduce.rank, 0);
    }

    #[test]
    fn test_broadcast() {
        let broadcast = Broadcast::new(0, 4, 0);
        assert!(broadcast.is_root());

        let broadcast = Broadcast::new(0, 4, 1);
        assert!(!broadcast.is_root());
    }
}

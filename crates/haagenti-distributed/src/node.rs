//! Node management for distributed inference

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// Node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node ID
    pub id: String,
    /// Node address
    pub address: SocketAddr,
    /// Node role
    pub role: NodeRole,
    /// GPU count
    pub gpu_count: usize,
    /// Memory capacity in bytes
    pub memory_capacity: u64,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            id: uuid_v4(),
            address: "127.0.0.1:9000".parse().unwrap(),
            role: NodeRole::Worker,
            gpu_count: 1,
            memory_capacity: 16 * 1024 * 1024 * 1024, // 16GB
            heartbeat_interval: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(30),
        }
    }
}

/// Generate UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("node-{:x}", ts)
}

/// Node role in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Coordinator node
    Coordinator,
    /// Worker node
    Worker,
    /// Hybrid (both coordinator and worker)
    Hybrid,
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is starting up
    Starting,
    /// Node is ready for work
    Ready,
    /// Node is busy with a job
    Busy,
    /// Node is draining (finishing current work)
    Draining,
    /// Node is offline
    Offline,
    /// Node has failed
    Failed,
}

/// Resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (0.0 - 1.0)
    pub cpu: f32,
    /// Memory used in bytes
    pub memory_used: u64,
    /// GPU memory used per GPU
    pub gpu_memory: Vec<u64>,
    /// Network bandwidth used (bytes/sec)
    pub network_bandwidth: u64,
}

/// Node in the distributed cluster
#[derive(Debug)]
pub struct Node {
    /// Configuration
    config: NodeConfig,
    /// Current status
    status: NodeStatus,
    /// Last heartbeat time
    last_heartbeat: Instant,
    /// Resource usage
    resources: ResourceUsage,
    /// Assigned shards
    assigned_shards: Vec<ShardAssignment>,
    /// Active connections
    connections: HashMap<String, ConnectionInfo>,
}

/// Shard assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Shard ID
    pub shard_id: String,
    /// Layer range (start, end)
    pub layer_range: (usize, usize),
    /// Memory required
    pub memory_required: u64,
}

/// Connection info
#[derive(Debug, Clone)]
struct ConnectionInfo {
    peer_id: String,
    connected_at: Instant,
    bytes_sent: u64,
    bytes_received: u64,
}

impl Node {
    /// Create new node
    pub fn new(config: NodeConfig) -> Self {
        Self {
            config,
            status: NodeStatus::Starting,
            last_heartbeat: Instant::now(),
            resources: ResourceUsage::default(),
            assigned_shards: Vec::new(),
            connections: HashMap::new(),
        }
    }

    /// Get node ID
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Get node address
    pub fn address(&self) -> SocketAddr {
        self.config.address
    }

    /// Get node role
    pub fn role(&self) -> NodeRole {
        self.config.role
    }

    /// Get current status
    pub fn status(&self) -> NodeStatus {
        self.status
    }

    /// Set status
    pub fn set_status(&mut self, status: NodeStatus) {
        self.status = status;
    }

    /// Update heartbeat
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Check if node is alive
    pub fn is_alive(&self) -> bool {
        self.last_heartbeat.elapsed() < self.config.heartbeat_interval * 3
    }

    /// Get resource usage
    pub fn resources(&self) -> &ResourceUsage {
        &self.resources
    }

    /// Update resource usage
    pub fn update_resources(&mut self, resources: ResourceUsage) {
        self.resources = resources;
    }

    /// Get available memory
    pub fn available_memory(&self) -> u64 {
        self.config.memory_capacity.saturating_sub(self.resources.memory_used)
    }

    /// Assign shard to node
    pub fn assign_shard(&mut self, assignment: ShardAssignment) -> Result<()> {
        if assignment.memory_required > self.available_memory() {
            return Err(DistributedError::PartitionError(
                "Insufficient memory for shard".into(),
            ));
        }
        self.assigned_shards.push(assignment);
        Ok(())
    }

    /// Get assigned shards
    pub fn shards(&self) -> &[ShardAssignment] {
        &self.assigned_shards
    }

    /// Clear shard assignments
    pub fn clear_shards(&mut self) {
        self.assigned_shards.clear();
    }

    /// Record connection
    pub fn add_connection(&mut self, peer_id: String) {
        self.connections.insert(
            peer_id.clone(),
            ConnectionInfo {
                peer_id,
                connected_at: Instant::now(),
                bytes_sent: 0,
                bytes_received: 0,
            },
        );
    }

    /// Remove connection
    pub fn remove_connection(&mut self, peer_id: &str) {
        self.connections.remove(peer_id);
    }

    /// Connection count
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

/// Node registry for tracking all nodes
#[derive(Debug, Default)]
pub struct NodeRegistry {
    /// Nodes by ID
    nodes: HashMap<String, Node>,
    /// Coordinator node ID
    coordinator_id: Option<String>,
}

impl NodeRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node
    pub fn register(&mut self, node: Node) {
        let id = node.id().to_string();
        if node.role() == NodeRole::Coordinator || node.role() == NodeRole::Hybrid {
            if self.coordinator_id.is_none() {
                self.coordinator_id = Some(id.clone());
            }
        }
        self.nodes.insert(id, node);
    }

    /// Unregister a node
    pub fn unregister(&mut self, id: &str) -> Option<Node> {
        if self.coordinator_id.as_deref() == Some(id) {
            self.coordinator_id = None;
        }
        self.nodes.remove(id)
    }

    /// Get node by ID
    pub fn get(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Get coordinator node
    pub fn coordinator(&self) -> Option<&Node> {
        self.coordinator_id.as_ref().and_then(|id| self.nodes.get(id))
    }

    /// Get all worker nodes
    pub fn workers(&self) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.role() == NodeRole::Worker || n.role() == NodeRole::Hybrid)
            .collect()
    }

    /// Get ready workers
    pub fn ready_workers(&self) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| {
                (n.role() == NodeRole::Worker || n.role() == NodeRole::Hybrid)
                    && n.status() == NodeStatus::Ready
            })
            .collect()
    }

    /// Total node count
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Update heartbeat for node
    pub fn heartbeat(&mut self, id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.heartbeat();
            true
        } else {
            false
        }
    }

    /// Check for dead nodes
    pub fn check_health(&mut self) -> Vec<String> {
        let mut dead = Vec::new();

        for (id, node) in &mut self.nodes {
            if !node.is_alive() && node.status() != NodeStatus::Offline {
                node.set_status(NodeStatus::Failed);
                dead.push(id.clone());
            }
        }

        dead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_config_default() {
        let config = NodeConfig::default();
        assert!(config.id.starts_with("node-"));
        assert_eq!(config.gpu_count, 1);
    }

    #[test]
    fn test_node_creation() {
        let config = NodeConfig::default();
        let node = Node::new(config);

        assert_eq!(node.status(), NodeStatus::Starting);
        assert!(node.is_alive());
    }

    #[test]
    fn test_node_resources() {
        let config = NodeConfig {
            memory_capacity: 1024,
            ..Default::default()
        };
        let mut node = Node::new(config);

        assert_eq!(node.available_memory(), 1024);

        node.update_resources(ResourceUsage {
            memory_used: 512,
            ..Default::default()
        });

        assert_eq!(node.available_memory(), 512);
    }

    #[test]
    fn test_shard_assignment() {
        let config = NodeConfig {
            memory_capacity: 1024,
            ..Default::default()
        };
        let mut node = Node::new(config);

        let assignment = ShardAssignment {
            shard_id: "shard-1".into(),
            layer_range: (0, 10),
            memory_required: 512,
        };

        node.assign_shard(assignment).unwrap();
        assert_eq!(node.shards().len(), 1);
    }

    #[test]
    fn test_node_registry() {
        let mut registry = NodeRegistry::new();

        let config = NodeConfig {
            role: NodeRole::Coordinator,
            ..Default::default()
        };
        let node = Node::new(config);
        let id = node.id().to_string();

        registry.register(node);
        assert_eq!(registry.len(), 1);
        assert!(registry.coordinator().is_some());

        registry.unregister(&id);
        assert!(registry.is_empty());
    }
}

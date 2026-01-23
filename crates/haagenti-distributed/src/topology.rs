//! Network topology for distributed inference

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Topology type
    pub topology_type: TopologyType,
    /// Enable automatic topology detection
    pub auto_detect: bool,
    /// NUMA awareness
    pub numa_aware: bool,
    /// Network bandwidth estimation (Gbps)
    pub estimated_bandwidth_gbps: f64,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Ring,
            auto_detect: true,
            numa_aware: true,
            estimated_bandwidth_gbps: 100.0,
        }
    }
}

/// Topology type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    /// Ring topology (for all-reduce)
    Ring,
    /// 2D mesh/torus
    Mesh,
    /// Fully connected
    FullyConnected,
    /// Tree (for broadcast)
    Tree,
    /// Hierarchical (multi-level)
    Hierarchical,
}

/// Network topology abstraction
#[derive(Debug)]
pub struct Topology {
    /// Configuration
    config: TopologyConfig,
    /// Node IDs in order
    nodes: Vec<String>,
    /// Node to rank mapping
    node_to_rank: HashMap<String, usize>,
    /// Adjacency list
    adjacency: Vec<HashSet<usize>>,
}

impl Topology {
    /// Create new topology
    pub fn new(config: TopologyConfig, nodes: Vec<String>) -> Self {
        let node_to_rank: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let adjacency = Self::build_adjacency(&config.topology_type, nodes.len());

        Self {
            config,
            nodes,
            node_to_rank,
            adjacency,
        }
    }

    /// Build adjacency list for topology type
    fn build_adjacency(topology_type: &TopologyType, size: usize) -> Vec<HashSet<usize>> {
        let mut adj = vec![HashSet::new(); size];

        match topology_type {
            TopologyType::Ring => {
                for (i, set) in adj.iter_mut().enumerate() {
                    let prev = (i + size - 1) % size;
                    let next = (i + 1) % size;
                    set.insert(prev);
                    set.insert(next);
                }
            }
            TopologyType::Mesh => {
                // 2D mesh (approximately square)
                let side = (size as f64).sqrt().ceil() as usize;
                for (i, set) in adj.iter_mut().enumerate() {
                    let row = i / side;
                    let col = i % side;

                    // Left neighbor
                    if col > 0 {
                        set.insert(i - 1);
                    }
                    // Right neighbor
                    if col < side - 1 && i + 1 < size {
                        set.insert(i + 1);
                    }
                    // Top neighbor
                    if row > 0 {
                        set.insert(i - side);
                    }
                    // Bottom neighbor
                    if i + side < size {
                        set.insert(i + side);
                    }
                }
            }
            TopologyType::FullyConnected => {
                for (i, set) in adj.iter_mut().enumerate() {
                    for j in 0..size {
                        if i != j {
                            set.insert(j);
                        }
                    }
                }
            }
            TopologyType::Tree => {
                // Binary tree with root at 0
                for (i, set) in adj.iter_mut().enumerate() {
                    // Parent
                    if i > 0 {
                        set.insert((i - 1) / 2);
                    }
                    // Left child
                    let left = 2 * i + 1;
                    if left < size {
                        set.insert(left);
                    }
                    // Right child
                    let right = 2 * i + 2;
                    if right < size {
                        set.insert(right);
                    }
                }
            }
            TopologyType::Hierarchical => {
                // Two-level hierarchy
                // Level 0: Intra-group (ring)
                // Level 1: Inter-group (all-to-all between leaders)
                let group_size = 4;
                let num_groups = size.div_ceil(group_size);

                for (i, set) in adj.iter_mut().enumerate() {
                    let group = i / group_size;
                    let pos_in_group = i % group_size;

                    // Intra-group ring
                    let group_start = group * group_size;
                    let group_end = (group_start + group_size).min(size);
                    let actual_group_size = group_end - group_start;

                    if actual_group_size > 1 {
                        let prev = group_start
                            + (pos_in_group + actual_group_size - 1) % actual_group_size;
                        let next = group_start + (pos_in_group + 1) % actual_group_size;
                        set.insert(prev);
                        set.insert(next);
                    }

                    // Inter-group (leaders only, pos 0 in each group)
                    if pos_in_group == 0 {
                        for other_group in 0..num_groups {
                            if other_group != group {
                                let leader = other_group * group_size;
                                if leader < size {
                                    set.insert(leader);
                                }
                            }
                        }
                    }
                }
            }
        }

        adj
    }

    /// Get rank for node ID
    pub fn rank(&self, node_id: &str) -> Option<usize> {
        self.node_to_rank.get(node_id).copied()
    }

    /// Get node ID for rank
    pub fn node(&self, rank: usize) -> Option<&str> {
        self.nodes.get(rank).map(|s| s.as_str())
    }

    /// Get neighbors for rank
    pub fn neighbors(&self, rank: usize) -> &HashSet<usize> {
        &self.adjacency[rank]
    }

    /// World size
    pub fn world_size(&self) -> usize {
        self.nodes.len()
    }

    /// Topology type
    pub fn topology_type(&self) -> TopologyType {
        self.config.topology_type
    }

    /// Find shortest path between two ranks
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return None;
        }

        if from == to {
            return Some(vec![from]);
        }

        // BFS
        let mut visited = vec![false; self.nodes.len()];
        let mut parent = vec![None; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();

        visited[from] = true;
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = to;
                while let Some(p) = parent[node] {
                    path.push(node);
                    node = p;
                }
                path.push(from);
                path.reverse();
                return Some(path);
            }

            for &neighbor in &self.adjacency[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Calculate diameter (longest shortest path)
    pub fn diameter(&self) -> usize {
        let mut max_dist = 0;

        for i in 0..self.nodes.len() {
            for j in i + 1..self.nodes.len() {
                if let Some(path) = self.shortest_path(i, j) {
                    max_dist = max_dist.max(path.len() - 1);
                }
            }
        }

        max_dist
    }

    /// Check if topology is connected
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![0usize];

        while let Some(node) = stack.pop() {
            if !visited[node] {
                visited[node] = true;
                for &neighbor in &self.adjacency[node] {
                    stack.push(neighbor);
                }
            }
        }

        visited.iter().all(|&v| v)
    }
}

/// Ring topology helper
pub struct Ring;

impl Ring {
    /// Get left neighbor in ring
    pub fn left(rank: usize, world_size: usize) -> usize {
        (rank + world_size - 1) % world_size
    }

    /// Get right neighbor in ring
    pub fn right(rank: usize, world_size: usize) -> usize {
        (rank + 1) % world_size
    }

    /// Distance in ring (minimum hops)
    pub fn distance(from: usize, to: usize, world_size: usize) -> usize {
        let forward = (to + world_size - from) % world_size;
        let backward = (from + world_size - to) % world_size;
        forward.min(backward)
    }
}

/// Mesh topology helper
pub struct Mesh;

impl Mesh {
    /// Get mesh dimensions for given size
    pub fn dimensions(size: usize) -> (usize, usize) {
        let side = (size as f64).sqrt().ceil() as usize;
        let rows = size.div_ceil(side);
        (rows, side)
    }

    /// Get (row, col) for rank
    pub fn position(rank: usize, cols: usize) -> (usize, usize) {
        (rank / cols, rank % cols)
    }

    /// Get rank for (row, col)
    pub fn rank(row: usize, col: usize, cols: usize) -> usize {
        row * cols + col
    }

    /// Manhattan distance
    pub fn distance(from: usize, to: usize, cols: usize) -> usize {
        let (r1, c1) = Self::position(from, cols);
        let (r2, c2) = Self::position(to, cols);
        (r1 as isize - r2 as isize).unsigned_abs() + (c1 as isize - c2 as isize).unsigned_abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_topology() {
        let nodes = vec!["n0".into(), "n1".into(), "n2".into(), "n3".into()];
        let config = TopologyConfig {
            topology_type: TopologyType::Ring,
            ..Default::default()
        };
        let topology = Topology::new(config, nodes);

        assert_eq!(topology.world_size(), 4);
        assert!(topology.neighbors(0).contains(&3));
        assert!(topology.neighbors(0).contains(&1));
        assert!(topology.is_connected());
    }

    #[test]
    fn test_mesh_topology() {
        let nodes: Vec<String> = (0..9).map(|i| format!("n{}", i)).collect();
        let config = TopologyConfig {
            topology_type: TopologyType::Mesh,
            ..Default::default()
        };
        let topology = Topology::new(config, nodes);

        // Node 4 (center of 3x3) should have 4 neighbors
        assert_eq!(topology.neighbors(4).len(), 4);
        assert!(topology.is_connected());
    }

    #[test]
    fn test_shortest_path() {
        let nodes = vec!["n0".into(), "n1".into(), "n2".into(), "n3".into()];
        let config = TopologyConfig {
            topology_type: TopologyType::Ring,
            ..Default::default()
        };
        let topology = Topology::new(config, nodes);

        let path = topology.shortest_path(0, 2).unwrap();
        assert_eq!(path.len(), 3); // 0 -> 1 -> 2 or 0 -> 3 -> 2
    }

    #[test]
    fn test_ring_helpers() {
        assert_eq!(Ring::left(0, 4), 3);
        assert_eq!(Ring::right(3, 4), 0);
        assert_eq!(Ring::distance(0, 2, 4), 2);
    }

    #[test]
    fn test_mesh_helpers() {
        assert_eq!(Mesh::dimensions(9), (3, 3));
        assert_eq!(Mesh::position(4, 3), (1, 1));
        assert_eq!(Mesh::rank(1, 1, 3), 4);
        assert_eq!(Mesh::distance(0, 8, 3), 4); // (0,0) to (2,2)
    }

    #[test]
    fn test_tree_topology() {
        let nodes: Vec<String> = (0..7).map(|i| format!("n{}", i)).collect();
        let config = TopologyConfig {
            topology_type: TopologyType::Tree,
            ..Default::default()
        };
        let topology = Topology::new(config, nodes);

        // Root has 2 children
        assert_eq!(topology.neighbors(0).len(), 2);
        assert!(topology.neighbors(0).contains(&1));
        assert!(topology.neighbors(0).contains(&2));
        assert!(topology.is_connected());
    }
}

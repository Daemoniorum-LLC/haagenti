//! GPU memory snapshot and restore for fast recovery

use crate::{Result, ServerlessError};
use arcanum_primitives::prelude::Blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Snapshot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Snapshot directory
    pub snapshot_dir: PathBuf,
    /// Enable compression
    pub compression: bool,
    /// Compression level (1-22 for zstd)
    pub compression_level: i32,
    /// Enable incremental snapshots
    pub incremental: bool,
    /// Maximum snapshot age in seconds
    pub max_age_seconds: u64,
    /// Enable checksum verification
    pub verify_checksum: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            snapshot_dir: PathBuf::from("/tmp/haagenti-snapshots"),
            compression: true,
            compression_level: 3,
            incremental: true,
            max_age_seconds: 3600, // 1 hour
            verify_checksum: true,
        }
    }
}

/// GPU memory snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSnapshot {
    /// Snapshot ID
    pub id: String,
    /// Version
    pub version: u32,
    /// Creation timestamp (unix ms)
    pub created_at: u64,
    /// Total size in bytes
    pub total_size: u64,
    /// Buffer snapshots
    pub buffers: Vec<BufferSnapshot>,
    /// Model weights hash
    pub weights_hash: String,
    /// Checksum
    pub checksum: String,
}

/// Individual buffer snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferSnapshot {
    /// Buffer name
    pub name: String,
    /// Buffer size
    pub size: u64,
    /// Offset in snapshot file
    pub offset: u64,
    /// Compressed size (if compression enabled)
    pub compressed_size: Option<u64>,
    /// Buffer type
    pub buffer_type: BufferType,
}

/// Buffer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferType {
    /// Model weights (read-only)
    Weights,
    /// KV cache
    KvCache,
    /// Activations
    Activations,
    /// Gradients
    Gradients,
    /// Optimizer state
    OptimizerState,
    /// Other
    Other,
}

impl GpuSnapshot {
    /// Create new snapshot
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: 1,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            total_size: 0,
            buffers: Vec::new(),
            weights_hash: String::new(),
            checksum: String::new(),
        }
    }

    /// Add buffer to snapshot
    pub fn add_buffer(&mut self, name: impl Into<String>, size: u64, buffer_type: BufferType) {
        let offset = self.total_size;
        self.buffers.push(BufferSnapshot {
            name: name.into(),
            size,
            offset,
            compressed_size: None,
            buffer_type,
        });
        self.total_size += size;
    }

    /// Get buffer by name
    pub fn get_buffer(&self, name: &str) -> Option<&BufferSnapshot> {
        self.buffers.iter().find(|b| b.name == name)
    }

    /// Get all buffers of a type
    pub fn get_buffers_by_type(&self, buffer_type: BufferType) -> Vec<&BufferSnapshot> {
        self.buffers
            .iter()
            .filter(|b| b.buffer_type == buffer_type)
            .collect()
    }

    /// Compute checksum
    pub fn compute_checksum(&mut self, data: &[u8]) {
        let hash = Blake3::hash(data);
        self.checksum = hash.iter().map(|b| format!("{:02x}", b)).collect();
    }

    /// Verify checksum
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        let hash = Blake3::hash(data);
        let computed: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        computed == self.checksum
    }
}

/// Snapshot manager
#[derive(Debug)]
pub struct SnapshotManager {
    /// Configuration
    config: SnapshotConfig,
    /// Cached snapshots
    snapshots: HashMap<String, GpuSnapshot>,
    /// Statistics
    stats: SnapshotStats,
}

/// Snapshot statistics
#[derive(Debug, Default)]
pub struct SnapshotStats {
    /// Total snapshots created
    pub created: u64,
    /// Total snapshots restored
    pub restored: u64,
    /// Total bytes saved
    pub bytes_saved: u64,
    /// Total bytes restored
    pub bytes_restored: u64,
    /// Average save time ms
    pub avg_save_ms: f64,
    /// Average restore time ms
    pub avg_restore_ms: f64,
}

impl SnapshotManager {
    /// Create new manager
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            snapshots: HashMap::new(),
            stats: SnapshotStats::default(),
        }
    }

    /// Create a snapshot
    pub async fn create_snapshot(
        &mut self,
        id: impl Into<String>,
        buffers: Vec<(String, Vec<u8>, BufferType)>,
    ) -> Result<GpuSnapshot> {
        let start = Instant::now();
        let id = id.into();

        let mut snapshot = GpuSnapshot::new(&id);
        let mut data = Vec::new();

        for (name, buffer_data, buffer_type) in buffers {
            snapshot.add_buffer(&name, buffer_data.len() as u64, buffer_type);

            if self.config.compression {
                // In real implementation, compress with zstd
                data.extend_from_slice(&buffer_data);
            } else {
                data.extend_from_slice(&buffer_data);
            }
        }

        if self.config.verify_checksum {
            snapshot.compute_checksum(&data);
        }

        // Save to disk
        self.save_to_disk(&snapshot, &data).await?;

        self.snapshots.insert(id, snapshot.clone());
        self.stats.created += 1;
        self.stats.bytes_saved += snapshot.total_size;

        let elapsed = start.elapsed().as_millis() as f64;
        self.stats.avg_save_ms = (self.stats.avg_save_ms * (self.stats.created - 1) as f64
            + elapsed)
            / self.stats.created as f64;

        Ok(snapshot)
    }

    /// Restore a snapshot
    pub async fn restore_snapshot(&mut self, id: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let start = Instant::now();

        // Try to load from cache
        let snapshot = if let Some(s) = self.snapshots.get(id) {
            s.clone()
        } else {
            // Load from disk
            self.load_from_disk(id).await?
        };

        // Load data from disk
        let data = self.load_data(id).await?;

        // Verify checksum
        if self.config.verify_checksum && !snapshot.verify_checksum(&data) {
            return Err(ServerlessError::SnapshotError(
                "Checksum verification failed".into(),
            ));
        }

        // Extract buffers
        let mut buffers = Vec::new();
        for buffer in &snapshot.buffers {
            let start = buffer.offset as usize;
            let end = start + buffer.size as usize;
            let buffer_data = data[start..end].to_vec();
            buffers.push((buffer.name.clone(), buffer_data));
        }

        self.stats.restored += 1;
        self.stats.bytes_restored += snapshot.total_size;

        let elapsed = start.elapsed().as_millis() as f64;
        self.stats.avg_restore_ms = (self.stats.avg_restore_ms * (self.stats.restored - 1) as f64
            + elapsed)
            / self.stats.restored as f64;

        Ok(buffers)
    }

    /// Save snapshot to disk
    async fn save_to_disk(&self, snapshot: &GpuSnapshot, data: &[u8]) -> Result<()> {
        let dir = &self.config.snapshot_dir;
        std::fs::create_dir_all(dir)?;

        // Save metadata
        let meta_path = dir.join(format!("{}.meta.json", snapshot.id));
        let meta_json = serde_json::to_string_pretty(snapshot)
            .map_err(|e| ServerlessError::SerializationError(e.to_string()))?;
        std::fs::write(&meta_path, meta_json)?;

        // Save data
        let data_path = dir.join(format!("{}.data", snapshot.id));
        std::fs::write(&data_path, data)?;

        Ok(())
    }

    /// Load snapshot from disk
    async fn load_from_disk(&mut self, id: &str) -> Result<GpuSnapshot> {
        let meta_path = self.config.snapshot_dir.join(format!("{}.meta.json", id));
        let meta_json = std::fs::read_to_string(&meta_path)?;
        let snapshot: GpuSnapshot = serde_json::from_str(&meta_json)
            .map_err(|e| ServerlessError::DeserializationError(e.to_string()))?;

        self.snapshots.insert(id.to_string(), snapshot.clone());
        Ok(snapshot)
    }

    /// Load data from disk
    async fn load_data(&self, id: &str) -> Result<Vec<u8>> {
        let data_path = self.config.snapshot_dir.join(format!("{}.data", id));
        let data = std::fs::read(&data_path)?;
        Ok(data)
    }

    /// List available snapshots
    pub fn list_snapshots(&self) -> Vec<&str> {
        self.snapshots.keys().map(|s| s.as_str()).collect()
    }

    /// Delete snapshot
    pub fn delete_snapshot(&mut self, id: &str) -> Result<()> {
        self.snapshots.remove(id);

        let meta_path = self.config.snapshot_dir.join(format!("{}.meta.json", id));
        let data_path = self.config.snapshot_dir.join(format!("{}.data", id));

        if meta_path.exists() {
            std::fs::remove_file(meta_path)?;
        }
        if data_path.exists() {
            std::fs::remove_file(data_path)?;
        }

        Ok(())
    }

    /// Clear old snapshots
    pub fn clear_old(&mut self) -> Result<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let max_age_ms = self.config.max_age_seconds * 1000;
        let mut to_delete = Vec::new();

        for (id, snapshot) in &self.snapshots {
            if now - snapshot.created_at > max_age_ms {
                to_delete.push(id.clone());
            }
        }

        for id in &to_delete {
            self.delete_snapshot(id)?;
        }

        Ok(to_delete.len())
    }

    /// Get statistics
    pub fn stats(&self) -> &SnapshotStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SnapshotConfig::default();
        assert!(config.compression);
        assert!(config.verify_checksum);
    }

    #[test]
    fn test_snapshot_creation() {
        let mut snapshot = GpuSnapshot::new("test-snapshot");

        snapshot.add_buffer("weights", 1024, BufferType::Weights);
        snapshot.add_buffer("kv_cache", 512, BufferType::KvCache);

        assert_eq!(snapshot.buffers.len(), 2);
        assert_eq!(snapshot.total_size, 1536);
    }

    #[test]
    fn test_buffer_lookup() {
        let mut snapshot = GpuSnapshot::new("test");

        snapshot.add_buffer("weights", 1024, BufferType::Weights);
        snapshot.add_buffer("cache", 512, BufferType::KvCache);

        assert!(snapshot.get_buffer("weights").is_some());
        assert!(snapshot.get_buffer("nonexistent").is_none());

        let weights = snapshot.get_buffers_by_type(BufferType::Weights);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_checksum() {
        let mut snapshot = GpuSnapshot::new("test");
        let data = vec![1, 2, 3, 4, 5];

        snapshot.compute_checksum(&data);
        assert!(!snapshot.checksum.is_empty());
        assert!(snapshot.verify_checksum(&data));
        assert!(!snapshot.verify_checksum(&[1, 2, 3]));
    }

    #[test]
    fn test_manager_creation() {
        let config = SnapshotConfig::default();
        let manager = SnapshotManager::new(config);

        assert!(manager.list_snapshots().is_empty());
    }
}

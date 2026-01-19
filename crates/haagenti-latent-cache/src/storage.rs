//! Storage for cached latents

use crate::{CacheError, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for latent storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage directory
    pub path: PathBuf,
    /// Maximum storage size (bytes)
    pub max_size: u64,
    /// Maximum entries
    pub max_entries: usize,
    /// Compress latents
    pub compress: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./latent_cache"),
            max_size: 5 * 1024 * 1024 * 1024, // 5GB
            max_entries: 10000,
            compress: true,
        }
    }
}

/// A stored latent at a specific step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredLatent {
    /// Step number
    pub step: u32,
    /// Shape of the latent
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Size in bytes
    pub size: usize,
    /// Whether compressed
    pub compressed: bool,
}

/// Latent entry with all checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentEntry {
    /// Entry ID (hash of prompt + seed + params)
    pub id: String,
    /// Original prompt
    pub prompt: String,
    /// Generation seed
    pub seed: u64,
    /// Model ID
    pub model_id: String,
    /// Available latent checkpoints
    pub checkpoints: HashMap<u32, StoredLatent>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last accessed timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u32,
    /// Total size of all checkpoints
    pub total_size: u64,
}

impl LatentEntry {
    /// Create a new entry
    pub fn new(id: String, prompt: String, seed: u64, model_id: String) -> Self {
        let now = now();
        Self {
            id,
            prompt,
            seed,
            model_id,
            checkpoints: HashMap::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            total_size: 0,
        }
    }

    /// Check if has a checkpoint at or before the given step
    pub fn has_checkpoint_before(&self, step: u32) -> Option<u32> {
        self.checkpoints
            .keys()
            .filter(|&&s| s <= step)
            .max()
            .copied()
    }

    /// Update access timestamp
    pub fn touch(&mut self) {
        self.last_accessed = now();
        self.access_count += 1;
    }

    /// Eviction score (lower = evict first)
    pub fn eviction_score(&self) -> f64 {
        let age = (now() - self.last_accessed) as f64;
        let recency = 1.0 / (age + 1.0);
        let frequency = (self.access_count as f64).ln().max(1.0);

        recency * frequency
    }
}

/// Latent storage manager
pub struct LatentStorage {
    config: StorageConfig,
    /// Metadata index
    index: Arc<RwLock<HashMap<String, LatentEntry>>>,
    /// Current storage size
    current_size: Arc<RwLock<u64>>,
}

impl LatentStorage {
    /// Open or create storage
    pub async fn open(config: StorageConfig) -> Result<Self> {
        fs::create_dir_all(&config.path).await?;
        fs::create_dir_all(config.path.join("data")).await?;

        let storage = Self {
            config,
            index: Arc::new(RwLock::new(HashMap::new())),
            current_size: Arc::new(RwLock::new(0)),
        };

        storage.load_index().await?;

        Ok(storage)
    }

    /// Load index from disk
    async fn load_index(&self) -> Result<()> {
        let index_path = self.config.path.join("index.json");

        if !index_path.exists() {
            return Ok(());
        }

        let data = fs::read_to_string(&index_path).await?;
        let entries: Vec<LatentEntry> = serde_json::from_str(&data)
            .map_err(|e| CacheError::Storage(e.to_string()))?;

        let mut index = self.index.write().await;
        let mut total_size = 0u64;

        for entry in entries {
            total_size += entry.total_size;
            index.insert(entry.id.clone(), entry);
        }

        *self.current_size.write().await = total_size;

        info!("Loaded latent storage index: {} entries", index.len());

        Ok(())
    }

    /// Save index to disk
    pub async fn save_index(&self) -> Result<()> {
        let index = self.index.read().await;
        let entries: Vec<&LatentEntry> = index.values().collect();

        let data = serde_json::to_string_pretty(&entries)
            .map_err(|e| CacheError::Storage(e.to_string()))?;

        let index_path = self.config.path.join("index.json");
        let tmp_path = index_path.with_extension("tmp");

        fs::write(&tmp_path, data).await?;
        fs::rename(&tmp_path, &index_path).await?;

        Ok(())
    }

    /// Store a latent checkpoint
    pub async fn store(
        &self,
        entry_id: &str,
        step: u32,
        data: Bytes,
        shape: Vec<usize>,
        dtype: &str,
    ) -> Result<()> {
        let size = data.len();

        // Check capacity
        self.ensure_capacity(size as u64).await?;

        // Write data
        let data_path = self.latent_path(entry_id, step);
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let write_data = if self.config.compress {
            // Simple compression would go here
            data.to_vec()
        } else {
            data.to_vec()
        };

        fs::write(&data_path, &write_data).await?;

        // Update index
        let mut index = self.index.write().await;
        let entry = index.entry(entry_id.to_string()).or_insert_with(|| {
            LatentEntry::new(entry_id.to_string(), String::new(), 0, String::new())
        });

        entry.checkpoints.insert(step, StoredLatent {
            step,
            shape,
            dtype: dtype.to_string(),
            size,
            compressed: self.config.compress,
        });
        entry.total_size += size as u64;

        *self.current_size.write().await += size as u64;

        debug!("Stored latent {} step {} ({} bytes)", entry_id, step, size);

        Ok(())
    }

    /// Load a latent checkpoint
    pub async fn load(&self, entry_id: &str, step: u32) -> Result<Bytes> {
        // Update access
        {
            let mut index = self.index.write().await;
            if let Some(entry) = index.get_mut(entry_id) {
                entry.touch();
            }
        }

        let data_path = self.latent_path(entry_id, step);

        if !data_path.exists() {
            return Err(CacheError::NotFound(format!(
                "{} step {}",
                entry_id, step
            )));
        }

        let data = fs::read(&data_path).await?;

        Ok(Bytes::from(data))
    }

    /// Get entry metadata
    pub async fn get_entry(&self, entry_id: &str) -> Option<LatentEntry> {
        self.index.read().await.get(entry_id).cloned()
    }

    /// Check if entry exists
    pub async fn contains(&self, entry_id: &str) -> bool {
        self.index.read().await.contains_key(entry_id)
    }

    /// Find best checkpoint for a step
    pub async fn find_checkpoint(&self, entry_id: &str, target_step: u32) -> Option<u32> {
        let index = self.index.read().await;
        index.get(entry_id).and_then(|e| e.has_checkpoint_before(target_step))
    }

    /// Ensure we have capacity for new data
    async fn ensure_capacity(&self, needed: u64) -> Result<()> {
        let mut current = *self.current_size.read().await;

        if current + needed <= self.config.max_size {
            return Ok(());
        }

        // Evict until we have space
        let target = self.config.max_size - needed - self.config.max_size / 10;

        let mut index = self.index.write().await;
        let mut entries: Vec<_> = index.values().collect();
        entries.sort_by(|a, b| {
            a.eviction_score()
                .partial_cmp(&b.eviction_score())
                .unwrap()
        });

        let mut to_remove = Vec::new();
        let mut freed = 0u64;

        for entry in entries {
            if current - freed <= target {
                break;
            }

            freed += entry.total_size;
            to_remove.push(entry.id.clone());
        }

        for id in &to_remove {
            if let Some(entry) = index.remove(id) {
                // Delete data files
                for step in entry.checkpoints.keys() {
                    let path = self.latent_path(id, *step);
                    let _ = fs::remove_file(&path).await;
                }
            }
        }

        drop(index);
        *self.current_size.write().await = current - freed;

        if !to_remove.is_empty() {
            info!("Evicted {} latent entries", to_remove.len());
        }

        Ok(())
    }

    /// Get path for a latent file
    fn latent_path(&self, entry_id: &str, step: u32) -> PathBuf {
        self.config
            .path
            .join("data")
            .join(&entry_id[..2])
            .join(format!("{}_{}.bin", entry_id, step))
    }

    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        let index = self.index.read().await;
        let current_size = *self.current_size.read().await;

        let total_checkpoints: usize = index.values().map(|e| e.checkpoints.len()).sum();

        StorageStats {
            entries: index.len(),
            total_checkpoints,
            size_bytes: current_size,
            max_size_bytes: self.config.max_size,
            utilization: current_size as f64 / self.config.max_size as f64,
        }
    }

    /// Clear all storage
    pub async fn clear(&self) -> Result<()> {
        let data_path = self.config.path.join("data");
        if data_path.exists() {
            fs::remove_dir_all(&data_path).await?;
            fs::create_dir_all(&data_path).await?;
        }

        self.index.write().await.clear();
        *self.current_size.write().await = 0;

        Ok(())
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of entries
    pub entries: usize,
    /// Total checkpoints
    pub total_checkpoints: usize,
    /// Current size in bytes
    pub size_bytes: u64,
    /// Maximum size in bytes
    pub max_size_bytes: u64,
    /// Utilization (0.0 - 1.0)
    pub utilization: f64,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_store_and_load() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let storage = LatentStorage::open(config).await.unwrap();

        let data = Bytes::from(vec![1u8; 1024]);
        storage
            .store("test_entry", 5, data.clone(), vec![1, 4, 64, 64], "float16")
            .await
            .unwrap();

        let loaded = storage.load("test_entry", 5).await.unwrap();
        assert_eq!(loaded.len(), 1024);
    }

    #[tokio::test]
    async fn test_find_checkpoint() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let storage = LatentStorage::open(config).await.unwrap();

        // Store checkpoints at steps 5, 10, 15
        for step in [5, 10, 15] {
            storage
                .store(
                    "test_entry",
                    step,
                    Bytes::from(vec![0u8; 100]),
                    vec![1, 4, 64, 64],
                    "float16",
                )
                .await
                .unwrap();
        }

        // Find best checkpoint for step 12
        let checkpoint = storage.find_checkpoint("test_entry", 12).await;
        assert_eq!(checkpoint, Some(10));

        // Find best checkpoint for step 20
        let checkpoint = storage.find_checkpoint("test_entry", 20).await;
        assert_eq!(checkpoint, Some(15));
    }
}

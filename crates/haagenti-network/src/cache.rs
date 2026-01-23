//! Fragment caching with disk and memory tiers

use crate::{NetworkError, Result};
use bytes::Bytes;
use dashmap::DashMap;
use haagenti_fragments::FragmentId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info};

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache directory
    pub path: PathBuf,
    /// Maximum cache size (bytes)
    pub max_size: u64,
    /// Maximum memory cache size (bytes)
    pub max_memory_size: u64,
    /// Eviction threshold (0.0 - 1.0)
    pub eviction_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./fragment_cache"),
            max_size: 10 * 1024 * 1024 * 1024,  // 10GB
            max_memory_size: 512 * 1024 * 1024, // 512MB
            eviction_threshold: 0.9,
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Fragment ID
    pub fragment_id: FragmentId,
    /// Size in bytes
    pub size: u64,
    /// ETag for validation
    pub etag: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<String>,
    /// Cache timestamp
    pub cached_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u32,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(fragment_id: FragmentId, size: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            fragment_id,
            size,
            etag: None,
            last_modified: None,
            cached_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    /// With etag
    pub fn with_etag(mut self, etag: impl Into<String>) -> Self {
        self.etag = Some(etag.into());
        self
    }

    /// With last modified
    pub fn with_last_modified(mut self, last_modified: impl Into<String>) -> Self {
        self.last_modified = Some(last_modified.into());
        self
    }

    /// Update access timestamp
    pub fn touch(&mut self) {
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.access_count += 1;
    }

    /// Compute eviction score (lower = evict first)
    pub fn eviction_score(&self) -> f64 {
        let age = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - self.last_accessed;

        // LRU-K hybrid: consider both recency and frequency
        let recency = 1.0 / (age as f64 + 1.0);
        let frequency = (self.access_count as f64).ln().max(1.0);

        recency * frequency
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total entries
    pub entries: usize,
    /// Disk cache size (bytes)
    pub disk_size: u64,
    /// Memory cache size (bytes)
    pub memory_size: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evictions
    pub evictions: u64,
}

impl CacheStats {
    /// Hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Two-tier fragment cache (memory + disk)
pub struct FragmentCache {
    config: CacheConfig,
    /// Memory cache
    memory: DashMap<FragmentId, Arc<Bytes>>,
    /// Disk cache metadata
    metadata: DashMap<FragmentId, CacheEntry>,
    /// Current disk size
    disk_size: AtomicU64,
    /// Current memory size
    memory_size: AtomicU64,
    /// Statistics
    stats: Arc<CacheStatsInner>,
}

struct CacheStatsInner {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl FragmentCache {
    /// Create or open a cache
    pub async fn open(config: CacheConfig) -> Result<Self> {
        fs::create_dir_all(&config.path).await?;

        let cache = Self {
            config,
            memory: DashMap::new(),
            metadata: DashMap::new(),
            disk_size: AtomicU64::new(0),
            memory_size: AtomicU64::new(0),
            stats: Arc::new(CacheStatsInner {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
            }),
        };

        // Load existing metadata
        cache.load_metadata().await?;

        Ok(cache)
    }

    /// Load cache metadata from disk
    async fn load_metadata(&self) -> Result<()> {
        let meta_path = self.config.path.join("metadata.bin");

        if !meta_path.exists() {
            return Ok(());
        }

        let data = fs::read(&meta_path).await?;
        let entries: Vec<CacheEntry> =
            bincode::deserialize(&data).map_err(|e| NetworkError::Cache(e.to_string()))?;

        let mut total_size = 0u64;
        for entry in entries {
            total_size += entry.size;
            self.metadata.insert(entry.fragment_id, entry);
        }

        self.disk_size.store(total_size, Ordering::Relaxed);
        info!(
            "Loaded cache metadata: {} entries, {} bytes",
            self.metadata.len(),
            total_size
        );

        Ok(())
    }

    /// Save cache metadata to disk
    async fn save_metadata(&self) -> Result<()> {
        let entries: Vec<CacheEntry> = self.metadata.iter().map(|e| e.value().clone()).collect();
        let data = bincode::serialize(&entries).map_err(|e| NetworkError::Cache(e.to_string()))?;

        let meta_path = self.config.path.join("metadata.bin");
        let tmp_path = meta_path.with_extension("tmp");

        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &meta_path).await?;

        Ok(())
    }

    /// Get a fragment from cache
    pub async fn get(&self, fragment_id: &FragmentId) -> Option<Bytes> {
        // Check memory cache first
        if let Some(data) = self.memory.get(fragment_id) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            if let Some(mut entry) = self.metadata.get_mut(fragment_id) {
                entry.touch();
            }
            return Some(data.as_ref().clone());
        }

        // Check disk cache
        if let Some(mut entry) = self.metadata.get_mut(fragment_id) {
            let path = self.fragment_path(fragment_id);
            if let Ok(data) = fs::read(&path).await {
                let bytes = Bytes::from(data);
                entry.touch();

                // Promote to memory cache if space available
                self.promote_to_memory(fragment_id, bytes.clone());

                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(bytes);
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Put a fragment into cache
    pub async fn put(&self, fragment_id: FragmentId, data: Bytes, entry: CacheEntry) -> Result<()> {
        let size = data.len() as u64;

        // Evict if needed
        self.maybe_evict(size).await?;

        // Write to disk
        let path = self.fragment_path(&fragment_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::write(&path, &data).await?;

        // Update metadata
        self.metadata.insert(fragment_id, entry);
        self.disk_size.fetch_add(size, Ordering::Relaxed);

        // Add to memory cache if space available
        self.promote_to_memory(&fragment_id, data);

        Ok(())
    }

    /// Promote to memory cache
    fn promote_to_memory(&self, fragment_id: &FragmentId, data: Bytes) {
        let size = data.len() as u64;
        let current = self.memory_size.load(Ordering::Relaxed);

        if current + size <= self.config.max_memory_size {
            self.memory.insert(*fragment_id, Arc::new(data));
            self.memory_size.fetch_add(size, Ordering::Relaxed);
        }
    }

    /// Maybe evict entries
    async fn maybe_evict(&self, needed_size: u64) -> Result<()> {
        let current = self.disk_size.load(Ordering::Relaxed);
        let threshold =
            (self.config.max_size as f64 * self.config.eviction_threshold as f64) as u64;

        if current + needed_size < threshold {
            return Ok(());
        }

        // Collect entries sorted by eviction score
        let mut entries: Vec<_> = self.metadata.iter().map(|e| e.value().clone()).collect();
        entries.sort_by(|a, b| a.eviction_score().partial_cmp(&b.eviction_score()).unwrap());

        // Evict until we have enough space
        let target = self.config.max_size - needed_size - (self.config.max_size / 10); // 10% buffer
        let mut freed = 0u64;

        for entry in entries {
            if current - freed <= target {
                break;
            }

            if self.evict(&entry.fragment_id).await.is_ok() {
                freed += entry.size;
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        debug!("Evicted {} bytes from cache", freed);
        Ok(())
    }

    /// Evict a single entry
    async fn evict(&self, fragment_id: &FragmentId) -> Result<()> {
        // Remove from memory
        if let Some((_, data)) = self.memory.remove(fragment_id) {
            self.memory_size
                .fetch_sub(data.len() as u64, Ordering::Relaxed);
        }

        // Remove from disk
        if let Some((_, entry)) = self.metadata.remove(fragment_id) {
            let path = self.fragment_path(fragment_id);
            if path.exists() {
                fs::remove_file(&path).await?;
            }
            self.disk_size.fetch_sub(entry.size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Check if fragment exists in cache
    pub fn contains(&self, fragment_id: &FragmentId) -> bool {
        self.metadata.contains_key(fragment_id)
    }

    /// Get cache entry metadata
    pub fn get_entry(&self, fragment_id: &FragmentId) -> Option<CacheEntry> {
        self.metadata.get(fragment_id).map(|e| e.value().clone())
    }

    /// Validate cache entry against remote
    pub fn needs_revalidation(&self, fragment_id: &FragmentId, etag: Option<&str>) -> bool {
        if let Some(entry) = self.metadata.get(fragment_id) {
            if let (Some(cached_etag), Some(remote_etag)) = (&entry.etag, etag) {
                return cached_etag != remote_etag;
            }
        }
        true
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.metadata.len(),
            disk_size: self.disk_size.load(Ordering::Relaxed),
            memory_size: self.memory_size.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
        }
    }

    /// Clear all cached data
    pub async fn clear(&self) -> Result<()> {
        self.memory.clear();
        self.metadata.clear();
        self.disk_size.store(0, Ordering::Relaxed);
        self.memory_size.store(0, Ordering::Relaxed);

        // Remove all files
        let fragments_dir = self.config.path.join("fragments");
        if fragments_dir.exists() {
            fs::remove_dir_all(&fragments_dir).await?;
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Persist cache state
    pub async fn sync(&self) -> Result<()> {
        self.save_metadata().await
    }

    /// Get fragment path
    fn fragment_path(&self, id: &FragmentId) -> PathBuf {
        let hex = id.to_hex();
        self.config
            .path
            .join("fragments")
            .join(&hex[..2])
            .join(format!("{}.bin", hex))
    }
}

impl Drop for FragmentCache {
    fn drop(&mut self) {
        // Best effort sync on drop
        let meta = self
            .metadata
            .iter()
            .map(|e| e.value().clone())
            .collect::<Vec<_>>();
        if let Ok(data) = bincode::serialize(&meta) {
            let _ = std::fs::write(self.config.path.join("metadata.bin"), data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_cache_put_get() {
        let dir = tempdir().unwrap();
        let config = CacheConfig {
            path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let cache = FragmentCache::open(config).await.unwrap();

        let fragment_id = FragmentId::new([1; 16]);
        let data = Bytes::from(vec![42u8; 1024]);
        let entry = CacheEntry::new(fragment_id, 1024);

        cache.put(fragment_id, data.clone(), entry).await.unwrap();

        let retrieved = cache.get(&fragment_id).await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let dir = tempdir().unwrap();
        let config = CacheConfig {
            path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let cache = FragmentCache::open(config).await.unwrap();

        let fragment_id = FragmentId::new([99; 16]);
        assert!(cache.get(&fragment_id).await.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }
}

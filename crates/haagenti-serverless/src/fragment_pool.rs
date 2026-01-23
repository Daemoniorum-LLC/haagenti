//! Pre-warmed fragment pools for fast cold starts

use crate::{Result, ServerlessError};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum pool size in bytes
    pub max_size: u64,
    /// Maximum fragments per pool
    pub max_fragments: usize,
    /// Fragment expiry time in seconds
    pub expiry_seconds: u64,
    /// Enable deduplication
    pub dedup_enabled: bool,
    /// Pre-warm common fragments
    pub prewarm_common: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 512 * 1024 * 1024, // 512MB
            max_fragments: 1000,
            expiry_seconds: 300, // 5 minutes
            dedup_enabled: true,
            prewarm_common: true,
        }
    }
}

/// A pooled fragment
#[derive(Debug)]
pub struct PooledFragment {
    /// Fragment ID
    pub id: u64,
    /// Fragment key
    pub key: String,
    /// Fragment data
    pub data: Arc<Vec<u8>>,
    /// Content hash for deduplication
    pub hash: u64,
    /// Size in bytes
    pub size: u64,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: AtomicU64,
}

impl Clone for PooledFragment {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            key: self.key.clone(),
            data: self.data.clone(),
            hash: self.hash,
            size: self.size,
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
        }
    }
}

impl PooledFragment {
    /// Create new fragment
    pub fn new(key: impl Into<String>, data: Vec<u8>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        let hash = Self::compute_hash(&data);
        let size = data.len() as u64;
        let now = Instant::now();

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            key: key.into(),
            data: Arc::new(data),
            hash,
            size,
            created_at: now,
            last_accessed: now,
            access_count: AtomicU64::new(1),
        }
    }

    /// Compute content hash
    fn compute_hash(data: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Record access
    pub fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get access count
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Check if expired
    pub fn is_expired(&self, expiry_seconds: u64) -> bool {
        self.created_at.elapsed().as_secs() > expiry_seconds
    }
}

/// Fragment pool for pre-warmed fragments
#[derive(Debug)]
pub struct FragmentPool {
    /// Configuration
    config: PoolConfig,
    /// Fragments by key
    fragments: DashMap<String, PooledFragment>,
    /// Hash to key mapping for deduplication
    hash_to_key: DashMap<u64, String>,
    /// Total size
    total_size: AtomicU64,
    /// Pool statistics
    stats: PoolStats,
}

/// Pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Cache hits
    hits: AtomicU64,
    /// Cache misses
    misses: AtomicU64,
    /// Evictions
    evictions: AtomicU64,
    /// Dedup savings in bytes
    dedup_savings: AtomicU64,
}

impl FragmentPool {
    /// Create new pool
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            fragments: DashMap::new(),
            hash_to_key: DashMap::new(),
            total_size: AtomicU64::new(0),
            stats: PoolStats::default(),
        }
    }

    /// Get or create fragment
    pub fn get_or_create<F>(&self, key: &str, create_fn: F) -> Result<Arc<Vec<u8>>>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        // Check for existing fragment
        if let Some(fragment) = self.fragments.get(key) {
            if !fragment.is_expired(self.config.expiry_seconds) {
                fragment.record_access();
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::clone(&fragment.data));
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);

        // Create new fragment
        let data = create_fn()?;

        // Check for dedup
        if self.config.dedup_enabled {
            let hash = PooledFragment::compute_hash(&data);
            if let Some(existing_key) = self.hash_to_key.get(&hash) {
                if let Some(existing) = self.fragments.get(existing_key.value()) {
                    self.stats
                        .dedup_savings
                        .fetch_add(data.len() as u64, Ordering::Relaxed);
                    return Ok(Arc::clone(&existing.data));
                }
            }
        }

        // Ensure space
        self.ensure_space(data.len() as u64)?;

        // Insert fragment
        let fragment = PooledFragment::new(key, data);
        let data_ref = Arc::clone(&fragment.data);
        let hash = fragment.hash;

        self.total_size.fetch_add(fragment.size, Ordering::SeqCst);
        self.fragments.insert(key.to_string(), fragment);

        if self.config.dedup_enabled {
            self.hash_to_key.insert(hash, key.to_string());
        }

        Ok(data_ref)
    }

    /// Get fragment by key
    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        if let Some(fragment) = self.fragments.get(key) {
            if !fragment.is_expired(self.config.expiry_seconds) {
                fragment.record_access();
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(Arc::clone(&fragment.data));
            }
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert fragment directly
    pub fn insert(&self, key: impl Into<String>, data: Vec<u8>) -> Result<()> {
        let size = data.len() as u64;
        self.ensure_space(size)?;

        let fragment = PooledFragment::new(key, data);
        let hash = fragment.hash;
        let key = fragment.key.clone();

        self.total_size.fetch_add(fragment.size, Ordering::SeqCst);
        self.fragments.insert(key.clone(), fragment);

        if self.config.dedup_enabled {
            self.hash_to_key.insert(hash, key);
        }

        Ok(())
    }

    /// Remove fragment
    pub fn remove(&self, key: &str) -> Option<PooledFragment> {
        if let Some((_, fragment)) = self.fragments.remove(key) {
            self.total_size.fetch_sub(fragment.size, Ordering::SeqCst);
            self.hash_to_key.remove(&fragment.hash);
            Some(fragment)
        } else {
            None
        }
    }

    /// Ensure space for new fragment
    fn ensure_space(&self, required: u64) -> Result<()> {
        // Check fragment count
        while self.fragments.len() >= self.config.max_fragments {
            self.evict_one()?;
        }

        // Check size
        while self.total_size.load(Ordering::SeqCst) + required > self.config.max_size {
            self.evict_one()?;
        }

        Ok(())
    }

    /// Evict one fragment (LRU)
    fn evict_one(&self) -> Result<()> {
        let mut oldest_key = None;
        let mut oldest_time = None;

        for entry in self.fragments.iter() {
            let time = entry.value().last_accessed;
            if oldest_time.is_none() || time < oldest_time.unwrap() {
                oldest_key = Some(entry.key().clone());
                oldest_time = Some(time);
            }
        }

        if let Some(key) = oldest_key {
            self.remove(&key);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(ServerlessError::PoolError("No fragments to evict".into()))
        }
    }

    /// Clear expired fragments
    pub fn clear_expired(&self) -> usize {
        let mut expired = Vec::new();

        for entry in self.fragments.iter() {
            if entry.value().is_expired(self.config.expiry_seconds) {
                expired.push(entry.key().clone());
            }
        }

        for key in &expired {
            self.remove(key);
        }

        expired.len()
    }

    /// Clear all fragments
    pub fn clear(&self) {
        self.fragments.clear();
        self.hash_to_key.clear();
        self.total_size.store(0, Ordering::SeqCst);
    }

    /// Total size
    pub fn total_size(&self) -> u64 {
        self.total_size.load(Ordering::SeqCst)
    }

    /// Fragment count
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.stats.hits.load(Ordering::Relaxed),
            self.stats.misses.load(Ordering::Relaxed),
            self.stats.evictions.load(Ordering::Relaxed),
            self.stats.dedup_savings.load(Ordering::Relaxed),
        )
    }

    /// Hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

/// Pre-warmer for common fragments
#[derive(Debug)]
pub struct FragmentPrewarmer {
    /// Pool to pre-warm
    pool: Arc<FragmentPool>,
    /// Pre-warm list
    prewarm_list: Vec<PrewarmEntry>,
}

/// Pre-warm entry (internal to FragmentPrewarmer)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PrewarmEntry {
    key: String,
    priority: u32,
    size_hint: u64,
}

impl FragmentPrewarmer {
    /// Create new pre-warmer
    pub fn new(pool: Arc<FragmentPool>) -> Self {
        Self {
            pool,
            prewarm_list: Vec::new(),
        }
    }

    /// Add fragment to pre-warm list
    pub fn add(&mut self, key: impl Into<String>, priority: u32, size_hint: u64) {
        self.prewarm_list.push(PrewarmEntry {
            key: key.into(),
            priority,
            size_hint,
        });
    }

    /// Sort by priority
    pub fn sort(&mut self) {
        self.prewarm_list
            .sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Pre-warm fragments
    pub async fn prewarm<F>(&self, loader: F) -> Result<usize>
    where
        F: Fn(&str) -> Result<Vec<u8>>,
    {
        let mut count = 0;

        for entry in &self.prewarm_list {
            if self.pool.get(&entry.key).is_none() {
                let data = loader(&entry.key)?;
                self.pool.insert(entry.key.clone(), data)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Get pre-warm list size
    pub fn list_size(&self) -> usize {
        self.prewarm_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_size, 512 * 1024 * 1024);
        assert!(config.dedup_enabled);
    }

    #[test]
    fn test_pooled_fragment() {
        let fragment = PooledFragment::new("test", vec![1, 2, 3, 4]);

        assert_eq!(fragment.key, "test");
        assert_eq!(fragment.size, 4);
        assert_eq!(fragment.get_access_count(), 1);

        fragment.record_access();
        assert_eq!(fragment.get_access_count(), 2);
    }

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig::default();
        let pool = FragmentPool::new(config);

        assert!(pool.is_empty());
        assert_eq!(pool.total_size(), 0);
    }

    #[test]
    fn test_pool_insert_get() {
        let config = PoolConfig::default();
        let pool = FragmentPool::new(config);

        pool.insert("key1", vec![1, 2, 3, 4]).unwrap();
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.total_size(), 4);

        let data = pool.get("key1").unwrap();
        assert_eq!(*data, vec![1, 2, 3, 4]);

        assert!(pool.get("nonexistent").is_none());
    }

    #[test]
    fn test_pool_dedup() {
        let config = PoolConfig {
            dedup_enabled: true,
            ..Default::default()
        };
        let pool = FragmentPool::new(config);

        // Insert same data with different keys
        let data = vec![1, 2, 3, 4];
        pool.insert("key1", data.clone()).unwrap();

        let result = pool.get_or_create("key2", || Ok(data.clone()));
        assert!(result.is_ok());

        // Should have dedup savings
        let (_, _, _, dedup) = pool.stats();
        assert!(dedup > 0);
    }

    #[test]
    fn test_pool_eviction() {
        let config = PoolConfig {
            max_fragments: 2,
            ..Default::default()
        };
        let pool = FragmentPool::new(config);

        pool.insert("key1", vec![1]).unwrap();
        pool.insert("key2", vec![2]).unwrap();
        pool.insert("key3", vec![3]).unwrap();

        // Should have evicted one
        assert_eq!(pool.len(), 2);

        let (_, _, evictions, _) = pool.stats();
        assert_eq!(evictions, 1);
    }

    #[test]
    fn test_prewarmer() {
        let config = PoolConfig::default();
        let pool = Arc::new(FragmentPool::new(config));
        let mut prewarmer = FragmentPrewarmer::new(Arc::clone(&pool));

        prewarmer.add("layer0", 10, 1024);
        prewarmer.add("layer1", 5, 1024);
        prewarmer.sort();

        assert_eq!(prewarmer.list_size(), 2);
    }
}

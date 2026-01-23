//! Fragment caching for WebGPU inference

use crate::buffer::{BufferPool, BufferUsage, GpuBuffer};
use crate::{Result, WebGpuError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size: u64,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Entry TTL (time-to-live)
    pub ttl_seconds: u64,
    /// Enable LRU eviction
    pub lru_eviction: bool,
    /// Preload common fragments
    pub preload_common: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 256 * 1024 * 1024, // 256MB
            max_entries: 1000,
            ttl_seconds: 300, // 5 minutes
            lru_eviction: true,
            preload_common: true,
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Buffer ID
    pub buffer_id: u64,
    /// Entry key
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Creation time
    pub created_at: Instant,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Fragment hash for deduplication
    pub content_hash: u64,
}

impl CacheEntry {
    /// Create new cache entry
    pub fn new(buffer_id: u64, key: impl Into<String>, size: u64, content_hash: u64) -> Self {
        let now = Instant::now();
        Self {
            buffer_id,
            key: key.into(),
            size,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            content_hash,
        }
    }

    /// Check if entry is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Fragment cache for WebGPU buffers
#[derive(Debug)]
pub struct FragmentCache {
    /// Configuration
    config: CacheConfig,
    /// Cache entries by key
    entries: HashMap<String, CacheEntry>,
    /// Content hash to key mapping for deduplication
    hash_to_key: HashMap<u64, String>,
    /// Buffer pool for allocations
    buffer_pool: BufferPool,
    /// Total cached size
    total_size: u64,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Bytes saved by deduplication
    pub dedup_savings: u64,
    /// Total bytes cached
    pub total_bytes_cached: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl FragmentCache {
    /// Create new fragment cache
    pub fn new(config: CacheConfig) -> Self {
        let buffer_pool = BufferPool::new(config.max_size);

        Self {
            config,
            entries: HashMap::new(),
            hash_to_key: HashMap::new(),
            buffer_pool,
            total_size: 0,
            stats: CacheStats::default(),
        }
    }

    /// Get or create a cached fragment
    pub fn get_or_create<F>(
        &mut self,
        key: &str,
        content_hash: u64,
        size: u64,
        create_fn: F,
    ) -> Result<&CacheEntry>
    where
        F: FnOnce(&mut GpuBuffer) -> Result<()>,
    {
        // Check for existing entry
        if let Some(entry) = self.entries.get_mut(key) {
            if !entry.is_expired(Duration::from_secs(self.config.ttl_seconds)) {
                entry.record_access();
                self.stats.hits += 1;
                return Ok(self.entries.get(key).unwrap());
            }
            // Entry expired, will be replaced
        }

        // Check for deduplicated entry
        if let Some(existing_key) = self.hash_to_key.get(&content_hash).cloned() {
            if let Some(existing_entry) = self.entries.get(&existing_key) {
                // Create alias to existing buffer
                let entry = CacheEntry::new(
                    existing_entry.buffer_id,
                    key,
                    existing_entry.size,
                    content_hash,
                );
                self.entries.insert(key.to_string(), entry);
                self.stats.hits += 1;
                self.stats.dedup_savings += size;
                return Ok(self.entries.get(key).unwrap());
            }
        }

        self.stats.misses += 1;

        // Evict if necessary
        self.ensure_space(size)?;

        // Allocate new buffer
        let mut buffer = self.buffer_pool.allocate(
            size,
            vec![
                BufferUsage::Storage,
                BufferUsage::CopySrc,
                BufferUsage::CopyDst,
            ],
            key,
        )?;

        // Initialize buffer content
        create_fn(&mut buffer)?;

        let buffer_id = buffer.id;

        // Create entry
        let entry = CacheEntry::new(buffer_id, key, size, content_hash);
        self.entries.insert(key.to_string(), entry);
        self.hash_to_key.insert(content_hash, key.to_string());
        self.total_size += size;
        self.stats.total_bytes_cached += size;

        Ok(self.entries.get(key).unwrap())
    }

    /// Get cached entry
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            if !entry.is_expired(Duration::from_secs(self.config.ttl_seconds)) {
                entry.record_access();
                self.stats.hits += 1;
                return Some(self.entries.get(key).unwrap());
            }
        }
        self.stats.misses += 1;
        None
    }

    /// Check if key exists and is valid
    pub fn contains(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .is_some_and(|e| !e.is_expired(Duration::from_secs(self.config.ttl_seconds)))
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.hash_to_key.remove(&entry.content_hash);
            self.buffer_pool.release(entry.buffer_id);
            self.total_size = self.total_size.saturating_sub(entry.size);
            Some(entry)
        } else {
            None
        }
    }

    /// Ensure space for new entry
    fn ensure_space(&mut self, required: u64) -> Result<()> {
        // Check entry count limit
        while self.entries.len() >= self.config.max_entries {
            self.evict_one()?;
        }

        // Check size limit
        while self.total_size + required > self.config.max_size {
            self.evict_one()?;
        }

        Ok(())
    }

    /// Evict one entry (LRU or oldest)
    fn evict_one(&mut self) -> Result<()> {
        let key_to_evict = if self.config.lru_eviction {
            // Find least recently used
            self.entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone())
        } else {
            // Find oldest
            self.entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone())
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
            self.stats.evictions += 1;
            Ok(())
        } else {
            Err(WebGpuError::CacheError("No entries to evict".into()))
        }
    }

    /// Clear all expired entries
    pub fn clear_expired(&mut self) -> usize {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl))
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.remove(&key);
        }

        count
    }

    /// Clear entire cache
    pub fn clear(&mut self) {
        for (_, entry) in self.entries.drain() {
            self.buffer_pool.release(entry.buffer_id);
        }
        self.hash_to_key.clear();
        self.total_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get current cache size
    pub fn size(&self) -> u64 {
        self.total_size
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

/// Pre-warmed cache for common model fragments
#[allow(dead_code)]
#[derive(Debug)]
pub struct PrewarmedCache {
    /// Fragment cache
    cache: FragmentCache,
    /// Pre-warm list (key, size, priority)
    prewarm_list: Vec<(String, u64, u32)>,
}

#[allow(dead_code)]
impl PrewarmedCache {
    /// Create new pre-warmed cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: FragmentCache::new(config),
            prewarm_list: Vec::new(),
        }
    }

    /// Add fragment to pre-warm list
    pub fn add_prewarm(&mut self, key: impl Into<String>, size: u64, priority: u32) {
        self.prewarm_list.push((key.into(), size, priority));
    }

    /// Sort pre-warm list by priority
    pub fn sort_prewarm_list(&mut self) {
        self.prewarm_list.sort_by(|a, b| b.2.cmp(&a.2));
    }

    /// Get next fragment to pre-warm
    pub fn next_prewarm(&self) -> Option<&(String, u64, u32)> {
        self.prewarm_list.first()
    }

    /// Mark fragment as pre-warmed
    pub fn mark_prewarmed(&mut self, key: &str) {
        self.prewarm_list.retain(|(k, _, _)| k != key);
    }

    /// Access underlying cache
    pub fn cache(&self) -> &FragmentCache {
        &self.cache
    }

    /// Access underlying cache mutably
    pub fn cache_mut(&mut self) -> &mut FragmentCache {
        &mut self.cache
    }
}

/// Layer-specific cache for transformer models
#[allow(dead_code)]
#[derive(Debug)]
pub struct LayerCache {
    /// Per-layer caches
    layers: HashMap<usize, FragmentCache>,
    /// Global configuration
    config: CacheConfig,
    /// Number of layers
    num_layers: usize,
}

#[allow(dead_code)]
impl LayerCache {
    /// Create new layer cache
    pub fn new(num_layers: usize, config: CacheConfig) -> Self {
        let mut layers = HashMap::new();

        // Create per-layer budget
        let per_layer_size = config.max_size / num_layers as u64;
        let per_layer_entries = config.max_entries / num_layers;

        for i in 0..num_layers {
            let layer_config = CacheConfig {
                max_size: per_layer_size,
                max_entries: per_layer_entries,
                ..config.clone()
            };
            layers.insert(i, FragmentCache::new(layer_config));
        }

        Self {
            layers,
            config,
            num_layers,
        }
    }

    /// Get cache for specific layer
    pub fn layer(&mut self, layer_idx: usize) -> Option<&mut FragmentCache> {
        self.layers.get_mut(&layer_idx)
    }

    /// Get total stats across all layers
    pub fn total_stats(&self) -> CacheStats {
        let mut total = CacheStats::default();

        for cache in self.layers.values() {
            let stats = cache.stats();
            total.hits += stats.hits;
            total.misses += stats.misses;
            total.evictions += stats.evictions;
            total.dedup_savings += stats.dedup_savings;
            total.total_bytes_cached += stats.total_bytes_cached;
        }

        total
    }

    /// Get total cached size
    pub fn total_size(&self) -> u64 {
        self.layers.values().map(|c| c.size()).sum()
    }

    /// Clear all layer caches
    pub fn clear_all(&mut self) {
        for cache in self.layers.values_mut() {
            cache.clear();
        }
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache = FragmentCache::new(config);

        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_entry() {
        let entry = CacheEntry::new(1, "test", 1024, 12345);

        assert_eq!(entry.buffer_id, 1);
        assert_eq!(entry.key, "test");
        assert_eq!(entry.size, 1024);
        assert_eq!(entry.access_count, 1);
        assert!(!entry.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.hits = 80;
        stats.misses = 20;

        assert!((stats.hit_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_cache() {
        let config = CacheConfig {
            max_size: 10 * 1024 * 1024, // 10MB
            max_entries: 100,
            ..Default::default()
        };

        let mut cache = LayerCache::new(12, config);

        assert_eq!(cache.num_layers(), 12);
        assert!(cache.layer(0).is_some());
        assert!(cache.layer(11).is_some());
        assert!(cache.layer(12).is_none());
    }

    #[test]
    fn test_prewarmed_cache() {
        let config = CacheConfig::default();
        let mut cache = PrewarmedCache::new(config);

        cache.add_prewarm("layer0_weights", 1024, 10);
        cache.add_prewarm("layer1_weights", 1024, 5);
        cache.add_prewarm("embeddings", 2048, 20);

        cache.sort_prewarm_list();

        let next = cache.next_prewarm().unwrap();
        assert_eq!(next.0, "embeddings");
        assert_eq!(next.2, 20);

        cache.mark_prewarmed("embeddings");
        let next = cache.next_prewarm().unwrap();
        assert_eq!(next.0, "layer0_weights");
    }
}

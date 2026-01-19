//! Main latent cache implementation

use arcanum_primitives::prelude::Blake3;
use crate::{
    CacheError, ClipEmbedding, DivergencePoint, DivergencePredictor,
    EmbeddingProvider, HnswConfig, LatentStorage, Result, SearchResult,
    SimilaritySearch, StorageConfig,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for the latent cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Storage configuration
    pub storage: StorageConfig,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Minimum similarity for cache hit
    pub min_similarity: f32,
    /// Total denoising steps
    pub total_steps: u32,
    /// Steps at which to cache latents
    pub checkpoint_steps: Vec<u32>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            hnsw: HnswConfig::default(),
            min_similarity: 0.85,
            total_steps: 20,
            checkpoint_steps: vec![5, 10, 15, 18],
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total lookups
    pub lookups: u64,
    /// Cache hits
    pub hits: u64,
    /// Steps saved by cache hits
    pub steps_saved: u64,
    /// Entries in cache
    pub entries: usize,
    /// Storage size
    pub storage_bytes: u64,
}

impl CacheStats {
    /// Hit rate
    pub fn hit_rate(&self) -> f32 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f32 / self.lookups as f32
        }
    }

    /// Average steps saved per hit
    pub fn avg_steps_saved(&self) -> f32 {
        if self.hits == 0 {
            0.0
        } else {
            self.steps_saved as f32 / self.hits as f32
        }
    }
}

/// Cache entry with full information
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Entry ID
    pub id: String,
    /// Original prompt
    pub prompt: String,
    /// Similarity to query
    pub similarity: f32,
    /// Recommended divergence point
    pub divergence: DivergencePoint,
    /// Available checkpoint steps
    pub available_steps: Vec<u32>,
}

/// Main latent cache
pub struct LatentCache<E: EmbeddingProvider> {
    config: CacheConfig,
    embedding_provider: Arc<E>,
    similarity_search: SimilaritySearch,
    storage: LatentStorage,
    divergence_predictor: DivergencePredictor,
    stats: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
    steps_saved: std::sync::atomic::AtomicU64,
}

impl<E: EmbeddingProvider> LatentCache<E> {
    /// Create a new latent cache
    pub async fn new(config: CacheConfig, embedding_provider: Arc<E>) -> Result<Self> {
        let storage = LatentStorage::open(config.storage.clone()).await?;
        let similarity_search = SimilaritySearch::new(config.hnsw.clone());
        let divergence_predictor =
            DivergencePredictor::new(config.total_steps, config.min_similarity);

        Ok(Self {
            config,
            embedding_provider,
            similarity_search,
            storage,
            divergence_predictor,
            stats: std::sync::atomic::AtomicU64::new(0),
            hits: std::sync::atomic::AtomicU64::new(0),
            steps_saved: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Find a cached latent for a prompt
    pub async fn find(&self, prompt: &str) -> Result<Option<CacheEntry>> {
        self.stats.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Get embedding for the prompt
        let embedding = self.embedding_provider.embed(prompt).await?;

        // Search for similar
        let result = self
            .similarity_search
            .find_similar(&embedding, self.config.min_similarity);

        if let Some(search_result) = result {
            // Get entry from storage
            if let Some(entry) = self.storage.get_entry(&search_result.id).await {
                // Predict divergence point
                if let Some(divergence) = self.divergence_predictor.predict(search_result.similarity)
                {
                    self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.steps_saved.fetch_add(
                        divergence.steps_saved as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );

                    let available_steps: Vec<u32> =
                        entry.checkpoints.keys().copied().collect();

                    return Ok(Some(CacheEntry {
                        id: entry.id,
                        prompt: entry.prompt,
                        similarity: search_result.similarity,
                        divergence,
                        available_steps,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Store latents for a generation
    pub async fn store(
        &self,
        prompt: &str,
        seed: u64,
        model_id: &str,
        latents: Vec<(u32, Bytes, Vec<usize>, String)>,
    ) -> Result<String> {
        // Generate entry ID
        let entry_id = self.generate_entry_id(prompt, seed, model_id);

        // Get and store embedding
        let embedding = self.embedding_provider.embed(prompt).await?;
        self.similarity_search
            .insert(entry_id.clone(), embedding);

        // Store each latent checkpoint
        for (step, data, shape, dtype) in latents {
            self.storage
                .store(&entry_id, step, data, shape, &dtype)
                .await?;
        }

        // Update entry metadata
        {
            if let Some(mut entry) = self.storage.get_entry(&entry_id).await {
                entry.prompt = prompt.to_string();
                entry.seed = seed;
                entry.model_id = model_id.to_string();
            }
        }

        info!("Cached latents for '{}' as {}", prompt, entry_id);

        Ok(entry_id)
    }

    /// Load a cached latent
    pub async fn load(&self, entry_id: &str, step: u32) -> Result<Bytes> {
        // Find best available checkpoint
        let checkpoint = self
            .storage
            .find_checkpoint(entry_id, step)
            .await
            .ok_or_else(|| {
                CacheError::NotFound(format!("No checkpoint at or before step {}", step))
            })?;

        self.storage.load(entry_id, checkpoint).await
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let storage_stats = self.storage.stats().await;

        CacheStats {
            lookups: self.stats.load(std::sync::atomic::Ordering::Relaxed),
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            steps_saved: self.steps_saved.load(std::sync::atomic::Ordering::Relaxed),
            entries: storage_stats.entries,
            storage_bytes: storage_stats.size_bytes,
        }
    }

    /// Clear the cache
    pub async fn clear(&self) -> Result<()> {
        self.similarity_search.clear();
        self.storage.clear().await?;

        self.stats.store(0, std::sync::atomic::Ordering::Relaxed);
        self.hits.store(0, std::sync::atomic::Ordering::Relaxed);
        self.steps_saved.store(0, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Get checkpoint steps configuration
    pub fn checkpoint_steps(&self) -> &[u32] {
        &self.config.checkpoint_steps
    }

    /// Generate entry ID
    fn generate_entry_id(&self, prompt: &str, seed: u64, model_id: &str) -> String {
        let input = format!("{}:{}:{}", prompt, seed, model_id);
        let hash = Blake3::hash(input.as_bytes());
        // Convert first 8 bytes to hex (16 chars)
        hash[..8].iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingProvider;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_cache_store_and_find() {
        let dir = tempdir().unwrap();
        let config = CacheConfig {
            storage: StorageConfig {
                path: dir.path().to_path_buf(),
                ..Default::default()
            },
            ..Default::default()
        };

        let provider = Arc::new(MockEmbeddingProvider::new(768));
        let cache = LatentCache::new(config, provider).await.unwrap();

        // Store some latents
        let latents = vec![
            (5, Bytes::from(vec![1u8; 1024]), vec![1, 4, 64, 64], "float16".to_string()),
            (10, Bytes::from(vec![2u8; 1024]), vec![1, 4, 64, 64], "float16".to_string()),
        ];

        cache
            .store("a cat sitting on a windowsill", 42, "sdxl", latents)
            .await
            .unwrap();

        // Find with exact same prompt
        let result = cache.find("a cat sitting on a windowsill").await.unwrap();
        assert!(result.is_some());

        let entry = result.unwrap();
        assert!(entry.similarity > 0.99);
        assert!(entry.available_steps.contains(&5));
        assert!(entry.available_steps.contains(&10));
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let dir = tempdir().unwrap();
        let config = CacheConfig {
            storage: StorageConfig {
                path: dir.path().to_path_buf(),
                ..Default::default()
            },
            min_similarity: 0.9,
            ..Default::default()
        };

        let provider = Arc::new(MockEmbeddingProvider::new(768));
        let cache = LatentCache::new(config, provider).await.unwrap();

        // Find with no entries
        let result = cache.find("completely different prompt").await.unwrap();
        assert!(result.is_none());
    }
}

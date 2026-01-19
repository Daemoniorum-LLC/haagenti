//! Embedding providers for semantic similarity

use arcanum_primitives::prelude::Blake3;
use crate::{CacheError, Result};
use serde::{Deserialize, Serialize};

/// CLIP embedding (768 dimensions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipEmbedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Model used to create embedding
    pub model: String,
}

impl ClipEmbedding {
    /// Create from a vector
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            model: "clip-vit-large-patch14".to_string(),
        }
    }

    /// Cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &ClipEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let mag_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }

    /// Euclidean distance
    pub fn euclidean_distance(&self, other: &ClipEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dimension of the embedding
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Trait for embedding providers
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a text prompt
    async fn embed(&self, text: &str) -> Result<ClipEmbedding>;

    /// Embed multiple prompts (batched)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<ClipEmbedding>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;
}

/// Mock embedding provider for testing
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    /// Create a new mock provider
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Create a deterministic embedding from text
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let hash = Blake3::hash(text.as_bytes());
        let bytes = &hash;

        (0..self.dimension)
            .map(|i| {
                let byte = bytes[i % 32];
                (byte as f32 / 255.0) * 2.0 - 1.0
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<ClipEmbedding> {
        Ok(ClipEmbedding::new(self.hash_embed(text)))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<ClipEmbedding>> {
        Ok(texts.iter().map(|t| ClipEmbedding::new(self.hash_embed(t))).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Cached embedding provider (wraps another provider)
pub struct CachedEmbeddingProvider<P: EmbeddingProvider> {
    inner: P,
    cache: dashmap::DashMap<String, ClipEmbedding>,
    max_cache_size: usize,
}

impl<P: EmbeddingProvider> CachedEmbeddingProvider<P> {
    /// Create a new cached provider
    pub fn new(inner: P, max_cache_size: usize) -> Self {
        Self {
            inner,
            cache: dashmap::DashMap::new(),
            max_cache_size,
        }
    }
}

#[async_trait::async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for CachedEmbeddingProvider<P> {
    async fn embed(&self, text: &str) -> Result<ClipEmbedding> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        let embedding = self.inner.embed(text).await?;

        if self.cache.len() < self.max_cache_size {
            self.cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<ClipEmbedding>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut to_embed = Vec::new();
        let mut to_embed_indices = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(cached) = self.cache.get(*text) {
                results.push(Some(cached.clone()));
            } else {
                results.push(None);
                to_embed.push(*text);
                to_embed_indices.push(i);
            }
        }

        if !to_embed.is_empty() {
            let embeddings = self.inner.embed_batch(&to_embed).await?;
            for (embed_idx, (original_idx, embedding)) in to_embed_indices.iter().copied().zip(embeddings).enumerate() {
                if self.cache.len() < self.max_cache_size {
                    self.cache.insert(to_embed[embed_idx].to_string(), embedding.clone());
                }
                results[original_idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|o| o.unwrap()).collect())
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = ClipEmbedding::new(vec![1.0, 0.0, 0.0]);
        let b = ClipEmbedding::new(vec![1.0, 0.0, 0.0]);
        let c = ClipEmbedding::new(vec![0.0, 1.0, 0.0]);

        assert!((a.cosine_similarity(&b) - 1.0).abs() < 0.001);
        assert!(a.cosine_similarity(&c).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mock_provider() {
        let provider = MockEmbeddingProvider::new(768);

        let emb1 = provider.embed("hello world").await.unwrap();
        let emb2 = provider.embed("hello world").await.unwrap();
        let emb3 = provider.embed("different text").await.unwrap();

        // Same text should give same embedding
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.001);

        // Different text should give different embedding
        assert!(emb1.cosine_similarity(&emb3) < 0.9);
    }
}

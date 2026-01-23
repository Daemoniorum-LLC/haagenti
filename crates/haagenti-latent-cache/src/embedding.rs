//! Embedding providers for semantic similarity

use crate::Result;
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
#[cfg(test)]
pub struct MockEmbeddingProvider {
    dim: usize,
}

#[cfg(test)]
impl MockEmbeddingProvider {
    /// Create a mock provider with given embedding dimension
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<ClipEmbedding> {
        // Create deterministic embedding based on text hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut rng_state = hash;
        let vector: Vec<f32> = (0..self.dim)
            .map(|_| {
                // Simple LCG random number generator
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        Ok(ClipEmbedding::new(vector))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<ClipEmbedding>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dim
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

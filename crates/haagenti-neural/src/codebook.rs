//! Codebook definitions for neural compression

use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for a codebook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookConfig {
    /// Number of centroids (codewords)
    pub num_centroids: usize,
    /// Dimension of each centroid
    pub centroid_dim: usize,
    /// Bits per index (log2 of num_centroids)
    pub index_bits: u8,
    /// Whether to use product quantization
    pub product_quantization: bool,
    /// Number of subspaces for PQ
    pub pq_subspaces: usize,
}

impl CodebookConfig {
    /// Create config for attention Q/K weights
    pub fn attention_qk() -> Self {
        Self {
            num_centroids: 4096,
            centroid_dim: 64,
            index_bits: 12,
            product_quantization: false,
            pq_subspaces: 1,
        }
    }

    /// Create config for attention V/O weights
    pub fn attention_vo() -> Self {
        Self {
            num_centroids: 8192,
            centroid_dim: 64,
            index_bits: 13,
            product_quantization: false,
            pq_subspaces: 1,
        }
    }

    /// Create config for FFN weights
    pub fn ffn() -> Self {
        Self {
            num_centroids: 2048,
            centroid_dim: 128,
            index_bits: 11,
            product_quantization: true,
            pq_subspaces: 2,
        }
    }

    /// Create config for normalization weights
    pub fn normalization() -> Self {
        Self {
            num_centroids: 256,
            centroid_dim: 32,
            index_bits: 8,
            product_quantization: false,
            pq_subspaces: 1,
        }
    }

    /// Compute memory size for codebook in bytes
    pub fn memory_size(&self) -> usize {
        self.num_centroids * self.centroid_dim * 4 // FP32 centroids
    }

    /// Compute bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        if self.product_quantization {
            self.index_bits as f32 * self.pq_subspaces as f32 / self.centroid_dim as f32
        } else {
            self.index_bits as f32 / self.centroid_dim as f32
        }
    }
}

/// A codebook containing centroids for vector quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Configuration
    pub config: CodebookConfig,
    /// Centroid vectors [num_centroids × centroid_dim]
    pub centroids: Vec<f32>,
    /// Codebook identifier
    pub id: String,
    /// Training statistics
    pub stats: CodebookStats,
}

/// Statistics from codebook training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodebookStats {
    /// Mean squared error after training
    pub mse: f32,
    /// Number of training iterations
    pub iterations: usize,
    /// Centroid usage distribution (min, max, mean)
    pub usage_distribution: (f32, f32, f32),
    /// Training samples seen
    pub samples_seen: usize,
}

impl Codebook {
    /// Create a new codebook with random centroids
    pub fn new(config: CodebookConfig, id: impl Into<String>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let centroids: Vec<f32> = (0..config.num_centroids * config.centroid_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            config,
            centroids,
            id: id.into(),
            stats: CodebookStats::default(),
        }
    }

    /// Create from existing centroids
    pub fn from_centroids(
        config: CodebookConfig,
        centroids: Vec<f32>,
        id: impl Into<String>,
    ) -> Result<Self> {
        let expected = config.num_centroids * config.centroid_dim;
        if centroids.len() != expected {
            return Err(NeuralError::DimensionMismatch {
                expected,
                actual: centroids.len(),
            });
        }

        Ok(Self {
            config,
            centroids,
            id: id.into(),
            stats: CodebookStats::default(),
        })
    }

    /// Find nearest centroid for a vector
    pub fn find_nearest(&self, vector: &[f32]) -> Result<(usize, f32)> {
        if vector.len() != self.config.centroid_dim {
            return Err(NeuralError::DimensionMismatch {
                expected: self.config.centroid_dim,
                actual: vector.len(),
            });
        }

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for i in 0..self.config.num_centroids {
            let offset = i * self.config.centroid_dim;
            let dist =
                self.squared_distance(vector, &self.centroids[offset..offset + vector.len()]);

            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Ok((best_idx, best_dist.sqrt()))
    }

    /// Find K nearest centroids
    pub fn find_k_nearest(&self, vector: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if vector.len() != self.config.centroid_dim {
            return Err(NeuralError::DimensionMismatch {
                expected: self.config.centroid_dim,
                actual: vector.len(),
            });
        }

        let mut distances: Vec<(usize, f32)> = (0..self.config.num_centroids)
            .map(|i| {
                let offset = i * self.config.centroid_dim;
                let dist =
                    self.squared_distance(vector, &self.centroids[offset..offset + vector.len()]);
                (i, dist.sqrt())
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Get centroid by index
    pub fn get_centroid(&self, index: usize) -> Option<&[f32]> {
        if index >= self.config.num_centroids {
            return None;
        }
        let offset = index * self.config.centroid_dim;
        Some(&self.centroids[offset..offset + self.config.centroid_dim])
    }

    /// Squared L2 distance between two vectors
    fn squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Encode a batch of vectors to indices
    pub fn encode_batch(&self, vectors: &[f32], vector_dim: usize) -> Result<Vec<u16>> {
        if vector_dim != self.config.centroid_dim {
            return Err(NeuralError::DimensionMismatch {
                expected: self.config.centroid_dim,
                actual: vector_dim,
            });
        }

        let num_vectors = vectors.len() / vector_dim;
        let mut indices = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let offset = i * vector_dim;
            let vector = &vectors[offset..offset + vector_dim];
            let (idx, _) = self.find_nearest(vector)?;
            indices.push(idx as u16);
        }

        Ok(indices)
    }

    /// Decode indices to vectors
    pub fn decode_batch(&self, indices: &[u16]) -> Vec<f32> {
        let dim = self.config.centroid_dim;
        let mut vectors = Vec::with_capacity(indices.len() * dim);

        for &idx in indices {
            if let Some(centroid) = self.get_centroid(idx as usize) {
                vectors.extend_from_slice(centroid);
            } else {
                // Fallback to zeros
                vectors.extend(std::iter::repeat_n(0.0, dim));
            }
        }

        vectors
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| NeuralError::InvalidFormat(e.to_string()))
    }
}

/// Layer type for codebook selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Attention Q/K projection
    AttentionQK,
    /// Attention V/O projection
    AttentionVO,
    /// Feed-forward network
    FFN,
    /// Layer normalization
    Normalization,
    /// Embedding layer
    Embedding,
    /// Output head
    OutputHead,
}

impl LayerType {
    /// Get default codebook config for this layer type
    pub fn default_config(&self) -> CodebookConfig {
        match self {
            LayerType::AttentionQK => CodebookConfig::attention_qk(),
            LayerType::AttentionVO => CodebookConfig::attention_vo(),
            LayerType::FFN => CodebookConfig::ffn(),
            LayerType::Normalization => CodebookConfig::normalization(),
            LayerType::Embedding => CodebookConfig::ffn(), // Similar to FFN
            LayerType::OutputHead => CodebookConfig::attention_vo(), // High precision
        }
    }
}

/// Collection of codebooks for different layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCodebook {
    /// Codebooks by layer type
    codebooks: HashMap<LayerType, Codebook>,
    /// Model identifier
    pub model_id: String,
    /// Total memory used by all codebooks
    pub total_memory: usize,
}

impl LayerCodebook {
    /// Create a new layer codebook collection
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            codebooks: HashMap::new(),
            model_id: model_id.into(),
            total_memory: 0,
        }
    }

    /// Add a codebook for a layer type
    pub fn add(&mut self, layer_type: LayerType, codebook: Codebook) {
        self.total_memory += codebook.config.memory_size();
        self.codebooks.insert(layer_type, codebook);
    }

    /// Get codebook for a layer type
    pub fn get(&self, layer_type: LayerType) -> Option<&Codebook> {
        self.codebooks.get(&layer_type)
    }

    /// Initialize with default codebooks for all layer types
    pub fn with_defaults(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let mut lc = Self::new(&model_id);

        for layer_type in &[
            LayerType::AttentionQK,
            LayerType::AttentionVO,
            LayerType::FFN,
            LayerType::Normalization,
        ] {
            let config = layer_type.default_config();
            let id = format!("{}_{:?}", model_id, layer_type);
            lc.add(*layer_type, Codebook::new(config, id));
        }

        lc
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| NeuralError::InvalidFormat(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_creation() {
        let config = CodebookConfig::attention_qk();
        let codebook = Codebook::new(config.clone(), "test");

        assert_eq!(codebook.config.num_centroids, 4096);
        assert_eq!(codebook.centroids.len(), 4096 * 64);
    }

    #[test]
    fn test_find_nearest() {
        let config = CodebookConfig {
            num_centroids: 4,
            centroid_dim: 3,
            index_bits: 2,
            product_quantization: false,
            pq_subspaces: 1,
        };

        let centroids = vec![
            1.0, 0.0, 0.0, // Centroid 0
            0.0, 1.0, 0.0, // Centroid 1
            0.0, 0.0, 1.0, // Centroid 2
            1.0, 1.0, 1.0, // Centroid 3
        ];

        let codebook = Codebook::from_centroids(config, centroids, "test").unwrap();

        // Vector close to centroid 1
        let (idx, _) = codebook.find_nearest(&[0.1, 0.9, 0.1]).unwrap();
        assert_eq!(idx, 1);

        // Vector close to centroid 3
        let (idx, _) = codebook.find_nearest(&[0.9, 0.9, 0.9]).unwrap();
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_encode_decode() {
        let config = CodebookConfig {
            num_centroids: 256,
            centroid_dim: 4,
            index_bits: 8,
            product_quantization: false,
            pq_subspaces: 1,
        };

        let codebook = Codebook::new(config, "test");

        // Create some test vectors
        let vectors: Vec<f32> = (0..40).map(|i| i as f32 * 0.01).collect();

        // Encode
        let indices = codebook.encode_batch(&vectors, 4).unwrap();
        assert_eq!(indices.len(), 10);

        // Decode
        let decoded = codebook.decode_batch(&indices);
        assert_eq!(decoded.len(), 40);
    }

    #[test]
    fn test_layer_codebook() {
        let lc = LayerCodebook::with_defaults("sdxl");

        assert!(lc.get(LayerType::AttentionQK).is_some());
        assert!(lc.get(LayerType::FFN).is_some());
        assert!(lc.total_memory > 0);
    }

    #[test]
    fn test_memory_size() {
        let config = CodebookConfig::attention_qk();
        assert_eq!(config.memory_size(), 4096 * 64 * 4); // 1MB
    }
}

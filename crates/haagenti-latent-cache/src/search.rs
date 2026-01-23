//! HNSW-based similarity search for embeddings

use crate::ClipEmbedding;
use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Configuration for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of neighbors to consider during construction
    pub ef_construction: usize,
    /// Number of neighbors to consider during search
    pub ef_search: usize,
    /// Maximum number of connections per node
    pub m: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            ef_search: 50,
            m: 16,
        }
    }
}

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The ID of the matching entry
    pub id: String,
    /// Similarity score (0.0 - 1.0, higher is more similar)
    pub similarity: f32,
    /// The matched embedding
    pub embedding: ClipEmbedding,
}

/// Point wrapper for HNSW
#[derive(Clone)]
struct EmbeddingPoint {
    id: String,
    embedding: ClipEmbedding,
}

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Use 1 - cosine_similarity as distance (lower is closer)
        1.0 - self.embedding.cosine_similarity(&other.embedding)
    }
}

/// Similarity search using HNSW
pub struct SimilaritySearch {
    config: HnswConfig,
    /// The HNSW index (rebuilt on modifications)
    index: Arc<RwLock<Option<HnswMap<EmbeddingPoint, String>>>>,
    /// All points (for rebuilding)
    points: Arc<RwLock<Vec<EmbeddingPoint>>>,
    /// Whether index needs rebuild
    needs_rebuild: Arc<RwLock<bool>>,
}

impl SimilaritySearch {
    /// Create a new similarity search
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            index: Arc::new(RwLock::new(None)),
            points: Arc::new(RwLock::new(Vec::new())),
            needs_rebuild: Arc::new(RwLock::new(false)),
        }
    }

    /// Add an embedding to the index
    pub fn insert(&self, id: String, embedding: ClipEmbedding) {
        let point = EmbeddingPoint {
            id: id.clone(),
            embedding,
        };

        self.points.write().unwrap().push(point);
        *self.needs_rebuild.write().unwrap() = true;
    }

    /// Remove an entry from the index
    pub fn remove(&self, id: &str) -> bool {
        let mut points = self.points.write().unwrap();
        let len_before = points.len();
        points.retain(|p| p.id != id);

        if points.len() != len_before {
            *self.needs_rebuild.write().unwrap() = true;
            true
        } else {
            false
        }
    }

    /// Rebuild the index if needed
    pub fn rebuild_if_needed(&self) {
        if !*self.needs_rebuild.read().unwrap() {
            return;
        }

        let points = self.points.read().unwrap();
        if points.is_empty() {
            *self.index.write().unwrap() = None;
            *self.needs_rebuild.write().unwrap() = false;
            return;
        }

        // Build new index
        let values: Vec<String> = points.iter().map(|p| p.id.clone()).collect();
        let points_vec: Vec<EmbeddingPoint> = points.clone();

        let hnsw = Builder::default()
            .ef_construction(self.config.ef_construction)
            .build(points_vec, values);

        *self.index.write().unwrap() = Some(hnsw);
        *self.needs_rebuild.write().unwrap() = false;
    }

    /// Search for similar embeddings
    pub fn search(&self, query: &ClipEmbedding, k: usize) -> Vec<SearchResult> {
        self.rebuild_if_needed();

        let index_guard = self.index.read().unwrap();
        let index = match index_guard.as_ref() {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let query_point = EmbeddingPoint {
            id: String::new(),
            embedding: query.clone(),
        };

        let mut search = Search::default();
        let results = index.search(&query_point, &mut search);

        results
            .take(k)
            .map(|item| {
                let point = &index.values[item.pid.into_inner() as usize];
                let distance = item.distance;
                let similarity = 1.0 - distance;

                // Get the original embedding
                let points = self.points.read().unwrap();
                let embedding = points
                    .iter()
                    .find(|p| p.id == *point)
                    .map(|p| p.embedding.clone())
                    .unwrap_or_else(|| ClipEmbedding::new(vec![]));

                SearchResult {
                    id: point.clone(),
                    similarity,
                    embedding,
                }
            })
            .collect()
    }

    /// Find the most similar entry above a threshold
    pub fn find_similar(&self, query: &ClipEmbedding, min_similarity: f32) -> Option<SearchResult> {
        let results = self.search(query, 1);
        results.into_iter().find(|r| r.similarity >= min_similarity)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.points.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.points.read().unwrap().is_empty()
    }

    /// Clear the index
    pub fn clear(&self) {
        self.points.write().unwrap().clear();
        *self.index.write().unwrap() = None;
        *self.needs_rebuild.write().unwrap() = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search() {
        let search = SimilaritySearch::new(HnswConfig::default());

        // Add some embeddings
        search.insert("a".to_string(), ClipEmbedding::new(vec![1.0, 0.0, 0.0]));
        search.insert("b".to_string(), ClipEmbedding::new(vec![0.9, 0.1, 0.0]));
        search.insert("c".to_string(), ClipEmbedding::new(vec![0.0, 1.0, 0.0]));

        // Search for something similar to "a"
        let query = ClipEmbedding::new(vec![0.95, 0.05, 0.0]);
        let results = search.search(&query, 2);

        assert_eq!(results.len(), 2);
        // Should find "a" or "b" as most similar
        assert!(results[0].id == "a" || results[0].id == "b");
    }

    #[test]
    fn test_similarity_threshold() {
        let search = SimilaritySearch::new(HnswConfig::default());

        search.insert("a".to_string(), ClipEmbedding::new(vec![1.0, 0.0, 0.0]));
        search.insert("b".to_string(), ClipEmbedding::new(vec![0.0, 1.0, 0.0]));

        // Query orthogonal to both
        let query = ClipEmbedding::new(vec![0.0, 0.0, 1.0]);

        // Should not find anything above 0.5 similarity
        let result = search.find_similar(&query, 0.5);
        assert!(result.is_none());
    }
}

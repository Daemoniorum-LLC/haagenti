//! Similarity index for efficient fragment matching
//!
//! Uses LSH (Locality-Sensitive Hashing) to enable sub-linear
//! nearest neighbor search across millions of fragments.

use arcanum_primitives::prelude::Blake3;
use crate::{Fragment, FragmentId, FragmentSignature, Result, SignatureConfig};
use dashmap::DashMap;
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Threshold configuration for similarity matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityThreshold {
    /// Minimum similarity for exact match (byte-identical)
    pub exact: f32,
    /// Minimum similarity for near-duplicate
    pub near_duplicate: f32,
    /// Minimum similarity for similar fragments
    pub similar: f32,
}

impl Default for SimilarityThreshold {
    fn default() -> Self {
        Self {
            exact: 0.9999,
            near_duplicate: 0.995,
            similar: 0.90,
        }
    }
}

/// A match found in the similarity index
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    /// The matched fragment ID
    pub fragment_id: FragmentId,
    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Match type
    pub match_type: MatchType,
}

/// Type of similarity match
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    /// Byte-identical match
    Exact,
    /// Near-duplicate (>99.5% similar)
    NearDuplicate,
    /// Similar fragment (>90% similar)
    Similar,
}

/// LSH band for indexing
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BandHash([u8; 8]);

impl BandHash {
    fn from_signature(sig: &[u8; 32], band_idx: usize, rows_per_band: usize) -> Self {
        let start = band_idx * rows_per_band;
        let end = (start + rows_per_band).min(32);

        let mut hash = [0u8; 8];
        let hash_input: Vec<u8> = sig[start..end].to_vec();
        let h = Blake3::hash(&hash_input);
        hash.copy_from_slice(&h[..8]);

        BandHash(hash)
    }
}

/// Similarity index using Locality-Sensitive Hashing
pub struct SimilarityIndex {
    /// Signature configuration
    config: SignatureConfig,
    /// Thresholds for matching
    thresholds: SimilarityThreshold,
    /// LSH band buckets (band_idx → hash → fragment IDs)
    bands: Vec<DashMap<BandHash, IndexSet<FragmentId>>>,
    /// Fragment signatures (for verification)
    signatures: DashMap<FragmentId, Arc<FragmentSignature>>,
    /// Number of bands
    num_bands: usize,
    /// Rows per band
    rows_per_band: usize,
}

impl SimilarityIndex {
    /// Create a new similarity index
    pub fn new(config: SignatureConfig, thresholds: SimilarityThreshold) -> Self {
        let num_bands = config.num_bands;
        let rows_per_band = 32 / num_bands; // 32 bytes in simhash

        let bands = (0..num_bands).map(|_| DashMap::new()).collect();

        Self {
            config,
            thresholds,
            bands,
            signatures: DashMap::new(),
            num_bands,
            rows_per_band,
        }
    }

    /// Index a fragment for similarity search
    pub fn index(&self, fragment: &Fragment) {
        let signature = FragmentSignature::compute(&fragment.data, &self.config);

        // Store signature
        self.signatures
            .insert(fragment.id, Arc::new(signature.clone()));

        // Add to LSH bands
        for (band_idx, band_map) in self.bands.iter().enumerate() {
            let band_hash = BandHash::from_signature(&signature.simhash, band_idx, self.rows_per_band);

            band_map
                .entry(band_hash)
                .or_insert_with(IndexSet::new)
                .insert(fragment.id);
        }
    }

    /// Remove a fragment from the index
    pub fn remove(&self, fragment_id: &FragmentId) {
        if let Some((_, signature)) = self.signatures.remove(fragment_id) {
            // Remove from all bands
            for (band_idx, band_map) in self.bands.iter().enumerate() {
                let band_hash =
                    BandHash::from_signature(&signature.simhash, band_idx, self.rows_per_band);

                if let Some(mut bucket) = band_map.get_mut(&band_hash) {
                    bucket.swap_remove(fragment_id);
                }
            }
        }
    }

    /// Find similar fragments
    pub fn find_similar(&self, data: &[u8], max_results: usize) -> Vec<SimilarityMatch> {
        let query_sig = FragmentSignature::compute(data, &self.config);
        self.find_similar_by_signature(&query_sig, max_results)
    }

    /// Find similar fragments by signature
    pub fn find_similar_by_signature(
        &self,
        query_sig: &FragmentSignature,
        max_results: usize,
    ) -> Vec<SimilarityMatch> {
        // Collect candidate IDs from LSH bands
        let mut candidates = IndexSet::new();

        for (band_idx, band_map) in self.bands.iter().enumerate() {
            let band_hash =
                BandHash::from_signature(&query_sig.simhash, band_idx, self.rows_per_band);

            if let Some(bucket) = band_map.get(&band_hash) {
                candidates.extend(bucket.iter().copied());
            }
        }

        // Compute exact similarities for candidates
        let mut matches: Vec<SimilarityMatch> = candidates
            .into_iter()
            .filter_map(|frag_id| {
                let sig = self.signatures.get(&frag_id)?;
                let similarity = query_sig.similarity(&sig);

                if similarity >= self.thresholds.similar {
                    let match_type = if similarity >= self.thresholds.exact {
                        MatchType::Exact
                    } else if similarity >= self.thresholds.near_duplicate {
                        MatchType::NearDuplicate
                    } else {
                        MatchType::Similar
                    };

                    Some(SimilarityMatch {
                        fragment_id: frag_id,
                        similarity,
                        match_type,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity (descending)
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches.truncate(max_results);

        matches
    }

    /// Find exact or near-duplicate match
    pub fn find_duplicate(&self, data: &[u8]) -> Option<SimilarityMatch> {
        let matches = self.find_similar(data, 1);
        matches
            .into_iter()
            .find(|m| m.match_type == MatchType::Exact || m.match_type == MatchType::NearDuplicate)
    }

    /// Get index statistics
    pub fn stats(&self) -> SimilarityIndexStats {
        let total_fragments = self.signatures.len();
        let total_buckets: usize = self.bands.iter().map(|b| b.len()).sum();
        let avg_bucket_size = if total_buckets > 0 {
            total_fragments as f32 / total_buckets as f32
        } else {
            0.0
        };

        SimilarityIndexStats {
            total_fragments,
            num_bands: self.num_bands,
            total_buckets,
            avg_bucket_size,
        }
    }
}

/// Statistics about the similarity index
#[derive(Debug, Clone)]
pub struct SimilarityIndexStats {
    /// Total fragments indexed
    pub total_fragments: usize,
    /// Number of LSH bands
    pub num_bands: usize,
    /// Total buckets across all bands
    pub total_buckets: usize,
    /// Average fragments per bucket
    pub avg_bucket_size: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FragmentType;

    #[test]
    fn test_find_exact_duplicate() {
        let config = SignatureConfig::default();
        let thresholds = SimilarityThreshold::default();
        let index = SimilarityIndex::new(config, thresholds);

        let data = vec![42u8; 1024];
        let fragment = Fragment::new(
            data.clone(),
            FragmentType::Generic,
            smallvec::smallvec![32, 32],
            "fp16",
            "lz4",
            0.5,
        );

        index.index(&fragment);

        let result = index.find_duplicate(&data);
        assert!(result.is_some());
        assert_eq!(result.unwrap().fragment_id, fragment.id);
    }

    #[test]
    fn test_find_similar_fragments() {
        let config = SignatureConfig::default();
        let thresholds = SimilarityThreshold::default();
        let index = SimilarityIndex::new(config, thresholds);

        // Create several fragments
        for i in 0..10 {
            let data: Vec<u8> = (0..1024).map(|x| ((x + i * 10) % 256) as u8).collect();
            let fragment = Fragment::new(
                data,
                FragmentType::Generic,
                smallvec::smallvec![32, 32],
                "fp16",
                "lz4",
                0.5,
            );
            index.index(&fragment);
        }

        let stats = index.stats();
        assert_eq!(stats.total_fragments, 10);
    }
}

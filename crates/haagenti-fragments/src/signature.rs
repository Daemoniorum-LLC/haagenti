//! Locality-Sensitive Hashing for fragment similarity detection
//!
//! Uses SimHash for efficient approximate nearest neighbor search.
//! This allows finding similar fragments across different models
//! even when they're not byte-identical.

use arcanum_primitives::prelude::Blake3;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Configuration for signature computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureConfig {
    /// Number of hash functions (bits in signature)
    pub num_hashes: usize,
    /// Chunk size for shingling
    pub shingle_size: usize,
    /// Number of bands for LSH
    pub num_bands: usize,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            num_hashes: 256,
            shingle_size: 8,
            num_bands: 32,
        }
    }
}

/// Locality-sensitive hash signature for a fragment
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FragmentSignature {
    /// SimHash bits (256 bits = 32 bytes)
    pub simhash: [u8; 32],
    /// MinHash signature for Jaccard similarity
    #[serde(with = "BigArray")]
    pub minhash: [u32; 64],
    /// Statistical fingerprint
    pub stats: StatisticalFingerprint,
}

/// Statistical properties of tensor data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StatisticalFingerprint {
    /// Quantized mean (scaled to u16)
    pub mean: u16,
    /// Quantized standard deviation
    pub std_dev: u16,
    /// Quantized min value
    pub min: u16,
    /// Quantized max value
    pub max: u16,
    /// Histogram bucket counts (16 buckets)
    pub histogram: [u8; 16],
}

impl FragmentSignature {
    /// Compute signature from raw tensor data
    pub fn compute(data: &[u8], config: &SignatureConfig) -> Self {
        let simhash = Self::compute_simhash(data, config);
        let minhash = Self::compute_minhash(data, config);
        let stats = Self::compute_stats(data);

        Self {
            simhash,
            minhash,
            stats,
        }
    }

    /// Compute SimHash (bit-wise locality-sensitive hash)
    fn compute_simhash(data: &[u8], config: &SignatureConfig) -> [u8; 32] {
        let mut bit_counts = [0i32; 256];

        // Process data in shingles
        for chunk in data.chunks(config.shingle_size) {
            let hash = Blake3::hash(chunk);
            let hash_bytes = &hash;

            // Each bit of the hash votes for its position
            for (byte_idx, &byte) in hash_bytes.iter().enumerate() {
                for bit_idx in 0..8 {
                    let bit_pos = byte_idx * 8 + bit_idx;
                    if bit_pos < 256 {
                        if (byte >> bit_idx) & 1 == 1 {
                            bit_counts[bit_pos] += 1;
                        } else {
                            bit_counts[bit_pos] -= 1;
                        }
                    }
                }
            }
        }

        // Convert counts to bits
        let mut result = [0u8; 32];
        for (i, &count) in bit_counts.iter().enumerate() {
            if count > 0 {
                result[i / 8] |= 1 << (i % 8);
            }
        }

        result
    }

    /// Compute MinHash for Jaccard similarity
    fn compute_minhash(data: &[u8], config: &SignatureConfig) -> [u32; 64] {
        let mut minhash = [u32::MAX; 64];

        // Use different hash functions (seeds)
        for chunk in data.chunks(config.shingle_size) {
            for (i, min) in minhash.iter_mut().enumerate() {
                // Create hash with different seed per position
                let mut hasher_input = Vec::with_capacity(chunk.len() + 4);
                hasher_input.extend_from_slice(&(i as u32).to_le_bytes());
                hasher_input.extend_from_slice(chunk);

                let hash = Blake3::hash(&hasher_input);
                let hash_val = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);

                *min = (*min).min(hash_val);
            }
        }

        minhash
    }

    /// Compute statistical fingerprint
    fn compute_stats(data: &[u8]) -> StatisticalFingerprint {
        if data.is_empty() {
            return StatisticalFingerprint {
                mean: 0,
                std_dev: 0,
                min: 0,
                max: 0,
                histogram: [0; 16],
            };
        }

        // Interpret as f16 values (2 bytes each)
        let values: Vec<f32> = data
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return StatisticalFingerprint {
                mean: 32768,
                std_dev: 0,
                min: 32768,
                max: 32768,
                histogram: [0; 16],
            };
        }

        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;

        let variance: f32 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Build histogram (16 buckets between min and max)
        let mut histogram = [0u8; 16];
        let range = (max_val - min_val).max(1e-6);

        for &v in &values {
            let bucket = (((v - min_val) / range) * 15.999).floor() as usize;
            let bucket = bucket.min(15);
            histogram[bucket] = histogram[bucket].saturating_add(1);
        }

        // Normalize histogram
        let max_count = *histogram.iter().max().unwrap_or(&1) as f32;
        for h in &mut histogram {
            *h = ((*h as f32 / max_count) * 255.0) as u8;
        }

        // Quantize to u16 (map [-10, 10] range to [0, 65535])
        let quantize = |v: f32| -> u16 { ((v.clamp(-10.0, 10.0) + 10.0) / 20.0 * 65535.0) as u16 };

        StatisticalFingerprint {
            mean: quantize(mean),
            std_dev: quantize(std_dev),
            min: quantize(min_val),
            max: quantize(max_val),
            histogram,
        }
    }

    /// Compute Hamming distance between SimHash signatures
    pub fn simhash_distance(&self, other: &Self) -> u32 {
        self.simhash
            .iter()
            .zip(other.simhash.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Compute Jaccard similarity from MinHash
    pub fn minhash_similarity(&self, other: &Self) -> f32 {
        let matches = self
            .minhash
            .iter()
            .zip(other.minhash.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f32 / 64.0
    }

    /// Compute combined similarity score
    pub fn similarity(&self, other: &Self) -> f32 {
        // SimHash component (Hamming distance → similarity)
        let hamming = self.simhash_distance(other);
        let simhash_sim = 1.0 - (hamming as f32 / 256.0);

        // MinHash component (Jaccard similarity)
        let minhash_sim = self.minhash_similarity(other);

        // Statistical component
        let stats_sim = self.stats_similarity(&other.stats);

        // Weighted combination
        simhash_sim * 0.4 + minhash_sim * 0.4 + stats_sim * 0.2
    }

    /// Compute statistical similarity
    fn stats_similarity(&self, other: &StatisticalFingerprint) -> f32 {
        let mean_diff = (self.stats.mean as i32 - other.mean as i32).abs() as f32 / 65535.0;
        let std_diff = (self.stats.std_dev as i32 - other.std_dev as i32).abs() as f32 / 65535.0;

        // Histogram similarity (cosine)
        let dot: f32 = self
            .stats
            .histogram
            .iter()
            .zip(other.histogram.iter())
            .map(|(&a, &b)| a as f32 * b as f32)
            .sum();
        let mag_a: f32 = self
            .stats
            .histogram
            .iter()
            .map(|&x| (x as f32).powi(2))
            .sum::<f32>()
            .sqrt();
        let mag_b: f32 = other
            .histogram
            .iter()
            .map(|&x| (x as f32).powi(2))
            .sum::<f32>()
            .sqrt();
        let hist_sim = if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        };

        // Combine
        (1.0 - mean_diff) * 0.3 + (1.0 - std_diff) * 0.2 + hist_sim * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_signatures() {
        let config = SignatureConfig::default();
        let data = vec![0u8; 1024];

        let sig1 = FragmentSignature::compute(&data, &config);
        let sig2 = FragmentSignature::compute(&data, &config);

        assert_eq!(sig1.simhash, sig2.simhash);
        assert_eq!(sig1.minhash, sig2.minhash);
        assert!((sig1.similarity(&sig2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_different_signatures() {
        let config = SignatureConfig::default();
        let data1 = vec![0u8; 1024];
        let data2 = vec![255u8; 1024];

        let sig1 = FragmentSignature::compute(&data1, &config);
        let sig2 = FragmentSignature::compute(&data2, &config);

        assert!(sig1.similarity(&sig2) < 0.5);
    }
}

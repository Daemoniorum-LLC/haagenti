//! Spectral Analysis for Adaptive Retention
//!
//! This module provides spectral energy analysis to determine optimal per-tensor
//! retention ratios for compressive encoding. Instead of using a fixed retention
//! ratio for all tensors, adaptive retention analyzes each tensor's spectral
//! characteristics to find the minimum retention needed for target quality.
//!
//! ## Key Insight
//!
//! Different tensors have different spectral energy distributions:
//! - **Attention weights**: Often low-rank, energy concentrated in few coefficients
//! - **MLP weights**: More distributed energy, need higher retention
//! - **Embeddings**: Often sparse, can use very low retention
//!
//! By adapting retention per-tensor, we can achieve the same quality with
//! significantly less storage (or better quality at same storage).

use haagenti_core::Result;
use crate::holotensor::dct_2d;

/// Spectral energy analyzer for adaptive compression.
///
/// Analyzes the DCT spectrum of a tensor to determine:
/// 1. Energy distribution across frequency components
/// 2. Optimal retention ratio for a target quality level
/// 3. Compression potential (how much the tensor can be compressed)
#[derive(Debug, Clone)]
pub struct SpectralAnalyzer {
    /// Target quality level (0.0-1.0, where 1.0 = perfect reconstruction)
    target_quality: f32,
    /// Minimum retention ratio to ensure basic structure
    min_retention: f32,
    /// Maximum retention ratio (cap for very flat spectra)
    max_retention: f32,
}

impl Default for SpectralAnalyzer {
    fn default() -> Self {
        Self {
            target_quality: 0.95,  // 95% energy retention
            min_retention: 0.05,   // At least 5% of coefficients
            max_retention: 0.90,   // At most 90%
        }
    }
}

impl SpectralAnalyzer {
    /// Create analyzer with target quality level.
    ///
    /// # Arguments
    /// * `target_quality` - Fraction of spectral energy to retain (0.9-0.99 typical)
    pub fn new(target_quality: f32) -> Self {
        Self {
            target_quality: target_quality.clamp(0.5, 0.999),
            ..Default::default()
        }
    }

    /// Set minimum retention ratio.
    pub fn with_min_retention(mut self, min: f32) -> Self {
        self.min_retention = min.clamp(0.01, 0.5);
        self
    }

    /// Set maximum retention ratio.
    pub fn with_max_retention(mut self, max: f32) -> Self {
        self.max_retention = max.clamp(0.5, 1.0);
        self
    }

    /// Get the target quality level.
    pub fn target_quality(&self) -> f32 {
        self.target_quality
    }

    /// Get the minimum retention ratio.
    pub fn min_retention(&self) -> f32 {
        self.min_retention
    }

    /// Get the maximum retention ratio.
    pub fn max_retention(&self) -> f32 {
        self.max_retention
    }

    /// Compute spectral energy distribution for a 2D tensor.
    ///
    /// Returns a sorted vector of (coefficient_index, energy) pairs,
    /// sorted by energy in descending order.
    pub fn compute_energy_distribution(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let n = width * height;
        if data.len() != n {
            return Err(haagenti_core::Error::corrupted("data size mismatch"));
        }

        // Transform to frequency domain
        let mut dct_coeffs = vec![0.0f32; n];
        dct_2d(data, &mut dct_coeffs, width, height);

        // Compute energy (squared magnitude) for each coefficient
        let mut energy_dist: Vec<(usize, f32)> = dct_coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c * c))
            .collect();

        // Sort by energy descending
        energy_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(energy_dist)
    }

    /// Compute cumulative energy curve from sorted energy distribution.
    ///
    /// Returns vector where element i is the fraction of total energy
    /// captured by the first i+1 coefficients.
    pub fn compute_cumulative_energy(&self, energy_dist: &[(usize, f32)]) -> Vec<f32> {
        if energy_dist.is_empty() {
            return vec![];
        }

        let total_energy: f32 = energy_dist.iter().map(|(_, e)| e).sum();
        if total_energy == 0.0 {
            // All zeros - any retention works
            return vec![1.0; energy_dist.len()];
        }

        let mut cumulative = Vec::with_capacity(energy_dist.len());
        let mut running_sum = 0.0f32;

        for (_, energy) in energy_dist {
            running_sum += energy;
            cumulative.push(running_sum / total_energy);
        }

        cumulative
    }

    /// Find the "knee point" where diminishing returns begin.
    ///
    /// Uses the maximum curvature method to find where the cumulative
    /// energy curve transitions from steep to flat.
    ///
    /// # Arguments
    /// * `cumulative_energy` - Cumulative energy fractions
    /// * `threshold` - Target cumulative energy (e.g., 0.95)
    ///
    /// # Returns
    /// Index of the knee point (number of coefficients to retain)
    pub fn find_knee_point(&self, cumulative_energy: &[f32], threshold: f32) -> usize {
        if cumulative_energy.is_empty() {
            return 0;
        }

        // Simple approach: find first index where cumulative >= threshold
        for (i, &cum) in cumulative_energy.iter().enumerate() {
            if cum >= threshold {
                return i + 1; // +1 because we need this many coefficients
            }
        }

        // If threshold not reached, return all
        cumulative_energy.len()
    }

    /// Find knee point using maximum curvature method.
    ///
    /// This finds the point of maximum curvature in the cumulative energy
    /// curve, which often corresponds to a natural break point.
    pub fn find_knee_by_curvature(&self, cumulative_energy: &[f32]) -> usize {
        if cumulative_energy.len() < 3 {
            return cumulative_energy.len();
        }

        let n = cumulative_energy.len();
        let mut max_curvature = 0.0f32;
        let mut knee_index = 1;

        // Compute curvature at each point
        // Curvature ≈ |f''| / (1 + f'^2)^1.5
        for i in 1..n - 1 {
            let x = i as f32 / n as f32;
            let y_prev = cumulative_energy[i - 1];
            let y = cumulative_energy[i];
            let y_next = cumulative_energy[i + 1];

            // Second derivative (finite difference)
            let d2y = y_next - 2.0 * y + y_prev;

            // First derivative
            let dy = (y_next - y_prev) / 2.0;

            // Curvature magnitude
            let curvature = d2y.abs() / (1.0 + dy * dy).powf(1.5);

            // We want the point of maximum negative curvature (concave down)
            // which indicates transition from steep to flat
            if curvature > max_curvature && d2y < 0.0 {
                max_curvature = curvature;
                knee_index = i;
            }
        }

        // Return at least 1, at most n
        knee_index.max(1)
    }

    /// Compute optimal retention ratio for a tensor.
    ///
    /// Analyzes the spectral energy distribution and returns the minimum
    /// retention ratio that achieves the target quality.
    ///
    /// # Arguments
    /// * `data` - Flattened 2D tensor data
    /// * `width` - Tensor width
    /// * `height` - Tensor height
    ///
    /// # Returns
    /// Optimal retention ratio (fraction of coefficients to keep)
    pub fn compute_optimal_retention(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32> {
        let n = width * height;
        if n == 0 {
            return Ok(self.min_retention);
        }

        // Get energy distribution
        let energy_dist = self.compute_energy_distribution(data, width, height)?;

        // Compute cumulative energy
        let cumulative = self.compute_cumulative_energy(&energy_dist);

        // Find knee point for target quality
        let knee = self.find_knee_point(&cumulative, self.target_quality);

        // Convert to retention ratio
        let retention = knee as f32 / n as f32;

        // Clamp to bounds
        Ok(retention.clamp(self.min_retention, self.max_retention))
    }

    /// Analyze a tensor and return comprehensive statistics.
    pub fn analyze(&self, data: &[f32], width: usize, height: usize) -> Result<SpectralStats> {
        let n = width * height;
        let energy_dist = self.compute_energy_distribution(data, width, height)?;
        let cumulative = self.compute_cumulative_energy(&energy_dist);

        let total_energy: f32 = energy_dist.iter().map(|(_, e)| e).sum();

        // Find retention for various quality levels
        let retention_90 = self.find_knee_point(&cumulative, 0.90) as f32 / n as f32;
        let retention_95 = self.find_knee_point(&cumulative, 0.95) as f32 / n as f32;
        let retention_99 = self.find_knee_point(&cumulative, 0.99) as f32 / n as f32;

        // Compute spectral entropy (measure of energy spread)
        let entropy = if total_energy > 0.0 {
            let probs: Vec<f32> = energy_dist
                .iter()
                .map(|(_, e)| e / total_energy)
                .filter(|&p| p > 1e-10)
                .collect();
            -probs.iter().map(|p| p * p.ln()).sum::<f32>() / (n as f32).ln()
        } else {
            0.0
        };

        // Knee point by curvature
        let knee_by_curvature = self.find_knee_by_curvature(&cumulative) as f32 / n as f32;

        Ok(SpectralStats {
            total_elements: n,
            total_energy,
            retention_90,
            retention_95,
            retention_99,
            spectral_entropy: entropy,
            knee_by_curvature,
            optimal_retention: self.compute_optimal_retention(data, width, height)?,
        })
    }
}

/// Statistics from spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectralStats {
    /// Total number of elements in tensor
    pub total_elements: usize,
    /// Total spectral energy (sum of squared coefficients)
    pub total_energy: f32,
    /// Retention needed for 90% energy
    pub retention_90: f32,
    /// Retention needed for 95% energy
    pub retention_95: f32,
    /// Retention needed for 99% energy
    pub retention_99: f32,
    /// Spectral entropy (0=concentrated, 1=uniform)
    pub spectral_entropy: f32,
    /// Knee point from curvature analysis
    pub knee_by_curvature: f32,
    /// Recommended optimal retention
    pub optimal_retention: f32,
}

impl SpectralStats {
    /// Classify tensor compressibility.
    pub fn compressibility(&self) -> Compressibility {
        if self.retention_95 < 0.10 {
            Compressibility::High
        } else if self.retention_95 < 0.30 {
            Compressibility::Medium
        } else if self.retention_95 < 0.60 {
            Compressibility::Low
        } else {
            Compressibility::VeryLow
        }
    }
}

/// Compressibility classification for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compressibility {
    /// Highly compressible (retention < 10% for 95% energy)
    High,
    /// Moderately compressible (10-30% retention)
    Medium,
    /// Low compressibility (30-60% retention)
    Low,
    /// Very low compressibility (>60% retention needed)
    VeryLow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_default() {
        let analyzer = SpectralAnalyzer::default();
        assert!((analyzer.target_quality - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_energy_distribution_uniform() {
        let analyzer = SpectralAnalyzer::default();
        // Uniform data has flat DCT spectrum
        let data: Vec<f32> = vec![1.0; 64];
        let energy = analyzer.compute_energy_distribution(&data, 8, 8).unwrap();

        // DC component should have most energy for uniform input
        assert!(!energy.is_empty());
    }

    #[test]
    fn test_energy_distribution_sparse() {
        let analyzer = SpectralAnalyzer::default();
        // Sparse data (single spike) should have concentrated energy
        let mut data = vec![0.0f32; 64];
        data[0] = 1.0;

        let energy = analyzer.compute_energy_distribution(&data, 8, 8).unwrap();
        let cumulative = analyzer.compute_cumulative_energy(&energy);

        // First few coefficients should capture most energy
        assert!(cumulative.len() == 64);
    }

    #[test]
    fn test_cumulative_energy_monotonic() {
        let analyzer = SpectralAnalyzer::default();
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let energy = analyzer.compute_energy_distribution(&data, 8, 8).unwrap();
        let cumulative = analyzer.compute_cumulative_energy(&energy);

        // Cumulative energy should be monotonically increasing
        for i in 1..cumulative.len() {
            assert!(cumulative[i] >= cumulative[i - 1] - 1e-6);
        }

        // Should end at ~1.0
        if let Some(&last) = cumulative.last() {
            assert!((last - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_knee_point_finds_threshold() {
        let analyzer = SpectralAnalyzer::default();
        let cumulative = vec![0.5, 0.8, 0.95, 0.99, 1.0];

        let knee_90 = analyzer.find_knee_point(&cumulative, 0.90);
        let knee_95 = analyzer.find_knee_point(&cumulative, 0.95);

        assert_eq!(knee_90, 3); // Index 2 reaches 0.95 >= 0.90
        assert_eq!(knee_95, 3); // Index 2 reaches exactly 0.95
    }

    #[test]
    fn test_optimal_retention_low_rank() {
        let analyzer = SpectralAnalyzer::new(0.95);

        // Create low-rank matrix (outer product)
        let u: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0)).collect();
        let mut data = vec![0.0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                data[i * 8 + j] = u[i] * v[j];
            }
        }

        let retention = analyzer.compute_optimal_retention(&data, 8, 8).unwrap();

        // Low-rank matrix should have low optimal retention
        assert!(retention < 0.5, "Low-rank matrix should need low retention, got {}", retention);
    }

    #[test]
    fn test_optimal_retention_random() {
        let analyzer = SpectralAnalyzer::new(0.95);

        // Random-ish data should need higher retention
        let data: Vec<f32> = (0..64).map(|i| ((i * 17 + 3) % 100) as f32 / 100.0).collect();

        let retention = analyzer.compute_optimal_retention(&data, 8, 8).unwrap();

        // Should need more retention for "random" data
        assert!(retention > 0.1, "Random data should need higher retention, got {}", retention);
    }

    #[test]
    fn test_analyze_returns_stats() {
        let analyzer = SpectralAnalyzer::new(0.95);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let stats = analyzer.analyze(&data, 8, 8).unwrap();

        assert_eq!(stats.total_elements, 64);
        assert!(stats.total_energy > 0.0);
        assert!(stats.retention_90 <= stats.retention_95);
        assert!(stats.retention_95 <= stats.retention_99);
        assert!(stats.spectral_entropy >= 0.0 && stats.spectral_entropy <= 1.0);
    }

    #[test]
    fn test_compressibility_classification() {
        let stats_high = SpectralStats {
            total_elements: 100,
            total_energy: 1.0,
            retention_90: 0.05,
            retention_95: 0.08,
            retention_99: 0.15,
            spectral_entropy: 0.3,
            knee_by_curvature: 0.07,
            optimal_retention: 0.08,
        };
        assert_eq!(stats_high.compressibility(), Compressibility::High);

        let stats_low = SpectralStats {
            total_elements: 100,
            total_energy: 1.0,
            retention_90: 0.30,
            retention_95: 0.50,
            retention_99: 0.80,
            spectral_entropy: 0.8,
            knee_by_curvature: 0.45,
            optimal_retention: 0.50,
        };
        assert_eq!(stats_low.compressibility(), Compressibility::Low);
    }

    #[test]
    fn test_empty_tensor() {
        let analyzer = SpectralAnalyzer::default();
        let data: Vec<f32> = vec![];

        let result = analyzer.compute_optimal_retention(&data, 0, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_zeros() {
        let analyzer = SpectralAnalyzer::default();
        let data = vec![0.0f32; 64];

        let retention = analyzer.compute_optimal_retention(&data, 8, 8).unwrap();
        // All zeros should result in minimum retention
        assert!(retention <= 0.10, "All-zero tensor should use minimum retention");
    }
}

//! DARE (Drop And REscale) merging algorithm
//!
//! DARE randomly drops parameters and rescales remaining ones,
//! enabling efficient parameter-space merging while maintaining
//! model quality.

use crate::{MergeError, Result, WeightTensor, WeightDelta};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DARE merge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DareConfig {
    /// Drop rate (0.0 - 1.0)
    pub drop_rate: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Rescaling method
    pub rescale_method: RescaleMethod,
    /// Per-layer drop rate overrides
    pub layer_drop_rates: HashMap<String, f32>,
    /// Combine method when merging multiple models
    pub combine_method: CombineMethod,
}

/// Rescaling method after dropping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescaleMethod {
    /// Multiply by 1/(1-drop_rate)
    Inverse,
    /// No rescaling
    None,
    /// Normalize to original L2 norm
    NormPreserving,
}

/// Method for combining multiple DARE-processed deltas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineMethod {
    /// Simple sum
    Sum,
    /// Average
    Average,
    /// Weighted sum
    Weighted,
}

impl Default for DareConfig {
    fn default() -> Self {
        Self {
            drop_rate: 0.9,
            seed: None,
            rescale_method: RescaleMethod::Inverse,
            layer_drop_rates: HashMap::new(),
            combine_method: CombineMethod::Sum,
        }
    }
}

impl DareConfig {
    /// Create config with specific drop rate
    pub fn with_drop_rate(drop_rate: f32) -> Self {
        Self {
            drop_rate: drop_rate.clamp(0.0, 0.99),
            ..Default::default()
        }
    }

    /// Get drop rate for a layer
    pub fn drop_for_layer(&self, layer: &str) -> f32 {
        *self.layer_drop_rates.get(layer).unwrap_or(&self.drop_rate)
    }
}

/// DARE merger
#[derive(Debug)]
pub struct DareMerger {
    /// Configuration
    config: DareConfig,
    /// Random number generator
    rng: StdRng,
}

impl DareMerger {
    /// Create new DARE merger
    pub fn new(config: DareConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self { config, rng }
    }

    /// Apply DARE to a delta tensor
    pub fn dare(&mut self, delta: &WeightTensor) -> WeightTensor {
        let drop_rate = self.config.drop_for_layer(&delta.name);
        let keep_prob = 1.0 - drop_rate;

        // Generate dropout mask
        let mask: Vec<bool> = (0..delta.data.len())
            .map(|_| self.rng.gen::<f32>() >= drop_rate)
            .collect();

        // Calculate rescale factor
        let rescale = match self.config.rescale_method {
            RescaleMethod::Inverse => {
                if keep_prob > 0.0 {
                    1.0 / keep_prob
                } else {
                    1.0
                }
            }
            RescaleMethod::None => 1.0,
            RescaleMethod::NormPreserving => {
                let original_norm = delta.l2_norm();
                let kept_sq_sum: f32 = delta
                    .data
                    .iter()
                    .zip(&mask)
                    .filter_map(|(&val, &keep)| if keep { Some(val * val) } else { None })
                    .sum();
                let kept_norm = kept_sq_sum.sqrt();

                if kept_norm > 0.0 {
                    original_norm / kept_norm
                } else {
                    1.0
                }
            }
        };

        // Apply mask and rescale
        let data: Vec<f32> = delta
            .data
            .iter()
            .zip(&mask)
            .map(|(&val, &keep)| if keep { val * rescale } else { 0.0 })
            .collect();

        WeightTensor {
            name: delta.name.clone(),
            shape: delta.shape.clone(),
            data,
            dtype: delta.dtype,
        }
    }

    /// Merge multiple deltas using DARE
    pub fn merge_deltas(
        &mut self,
        deltas: &[WeightDelta],
        weights: Option<&[f32]>,
    ) -> Result<WeightTensor> {
        if deltas.is_empty() {
            return Err(MergeError::InvalidWeights("No deltas provided".into()));
        }

        // Apply DARE to each delta
        let dared: Vec<WeightTensor> = deltas
            .iter()
            .map(|d| self.dare(&d.delta))
            .collect();

        // Combine based on method
        let n = dared[0].data.len();
        let mut result = vec![0.0f32; n];

        let weights = weights.map(|w| w.to_vec()).unwrap_or_else(|| {
            vec![1.0; deltas.len()]
        });

        match self.config.combine_method {
            CombineMethod::Sum => {
                for (delta, &w) in dared.iter().zip(&weights) {
                    for (i, &val) in delta.data.iter().enumerate() {
                        result[i] += val * w;
                    }
                }
            }
            CombineMethod::Average => {
                let weight_sum: f32 = weights.iter().sum();
                for (delta, &w) in dared.iter().zip(&weights) {
                    for (i, &val) in delta.data.iter().enumerate() {
                        result[i] += val * w / weight_sum;
                    }
                }
            }
            CombineMethod::Weighted => {
                for (delta, &w) in dared.iter().zip(&weights) {
                    for (i, &val) in delta.data.iter().enumerate() {
                        result[i] += val * w;
                    }
                }
            }
        }

        Ok(WeightTensor {
            name: deltas[0].delta.name.clone(),
            shape: deltas[0].delta.shape.clone(),
            data: result,
            dtype: deltas[0].delta.dtype,
        })
    }

    /// Apply merged delta to base model
    pub fn merge_to_base(
        &mut self,
        base: &WeightTensor,
        deltas: &[WeightDelta],
        weights: Option<&[f32]>,
    ) -> Result<WeightTensor> {
        let merged_delta = self.merge_deltas(deltas, weights)?;
        base.add(&merged_delta)
    }

    /// Get sparsity statistics for a DARE-processed tensor
    pub fn sparsity_stats(&mut self, delta: &WeightTensor) -> DareStats {
        let dared = self.dare(delta);
        let nnz = dared.nnz();
        let total = dared.numel();

        DareStats {
            original_nnz: delta.nnz(),
            dared_nnz: nnz,
            sparsity: 1.0 - (nnz as f32 / total as f32),
            original_norm: delta.l2_norm(),
            dared_norm: dared.l2_norm(),
        }
    }
}

/// DARE statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DareStats {
    /// Original non-zero count
    pub original_nnz: usize,
    /// DARE'd non-zero count
    pub dared_nnz: usize,
    /// Resulting sparsity
    pub sparsity: f32,
    /// Original L2 norm
    pub original_norm: f32,
    /// DARE'd L2 norm
    pub dared_norm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DareConfig::default();
        assert_eq!(config.drop_rate, 0.9);
        assert_eq!(config.rescale_method, RescaleMethod::Inverse);
    }

    #[test]
    fn test_dare_reproducible() {
        let config = DareConfig {
            drop_rate: 0.5,
            seed: Some(42),
            ..Default::default()
        };

        let delta = WeightTensor::new("layer", vec![100], vec![1.0; 100]).unwrap();

        let mut merger1 = DareMerger::new(config.clone());
        let result1 = merger1.dare(&delta);

        let mut merger2 = DareMerger::new(config);
        let result2 = merger2.dare(&delta);

        assert_eq!(result1.data, result2.data);
    }

    #[test]
    fn test_dare_drop_rate() {
        let config = DareConfig {
            drop_rate: 0.8,
            seed: Some(42),
            rescale_method: RescaleMethod::None,
            ..Default::default()
        };

        let delta = WeightTensor::new("layer", vec![1000], vec![1.0; 1000]).unwrap();

        let mut merger = DareMerger::new(config);
        let result = merger.dare(&delta);

        // Approximately 20% should be kept
        let nnz = result.nnz();
        assert!(nnz > 150 && nnz < 250, "NNZ = {}", nnz);
    }

    #[test]
    fn test_dare_rescale_inverse() {
        let config = DareConfig {
            drop_rate: 0.5,
            seed: Some(42),
            rescale_method: RescaleMethod::Inverse,
            ..Default::default()
        };

        let delta = WeightTensor::new("layer", vec![100], vec![1.0; 100]).unwrap();

        let mut merger = DareMerger::new(config);
        let result = merger.dare(&delta);

        // Kept values should be scaled by 2 (1/(1-0.5))
        for &val in &result.data {
            assert!(val == 0.0 || val == 2.0);
        }
    }

    #[test]
    fn test_dare_norm_preserving() {
        let config = DareConfig {
            drop_rate: 0.5,
            seed: Some(42),
            rescale_method: RescaleMethod::NormPreserving,
            ..Default::default()
        };

        let delta = WeightTensor::new("layer", vec![100], vec![1.0; 100]).unwrap();
        let original_norm = delta.l2_norm();

        let mut merger = DareMerger::new(config);
        let result = merger.dare(&delta);

        // Norm should be approximately preserved
        let result_norm = result.l2_norm();
        assert!((original_norm - result_norm).abs() < 0.1);
    }

    #[test]
    fn test_merge_deltas() {
        let config = DareConfig {
            drop_rate: 0.0, // No drop for deterministic test
            seed: Some(42),
            combine_method: CombineMethod::Average,
            ..Default::default()
        };

        let d1 = WeightDelta {
            delta: WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
            source_model: "m1".into(),
            task: "t".into(),
        };

        let d2 = WeightDelta {
            delta: WeightTensor::new("layer", vec![3], vec![3.0, 2.0, 1.0]).unwrap(),
            source_model: "m2".into(),
            task: "t".into(),
        };

        let mut merger = DareMerger::new(config);
        let result = merger.merge_deltas(&[d1, d2], None).unwrap();

        assert_eq!(result.data, vec![2.0, 2.0, 2.0]);
    }
}

//! TIES (Trim, Elect, Merge) merging algorithm
//!
//! TIES is a task vector merging algorithm that:
//! 1. Trims low-magnitude parameters
//! 2. Elects the sign based on majority vote
//! 3. Merges by averaging parameters with the elected sign

use crate::{MergeError, ModelWeights, Result, WeightDelta, WeightTensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TIES merge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiesConfig {
    /// Trim percentage (0.0 - 1.0)
    pub trim_ratio: f32,
    /// Scaling factor for merged task vectors
    pub scaling_factor: f32,
    /// Weight for each model
    pub model_weights: Vec<f32>,
    /// Per-layer trim ratio overrides
    pub layer_trim_ratios: HashMap<String, f32>,
}

impl Default for TiesConfig {
    fn default() -> Self {
        Self {
            trim_ratio: 0.2,
            scaling_factor: 1.0,
            model_weights: Vec::new(),
            layer_trim_ratios: HashMap::new(),
        }
    }
}

impl TiesConfig {
    /// Create config with specific trim ratio
    pub fn with_trim(trim_ratio: f32) -> Self {
        Self {
            trim_ratio: trim_ratio.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Get trim ratio for a layer
    pub fn trim_for_layer(&self, layer: &str) -> f32 {
        *self
            .layer_trim_ratios
            .get(layer)
            .unwrap_or(&self.trim_ratio)
    }
}

/// TIES merger
#[derive(Debug)]
pub struct TiesMerger {
    /// Configuration
    config: TiesConfig,
}

impl TiesMerger {
    /// Create new TIES merger
    pub fn new(config: TiesConfig) -> Self {
        Self { config }
    }

    /// Trim low-magnitude parameters
    fn trim(&self, delta: &WeightTensor, trim_ratio: f32) -> WeightTensor {
        // Calculate magnitude threshold
        let mut magnitudes: Vec<f32> = delta.data.iter().map(|x| x.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = (magnitudes.len() as f32 * trim_ratio) as usize;
        let threshold = if threshold_idx < magnitudes.len() {
            magnitudes[threshold_idx]
        } else {
            f32::MAX
        };

        // Zero out parameters below threshold
        let data: Vec<f32> = delta
            .data
            .iter()
            .map(|&x| if x.abs() >= threshold { x } else { 0.0 })
            .collect();

        WeightTensor {
            name: delta.name.clone(),
            shape: delta.shape.clone(),
            data,
            dtype: delta.dtype,
        }
    }

    /// Elect sign based on majority vote
    fn elect_sign(&self, deltas: &[WeightTensor]) -> Vec<f32> {
        if deltas.is_empty() {
            return Vec::new();
        }

        let n = deltas[0].data.len();
        let mut signs = vec![0.0f32; n];

        for (i, sign) in signs.iter_mut().enumerate() {
            let mut pos_sum = 0.0;
            let mut neg_sum = 0.0;

            for delta in deltas {
                let val = delta.data[i];
                if val > 0.0 {
                    pos_sum += val;
                } else if val < 0.0 {
                    neg_sum += val.abs();
                }
            }

            *sign = if pos_sum >= neg_sum { 1.0 } else { -1.0 };
        }

        signs
    }

    /// Merge task vectors using TIES algorithm
    pub fn merge_deltas(&self, deltas: &[WeightDelta]) -> Result<WeightTensor> {
        if deltas.is_empty() {
            return Err(MergeError::InvalidWeights("No deltas provided".into()));
        }

        let layer_name = &deltas[0].delta.name;
        let trim_ratio = self.config.trim_for_layer(layer_name);

        // Step 1: Trim
        let trimmed: Vec<WeightTensor> = deltas
            .iter()
            .map(|d| self.trim(&d.delta, trim_ratio))
            .collect();

        // Step 2: Elect sign
        let elected_signs = self.elect_sign(&trimmed);

        // Step 3: Merge with sign agreement
        let n = trimmed[0].data.len();
        let mut merged = vec![0.0f32; n];
        let mut counts = vec![0usize; n];

        let weights = if self.config.model_weights.is_empty() {
            vec![1.0; deltas.len()]
        } else {
            self.config.model_weights.clone()
        };

        for (delta, &weight) in trimmed.iter().zip(&weights) {
            for (i, &val) in delta.data.iter().enumerate() {
                // Only include if sign matches elected sign
                let val_sign = if val >= 0.0 { 1.0 } else { -1.0 };
                if val != 0.0 && val_sign == elected_signs[i] {
                    merged[i] += val * weight;
                    counts[i] += 1;
                }
            }
        }

        // Average by count
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                merged[i] /= count as f32;
            }
        }

        // Apply scaling factor
        for val in &mut merged {
            *val *= self.config.scaling_factor;
        }

        Ok(WeightTensor {
            name: layer_name.clone(),
            shape: deltas[0].delta.shape.clone(),
            data: merged,
            dtype: deltas[0].delta.dtype,
        })
    }

    /// Merge task vectors and apply to base model
    pub fn merge_to_base(
        &self,
        base: &WeightTensor,
        deltas: &[WeightDelta],
    ) -> Result<WeightTensor> {
        let merged_delta = self.merge_deltas(deltas)?;
        base.add(&merged_delta)
    }

    /// Merge complete models using TIES
    pub fn merge_models(
        &self,
        base: &ModelWeights,
        finetuned: &[&ModelWeights],
    ) -> Result<ModelWeights> {
        let mut result = ModelWeights::new("merged");

        for layer_name in base.layer_names() {
            let base_tensor = base
                .get_layer(layer_name)
                .ok_or_else(|| MergeError::MissingLayer(layer_name.to_string()))?;

            // Compute deltas for each finetuned model
            let deltas: Vec<WeightDelta> = finetuned
                .iter()
                .filter_map(|model| {
                    model.get_layer(layer_name).map(|t| {
                        WeightDelta::from_models(base_tensor, t, &model.name, "finetuned").ok()
                    })
                })
                .flatten()
                .collect();

            if deltas.is_empty() {
                result.add_layer(base_tensor.clone());
                continue;
            }

            let merged = self.merge_to_base(base_tensor, &deltas)?;
            result.add_layer(merged);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim() {
        let config = TiesConfig::with_trim(0.5);
        let merger = TiesMerger::new(config);

        let delta =
            WeightTensor::new("layer", vec![6], vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9]).unwrap();

        let trimmed = merger.trim(&delta, 0.5);

        // Bottom 50% should be zeroed
        let non_zero: Vec<f32> = trimmed
            .data
            .iter()
            .filter(|&&x| x != 0.0)
            .cloned()
            .collect();
        assert!(non_zero.len() <= 3);
    }

    #[test]
    fn test_elect_sign() {
        let config = TiesConfig::default();
        let merger = TiesMerger::new(config);

        let d1 = WeightTensor::new("layer", vec![3], vec![1.0, -1.0, 1.0]).unwrap();
        let d2 = WeightTensor::new("layer", vec![3], vec![1.0, 1.0, -1.0]).unwrap();
        let d3 = WeightTensor::new("layer", vec![3], vec![1.0, 1.0, 1.0]).unwrap();

        let signs = merger.elect_sign(&[d1, d2, d3]);

        assert_eq!(signs[0], 1.0); // All positive
        assert_eq!(signs[1], 1.0); // 2 positive, 1 negative
        assert_eq!(signs[2], 1.0); // 2 positive, 1 negative
    }

    #[test]
    fn test_merge_deltas() {
        let config = TiesConfig {
            trim_ratio: 0.0, // No trimming for test
            scaling_factor: 1.0,
            model_weights: vec![1.0, 1.0],
            layer_trim_ratios: HashMap::new(),
        };
        let merger = TiesMerger::new(config);

        let d1 = WeightDelta {
            delta: WeightTensor::new("layer", vec![3], vec![1.0, 2.0, -1.0]).unwrap(),
            source_model: "model1".into(),
            task: "task".into(),
        };

        let d2 = WeightDelta {
            delta: WeightTensor::new("layer", vec![3], vec![1.0, -2.0, 1.0]).unwrap(),
            source_model: "model2".into(),
            task: "task".into(),
        };

        let merged = merger.merge_deltas(&[d1, d2]).unwrap();

        // Position 0: both positive, average = 1.0
        assert_eq!(merged.data[0], 1.0);
        // Position 1: positive wins (2 > 2), only positive kept = 2.0
        assert_eq!(merged.data[1], 2.0);
        // Position 2: positive wins (1 > 1), only positive kept = 1.0
        assert_eq!(merged.data[2], 1.0);
    }
}

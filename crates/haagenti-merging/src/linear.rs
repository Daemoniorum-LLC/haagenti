//! Linear weight averaging for model merging

use crate::{MergeError, ModelWeights, Result, WeightTensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Linear merge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Weights for each model (should sum to 1.0)
    pub weights: Vec<f32>,
    /// Normalize weights to sum to 1.0
    pub normalize: bool,
    /// Per-layer weight overrides
    pub layer_weights: HashMap<String, Vec<f32>>,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            weights: vec![0.5, 0.5],
            normalize: true,
            layer_weights: HashMap::new(),
        }
    }
}

impl LinearConfig {
    /// Create config for N models with equal weights
    pub fn equal(n: usize) -> Self {
        let w = 1.0 / n as f32;
        Self {
            weights: vec![w; n],
            normalize: false,
            layer_weights: HashMap::new(),
        }
    }

    /// Create config with specific weights
    pub fn weighted(weights: Vec<f32>) -> Self {
        Self {
            weights,
            normalize: true,
            layer_weights: HashMap::new(),
        }
    }

    /// Validate configuration
    pub fn validate(&self, n_models: usize) -> Result<()> {
        if self.weights.len() != n_models {
            return Err(MergeError::ConfigError(format!(
                "Expected {} weights, got {}",
                n_models,
                self.weights.len()
            )));
        }

        for &w in &self.weights {
            if w < 0.0 {
                return Err(MergeError::ConfigError(
                    "Weights must be non-negative".into(),
                ));
            }
        }

        Ok(())
    }

    /// Get normalized weights
    pub fn normalized_weights(&self) -> Vec<f32> {
        if !self.normalize {
            return self.weights.clone();
        }

        let sum: f32 = self.weights.iter().sum();
        if sum == 0.0 {
            return vec![1.0 / self.weights.len() as f32; self.weights.len()];
        }

        self.weights.iter().map(|w| w / sum).collect()
    }

    /// Get weights for a specific layer
    pub fn weights_for_layer(&self, layer: &str) -> Vec<f32> {
        self.layer_weights
            .get(layer)
            .cloned()
            .unwrap_or_else(|| self.normalized_weights())
    }
}

/// Linear merger for averaging model weights
#[derive(Debug)]
pub struct LinearMerger {
    /// Configuration
    config: LinearConfig,
}

impl LinearMerger {
    /// Create new linear merger
    pub fn new(config: LinearConfig) -> Self {
        Self { config }
    }

    /// Merge multiple weight tensors
    pub fn merge_tensors(&self, tensors: &[&WeightTensor]) -> Result<WeightTensor> {
        if tensors.is_empty() {
            return Err(MergeError::InvalidWeights("No tensors provided".into()));
        }

        let weights = self.config.weights_for_layer(&tensors[0].name);
        if weights.len() != tensors.len() {
            return Err(MergeError::ConfigError(format!(
                "Weight count {} doesn't match tensor count {}",
                weights.len(),
                tensors.len()
            )));
        }

        // Verify all shapes match
        let shape = &tensors[0].shape;
        for tensor in &tensors[1..] {
            if &tensor.shape != shape {
                return Err(MergeError::ShapeMismatch {
                    expected: shape.clone(),
                    got: tensor.shape.clone(),
                });
            }
        }

        // Compute weighted average
        let mut result = WeightTensor::zeros(&tensors[0].name, shape.clone());

        for (tensor, &weight) in tensors.iter().zip(&weights) {
            for (i, &val) in tensor.data.iter().enumerate() {
                result.data[i] += val * weight;
            }
        }

        Ok(result)
    }

    /// Merge complete models
    pub fn merge_models(&self, models: &[&ModelWeights]) -> Result<ModelWeights> {
        if models.is_empty() {
            return Err(MergeError::InvalidWeights("No models provided".into()));
        }

        self.config.validate(models.len())?;

        // Verify compatibility
        let base = models[0];
        for model in &models[1..] {
            if !base.is_compatible(model) {
                return Err(MergeError::IncompatibleModels(format!(
                    "{} and {} have different architectures",
                    base.name, model.name
                )));
            }
        }

        let mut result = ModelWeights::new("merged");

        for layer_name in base.layer_names() {
            let tensors: Vec<&WeightTensor> = models
                .iter()
                .filter_map(|m| m.get_layer(layer_name))
                .collect();

            if tensors.len() != models.len() {
                return Err(MergeError::MissingLayer(layer_name.to_string()));
            }

            let merged = self.merge_tensors(&tensors)?;
            result.add_layer(merged);
        }

        Ok(result)
    }

    /// Interpolate between two tensors
    pub fn interpolate(a: &WeightTensor, b: &WeightTensor, t: f32) -> Result<WeightTensor> {
        if a.shape != b.shape {
            return Err(MergeError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }

        let data: Vec<f32> = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(va, vb)| va * (1.0 - t) + vb * t)
            .collect();

        Ok(WeightTensor {
            name: a.name.clone(),
            shape: a.shape.clone(),
            data,
            dtype: a.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_equal() {
        let config = LinearConfig::equal(4);
        assert_eq!(config.weights, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_normalized_weights() {
        let config = LinearConfig::weighted(vec![1.0, 3.0]);
        let normalized = config.normalized_weights();
        assert_eq!(normalized, vec![0.25, 0.75]);
    }

    #[test]
    fn test_merge_tensors() {
        let t1 = WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let t2 = WeightTensor::new("layer", vec![3], vec![3.0, 4.0, 5.0]).unwrap();

        let config = LinearConfig::equal(2);
        let merger = LinearMerger::new(config);

        let result = merger.merge_tensors(&[&t1, &t2]).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_interpolate() {
        let a = WeightTensor::new("layer", vec![3], vec![0.0, 0.0, 0.0]).unwrap();
        let b = WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        let result = LinearMerger::interpolate(&a, &b, 0.5).unwrap();
        assert_eq!(result.data, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_layer_specific_weights() {
        let mut config = LinearConfig::equal(2);
        config
            .layer_weights
            .insert("important_layer".into(), vec![0.9, 0.1]);

        let weights = config.weights_for_layer("important_layer");
        assert_eq!(weights, vec![0.9, 0.1]);

        let weights = config.weights_for_layer("other_layer");
        assert_eq!(weights, vec![0.5, 0.5]);
    }
}

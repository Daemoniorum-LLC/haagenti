//! Weight tensor utilities for model merging

use crate::{MergeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Weight tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Flattened data
    pub data: Vec<f32>,
    /// Data type
    pub dtype: DataType,
}

/// Data type for weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

impl WeightTensor {
    /// Create new weight tensor
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(MergeError::ShapeMismatch {
                expected: vec![expected_size],
                got: vec![data.len()],
            });
        }

        Ok(Self {
            name: name.into(),
            shape,
            data,
            dtype: DataType::Float32,
        })
    }

    /// Create zero tensor
    pub fn zeros(name: impl Into<String>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            name: name.into(),
            shape,
            data: vec![0.0; size],
            dtype: DataType::Float32,
        }
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 0.0 {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }

    /// Element-wise add
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data,
            dtype: self.dtype,
        })
    }

    /// Element-wise subtract
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Ok(Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data,
            dtype: self.dtype,
        })
    }

    /// Scalar multiply
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x * factor).collect(),
            dtype: self.dtype,
        }
    }

    /// Element-wise multiply (Hadamard product)
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Ok(Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data,
            dtype: self.dtype,
        })
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> Result<f32> {
        self.check_compatible(other)?;
        Ok(self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum())
    }

    /// Check compatibility for operations
    fn check_compatible(&self, other: &Self) -> Result<()> {
        if self.shape != other.shape {
            return Err(MergeError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        Ok(())
    }

    /// Get sign mask (1.0 for positive, -1.0 for negative, 0.0 for zero)
    pub fn sign(&self) -> Self {
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.signum()).collect(),
            dtype: self.dtype,
        }
    }

    /// Create mask where |value| > threshold
    pub fn magnitude_mask(&self, threshold: f32) -> Vec<bool> {
        self.data.iter().map(|x| x.abs() > threshold).collect()
    }

    /// Apply mask (zero out where mask is false)
    pub fn apply_mask(&self, mask: &[bool]) -> Self {
        assert_eq!(mask.len(), self.data.len());
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(mask)
                .map(|(x, &m)| if m { *x } else { 0.0 })
                .collect(),
            dtype: self.dtype,
        }
    }

    /// Count non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&x| x != 0.0).count()
    }

    /// Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.nnz() as f32 / self.numel() as f32)
    }
}

/// Weight delta (task vector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDelta {
    /// Delta tensor
    pub delta: WeightTensor,
    /// Source model name
    pub source_model: String,
    /// Task description
    pub task: String,
}

impl WeightDelta {
    /// Compute delta from base and fine-tuned model weights
    pub fn from_models(
        base: &WeightTensor,
        finetuned: &WeightTensor,
        source_model: impl Into<String>,
        task: impl Into<String>,
    ) -> Result<Self> {
        let delta = finetuned.sub(base)?;
        Ok(Self {
            delta,
            source_model: source_model.into(),
            task: task.into(),
        })
    }

    /// Apply delta to base model with scaling
    pub fn apply(&self, base: &WeightTensor, scale: f32) -> Result<WeightTensor> {
        let scaled = self.delta.scale(scale);
        base.add(&scaled)
    }
}

/// Complete model weights
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Model name
    pub name: String,
    /// Weights by layer name
    pub layers: HashMap<String, WeightTensor>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ModelWeights {
    /// Create new model weights
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add layer weights
    pub fn add_layer(&mut self, tensor: WeightTensor) {
        self.layers.insert(tensor.name.clone(), tensor);
    }

    /// Get layer weights
    pub fn get_layer(&self, name: &str) -> Option<&WeightTensor> {
        self.layers.get(name)
    }

    /// Layer names
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.keys().map(|s| s.as_str()).collect()
    }

    /// Total parameters
    pub fn total_params(&self) -> usize {
        self.layers.values().map(|t| t.numel()).sum()
    }

    /// Check compatibility with another model
    pub fn is_compatible(&self, other: &Self) -> bool {
        if self.layers.len() != other.layers.len() {
            return false;
        }

        for (name, tensor) in &self.layers {
            match other.layers.get(name) {
                Some(other_tensor) => {
                    if tensor.shape != other_tensor.shape {
                        return false;
                    }
                }
                None => return false,
            }
        }

        true
    }

    /// Compute task vector (delta) from base model
    pub fn compute_delta(&self, base: &Self) -> Result<HashMap<String, WeightDelta>> {
        let mut deltas = HashMap::new();

        for (name, tensor) in &self.layers {
            let base_tensor = base.layers.get(name).ok_or_else(|| {
                MergeError::MissingLayer(name.clone())
            })?;

            let delta = WeightDelta::from_models(
                base_tensor,
                tensor,
                &self.name,
                "finetuned",
            )?;

            deltas.insert(name.clone(), delta);
        }

        Ok(deltas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_tensor() {
        let tensor = WeightTensor::new("test", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[test]
    fn test_l2_norm() {
        let tensor = WeightTensor::new("test", vec![4], vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        assert_eq!(tensor.l2_norm(), 5.0);
    }

    #[test]
    fn test_add_sub() {
        let a = WeightTensor::new("a", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = WeightTensor::new("b", vec![3], vec![4.0, 5.0, 6.0]).unwrap();

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);

        let diff = b.sub(&a).unwrap();
        assert_eq!(diff.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let tensor = WeightTensor::new("test", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let scaled = tensor.scale(2.0);
        assert_eq!(scaled.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sign() {
        let tensor = WeightTensor::new("test", vec![4], vec![-1.0, 0.0, 2.0, -3.0]).unwrap();
        let signs = tensor.sign();
        assert_eq!(signs.data, vec![-1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_magnitude_mask() {
        let tensor = WeightTensor::new("test", vec![4], vec![0.1, 0.5, 0.3, 0.7]).unwrap();
        let mask = tensor.magnitude_mask(0.4);
        assert_eq!(mask, vec![false, true, false, true]);
    }

    #[test]
    fn test_weight_delta() {
        let base = WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let finetuned = WeightTensor::new("layer", vec![3], vec![1.5, 2.5, 3.5]).unwrap();

        let delta = WeightDelta::from_models(&base, &finetuned, "model", "task").unwrap();
        assert_eq!(delta.delta.data, vec![0.5, 0.5, 0.5]);

        let result = delta.apply(&base, 1.0).unwrap();
        assert_eq!(result.data, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_model_weights() {
        let mut model = ModelWeights::new("test_model");
        let tensor = WeightTensor::new("layer1", vec![10], vec![0.0; 10]).unwrap();
        model.add_layer(tensor);

        assert_eq!(model.total_params(), 10);
        assert!(model.get_layer("layer1").is_some());
    }
}

//! Spherical linear interpolation (SLERP) for model merging

use crate::{MergeError, Result, WeightTensor};
use serde::{Deserialize, Serialize};

/// SLERP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlerpConfig {
    /// Interpolation factor (0.0 = model A, 1.0 = model B)
    pub t: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Fallback to linear interpolation when vectors are nearly parallel
    pub linear_fallback: bool,
}

impl Default for SlerpConfig {
    fn default() -> Self {
        Self {
            t: 0.5,
            epsilon: 1e-8,
            linear_fallback: true,
        }
    }
}

impl SlerpConfig {
    /// Create config with specific interpolation factor
    pub fn with_t(t: f32) -> Self {
        Self {
            t: t.clamp(0.0, 1.0),
            ..Default::default()
        }
    }
}

/// SLERP merger for smooth weight interpolation
#[derive(Debug)]
pub struct SlerpMerger {
    /// Configuration
    config: SlerpConfig,
}

impl SlerpMerger {
    /// Create new SLERP merger
    pub fn new(config: SlerpConfig) -> Self {
        Self { config }
    }

    /// SLERP between two weight tensors
    pub fn slerp(&self, a: &WeightTensor, b: &WeightTensor) -> Result<WeightTensor> {
        if a.shape != b.shape {
            return Err(MergeError::ShapeMismatch {
                expected: a.shape.clone(),
                got: b.shape.clone(),
            });
        }

        // Normalize vectors
        let norm_a = a.l2_norm();
        let norm_b = b.l2_norm();

        if norm_a < self.config.epsilon || norm_b < self.config.epsilon {
            // One vector is zero, fall back to linear
            return self.linear_interpolate(a, b);
        }

        // Compute dot product of normalized vectors
        let dot: f32 = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(va, vb)| (va / norm_a) * (vb / norm_b))
            .sum();

        // Clamp dot product to [-1, 1]
        let dot = dot.clamp(-1.0, 1.0);

        // If vectors are nearly parallel, use linear interpolation
        if dot.abs() > 1.0 - self.config.epsilon {
            if self.config.linear_fallback {
                return self.linear_interpolate(a, b);
            }
        }

        // Compute SLERP
        let theta = dot.acos();
        let sin_theta = theta.sin();

        if sin_theta < self.config.epsilon {
            return self.linear_interpolate(a, b);
        }

        let t = self.config.t;
        let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
        let scale_b = (t * theta).sin() / sin_theta;

        // Interpolate norms
        let interp_norm = norm_a * (1.0 - t) + norm_b * t;

        let data: Vec<f32> = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(va, vb)| {
                let unit_a = va / norm_a;
                let unit_b = vb / norm_b;
                (scale_a * unit_a + scale_b * unit_b) * interp_norm
            })
            .collect();

        Ok(WeightTensor {
            name: a.name.clone(),
            shape: a.shape.clone(),
            data,
            dtype: a.dtype,
        })
    }

    /// Linear interpolation fallback
    fn linear_interpolate(&self, a: &WeightTensor, b: &WeightTensor) -> Result<WeightTensor> {
        let t = self.config.t;
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

    /// SLERP with per-layer interpolation factors
    pub fn slerp_with_schedule(
        &self,
        a: &WeightTensor,
        b: &WeightTensor,
        t: f32,
    ) -> Result<WeightTensor> {
        let mut merger = Self::new(SlerpConfig::with_t(t));
        merger.slerp(a, b)
    }

    /// Generate interpolation schedule for layers
    pub fn gradient_schedule(num_layers: usize, start_t: f32, end_t: f32) -> Vec<f32> {
        (0..num_layers)
            .map(|i| {
                let ratio = i as f32 / (num_layers - 1).max(1) as f32;
                start_t + (end_t - start_t) * ratio
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slerp_config() {
        let config = SlerpConfig::with_t(0.3);
        assert_eq!(config.t, 0.3);
    }

    #[test]
    fn test_slerp_midpoint() {
        let a = WeightTensor::new("layer", vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let b = WeightTensor::new("layer", vec![3], vec![0.0, 1.0, 0.0]).unwrap();

        let config = SlerpConfig::with_t(0.5);
        let merger = SlerpMerger::new(config);

        let result = merger.slerp(&a, &b).unwrap();

        // At t=0.5, should be on the unit circle between a and b
        let norm = result.l2_norm();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = WeightTensor::new("layer", vec![3], vec![4.0, 5.0, 6.0]).unwrap();

        // t = 0 should give a
        let config = SlerpConfig::with_t(0.0);
        let merger = SlerpMerger::new(config);
        let result = merger.slerp(&a, &b).unwrap();
        for (r, expected) in result.data.iter().zip(&a.data) {
            assert!((r - expected).abs() < 0.01);
        }

        // t = 1 should give b
        let config = SlerpConfig::with_t(1.0);
        let merger = SlerpMerger::new(config);
        let result = merger.slerp(&a, &b).unwrap();
        for (r, expected) in result.data.iter().zip(&b.data) {
            assert!((r - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_gradient_schedule() {
        let schedule = SlerpMerger::gradient_schedule(5, 0.0, 1.0);
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule[0], 0.0);
        assert_eq!(schedule[4], 1.0);
        assert!((schedule[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_slerp_parallel_vectors() {
        let a = WeightTensor::new("layer", vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = WeightTensor::new("layer", vec![3], vec![2.0, 4.0, 6.0]).unwrap();

        let config = SlerpConfig::with_t(0.5);
        let merger = SlerpMerger::new(config);

        // Should fall back to linear interpolation
        let result = merger.slerp(&a, &b).unwrap();
        assert_eq!(result.data, vec![1.5, 3.0, 4.5]);
    }
}

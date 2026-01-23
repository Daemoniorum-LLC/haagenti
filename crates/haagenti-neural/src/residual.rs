//! Residual refinement for neural compression

use crate::{NeuralError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for residual refiner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinerConfig {
    /// Number of refinement iterations
    pub iterations: usize,
    /// Learning rate for refinement
    pub learning_rate: f32,
    /// Whether to use momentum
    pub use_momentum: bool,
    /// Momentum coefficient
    pub momentum: f32,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for RefinerConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            learning_rate: 0.1,
            use_momentum: true,
            momentum: 0.9,
            convergence_threshold: 1e-6,
        }
    }
}

/// Residual refiner for improving reconstruction quality
#[derive(Debug)]
pub struct ResidualRefiner {
    config: RefinerConfig,
}

impl ResidualRefiner {
    /// Create a new refiner
    pub fn new(config: RefinerConfig) -> Self {
        Self { config }
    }

    /// Refine residuals to minimize reconstruction error
    pub fn refine(
        &self,
        original: &[f32],
        quantized: &[f32],
        initial_residuals: &[i8],
        scale: f32,
    ) -> Result<(Vec<i8>, f32)> {
        if original.len() != quantized.len() || original.len() != initial_residuals.len() {
            return Err(NeuralError::DimensionMismatch {
                expected: original.len(),
                actual: quantized.len(),
            });
        }

        // Convert residuals to float for refinement
        let mut residuals: Vec<f32> = initial_residuals
            .iter()
            .map(|&r| r as f32 * scale)
            .collect();

        let mut momentum_buffer: Vec<f32> = vec![0.0; residuals.len()];
        let mut best_mse = self.compute_mse(original, quantized, &residuals);
        let mut best_residuals = residuals.clone();

        for _ in 0..self.config.iterations {
            // Compute gradients
            let gradients: Vec<f32> = original
                .iter()
                .zip(quantized.iter())
                .zip(residuals.iter())
                .map(|((o, q), r)| {
                    let reconstruction = q + r;
                    2.0 * (reconstruction - o)
                })
                .collect();

            // Update with momentum
            for (i, (r, g)) in residuals.iter_mut().zip(gradients.iter()).enumerate() {
                if self.config.use_momentum {
                    momentum_buffer[i] =
                        self.config.momentum * momentum_buffer[i] - self.config.learning_rate * g;
                    *r += momentum_buffer[i];
                } else {
                    *r -= self.config.learning_rate * g;
                }
            }

            // Check convergence
            let mse = self.compute_mse(original, quantized, &residuals);
            if mse < best_mse {
                best_mse = mse;
                best_residuals = residuals.clone();
            }

            if (best_mse - mse).abs() < self.config.convergence_threshold {
                break;
            }
        }

        // Find optimal scale for quantized residuals
        let new_scale = self.find_optimal_scale(&best_residuals);

        // Quantize to i8
        let quantized_residuals: Vec<i8> = best_residuals
            .iter()
            .map(|&r| (r / new_scale).clamp(-127.0, 127.0) as i8)
            .collect();

        Ok((quantized_residuals, new_scale))
    }

    /// Compute MSE with residuals applied
    fn compute_mse(&self, original: &[f32], quantized: &[f32], residuals: &[f32]) -> f32 {
        if original.is_empty() {
            return 0.0;
        }

        let sum: f32 = original
            .iter()
            .zip(quantized.iter())
            .zip(residuals.iter())
            .map(|((o, q), r)| {
                let reconstruction = q + r;
                (o - reconstruction).powi(2)
            })
            .sum();

        sum / original.len() as f32
    }

    /// Find optimal scale to minimize quantization error
    fn find_optimal_scale(&self, residuals: &[f32]) -> f32 {
        let max_abs = residuals.iter().map(|r| r.abs()).fold(0.0f32, f32::max);

        if max_abs < 1e-8 {
            return 1.0;
        }

        // Scale to fit in i8 range with some headroom
        max_abs / 120.0
    }

    /// Apply multi-level residual coding
    pub fn multi_level_refine(
        &self,
        original: &[f32],
        quantized: &[f32],
        levels: usize,
    ) -> Result<MultiLevelResidual> {
        if original.len() != quantized.len() {
            return Err(NeuralError::DimensionMismatch {
                expected: original.len(),
                actual: quantized.len(),
            });
        }

        let mut current_reconstruction = quantized.to_vec();
        let mut residual_levels = Vec::with_capacity(levels);

        for level in 0..levels {
            // Compute residual at this level
            let residual: Vec<f32> = original
                .iter()
                .zip(current_reconstruction.iter())
                .map(|(o, r)| o - r)
                .collect();

            // Find scale and quantize
            let scale = self.find_optimal_scale(&residual);
            let quantized_residual: Vec<i8> = residual
                .iter()
                .map(|&r| (r / scale).clamp(-127.0, 127.0) as i8)
                .collect();

            // Update reconstruction
            for (recon, &qr) in current_reconstruction
                .iter_mut()
                .zip(quantized_residual.iter())
            {
                *recon += qr as f32 * scale;
            }

            // Compute MSE at this level
            let mse = self.compute_mse(
                original,
                &vec![0.0; original.len()],
                &current_reconstruction,
            );

            residual_levels.push(ResidualLevel {
                level,
                residuals: quantized_residual,
                scale,
                mse,
            });

            // Early termination if good enough
            if mse < self.config.convergence_threshold {
                break;
            }
        }

        Ok(MultiLevelResidual {
            levels: residual_levels,
            final_mse: self.compute_mse(
                original,
                &vec![0.0; original.len()],
                &current_reconstruction,
            ),
        })
    }
}

/// A single level of residual coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualLevel {
    /// Level index
    pub level: usize,
    /// Quantized residuals
    pub residuals: Vec<i8>,
    /// Scale factor
    pub scale: f32,
    /// MSE after this level
    pub mse: f32,
}

/// Multi-level residual coding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelResidual {
    /// Residual levels
    pub levels: Vec<ResidualLevel>,
    /// Final MSE
    pub final_mse: f32,
}

impl MultiLevelResidual {
    /// Apply all levels to a quantized vector
    pub fn apply(&self, quantized: &mut [f32]) {
        for level in &self.levels {
            for (q, &r) in quantized.iter_mut().zip(level.residuals.iter()) {
                *q += r as f32 * level.scale;
            }
        }
    }

    /// Total size in bytes
    pub fn size(&self) -> usize {
        self.levels.iter().map(|l| l.residuals.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_residuals() {
        let refiner = ResidualRefiner::new(RefinerConfig::default());

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        let initial_residuals = vec![0i8; 5];
        let scale = 0.01;

        let (refined, new_scale) = refiner
            .refine(&original, &quantized, &initial_residuals, scale)
            .unwrap();

        assert_eq!(refined.len(), 5);
        assert!(new_scale > 0.0);
    }

    #[test]
    fn test_multi_level() {
        let refiner = ResidualRefiner::new(RefinerConfig::default());

        let original: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let quantized: Vec<f32> = original.iter().map(|x| (x * 2.0).round() / 2.0).collect();

        let multi = refiner
            .multi_level_refine(&original, &quantized, 3)
            .unwrap();

        assert!(!multi.levels.is_empty());
        // Each level should reduce MSE
        if multi.levels.len() > 1 {
            assert!(multi.levels[0].mse >= multi.levels.last().unwrap().mse);
        }
    }

    #[test]
    fn test_apply_multi_level() {
        let multi = MultiLevelResidual {
            levels: vec![
                ResidualLevel {
                    level: 0,
                    residuals: vec![10, -10, 20, -20],
                    scale: 0.01,
                    mse: 0.1,
                },
                ResidualLevel {
                    level: 1,
                    residuals: vec![5, -5, 10, -10],
                    scale: 0.001,
                    mse: 0.01,
                },
            ],
            final_mse: 0.01,
        };

        let mut quantized = vec![1.0, 2.0, 3.0, 4.0];
        multi.apply(&mut quantized);

        assert!((quantized[0] - 1.105).abs() < 0.001);
        assert!((quantized[1] - 1.895).abs() < 0.001);
    }
}

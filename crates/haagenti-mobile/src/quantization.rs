//! INT4 quantization for mobile deployment

use crate::{MobileError, Result};
use serde::{Deserialize, Serialize};

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Group size for per-group quantization
    pub group_size: usize,
    /// Symmetric quantization (vs asymmetric)
    pub symmetric: bool,
    /// Per-channel quantization
    pub per_channel: bool,
    /// Calibration samples
    pub calibration_samples: usize,
    /// Clipping percentile for outliers
    pub clip_percentile: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            symmetric: true,
            per_channel: true,
            calibration_samples: 256,
            clip_percentile: 99.9,
        }
    }
}

/// Quantized tensor storage
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed INT4 values (2 values per byte)
    pub data: Vec<u8>,
    /// Scale factors per group
    pub scales: Vec<f32>,
    /// Zero points per group (for asymmetric)
    pub zero_points: Option<Vec<i8>>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Group size used
    pub group_size: usize,
}

impl QuantizedTensor {
    /// Create empty quantized tensor
    pub fn new(shape: Vec<usize>, group_size: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let packed_size = total_elements.div_ceil(2);
        let num_groups = total_elements.div_ceil(group_size);

        Self {
            data: vec![0; packed_size],
            scales: vec![1.0; num_groups],
            zero_points: None,
            shape,
            group_size,
        }
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get storage size in bytes
    pub fn storage_size(&self) -> usize {
        let data_size = self.data.len();
        let scale_size = self.scales.len() * 4;
        let zp_size = self.zero_points.as_ref().map_or(0, |zp| zp.len());
        data_size + scale_size + zp_size
    }

    /// Compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.numel() * 4; // FP32
        original_size as f32 / self.storage_size() as f32
    }

    /// Dequantize to FP32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.numel());

        for (group_idx, group_scale) in self.scales.iter().enumerate() {
            let start = group_idx * self.group_size;
            let end = (start + self.group_size).min(self.numel());

            let zero_point = self
                .zero_points
                .as_ref()
                .map_or(0, |zp| zp[group_idx] as i32);

            for i in start..end {
                let packed_idx = i / 2;
                let int4_val = if i % 2 == 0 {
                    (self.data[packed_idx] & 0x0F) as i32
                } else {
                    ((self.data[packed_idx] >> 4) & 0x0F) as i32
                };

                // Convert from unsigned (0-15) to signed (-8 to 7)
                let signed_val = int4_val - 8;
                let dequant_val = (signed_val - zero_point) as f32 * group_scale;
                result.push(dequant_val);
            }
        }

        result
    }
}

/// INT4 quantizer
#[derive(Debug)]
pub struct Int4Quantizer {
    /// Configuration
    config: QuantizationConfig,
    /// Calibration data statistics
    calibration_stats: Option<CalibrationStats>,
}

/// Calibration statistics (internal)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CalibrationStats {
    /// Min values per group
    mins: Vec<f32>,
    /// Max values per group
    maxs: Vec<f32>,
    /// Running count
    count: usize,
}

impl Int4Quantizer {
    /// Create new quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_stats: None,
        }
    }

    /// Calibrate quantizer with sample data
    pub fn calibrate(&mut self, samples: &[&[f32]]) -> Result<()> {
        if samples.is_empty() {
            return Err(MobileError::QuantizationError(
                "No calibration samples provided".into(),
            ));
        }

        let sample_len = samples[0].len();
        let num_groups = sample_len.div_ceil(self.config.group_size);

        let mut mins = vec![f32::MAX; num_groups];
        let mut maxs = vec![f32::MIN; num_groups];

        for sample in samples {
            for (group_idx, chunk) in sample.chunks(self.config.group_size).enumerate() {
                for &val in chunk {
                    mins[group_idx] = mins[group_idx].min(val);
                    maxs[group_idx] = maxs[group_idx].max(val);
                }
            }
        }

        // Apply clipping
        if self.config.clip_percentile < 100.0 {
            // In a real implementation, we'd compute actual percentiles
            // For now, just shrink the range slightly
            let shrink = (100.0 - self.config.clip_percentile) / 100.0;
            for i in 0..num_groups {
                let range = maxs[i] - mins[i];
                mins[i] += range * shrink;
                maxs[i] -= range * shrink;
            }
        }

        self.calibration_stats = Some(CalibrationStats {
            mins,
            maxs,
            count: samples.len(),
        });

        Ok(())
    }

    /// Quantize tensor to INT4
    pub fn quantize(&self, tensor: &[f32]) -> Result<QuantizedTensor> {
        let shape = vec![tensor.len()];
        let mut result = QuantizedTensor::new(shape, self.config.group_size);

        // Compute scales per group
        for (group_idx, chunk) in tensor.chunks(self.config.group_size).enumerate() {
            let (min_val, max_val) = if let Some(stats) = &self.calibration_stats {
                (stats.mins[group_idx], stats.maxs[group_idx])
            } else {
                // Dynamic range
                let min = chunk.iter().cloned().fold(f32::MAX, f32::min);
                let max = chunk.iter().cloned().fold(f32::MIN, f32::max);
                (min, max)
            };

            // Symmetric quantization: scale to [-8, 7]
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };
            result.scales[group_idx] = scale;
        }

        // Quantize values
        for (i, &val) in tensor.iter().enumerate() {
            let group_idx = i / self.config.group_size;
            let scale = result.scales[group_idx];

            // Quantize to [-8, 7] then shift to [0, 15]
            let quantized = (val / scale).round().clamp(-8.0, 7.0) as i32 + 8;
            let uint4_val = quantized as u8;

            let packed_idx = i / 2;
            if i % 2 == 0 {
                result.data[packed_idx] = (result.data[packed_idx] & 0xF0) | (uint4_val & 0x0F);
            } else {
                result.data[packed_idx] =
                    (result.data[packed_idx] & 0x0F) | ((uint4_val & 0x0F) << 4);
            }
        }

        Ok(result)
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Check if calibrated
    pub fn is_calibrated(&self) -> bool {
        self.calibration_stats.is_some()
    }
}

/// Quantization error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetrics {
    /// Mean squared error
    pub mse: f32,
    /// Mean absolute error
    pub mae: f32,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Maximum absolute error
    pub max_error: f32,
}

impl QuantizationMetrics {
    /// Compute metrics between original and dequantized tensors
    pub fn compute(original: &[f32], dequantized: &[f32]) -> Self {
        assert_eq!(original.len(), dequantized.len());

        let mut sum_sq_error = 0.0f64;
        let mut sum_abs_error = 0.0f64;
        let mut sum_sq_signal = 0.0f64;
        let mut max_error = 0.0f32;

        for (o, d) in original.iter().zip(dequantized.iter()) {
            let error = (o - d).abs();
            sum_sq_error += (error * error) as f64;
            sum_abs_error += error as f64;
            sum_sq_signal += (o * o) as f64;
            max_error = max_error.max(error);
        }

        let n = original.len() as f64;
        let mse = (sum_sq_error / n) as f32;
        let mae = (sum_abs_error / n) as f32;
        let snr_db = if sum_sq_error > 0.0 {
            (10.0 * (sum_sq_signal / sum_sq_error).log10()) as f32
        } else {
            f32::INFINITY
        };

        Self {
            mse,
            mae,
            snr_db,
            max_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.group_size, 128);
        assert!(config.symmetric);
    }

    #[test]
    fn test_quantized_tensor() {
        let tensor = QuantizedTensor::new(vec![256], 128);
        assert_eq!(tensor.numel(), 256);
        assert_eq!(tensor.scales.len(), 2);
    }

    #[test]
    fn test_quantize_dequantize() {
        let config = QuantizationConfig {
            group_size: 8,
            ..Default::default()
        };
        let quantizer = Int4Quantizer::new(config);

        let original: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let quantized = quantizer.quantize(&original).unwrap();
        let dequantized = quantized.dequantize();

        // Check approximate reconstruction
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.2, "Original: {}, Dequantized: {}", o, d);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let tensor = QuantizedTensor::new(vec![1024], 128);
        let ratio = tensor.compression_ratio();
        // INT4 should be close to 8:1 compression vs FP32
        assert!(ratio > 6.0 && ratio < 9.0, "Compression ratio: {}", ratio);
    }

    #[test]
    fn test_quantization_metrics() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let dequantized = vec![1.1, 1.9, 3.1, 3.9];

        let metrics = QuantizationMetrics::compute(&original, &dequantized);
        assert!(metrics.mse < 0.1);
        assert!(metrics.mae < 0.2);
        assert!(metrics.snr_db > 20.0);
    }

    #[test]
    fn test_calibration() {
        let config = QuantizationConfig {
            group_size: 4,
            ..Default::default()
        };
        let mut quantizer = Int4Quantizer::new(config);

        let samples = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        ];

        let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        quantizer.calibrate(&sample_refs).unwrap();

        assert!(quantizer.is_calibrated());
    }
}

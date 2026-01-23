//! Neural encoder for compressing model weights

use crate::{Codebook, LayerCodebook, LayerType, NeuralError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Target compression ratio
    pub target_ratio: f32,
    /// Maximum quality loss allowed
    pub max_quality_loss: f32,
    /// Whether to use residual encoding
    pub use_residual: bool,
    /// Residual quantization bits
    pub residual_bits: u8,
    /// Batch size for encoding
    pub batch_size: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            target_ratio: 10.0,
            max_quality_loss: 0.01,
            use_residual: true,
            residual_bits: 4,
            batch_size: 1024,
        }
    }
}

/// Encoded tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedTensor {
    /// Original tensor name
    pub name: String,
    /// Original shape
    pub shape: Vec<usize>,
    /// Original dtype
    pub dtype: String,
    /// Layer type for codebook selection
    pub layer_type: LayerType,
    /// Quantized indices
    pub indices: Vec<u16>,
    /// Residual data (if enabled)
    pub residuals: Option<Vec<i8>>,
    /// Scale factor for residuals
    pub residual_scale: f32,
    /// Compression statistics
    pub stats: EncodingStats,
}

/// Statistics from encoding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncodingStats {
    /// Mean squared error
    pub mse: f32,
    /// Peak signal-to-noise ratio
    pub psnr: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
}

/// Neural encoder for model compression
#[derive(Debug)]
pub struct NeuralEncoder {
    config: EncoderConfig,
    codebooks: LayerCodebook,
}

impl NeuralEncoder {
    /// Create a new encoder with codebooks
    pub fn new(config: EncoderConfig, codebooks: LayerCodebook) -> Self {
        Self { config, codebooks }
    }

    /// Encode a single tensor
    pub fn encode_tensor(
        &self,
        name: &str,
        data: &[f32],
        shape: &[usize],
        layer_type: LayerType,
    ) -> Result<EncodedTensor> {
        let codebook = self
            .codebooks
            .get(layer_type)
            .ok_or_else(|| NeuralError::CodebookNotFound(format!("{:?}", layer_type)))?;

        let dim = codebook.config.centroid_dim;
        let num_vectors = data.len() / dim;

        if !data.len().is_multiple_of(dim) {
            return Err(NeuralError::DimensionMismatch {
                expected: (num_vectors + 1) * dim,
                actual: data.len(),
            });
        }

        // Encode to indices
        let indices = codebook.encode_batch(data, dim)?;

        // Compute residuals if enabled
        let (residuals, residual_scale) = if self.config.use_residual {
            self.compute_residuals(data, &indices, codebook)
        } else {
            (None, 0.0)
        };

        // Compute statistics
        let reconstructed = codebook.decode_batch(&indices);
        let mse = self.compute_mse(data, &reconstructed);
        let psnr = self.compute_psnr(mse);

        let original_size = data.len() * 4; // FP32
        let index_size = indices.len() * 2; // U16
        let residual_size = residuals.as_ref().map(|r| r.len()).unwrap_or(0);
        let compressed_size = index_size + residual_size;

        let stats = EncodingStats {
            mse,
            psnr,
            compression_ratio: original_size as f32 / compressed_size as f32,
            original_size,
            compressed_size,
        };

        Ok(EncodedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: "float32".to_string(),
            layer_type,
            indices,
            residuals,
            residual_scale,
            stats,
        })
    }

    /// Compute residuals between original and quantized
    fn compute_residuals(
        &self,
        original: &[f32],
        indices: &[u16],
        codebook: &Codebook,
    ) -> (Option<Vec<i8>>, f32) {
        let reconstructed = codebook.decode_batch(indices);

        // Compute difference
        let residuals: Vec<f32> = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| o - r)
            .collect();

        // Find scale to fit in i8 range
        let max_abs = residuals.iter().map(|r| r.abs()).fold(0.0f32, f32::max);

        if max_abs < 1e-6 {
            return (None, 0.0);
        }

        let scale = max_abs / 127.0;

        // Quantize residuals to i8
        let quantized: Vec<i8> = residuals
            .iter()
            .map(|r| (r / scale).clamp(-127.0, 127.0) as i8)
            .collect();

        (Some(quantized), scale)
    }

    /// Compute mean squared error
    fn compute_mse(&self, original: &[f32], reconstructed: &[f32]) -> f32 {
        if original.len() != reconstructed.len() || original.is_empty() {
            return f32::INFINITY;
        }

        let sum: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum();

        sum / original.len() as f32
    }

    /// Compute PSNR from MSE
    fn compute_psnr(&self, mse: f32) -> f32 {
        if mse < 1e-10 {
            return 100.0; // Perfect reconstruction
        }
        10.0 * (1.0 / mse).log10()
    }

    /// Encode multiple tensors
    pub fn encode_tensors(
        &self,
        tensors: &[(String, Vec<f32>, Vec<usize>, LayerType)],
    ) -> Result<Vec<EncodedTensor>> {
        tensors
            .iter()
            .map(|(name, data, shape, layer_type)| {
                self.encode_tensor(name, data, shape, *layer_type)
            })
            .collect()
    }

    /// Get encoder configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_tensor() {
        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(EncoderConfig::default(), codebooks);

        // Create test data matching centroid dimension (64 for QK)
        let data: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let shape = vec![10, 64];

        let encoded = encoder
            .encode_tensor("test.weight", &data, &shape, LayerType::AttentionQK)
            .unwrap();

        assert_eq!(encoded.name, "test.weight");
        assert_eq!(encoded.indices.len(), 10);
        assert!(encoded.stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_residual_encoding() {
        let config = EncoderConfig {
            use_residual: true,
            ..Default::default()
        };

        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(config, codebooks);

        let data: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let shape = vec![10, 64];

        let encoded = encoder
            .encode_tensor("test.weight", &data, &shape, LayerType::AttentionQK)
            .unwrap();

        assert!(encoded.residuals.is_some());
        assert!(encoded.residual_scale > 0.0);
    }

    #[test]
    fn test_psnr_calculation() {
        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(EncoderConfig::default(), codebooks);

        // Low MSE should give high PSNR (MSE=0.001 -> PSNR=30.0)
        assert!(encoder.compute_psnr(0.001) >= 30.0);
        // High MSE should give low PSNR
        assert!(encoder.compute_psnr(0.1) < 20.0);
    }
}

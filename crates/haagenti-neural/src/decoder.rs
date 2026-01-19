//! Neural decoder for decompressing model weights

use crate::{Codebook, EncodedTensor, LayerCodebook, LayerType, NeuralError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Whether to apply residual refinement
    pub apply_residuals: bool,
    /// Batch size for decoding
    pub batch_size: usize,
    /// Use GPU acceleration (future)
    pub use_gpu: bool,
    /// Prefetch next layer while decoding
    pub prefetch: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            apply_residuals: true,
            batch_size: 1024,
            use_gpu: false,
            prefetch: true,
        }
    }
}

/// Decoded tensor result
#[derive(Debug, Clone)]
pub struct DecodedTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Decoded data
    pub data: Vec<f32>,
    /// Decoding statistics
    pub stats: DecodingStats,
}

/// Statistics from decoding
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodingStats {
    /// Time to decode in microseconds
    pub decode_time_us: u64,
    /// Codebook lookup time
    pub lookup_time_us: u64,
    /// Residual application time
    pub residual_time_us: u64,
    /// Number of vectors decoded
    pub vectors_decoded: usize,
}

/// Neural decoder for model decompression
#[derive(Debug)]
pub struct NeuralDecoder {
    config: DecoderConfig,
    codebooks: LayerCodebook,
}

impl NeuralDecoder {
    /// Create a new decoder with codebooks
    pub fn new(config: DecoderConfig, codebooks: LayerCodebook) -> Self {
        Self { config, codebooks }
    }

    /// Decode an encoded tensor
    pub fn decode_tensor(&self, encoded: &EncodedTensor) -> Result<DecodedTensor> {
        let start = std::time::Instant::now();

        let codebook = self.codebooks.get(encoded.layer_type).ok_or_else(|| {
            NeuralError::CodebookNotFound(format!("{:?}", encoded.layer_type))
        })?;

        // Decode indices to vectors
        let lookup_start = std::time::Instant::now();
        let mut data = codebook.decode_batch(&encoded.indices);
        let lookup_time = lookup_start.elapsed();

        // Apply residuals if present
        let residual_start = std::time::Instant::now();
        if self.config.apply_residuals {
            if let Some(residuals) = encoded.residuals.as_ref() {
                self.apply_residuals(&mut data, residuals, encoded.residual_scale);
            }
        }
        let residual_time = residual_start.elapsed();

        let total_time = start.elapsed();

        let stats = DecodingStats {
            decode_time_us: total_time.as_micros() as u64,
            lookup_time_us: lookup_time.as_micros() as u64,
            residual_time_us: residual_time.as_micros() as u64,
            vectors_decoded: encoded.indices.len(),
        };

        Ok(DecodedTensor {
            name: encoded.name.clone(),
            shape: encoded.shape.clone(),
            data,
            stats,
        })
    }

    /// Apply residual refinement
    fn apply_residuals(&self, data: &mut [f32], residuals: &[i8], scale: f32) {
        for (d, &r) in data.iter_mut().zip(residuals.iter()) {
            *d += r as f32 * scale;
        }
    }

    /// Decode multiple tensors
    pub fn decode_tensors(&self, encoded: &[EncodedTensor]) -> Result<Vec<DecodedTensor>> {
        encoded.iter().map(|e| self.decode_tensor(e)).collect()
    }

    /// Decode with streaming (yields chunks)
    pub fn decode_streaming<'a>(
        &'a self,
        encoded: &'a EncodedTensor,
        chunk_size: usize,
    ) -> impl Iterator<Item = Result<Vec<f32>>> + 'a {
        let codebook = self.codebooks.get(encoded.layer_type);

        encoded.indices.chunks(chunk_size).enumerate().map(move |(i, chunk)| {
            let cb = codebook.ok_or_else(|| {
                NeuralError::CodebookNotFound(format!("{:?}", encoded.layer_type))
            })?;

            let mut data: Vec<f32> = chunk
                .iter()
                .flat_map(|&idx| {
                    cb.get_centroid(idx as usize)
                        .map(|c| c.to_vec())
                        .unwrap_or_else(|| vec![0.0; cb.config.centroid_dim])
                })
                .collect();

            // Apply residuals for this chunk
            if self.config.apply_residuals {
                if let Some(residuals) = encoded.residuals.as_ref() {
                    let dim = cb.config.centroid_dim;
                    let start = i * chunk_size * dim;
                    let end = (start + data.len()).min(residuals.len());
                    if start < residuals.len() {
                        self.apply_residuals(&mut data[..end - start], &residuals[start..end], encoded.residual_scale);
                    }
                }
            }

            Ok(data)
        })
    }

    /// Get decoder configuration
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Estimate decode throughput in GB/s
    pub fn estimate_throughput(&self, stats: &DecodingStats) -> f32 {
        if stats.decode_time_us == 0 {
            return 0.0;
        }

        let bytes_decoded = stats.vectors_decoded * 64 * 4; // Assuming 64-dim FP32
        let seconds = stats.decode_time_us as f64 / 1_000_000.0;
        (bytes_decoded as f64 / seconds / 1e9) as f32
    }
}

/// GPU-accelerated decoder (placeholder for CUDA implementation)
#[derive(Debug)]
pub struct GpuDecoder {
    /// Codebook data on GPU
    codebook_buffers: Vec<GpuBuffer>,
    /// Output buffer
    output_buffer: GpuBuffer,
}

/// Placeholder for GPU buffer
#[derive(Debug, Default)]
struct GpuBuffer {
    size: usize,
    device_ptr: u64,
}

impl GpuDecoder {
    /// Create GPU decoder (requires CUDA)
    pub fn new(_codebooks: &LayerCodebook) -> Result<Self> {
        // In real implementation, this would:
        // 1. Allocate GPU memory for codebooks
        // 2. Copy codebook data to GPU
        // 3. Compile decode kernels

        Ok(Self {
            codebook_buffers: Vec::new(),
            output_buffer: GpuBuffer::default(),
        })
    }

    /// Decode on GPU
    pub fn decode_gpu(&self, _indices: &[u16], _layer_type: LayerType) -> Result<Vec<f32>> {
        // Placeholder - would launch CUDA kernel
        Err(NeuralError::DecodingError("GPU not available".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncoderConfig, NeuralEncoder};

    #[test]
    fn test_decode_tensor() {
        // First encode something
        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(EncoderConfig::default(), codebooks.clone());

        let original: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let shape = vec![10, 64];

        let encoded = encoder
            .encode_tensor("test.weight", &original, &shape, LayerType::AttentionQK)
            .unwrap();

        // Now decode
        let decoder = NeuralDecoder::new(DecoderConfig::default(), codebooks);
        let decoded = decoder.decode_tensor(&encoded).unwrap();

        assert_eq!(decoded.name, "test.weight");
        assert_eq!(decoded.shape, vec![10, 64]);
        assert_eq!(decoded.data.len(), 640);
    }

    #[test]
    fn test_decode_streaming() {
        let codebooks = LayerCodebook::with_defaults("test");
        let encoder = NeuralEncoder::new(EncoderConfig::default(), codebooks.clone());

        let original: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let encoded = encoder
            .encode_tensor("test", &original, &[10, 64], LayerType::AttentionQK)
            .unwrap();

        let decoder = NeuralDecoder::new(DecoderConfig::default(), codebooks);

        let chunks: Vec<_> = decoder
            .decode_streaming(&encoded, 3)
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_residual_application() {
        let decoder = NeuralDecoder::new(
            DecoderConfig::default(),
            LayerCodebook::with_defaults("test"),
        );

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let residuals = vec![10i8, -10, 20, -20];
        let scale = 0.01;

        decoder.apply_residuals(&mut data, &residuals, scale);

        assert!((data[0] - 1.1).abs() < 0.001);
        assert!((data[1] - 1.9).abs() < 0.001);
        assert!((data[2] - 3.2).abs() < 0.001);
        assert!((data[3] - 3.8).abs() < 0.001);
    }
}

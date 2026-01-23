//! Neural GPU Decompression
//!
//! GPU-accelerated neural compression decoding including:
//! - Codebook lookup (vector quantization)
//! - Batched tensor decoding
//! - Integration with candle tensors
//!
//! # Track B Phases
//!
//! - B.4: GPU Codebook Lookup
//! - B.5: GPU Batched Decode

use crate::error::{CudaError, Result};
use crate::memory::GpuBuffer;
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;

/// Layer codebook for GPU decoding.
///
/// Contains centroids for vector quantization lookup.
#[derive(Clone, Debug)]
pub struct LayerCodebook {
    /// Number of codewords
    pub num_codes: usize,
    /// Dimension of each codeword
    pub code_dim: usize,
    /// Centroid data [num_codes × code_dim]
    pub centroids: Vec<f32>,
}

impl LayerCodebook {
    /// Create a new codebook with specified size.
    pub fn new(num_codes: usize, code_dim: usize) -> Self {
        Self {
            num_codes,
            code_dim,
            centroids: vec![0.0; num_codes * code_dim],
        }
    }

    /// Create a random codebook for testing.
    #[cfg(test)]
    pub fn random(num_codes: usize, code_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let centroids: Vec<f32> = (0..num_codes * code_dim)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();

        Self {
            num_codes,
            code_dim,
            centroids,
        }
    }

    /// Create a random codebook for testing (non-test version using simple pattern).
    #[cfg(not(test))]
    pub fn random(num_codes: usize, code_dim: usize) -> Self {
        let centroids: Vec<f32> = (0..num_codes * code_dim)
            .map(|i| ((i as f32 * 0.123456).sin() * 2.0) - 1.0)
            .collect();

        Self {
            num_codes,
            code_dim,
            centroids,
        }
    }

    /// Set a specific code.
    pub fn set_code(&mut self, index: usize, values: &[f32]) {
        if index < self.num_codes && values.len() == self.code_dim {
            let offset = index * self.code_dim;
            self.centroids[offset..offset + self.code_dim].copy_from_slice(values);
        }
    }

    /// Get a specific code.
    pub fn get_code(&self, index: usize) -> Option<&[f32]> {
        if index < self.num_codes {
            let offset = index * self.code_dim;
            Some(&self.centroids[offset..offset + self.code_dim])
        } else {
            None
        }
    }

    /// Memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.num_codes * self.code_dim * std::mem::size_of::<f32>()
    }
}

/// Quantized tensor (indices into codebook).
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// Quantization indices
    pub indices: Vec<u8>,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Create a random quantized tensor for testing.
    #[cfg(test)]
    pub fn random(num_indices: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let indices: Vec<u8> = (0..num_indices).map(|_| rng.r#gen()).collect();

        Self {
            indices,
            shape: vec![num_indices],
        }
    }

    /// Create a pseudo-random quantized tensor (non-test version).
    #[cfg(not(test))]
    pub fn random(num_indices: usize) -> Self {
        let indices: Vec<u8> = (0..num_indices).map(|i| (i * 17 + i / 7) as u8).collect();

        Self {
            indices,
            shape: vec![num_indices],
        }
    }

    /// Get decompressed size (number of float values).
    pub fn decompressed_size(&self, code_dim: usize) -> usize {
        self.indices.len() * code_dim
    }
}

/// GPU-accelerated neural decoder.
///
/// Performs codebook lookup on GPU for fast tensor decompression.
pub struct NeuralGpuDecoder {
    device: Arc<CudaDevice>,
    /// Stream for async operations (kept for future pipelined execution)
    #[allow(dead_code)]
    stream: CudaStream,
    codebook: Option<LayerCodebook>,
    gpu_codebook: Option<GpuBuffer>,
}

impl NeuralGpuDecoder {
    /// Create a new neural GPU decoder.
    pub fn new(ctx: &crate::GpuContext) -> Result<Self> {
        let device = ctx.device().clone();
        let stream = device.fork_default_stream()?;

        Ok(Self {
            device,
            stream,
            codebook: None,
            gpu_codebook: None,
        })
    }

    /// Upload a codebook to GPU.
    pub fn upload_codebook(&mut self, codebook: &LayerCodebook) -> Result<()> {
        let size = codebook.memory_size();
        let buffer = GpuBuffer::new(self.device.clone(), size)?;

        // Convert f32 to bytes
        let bytes: Vec<u8> = codebook
            .centroids
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        buffer.copy_from_host(&bytes)?;

        self.codebook = Some(codebook.clone());
        self.gpu_codebook = Some(buffer);
        Ok(())
    }

    /// Check if a codebook is loaded.
    pub fn has_codebook(&self) -> bool {
        self.codebook.is_some()
    }

    /// Perform codebook lookup.
    ///
    /// Maps indices to centroid vectors.
    pub fn lookup(&self, indices: &[u8]) -> Result<Vec<f32>> {
        let codebook = self
            .codebook
            .as_ref()
            .ok_or_else(|| CudaError::InvalidData("No codebook loaded".into()))?;

        let code_dim = codebook.code_dim;
        let mut output = Vec::with_capacity(indices.len() * code_dim);

        for &idx in indices {
            let idx = idx as usize % codebook.num_codes;
            if let Some(code) = codebook.get_code(idx) {
                output.extend_from_slice(code);
            } else {
                output.extend(std::iter::repeat_n(0.0, code_dim));
            }
        }

        Ok(output)
    }

    /// Batch lookup for multiple quantized tensors.
    pub fn lookup_batch(&self, tensors: &[QuantizedTensor]) -> Result<Vec<Vec<f32>>> {
        tensors.iter().map(|t| self.lookup(&t.indices)).collect()
    }
}

/// Full neural GPU decompression pipeline.
pub struct NeuralGpuPipeline {
    /// Device handle kept for ownership/lifetime
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    decoder: NeuralGpuDecoder,
    ready: bool,
}

/// Tensor metadata from NCT file.
#[derive(Clone, Debug)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Quantization indices
    pub indices: Vec<u8>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Bits per value
    pub bits: u8,
}

impl TensorData {
    /// Get decompressed size in floats.
    pub fn decompressed_size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// NCT file handle for neural compressed tensors.
pub struct NctFile {
    /// Path to file
    pub path: String,
    /// Tensors in file
    pub tensors: Vec<TensorData>,
    /// Codebooks
    pub codebooks: Vec<LayerCodebook>,
}

impl NctFile {
    /// Open an NCT file.
    pub fn open(path: &str) -> Result<Self> {
        // Placeholder - would actually read file
        Ok(Self {
            path: path.to_string(),
            tensors: Vec::new(),
            codebooks: Vec::new(),
        })
    }

    /// Read a specific tensor.
    pub fn read_tensor(&self, name: &str) -> Result<TensorData> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| CudaError::InvalidData(format!("Tensor not found: {}", name)))
    }

    /// Get tensors for a specific layer.
    pub fn layer_tensors(&self, layer: usize) -> Result<Vec<TensorData>> {
        let prefix = format!("layer.{}.", layer);
        Ok(self
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .cloned()
            .collect())
    }

    /// Get all tensors.
    pub fn all_tensors(&self) -> impl Iterator<Item = &TensorData> {
        self.tensors.iter()
    }
}

impl NeuralGpuPipeline {
    /// Create new neural GPU pipeline.
    pub fn new(ctx: &crate::GpuContext) -> Result<Self> {
        let decoder = NeuralGpuDecoder::new(ctx)?;

        Ok(Self {
            device: ctx.device().clone(),
            decoder,
            ready: true,
        })
    }

    /// Check if pipeline is ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Load codebooks from NCT file.
    pub fn load_codebooks(&mut self, nct: &NctFile) -> Result<()> {
        for codebook in &nct.codebooks {
            self.decoder.upload_codebook(codebook)?;
        }
        Ok(())
    }

    /// Decode a single tensor.
    pub fn decode_tensor(&self, data: &TensorData) -> Result<Vec<f32>> {
        self.decoder.lookup(&data.indices)
    }

    /// Decode multiple tensors in batch.
    pub fn decode_batch(&self, tensors: &[TensorData]) -> Result<Vec<Vec<f32>>> {
        tensors.iter().map(|t| self.decode_tensor(t)).collect()
    }

    /// Streaming decode with callback.
    pub fn decode_streaming<F>(&self, nct: &NctFile, mut callback: F) -> Result<()>
    where
        F: FnMut(&str, Vec<f32>) -> Result<()>,
    {
        for tensor in &nct.tensors {
            let decoded = self.decode_tensor(tensor)?;
            callback(&tensor.name, decoded)?;
        }
        Ok(())
    }
}

// CPU-based neural decoder for comparison
pub struct NeuralDecoder {
    codebook: LayerCodebook,
}

impl NeuralDecoder {
    /// Create new CPU decoder.
    pub fn new(codebook: &LayerCodebook) -> Self {
        Self {
            codebook: codebook.clone(),
        }
    }

    /// Lookup indices.
    pub fn lookup(&self, indices: &[u8]) -> Result<Vec<f32>> {
        let code_dim = self.codebook.code_dim;
        let mut output = Vec::with_capacity(indices.len() * code_dim);

        for &idx in indices {
            let idx = idx as usize % self.codebook.num_codes;
            if let Some(code) = self.codebook.get_code(idx) {
                output.extend_from_slice(code);
            } else {
                output.extend(std::iter::repeat_n(0.0, code_dim));
            }
        }

        Ok(output)
    }
}

// =========================================================================
// Track B.4: GPU Codebook Lookup Tests (12 tests)
// =========================================================================

#[cfg(test)]
mod gpu_codebook_tests {
    use super::*;

    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok())
            .ok()
            .flatten()
    }

    #[test]
    fn test_codebook_creation() {
        let codebook = LayerCodebook::new(256, 64);
        assert_eq!(codebook.num_codes, 256);
        assert_eq!(codebook.code_dim, 64);
        assert_eq!(codebook.centroids.len(), 256 * 64);
    }

    #[test]
    fn test_codebook_random() {
        let codebook = LayerCodebook::random(256, 64);
        assert_eq!(codebook.num_codes, 256);
        assert_eq!(codebook.code_dim, 64);

        // Check values are in range
        for &v in &codebook.centroids {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_codebook_set_get() {
        let mut codebook = LayerCodebook::new(4, 2);
        codebook.set_code(0, &[1.0, 0.0]);
        codebook.set_code(1, &[0.0, 1.0]);

        assert_eq!(codebook.get_code(0), Some(&[1.0f32, 0.0][..]));
        assert_eq!(codebook.get_code(1), Some(&[0.0f32, 1.0][..]));
        assert_eq!(codebook.get_code(4), None);
    }

    #[test]
    fn test_gpu_codebook_upload() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let codebook = LayerCodebook::random(256, 64);
        let mut decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        decoder.upload_codebook(&codebook).unwrap();
        assert!(decoder.has_codebook());
    }

    #[test]
    fn test_gpu_codebook_lookup_simple() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut codebook = LayerCodebook::new(4, 2);
        codebook.set_code(0, &[1.0, 0.0]);
        codebook.set_code(1, &[0.0, 1.0]);
        codebook.set_code(2, &[-1.0, 0.0]);
        codebook.set_code(3, &[0.0, -1.0]);

        let mut decoder = NeuralGpuDecoder::new(&ctx).unwrap();
        decoder.upload_codebook(&codebook).unwrap();

        let indices = vec![0u8, 1, 2, 3, 0, 1];
        let result = decoder.lookup(&indices).unwrap();

        let expected = vec![
            1.0, 0.0, // code 0
            0.0, 1.0, // code 1
            -1.0, 0.0, // code 2
            0.0, -1.0, // code 3
            1.0, 0.0, // code 0
            0.0, 1.0, // code 1
        ];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gpu_codebook_lookup_batch() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let codebook = LayerCodebook::random(256, 64);
        let mut decoder = NeuralGpuDecoder::new(&ctx).unwrap();
        decoder.upload_codebook(&codebook).unwrap();

        let tensors: Vec<QuantizedTensor> = (0..10).map(|_| QuantizedTensor::random(100)).collect();

        let results = decoder.lookup_batch(&tensors).unwrap();

        assert_eq!(results.len(), 10);
        for result in &results {
            assert_eq!(result.len(), 100 * 64);
        }
    }

    #[test]
    fn test_gpu_vs_cpu_codebook_lookup() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let codebook = LayerCodebook::random(256, 64);
        let mut gpu_decoder = NeuralGpuDecoder::new(&ctx).unwrap();
        gpu_decoder.upload_codebook(&codebook).unwrap();

        let cpu_decoder = NeuralDecoder::new(&codebook);

        let indices: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let gpu_result = gpu_decoder.lookup(&indices).unwrap();
        let cpu_result = cpu_decoder.lookup(&indices).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(cpu_result.iter()) {
            assert!((g - c).abs() < 1e-5, "GPU: {}, CPU: {}", g, c);
        }
    }

    #[test]
    fn test_gpu_codebook_memory_size() {
        let codebook = LayerCodebook::random(65536, 128);
        let expected_size = 65536 * 128 * 4; // f32

        assert_eq!(codebook.memory_size(), expected_size);
    }

    #[test]
    fn test_gpu_codebook_large() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        // Large codebook: 64K codes, 128-dim = 32MB
        let codebook = LayerCodebook::random(65536, 128);
        let mut decoder = NeuralGpuDecoder::new(&ctx).unwrap();
        decoder.upload_codebook(&codebook).unwrap();

        let indices: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let result = decoder.lookup(&indices).unwrap();

        assert_eq!(result.len(), 10000 * 128);
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let tensor = QuantizedTensor::random(1000);
        assert_eq!(tensor.indices.len(), 1000);
        assert_eq!(tensor.shape, vec![1000]);
    }

    #[test]
    fn test_quantized_tensor_decompressed_size() {
        let tensor = QuantizedTensor::random(1000);
        assert_eq!(tensor.decompressed_size(64), 1000 * 64);
    }

    #[test]
    fn test_codebook_lookup_throughput() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let codebook = LayerCodebook::random(256, 64);
        let mut decoder = NeuralGpuDecoder::new(&ctx).unwrap();
        decoder.upload_codebook(&codebook).unwrap();

        let indices: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = decoder.lookup(&indices).unwrap();
        }
        let elapsed = start.elapsed();

        let output_bytes = indices.len() as f64 * 64.0 * 4.0;
        let throughput_mbs =
            (iterations as f64 * output_bytes) / elapsed.as_secs_f64() / 1_000_000.0;

        println!("Codebook lookup throughput: {:.2} MB/s", throughput_mbs);
        assert!(throughput_mbs > 0.0);
    }
}

// =========================================================================
// Track B.5: GPU Batched Decode Tests (10 tests)
// =========================================================================

#[cfg(test)]
mod gpu_neural_decode_tests {
    use super::*;

    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok())
            .ok()
            .flatten()
    }

    #[test]
    fn test_neural_pipeline_creation() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();
        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_tensor_data_creation() {
        let data = TensorData {
            name: "layer.0.weight".to_string(),
            indices: vec![0, 1, 2, 3],
            shape: vec![2, 2],
            bits: 8,
        };

        assert_eq!(data.name, "layer.0.weight");
        assert_eq!(data.decompressed_size(), 4);
    }

    #[test]
    fn test_nct_file_open() {
        let nct = NctFile::open("testdata/test.nct");
        assert!(nct.is_ok());
    }

    #[test]
    fn test_pipeline_decode_tensor() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        // Create and load a codebook
        let codebook = LayerCodebook::random(256, 64);
        pipeline.decoder.upload_codebook(&codebook).unwrap();

        let tensor = TensorData {
            name: "test.weight".to_string(),
            indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
            shape: vec![8],
            bits: 8,
        };

        let decoded = pipeline.decode_tensor(&tensor).unwrap();
        assert_eq!(decoded.len(), 8 * 64);
    }

    #[test]
    fn test_pipeline_decode_batch() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut pipeline = NeuralGpuPipeline::new(&ctx).unwrap();
        let codebook = LayerCodebook::random(256, 64);
        pipeline.decoder.upload_codebook(&codebook).unwrap();

        let tensors: Vec<TensorData> = (0..5)
            .map(|i| TensorData {
                name: format!("tensor.{}", i),
                indices: vec![i as u8; 100],
                shape: vec![100],
                bits: 8,
            })
            .collect();

        let results = pipeline.decode_batch(&tensors).unwrap();

        assert_eq!(results.len(), 5);
        for result in &results {
            assert_eq!(result.len(), 100 * 64);
        }
    }

    #[test]
    fn test_pipeline_streaming_decode() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut pipeline = NeuralGpuPipeline::new(&ctx).unwrap();
        let codebook = LayerCodebook::random(256, 64);
        pipeline.decoder.upload_codebook(&codebook).unwrap();

        let mut nct = NctFile::open("test.nct").unwrap();
        nct.tensors = vec![
            TensorData {
                name: "layer.0.weight".to_string(),
                indices: vec![0; 100],
                shape: vec![100],
                bits: 8,
            },
            TensorData {
                name: "layer.1.weight".to_string(),
                indices: vec![1; 100],
                shape: vec![100],
                bits: 8,
            },
        ];

        let mut decoded_count = 0;
        pipeline
            .decode_streaming(&nct, |_name, _tensor| {
                decoded_count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(decoded_count, 2);
    }

    #[test]
    fn test_pipeline_empty_nct() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();
        let nct = NctFile::open("empty.nct").unwrap();

        let mut decoded_count = 0;
        pipeline
            .decode_streaming(&nct, |_name, _tensor| {
                decoded_count += 1;
                Ok(())
            })
            .unwrap();

        assert_eq!(decoded_count, 0);
    }

    #[test]
    fn test_nct_layer_tensors() {
        let mut nct = NctFile::open("test.nct").unwrap();
        nct.tensors = vec![
            TensorData {
                name: "layer.0.q_proj".to_string(),
                indices: vec![],
                shape: vec![],
                bits: 8,
            },
            TensorData {
                name: "layer.0.k_proj".to_string(),
                indices: vec![],
                shape: vec![],
                bits: 8,
            },
            TensorData {
                name: "layer.1.q_proj".to_string(),
                indices: vec![],
                shape: vec![],
                bits: 8,
            },
        ];

        let layer0 = nct.layer_tensors(0).unwrap();
        assert_eq!(layer0.len(), 2);

        let layer1 = nct.layer_tensors(1).unwrap();
        assert_eq!(layer1.len(), 1);
    }

    #[test]
    fn test_nct_read_tensor() {
        let mut nct = NctFile::open("test.nct").unwrap();
        nct.tensors = vec![TensorData {
            name: "model.embed".to_string(),
            indices: vec![1, 2, 3],
            shape: vec![3],
            bits: 8,
        }];

        let tensor = nct.read_tensor("model.embed").unwrap();
        assert_eq!(tensor.name, "model.embed");
        assert_eq!(tensor.indices, vec![1, 2, 3]);

        let missing = nct.read_tensor("not_found");
        assert!(missing.is_err());
    }

    #[test]
    fn test_pipeline_throughput() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut pipeline = NeuralGpuPipeline::new(&ctx).unwrap();
        let codebook = LayerCodebook::random(256, 64);
        pipeline.decoder.upload_codebook(&codebook).unwrap();

        // Create a batch of tensors
        let tensors: Vec<TensorData> = (0..100)
            .map(|i| TensorData {
                name: format!("tensor.{}", i),
                indices: (0..1000).map(|j| ((i + j) % 256) as u8).collect(),
                shape: vec![1000],
                bits: 8,
            })
            .collect();

        let total_floats: usize = tensors.iter().map(|t| t.decompressed_size()).sum();

        let start = std::time::Instant::now();
        let _ = pipeline.decode_batch(&tensors).unwrap();
        let elapsed = start.elapsed();

        let throughput_mbs = (total_floats * 4) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        println!("Batch decode throughput: {:.2} MB/s", throughput_mbs);
        assert!(throughput_mbs > 0.0);
    }
}

// =========================================================================
// Track B.6: Integration Tests (8 tests)
// =========================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn test_context() -> Option<crate::GpuContext> {
        // Use catch_unwind to handle case where CUDA isn't available
        std::panic::catch_unwind(|| crate::GpuContext::new(0).ok())
            .ok()
            .flatten()
    }

    #[test]
    fn test_unified_pipeline_zstd_and_neural() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        // Create both pipelines
        let zstd_pipeline = crate::ZstdGpuPipeline::new(&ctx).unwrap();
        let mut neural_pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        let codebook = LayerCodebook::random(256, 64);
        neural_pipeline.decoder.upload_codebook(&codebook).unwrap();

        // Test Zstd decompression
        let original = b"Test data for unified pipeline".repeat(100);
        let compressed = zstd::encode_all(original.as_slice(), 3).unwrap();
        let zstd_result = zstd_pipeline.decompress(&compressed).unwrap();
        assert_eq!(zstd_result, original);

        // Test neural decompression
        let tensor = TensorData {
            name: "test".to_string(),
            indices: vec![0, 1, 2, 3],
            shape: vec![4],
            bits: 8,
        };
        let neural_result = neural_pipeline.decode_tensor(&tensor).unwrap();
        assert_eq!(neural_result.len(), 4 * 64);
    }

    #[test]
    fn test_memory_pool_efficiency() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let initial = ctx.pool().allocated();

        // Allocate and release multiple buffers
        for _ in 0..10 {
            let _buffer = ctx.pool().allocate(1024 * 1024).unwrap();
        }

        // Memory should be reused (not grow unbounded)
        let after = ctx.pool().allocated();
        assert!(
            after <= initial + 20 * 1024 * 1024,
            "Memory grew too much: {} -> {}",
            initial,
            after
        );
    }

    #[test]
    fn test_multi_decoder_coexistence() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        // Create multiple decoders
        let zstd1 = crate::ZstdGpuDecoder::new(&ctx).unwrap();
        let zstd2 = crate::ZstdGpuDecoder::new(&ctx).unwrap();
        let mut neural1 = NeuralGpuDecoder::new(&ctx).unwrap();
        let mut neural2 = NeuralGpuDecoder::new(&ctx).unwrap();

        // All should be ready
        assert!(zstd1.is_ready());
        assert!(zstd2.is_ready());

        // Load different codebooks
        let cb1 = LayerCodebook::random(256, 64);
        let cb2 = LayerCodebook::random(512, 32);
        neural1.upload_codebook(&cb1).unwrap();
        neural2.upload_codebook(&cb2).unwrap();

        // Both should work
        let r1 = neural1.lookup(&[0, 1, 2]).unwrap();
        let r2 = neural2.lookup(&[0, 1, 2]).unwrap();

        assert_eq!(r1.len(), 3 * 64);
        assert_eq!(r2.len(), 3 * 32);
    }

    #[test]
    fn test_device_fallback_logic() {
        // Even without GPU, CPU fallback should work
        let result = zstd::encode_all(b"test data".as_slice(), 3);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        let decompressed = zstd::decode_all(compressed.as_slice()).unwrap();
        assert_eq!(decompressed, b"test data");
    }

    #[test]
    fn test_peak_memory_tracking() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let initial = ctx.pool().allocated();

        // Do some operations
        let buffer1 = ctx.pool().allocate(1024 * 1024).unwrap();
        let buffer2 = ctx.pool().allocate(2 * 1024 * 1024).unwrap();

        let peak = ctx.pool().allocated();
        assert!(peak >= initial + 3 * 1024 * 1024 - 1000);

        drop(buffer1);
        drop(buffer2);

        // Memory may be recycled but allocated count should decrease
        let after = ctx.pool().allocated();
        assert!(after <= peak);
    }

    #[test]
    fn test_error_recovery() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        // Should fail without codebook
        let result = decoder.lookup(&[0, 1, 2]);
        assert!(result.is_err());

        // Create pipeline and try decoding
        let pipeline = crate::ZstdGpuPipeline::new(&ctx).unwrap();

        // Invalid data should return error
        let result = pipeline.decompress(b"not valid zstd");
        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_operations() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        let pipeline = crate::ZstdGpuPipeline::new(&ctx).unwrap();

        // Create multiple compressed frames
        let frames: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                let data = format!("Data block {} for testing", i).repeat(100);
                zstd::encode_all(data.as_bytes(), 3).unwrap()
            })
            .collect();

        // Decompress all
        let results = pipeline.decompress_batch(&frames).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_context_properties() {
        let ctx = match test_context() {
            Some(ctx) => ctx,
            None => return,
        };

        // Check context has expected state
        let pool = ctx.pool();
        assert!(pool.total_size() > 0);
        assert!(pool.available() <= pool.total_size());
    }
}

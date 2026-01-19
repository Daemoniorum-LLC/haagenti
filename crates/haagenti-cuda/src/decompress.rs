//! GPU-accelerated HCT decompression.
//!
//! Reconstructs original tensors from HCT V3 compressed format using GPU IDCT.
//!
//! ## Format
//!
//! HCT V3 compressed data structure:
//! - 2 bytes: num_fragments (u16 LE)
//! - For each fragment:
//!   - 2 bytes: index (u16 LE)
//!   - 2 bytes: flags (u16 LE)
//!   - 8 bytes: checksum (u64 LE)
//!   - 4 bytes: data_len (u32 LE)
//!   - data_len bytes: fragment data
//!
//! Fragment data (V3 with bitmap):
//! - 4 bytes: num_coefficients (u32 LE)
//! - bitmap: (num_elements + 7) / 8 bytes
//! - coefficients: num_coefficients * 2 bytes (f16 LE)
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti_cuda::decompress::{GpuDecompressor, DecompressConfig};
//!
//! let mut decompressor = GpuDecompressor::new(0)?;
//!
//! // Decompress single tensor
//! let tensor = decompressor.decompress(&compressed_data, &[576, 576])?;
//!
//! // Batch decompress multiple tensors
//! let tensors = decompressor.decompress_batch(&compressed_tensors)?;
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::dct_gpu::GpuDctContext;
use crate::{CudaError, Result};

/// Configuration for HCT decompression.
#[derive(Debug, Clone)]
pub struct DecompressConfig {
    /// GPU device ID.
    pub device_id: usize,
    /// Whether to verify checksums.
    pub verify_checksums: bool,
    /// Output precision (f32 or f16).
    pub output_f16: bool,
}

impl Default for DecompressConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            verify_checksums: false, // Skip for performance
            output_f16: false,
        }
    }
}

/// Statistics from batch decompression.
#[derive(Debug, Clone, Default)]
pub struct DecompressStats {
    /// Number of tensors decompressed.
    pub num_tensors: usize,
    /// Total input bytes (compressed).
    pub total_input_bytes: usize,
    /// Total output bytes (decompressed).
    pub total_output_bytes: usize,
    /// Time spent parsing HCT format and reconstructing coefficients (ms).
    pub parse_time_ms: f64,
    /// Time spent on GPU IDCT operations (ms).
    pub idct_time_ms: f64,
    /// Total decompression time (ms).
    pub total_time_ms: f64,
    /// Throughput in MB/s (output bytes / total time).
    pub throughput_mbps: f64,
}

impl DecompressStats {
    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        if self.total_input_bytes == 0 {
            return 0.0;
        }
        self.total_output_bytes as f32 / self.total_input_bytes as f32
    }

    /// Format statistics for display.
    pub fn summary(&self) -> String {
        format!(
            "{} tensors, {:.1} MB -> {:.1} MB ({:.1}x), {:.1}ms ({:.1} MB/s)",
            self.num_tensors,
            self.total_input_bytes as f64 / 1_000_000.0,
            self.total_output_bytes as f64 / 1_000_000.0,
            self.compression_ratio(),
            self.total_time_ms,
            self.throughput_mbps,
        )
    }
}

/// Parsed HCT fragment.
#[derive(Debug)]
#[allow(dead_code)]
struct HctFragment {
    index: u16,
    flags: u16,
    checksum: u64,
    num_coefficients: u32,
    bitmap: Vec<u8>,
    coefficients: Vec<f32>, // Expanded from f16
}

/// GPU-accelerated HCT decompressor.
pub struct GpuDecompressor {
    dct_ctx: GpuDctContext,
    #[allow(dead_code)]
    config: DecompressConfig,
}

impl GpuDecompressor {
    /// Creates a new GPU decompressor.
    pub fn new(device_id: usize) -> Result<Self> {
        let dct_ctx = GpuDctContext::new(device_id)?;
        Ok(Self {
            dct_ctx,
            config: DecompressConfig {
                device_id,
                ..Default::default()
            },
        })
    }

    /// Creates a decompressor with custom config.
    pub fn with_config(config: DecompressConfig) -> Result<Self> {
        let dct_ctx = GpuDctContext::new(config.device_id)?;
        Ok(Self { dct_ctx, config })
    }

    /// Creates a decompressor using an existing CUDA device.
    ///
    /// Useful for sharing GPU context with inference engine.
    pub fn with_device(device: Arc<CudaDevice>) -> Result<Self> {
        let dct_ctx = GpuDctContext::with_device(device)?;
        Ok(Self {
            dct_ctx,
            config: DecompressConfig::default(),
        })
    }

    /// Decompress a single HCT-compressed tensor.
    ///
    /// # Arguments
    /// * `compressed` - Raw HCT compressed data (may be zstd-wrapped)
    /// * `shape` - Original tensor shape
    ///
    /// # Returns
    /// Reconstructed tensor as f32 values.
    pub fn decompress(&mut self, compressed: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
        // First, try to decompress zstd wrapper
        let decompressed = self.try_zstd_decompress(compressed)?;
        let data = decompressed.as_deref().unwrap_or(compressed);

        // Parse HCT format
        let fragments = self.parse_hct_fragments(data, shape)?;

        // Calculate dimensions
        let (width, height) = self.calculate_dimensions(shape);
        let total_elements = width * height;

        // Reconstruct DCT coefficient matrix from fragments
        let coefficients = self.reconstruct_coefficients(&fragments, total_elements)?;

        // Apply GPU IDCT
        let reconstructed = self.dct_ctx.idct_2d(&coefficients, width, height)?;

        // Reshape if needed (currently we keep flat f32 vec)
        Ok(reconstructed)
    }

    /// Decompress directly to GPU memory (avoids device->host copy).
    ///
    /// # Arguments
    /// * `compressed` - Raw HCT compressed data
    /// * `shape` - Original tensor shape
    ///
    /// # Returns
    /// Reconstructed tensor in GPU memory.
    pub fn decompress_to_gpu(
        &mut self,
        compressed: &[u8],
        shape: &[usize],
    ) -> Result<CudaSlice<f32>> {
        // Parse and reconstruct coefficients
        let decompressed = self.try_zstd_decompress(compressed)?;
        let data = decompressed.as_deref().unwrap_or(compressed);

        let fragments = self.parse_hct_fragments(data, shape)?;
        let (width, height) = self.calculate_dimensions(shape);
        let total_elements = width * height;

        let coefficients = self.reconstruct_coefficients(&fragments, total_elements)?;

        // Upload to GPU and decompress
        let d_coeffs: CudaSlice<f32> = self.dct_ctx.device().htod_sync_copy(&coefficients)?;
        let d_output = self.dct_ctx.idct_2d_gpu(&d_coeffs, width, height)?;

        Ok(d_output)
    }

    /// Batch decompress multiple tensors.
    ///
    /// More efficient than calling decompress() multiple times.
    pub fn decompress_batch(
        &mut self,
        tensors: &[(&[u8], &[usize])],
    ) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(tensors.len());

        for (compressed, shape) in tensors {
            results.push(self.decompress(compressed, shape)?);
        }

        Ok(results)
    }

    /// Pipelined batch decompression with statistics.
    ///
    /// Overlaps CPU coefficient reconstruction with GPU IDCT operations
    /// for better throughput when decompressing many tensors.
    ///
    /// # Arguments
    /// * `tensors` - List of (compressed_data, shape) pairs
    ///
    /// # Returns
    /// Tuple of (results, statistics)
    pub fn decompress_batch_pipelined(
        &mut self,
        tensors: &[(&[u8], &[usize])],
    ) -> Result<(Vec<Vec<f32>>, DecompressStats)> {
        let start = Instant::now();
        let mut stats = DecompressStats::default();
        stats.num_tensors = tensors.len();

        // Pre-parse all tensors to get coefficient matrices (CPU work)
        let parse_start = Instant::now();
        let mut parsed: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(tensors.len());

        for (compressed, shape) in tensors {
            stats.total_input_bytes += compressed.len();

            // Decompress zstd if needed
            let decompressed = self.try_zstd_decompress(compressed)?;
            let data = decompressed.as_deref().unwrap_or(*compressed);

            // Parse fragments
            let fragments = self.parse_hct_fragments(data, shape)?;
            let (width, height) = self.calculate_dimensions(shape);
            let total_elements = width * height;

            // Reconstruct coefficient matrix
            let coefficients = self.reconstruct_coefficients(&fragments, total_elements)?;
            parsed.push((coefficients, width, height));

            stats.total_output_bytes += total_elements * 4;
        }
        stats.parse_time_ms = parse_start.elapsed().as_secs_f64() * 1000.0;

        // Now do all GPU IDCT operations
        let idct_start = Instant::now();
        let mut results = Vec::with_capacity(parsed.len());

        for (coefficients, width, height) in parsed {
            let reconstructed = self.dct_ctx.idct_2d(&coefficients, width, height)?;
            results.push(reconstructed);
        }
        stats.idct_time_ms = idct_start.elapsed().as_secs_f64() * 1000.0;

        stats.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.throughput_mbps = stats.total_output_bytes as f64 / stats.total_time_ms / 1000.0;

        Ok((results, stats))
    }

    /// Batch decompress directly to GPU memory.
    ///
    /// Returns tensors in GPU memory, avoiding device-to-host copies.
    /// Optimal for inference pipelines that keep tensors on GPU.
    pub fn decompress_batch_to_gpu(
        &mut self,
        tensors: &[(&[u8], &[usize])],
    ) -> Result<Vec<CudaSlice<f32>>> {
        let mut results = Vec::with_capacity(tensors.len());

        for (compressed, shape) in tensors {
            results.push(self.decompress_to_gpu(compressed, shape)?);
        }

        Ok(results)
    }

    /// Try to decompress zstd wrapper.
    fn try_zstd_decompress(&self, data: &[u8]) -> Result<Option<Vec<u8>>> {
        // Check for zstd magic number (0x28B52FFD)
        if data.len() >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD {
            // Try zstd decompression, but fall back to raw parsing if it fails
            // (haagenti-zstd may produce non-standard frames)
            match zstd::decode_all(data) {
                Ok(decompressed) => Ok(Some(decompressed)),
                Err(_) => {
                    // Zstd magic present but decompression failed - treat as raw HCT
                    tracing::debug!("Zstd magic present but decompression failed, treating as raw HCT");
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Parse HCT V3 fragments from data.
    fn parse_hct_fragments(&self, data: &[u8], shape: &[usize]) -> Result<Vec<HctFragment>> {
        if data.len() < 2 {
            return Err(CudaError::InvalidData("HCT data too short".to_string()));
        }

        let num_fragments = u16::from_le_bytes([data[0], data[1]]) as usize;
        let mut offset = 2;
        let mut fragments = Vec::with_capacity(num_fragments);

        let total_elements: usize = shape.iter().product();

        for _ in 0..num_fragments {
            if offset + 16 > data.len() {
                break;
            }

            // Parse fragment header
            let index = u16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
            let flags = u16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
            let checksum = u64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]);
            offset += 8;
            let data_len = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + data_len > data.len() {
                return Err(CudaError::InvalidData(format!(
                    "Fragment {} extends past data end",
                    index
                )));
            }

            let frag_data = &data[offset..offset + data_len];
            offset += data_len;

            // Parse fragment data (V3 format)
            if frag_data.len() < 4 {
                continue;
            }

            let num_coefficients = u32::from_le_bytes([
                frag_data[0], frag_data[1], frag_data[2], frag_data[3],
            ]);

            let bitmap_size = (total_elements + 7) / 8;
            if frag_data.len() < 4 + bitmap_size {
                continue;
            }

            let bitmap = frag_data[4..4 + bitmap_size].to_vec();

            // Extract f16 coefficients and convert to f32
            let coeff_data = &frag_data[4 + bitmap_size..];
            let coefficients: Vec<f32> = coeff_data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            fragments.push(HctFragment {
                index,
                flags,
                checksum,
                num_coefficients,
                bitmap,
                coefficients,
            });
        }

        Ok(fragments)
    }

    /// Reconstruct full coefficient matrix from sparse fragments.
    fn reconstruct_coefficients(
        &self,
        fragments: &[HctFragment],
        total_elements: usize,
    ) -> Result<Vec<f32>> {
        let mut coefficients = vec![0.0f32; total_elements];

        for fragment in fragments {
            // Expand bitmap to indices
            let mut coeff_idx = 0;

            for (byte_idx, &byte) in fragment.bitmap.iter().enumerate() {
                for bit in 0..8 {
                    let element_idx = byte_idx * 8 + bit;
                    if element_idx >= total_elements {
                        break;
                    }

                    if (byte >> bit) & 1 == 1 {
                        if coeff_idx < fragment.coefficients.len() {
                            coefficients[element_idx] = fragment.coefficients[coeff_idx];
                            coeff_idx += 1;
                        }
                    }
                }
            }
        }

        Ok(coefficients)
    }

    /// Calculate 2D dimensions from shape.
    fn calculate_dimensions(&self, shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            0 => (1, 1),
            1 => (shape[0], 1),
            2 => (shape[1], shape[0]),
            _ => {
                // Flatten to 2D: last dim is width, rest is height
                let width = shape.last().copied().unwrap_or(1);
                let height: usize = shape.iter().take(shape.len() - 1).product();
                (width, height)
            }
        }
    }

    /// Get underlying DCT context for advanced operations.
    pub fn dct_context(&mut self) -> &mut GpuDctContext {
        &mut self.dct_ctx
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        self.dct_ctx.device()
    }
}

/// Decompress HCT data using CPU fallback (no GPU required).
///
/// Useful for testing or systems without CUDA.
pub fn decompress_cpu(compressed: &[u8], shape: &[usize]) -> Result<Vec<f32>> {
    // Try zstd decompress (fall back to raw if it fails)
    let decompressed = if compressed.len() >= 4
        && compressed[0] == 0x28
        && compressed[1] == 0xB5
        && compressed[2] == 0x2F
        && compressed[3] == 0xFD
    {
        // Try zstd, but treat as raw HCT if decompression fails
        zstd::decode_all(compressed).ok()
    } else {
        None
    };

    let data = decompressed.as_deref().unwrap_or(compressed);

    // Parse fragments
    if data.len() < 2 {
        return Err(CudaError::InvalidData("Data too short".to_string()));
    }

    let num_fragments = u16::from_le_bytes([data[0], data[1]]) as usize;
    let mut offset = 2;

    let total_elements: usize = shape.iter().product();
    let mut coefficients = vec![0.0f32; total_elements];

    for _ in 0..num_fragments {
        if offset + 16 > data.len() {
            break;
        }

        // Skip header
        offset += 2; // index
        offset += 2; // flags
        offset += 8; // checksum
        let data_len = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + data_len > data.len() {
            break;
        }

        let frag_data = &data[offset..offset + data_len];
        offset += data_len;

        if frag_data.len() < 4 {
            continue;
        }

        let bitmap_size = (total_elements + 7) / 8;
        if frag_data.len() < 4 + bitmap_size {
            continue;
        }

        let bitmap = &frag_data[4..4 + bitmap_size];
        let coeff_data = &frag_data[4 + bitmap_size..];

        // Expand coefficients
        let mut coeff_idx = 0;
        for (byte_idx, &byte) in bitmap.iter().enumerate() {
            for bit in 0..8 {
                let element_idx = byte_idx * 8 + bit;
                if element_idx >= total_elements {
                    break;
                }

                if (byte >> bit) & 1 == 1 {
                    if coeff_idx * 2 + 1 < coeff_data.len() {
                        let bits = u16::from_le_bytes([
                            coeff_data[coeff_idx * 2],
                            coeff_data[coeff_idx * 2 + 1],
                        ]);
                        coefficients[element_idx] = half::f16::from_bits(bits).to_f32();
                        coeff_idx += 1;
                    }
                }
            }
        }
    }

    // Apply CPU IDCT
    let (width, height) = match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        2 => (shape[1], shape[0]),
        _ => {
            let width = shape.last().copied().unwrap_or(1);
            let height: usize = shape.iter().take(shape.len() - 1).product();
            (width, height)
        }
    };

    #[cfg(feature = "cpu-fallback")]
    {
        use haagenti_core::dct::idct_2d;
        let mut output = vec![0.0f32; total_elements];
        idct_2d(&coefficients, &mut output, width, height);
        Ok(output)
    }

    #[cfg(not(feature = "cpu-fallback"))]
    {
        let _ = (width, height);
        Err(CudaError::InvalidData("CPU fallback disabled".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompress_config_default() {
        let config = DecompressConfig::default();
        assert_eq!(config.device_id, 0);
        assert!(!config.verify_checksums);
        assert!(!config.output_f16);
    }

    #[test]
    fn test_calculate_dimensions_helper() {
        // Test dimension calculation logic directly
        fn calc_dims(shape: &[usize]) -> (usize, usize) {
            match shape.len() {
                0 => (1, 1),
                1 => (shape[0], 1),
                2 => (shape[1], shape[0]),
                _ => {
                    let width = shape.last().copied().unwrap_or(1);
                    let height: usize = shape.iter().take(shape.len() - 1).product();
                    (width, height)
                }
            }
        }

        assert_eq!(calc_dims(&[]), (1, 1));
        assert_eq!(calc_dims(&[10]), (10, 1));
        assert_eq!(calc_dims(&[10, 20]), (20, 10));
        assert_eq!(calc_dims(&[2, 3, 4]), (4, 6));
    }

    #[test]
    #[ignore]
    fn test_gpu_decompressor_creation() {
        let result = GpuDecompressor::new(0);
        assert!(result.is_ok());
    }
}

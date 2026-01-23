//! CUDA GPU Decompression for Haagenti
//!
//! Provides GPU-accelerated decompression for LZ4 and Zstd compressed tensors,
//! enabling zero-copy loading directly to GPU memory.
//!
//! # Architecture
//!
//! ```text
//! Traditional Pipeline:
//!   Disk → CPU RAM → Decompress (CPU) → GPU Transfer → Inference
//!   [5s]   [2GB]      [500ms]            [200ms]        [ready]
//!
//! GPU Decompression Pipeline:
//!   Disk → Pinned Memory → GPU Transfer → Decompress (GPU) → Inference
//!   [3s]   [staged]        [150ms]        [50ms]             [ready]
//! ```
//!
//! # Key Features
//!
//! - **Zero-copy**: Decompressed data stays on GPU, never touches CPU RAM
//! - **Streaming**: Overlap disk I/O with GPU decompression
//! - **Memory Pool**: Reusable GPU buffers eliminate allocation overhead
//! - **Async**: Non-blocking decompression with CUDA streams
//!
//! # Example
//!
//! ```ignore
//! use haagenti_cuda::{GpuDecompressor, MemoryPool};
//!
//! let pool = MemoryPool::new(512 * 1024 * 1024)?; // 512MB pool
//! let decompressor = GpuDecompressor::new(&pool)?;
//!
//! // Decompress directly to GPU tensor
//! let gpu_tensor = decompressor.decompress_lz4(&compressed_data)?;
//! ```

#[cfg(feature = "cufft")]
pub mod cufft_ffi;
pub mod dct_gpu;
pub mod decompress;
pub mod error;
pub mod kernels;
pub mod memory;
pub mod native_kernels;
pub mod neural_gpu;
pub mod pipeline;
pub mod stream;
pub mod zstd_gpu;

#[cfg(feature = "cufft")]
pub use cufft_ffi::{CufftPlan, CufftType, FftDctContext};
pub use dct_gpu::{BatchDctConfig, DctMode, GpuDctContext};
pub use decompress::{decompress_cpu, DecompressConfig, DecompressStats, GpuDecompressor};
pub use error::{CudaError, Result};
pub use kernels::{Lz4GpuDecompressor, ZstdGpuDecompressor};
pub use memory::{GpuBuffer, MemoryPool, PinnedBuffer};
pub use native_kernels::{KernelStats, NativeKernels};
pub use neural_gpu::{
    LayerCodebook, NctFile, NeuralDecoder, NeuralGpuDecoder, NeuralGpuPipeline, QuantizedTensor,
    TensorData,
};
pub use pipeline::{DecompressionPipeline, PipelineConfig};
pub use stream::{AsyncDecompressor, StreamingDecoder};
pub use zstd_gpu::{FseGpuDecoder, FseTable, Sequence, ZstdGpuDecoder, ZstdGpuPipeline};

use cudarc::driver::{sys, CudaDevice, CudaStream};
use std::sync::Arc;

/// Get compute capability of a CUDA device.
fn get_compute_capability(device: &CudaDevice) -> (usize, usize) {
    let major = device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0) as usize;
    let minor = device
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0) as usize;
    (major, minor)
}

/// GPU decompression context.
///
/// Manages CUDA device, memory pool, and decompression kernels.
/// Automatically uses native kernels on SM 7.0+ GPUs for best performance.
pub struct GpuContext {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    pool: MemoryPool,
    lz4: Lz4GpuDecompressor,
    zstd: Option<ZstdGpuDecompressor>,
    native: Option<NativeKernels>,
    use_native: bool,
}

// SAFETY: GpuContext is safe to send between threads when protected by a Mutex.
// The CudaStream internally synchronizes CUDA operations, and we ensure
// single-threaded access through external synchronization (Mutex wrapper).
// The device (Arc<CudaDevice>) is already Send + Sync.
unsafe impl Send for GpuContext {}

// SAFETY: GpuContext can be shared between threads when protected by a Mutex.
// All CUDA operations are synchronized through the stream, and we ensure
// exclusive access through the Mutex wrapper in calling code.
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Create a new GPU context on the specified device.
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)?;
        // Note: CudaDevice::new already returns Arc<CudaDevice>
        let stream = device.fork_default_stream()?;

        // Create memory pool with 256MB initial size
        let pool = MemoryPool::new(device.clone(), 256 * 1024 * 1024)?;

        // Initialize decompressors
        let lz4 = Lz4GpuDecompressor::new(device.clone())?;
        let zstd = ZstdGpuDecompressor::new(device.clone()).ok();

        // Try to load native kernels (SM 7.0+)
        let native = NativeKernels::new(device.clone()).ok();
        let use_native = native.is_some();

        if use_native {
            tracing::info!("Native CUDA kernels loaded - using warp-level parallelism");
        }

        Ok(GpuContext {
            device,
            stream,
            pool,
            lz4,
            zstd,
            native,
            use_native,
        })
    }

    /// Create with explicit native kernel preference.
    pub fn with_native_preference(device_id: usize, prefer_native: bool) -> Result<Self> {
        let mut ctx = Self::new(device_id)?;
        ctx.use_native = prefer_native && ctx.native.is_some();
        Ok(ctx)
    }

    /// Check if native kernels are available.
    pub fn has_native_kernels(&self) -> bool {
        self.native.is_some()
    }

    /// Enable/disable native kernels.
    pub fn set_use_native(&mut self, use_native: bool) {
        self.use_native = use_native && self.native.is_some();
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the memory pool.
    pub fn pool(&self) -> &MemoryPool {
        &self.pool
    }

    /// Decompress LZ4 data directly to GPU.
    ///
    /// Returns a GPU buffer containing the decompressed data.
    pub fn decompress_lz4(&self, compressed: &[u8], decompressed_size: usize) -> Result<GpuBuffer> {
        // Allocate output buffer from pool
        let output = self.pool.allocate(decompressed_size)?;

        // Stage compressed data in pinned memory
        let mut pinned = self.pool.allocate_pinned(compressed.len())?;
        pinned.copy_from_host(compressed)?;

        // Copy to device
        let device_input = self.pool.allocate(compressed.len())?;
        device_input.copy_from_pinned(&pinned)?;

        // Decompress on GPU
        self.lz4.decompress(
            &device_input,
            &output,
            compressed.len(),
            decompressed_size,
            &self.stream,
        )?;

        // Sync and return
        self.device.synchronize()?;
        Ok(output)
    }

    /// Decompress Zstd data directly to GPU.
    pub fn decompress_zstd(
        &self,
        compressed: &[u8],
        decompressed_size: usize,
    ) -> Result<GpuBuffer> {
        let zstd = self.zstd.as_ref().ok_or(CudaError::UnsupportedAlgorithm)?;

        let output = self.pool.allocate(decompressed_size)?;
        let mut pinned = self.pool.allocate_pinned(compressed.len())?;
        pinned.copy_from_host(compressed)?;

        let device_input = self.pool.allocate(compressed.len())?;
        device_input.copy_from_pinned(&pinned)?;

        zstd.decompress(
            &device_input,
            &output,
            compressed.len(),
            decompressed_size,
            &self.stream,
        )?;

        self.device.synchronize()?;
        Ok(output)
    }

    /// Create a streaming decompression pipeline.
    pub fn create_pipeline(&self, config: PipelineConfig) -> Result<DecompressionPipeline> {
        DecompressionPipeline::new(self.device.clone(), self.pool.clone(), config)
    }
}

/// Check if CUDA GPU decompression is available.
pub fn is_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Get memory info for a device (free, total).
fn get_memory_info() -> (usize, usize) {
    // Use cudarc's mem_get_info which wraps cuMemGetInfo
    cudarc::driver::result::mem_get_info().unwrap_or((0, 0))
}

/// Get information about available CUDA devices.
pub fn device_info() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    // Use catch_unwind to handle case where CUDA isn't available
    for i in 0..8 {
        let device_result = std::panic::catch_unwind(|| CudaDevice::new(i));
        match device_result {
            Ok(Ok(device)) => {
                // Get memory info (returns (free, total))
                let (free, total) = get_memory_info();
                devices.push(DeviceInfo {
                    id: i,
                    name: device.name().unwrap_or_default(),
                    compute_capability: get_compute_capability(&device),
                    total_memory: total,
                    free_memory: free,
                });
            }
            Ok(Err(_)) | Err(_) => {
                // Device not available or panic, stop checking
                break;
            }
        }
    }

    devices
}

/// Information about a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: usize,
    pub name: String,
    pub compute_capability: (usize, usize),
    pub total_memory: usize,
    pub free_memory: usize,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU {}: {} (CC {}.{}, {:.1}GB/{:.1}GB free)",
            self.id,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.free_memory as f64 / 1e9,
            self.total_memory as f64 / 1e9,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info() {
        let devices = device_info();
        for device in devices {
            println!("{}", device);
        }
    }
}

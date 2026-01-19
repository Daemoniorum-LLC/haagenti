//! GPU Memory Management for Haagenti CUDA.
//!
//! Provides efficient memory pooling and buffer management for GPU decompression.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       Memory Pool                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Device Memory (GPU VRAM)                                        │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
//! │  │ Buffer 1 │ │ Buffer 2 │ │ Buffer 3 │ │   Free   │            │
//! │  │  (in use)│ │ (free)   │ │ (in use) │ │          │            │
//! │  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Pinned Memory (CPU RAM, DMA accessible)                         │
//! │  ┌──────────┐ ┌──────────┐                                       │
//! │  │ Staging  │ │ Staging  │                                       │
//! │  │ Buffer 1 │ │ Buffer 2 │                                       │
//! │  └──────────┘ └──────────┘                                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::error::{CudaError, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A buffer in GPU device memory.
pub struct GpuBuffer {
    data: CudaSlice<u8>,
    size: usize,
    pool: Option<Arc<MemoryPoolInner>>,
}

impl GpuBuffer {
    /// Get the size of the buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the length of the buffer in elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get a reference to the underlying CUDA slice.
    pub fn as_slice(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Copy data from pinned memory.
    pub fn copy_from_pinned(&self, pinned: &PinnedBuffer) -> Result<()> {
        if pinned.size() > self.size {
            return Err(CudaError::SizeMismatch {
                expected: self.size,
                actual: pinned.size(),
            });
        }
        let device = self.data.device();
        // cudarc 0.12: use device.htod_sync_copy_into for synchronous host-to-device copy
        device.htod_sync_copy_into(pinned.as_slice(), &mut self.data.clone())?;
        Ok(())
    }

    /// Copy data to host memory.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() < self.size {
            return Err(CudaError::SizeMismatch {
                expected: self.size,
                actual: dst.len(),
            });
        }
        let device = self.data.device();
        // cudarc 0.12: use device.dtoh_sync_copy_into for synchronous device-to-host copy
        device.dtoh_sync_copy_into(&self.data, dst)?;
        Ok(())
    }

    /// Get the device this buffer is allocated on.
    pub fn device(&self) -> Arc<CudaDevice> {
        self.data.device()
    }

    /// Get the raw CUDA device pointer.
    ///
    /// This is needed for launching kernels that require raw pointers.
    pub fn as_ptr(&self) -> sys::CUdeviceptr {
        *self.data.device_ptr()
    }

    /// Copy data from pinned memory with stream ordering.
    ///
    /// This is an alias for copy_from_pinned that ignores the stream parameter
    /// since cudarc 0.12 uses synchronous copies.
    pub fn copy_from_pinned_stream(&mut self, pinned: &PinnedBuffer, _stream: &CudaStream) -> Result<()> {
        self.copy_from_pinned(pinned)
    }

    /// Copy data from host memory.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        if src.len() > self.size {
            return Err(CudaError::SizeMismatch {
                expected: self.size,
                actual: src.len(),
            });
        }
        let device = self.data.device();
        // cudarc 0.12: use device.htod_sync_copy_into for synchronous host-to-device copy
        device.htod_sync_copy_into(src, &mut self.data.clone())?;
        Ok(())
    }

    /// Copy entire buffer to host memory.
    pub fn to_host(&self) -> Result<Vec<u8>> {
        let mut dst = vec![0u8; self.size];
        let device = self.data.device();
        // cudarc 0.12: use device.dtoh_sync_copy_into for synchronous device-to-host copy
        device.dtoh_sync_copy_into(&self.data, &mut dst)?;
        Ok(dst)
    }

    /// Create a new GPU buffer directly from device.
    pub fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self> {
        let data = device.alloc_zeros::<u8>(size)?;
        Ok(GpuBuffer {
            data,
            size,
            pool: None,
        })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Return to pool if from a pool
        if let Some(pool) = &self.pool {
            pool.return_buffer(self.size);
        }
    }
}

/// A buffer in pinned (page-locked) CPU memory.
///
/// Pinned memory enables faster DMA transfers to/from GPU.
pub struct PinnedBuffer {
    data: Vec<u8>,
    size: usize,
}

impl PinnedBuffer {
    /// Create a new pinned buffer.
    pub fn new(size: usize) -> Result<Self> {
        // In a real implementation, this would use cudaHostAlloc
        // For now, we use regular allocation as a placeholder
        let data = vec![0u8; size];
        Ok(PinnedBuffer { data, size })
    }

    /// Get the size of the buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a slice of the data.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.size]
    }

    /// Get a mutable slice of the data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.size]
    }

    /// Copy data from host.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.data.len() {
            return Err(CudaError::SizeMismatch {
                expected: self.data.len(),
                actual: src.len(),
            });
        }
        self.data[..src.len()].copy_from_slice(src);
        self.size = src.len();
        Ok(())
    }
}

/// Internal memory pool state.
struct MemoryPoolInner {
    device: Arc<CudaDevice>,
    total_size: usize,
    allocated: Mutex<usize>,
    free_buffers: Mutex<VecDeque<(usize, CudaSlice<u8>)>>,
}

impl MemoryPoolInner {
    fn return_buffer(&self, size: usize) {
        let mut allocated = self.allocated.lock().unwrap();
        *allocated = allocated.saturating_sub(size);
    }
}

/// GPU memory pool for efficient buffer reuse.
///
/// Pre-allocates GPU memory and manages buffer lifecycle to avoid
/// repeated allocation/deallocation overhead.
#[derive(Clone)]
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

impl MemoryPool {
    /// Create a new memory pool with the specified size.
    pub fn new(device: Arc<CudaDevice>, total_size: usize) -> Result<Self> {
        Ok(MemoryPool {
            inner: Arc::new(MemoryPoolInner {
                device,
                total_size,
                allocated: Mutex::new(0),
                free_buffers: Mutex::new(VecDeque::new()),
            }),
        })
    }

    /// Get the device for this pool.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.inner.device
    }

    /// Get the total pool size.
    pub fn total_size(&self) -> usize {
        self.inner.total_size
    }

    /// Get the currently allocated size.
    pub fn allocated(&self) -> usize {
        *self.inner.allocated.lock().unwrap()
    }

    /// Get the available size.
    pub fn available(&self) -> usize {
        self.inner.total_size.saturating_sub(self.allocated())
    }

    /// Allocate a buffer from the pool.
    pub fn allocate(&self, size: usize) -> Result<GpuBuffer> {
        // Check if we have a suitable free buffer
        {
            let mut free = self.inner.free_buffers.lock().unwrap();
            if let Some(pos) = free.iter().position(|(s, _)| *s >= size) {
                let (buf_size, data) = free.remove(pos).unwrap();
                let mut allocated = self.inner.allocated.lock().unwrap();
                *allocated += buf_size;
                return Ok(GpuBuffer {
                    data,
                    size: buf_size,
                    pool: Some(self.inner.clone()),
                });
            }
        }

        // Check if we have space
        let mut allocated = self.inner.allocated.lock().unwrap();
        if *allocated + size > self.inner.total_size {
            // Try to free some buffers
            let mut free = self.inner.free_buffers.lock().unwrap();
            while *allocated + size > self.inner.total_size && !free.is_empty() {
                let (freed_size, _) = free.pop_front().unwrap();
                // Buffer is dropped, memory freed
                *allocated = allocated.saturating_sub(freed_size);
            }

            if *allocated + size > self.inner.total_size {
                return Err(CudaError::OutOfMemory {
                    requested: size,
                    available: self.inner.total_size - *allocated,
                });
            }
        }

        // Allocate new buffer
        let data = self.inner.device.alloc_zeros::<u8>(size)?;
        *allocated += size;

        Ok(GpuBuffer {
            data,
            size,
            pool: Some(self.inner.clone()),
        })
    }

    /// Allocate a pinned CPU buffer.
    pub fn allocate_pinned(&self, size: usize) -> Result<PinnedBuffer> {
        PinnedBuffer::new(size)
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// Note: In cudarc 0.12, we cannot easily extract the CudaSlice from a GpuBuffer
    /// due to Drop semantics. For now, buffers are just dropped and memory is reclaimed
    /// by the CUDA driver.
    pub fn recycle(&self, buffer: GpuBuffer) {
        // In cudarc 0.12, we just drop the buffer - the memory is freed by CUDA
        // The pool's Drop handler on GpuBuffer will update the allocated count
        drop(buffer);
    }

    /// Clear all free buffers.
    pub fn clear_free(&self) {
        let mut free = self.inner.free_buffers.lock().unwrap();
        let mut allocated = self.inner.allocated.lock().unwrap();
        while let Some((size, _)) = free.pop_front() {
            *allocated = allocated.saturating_sub(size);
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let free = self.inner.free_buffers.lock().unwrap();
        PoolStats {
            total_size: self.inner.total_size,
            allocated: self.allocated(),
            free_buffers: free.len(),
            free_buffer_bytes: free.iter().map(|(s, _)| *s).sum(),
        }
    }
}

/// Memory pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_size: usize,
    pub allocated: usize,
    pub free_buffers: usize,
    pub free_buffer_bytes: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pool: {:.1}MB allocated, {:.1}MB total, {} free buffers ({:.1}MB)",
            self.allocated as f64 / 1e6,
            self.total_size as f64 / 1e6,
            self.free_buffers,
            self.free_buffer_bytes as f64 / 1e6,
        )
    }
}

/// Buffer that can be either GPU or CPU (for fallback).
pub enum HybridBuffer {
    Gpu(GpuBuffer),
    Cpu(Vec<u8>),
}

impl HybridBuffer {
    /// Get the size.
    pub fn size(&self) -> usize {
        match self {
            HybridBuffer::Gpu(b) => b.size(),
            HybridBuffer::Cpu(v) => v.len(),
        }
    }

    /// Check if this is a GPU buffer.
    pub fn is_gpu(&self) -> bool {
        matches!(self, HybridBuffer::Gpu(_))
    }

    /// Try to get as GPU buffer.
    pub fn as_gpu(&self) -> Option<&GpuBuffer> {
        match self {
            HybridBuffer::Gpu(b) => Some(b),
            HybridBuffer::Cpu(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_buffer() {
        let mut buf = PinnedBuffer::new(1024).unwrap();
        buf.copy_from_host(&[1, 2, 3, 4]).unwrap();
        assert_eq!(&buf.as_slice()[..4], &[1, 2, 3, 4]);
    }
}

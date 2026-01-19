//! GPU buffer management for WebGPU

use crate::{Result, WebGpuError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BufferUsage {
    /// Storage buffer (read/write in shaders)
    Storage,
    /// Uniform buffer (read-only constants)
    Uniform,
    /// Vertex buffer
    Vertex,
    /// Index buffer
    Index,
    /// Copy source
    CopySrc,
    /// Copy destination
    CopyDst,
    /// Map for reading
    MapRead,
    /// Map for writing
    MapWrite,
}

impl BufferUsage {
    /// Convert to WebGPU buffer usage flags
    #[cfg(target_arch = "wasm32")]
    pub fn to_webgpu(&self) -> u32 {
        match self {
            BufferUsage::Storage => web_sys::gpu_buffer_usage::STORAGE,
            BufferUsage::Uniform => web_sys::gpu_buffer_usage::UNIFORM,
            BufferUsage::Vertex => web_sys::gpu_buffer_usage::VERTEX,
            BufferUsage::Index => web_sys::gpu_buffer_usage::INDEX,
            BufferUsage::CopySrc => web_sys::gpu_buffer_usage::COPY_SRC,
            BufferUsage::CopyDst => web_sys::gpu_buffer_usage::COPY_DST,
            BufferUsage::MapRead => web_sys::gpu_buffer_usage::MAP_READ,
            BufferUsage::MapWrite => web_sys::gpu_buffer_usage::MAP_WRITE,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_webgpu(&self) -> u32 {
        match self {
            BufferUsage::Storage => 0x80,
            BufferUsage::Uniform => 0x40,
            BufferUsage::Vertex => 0x20,
            BufferUsage::Index => 0x10,
            BufferUsage::CopySrc => 0x04,
            BufferUsage::CopyDst => 0x08,
            BufferUsage::MapRead => 0x01,
            BufferUsage::MapWrite => 0x02,
        }
    }
}

/// A GPU buffer
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer ID
    pub id: u64,
    /// Size in bytes
    pub size: u64,
    /// Usage flags
    pub usage: Vec<BufferUsage>,
    /// Label for debugging
    pub label: String,
    /// Whether buffer is mapped
    mapped: bool,
}

impl GpuBuffer {
    /// Create a new buffer descriptor
    pub fn new(size: u64, usage: Vec<BufferUsage>, label: impl Into<String>) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            size,
            usage,
            label: label.into(),
            mapped: false,
        }
    }

    /// Create storage buffer
    pub fn storage(size: u64, label: impl Into<String>) -> Self {
        Self::new(
            size,
            vec![BufferUsage::Storage, BufferUsage::CopySrc, BufferUsage::CopyDst],
            label,
        )
    }

    /// Create uniform buffer
    pub fn uniform(size: u64, label: impl Into<String>) -> Self {
        Self::new(
            size,
            vec![BufferUsage::Uniform, BufferUsage::CopyDst],
            label,
        )
    }

    /// Create staging buffer for reading
    pub fn staging_read(size: u64, label: impl Into<String>) -> Self {
        Self::new(
            size,
            vec![BufferUsage::MapRead, BufferUsage::CopyDst],
            label,
        )
    }

    /// Create staging buffer for writing
    pub fn staging_write(size: u64, label: impl Into<String>) -> Self {
        Self::new(
            size,
            vec![BufferUsage::MapWrite, BufferUsage::CopySrc],
            label,
        )
    }

    /// Combined usage flags for WebGPU
    pub fn combined_usage(&self) -> u32 {
        self.usage.iter().map(|u| u.to_webgpu()).fold(0, |a, b| a | b)
    }

    /// Check if buffer is mapped
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Mark as mapped
    pub fn set_mapped(&mut self, mapped: bool) {
        self.mapped = mapped;
    }

    /// Align size to WebGPU requirements (256 bytes for storage)
    pub fn aligned_size(size: u64, alignment: u64) -> u64 {
        (size + alignment - 1) / alignment * alignment
    }
}

/// Buffer pool for efficient allocation
#[derive(Debug)]
pub struct BufferPool {
    /// Available buffers by size bucket
    available: HashMap<u64, Vec<GpuBuffer>>,
    /// All allocated buffers
    allocated: HashMap<u64, GpuBuffer>,
    /// Size buckets (power of 2)
    buckets: Vec<u64>,
    /// Total allocated memory
    total_allocated: u64,
    /// Maximum pool size
    max_size: u64,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(max_size: u64) -> Self {
        // Size buckets: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB
        let buckets = vec![
            1024,
            4 * 1024,
            16 * 1024,
            64 * 1024,
            256 * 1024,
            1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
        ];

        Self {
            available: HashMap::new(),
            allocated: HashMap::new(),
            buckets,
            total_allocated: 0,
            max_size,
        }
    }

    /// Find appropriate bucket for size
    fn bucket_for_size(&self, size: u64) -> u64 {
        for &bucket in &self.buckets {
            if size <= bucket {
                return bucket;
            }
        }
        // Round up to next power of 2
        size.next_power_of_two()
    }

    /// Allocate a buffer from pool
    pub fn allocate(
        &mut self,
        size: u64,
        usage: Vec<BufferUsage>,
        label: impl Into<String>,
    ) -> Result<GpuBuffer> {
        let bucket = self.bucket_for_size(size);

        // Check if we have available buffer in this bucket
        if let Some(buffers) = self.available.get_mut(&bucket) {
            if let Some(mut buffer) = buffers.pop() {
                buffer.label = label.into();
                buffer.usage = usage;
                let id = buffer.id;
                self.allocated.insert(id, buffer.clone());
                return Ok(buffer);
            }
        }

        // Check memory limit
        if self.total_allocated + bucket > self.max_size {
            return Err(WebGpuError::OutOfMemory {
                requested_mb: bucket / (1024 * 1024),
                available_mb: (self.max_size - self.total_allocated) / (1024 * 1024),
            });
        }

        // Create new buffer
        let buffer = GpuBuffer::new(bucket, usage, label);
        self.total_allocated += bucket;
        let id = buffer.id;
        self.allocated.insert(id, buffer.clone());

        Ok(buffer)
    }

    /// Release a buffer back to pool
    pub fn release(&mut self, buffer_id: u64) {
        if let Some(buffer) = self.allocated.remove(&buffer_id) {
            let bucket = buffer.size;
            self.available.entry(bucket).or_default().push(buffer);
        }
    }

    /// Clear all pooled buffers
    pub fn clear(&mut self) {
        self.available.clear();
        self.allocated.clear();
        self.total_allocated = 0;
    }

    /// Total allocated memory
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
    }

    /// Available memory
    pub fn available_memory(&self) -> u64 {
        self.max_size.saturating_sub(self.total_allocated)
    }

    /// Number of pooled buffers
    pub fn pooled_count(&self) -> usize {
        self.available.values().map(|v| v.len()).sum()
    }

    /// Number of active allocations
    pub fn active_count(&self) -> usize {
        self.allocated.len()
    }
}

/// Memory layout for structured data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayout {
    /// Total size in bytes
    pub size: u64,
    /// Alignment requirement
    pub alignment: u64,
    /// Field offsets
    pub fields: Vec<FieldLayout>,
}

/// Layout of a single field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLayout {
    /// Field name
    pub name: String,
    /// Offset in bytes
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

impl MemoryLayout {
    /// Create layout for an array of f32
    pub fn f32_array(count: usize) -> Self {
        Self {
            size: (count * 4) as u64,
            alignment: 4,
            fields: vec![FieldLayout {
                name: "data".into(),
                offset: 0,
                size: (count * 4) as u64,
            }],
        }
    }

    /// Create layout for matrix (column-major)
    pub fn matrix(rows: usize, cols: usize) -> Self {
        let size = (rows * cols * 4) as u64;
        Self {
            size,
            alignment: 16, // vec4 alignment
            fields: vec![FieldLayout {
                name: "matrix".into(),
                offset: 0,
                size,
            }],
        }
    }

    /// Aligned size
    pub fn aligned_size(&self) -> u64 {
        GpuBuffer::aligned_size(self.size, self.alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = GpuBuffer::storage(1024, "test");
        assert_eq!(buffer.size, 1024);
        assert!(buffer.usage.contains(&BufferUsage::Storage));
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(10 * 1024 * 1024); // 10MB

        // Allocate
        let buffer1 = pool.allocate(1000, vec![BufferUsage::Storage], "buf1").unwrap();
        assert!(buffer1.size >= 1000);

        let buffer2 = pool.allocate(2000, vec![BufferUsage::Storage], "buf2").unwrap();
        assert!(buffer2.size >= 2000);

        assert_eq!(pool.active_count(), 2);

        // Release
        pool.release(buffer1.id);
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.pooled_count(), 1);

        // Reallocate should reuse
        let buffer3 = pool.allocate(500, vec![BufferUsage::Storage], "buf3").unwrap();
        assert_eq!(pool.pooled_count(), 0); // Buffer was reused
    }

    #[test]
    fn test_bucket_sizes() {
        let pool = BufferPool::new(100 * 1024 * 1024);

        assert_eq!(pool.bucket_for_size(100), 1024);
        assert_eq!(pool.bucket_for_size(2000), 4096);
        assert_eq!(pool.bucket_for_size(1024 * 1024), 1024 * 1024);
    }

    #[test]
    fn test_memory_layout() {
        let layout = MemoryLayout::f32_array(100);
        assert_eq!(layout.size, 400);
        assert_eq!(layout.alignment, 4);

        let matrix = MemoryLayout::matrix(4, 4);
        assert_eq!(matrix.size, 64);
        assert_eq!(matrix.alignment, 16);
    }
}

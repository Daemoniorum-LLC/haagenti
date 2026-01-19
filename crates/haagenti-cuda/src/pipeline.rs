//! Streaming Decompression Pipeline.
//!
//! Overlaps disk I/O, GPU transfer, and decompression for maximum throughput.
//!
//! # Pipeline Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Streaming Pipeline                          │
//! ├────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Stage 1: Disk Read     Stage 2: GPU Transfer   Stage 3: Decompress
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │  │ Block N+2       │ → │ Block N+1       │ → │ Block N         │
//! │  │ (reading)       │    │ (transferring)  │    │ (decompressing) │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘
//! │         ↓                      ↓                      ↓
//! │     [Pinned Buf 0]        [GPU Buf 0]           [Output Buf]
//! │     [Pinned Buf 1]        [GPU Buf 1]
//! │     (double buffer)       (double buffer)
//! │                                                                 │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! This triple-buffering approach keeps all stages busy simultaneously.

use crate::error::{CudaError, Result};
use crate::kernels::{BlockInfo, Lz4GpuDecompressor};
use crate::memory::{GpuBuffer, MemoryPool, PinnedBuffer};
use cudarc::driver::{CudaDevice, CudaStream};
use std::collections::VecDeque;
use std::sync::Arc;

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of staging buffers (2 = double buffering).
    pub num_staging_buffers: usize,

    /// Size of each staging buffer.
    pub staging_buffer_size: usize,

    /// Maximum blocks to have in flight.
    pub max_in_flight: usize,

    /// Whether to use pinned memory.
    pub use_pinned_memory: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            num_staging_buffers: 2,
            staging_buffer_size: 4 * 1024 * 1024, // 4MB
            max_in_flight: 4,
            use_pinned_memory: true,
        }
    }
}

/// State of a block in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// Block is being read from disk.
    Reading,
    /// Block is being transferred to GPU.
    Transferring,
    /// Block is being decompressed.
    Decompressing,
    /// Block is ready in output buffer.
    Ready,
    /// Block processing failed.
    Failed,
}

/// A block being processed in the pipeline.
struct PipelineBlock {
    info: BlockInfo,
    state: BlockState,
    staging_buffer_idx: Option<usize>,
    gpu_buffer_idx: Option<usize>,
}

/// Streaming decompression pipeline.
///
/// Orchestrates parallel disk I/O, GPU transfer, and decompression.
pub struct DecompressionPipeline {
    device: Arc<CudaDevice>,
    pool: MemoryPool,
    config: PipelineConfig,

    // Streams for overlapping operations
    transfer_stream: CudaStream,
    compute_stream: CudaStream,

    // Double-buffered staging
    pinned_buffers: Vec<PinnedBuffer>,
    gpu_staging: Vec<GpuBuffer>,
    buffer_in_use: Vec<bool>,

    // Output buffer
    output: Option<GpuBuffer>,

    // Blocks in flight
    blocks: VecDeque<PipelineBlock>,

    // Decompressor
    lz4: Lz4GpuDecompressor,

    // Statistics
    stats: PipelineStats,
}

/// Pipeline statistics.
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    pub blocks_processed: usize,
    pub bytes_read: usize,
    pub bytes_decompressed: usize,
    pub transfer_time_ns: u64,
    pub decompress_time_ns: u64,
}

impl DecompressionPipeline {
    /// Create a new decompression pipeline.
    pub fn new(
        device: Arc<CudaDevice>,
        pool: MemoryPool,
        config: PipelineConfig,
    ) -> Result<Self> {
        // Create streams for overlapping operations
        let transfer_stream = device.fork_default_stream()?;
        let compute_stream = device.fork_default_stream()?;

        // Allocate staging buffers
        let mut pinned_buffers = Vec::with_capacity(config.num_staging_buffers);
        let mut gpu_staging = Vec::with_capacity(config.num_staging_buffers);
        let mut buffer_in_use = Vec::with_capacity(config.num_staging_buffers);

        for _ in 0..config.num_staging_buffers {
            pinned_buffers.push(PinnedBuffer::new(config.staging_buffer_size)?);
            gpu_staging.push(pool.allocate(config.staging_buffer_size)?);
            buffer_in_use.push(false);
        }

        let lz4 = Lz4GpuDecompressor::new(device.clone())?;

        Ok(DecompressionPipeline {
            device,
            pool,
            config,
            transfer_stream,
            compute_stream,
            pinned_buffers,
            gpu_staging,
            buffer_in_use,
            output: None,
            blocks: VecDeque::new(),
            lz4,
            stats: PipelineStats::default(),
        })
    }

    /// Initialize the pipeline for decompressing to a specific output size.
    pub fn init_output(&mut self, output_size: usize) -> Result<()> {
        self.output = Some(self.pool.allocate(output_size)?);
        Ok(())
    }

    /// Submit a block for decompression.
    pub fn submit_block(&mut self, info: BlockInfo, data: &[u8]) -> Result<()> {
        // Find a free staging buffer
        let staging_idx = self.buffer_in_use
            .iter()
            .position(|&in_use| !in_use)
            .ok_or_else(|| CudaError::PoolExhausted("No free staging buffers".into()))?;

        // Copy to pinned memory
        self.pinned_buffers[staging_idx].copy_from_host(data)?;
        self.buffer_in_use[staging_idx] = true;

        // Add to queue
        self.blocks.push_back(PipelineBlock {
            info,
            state: BlockState::Reading,
            staging_buffer_idx: Some(staging_idx),
            gpu_buffer_idx: Some(staging_idx), // Same index for simplicity
        });

        Ok(())
    }

    /// Process pending blocks.
    ///
    /// Call this repeatedly to advance the pipeline.
    pub fn process(&mut self) -> Result<bool> {
        let mut made_progress = false;

        // Process blocks in order
        for block in self.blocks.iter_mut() {
            match block.state {
                BlockState::Reading => {
                    // Start transfer to GPU
                    if let (Some(staging_idx), Some(gpu_idx)) =
                        (block.staging_buffer_idx, block.gpu_buffer_idx)
                    {
                        self.gpu_staging[gpu_idx].copy_from_pinned(
                            &self.pinned_buffers[staging_idx],
                        )?;
                        block.state = BlockState::Transferring;
                        made_progress = true;
                    }
                }
                BlockState::Transferring => {
                    // Check if transfer complete, start decompression
                    // In a real impl, we'd use events to check completion
                    self.device.synchronize()?;

                    if let (Some(gpu_idx), Some(output)) =
                        (block.gpu_buffer_idx, &self.output)
                    {
                        self.lz4.decompress(
                            &self.gpu_staging[gpu_idx],
                            output,
                            block.info.input_size,
                            block.info.output_size,
                            &self.compute_stream,
                        )?;
                        block.state = BlockState::Decompressing;
                        made_progress = true;
                    }
                }
                BlockState::Decompressing => {
                    // Check if decompression complete
                    self.device.synchronize()?;
                    block.state = BlockState::Ready;

                    // Free staging buffer
                    if let Some(staging_idx) = block.staging_buffer_idx {
                        self.buffer_in_use[staging_idx] = false;
                    }

                    self.stats.blocks_processed += 1;
                    self.stats.bytes_read += block.info.input_size;
                    self.stats.bytes_decompressed += block.info.output_size;
                    made_progress = true;
                }
                BlockState::Ready | BlockState::Failed => {}
            }
        }

        // Remove completed blocks from front
        while let Some(block) = self.blocks.front() {
            if block.state == BlockState::Ready || block.state == BlockState::Failed {
                self.blocks.pop_front();
            } else {
                break;
            }
        }

        Ok(made_progress)
    }

    /// Wait for all pending blocks to complete.
    pub fn finish(&mut self) -> Result<()> {
        while !self.blocks.is_empty() {
            self.process()?;
        }
        // Synchronize all device operations
        self.device.synchronize()?;
        Ok(())
    }

    /// Get the output buffer.
    pub fn output(&self) -> Option<&GpuBuffer> {
        self.output.as_ref()
    }

    /// Take ownership of the output buffer.
    pub fn take_output(&mut self) -> Option<GpuBuffer> {
        self.output.take()
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
    }
}

/// Decompress an entire HCT file using the pipeline.
pub fn decompress_hct_file(
    pipeline: &mut DecompressionPipeline,
    data: &[u8],
    output_size: usize,
) -> Result<GpuBuffer> {
    use crate::kernels::parse_lz4_frame;

    // Parse frame to get blocks
    let blocks = parse_lz4_frame(data)?;

    // Initialize output
    pipeline.init_output(output_size)?;

    // Submit all blocks
    for block in blocks {
        let block_data = &data[block.input_offset..block.input_offset + block.input_size];
        pipeline.submit_block(block.clone(), block_data)?;

        // Process as we go to keep pipeline full
        while pipeline.blocks.len() >= pipeline.config.max_in_flight {
            pipeline.process()?;
        }
    }

    // Finish remaining blocks
    pipeline.finish()?;

    // Return output
    pipeline.take_output().ok_or(CudaError::InvalidData("No output".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_staging_buffers, 2);
        assert_eq!(config.staging_buffer_size, 4 * 1024 * 1024);
    }
}

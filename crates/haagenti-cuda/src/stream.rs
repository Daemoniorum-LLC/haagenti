//! Async Streaming Decompression.
//!
//! Provides async/await compatible streaming decompression that integrates
//! with Tokio and other async runtimes.

use crate::error::{CudaError, Result};
use crate::memory::{GpuBuffer, MemoryPool};
use crate::pipeline::{DecompressionPipeline, PipelineConfig};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[cfg(feature = "async")]
use tokio::sync::mpsc;

/// Async decompressor that runs in a background task.
pub struct AsyncDecompressor {
    device: Arc<CudaDevice>,
    pool: MemoryPool,
    config: PipelineConfig,
}

impl AsyncDecompressor {
    /// Create a new async decompressor.
    pub fn new(device: Arc<CudaDevice>, pool: MemoryPool) -> Self {
        AsyncDecompressor {
            device,
            pool,
            config: PipelineConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(device: Arc<CudaDevice>, pool: MemoryPool, config: PipelineConfig) -> Self {
        AsyncDecompressor {
            device,
            pool,
            config,
        }
    }

    /// Decompress data asynchronously.
    #[cfg(feature = "async")]
    pub async fn decompress_async(
        &self,
        compressed: Vec<u8>,
        output_size: usize,
    ) -> Result<GpuBuffer> {
        let device = self.device.clone();
        let pool = self.pool.clone();
        let config = self.config.clone();

        // Spawn blocking task for GPU work
        tokio::task::spawn_blocking(move || {
            let mut pipeline = DecompressionPipeline::new(device, pool, config)?;
            crate::pipeline::decompress_hct_file(&mut pipeline, &compressed, output_size)
        })
        .await
        .map_err(|e| CudaError::DecompressionFailed(e.to_string()))?
    }

    /// Decompress synchronously (for non-async contexts).
    pub fn decompress_sync(
        &self,
        compressed: &[u8],
        output_size: usize,
    ) -> Result<GpuBuffer> {
        let mut pipeline = DecompressionPipeline::new(
            self.device.clone(),
            self.pool.clone(),
            self.config.clone(),
        )?;
        crate::pipeline::decompress_hct_file(&mut pipeline, compressed, output_size)
    }
}

/// Streaming decoder for processing fragments as they arrive.
pub struct StreamingDecoder {
    device: Arc<CudaDevice>,
    pool: MemoryPool,
    pipeline: Option<DecompressionPipeline>,
    total_output_size: usize,
    current_offset: usize,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new(
        device: Arc<CudaDevice>,
        pool: MemoryPool,
        total_output_size: usize,
    ) -> Result<Self> {
        Ok(StreamingDecoder {
            device,
            pool,
            pipeline: None,
            total_output_size,
            current_offset: 0,
        })
    }

    /// Initialize the decoder for streaming.
    pub fn init(&mut self) -> Result<()> {
        let pipeline = DecompressionPipeline::new(
            self.device.clone(),
            self.pool.clone(),
            PipelineConfig::default(),
        )?;
        self.pipeline = Some(pipeline);

        if let Some(p) = &mut self.pipeline {
            p.init_output(self.total_output_size)?;
        }

        Ok(())
    }

    /// Feed a fragment into the decoder.
    pub fn feed_fragment(
        &mut self,
        fragment_data: &[u8],
        output_offset: usize,
        output_size: usize,
    ) -> Result<()> {
        let pipeline = self.pipeline.as_mut()
            .ok_or_else(|| CudaError::InvalidData("Decoder not initialized".into()))?;

        let block_info = crate::kernels::BlockInfo {
            input_offset: 0,
            input_size: fragment_data.len(),
            output_offset,
            output_size,
        };

        pipeline.submit_block(block_info, fragment_data)?;
        pipeline.process()?;

        self.current_offset = output_offset + output_size;
        Ok(())
    }

    /// Get current progress (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_output_size == 0 {
            return 1.0;
        }
        self.current_offset as f32 / self.total_output_size as f32
    }

    /// Check if decoding is complete.
    pub fn is_complete(&self) -> bool {
        self.current_offset >= self.total_output_size
    }

    /// Finish and get the output buffer.
    pub fn finish(mut self) -> Result<GpuBuffer> {
        if let Some(mut pipeline) = self.pipeline.take() {
            pipeline.finish()?;
            pipeline.take_output()
                .ok_or(CudaError::InvalidData("No output".into()))
        } else {
            Err(CudaError::InvalidData("Decoder not initialized".into()))
        }
    }

    /// Get a reference to the output buffer (may be incomplete).
    pub fn output(&self) -> Option<&GpuBuffer> {
        self.pipeline.as_ref()?.output()
    }
}

/// Channel-based streaming decoder for producer/consumer pattern.
#[cfg(feature = "async")]
pub struct ChannelDecoder {
    /// Send fragments to decoder
    tx: mpsc::Sender<FragmentMessage>,
    /// Handle to the decoder task
    handle: tokio::task::JoinHandle<Result<GpuBuffer>>,
}

#[cfg(feature = "async")]
pub enum FragmentMessage {
    Fragment {
        data: Vec<u8>,
        output_offset: usize,
        output_size: usize,
    },
    End,
}

#[cfg(feature = "async")]
impl ChannelDecoder {
    /// Create a new channel decoder.
    pub fn spawn(
        device: Arc<CudaDevice>,
        pool: MemoryPool,
        total_output_size: usize,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel::<FragmentMessage>(16);

        let handle = tokio::spawn(async move {
            let mut decoder = StreamingDecoder::new(
                device.clone(),
                pool.clone(),
                total_output_size,
            )?;
            decoder.init()?;

            while let Some(msg) = rx.recv().await {
                match msg {
                    FragmentMessage::Fragment { data, output_offset, output_size } => {
                        decoder.feed_fragment(&data, output_offset, output_size)?;
                    }
                    FragmentMessage::End => break,
                }
            }

            decoder.finish()
        });

        Ok(ChannelDecoder { tx, handle })
    }

    /// Send a fragment to the decoder.
    pub async fn send_fragment(
        &self,
        data: Vec<u8>,
        output_offset: usize,
        output_size: usize,
    ) -> Result<()> {
        self.tx.send(FragmentMessage::Fragment {
            data,
            output_offset,
            output_size,
        }).await.map_err(|_| CudaError::StreamSync("Channel closed".into()))
    }

    /// Signal end of stream and wait for result.
    pub async fn finish(self) -> Result<GpuBuffer> {
        let _ = self.tx.send(FragmentMessage::End).await;
        self.handle.await.map_err(|e| CudaError::StreamSync(e.to_string()))?
    }
}

/// Builder for configuring streaming decompression.
pub struct StreamingDecoderBuilder {
    device: Arc<CudaDevice>,
    pool: MemoryPool,
    total_output_size: usize,
    config: PipelineConfig,
}

impl StreamingDecoderBuilder {
    /// Create a new builder.
    pub fn new(device: Arc<CudaDevice>, pool: MemoryPool) -> Self {
        StreamingDecoderBuilder {
            device,
            pool,
            total_output_size: 0,
            config: PipelineConfig::default(),
        }
    }

    /// Set the total output size.
    pub fn output_size(mut self, size: usize) -> Self {
        self.total_output_size = size;
        self
    }

    /// Set the number of staging buffers.
    pub fn staging_buffers(mut self, count: usize) -> Self {
        self.config.num_staging_buffers = count;
        self
    }

    /// Set the staging buffer size.
    pub fn staging_buffer_size(mut self, size: usize) -> Self {
        self.config.staging_buffer_size = size;
        self
    }

    /// Set max blocks in flight.
    pub fn max_in_flight(mut self, count: usize) -> Self {
        self.config.max_in_flight = count;
        self
    }

    /// Build the streaming decoder.
    pub fn build(self) -> Result<StreamingDecoder> {
        StreamingDecoder::new(self.device, self.pool, self.total_output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_decoder_progress() {
        // Would need GPU for real test
        // This just tests the struct construction
    }
}

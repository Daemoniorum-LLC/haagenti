//! Production pipeline for large model compression with checkpointing.
//!
//! This module provides a robust, resumable compression pipeline designed for
//! compressing very large models (100GB+) with:
//!
//! - **Streaming I/O**: Memory-mapped shard reading to avoid loading full files
//! - **Checkpointing**: Per-tensor state tracking for crash recovery
//! - **Incremental output**: Append-only writing with deferred header finalization
//! - **Quality validation**: Sampled reconstruction quality checks
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
//! │  Shard   │───▶│  Tensor  │───▶│ Compress │───▶│  Output  │
//! │ Scanner  │    │ Streamer │    │  Worker  │    │  Writer  │
//! └──────────┘    └──────────┘    └──────────┘    └──────────┘
//!      │               │               │               │
//!      └───────────────┴───────────────┴───────────────┘
//!                              │
//!                    ┌─────────▼─────────┐
//!                    │ Checkpoint Manager │
//!                    └───────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::pipeline::{CompressionPipeline, PipelineConfig};
//!
//! // Start fresh or resume from checkpoint
//! let config = PipelineConfig {
//!     model: "meta-llama/Llama-3.1-405B".to_string(),
//!     output_dir: PathBuf::from("./compressed"),
//!     retention: 0.70,
//!     ..Default::default()
//! };
//!
//! let mut pipeline = CompressionPipeline::new_or_resume(config)?;
//! let report = pipeline.run()?;
//!
//! println!("Compressed {} tensors, ratio: {:.1}x",
//!     report.tensors_completed,
//!     report.compression_ratio);
//! ```
//!
//! ## Resumption
//!
//! If the pipeline is interrupted, simply run again with `--resume`:
//!
//! ```bash
//! cargo run --release --example compress_405b -- \
//!     --model meta-llama/Llama-3.1-405B \
//!     --output ./compressed \
//!     --resume
//! ```

mod checkpoint;
mod incremental_writer;
mod orchestrator;
mod quality;
mod shard_reader;

#[cfg(feature = "parallel")]
pub mod turbo;

pub use checkpoint::{
    CompressionCheckpoint, CompressionConfig, ShardStatus, TensorStatus,
};
pub use incremental_writer::{IncrementalHctWriter, TensorIndexEntry};
pub use orchestrator::{
    CompressionPipeline, CompressionReport, PipelineConfig, TensorResult,
};
pub use quality::{QualityReport, QualitySampler, QualitySummary};
pub use shard_reader::{discover_shards, ShardReader, TensorEntry};

#[cfg(feature = "parallel")]
pub use turbo::{TurboPipeline, TurboConfig, TurboReport};

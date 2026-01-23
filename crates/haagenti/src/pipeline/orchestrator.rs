//! Main pipeline orchestrator for large model compression.
//!
//! Coordinates shard reading, compression, output writing, and checkpointing.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use super::checkpoint::{
    CompressionCheckpoint, CompressionConfig, ProgressInfo, ShardProcessingStatus, TensorStatus,
};
use super::incremental_writer::IncrementalHctWriter;
use super::quality::{QualityReport, QualitySampler, QualitySummary};
use super::shard_reader::{discover_shards, ShardReader, TensorEntry};

use crate::compressive::CompressiveSpectralEncoder;
use crate::{Error, Result};

#[cfg(feature = "zstd")]
use haagenti_core::CompressionLevel;
#[cfg(feature = "zstd")]
use haagenti_zstd::compress::CompressContext as ZstdCompressor;

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Model path or HuggingFace ID.
    pub model: String,
    /// Output directory for compressed model and checkpoints.
    pub output_dir: PathBuf,
    /// Retention ratio (0.0-1.0).
    pub retention: f32,
    /// Number of fragments per tensor.
    pub num_fragments: u16,
    /// Essential ratio for spectral encoding.
    pub essential_ratio: f32,
    /// Quality sample rate (0.0-1.0).
    pub quality_sample_rate: f32,
    /// Checkpoint interval (tensors between checkpoints).
    pub checkpoint_interval: usize,
    /// Minimum tensor size to compress (skip smaller).
    pub min_tensor_size: usize,
    /// Maximum tensor size to compress (skip larger).
    pub max_tensor_size: usize,
    /// Maximum retries for failed tensors.
    pub max_retries: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            output_dir: PathBuf::from("./compressed"),
            retention: 0.70,
            num_fragments: 4,
            essential_ratio: 0.20,
            quality_sample_rate: 0.05,
            checkpoint_interval: 10,
            min_tensor_size: 256,
            max_tensor_size: 100_000_000, // 100M elements
            max_retries: 3,
        }
    }
}

/// Result of compressing a single tensor.
#[derive(Debug, Clone)]
pub struct TensorResult {
    /// Tensor name.
    pub name: String,
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Compression ratio.
    pub ratio: f32,
    /// Quality report (if sampled).
    pub quality: Option<QualityReport>,
    /// Processing time.
    pub duration: Duration,
}

/// Final report after compression completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionReport {
    /// Model identifier.
    pub model_id: String,
    /// Output file path.
    pub output_path: PathBuf,
    /// Total tensors processed.
    pub tensors_processed: usize,
    /// Tensors completed successfully.
    pub tensors_completed: usize,
    /// Tensors that failed.
    pub tensors_failed: usize,
    /// Tensors skipped.
    pub tensors_skipped: usize,
    /// Total input size in bytes.
    pub total_input_bytes: u64,
    /// Total output size in bytes.
    pub total_output_bytes: u64,
    /// Overall compression ratio.
    pub compression_ratio: f32,
    /// Total processing time.
    pub elapsed_seconds: f64,
    /// Average throughput in MB/s.
    pub throughput_mbps: f64,
    /// Quality summary.
    pub quality: QualitySummary,
}

/// Main compression pipeline.
pub struct CompressionPipeline {
    /// Configuration.
    config: PipelineConfig,
    /// Checkpoint state.
    checkpoint: CompressionCheckpoint,
    /// Output writer.
    writer: IncrementalHctWriter,
    /// Quality sampler.
    sampler: QualitySampler,
    /// Encoder.
    encoder: CompressiveSpectralEncoder,
    /// Start time.
    start_time: Instant,
    /// Progress bar.
    progress: Option<ProgressBar>,
}

impl CompressionPipeline {
    /// Creates a new pipeline or resumes from checkpoint.
    pub fn new_or_resume(config: PipelineConfig) -> Result<Self> {
        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| Error::io(format!("failed to create output directory: {}", e)))?;

        let checkpoint_path = config.output_dir.join("checkpoint.json");
        let output_path = config.output_dir.join("model.safetensors");

        // Try to resume from checkpoint
        let (checkpoint, writer) = if checkpoint_path.exists() {
            let mut checkpoint = CompressionCheckpoint::load(&checkpoint_path)?;

            // Reset any in-progress tensors
            checkpoint.reset_in_progress();

            // Resume writer
            let writer = if output_path.exists() {
                IncrementalHctWriter::resume(&output_path)?
            } else {
                IncrementalHctWriter::create(&output_path)?
            };

            (checkpoint, writer)
        } else {
            // Fresh start - discover shards
            let shards = discover_shards(Path::new(&config.model))?;

            let compression_config = CompressionConfig {
                retention: config.retention,
                mode: "uniform".to_string(),
                target_dtype: "f16".to_string(),
                essential_ratio: config.essential_ratio,
                num_fragments: config.num_fragments,
            };

            let mut checkpoint =
                CompressionCheckpoint::new(&config.model, compression_config, &output_path);

            // Register all shards and tensors
            for (shard_idx, shard_path) in shards.iter().enumerate() {
                let reader = ShardReader::open(shard_path)?;

                checkpoint.add_shard(shard_path.clone(), reader.tensor_count());

                for entry in reader.tensors() {
                    checkpoint.register_tensor(&entry.name, shard_idx);
                }
            }

            // Save initial checkpoint
            checkpoint.save(&checkpoint_path)?;

            let writer = IncrementalHctWriter::create(&output_path)?;

            (checkpoint, writer)
        };

        // Create encoder
        let encoder = CompressiveSpectralEncoder::new(config.num_fragments, config.retention);

        // Create sampler
        let sampler = QualitySampler::new(config.quality_sample_rate, 42);

        Ok(Self {
            config,
            checkpoint,
            writer,
            sampler,
            encoder,
            start_time: Instant::now(),
            progress: None,
        })
    }

    /// Runs the full compression pipeline.
    pub fn run(&mut self) -> Result<CompressionReport> {
        // Setup progress bar
        let total = self.checkpoint.stats.total_tensors;
        let completed = self.checkpoint.stats.completed
            + self.checkpoint.stats.failed
            + self.checkpoint.stats.skipped;

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_position(completed as u64);
        self.progress = Some(pb);

        // Process shards
        for shard_idx in 0..self.checkpoint.shards.len() {
            if matches!(
                self.checkpoint.shards[shard_idx].status,
                ShardProcessingStatus::Completed | ShardProcessingStatus::Failed
            ) {
                continue;
            }

            self.process_shard(shard_idx)?;
        }

        // Finalize
        if let Some(pb) = &self.progress {
            pb.finish_with_message("Finalizing...");
        }

        // Checkpoint before finalize
        let checkpoint_path = self.config.output_dir.join("checkpoint.json");
        self.writer.checkpoint()?;
        self.checkpoint.save(&checkpoint_path)?;

        // Create report
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let input_bytes = self.checkpoint.stats.total_input_bytes;
        let output_bytes = self.checkpoint.stats.total_output_bytes;

        let report = CompressionReport {
            model_id: self.config.model.clone(),
            output_path: self.writer.path().to_path_buf(),
            tensors_processed: self.checkpoint.stats.total_tensors,
            tensors_completed: self.checkpoint.stats.completed,
            tensors_failed: self.checkpoint.stats.failed,
            tensors_skipped: self.checkpoint.stats.skipped,
            total_input_bytes: input_bytes,
            total_output_bytes: output_bytes,
            compression_ratio: if output_bytes > 0 {
                input_bytes as f32 / output_bytes as f32
            } else {
                0.0
            },
            elapsed_seconds: elapsed,
            throughput_mbps: if elapsed > 0.0 {
                input_bytes as f64 / elapsed / 1_000_000.0
            } else {
                0.0
            },
            quality: self.sampler.summary(),
        };

        Ok(report)
    }

    /// Processes a single shard.
    fn process_shard(&mut self, shard_idx: usize) -> Result<()> {
        let shard_path = self.checkpoint.shards[shard_idx].path.clone();

        // Update shard status
        self.checkpoint.shards[shard_idx].status = ShardProcessingStatus::InProgress;

        // Open shard reader (memory-mapped)
        let reader = match ShardReader::open(&shard_path) {
            Ok(r) => r,
            Err(e) => {
                self.checkpoint.shards[shard_idx].status = ShardProcessingStatus::Failed;
                return Err(e);
            }
        };

        // Get tensors for this shard
        let tensor_names: Vec<String> = self
            .checkpoint
            .tensors
            .iter()
            .filter(|(_, status)| status.shard() == shard_idx && status.is_pending())
            .map(|(name, _)| name.clone())
            .collect();

        // Process each tensor
        let mut tensors_since_checkpoint = 0;

        for name in tensor_names {
            let entry = match reader.get(&name) {
                Some(e) => e,
                None => {
                    self.checkpoint.skip_tensor(&name, "not found in shard");
                    continue;
                }
            };

            // Check size constraints
            let num_elements = entry.num_elements();
            if num_elements < self.config.min_tensor_size {
                self.checkpoint.skip_tensor(&name, "too small");
                self.update_progress(&name);
                continue;
            }
            if num_elements > self.config.max_tensor_size {
                self.checkpoint.skip_tensor(&name, "too large");
                self.update_progress(&name);
                continue;
            }

            // Skip 1D tensors (biases) - they don't compress well
            if entry.is_1d() {
                self.checkpoint.skip_tensor(&name, "1D tensor");
                self.update_progress(&name);
                continue;
            }

            // Process the tensor
            match self.process_tensor(&reader, entry) {
                Ok(result) => {
                    self.checkpoint.complete_tensor(
                        &name,
                        result.original_size,
                        result.compressed_size,
                        self.writer.data_size() - result.compressed_size as u64,
                        result.quality.as_ref().map(|q| q.cosine_similarity),
                    );
                }
                Err(e) => {
                    self.checkpoint.fail_tensor(&name, e.to_string());
                }
            }

            self.update_progress(&name);
            tensors_since_checkpoint += 1;

            // Periodic checkpoint
            if tensors_since_checkpoint >= self.config.checkpoint_interval {
                self.writer.checkpoint()?;
                let checkpoint_path = self.config.output_dir.join("checkpoint.json");
                self.checkpoint.save(&checkpoint_path)?;

                // Also save progress file
                let progress_path = self.config.output_dir.join("progress.json");
                let elapsed = self.start_time.elapsed().as_secs_f64();
                let progress_info = ProgressInfo::from_checkpoint(&self.checkpoint, elapsed);
                progress_info.save(&progress_path)?;

                tensors_since_checkpoint = 0;
            }
        }

        // Mark shard complete
        let has_failures = self
            .checkpoint
            .tensors
            .values()
            .any(|s| matches!(s, TensorStatus::Failed { shard, .. } if *shard == shard_idx));

        self.checkpoint.shards[shard_idx].status = if has_failures {
            ShardProcessingStatus::CompletedWithErrors
        } else {
            ShardProcessingStatus::Completed
        };

        Ok(())
    }

    /// Processes a single tensor.
    fn process_tensor(
        &mut self,
        reader: &ShardReader,
        entry: &TensorEntry,
    ) -> Result<TensorResult> {
        let start = Instant::now();

        // Mark as in-progress
        self.checkpoint.start_tensor(&entry.name);

        // Read tensor data as f32
        let original_f32 = reader.tensor_f32(&entry.name)?;
        let original_size = original_f32.len() * 4; // f32 bytes

        // Determine 2D dimensions for compression
        let (width, height) = if entry.shape.len() == 2 {
            (entry.shape[1], entry.shape[0])
        } else if entry.shape.len() == 1 {
            (entry.shape[0], 1)
        } else {
            // Flatten higher-dimensional tensors to 2D
            let total: usize = entry.shape.iter().product();
            let width = entry.shape.last().copied().unwrap_or(1);
            let height = total / width;
            (width, height)
        };

        // Compress
        let fragments = self.encoder.encode_2d(&original_f32, width, height)?;

        // Serialize fragments to bytes
        let mut compressed_data = Vec::new();

        // Simple format: [num_fragments: u16][fragment_data...]
        compressed_data.extend_from_slice(&(fragments.len() as u16).to_le_bytes());

        for frag in &fragments {
            // [index: u16][flags: u16][checksum: u64][data_len: u32][data...]
            compressed_data.extend_from_slice(&frag.index.to_le_bytes());
            compressed_data.extend_from_slice(&frag.flags.to_le_bytes());
            compressed_data.extend_from_slice(&frag.checksum.to_le_bytes());
            compressed_data.extend_from_slice(&(frag.data.len() as u32).to_le_bytes());
            compressed_data.extend_from_slice(&frag.data);
        }

        // Apply zstd entropy coding for actual compression
        #[cfg(feature = "zstd")]
        let final_data = {
            let mut zstd = ZstdCompressor::new(CompressionLevel::Fast);
            zstd.compress(&compressed_data)
                .map_err(|e| Error::corrupted(format!("zstd compression failed: {}", e)))?
        };
        #[cfg(not(feature = "zstd"))]
        let final_data = compressed_data;

        let compressed_size = final_data.len();

        // Quality validation (if sampled)
        let quality = if self.sampler.should_sample(&entry.name) {
            // Decompress for validation
            let mut decoder = crate::compressive::CompressiveSpectralDecoder::new();
            for frag in &fragments {
                if frag.index == 0 {
                    decoder.add_essentials(frag)?;
                } else {
                    decoder.add_detail(frag)?;
                }
            }

            if decoder.can_reconstruct() {
                let reconstructed = decoder.reconstruct()?;
                Some(crate::pipeline::quality::QualityReport::compute(
                    &entry.name,
                    &original_f32,
                    &reconstructed,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Write to output
        self.writer.write_tensor(
            &entry.name,
            &final_data,
            entry.shape.clone(),
            format!("{:?}", entry.dtype),
        )?;

        let duration = start.elapsed();

        Ok(TensorResult {
            name: entry.name.clone(),
            original_size,
            compressed_size,
            ratio: original_size as f32 / compressed_size as f32,
            quality,
            duration,
        })
    }

    /// Updates the progress bar.
    fn update_progress(&self, tensor_name: &str) {
        if let Some(pb) = &self.progress {
            pb.inc(1);
            pb.set_message(tensor_name.chars().take(40).collect::<String>());
        }
    }

    /// Returns the current checkpoint state.
    #[must_use]
    pub fn checkpoint(&self) -> &CompressionCheckpoint {
        &self.checkpoint
    }

    /// Returns the pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!((config.retention - 0.70).abs() < 0.01);
        assert_eq!(config.checkpoint_interval, 10);
    }

    #[test]
    fn test_tensor_result() {
        let result = TensorResult {
            name: "test.weight".to_string(),
            original_size: 1000,
            compressed_size: 100,
            ratio: 10.0,
            quality: None,
            duration: Duration::from_millis(50),
        };

        assert_eq!(result.name, "test.weight");
        assert!((result.ratio - 10.0).abs() < 0.01);
    }

    // Integration tests require actual model files
    #[test]
    #[ignore]
    fn test_pipeline_creation() {
        let config = PipelineConfig {
            model: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            output_dir: PathBuf::from("/tmp/test-pipeline"),
            ..Default::default()
        };

        let pipeline = CompressionPipeline::new_or_resume(config);
        assert!(pipeline.is_ok());
    }
}

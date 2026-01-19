//! Checkpoint state management for resumable compression.
//!
//! Provides atomic checkpoint saves and recovery for long-running compression jobs.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Compression configuration stored in checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Retention ratio (0.0-1.0).
    pub retention: f32,
    /// Compression mode name.
    pub mode: String,
    /// Target output dtype.
    pub target_dtype: String,
    /// Essential ratio for spectral encoding.
    pub essential_ratio: f32,
    /// Number of fragments.
    pub num_fragments: u16,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            retention: 0.70,
            mode: "uniform".to_string(),
            target_dtype: "f16".to_string(),
            essential_ratio: 0.20,
            num_fragments: 4,
        }
    }
}

/// Status of a shard in the compression pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardStatus {
    /// Shard file path.
    pub path: PathBuf,
    /// Processing status.
    pub status: ShardProcessingStatus,
    /// Number of tensors in this shard.
    pub tensor_count: usize,
    /// Number of tensors completed.
    pub tensors_completed: usize,
}

/// Shard processing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardProcessingStatus {
    /// Not yet started.
    Pending,
    /// Currently being processed.
    InProgress,
    /// All tensors processed.
    Completed,
    /// Shard had errors (some tensors may have failed).
    CompletedWithErrors,
    /// Shard could not be read.
    Failed,
}

/// Status of a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum TensorStatus {
    /// Not yet processed.
    #[serde(rename = "pending")]
    Pending {
        /// Shard index containing this tensor.
        shard: usize,
    },
    /// Currently being compressed.
    #[serde(rename = "in_progress")]
    InProgress {
        /// Shard index.
        shard: usize,
        /// When processing started.
        #[serde(with = "system_time_serde")]
        started: SystemTime,
    },
    /// Successfully compressed.
    #[serde(rename = "completed")]
    Completed {
        /// Shard index.
        shard: usize,
        /// Original size in bytes.
        original_size: usize,
        /// Compressed size in bytes.
        compressed_size: usize,
        /// Offset in output file.
        output_offset: u64,
        /// Cosine similarity (if quality sampled).
        cosine: Option<f32>,
    },
    /// Compression failed.
    #[serde(rename = "failed")]
    Failed {
        /// Shard index.
        shard: usize,
        /// Error message.
        error: String,
        /// Number of retry attempts.
        retries: u32,
    },
    /// Skipped (too small, unsupported, etc.).
    #[serde(rename = "skipped")]
    Skipped {
        /// Shard index.
        shard: usize,
        /// Reason for skipping.
        reason: String,
    },
}

impl TensorStatus {
    /// Returns true if this tensor needs processing.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self, TensorStatus::Pending { .. } | TensorStatus::InProgress { .. })
    }

    /// Returns true if this tensor is done (completed, failed, or skipped).
    #[must_use]
    pub fn is_done(&self) -> bool {
        matches!(
            self,
            TensorStatus::Completed { .. }
                | TensorStatus::Failed { .. }
                | TensorStatus::Skipped { .. }
        )
    }

    /// Returns the shard index for this tensor.
    #[must_use]
    pub fn shard(&self) -> usize {
        match self {
            TensorStatus::Pending { shard }
            | TensorStatus::InProgress { shard, .. }
            | TensorStatus::Completed { shard, .. }
            | TensorStatus::Failed { shard, .. }
            | TensorStatus::Skipped { shard, .. } => *shard,
        }
    }
}

/// Aggregated statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Total tensor count.
    pub total_tensors: usize,
    /// Completed tensor count.
    pub completed: usize,
    /// Failed tensor count.
    pub failed: usize,
    /// Skipped tensor count.
    pub skipped: usize,
    /// Total input bytes processed.
    pub total_input_bytes: u64,
    /// Total output bytes written.
    pub total_output_bytes: u64,
    /// Average cosine similarity (from sampled tensors).
    pub avg_cosine: Option<f32>,
    /// Elapsed seconds.
    pub elapsed_seconds: f64,
}

/// Main checkpoint state for resumable compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionCheckpoint {
    /// Checkpoint format version.
    pub version: u32,
    /// Model identifier (path or HuggingFace ID).
    pub model_id: String,
    /// Compression configuration.
    pub config: CompressionConfig,
    /// List of shards with status.
    pub shards: Vec<ShardStatus>,
    /// Per-tensor status.
    pub tensors: HashMap<String, TensorStatus>,
    /// Output file path.
    pub output_path: PathBuf,
    /// Bytes written to output.
    pub bytes_written: u64,
    /// Aggregated statistics.
    pub stats: CompressionStats,
    /// Last update timestamp.
    #[serde(with = "system_time_serde")]
    pub last_updated: SystemTime,
}

impl CompressionCheckpoint {
    /// Current checkpoint format version.
    pub const VERSION: u32 = 1;

    /// Creates a new checkpoint for a fresh compression job.
    pub fn new(
        model_id: impl Into<String>,
        config: CompressionConfig,
        output_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            version: Self::VERSION,
            model_id: model_id.into(),
            config,
            shards: Vec::new(),
            tensors: HashMap::new(),
            output_path: output_path.into(),
            bytes_written: 0,
            stats: CompressionStats::default(),
            last_updated: SystemTime::now(),
        }
    }

    /// Adds a shard to track.
    pub fn add_shard(&mut self, path: PathBuf, tensor_count: usize) {
        self.shards.push(ShardStatus {
            path,
            status: ShardProcessingStatus::Pending,
            tensor_count,
            tensors_completed: 0,
        });
        self.stats.total_tensors += tensor_count;
    }

    /// Registers a tensor as pending.
    pub fn register_tensor(&mut self, name: impl Into<String>, shard: usize) {
        self.tensors.insert(name.into(), TensorStatus::Pending { shard });
    }

    /// Marks a tensor as in-progress.
    pub fn start_tensor(&mut self, name: &str) {
        if let Some(status) = self.tensors.get_mut(name) {
            let shard = status.shard();
            *status = TensorStatus::InProgress {
                shard,
                started: SystemTime::now(),
            };
        }
    }

    /// Marks a tensor as completed.
    pub fn complete_tensor(
        &mut self,
        name: &str,
        original_size: usize,
        compressed_size: usize,
        output_offset: u64,
        cosine: Option<f32>,
    ) {
        if let Some(status) = self.tensors.get_mut(name) {
            let shard = status.shard();
            *status = TensorStatus::Completed {
                shard,
                original_size,
                compressed_size,
                output_offset,
                cosine,
            };
            self.stats.completed += 1;
            self.stats.total_input_bytes += original_size as u64;
            self.stats.total_output_bytes += compressed_size as u64;
            self.bytes_written += compressed_size as u64;

            // Update shard progress
            if let Some(shard_status) = self.shards.get_mut(shard) {
                shard_status.tensors_completed += 1;
                if shard_status.tensors_completed >= shard_status.tensor_count {
                    shard_status.status = ShardProcessingStatus::Completed;
                }
            }
        }
        self.last_updated = SystemTime::now();
    }

    /// Marks a tensor as failed.
    pub fn fail_tensor(&mut self, name: &str, error: impl Into<String>) {
        if let Some(status) = self.tensors.get_mut(name) {
            let shard = status.shard();
            let retries = match status {
                TensorStatus::Failed { retries, .. } => *retries + 1,
                _ => 1,
            };
            *status = TensorStatus::Failed {
                shard,
                error: error.into(),
                retries,
            };
            self.stats.failed += 1;
        }
        self.last_updated = SystemTime::now();
    }

    /// Marks a tensor as skipped.
    pub fn skip_tensor(&mut self, name: &str, reason: impl Into<String>) {
        if let Some(status) = self.tensors.get_mut(name) {
            let shard = status.shard();
            *status = TensorStatus::Skipped {
                shard,
                reason: reason.into(),
            };
            self.stats.skipped += 1;
        }
        self.last_updated = SystemTime::now();
    }

    /// Returns the next tensor that needs processing.
    #[must_use]
    pub fn next_pending(&self) -> Option<&str> {
        // Find tensors that are pending and in the current or earlier shards
        let current_shard = self.current_shard_index();

        self.tensors
            .iter()
            .filter(|(_, status)| status.is_pending() && status.shard() <= current_shard)
            .min_by_key(|(name, status)| (status.shard(), name.as_str()))
            .map(|(name, _)| name.as_str())
    }

    /// Returns the index of the current shard being processed.
    #[must_use]
    pub fn current_shard_index(&self) -> usize {
        self.shards
            .iter()
            .position(|s| {
                matches!(
                    s.status,
                    ShardProcessingStatus::Pending | ShardProcessingStatus::InProgress
                )
            })
            .unwrap_or(self.shards.len().saturating_sub(1))
    }

    /// Returns true if all tensors are done.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.tensors.values().all(|s| s.is_done())
    }

    /// Returns completion progress as a fraction (0.0-1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.stats.total_tensors == 0 {
            return 1.0;
        }
        let done = self.stats.completed + self.stats.failed + self.stats.skipped;
        done as f32 / self.stats.total_tensors as f32
    }

    /// Saves checkpoint atomically to a file.
    ///
    /// Uses write-to-temp + rename for atomicity.
    pub fn save(&self, path: &Path) -> Result<()> {
        let temp_path = path.with_extension("tmp");

        // Write to temp file
        let file = File::create(&temp_path).map_err(|e| {
            Error::io(format!("failed to create checkpoint temp file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| {
            Error::io(format!("failed to serialize checkpoint: {}", e))
        })?;

        // Atomic rename
        fs::rename(&temp_path, path).map_err(|e| {
            Error::io(format!("failed to rename checkpoint: {}", e))
        })?;

        Ok(())
    }

    /// Loads checkpoint from file.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            Error::io(format!("failed to open checkpoint: {}", e))
        })?;
        let reader = BufReader::new(file);
        let checkpoint: Self = serde_json::from_reader(reader).map_err(|e| {
            Error::corrupted(format!("failed to parse checkpoint: {}", e))
        })?;

        if checkpoint.version != Self::VERSION {
            return Err(Error::corrupted(format!(
                "checkpoint version mismatch: expected {}, got {}",
                Self::VERSION,
                checkpoint.version
            )));
        }

        Ok(checkpoint)
    }

    /// Returns true if this checkpoint can be resumed.
    #[must_use]
    pub fn can_resume(&self) -> bool {
        // Check if output file exists and matches expected size
        if !self.output_path.exists() {
            return false;
        }

        // Check we have pending work
        !self.is_complete()
    }

    /// Resets all in-progress tensors back to pending.
    ///
    /// Call this when resuming to handle tensors that were interrupted mid-compression.
    pub fn reset_in_progress(&mut self) {
        for status in self.tensors.values_mut() {
            if let TensorStatus::InProgress { shard, .. } = status {
                *status = TensorStatus::Pending { shard: *shard };
            }
        }
    }
}

/// Lightweight progress tracking file (updated frequently).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Current shard index.
    pub current_shard: usize,
    /// Current tensor name.
    pub current_tensor: String,
    /// Tensors completed.
    pub tensors_completed: usize,
    /// Total tensor count.
    pub tensors_total: usize,
    /// Bytes written.
    pub bytes_written: u64,
    /// Elapsed seconds.
    pub elapsed_seconds: f64,
    /// Estimated remaining seconds.
    pub estimated_remaining_seconds: f64,
    /// Throughput in MB/s.
    pub throughput_mbps: f64,
}

impl ProgressInfo {
    /// Creates progress info from checkpoint state.
    pub fn from_checkpoint(checkpoint: &CompressionCheckpoint, elapsed: f64) -> Self {
        let completed = checkpoint.stats.completed;
        let total = checkpoint.stats.total_tensors;
        let remaining = if completed > 0 && elapsed > 0.0 {
            let rate = completed as f64 / elapsed;
            (total - completed) as f64 / rate
        } else {
            0.0
        };

        let throughput = if elapsed > 0.0 {
            checkpoint.stats.total_input_bytes as f64 / elapsed / 1_000_000.0
        } else {
            0.0
        };

        let current_tensor = checkpoint
            .tensors
            .iter()
            .find(|(_, s)| matches!(s, TensorStatus::InProgress { .. }))
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        Self {
            current_shard: checkpoint.current_shard_index(),
            current_tensor,
            tensors_completed: completed,
            tensors_total: total,
            bytes_written: checkpoint.bytes_written,
            elapsed_seconds: elapsed,
            estimated_remaining_seconds: remaining,
            throughput_mbps: throughput,
        }
    }

    /// Saves to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            Error::io(format!("failed to create progress file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self).map_err(|e| {
            Error::io(format!("failed to write progress: {}", e))
        })?;
        writer.flush().map_err(|e| {
            Error::io(format!("failed to flush progress: {}", e))
        })?;
        Ok(())
    }
}

/// Custom serde for SystemTime.
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        let millis = duration.as_millis() as u64;
        millis.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_new() {
        let checkpoint = CompressionCheckpoint::new(
            "test-model",
            CompressionConfig::default(),
            "/tmp/output.hct",
        );

        assert_eq!(checkpoint.version, CompressionCheckpoint::VERSION);
        assert_eq!(checkpoint.model_id, "test-model");
        assert!(checkpoint.tensors.is_empty());
        assert!(checkpoint.shards.is_empty());
    }

    #[test]
    fn test_checkpoint_tensor_lifecycle() {
        let mut checkpoint = CompressionCheckpoint::new(
            "test-model",
            CompressionConfig::default(),
            "/tmp/output.hct",
        );

        // Add shard and tensor
        checkpoint.add_shard(PathBuf::from("shard-0.safetensors"), 10);
        checkpoint.register_tensor("layer.0.weight", 0);

        assert!(checkpoint.tensors.get("layer.0.weight").unwrap().is_pending());

        // Start processing
        checkpoint.start_tensor("layer.0.weight");
        assert!(matches!(
            checkpoint.tensors.get("layer.0.weight"),
            Some(TensorStatus::InProgress { .. })
        ));

        // Complete
        checkpoint.complete_tensor("layer.0.weight", 1000, 100, 0, Some(0.99));
        assert!(matches!(
            checkpoint.tensors.get("layer.0.weight"),
            Some(TensorStatus::Completed { .. })
        ));

        assert_eq!(checkpoint.stats.completed, 1);
        assert_eq!(checkpoint.stats.total_input_bytes, 1000);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        let mut checkpoint = CompressionCheckpoint::new(
            "test-model",
            CompressionConfig::default(),
            "/tmp/output.hct",
        );
        checkpoint.add_shard(PathBuf::from("shard-0.safetensors"), 5);
        checkpoint.register_tensor("tensor.0", 0);
        checkpoint.complete_tensor("tensor.0", 500, 50, 0, None);

        // Save
        checkpoint.save(&path).unwrap();
        assert!(path.exists());

        // Load
        let loaded = CompressionCheckpoint::load(&path).unwrap();
        assert_eq!(loaded.model_id, "test-model");
        assert_eq!(loaded.stats.completed, 1);
        assert!(loaded.tensors.contains_key("tensor.0"));
    }

    #[test]
    fn test_checkpoint_progress() {
        let mut checkpoint = CompressionCheckpoint::new(
            "test-model",
            CompressionConfig::default(),
            "/tmp/output.hct",
        );

        checkpoint.add_shard(PathBuf::from("shard-0.safetensors"), 10);
        for i in 0..10 {
            checkpoint.register_tensor(format!("tensor.{}", i), 0);
        }

        assert!((checkpoint.progress() - 0.0).abs() < 0.01);

        // Complete 5 tensors
        for i in 0..5 {
            let name = format!("tensor.{}", i);
            checkpoint.complete_tensor(&name, 100, 10, i as u64 * 10, None);
        }

        assert!((checkpoint.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_reset_in_progress() {
        let mut checkpoint = CompressionCheckpoint::new(
            "test-model",
            CompressionConfig::default(),
            "/tmp/output.hct",
        );

        checkpoint.add_shard(PathBuf::from("shard-0.safetensors"), 2);
        checkpoint.register_tensor("tensor.0", 0);
        checkpoint.register_tensor("tensor.1", 0);

        checkpoint.start_tensor("tensor.0");
        assert!(matches!(
            checkpoint.tensors.get("tensor.0"),
            Some(TensorStatus::InProgress { .. })
        ));

        checkpoint.reset_in_progress();
        assert!(matches!(
            checkpoint.tensors.get("tensor.0"),
            Some(TensorStatus::Pending { .. })
        ));
    }
}

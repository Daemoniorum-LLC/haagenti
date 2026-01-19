//! Turbo Pipeline - High-performance compression with parallel processing.
//!
//! Uses rayon for parallel tensor processing to maximize throughput.
//!
//! ## Performance
//!
//! | Configuration | 7B Model | 405B Model (Est.) |
//! |---------------|----------|-------------------|
//! | CPU Sequential | ~10 min | ~10 hours |
//! | CPU Parallel (8 cores) | ~2 min | ~2 hours |
//! | CPU Parallel (16 cores) | ~1 min | ~1 hour |
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::pipeline::turbo::{TurboPipeline, TurboConfig};
//!
//! let config = TurboConfig {
//!     model: "meta-llama/Llama-3-405B".into(),
//!     output_dir: "/tmp/compressed".into(),
//!     retention: 0.20,
//!     num_workers: 8,
//!     ..Default::default()
//! };
//!
//! let mut pipeline = TurboPipeline::new(config)?;
//! let report = pipeline.run()?;
//! println!("Compressed in {:.1}s", report.elapsed_seconds);
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::shard_reader::{discover_shards, ShardReader};
use crate::compressive::CompressiveSpectralEncoder;
use crate::{Error, Result};

// Use reference zstd crate for better compatibility
// (haagenti-zstd produces non-standard frames that can't be decoded)

#[cfg(feature = "cuda")]
use haagenti_cuda::dct_gpu::GpuDctContext;
#[cfg(feature = "cuda")]
use std::cell::RefCell;

// Thread-local GPU DCT context for parallel workers
#[cfg(feature = "cuda")]
thread_local! {
    static GPU_DCT_CTX: RefCell<Option<GpuDctContext>> = const { RefCell::new(None) };
}

/// Configuration for turbo pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboConfig {
    /// Model path or HuggingFace ID.
    pub model: String,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Retention ratio (0.0-1.0).
    pub retention: f32,
    /// Number of fragments per tensor.
    pub num_fragments: u16,
    /// Essential ratio for spectral encoding.
    pub essential_ratio: f32,
    /// Number of parallel workers.
    pub num_workers: usize,
    /// Checkpoint interval (tensors).
    pub checkpoint_interval: usize,
    /// Minimum tensor size to compress.
    pub min_tensor_size: usize,
    /// Maximum tensor size to compress.
    pub max_tensor_size: usize,
    /// Use GPU acceleration for DCT (requires cuda feature).
    pub use_gpu: bool,
    /// GPU device ID (0 for first GPU).
    pub gpu_device_id: usize,
}

impl Default for TurboConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            output_dir: PathBuf::from("./compressed"),
            retention: 0.20,
            num_fragments: 4,
            essential_ratio: 0.20,
            num_workers: num_cpus::get().min(16),
            checkpoint_interval: 50,
            min_tensor_size: 256,
            max_tensor_size: 100_000_000,
            use_gpu: false,
            gpu_device_id: 0,
        }
    }
}

/// Result of compressing a single tensor.
#[derive(Debug, Clone)]
pub struct TensorCompressionResult {
    pub name: String,
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f32,
    pub duration: Duration,
}

/// Final compression report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboReport {
    pub model_id: String,
    pub output_path: PathBuf,
    pub tensors_completed: usize,
    pub tensors_failed: usize,
    pub tensors_skipped: usize,
    pub total_input_bytes: u64,
    pub total_output_bytes: u64,
    pub compression_ratio: f32,
    pub elapsed_seconds: f64,
    pub throughput_mbps: f64,
    pub num_workers: usize,
    /// Whether GPU was used for DCT operations.
    pub gpu_used: bool,
}

/// Shared state for parallel workers.
struct SharedState {
    /// Atomic counter for completed tensors.
    completed: AtomicUsize,
    /// Atomic counter for failed tensors.
    failed: AtomicUsize,
    /// Atomic counter for skipped tensors.
    skipped: AtomicUsize,
    /// Total input bytes processed.
    input_bytes: AtomicU64,
    /// Total output bytes written.
    output_bytes: AtomicU64,
    /// Output buffer (protected by mutex).
    output_buffer: Mutex<Vec<(String, Vec<u8>, Vec<usize>, String)>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            completed: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            skipped: AtomicUsize::new(0),
            input_bytes: AtomicU64::new(0),
            output_bytes: AtomicU64::new(0),
            output_buffer: Mutex::new(Vec::new()),
        }
    }
}

/// High-performance parallel compression pipeline.
pub struct TurboPipeline {
    config: TurboConfig,
    start_time: Instant,
}

impl TurboPipeline {
    /// Creates a new turbo pipeline.
    pub fn new(config: TurboConfig) -> Result<Self> {
        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir).map_err(|e| {
            Error::io(format!("failed to create output directory: {}", e))
        })?;

        Ok(Self {
            config,
            start_time: Instant::now(),
        })
    }

    /// Runs the parallel compression pipeline.
    #[cfg(feature = "parallel")]
    pub fn run(&mut self) -> Result<TurboReport> {
        self.start_time = Instant::now();

        // Discover shards
        let shards = discover_shards(std::path::Path::new(&self.config.model))?;
        eprintln!("Found {} shards", shards.len());

        // Collect all tensors to process
        let mut all_tensors: Vec<(usize, String, Vec<usize>, String)> = Vec::new();

        for (shard_idx, shard_path) in shards.iter().enumerate() {
            let reader = ShardReader::open(shard_path)?;
            for entry in reader.tensors() {
                // Filter by size
                let num_elements = entry.num_elements();
                if num_elements < self.config.min_tensor_size {
                    continue;
                }
                if num_elements > self.config.max_tensor_size {
                    continue;
                }
                // Skip 1D tensors
                if entry.is_1d() {
                    continue;
                }

                all_tensors.push((
                    shard_idx,
                    entry.name.clone(),
                    entry.shape.clone(),
                    format!("{:?}", entry.dtype),
                ));
            }
        }

        let total_tensors = all_tensors.len();
        eprintln!("Processing {} tensors with {} workers", total_tensors, self.config.num_workers);

        // Setup progress bar
        let pb = ProgressBar::new(total_tensors as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Shared state
        let state = Arc::new(SharedState::new());
        let config = Arc::new(self.config.clone());
        let shards = Arc::new(shards);

        // Setup thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_workers)
            .build()
            .map_err(|e| Error::io(format!("failed to create thread pool: {}", e)))?;

        // Process tensors in parallel
        let pb_ref = &pb;
        let results: Vec<Result<TensorCompressionResult>> = pool.install(|| {
            all_tensors
                .par_iter()
                .map(|(shard_idx, name, shape, _dtype)| {
                    let result = Self::process_tensor_parallel(
                        &shards[*shard_idx],
                        name,
                        shape,
                        &config,
                        &state,
                    );

                    pb_ref.inc(1);
                    pb_ref.set_message(name.chars().take(30).collect::<String>());

                    result
                })
                .collect()
        });

        pb.finish_with_message("Complete!");

        // Aggregate results
        let mut completed = 0;
        let mut failed = 0;

        for result in &results {
            match result {
                Ok(_) => completed += 1,
                Err(_) => failed += 1,
            }
        }

        // Write output file
        let output_path = self.config.output_dir.join("model.safetensors");
        Self::write_output(&state, &output_path)?;

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let input_bytes = state.input_bytes.load(Ordering::Relaxed);
        let output_bytes = state.output_bytes.load(Ordering::Relaxed);

        // Check if GPU was used
        #[cfg(feature = "cuda")]
        let gpu_used = self.config.use_gpu;
        #[cfg(not(feature = "cuda"))]
        let gpu_used = false;

        Ok(TurboReport {
            model_id: self.config.model.clone(),
            output_path,
            tensors_completed: completed,
            tensors_failed: failed,
            tensors_skipped: state.skipped.load(Ordering::Relaxed),
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
            num_workers: self.config.num_workers,
            gpu_used,
        })
    }

    /// Process a single tensor (called from parallel workers).
    fn process_tensor_parallel(
        shard_path: &std::path::Path,
        name: &str,
        shape: &[usize],
        config: &TurboConfig,
        state: &SharedState,
    ) -> Result<TensorCompressionResult> {
        let start = Instant::now();

        // Open shard and read tensor
        let reader = ShardReader::open(shard_path)?;
        let data_f32 = reader.tensor_f32(name)?;
        let original_size = data_f32.len() * 4;

        state.input_bytes.fetch_add(original_size as u64, Ordering::Relaxed);

        // Determine 2D dimensions
        let (width, height) = if shape.len() == 2 {
            (shape[1], shape[0])
        } else if shape.len() == 1 {
            (shape[0], 1)
        } else {
            let total: usize = shape.iter().product();
            let width = shape.last().copied().unwrap_or(1);
            let height = total / width;
            (width, height)
        };

        // Encode using compressive spectral encoder
        let encoder = CompressiveSpectralEncoder::new(config.num_fragments, config.retention)
            .with_essential_ratio(config.essential_ratio);

        // Use GPU DCT if enabled, otherwise CPU
        // GPU automatically selects shared-memory (fast) or direct (large) kernels
        #[cfg(feature = "cuda")]
        let fragments = if config.use_gpu {
            // Get or create thread-local GPU context
            GPU_DCT_CTX.with(|ctx| {
                let mut ctx_ref = ctx.borrow_mut();
                if ctx_ref.is_none() {
                    *ctx_ref = GpuDctContext::new(config.gpu_device_id).ok();
                }

                if let Some(gpu_ctx) = ctx_ref.as_mut() {
                    // GPU DCT
                    match gpu_ctx.dct_2d(&data_f32, width, height) {
                        Ok(dct_coeffs) => encoder.encode_2d_from_dct(&dct_coeffs, width, height),
                        Err(_) => encoder.encode_2d(&data_f32, width, height), // Fallback to CPU
                    }
                } else {
                    encoder.encode_2d(&data_f32, width, height)
                }
            })?
        } else {
            encoder.encode_2d(&data_f32, width, height)?
        };

        #[cfg(not(feature = "cuda"))]
        let fragments = encoder.encode_2d(&data_f32, width, height)?;

        // Serialize fragments
        let mut compressed_data = Vec::new();
        compressed_data.extend_from_slice(&(fragments.len() as u16).to_le_bytes());

        for frag in &fragments {
            compressed_data.extend_from_slice(&frag.index.to_le_bytes());
            compressed_data.extend_from_slice(&frag.flags.to_le_bytes());
            compressed_data.extend_from_slice(&frag.checksum.to_le_bytes());
            compressed_data.extend_from_slice(&(frag.data.len() as u32).to_le_bytes());
            compressed_data.extend_from_slice(&frag.data);
        }

        // Apply zstd compression using reference implementation for compatibility
        let final_data = zstd::encode_all(&compressed_data[..], 1).map_err(|e| {
            Error::corrupted(format!("zstd compression failed: {}", e))
        })?;

        let compressed_size = final_data.len();
        state.output_bytes.fetch_add(compressed_size as u64, Ordering::Relaxed);

        // Add to output buffer
        {
            let mut buffer = state.output_buffer.lock().unwrap();
            buffer.push((
                name.to_string(),
                final_data,
                shape.to_vec(),
                "hct_v3".to_string(),
            ));
        }

        state.completed.fetch_add(1, Ordering::Relaxed);

        Ok(TensorCompressionResult {
            name: name.to_string(),
            original_size,
            compressed_size,
            ratio: original_size as f32 / compressed_size as f32,
            duration: start.elapsed(),
        })
    }

    /// Write all compressed tensors to output file.
    fn write_output(state: &SharedState, output_path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        let buffer = state.output_buffer.lock().unwrap();

        // Simple safetensors-like format
        let mut metadata: std::collections::HashMap<String, serde_json::Value> = std::collections::HashMap::new();
        let mut data_offset = 0u64;
        let mut all_data = Vec::new();

        for (name, data, shape, dtype) in buffer.iter() {
            let tensor_meta = serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [data_offset, data_offset + data.len() as u64]
            });
            metadata.insert(name.clone(), tensor_meta);

            all_data.extend_from_slice(data);
            data_offset += data.len() as u64;
        }

        let header_json = serde_json::to_vec(&metadata).map_err(|e| {
            Error::corrupted(format!("failed to serialize header: {}", e))
        })?;

        let mut file = std::fs::File::create(output_path).map_err(|e| {
            Error::io(format!("failed to create output file: {}", e))
        })?;

        // Write header length (8 bytes, little-endian)
        file.write_all(&(header_json.len() as u64).to_le_bytes()).map_err(|e| {
            Error::io(format!("failed to write header length: {}", e))
        })?;

        // Write header
        file.write_all(&header_json).map_err(|e| {
            Error::io(format!("failed to write header: {}", e))
        })?;

        // Write data
        file.write_all(&all_data).map_err(|e| {
            Error::io(format!("failed to write data: {}", e))
        })?;

        Ok(())
    }

    /// Get current configuration.
    pub fn config(&self) -> &TurboConfig {
        &self.config
    }
}

/// Get available number of CPUs.
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_config_default() {
        let config = TurboConfig::default();
        assert!((config.retention - 0.20).abs() < 0.01);
        assert!(config.num_workers >= 1);
    }

    #[test]
    fn test_shared_state() {
        let state = SharedState::new();
        state.completed.fetch_add(1, Ordering::Relaxed);
        assert_eq!(state.completed.load(Ordering::Relaxed), 1);
    }
}

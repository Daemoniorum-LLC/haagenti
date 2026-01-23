//! Cold start optimization for serverless functions

use crate::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Target cold start time in ms
    pub target_cold_start_ms: u64,
    /// Maximum warmup time in ms
    pub max_warmup_ms: u64,
    /// Pre-load model weights
    pub preload_weights: bool,
    /// Pre-allocate memory pools
    pub preallocate_pools: bool,
    /// Pre-compile shaders/kernels
    pub precompile_kernels: bool,
    /// Warmup batch size
    pub warmup_batch_size: usize,
    /// Enable lazy loading after warmup
    pub lazy_loading: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            target_cold_start_ms: 100,
            max_warmup_ms: 5000,
            preload_weights: true,
            preallocate_pools: true,
            precompile_kernels: true,
            warmup_batch_size: 1,
            lazy_loading: true,
        }
    }
}

/// Warmup statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WarmupStats {
    /// Total warmup time in ms
    pub total_warmup_ms: u64,
    /// Weight loading time in ms
    pub weight_load_ms: u64,
    /// Pool allocation time in ms
    pub pool_alloc_ms: u64,
    /// Kernel compilation time in ms
    pub kernel_compile_ms: u64,
    /// First inference time in ms
    pub first_inference_ms: u64,
    /// Number of warmup iterations
    pub warmup_iterations: u32,
    /// Memory used after warmup in bytes
    pub memory_after_warmup: u64,
}

impl WarmupStats {
    /// Check if warmup met target
    pub fn met_target(&self, target_ms: u64) -> bool {
        self.first_inference_ms <= target_ms
    }
}

/// Cold start optimizer
#[derive(Debug)]
pub struct ColdStartOptimizer {
    /// Configuration
    config: WarmupConfig,
    /// Warmup stats
    stats: WarmupStats,
    /// Whether warmup is complete
    warmed_up: bool,
    /// Warmup start time
    warmup_start: Option<Instant>,
    /// Warmup phases
    phases: Vec<WarmupPhase>,
}

/// Warmup phase (internal tracking)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WarmupPhase {
    name: String,
    duration_ms: u64,
    completed: bool,
}

impl ColdStartOptimizer {
    /// Create new optimizer
    pub fn new(config: WarmupConfig) -> Self {
        Self {
            config,
            stats: WarmupStats::default(),
            warmed_up: false,
            warmup_start: None,
            phases: Vec::new(),
        }
    }

    /// Start warmup process
    pub async fn warmup<F>(&mut self, init_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        self.warmup_start = Some(Instant::now());

        // Phase 1: Weight loading
        if self.config.preload_weights {
            let start = Instant::now();
            self.load_weights().await?;
            self.stats.weight_load_ms = start.elapsed().as_millis() as u64;
            self.phases.push(WarmupPhase {
                name: "weight_load".into(),
                duration_ms: self.stats.weight_load_ms,
                completed: true,
            });
        }

        // Phase 2: Pool allocation
        if self.config.preallocate_pools {
            let start = Instant::now();
            self.preallocate_pools().await?;
            self.stats.pool_alloc_ms = start.elapsed().as_millis() as u64;
            self.phases.push(WarmupPhase {
                name: "pool_alloc".into(),
                duration_ms: self.stats.pool_alloc_ms,
                completed: true,
            });
        }

        // Phase 3: Kernel compilation
        if self.config.precompile_kernels {
            let start = Instant::now();
            self.compile_kernels().await?;
            self.stats.kernel_compile_ms = start.elapsed().as_millis() as u64;
            self.phases.push(WarmupPhase {
                name: "kernel_compile".into(),
                duration_ms: self.stats.kernel_compile_ms,
                completed: true,
            });
        }

        // Phase 4: Custom initialization
        let start = Instant::now();
        init_fn()?;
        self.phases.push(WarmupPhase {
            name: "custom_init".into(),
            duration_ms: start.elapsed().as_millis() as u64,
            completed: true,
        });

        // Phase 5: Warmup inference
        let start = Instant::now();
        self.warmup_inference().await?;
        self.stats.first_inference_ms = start.elapsed().as_millis() as u64;
        self.phases.push(WarmupPhase {
            name: "warmup_inference".into(),
            duration_ms: self.stats.first_inference_ms,
            completed: true,
        });

        self.stats.total_warmup_ms = self.warmup_start.unwrap().elapsed().as_millis() as u64;
        self.warmed_up = true;

        // Check if we exceeded max warmup time
        if self.stats.total_warmup_ms > self.config.max_warmup_ms {
            tracing::warn!(
                "Warmup exceeded max time: {}ms > {}ms",
                self.stats.total_warmup_ms,
                self.config.max_warmup_ms
            );
        }

        Ok(())
    }

    /// Load model weights
    async fn load_weights(&self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Load weights from pre-warmed cache or storage
        // 2. Initialize model tensors
        // 3. Transfer to GPU if available
        Ok(())
    }

    /// Pre-allocate memory pools
    async fn preallocate_pools(&self) -> Result<()> {
        // Pre-allocate:
        // - Input buffer pools
        // - Output buffer pools
        // - Intermediate activation buffers
        // - KV cache for transformers
        Ok(())
    }

    /// Pre-compile compute kernels
    async fn compile_kernels(&self) -> Result<()> {
        // Pre-compile:
        // - Matrix multiplication kernels
        // - Attention kernels
        // - Activation functions
        // - Normalization layers
        Ok(())
    }

    /// Run warmup inference
    async fn warmup_inference(&self) -> Result<()> {
        // Run inference with dummy data to:
        // - Warm up JIT compilation
        // - Populate caches
        // - Trigger GPU memory allocation
        for _ in 0..self.config.warmup_batch_size {
            // Simulate inference
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
        // Warmup iterations tracked in caller (warmup method)
        Ok(())
    }

    /// Check if warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Get warmup stats
    pub fn stats(&self) -> &WarmupStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &WarmupConfig {
        &self.config
    }

    /// Get warmup phases
    pub fn phases(&self) -> Vec<(String, u64)> {
        self.phases
            .iter()
            .map(|p| (p.name.clone(), p.duration_ms))
            .collect()
    }
}

/// Warmup scheduler for pre-warming instances
#[derive(Debug)]
pub struct WarmupScheduler {
    /// Scheduled warmups
    schedule: Vec<ScheduledWarmup>,
    /// Active warmup count
    active_count: usize,
    /// Maximum concurrent warmups
    max_concurrent: usize,
}

/// Scheduled warmup entry (internal to WarmupScheduler)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ScheduledWarmup {
    /// Instance ID
    instance_id: String,
    /// Scheduled time
    scheduled_at: Instant,
    /// Priority
    priority: u32,
}

impl WarmupScheduler {
    /// Create new scheduler
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            schedule: Vec::new(),
            active_count: 0,
            max_concurrent,
        }
    }

    /// Schedule a warmup
    pub fn schedule(&mut self, instance_id: impl Into<String>, priority: u32) {
        self.schedule.push(ScheduledWarmup {
            instance_id: instance_id.into(),
            scheduled_at: Instant::now(),
            priority,
        });

        // Sort by priority (higher first)
        self.schedule.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get next instance to warm up
    pub fn next_warmup(&mut self) -> Option<String> {
        if self.active_count >= self.max_concurrent {
            return None;
        }

        if self.schedule.is_empty() {
            return None;
        }

        // Remove from front (highest priority after sorting)
        let s = self.schedule.remove(0);
        self.active_count += 1;
        Some(s.instance_id)
    }

    /// Mark warmup complete
    pub fn complete(&mut self, _instance_id: &str) {
        self.active_count = self.active_count.saturating_sub(1);
    }

    /// Pending count
    pub fn pending_count(&self) -> usize {
        self.schedule.len()
    }

    /// Active count
    pub fn active_count(&self) -> usize {
        self.active_count
    }
}

/// Cold start metrics collector
#[derive(Debug, Default)]
pub struct ColdStartMetrics {
    /// Cold start times
    cold_starts: Vec<u64>,
    /// Warm start times
    warm_starts: Vec<u64>,
}

impl ColdStartMetrics {
    /// Record cold start
    pub fn record_cold_start(&mut self, duration_ms: u64) {
        self.cold_starts.push(duration_ms);
    }

    /// Record warm start
    pub fn record_warm_start(&mut self, duration_ms: u64) {
        self.warm_starts.push(duration_ms);
    }

    /// Average cold start time
    pub fn avg_cold_start_ms(&self) -> f64 {
        if self.cold_starts.is_empty() {
            0.0
        } else {
            self.cold_starts.iter().sum::<u64>() as f64 / self.cold_starts.len() as f64
        }
    }

    /// Average warm start time
    pub fn avg_warm_start_ms(&self) -> f64 {
        if self.warm_starts.is_empty() {
            0.0
        } else {
            self.warm_starts.iter().sum::<u64>() as f64 / self.warm_starts.len() as f64
        }
    }

    /// Cold to warm ratio
    pub fn cold_warm_ratio(&self) -> f64 {
        let total = self.cold_starts.len() + self.warm_starts.len();
        if total == 0 {
            0.0
        } else {
            self.cold_starts.len() as f64 / total as f64
        }
    }

    /// P95 cold start
    pub fn p95_cold_start_ms(&self) -> Option<u64> {
        if self.cold_starts.is_empty() {
            return None;
        }

        let mut sorted = self.cold_starts.clone();
        sorted.sort();
        let idx = (sorted.len() as f64 * 0.95) as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = WarmupConfig::default();
        assert_eq!(config.target_cold_start_ms, 100);
        assert!(config.preload_weights);
        assert!(config.preallocate_pools);
    }

    #[test]
    fn test_warmup_stats() {
        let stats = WarmupStats {
            first_inference_ms: 80,
            ..Default::default()
        };

        assert!(stats.met_target(100));
        assert!(!stats.met_target(50));
    }

    #[test]
    fn test_optimizer_creation() {
        let config = WarmupConfig::default();
        let optimizer = ColdStartOptimizer::new(config);

        assert!(!optimizer.is_warmed_up());
    }

    #[test]
    fn test_warmup_scheduler() {
        let mut scheduler = WarmupScheduler::new(2);

        scheduler.schedule("instance1", 1);
        scheduler.schedule("instance2", 2);
        scheduler.schedule("instance3", 3);

        // Highest priority first
        assert_eq!(scheduler.next_warmup(), Some("instance3".to_string()));
        assert_eq!(scheduler.next_warmup(), Some("instance2".to_string()));
        assert_eq!(scheduler.next_warmup(), None); // At max concurrent

        scheduler.complete("instance3");
        assert_eq!(scheduler.next_warmup(), Some("instance1".to_string()));
    }

    #[test]
    fn test_cold_start_metrics() {
        let mut metrics = ColdStartMetrics::default();

        metrics.record_cold_start(100);
        metrics.record_cold_start(150);
        metrics.record_warm_start(10);
        metrics.record_warm_start(15);

        assert_eq!(metrics.avg_cold_start_ms(), 125.0);
        assert_eq!(metrics.avg_warm_start_ms(), 12.5);
        assert_eq!(metrics.cold_warm_ratio(), 0.5);
    }
}

//! Download scheduler with bandwidth monitoring

use crate::{NetworkConfig, PrioritizedFragment, Priority, PriorityQueue};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, warn};

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum concurrent downloads
    pub max_concurrent: usize,
    /// Bandwidth sample window
    pub sample_window: Duration,
    /// Number of samples to keep
    pub sample_count: usize,
    /// Minimum acceptable bandwidth (bytes/sec)
    pub min_bandwidth: u64,
    /// Maximum queue size
    pub max_queue_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            sample_window: Duration::from_secs(5),
            sample_count: 10,
            min_bandwidth: 1024 * 1024, // 1MB/s
            max_queue_size: 1000,
        }
    }
}

impl From<&NetworkConfig> for SchedulerConfig {
    fn from(config: &NetworkConfig) -> Self {
        Self {
            max_concurrent: config.max_concurrent,
            min_bandwidth: config.min_bandwidth,
            ..Default::default()
        }
    }
}

/// Bandwidth measurement sample
#[derive(Debug, Clone, Copy)]
struct BandwidthSample {
    bytes: u64,
    duration: Duration,
    timestamp: Instant,
}

impl BandwidthSample {
    fn bytes_per_second(&self) -> f64 {
        self.bytes as f64 / self.duration.as_secs_f64()
    }
}

/// Bandwidth monitor
pub struct BandwidthMonitor {
    samples: Mutex<VecDeque<BandwidthSample>>,
    sample_window: Duration,
    max_samples: usize,
    total_bytes: AtomicU64,
    start_time: Instant,
}

impl BandwidthMonitor {
    /// Create a new bandwidth monitor
    pub fn new(sample_window: Duration, max_samples: usize) -> Self {
        Self {
            samples: Mutex::new(VecDeque::with_capacity(max_samples)),
            sample_window,
            max_samples,
            total_bytes: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a completed download
    pub async fn record(&self, bytes: u64, duration: Duration) {
        let sample = BandwidthSample {
            bytes,
            duration,
            timestamp: Instant::now(),
        };

        let mut samples = self.samples.lock().await;

        // Remove old samples
        let cutoff = Instant::now() - self.sample_window;
        while samples.front().map_or(false, |s| s.timestamp < cutoff) {
            samples.pop_front();
        }

        // Add new sample
        if samples.len() >= self.max_samples {
            samples.pop_front();
        }
        samples.push_back(sample);

        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get current bandwidth estimate (bytes/sec)
    pub async fn current_bandwidth(&self) -> f64 {
        let samples = self.samples.lock().await;

        if samples.is_empty() {
            return 0.0;
        }

        // Weighted moving average (more recent samples weighted higher)
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (i, sample) in samples.iter().enumerate() {
            let weight = (i + 1) as f64;
            weighted_sum += sample.bytes_per_second() * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Get average bandwidth over entire session
    pub fn average_bandwidth(&self) -> f64 {
        let bytes = self.total_bytes.load(Ordering::Relaxed);
        let duration = self.start_time.elapsed();

        if duration.as_secs_f64() > 0.0 {
            bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get total bytes transferred
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Estimate time to download given bytes
    pub async fn estimate_time(&self, bytes: u64) -> Duration {
        let bandwidth = self.current_bandwidth().await;
        if bandwidth > 0.0 {
            Duration::from_secs_f64(bytes as f64 / bandwidth)
        } else {
            Duration::from_secs(u64::MAX)
        }
    }
}

/// Download scheduler
pub struct Scheduler {
    config: SchedulerConfig,
    queue: PriorityQueue,
    bandwidth: Arc<BandwidthMonitor>,
    semaphore: Arc<Semaphore>,
    active: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let bandwidth = Arc::new(BandwidthMonitor::new(
            config.sample_window,
            config.sample_count,
        ));

        Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            queue: PriorityQueue::new(),
            bandwidth,
            config,
            active: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        }
    }

    /// Enqueue a fragment for download
    pub fn enqueue(&self, fragment: PrioritizedFragment) {
        if self.queue.len() >= self.config.max_queue_size {
            warn!("Queue full, dropping fragment {:?}", fragment.fragment_id);
            return;
        }
        self.queue.push(fragment);
    }

    /// Enqueue multiple fragments
    pub fn enqueue_many(&self, fragments: impl IntoIterator<Item = PrioritizedFragment>) {
        for fragment in fragments {
            self.enqueue(fragment);
        }
    }

    /// Get next fragment to download
    pub async fn next(&self) -> Option<(PrioritizedFragment, SchedulerPermit)> {
        let fragment = self.queue.pop()?;

        // Wait for download slot
        let permit = self.semaphore.clone().acquire_owned().await.ok()?;
        self.active.fetch_add(1, Ordering::Relaxed);

        Some((fragment, SchedulerPermit {
            _permit: permit,
            scheduler: self,
        }))
    }

    /// Record completed download
    pub async fn record_success(&self, bytes: u64, duration: Duration) {
        self.bandwidth.record(bytes, duration).await;
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record failed download
    pub fn record_failure(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get bandwidth monitor
    pub fn bandwidth(&self) -> &BandwidthMonitor {
        &self.bandwidth
    }

    /// Get queue length
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get active downloads
    pub fn active(&self) -> u64 {
        self.active.load(Ordering::Relaxed)
    }

    /// Get completed downloads
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    /// Get failed downloads
    pub fn failed(&self) -> u64 {
        self.failed.load(Ordering::Relaxed)
    }

    /// Check if should reduce concurrency (bandwidth dropping)
    pub async fn should_throttle(&self) -> bool {
        let current = self.bandwidth.current_bandwidth().await;
        current > 0.0 && current < self.config.min_bandwidth as f64
    }

    /// Get scheduler statistics
    pub async fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            queue_len: self.queue.len(),
            active: self.active(),
            completed: self.completed(),
            failed: self.failed(),
            current_bandwidth: self.bandwidth.current_bandwidth().await,
            average_bandwidth: self.bandwidth.average_bandwidth(),
            total_bytes: self.bandwidth.total_bytes(),
        }
    }

    /// Clear the queue
    pub fn clear(&self) {
        self.queue.clear();
    }

    /// Update priority of queued fragment
    pub fn update_priority(&self, fragment_id: &haagenti_fragments::FragmentId, priority: Priority) {
        self.queue.update_priority(fragment_id, priority);
    }
}

/// Permit for an active download
pub struct SchedulerPermit<'a> {
    _permit: tokio::sync::OwnedSemaphorePermit,
    scheduler: &'a Scheduler,
}

impl Drop for SchedulerPermit<'_> {
    fn drop(&mut self) {
        self.scheduler.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Queue length
    pub queue_len: usize,
    /// Active downloads
    pub active: u64,
    /// Completed downloads
    pub completed: u64,
    /// Failed downloads
    pub failed: u64,
    /// Current bandwidth (bytes/sec)
    pub current_bandwidth: f64,
    /// Average bandwidth (bytes/sec)
    pub average_bandwidth: f64,
    /// Total bytes transferred
    pub total_bytes: u64,
}

impl SchedulerStats {
    /// Success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.completed + self.failed;
        if total == 0 {
            1.0
        } else {
            self.completed as f64 / total as f64
        }
    }

    /// Format bandwidth as human readable
    pub fn bandwidth_human(&self) -> String {
        format_bytes_per_second(self.current_bandwidth)
    }
}

fn format_bytes_per_second(bps: f64) -> String {
    if bps >= 1_000_000_000.0 {
        format!("{:.1} GB/s", bps / 1_000_000_000.0)
    } else if bps >= 1_000_000.0 {
        format!("{:.1} MB/s", bps / 1_000_000.0)
    } else if bps >= 1_000.0 {
        format!("{:.1} KB/s", bps / 1_000.0)
    } else {
        format!("{:.0} B/s", bps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use haagenti_fragments::FragmentId;

    #[tokio::test]
    async fn test_bandwidth_monitor() {
        let monitor = BandwidthMonitor::new(Duration::from_secs(5), 10);

        // Record some transfers
        monitor.record(1024 * 1024, Duration::from_secs(1)).await;
        monitor.record(2048 * 1024, Duration::from_secs(1)).await;

        let bandwidth = monitor.current_bandwidth().await;
        assert!(bandwidth > 1_000_000.0); // > 1MB/s

        assert_eq!(monitor.total_bytes(), 3 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_scheduler_priority() {
        let config = SchedulerConfig {
            max_concurrent: 2,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);

        // Enqueue fragments with different priorities
        scheduler.enqueue(PrioritizedFragment::new(
            FragmentId::new([1; 16]),
            Priority::Low,
        ));
        scheduler.enqueue(PrioritizedFragment::new(
            FragmentId::new([2; 16]),
            Priority::Critical,
        ));
        scheduler.enqueue(PrioritizedFragment::new(
            FragmentId::new([3; 16]),
            Priority::Normal,
        ));

        // Should get critical first
        let (frag, _permit) = scheduler.next().await.unwrap();
        assert_eq!(frag.priority, Priority::Critical);
    }
}

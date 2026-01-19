//! Runtime profiling and bottleneck detection

use crate::{OptError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profiler for measuring performance
#[derive(Debug)]
pub struct Profiler {
    /// Active spans
    active_spans: HashMap<String, Instant>,
    /// Completed measurements
    measurements: Vec<Measurement>,
    /// Aggregated stats
    stats: HashMap<String, AggregatedStats>,
    /// Maximum measurements to keep
    max_measurements: usize,
}

/// Single measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// Span name
    pub name: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Memory delta (bytes)
    pub memory_delta: i64,
    /// Timestamp
    pub timestamp: u64,
}

/// Aggregated statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedStats {
    /// Count
    pub count: usize,
    /// Total duration (us)
    pub total_us: u64,
    /// Min duration (us)
    pub min_us: u64,
    /// Max duration (us)
    pub max_us: u64,
    /// Mean duration (us)
    pub mean_us: f64,
    /// Standard deviation
    pub std_us: f64,
    /// Percentage of total time
    pub percentage: f32,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            active_spans: HashMap::new(),
            measurements: Vec::new(),
            stats: HashMap::new(),
            max_measurements: 10000,
        }
    }

    /// Start a span
    pub fn start(&mut self, name: impl Into<String>) {
        self.active_spans.insert(name.into(), Instant::now());
    }

    /// End a span
    pub fn end(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.active_spans.remove(name) {
            let duration = start.elapsed();
            let measurement = Measurement {
                name: name.into(),
                duration_us: duration.as_micros() as u64,
                memory_delta: 0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };

            self.record(measurement);
            Some(duration)
        } else {
            None
        }
    }

    /// Record a measurement directly
    pub fn record(&mut self, measurement: Measurement) {
        // Update stats
        let stats = self.stats.entry(measurement.name.clone()).or_default();
        let dur = measurement.duration_us;

        if stats.count == 0 {
            stats.min_us = dur;
            stats.max_us = dur;
        } else {
            stats.min_us = stats.min_us.min(dur);
            stats.max_us = stats.max_us.max(dur);
        }

        stats.count += 1;
        stats.total_us += dur;
        stats.mean_us = stats.total_us as f64 / stats.count as f64;

        // Store measurement
        self.measurements.push(measurement);

        // Trim if needed
        if self.measurements.len() > self.max_measurements {
            self.measurements.remove(0);
        }
    }

    /// Measure a closure
    pub fn measure<F, T>(&mut self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.start(name);
        let result = f();
        self.end(name);
        result
    }

    /// Get statistics for a span
    pub fn get_stats(&self, name: &str) -> Option<&AggregatedStats> {
        self.stats.get(name)
    }

    /// Get all statistics
    pub fn all_stats(&self) -> &HashMap<String, AggregatedStats> {
        &self.stats
    }

    /// Compute percentages
    pub fn compute_percentages(&mut self) {
        let total: u64 = self.stats.values().map(|s| s.total_us).sum();

        if total > 0 {
            for stats in self.stats.values_mut() {
                stats.percentage = stats.total_us as f32 / total as f32 * 100.0;
            }
        }
    }

    /// Get profile result
    pub fn result(&mut self) -> ProfileResult {
        self.compute_percentages();

        let mut sorted_stats: Vec<_> = self.stats.iter().collect();
        sorted_stats.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));

        let hotspots: Vec<String> = sorted_stats
            .iter()
            .take(5)
            .map(|(name, _)| name.to_string())
            .collect();

        let bottlenecks = self.detect_bottlenecks();

        ProfileResult {
            stats: self.stats.clone(),
            hotspots,
            bottlenecks,
            total_time_us: self.stats.values().map(|s| s.total_us).sum(),
        }
    }

    /// Detect bottlenecks
    fn detect_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (name, stats) in &self.stats {
            // High variance suggests inconsistent performance
            if stats.std_us > stats.mean_us * 0.5 && stats.count > 10 {
                bottlenecks.push(Bottleneck {
                    name: name.clone(),
                    bottleneck_type: BottleneckType::HighVariance,
                    severity: Severity::Medium,
                    suggestion: format!("Reduce variance in {}", name),
                });
            }

            // Long tail suggests outliers
            if stats.max_us > stats.mean_us as u64 * 10 {
                bottlenecks.push(Bottleneck {
                    name: name.clone(),
                    bottleneck_type: BottleneckType::LongTail,
                    severity: Severity::Low,
                    suggestion: format!("Investigate outliers in {}", name),
                });
            }

            // Dominant operation
            if stats.percentage > 50.0 {
                bottlenecks.push(Bottleneck {
                    name: name.clone(),
                    bottleneck_type: BottleneckType::DominantOperation,
                    severity: Severity::High,
                    suggestion: format!("Optimize {} - takes {:.1}% of time", name, stats.percentage),
                });
            }
        }

        bottlenecks.sort_by(|a, b| {
            b.severity.as_int().cmp(&a.severity.as_int())
        });

        bottlenecks
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.active_spans.clear();
        self.measurements.clear();
        self.stats.clear();
    }
}

/// Profile result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Statistics by span
    pub stats: HashMap<String, AggregatedStats>,
    /// Top hotspots
    pub hotspots: Vec<String>,
    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Total time (us)
    pub total_time_us: u64,
}

impl ProfileResult {
    /// Get top N spans by time
    pub fn top_n(&self, n: usize) -> Vec<(&String, &AggregatedStats)> {
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));
        sorted.into_iter().take(n).collect()
    }
}

/// Bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Operation name
    pub name: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity
    pub severity: Severity,
    /// Suggested fix
    pub suggestion: String,
}

/// Bottleneck type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Operation takes too much time
    DominantOperation,
    /// High variance in timing
    HighVariance,
    /// Occasional very slow executions
    LongTail,
    /// Memory bound
    MemoryBound,
    /// Compute bound
    ComputeBound,
    /// IO bound
    IoBound,
}

/// Bottleneck severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    fn as_int(&self) -> u8 {
        match self {
            Severity::Low => 0,
            Severity::Medium => 1,
            Severity::High => 2,
            Severity::Critical => 3,
        }
    }
}

/// Scoped timer that automatically records on drop
pub struct ScopedTimer<'a> {
    profiler: &'a mut Profiler,
    name: String,
}

impl<'a> ScopedTimer<'a> {
    /// Create new scoped timer
    pub fn new(profiler: &'a mut Profiler, name: impl Into<String>) -> Self {
        let name = name.into();
        profiler.start(&name);
        Self { profiler, name }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.profiler.end(&self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = Profiler::new();

        profiler.start("test");
        std::thread::sleep(Duration::from_millis(10));
        let duration = profiler.end("test").unwrap();

        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_profiler_measure() {
        let mut profiler = Profiler::new();

        let result = profiler.measure("computation", || {
            std::thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        assert!(profiler.get_stats("computation").is_some());
    }

    #[test]
    fn test_aggregated_stats() {
        let mut profiler = Profiler::new();

        for _ in 0..10 {
            profiler.record(Measurement {
                name: "test".into(),
                duration_us: 100,
                memory_delta: 0,
                timestamp: 0,
            });
        }

        let stats = profiler.get_stats("test").unwrap();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.total_us, 1000);
        assert_eq!(stats.mean_us, 100.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut profiler = Profiler::new();

        // Add a dominant operation
        for _ in 0..100 {
            profiler.record(Measurement {
                name: "slow_op".into(),
                duration_us: 1000,
                memory_delta: 0,
                timestamp: 0,
            });
        }

        // Add a fast operation
        for _ in 0..100 {
            profiler.record(Measurement {
                name: "fast_op".into(),
                duration_us: 10,
                memory_delta: 0,
                timestamp: 0,
            });
        }

        let result = profiler.result();
        assert!(!result.bottlenecks.is_empty());
    }

    #[test]
    fn test_profile_result() {
        let mut profiler = Profiler::new();

        profiler.record(Measurement {
            name: "a".into(),
            duration_us: 1000,
            memory_delta: 0,
            timestamp: 0,
        });

        profiler.record(Measurement {
            name: "b".into(),
            duration_us: 500,
            memory_delta: 0,
            timestamp: 0,
        });

        let result = profiler.result();
        let top = result.top_n(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "a");
    }
}

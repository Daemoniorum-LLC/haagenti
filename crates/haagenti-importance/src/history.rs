//! Historical usage tracking for fragment importance learning

use crate::Result;
use dashmap::DashMap;
use haagenti_fragments::FragmentId;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs;
use tracing::{debug, info};

/// Usage statistics for a single fragment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentUsage {
    /// Total times loaded
    pub load_count: u64,
    /// Times used in generation (vs prefetched but unused)
    pub use_count: u64,
    /// Average quality level when used
    pub avg_quality: f32,
    /// Average step at which this fragment was accessed
    pub avg_step: f32,
    /// Cumulative contribution to image quality (if measured)
    pub quality_contribution: f32,
    /// Last updated timestamp
    pub updated_at: u64,
}

impl FragmentUsage {
    /// Create a new usage record
    pub fn new() -> Self {
        Self {
            load_count: 0,
            use_count: 0,
            avg_quality: 1.0,
            avg_step: 0.0,
            quality_contribution: 0.0,
            updated_at: now(),
        }
    }

    /// Record a load event
    pub fn record_load(&mut self) {
        self.load_count += 1;
        self.updated_at = now();
    }

    /// Record a use event
    pub fn record_use(&mut self, quality: f32, step: u32) {
        self.use_count += 1;

        // Exponential moving average
        let alpha = 0.1;
        self.avg_quality = self.avg_quality * (1.0 - alpha) + quality * alpha;
        self.avg_step = self.avg_step * (1.0 - alpha) + step as f32 * alpha;

        self.updated_at = now();
    }

    /// Update quality contribution
    pub fn update_contribution(&mut self, contribution: f32) {
        let alpha = 0.1;
        self.quality_contribution =
            self.quality_contribution * (1.0 - alpha) + contribution * alpha;
        self.updated_at = now();
    }

    /// Get usage ratio (used / loaded)
    pub fn usage_ratio(&self) -> f32 {
        if self.load_count == 0 {
            0.0
        } else {
            self.use_count as f32 / self.load_count as f32
        }
    }

    /// Compute importance score from history
    pub fn importance_score(&self) -> f32 {
        // Combine multiple factors
        let usage = self.usage_ratio();
        let contribution = self.quality_contribution.max(0.0).min(1.0);
        let recency = 1.0 / (1.0 + (now() - self.updated_at) as f32 / 86400.0); // Decay over days

        // Weighted combination
        usage * 0.4 + contribution * 0.4 + recency * 0.2
    }
}

impl Default for FragmentUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated usage statistics
#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    /// Total fragments tracked
    pub total_fragments: usize,
    /// Total loads
    pub total_loads: u64,
    /// Total uses
    pub total_uses: u64,
    /// Average usage ratio
    pub avg_usage_ratio: f32,
    /// Fragments with high usage (> 0.8 ratio)
    pub high_usage_count: usize,
    /// Fragments with low usage (< 0.2 ratio)
    pub low_usage_count: usize,
}

/// Historical usage tracker
pub struct UsageHistory {
    /// Per-fragment usage data
    usage: DashMap<FragmentId, FragmentUsage>,
    /// Per-model usage data
    model_usage: DashMap<String, DashMap<FragmentId, FragmentUsage>>,
    /// Total operations
    total_operations: AtomicU64,
    /// Persistence path
    path: Option<std::path::PathBuf>,
}

impl UsageHistory {
    /// Create a new usage history
    pub fn new() -> Self {
        Self {
            usage: DashMap::new(),
            model_usage: DashMap::new(),
            total_operations: AtomicU64::new(0),
            path: None,
        }
    }

    /// Create with persistence
    pub async fn with_persistence(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut history = Self::new();
        history.path = Some(path.clone());

        // Load existing data
        if path.exists() {
            let data = fs::read(&path).await?;
            let saved: SavedHistory = bincode::deserialize(&data)?;

            for (id, usage) in saved.usage {
                history.usage.insert(id, usage);
            }

            for (model, fragments) in saved.model_usage {
                let map = DashMap::new();
                for (id, usage) in fragments {
                    map.insert(id, usage);
                }
                history.model_usage.insert(model, map);
            }

            info!("Loaded usage history: {} fragments", history.usage.len());
        }

        Ok(history)
    }

    /// Record a fragment load
    pub fn record_load(&self, fragment_id: FragmentId) {
        self.usage.entry(fragment_id).or_default().record_load();

        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fragment use
    pub fn record_use(&self, fragment_id: FragmentId, quality: f32, step: u32) {
        self.usage
            .entry(fragment_id)
            .or_default()
            .record_use(quality, step);
    }

    /// Record for a specific model
    pub fn record_model_use(
        &self,
        model_id: &str,
        fragment_id: FragmentId,
        quality: f32,
        step: u32,
    ) {
        self.model_usage
            .entry(model_id.to_string())
            .or_default()
            .entry(fragment_id)
            .or_default()
            .record_use(quality, step);
    }

    /// Get usage for a fragment
    pub fn get(&self, fragment_id: &FragmentId) -> Option<FragmentUsage> {
        self.usage.get(fragment_id).map(|r| r.value().clone())
    }

    /// Get importance score for a fragment
    pub fn importance(&self, fragment_id: &FragmentId) -> f32 {
        self.usage
            .get(fragment_id)
            .map(|r| r.importance_score())
            .unwrap_or(0.5) // Default to medium importance
    }

    /// Get importance for a specific model
    pub fn model_importance(&self, model_id: &str, fragment_id: &FragmentId) -> f32 {
        self.model_usage
            .get(model_id)
            .and_then(|model| model.get(fragment_id).map(|r| r.importance_score()))
            .unwrap_or_else(|| self.importance(fragment_id))
    }

    /// Get aggregated statistics
    pub fn stats(&self) -> UsageStats {
        let mut stats = UsageStats::default();
        stats.total_fragments = self.usage.len();

        let mut total_ratio = 0.0;

        for entry in self.usage.iter() {
            let usage = entry.value();
            stats.total_loads += usage.load_count;
            stats.total_uses += usage.use_count;

            let ratio = usage.usage_ratio();
            total_ratio += ratio;

            if ratio > 0.8 {
                stats.high_usage_count += 1;
            } else if ratio < 0.2 {
                stats.low_usage_count += 1;
            }
        }

        stats.avg_usage_ratio = if stats.total_fragments > 0 {
            total_ratio / stats.total_fragments as f32
        } else {
            0.0
        };

        stats
    }

    /// Get top N most important fragments
    pub fn top_fragments(&self, n: usize) -> Vec<(FragmentId, f32)> {
        let mut scores: Vec<_> = self
            .usage
            .iter()
            .map(|e| (*e.key(), e.value().importance_score()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(n);
        scores
    }

    /// Get least used fragments (candidates for eviction)
    pub fn least_used(&self, n: usize) -> Vec<(FragmentId, f32)> {
        let mut scores: Vec<_> = self
            .usage
            .iter()
            .map(|e| (*e.key(), e.value().importance_score()))
            .collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scores.truncate(n);
        scores
    }

    /// Persist to disk
    pub async fn persist(&self) -> Result<()> {
        let Some(ref path) = self.path else {
            return Ok(());
        };

        let saved = SavedHistory {
            usage: self
                .usage
                .iter()
                .map(|e| (*e.key(), e.value().clone()))
                .collect(),
            model_usage: self
                .model_usage
                .iter()
                .map(|e| {
                    (
                        e.key().clone(),
                        e.value()
                            .iter()
                            .map(|e2| (*e2.key(), e2.value().clone()))
                            .collect(),
                    )
                })
                .collect(),
        };

        let data = bincode::serialize(&saved)?;

        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &data).await?;
        fs::rename(&tmp, path).await?;

        debug!("Persisted usage history: {} fragments", self.usage.len());
        Ok(())
    }

    /// Clear all history
    pub fn clear(&self) {
        self.usage.clear();
        self.model_usage.clear();
        self.total_operations.store(0, Ordering::Relaxed);
    }
}

impl Default for UsageHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable history data
#[derive(Serialize, Deserialize)]
struct SavedHistory {
    usage: Vec<(FragmentId, FragmentUsage)>,
    model_usage: Vec<(String, Vec<(FragmentId, FragmentUsage)>)>,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_tracking() {
        let history = UsageHistory::new();

        let frag_id = FragmentId::new([1; 16]);

        history.record_load(frag_id);
        history.record_load(frag_id);
        history.record_use(frag_id, 1.0, 10);

        let usage = history.get(&frag_id).unwrap();
        assert_eq!(usage.load_count, 2);
        assert_eq!(usage.use_count, 1);
        assert_eq!(usage.usage_ratio(), 0.5);
    }

    #[test]
    fn test_importance_score() {
        let history = UsageHistory::new();

        let high_use = FragmentId::new([1; 16]);
        let low_use = FragmentId::new([2; 16]);

        // High usage fragment
        for _ in 0..10 {
            history.record_load(high_use);
            history.record_use(high_use, 1.0, 5);
        }

        // Low usage fragment
        for _ in 0..10 {
            history.record_load(low_use);
        }
        history.record_use(low_use, 0.5, 20);

        assert!(history.importance(&high_use) > history.importance(&low_use));
    }
}

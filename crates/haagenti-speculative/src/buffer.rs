//! Speculation buffer for managing prefetched fragments

use crate::{Intent, Result, SpeculativeError};
use dashmap::DashMap;
use haagenti_fragments::FragmentId;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::debug;

/// Configuration for speculation buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Maximum buffer size in bytes
    pub max_size: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Eviction threshold (0.0 - 1.0)
    pub eviction_threshold: f32,
    /// Time-to-live for speculated entries (ms)
    pub ttl_ms: u64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: 512 * 1024 * 1024, // 512MB
            max_entries: 1000,
            eviction_threshold: 0.8,
            ttl_ms: 30000, // 30 seconds
        }
    }
}

/// State of a buffer entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryState {
    /// Speculation started, loading in progress
    Loading,
    /// Fragment loaded and ready
    Ready,
    /// Loading was cancelled
    Cancelled,
    /// Entry has expired
    Expired,
}

/// An entry in the speculation buffer
#[derive(Debug, Clone)]
pub struct BufferEntry {
    /// Fragment ID
    pub fragment_id: FragmentId,
    /// Current state
    pub state: EntryState,
    /// Confidence when speculation started
    pub initial_confidence: f32,
    /// Current confidence
    pub current_confidence: f32,
    /// Associated intent
    pub intent_prompt: String,
    /// Size in bytes (0 if not loaded)
    pub size: usize,
    /// Creation time
    pub created_at: Instant,
    /// Priority (lower = higher priority)
    pub priority: u32,
}

impl BufferEntry {
    /// Check if entry is still valid
    pub fn is_valid(&self, ttl_ms: u64) -> bool {
        self.state == EntryState::Ready && self.created_at.elapsed().as_millis() < ttl_ms as u128
    }

    /// Check if entry should be evicted
    pub fn should_evict(&self, ttl_ms: u64) -> bool {
        self.state == EntryState::Cancelled
            || self.state == EntryState::Expired
            || self.created_at.elapsed().as_millis() > ttl_ms as u128
            || self.current_confidence < 0.3
    }
}

/// Statistics for the buffer
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Total entries
    pub total_entries: usize,
    /// Ready entries
    pub ready_entries: usize,
    /// Loading entries
    pub loading_entries: usize,
    /// Cancelled entries
    pub cancelled_entries: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Hit count (speculated and used)
    pub hits: u64,
    /// Miss count (speculated but not used)
    pub misses: u64,
    /// Wasted bytes (cancelled after loading)
    pub wasted_bytes: u64,
}

impl BufferStats {
    /// Hit rate
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Waste rate
    pub fn waste_rate(&self) -> f32 {
        if self.total_size == 0 {
            0.0
        } else {
            self.wasted_bytes as f32 / self.total_size as f32
        }
    }
}

/// Speculation buffer for managing prefetched fragments
pub struct SpeculationBuffer {
    config: BufferConfig,
    /// Entries by fragment ID
    entries: DashMap<FragmentId, BufferEntry>,
    /// Entries by intent prompt (for quick lookup)
    by_intent: DashMap<String, Vec<FragmentId>>,
    /// Current size in bytes
    current_size: AtomicU64,
    /// Statistics
    stats: Arc<RwLock<BufferStats>>,
}

impl SpeculationBuffer {
    /// Create a new speculation buffer
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            entries: DashMap::new(),
            by_intent: DashMap::new(),
            current_size: AtomicU64::new(0),
            stats: Arc::new(RwLock::new(BufferStats::default())),
        }
    }

    /// Start speculation for an intent
    pub async fn speculate(&self, intent: &Intent, fragment_ids: Vec<FragmentId>) -> Result<()> {
        // Check capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired().await;

            if self.entries.len() >= self.config.max_entries {
                return Err(SpeculativeError::BufferFull);
            }
        }

        let now = Instant::now();

        for (idx, fragment_id) in fragment_ids.iter().enumerate() {
            let entry = BufferEntry {
                fragment_id: *fragment_id,
                state: EntryState::Loading,
                initial_confidence: intent.confidence,
                current_confidence: intent.confidence,
                intent_prompt: intent.predicted_prompt.clone(),
                size: 0,
                created_at: now,
                priority: idx as u32,
            };

            self.entries.insert(*fragment_id, entry);
        }

        // Track by intent
        self.by_intent
            .entry(intent.predicted_prompt.clone())
            .or_default()
            .extend(fragment_ids);

        debug!(
            "Started speculation for '{}' ({} fragments)",
            intent.predicted_prompt,
            self.entries.len()
        );

        Ok(())
    }

    /// Mark a fragment as ready
    pub fn mark_ready(&self, fragment_id: &FragmentId, size: usize) {
        if let Some(mut entry) = self.entries.get_mut(fragment_id) {
            entry.state = EntryState::Ready;
            entry.size = size;
            self.current_size.fetch_add(size as u64, Ordering::Relaxed);
        }
    }

    /// Cancel speculation for a fragment
    pub fn cancel(&self, fragment_id: &FragmentId) {
        if let Some(mut entry) = self.entries.get_mut(fragment_id) {
            if entry.state == EntryState::Ready {
                // Already loaded, mark as wasted
                tokio::spawn({
                    let stats = self.stats.clone();
                    let size = entry.size;
                    async move {
                        let mut stats = stats.write().await;
                        stats.wasted_bytes += size as u64;
                    }
                });
            }
            entry.state = EntryState::Cancelled;
        }
    }

    /// Cancel all speculation for an intent
    pub fn cancel_intent(&self, intent_prompt: &str) {
        if let Some((_, fragment_ids)) = self.by_intent.remove(intent_prompt) {
            for fragment_id in fragment_ids {
                self.cancel(&fragment_id);
            }
            debug!("Cancelled speculation for '{}'", intent_prompt);
        }
    }

    /// Update confidence for an intent
    pub fn update_confidence(&self, intent_prompt: &str, new_confidence: f32) {
        if let Some(fragment_ids) = self.by_intent.get(intent_prompt) {
            for fragment_id in fragment_ids.iter() {
                if let Some(mut entry) = self.entries.get_mut(fragment_id) {
                    entry.current_confidence = new_confidence;
                }
            }
        }
    }

    /// Check if a fragment is ready in the buffer
    pub fn is_ready(&self, fragment_id: &FragmentId) -> bool {
        self.entries
            .get(fragment_id)
            .map(|e| e.state == EntryState::Ready && e.is_valid(self.config.ttl_ms))
            .unwrap_or(false)
    }

    /// Get a ready entry and mark as hit
    pub async fn get(&self, fragment_id: &FragmentId) -> Option<BufferEntry> {
        let entry = self.entries.get(fragment_id)?;

        if entry.state != EntryState::Ready || !entry.is_valid(self.config.ttl_ms) {
            return None;
        }

        // Record hit
        {
            let mut stats = self.stats.write().await;
            stats.hits += 1;
        }

        Some(entry.clone())
    }

    /// Evict expired entries
    pub async fn evict_expired(&self) {
        let ttl_ms = self.config.ttl_ms;
        let mut to_remove = Vec::new();
        let mut freed_size = 0u64;

        for entry in self.entries.iter() {
            if entry.should_evict(ttl_ms) {
                to_remove.push(*entry.key());
                freed_size += entry.size as u64;

                if entry.state == EntryState::Ready {
                    // Wasn't used, record as miss
                    let stats = self.stats.clone();
                    tokio::spawn(async move {
                        let mut stats = stats.write().await;
                        stats.misses += 1;
                    });
                }
            }
        }

        for id in &to_remove {
            self.entries.remove(id);
        }

        if freed_size > 0 {
            self.current_size.fetch_sub(freed_size, Ordering::Relaxed);
        }

        if !to_remove.is_empty() {
            debug!("Evicted {} expired entries", to_remove.len());
        }
    }

    /// Get buffer statistics
    pub async fn stats(&self) -> BufferStats {
        let mut stats = self.stats.read().await.clone();

        stats.total_entries = self.entries.len();
        stats.total_size = self.current_size.load(Ordering::Relaxed) as usize;

        stats.ready_entries = 0;
        stats.loading_entries = 0;
        stats.cancelled_entries = 0;

        for entry in self.entries.iter() {
            match entry.state {
                EntryState::Ready => stats.ready_entries += 1,
                EntryState::Loading => stats.loading_entries += 1,
                EntryState::Cancelled => stats.cancelled_entries += 1,
                EntryState::Expired => {}
            }
        }

        stats
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries.clear();
        self.by_intent.clear();
        self.current_size.store(0, Ordering::Relaxed);
    }

    /// Get entries for an intent
    pub fn entries_for_intent(&self, intent_prompt: &str) -> Vec<BufferEntry> {
        self.by_intent
            .get(intent_prompt)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.entries.get(id).map(|e| e.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::FragmentHint;
    use haagenti_importance::SemanticCategory;

    fn make_intent(prompt: &str, confidence: f32) -> Intent {
        Intent {
            predicted_prompt: prompt.to_string(),
            confidence,
            categories: smallvec::smallvec![SemanticCategory::Human],
            fragment_hints: vec![],
            is_commit: confidence >= 0.8,
        }
    }

    #[tokio::test]
    async fn test_speculation() {
        let buffer = SpeculationBuffer::new(BufferConfig::default());

        let intent = make_intent("portrait of a woman", 0.9);
        let fragment_ids = vec![FragmentId::new([1; 16]), FragmentId::new([2; 16])];

        buffer
            .speculate(&intent, fragment_ids.clone())
            .await
            .unwrap();

        assert_eq!(buffer.entries.len(), 2);

        // Mark one as ready
        buffer.mark_ready(&fragment_ids[0], 1024);
        assert!(buffer.is_ready(&fragment_ids[0]));
        assert!(!buffer.is_ready(&fragment_ids[1]));
    }

    #[tokio::test]
    async fn test_cancellation() {
        let buffer = SpeculationBuffer::new(BufferConfig::default());

        let intent = make_intent("portrait", 0.7);
        let fragment_ids = vec![FragmentId::new([1; 16])];

        buffer.speculate(&intent, fragment_ids).await.unwrap();
        buffer.cancel_intent("portrait");

        let stats = buffer.stats().await;
        assert_eq!(stats.cancelled_entries, 1);
    }
}

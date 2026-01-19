//! Priority queue for fragment fetching

use haagenti_fragments::FragmentId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

/// Priority level for fragment loading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Critical - needed immediately (blocking inference)
    Critical = 0,
    /// High - needed soon (next few steps)
    High = 1,
    /// Normal - standard priority
    Normal = 2,
    /// Low - prefetch/background
    Low = 3,
    /// Idle - load when nothing else to do
    Idle = 4,
}

impl Priority {
    /// Get numeric priority (lower = higher priority)
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric priority
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Priority::Critical,
            1 => Priority::High,
            2 => Priority::Normal,
            3 => Priority::Low,
            _ => Priority::Idle,
        }
    }
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower value = higher priority, so reverse the comparison
        other.as_u8().cmp(&self.as_u8())
    }
}

/// A fragment with associated priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedFragment {
    /// Fragment ID
    pub fragment_id: FragmentId,
    /// Priority level
    pub priority: Priority,
    /// Importance score (0.0 - 1.0, from ML model)
    pub importance: f32,
    /// Size in bytes (for bandwidth planning)
    pub size: usize,
    /// Deadline (if any)
    pub deadline_ms: Option<u64>,
    /// Creation timestamp
    pub created_at: u64,
}

impl PrioritizedFragment {
    /// Create a new prioritized fragment
    pub fn new(fragment_id: FragmentId, priority: Priority) -> Self {
        Self {
            fragment_id,
            priority,
            importance: 0.5,
            size: 0,
            deadline_ms: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Set importance score
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set size
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    /// Compute effective priority score (lower = higher priority)
    pub fn effective_priority(&self) -> f64 {
        let base = self.priority.as_u8() as f64;
        let importance_boost = (1.0 - self.importance as f64) * 0.5;

        // Deadline urgency
        let deadline_boost = if let Some(deadline) = self.deadline_ms {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            if deadline <= now {
                -1.0 // Past deadline, highest priority
            } else {
                let remaining = (deadline - now) as f64;
                let urgency = 1.0 - (remaining / 10000.0).min(1.0); // 10s window
                -urgency * 0.5
            }
        } else {
            0.0
        };

        base + importance_boost + deadline_boost
    }
}

impl PartialEq for PrioritizedFragment {
    fn eq(&self, other: &Self) -> bool {
        self.fragment_id == other.fragment_id
    }
}

impl Eq for PrioritizedFragment {}

impl PartialOrd for PrioritizedFragment {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedFragment {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower effective priority = should come first
        // BinaryHeap is max-heap, so we reverse
        other
            .effective_priority()
            .partial_cmp(&self.effective_priority())
            .unwrap_or(Ordering::Equal)
    }
}

/// Thread-safe priority queue for fragments
pub struct PriorityQueue {
    heap: Arc<Mutex<BinaryHeap<PrioritizedFragment>>>,
    pending: Arc<Mutex<std::collections::HashSet<FragmentId>>>,
}

impl PriorityQueue {
    /// Create a new priority queue
    pub fn new() -> Self {
        Self {
            heap: Arc::new(Mutex::new(BinaryHeap::new())),
            pending: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// Push a fragment onto the queue
    pub fn push(&self, fragment: PrioritizedFragment) {
        let mut pending = self.pending.lock().unwrap();
        if pending.contains(&fragment.fragment_id) {
            return; // Already queued
        }
        pending.insert(fragment.fragment_id);

        let mut heap = self.heap.lock().unwrap();
        heap.push(fragment);
    }

    /// Pop the highest priority fragment
    pub fn pop(&self) -> Option<PrioritizedFragment> {
        let mut heap = self.heap.lock().unwrap();
        let fragment = heap.pop()?;

        let mut pending = self.pending.lock().unwrap();
        pending.remove(&fragment.fragment_id);

        Some(fragment)
    }

    /// Peek at the highest priority fragment
    pub fn peek(&self) -> Option<PrioritizedFragment> {
        let heap = self.heap.lock().unwrap();
        heap.peek().cloned()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.heap.lock().unwrap().len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.heap.lock().unwrap().is_empty()
    }

    /// Check if a fragment is already queued
    pub fn contains(&self, fragment_id: &FragmentId) -> bool {
        self.pending.lock().unwrap().contains(fragment_id)
    }

    /// Clear the queue
    pub fn clear(&self) {
        self.heap.lock().unwrap().clear();
        self.pending.lock().unwrap().clear();
    }

    /// Update priority of a fragment
    pub fn update_priority(&self, fragment_id: &FragmentId, new_priority: Priority) {
        let mut heap = self.heap.lock().unwrap();

        // Remove and re-add with new priority
        let items: Vec<_> = heap.drain().collect();
        for mut item in items {
            if item.fragment_id == *fragment_id {
                item.priority = new_priority;
            }
            heap.push(item);
        }
    }

    /// Get all fragments at or above a priority level
    pub fn get_by_priority(&self, min_priority: Priority) -> Vec<PrioritizedFragment> {
        let heap = self.heap.lock().unwrap();
        heap.iter()
            .filter(|f| f.priority <= min_priority)
            .cloned()
            .collect()
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let queue = PriorityQueue::new();

        let low = PrioritizedFragment::new(FragmentId::new([1; 16]), Priority::Low);
        let high = PrioritizedFragment::new(FragmentId::new([2; 16]), Priority::High);
        let critical = PrioritizedFragment::new(FragmentId::new([3; 16]), Priority::Critical);

        queue.push(low);
        queue.push(high);
        queue.push(critical);

        assert_eq!(queue.pop().unwrap().priority, Priority::Critical);
        assert_eq!(queue.pop().unwrap().priority, Priority::High);
        assert_eq!(queue.pop().unwrap().priority, Priority::Low);
    }

    #[test]
    fn test_importance_affects_priority() {
        let queue = PriorityQueue::new();

        let normal_low_importance =
            PrioritizedFragment::new(FragmentId::new([1; 16]), Priority::Normal)
                .with_importance(0.2);
        let normal_high_importance =
            PrioritizedFragment::new(FragmentId::new([2; 16]), Priority::Normal)
                .with_importance(0.9);

        queue.push(normal_low_importance.clone());
        queue.push(normal_high_importance.clone());

        // Higher importance should come first
        let first = queue.pop().unwrap();
        assert_eq!(first.fragment_id, normal_high_importance.fragment_id);
    }
}

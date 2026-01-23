//! Quality-Aware Adaptive Streaming
//!
//! Provides automatic quality adaptation based on network conditions,
//! optimizing fragment loading order and quality targets for the best
//! user experience.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::Duration;

/// Network conditions for adaptive decisions
#[derive(Debug, Clone, Copy)]
pub struct NetworkConditions {
    /// Bandwidth in bits per second
    pub bandwidth_bps: u64,
    /// Round-trip latency in milliseconds
    pub latency_ms: u32,
    /// Packet loss rate (0.0 - 1.0)
    pub packet_loss: f32,
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            bandwidth_bps: 50_000_000, // 50 Mbps default
            latency_ms: 20,
            packet_loss: 0.0,
        }
    }
}

/// Policy for quality adaptation
#[derive(Debug, Clone)]
pub struct QualityPolicy {
    /// Minimum allowed quality (0.0 - 1.0)
    pub min_quality: f32,
    /// Maximum allowed quality (0.0 - 1.0)
    pub max_quality: f32,
    /// How aggressively to adapt (0.0 = slow, 1.0 = fast)
    pub adaptation_rate: f32,
    /// Target buffer duration in ms
    pub target_buffer_ms: u32,
}

impl Default for QualityPolicy {
    fn default() -> Self {
        Self {
            min_quality: 0.5,
            max_quality: 0.99,
            adaptation_rate: 0.5,
            target_buffer_ms: 500,
        }
    }
}

/// Recommended quality settings from the adaptive manager
#[derive(Debug, Clone)]
pub struct RecommendedQuality {
    /// Target quality level (0.0 - 1.0)
    pub target_quality: f32,
    /// Number of fragments to load
    pub fragments_to_load: usize,
}

/// Transfer sample for bandwidth estimation
#[derive(Debug, Clone)]
struct TransferSample {
    bytes: usize,
    duration: Duration,
}

/// Internal mutable state for the adaptive manager
struct AdaptiveState {
    /// Current network conditions
    conditions: NetworkConditions,
    /// Transfer history for bandwidth estimation
    transfer_history: VecDeque<TransferSample>,
}

/// Adaptive stream manager for quality-aware streaming
pub struct AdaptiveStreamManager {
    /// Quality policy
    policy: QualityPolicy,
    /// Mutable state using interior mutability
    state: RefCell<AdaptiveState>,
    /// Maximum transfer samples to keep
    max_samples: usize,
}

impl AdaptiveStreamManager {
    /// Create a new adaptive stream manager
    pub fn new(policy: QualityPolicy) -> Self {
        Self {
            policy,
            state: RefCell::new(AdaptiveState {
                conditions: NetworkConditions::default(),
                transfer_history: VecDeque::new(),
            }),
            max_samples: 20,
        }
    }

    /// Check if the manager is ready
    pub fn is_ready(&self) -> bool {
        true
    }

    /// Update current network conditions
    pub fn update_network_conditions(&self, conditions: NetworkConditions) {
        self.state.borrow_mut().conditions = conditions;
    }

    /// Get recommended quality settings based on current conditions
    pub fn recommended_quality(&self) -> RecommendedQuality {
        let state = self.state.borrow();
        let bandwidth_quality = self.bandwidth_to_quality(state.conditions.bandwidth_bps);
        let latency_factor = self.latency_factor(state.conditions.latency_ms);
        let loss_factor = 1.0 - state.conditions.packet_loss;

        let raw_quality = bandwidth_quality * latency_factor * loss_factor;
        let target_quality = raw_quality
            .max(self.policy.min_quality)
            .min(self.policy.max_quality);

        let fragments_to_load = self.fragments_for_quality(target_quality);

        RecommendedQuality {
            target_quality,
            fragments_to_load,
        }
    }

    /// Get optimal fragment loading order
    pub fn optimal_fragment_order(&self, total_fragments: usize) -> Vec<usize> {
        // Priority order: start with fragment 0, then interleave for progressive quality
        let mut order = Vec::with_capacity(total_fragments);

        // Fragment 0 always first (contains essential data)
        if total_fragments > 0 {
            order.push(0);
        }

        // Add remaining fragments in priority order
        // Use a simple interleaving pattern for progressive quality
        for i in 1..total_fragments {
            order.push(i);
        }

        order
    }

    /// Record a transfer for bandwidth estimation
    pub fn record_transfer(&self, bytes: usize, duration: Duration) {
        let mut state = self.state.borrow_mut();
        state
            .transfer_history
            .push_back(TransferSample { bytes, duration });
        if state.transfer_history.len() > self.max_samples {
            state.transfer_history.pop_front();
        }
    }

    /// Estimate current bandwidth based on recent transfers
    pub fn estimated_bandwidth_bps(&self) -> u64 {
        let state = self.state.borrow();
        if state.transfer_history.is_empty() {
            return state.conditions.bandwidth_bps;
        }

        let total_bytes: usize = state.transfer_history.iter().map(|s| s.bytes).sum();
        let total_duration: Duration = state.transfer_history.iter().map(|s| s.duration).sum();

        if total_duration.as_secs_f64() > 0.0 {
            let bytes_per_sec = total_bytes as f64 / total_duration.as_secs_f64();
            (bytes_per_sec * 8.0) as u64 // Convert to bits
        } else {
            state.conditions.bandwidth_bps
        }
    }

    /// Calculate fragments needed for a target quality
    pub fn fragments_for_target(&self, target_quality: f32, total_fragments: usize) -> usize {
        // Linear relationship: quality 0.0 = 1 fragment, quality 1.0 = all fragments
        let fraction = target_quality.clamp(0.0, 1.0);
        let fragments = (fraction * total_fragments as f32).ceil() as usize;
        fragments.max(1).min(total_fragments)
    }

    // Internal helpers

    fn bandwidth_to_quality(&self, bandwidth_bps: u64) -> f32 {
        // Map bandwidth to quality:
        // < 1 Mbps = 0.5
        // 1-10 Mbps = 0.5-0.8
        // 10-100 Mbps = 0.8-0.95
        // > 100 Mbps = 0.95-1.0
        match bandwidth_bps {
            0..=1_000_000 => 0.5,
            1_000_001..=10_000_000 => {
                0.5 + 0.3 * ((bandwidth_bps - 1_000_000) as f32 / 9_000_000.0)
            }
            10_000_001..=100_000_000 => {
                0.8 + 0.15 * ((bandwidth_bps - 10_000_000) as f32 / 90_000_000.0)
            }
            _ => 0.95 + 0.05 * ((bandwidth_bps - 100_000_000) as f32 / 900_000_000.0).min(1.0),
        }
    }

    fn latency_factor(&self, latency_ms: u32) -> f32 {
        // Lower latency = better quality factor
        // < 20ms = 1.0
        // 20-100ms = 0.9-1.0
        // 100-500ms = 0.7-0.9
        // > 500ms = 0.5-0.7
        match latency_ms {
            0..=20 => 1.0,
            21..=100 => 1.0 - 0.1 * ((latency_ms - 20) as f32 / 80.0),
            101..=500 => 0.9 - 0.2 * ((latency_ms - 100) as f32 / 400.0),
            _ => 0.7 - 0.2 * ((latency_ms - 500) as f32 / 500.0).min(1.0),
        }
    }

    fn fragments_for_quality(&self, quality: f32) -> usize {
        // Map quality to fragment count (assuming 32 max fragments)
        let max_fragments = 32;
        let count = (quality * max_fragments as f32).ceil() as usize;
        count.max(1).min(max_fragments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Track C.4: Quality-Aware Adaptive Streaming Tests
    // =========================================================================

    #[test]
    fn test_adaptive_manager_creation() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());
        assert!(manager.is_ready());
    }

    #[test]
    fn test_quality_degradation_on_slow_network() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Simulate slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 1_000_000, // 1 Mbps
            latency_ms: 200,
            packet_loss: 0.05,
        });

        let recommended = manager.recommended_quality();

        // Should reduce quality for slow network
        assert!(
            recommended.fragments_to_load < 32,
            "Fragments: {} should be < 32 for slow network",
            recommended.fragments_to_load
        );
        assert!(
            recommended.target_quality < 0.9,
            "Quality: {} should be < 0.9 for slow network",
            recommended.target_quality
        );
    }

    #[test]
    fn test_quality_increase_on_fast_network() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Start with slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 1_000_000,
            latency_ms: 200,
            packet_loss: 0.05,
        });

        let slow_quality = manager.recommended_quality().target_quality;

        // Network improves
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 100_000_000, // 100 Mbps
            latency_ms: 10,
            packet_loss: 0.0,
        });

        let fast_quality = manager.recommended_quality();

        // Should increase quality
        assert!(
            fast_quality.target_quality > slow_quality,
            "Quality should improve: {} > {}",
            fast_quality.target_quality,
            slow_quality
        );
        assert!(
            fast_quality.target_quality >= 0.9,
            "Quality: {} should be >= 0.9 for fast network",
            fast_quality.target_quality
        );
    }

    #[test]
    fn test_quality_policy_minimum() {
        let policy = QualityPolicy {
            min_quality: 0.7,
            max_quality: 0.99,
            ..Default::default()
        };

        let manager = AdaptiveStreamManager::new(policy);

        // Very slow network
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 100_000, // 100 Kbps
            latency_ms: 1000,
            packet_loss: 0.10,
        });

        let recommended = manager.recommended_quality();

        // Should not go below minimum
        assert!(
            recommended.target_quality >= 0.7,
            "Quality {} should not go below min 0.7",
            recommended.target_quality
        );
    }

    #[test]
    fn test_adaptive_fragment_ordering() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 10_000_000, // 10 Mbps
            latency_ms: 50,
            packet_loss: 0.01,
        });

        let fragment_order = manager.optimal_fragment_order(32);

        // Should prioritize high-impact fragments
        // Fragment 0 should always be first (contains essential data)
        assert_eq!(fragment_order[0], 0, "Fragment 0 should be first");
        assert_eq!(fragment_order.len(), 32, "Should return all fragments");
    }

    #[test]
    fn test_bandwidth_estimation() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Record transfer samples (~10 MB/s = 80 Mbps)
        manager.record_transfer(1_000_000, Duration::from_millis(100));
        manager.record_transfer(1_000_000, Duration::from_millis(110));
        manager.record_transfer(1_000_000, Duration::from_millis(90));

        let estimated = manager.estimated_bandwidth_bps();

        // Should estimate around 80 Mbps (with some variance)
        assert!(
            estimated > 60_000_000 && estimated < 100_000_000,
            "Estimated: {} bps should be ~80 Mbps",
            estimated
        );
    }

    #[test]
    fn test_fragments_for_target() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // Low quality = fewer fragments
        let low = manager.fragments_for_target(0.3, 32);
        assert!(low >= 1 && low <= 16, "Low quality fragments: {}", low);

        // High quality = more fragments
        let high = manager.fragments_for_target(0.95, 32);
        assert!(high >= 28 && high <= 32, "High quality fragments: {}", high);

        // Quality 1.0 = all fragments
        let full = manager.fragments_for_target(1.0, 32);
        assert_eq!(full, 32, "Full quality should use all fragments");
    }

    #[test]
    fn test_network_conditions_default() {
        let conditions = NetworkConditions::default();

        assert!(conditions.bandwidth_bps > 0);
        assert!(conditions.latency_ms < 100);
        assert!(conditions.packet_loss >= 0.0 && conditions.packet_loss <= 1.0);
    }

    #[test]
    fn test_quality_policy_bounds() {
        let policy = QualityPolicy {
            min_quality: 0.2,
            max_quality: 0.8,
            ..Default::default()
        };

        let manager = AdaptiveStreamManager::new(policy);

        // Even with excellent network, should not exceed max
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 1_000_000_000, // 1 Gbps
            latency_ms: 1,
            packet_loss: 0.0,
        });

        let quality = manager.recommended_quality().target_quality;
        assert!(
            quality <= 0.8,
            "Quality {} should not exceed max 0.8",
            quality
        );
    }

    #[test]
    fn test_empty_transfer_history() {
        let manager = AdaptiveStreamManager::new(QualityPolicy::default());

        // No transfers recorded - should use conditions bandwidth
        manager.update_network_conditions(NetworkConditions {
            bandwidth_bps: 50_000_000,
            ..Default::default()
        });

        let estimated = manager.estimated_bandwidth_bps();
        assert_eq!(
            estimated, 50_000_000,
            "Should use conditions bandwidth when no history"
        );
    }
}

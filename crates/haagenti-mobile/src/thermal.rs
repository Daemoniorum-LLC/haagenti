//! Thermal management for sustained mobile performance

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Thermal state levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operation
    Nominal,
    /// Slightly elevated, may throttle soon
    Fair,
    /// Throttling required
    Serious,
    /// Critical - must reduce workload significantly
    Critical,
}

impl ThermalState {
    /// Get throttle factor (1.0 = full speed, 0.0 = stopped)
    pub fn throttle_factor(&self) -> f32 {
        match self {
            ThermalState::Nominal => 1.0,
            ThermalState::Fair => 0.85,
            ThermalState::Serious => 0.5,
            ThermalState::Critical => 0.1,
        }
    }

    /// Get recommended delay between operations in ms
    pub fn recommended_delay_ms(&self) -> u64 {
        match self {
            ThermalState::Nominal => 0,
            ThermalState::Fair => 10,
            ThermalState::Serious => 50,
            ThermalState::Critical => 200,
        }
    }
}

/// Thermal management policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPolicy {
    /// Temperature threshold for Fair state (°C)
    pub fair_threshold: f32,
    /// Temperature threshold for Serious state (°C)
    pub serious_threshold: f32,
    /// Temperature threshold for Critical state (°C)
    pub critical_threshold: f32,
    /// Cooldown period before resuming full speed
    pub cooldown_duration: Duration,
    /// Enable adaptive workload reduction
    pub adaptive_throttling: bool,
}

impl Default for ThermalPolicy {
    fn default() -> Self {
        Self {
            fair_threshold: 35.0,
            serious_threshold: 40.0,
            critical_threshold: 45.0,
            cooldown_duration: Duration::from_secs(30),
            adaptive_throttling: true,
        }
    }
}

/// Thermal manager for monitoring and controlling device temperature
#[derive(Debug)]
pub struct ThermalManager {
    /// Current policy
    policy: ThermalPolicy,
    /// Current state
    current_state: ThermalState,
    /// Current temperature
    temperature: f32,
    /// Last state change time
    last_state_change: Instant,
    /// History of temperature readings
    history: Vec<(Instant, f32)>,
    /// Maximum history entries
    max_history: usize,
}

impl ThermalManager {
    /// Create new thermal manager
    pub fn new() -> Self {
        Self {
            policy: ThermalPolicy::default(),
            current_state: ThermalState::Nominal,
            temperature: 25.0,
            last_state_change: Instant::now(),
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create with custom policy
    pub fn with_policy(policy: ThermalPolicy) -> Self {
        Self {
            policy,
            ..Self::new()
        }
    }

    /// Update temperature reading
    pub fn update_temperature(&mut self, temp: f32) {
        self.temperature = temp;
        self.history.push((Instant::now(), temp));

        // Trim history
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Update state
        let new_state = self.compute_state(temp);
        if new_state != self.current_state {
            self.current_state = new_state;
            self.last_state_change = Instant::now();
        }
    }

    /// Compute state from temperature
    fn compute_state(&self, temp: f32) -> ThermalState {
        if temp >= self.policy.critical_threshold {
            ThermalState::Critical
        } else if temp >= self.policy.serious_threshold {
            ThermalState::Serious
        } else if temp >= self.policy.fair_threshold {
            ThermalState::Fair
        } else {
            ThermalState::Nominal
        }
    }

    /// Get current thermal state
    pub fn current_state(&self) -> ThermalState {
        // Refresh from system
        self.refresh_temperature();
        self.current_state
    }

    /// Get current temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Refresh temperature from system
    fn refresh_temperature(&self) {
        #[cfg(target_os = "ios")]
        {
            // iOS thermal state API
            // ProcessInfo.processInfo.thermalState
        }

        #[cfg(target_os = "android")]
        {
            // Android thermal API
            // PowerManager.getThermalHeadroom()
        }
    }

    /// Get temperature from system (platform-specific)
    pub fn read_system_temperature() -> Option<f32> {
        #[cfg(target_os = "ios")]
        {
            // No direct temperature API on iOS
            // Use ProcessInfo.thermalState as proxy
            None
        }

        #[cfg(target_os = "android")]
        {
            // Read from /sys/class/thermal/
            None
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            None
        }
    }

    /// Check if should throttle
    pub fn should_throttle(&self) -> bool {
        self.current_state != ThermalState::Nominal
    }

    /// Get recommended workload factor (0.0 - 1.0)
    pub fn workload_factor(&self) -> f32 {
        if !self.policy.adaptive_throttling {
            return self.current_state.throttle_factor();
        }

        // Adaptive: interpolate based on temperature
        let temp = self.temperature;

        if temp <= self.policy.fair_threshold {
            1.0
        } else if temp <= self.policy.serious_threshold {
            let range = self.policy.serious_threshold - self.policy.fair_threshold;
            let offset = temp - self.policy.fair_threshold;
            1.0 - (offset / range) * 0.35 // 1.0 -> 0.65
        } else if temp <= self.policy.critical_threshold {
            let range = self.policy.critical_threshold - self.policy.serious_threshold;
            let offset = temp - self.policy.serious_threshold;
            0.65 - (offset / range) * 0.55 // 0.65 -> 0.1
        } else {
            0.1
        }
    }

    /// Get recommended delay before next operation
    pub fn recommended_delay(&self) -> Duration {
        Duration::from_millis(self.current_state.recommended_delay_ms())
    }

    /// Check if in cooldown period
    pub fn in_cooldown(&self) -> bool {
        if self.current_state == ThermalState::Nominal {
            return false;
        }
        self.last_state_change.elapsed() < self.policy.cooldown_duration
    }

    /// Get temperature trend (positive = heating, negative = cooling)
    pub fn temperature_trend(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<&(Instant, f32)> = self.history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let newest = recent[0];
        let oldest = recent[recent.len() - 1];

        let time_diff = newest.0.duration_since(oldest.0).as_secs_f32();
        if time_diff == 0.0 {
            return 0.0;
        }

        (newest.1 - oldest.1) / time_diff
    }

    /// Get policy
    pub fn policy(&self) -> &ThermalPolicy {
        &self.policy
    }

    /// Set policy
    pub fn set_policy(&mut self, policy: ThermalPolicy) {
        self.policy = policy;
    }

    /// Get time since last state change
    pub fn time_in_state(&self) -> Duration {
        self.last_state_change.elapsed()
    }
}

impl Default for ThermalManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Thermal event for logging/monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalEvent {
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
    /// Previous state
    pub from_state: ThermalState,
    /// New state
    pub to_state: ThermalState,
    /// Temperature at event
    pub temperature: f32,
}

/// Thermal history for analysis
#[derive(Debug, Default)]
pub struct ThermalHistory {
    /// Events
    events: Vec<ThermalEvent>,
    /// Maximum events to keep
    max_events: usize,
}

impl ThermalHistory {
    /// Create new history
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            max_events,
        }
    }

    /// Record state change event
    pub fn record(&mut self, from: ThermalState, to: ThermalState, temp: f32) {
        let event = ThermalEvent {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            from_state: from,
            to_state: to,
            temperature: temp,
        };

        self.events.push(event);

        if self.events.len() > self.max_events {
            self.events.remove(0);
        }
    }

    /// Get all events
    pub fn events(&self) -> &[ThermalEvent] {
        &self.events
    }

    /// Count transitions to throttled states
    pub fn throttle_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.to_state != ThermalState::Nominal)
            .count()
    }

    /// Average time spent in non-nominal state
    pub fn avg_throttle_duration(&self) -> Duration {
        let mut total_ms = 0u64;
        let mut count = 0u64;

        for i in 0..self.events.len() {
            if self.events[i].from_state != ThermalState::Nominal && i + 1 < self.events.len() {
                total_ms += self.events[i + 1].timestamp_ms - self.events[i].timestamp_ms;
                count += 1;
            }
        }

        if count == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(total_ms / count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_state_throttle_factor() {
        assert_eq!(ThermalState::Nominal.throttle_factor(), 1.0);
        assert_eq!(ThermalState::Fair.throttle_factor(), 0.85);
        assert_eq!(ThermalState::Serious.throttle_factor(), 0.5);
        assert_eq!(ThermalState::Critical.throttle_factor(), 0.1);
    }

    #[test]
    fn test_thermal_policy_default() {
        let policy = ThermalPolicy::default();
        assert_eq!(policy.fair_threshold, 35.0);
        assert_eq!(policy.serious_threshold, 40.0);
        assert_eq!(policy.critical_threshold, 45.0);
    }

    #[test]
    fn test_thermal_manager_creation() {
        let manager = ThermalManager::new();
        assert_eq!(manager.current_state, ThermalState::Nominal);
        assert!(!manager.should_throttle());
    }

    #[test]
    fn test_temperature_update() {
        let mut manager = ThermalManager::new();

        manager.update_temperature(30.0);
        assert_eq!(manager.current_state, ThermalState::Nominal);

        manager.update_temperature(37.0);
        assert_eq!(manager.current_state, ThermalState::Fair);

        manager.update_temperature(42.0);
        assert_eq!(manager.current_state, ThermalState::Serious);

        manager.update_temperature(50.0);
        assert_eq!(manager.current_state, ThermalState::Critical);
    }

    #[test]
    fn test_workload_factor() {
        let mut manager = ThermalManager::new();

        manager.update_temperature(30.0);
        assert_eq!(manager.workload_factor(), 1.0);

        manager.update_temperature(50.0);
        assert!(manager.workload_factor() <= 0.1);
    }

    #[test]
    fn test_thermal_history() {
        let mut history = ThermalHistory::new(100);

        history.record(ThermalState::Nominal, ThermalState::Fair, 36.0);
        history.record(ThermalState::Fair, ThermalState::Nominal, 34.0);

        assert_eq!(history.events().len(), 2);
        assert_eq!(history.throttle_count(), 1);
    }
}

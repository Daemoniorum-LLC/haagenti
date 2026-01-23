//! Unified mobile runtime

use crate::{
    coreml::CoreMLRuntime,
    nnapi::NnapiRuntime,
    quantization::{Int4Quantizer, QuantizationConfig},
    thermal::{ThermalManager, ThermalState},
    MobileError, Result,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Completion handler type for async execution callbacks
pub type CompletionHandler = Box<dyn FnOnce(Result<Vec<f32>>) + Send>;

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Maximum memory usage in bytes
    pub max_memory: u64,
    /// Enable thermal management
    pub thermal_management: bool,
    /// Enable battery monitoring
    pub battery_monitoring: bool,
    /// Minimum battery level to run
    pub min_battery_level: u8,
    /// Default execution timeout
    pub timeout_ms: u64,
    /// Enable quantization
    pub use_quantization: bool,
    /// Quantization config
    pub quantization: QuantizationConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory: 512 * 1024 * 1024, // 512MB
            thermal_management: true,
            battery_monitoring: true,
            min_battery_level: 15,
            timeout_ms: 30000, // 30 seconds
            use_quantization: true,
            quantization: QuantizationConfig::default(),
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStats {
    /// Total inferences run
    pub total_inferences: u64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Min latency in ms
    pub min_latency_ms: f64,
    /// Max latency in ms
    pub max_latency_ms: f64,
    /// Thermal throttling events
    pub throttle_events: u64,
    /// Memory peak usage
    pub peak_memory: u64,
    /// Total bytes processed
    pub total_bytes: u64,
}

impl RuntimeStats {
    /// Record a new inference
    pub fn record_inference(&mut self, latency_ms: f64) {
        if self.total_inferences == 0 {
            self.min_latency_ms = latency_ms;
            self.max_latency_ms = latency_ms;
            self.avg_latency_ms = latency_ms;
        } else {
            self.min_latency_ms = self.min_latency_ms.min(latency_ms);
            self.max_latency_ms = self.max_latency_ms.max(latency_ms);
            // Running average
            self.avg_latency_ms = (self.avg_latency_ms * self.total_inferences as f64 + latency_ms)
                / (self.total_inferences + 1) as f64;
        }
        self.total_inferences += 1;
    }

    /// Record throttle event
    pub fn record_throttle(&mut self) {
        self.throttle_events += 1;
    }

    /// Update peak memory
    pub fn update_memory(&mut self, current: u64) {
        self.peak_memory = self.peak_memory.max(current);
    }
}

/// Unified mobile runtime
#[derive(Debug)]
pub struct MobileRuntime {
    /// Configuration
    config: RuntimeConfig,
    /// CoreML runtime (iOS) - platform-specific, used via cfg
    #[allow(dead_code)]
    coreml: Option<CoreMLRuntime>,
    /// NNAPI runtime (Android) - platform-specific, used via cfg
    #[allow(dead_code)]
    nnapi: Option<NnapiRuntime>,
    /// Thermal manager
    thermal: ThermalManager,
    /// Quantizer
    quantizer: Int4Quantizer,
    /// Statistics
    stats: RuntimeStats,
    /// Current memory usage
    current_memory: u64,
}

impl MobileRuntime {
    /// Create new mobile runtime
    pub fn new(config: RuntimeConfig) -> Self {
        let quantizer = Int4Quantizer::new(config.quantization.clone());
        let thermal = ThermalManager::new();

        #[cfg(target_os = "ios")]
        let coreml = Some(CoreMLRuntime::new());
        #[cfg(not(target_os = "ios"))]
        let coreml = None;

        #[cfg(target_os = "android")]
        let nnapi = Some(NnapiRuntime::new());
        #[cfg(not(target_os = "android"))]
        let nnapi = None;

        Self {
            config,
            coreml,
            nnapi,
            thermal,
            quantizer,
            stats: RuntimeStats::default(),
            current_memory: 0,
        }
    }

    /// Initialize runtime
    pub async fn initialize(&mut self) -> Result<()> {
        // Check thermal state
        if self.config.thermal_management {
            let state = self.thermal.current_state();
            if state == ThermalState::Critical {
                return Err(MobileError::ThermalThrottling {
                    state: "Critical".into(),
                    temp_celsius: self.thermal.temperature(),
                });
            }
        }

        // Check battery
        if self.config.battery_monitoring {
            let level = self.battery_level();
            if level < self.config.min_battery_level {
                return Err(MobileError::BatteryLow { level });
            }
        }

        Ok(())
    }

    /// Get battery level (0-100)
    fn battery_level(&self) -> u8 {
        #[cfg(target_os = "ios")]
        {
            // UIDevice.current.batteryLevel
            100
        }

        #[cfg(target_os = "android")]
        {
            // BatteryManager
            100
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            100
        }
    }

    /// Run inference on the appropriate backend
    pub async fn infer(&mut self, model_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        // Check thermal state
        if self.config.thermal_management {
            self.check_thermal_state()?;
        }

        let start = Instant::now();

        let result = self.infer_internal(model_name, input).await;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.record_inference(latency_ms);
        self.stats.total_bytes += (input.len() * 4) as u64;

        result
    }

    async fn infer_internal(&self, _model_name: &str, input: &[f32]) -> Result<Vec<f32>> {
        #[cfg(target_os = "ios")]
        {
            if let Some(ref coreml) = self.coreml {
                if let Some(model) = coreml.get(model_name) {
                    return model.predict(input).await;
                }
            }
        }

        #[cfg(target_os = "android")]
        {
            if let Some(ref nnapi) = self.nnapi {
                if let Some(model) = nnapi.get(model_name) {
                    return model.predict(input).await;
                }
            }
        }

        // Fallback: simulate inference
        Ok(vec![0.0; input.len()])
    }

    /// Check thermal state and throttle if needed
    fn check_thermal_state(&mut self) -> Result<()> {
        let state = self.thermal.current_state();

        match state {
            ThermalState::Nominal => Ok(()),
            ThermalState::Fair => {
                // Log warning but continue
                Ok(())
            }
            ThermalState::Serious => {
                self.stats.record_throttle();
                // Could add delay here
                Ok(())
            }
            ThermalState::Critical => {
                self.stats.record_throttle();
                Err(MobileError::ThermalThrottling {
                    state: "Critical".into(),
                    temp_celsius: self.thermal.temperature(),
                })
            }
        }
    }

    /// Allocate memory
    pub fn allocate(&mut self, size: u64) -> Result<()> {
        if self.current_memory + size > self.config.max_memory {
            return Err(MobileError::OutOfMemory {
                requested_mb: size / (1024 * 1024),
                available_mb: (self.config.max_memory - self.current_memory) / (1024 * 1024),
            });
        }

        self.current_memory += size;
        self.stats.update_memory(self.current_memory);
        Ok(())
    }

    /// Release memory
    pub fn release(&mut self, size: u64) {
        self.current_memory = self.current_memory.saturating_sub(size);
    }

    /// Get runtime statistics
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get thermal manager
    pub fn thermal(&self) -> &ThermalManager {
        &self.thermal
    }

    /// Get quantizer
    pub fn quantizer(&self) -> &Int4Quantizer {
        &self.quantizer
    }

    /// Current memory usage
    pub fn memory_usage(&self) -> u64 {
        self.current_memory
    }

    /// Available memory
    pub fn available_memory(&self) -> u64 {
        self.config.max_memory.saturating_sub(self.current_memory)
    }
}

/// Execution context for a single inference
pub struct ExecutionContext {
    /// Model name
    pub model_name: String,
    /// Timeout
    pub timeout: Duration,
    /// Priority (higher = more urgent)
    pub priority: u32,
    /// Callback on completion
    completion_handler: Option<CompletionHandler>,
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("model_name", &self.model_name)
            .field("timeout", &self.timeout)
            .field("priority", &self.priority)
            .field(
                "completion_handler",
                &self.completion_handler.as_ref().map(|_| "..."),
            )
            .finish()
    }
}

impl ExecutionContext {
    /// Create new execution context
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            timeout: Duration::from_secs(30),
            priority: 0,
            completion_handler: None,
        }
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set completion handler
    pub fn on_complete<F>(mut self, handler: F) -> Self
    where
        F: FnOnce(Result<Vec<f32>>) + Send + 'static,
    {
        self.completion_handler = Some(Box::new(handler));
        self
    }
}

/// Batch inference for multiple inputs
#[derive(Debug)]
pub struct BatchContext {
    /// Model name
    pub model_name: String,
    /// Inputs
    pub inputs: Vec<Vec<f32>>,
    /// Maximum batch size
    pub max_batch_size: usize,
}

impl BatchContext {
    /// Create new batch context
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            inputs: Vec::new(),
            max_batch_size: 4,
        }
    }

    /// Add input
    pub fn add(&mut self, input: Vec<f32>) {
        self.inputs.push(input);
    }

    /// Set max batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Get batch count
    pub fn batch_count(&self) -> usize {
        self.inputs.len().div_ceil(self.max_batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_memory, 512 * 1024 * 1024);
        assert!(config.thermal_management);
        assert!(config.use_quantization);
    }

    #[test]
    fn test_runtime_stats() {
        let mut stats = RuntimeStats::default();

        stats.record_inference(10.0);
        assert_eq!(stats.total_inferences, 1);
        assert_eq!(stats.avg_latency_ms, 10.0);

        stats.record_inference(20.0);
        assert_eq!(stats.total_inferences, 2);
        assert_eq!(stats.avg_latency_ms, 15.0);
        assert_eq!(stats.min_latency_ms, 10.0);
        assert_eq!(stats.max_latency_ms, 20.0);
    }

    #[test]
    fn test_runtime_creation() {
        let config = RuntimeConfig::default();
        let runtime = MobileRuntime::new(config);

        assert_eq!(runtime.memory_usage(), 0);
        assert!(runtime.available_memory() > 0);
    }

    #[test]
    fn test_memory_allocation() {
        let config = RuntimeConfig {
            max_memory: 1024 * 1024, // 1MB
            ..Default::default()
        };
        let mut runtime = MobileRuntime::new(config);

        // Allocate 512KB
        runtime.allocate(512 * 1024).unwrap();
        assert_eq!(runtime.memory_usage(), 512 * 1024);

        // Allocate another 256KB
        runtime.allocate(256 * 1024).unwrap();
        assert_eq!(runtime.memory_usage(), 768 * 1024);

        // Try to allocate more than available
        let result = runtime.allocate(512 * 1024);
        assert!(result.is_err());

        // Release memory
        runtime.release(256 * 1024);
        assert_eq!(runtime.memory_usage(), 512 * 1024);
    }

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new("model")
            .with_timeout(Duration::from_secs(10))
            .with_priority(5);

        assert_eq!(ctx.model_name, "model");
        assert_eq!(ctx.timeout, Duration::from_secs(10));
        assert_eq!(ctx.priority, 5);
    }

    #[test]
    fn test_batch_context() {
        let mut ctx = BatchContext::new("model").with_max_batch_size(4);

        ctx.add(vec![1.0, 2.0]);
        ctx.add(vec![3.0, 4.0]);
        ctx.add(vec![5.0, 6.0]);
        ctx.add(vec![7.0, 8.0]);
        ctx.add(vec![9.0, 10.0]);

        assert_eq!(ctx.inputs.len(), 5);
        assert_eq!(ctx.batch_count(), 2);
    }
}

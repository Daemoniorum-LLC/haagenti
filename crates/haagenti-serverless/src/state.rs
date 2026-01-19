//! Function state management for serverless hibernation

use crate::{Result, ServerlessError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Function state for hibernation/resume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionState {
    /// State version
    pub version: u32,
    /// Function name
    pub function_name: String,
    /// Created timestamp (unix ms)
    pub created_at: u64,
    /// Last modified timestamp (unix ms)
    pub modified_at: u64,
    /// Model state
    pub model_state: ModelState,
    /// Cache state
    pub cache_state: CacheState,
    /// Execution state
    pub execution_state: ExecutionState,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Model state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelState {
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Loaded layers
    pub loaded_layers: Vec<String>,
    /// Weights hash
    pub weights_hash: String,
    /// Quantization info
    pub quantization: Option<QuantizationInfo>,
}

/// Quantization info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization type
    pub qtype: String,
    /// Bits per weight
    pub bits: u8,
    /// Group size
    pub group_size: usize,
}

/// Cache state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheState {
    /// KV cache size
    pub kv_cache_size: u64,
    /// KV cache entries
    pub kv_entries: usize,
    /// Fragment cache size
    pub fragment_cache_size: u64,
    /// Fragment entries
    pub fragment_entries: usize,
    /// Cache hit rate
    pub hit_rate: f64,
}

/// Execution state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average latency ms
    pub avg_latency_ms: f64,
    /// Last request timestamp
    pub last_request_at: Option<u64>,
}

impl FunctionState {
    /// Create new function state
    pub fn new(function_name: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            version: 1,
            function_name: function_name.into(),
            created_at: now,
            modified_at: now,
            model_state: ModelState::default(),
            cache_state: CacheState::default(),
            execution_state: ExecutionState::default(),
            metadata: HashMap::new(),
        }
    }

    /// Update modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
        self.touch();
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Record successful request
    pub fn record_success(&mut self, latency_ms: f64) {
        self.execution_state.total_requests += 1;
        self.execution_state.successful_requests += 1;

        let count = self.execution_state.successful_requests as f64;
        self.execution_state.avg_latency_ms =
            (self.execution_state.avg_latency_ms * (count - 1.0) + latency_ms) / count;

        self.execution_state.last_request_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        self.touch();
    }

    /// Record failed request
    pub fn record_failure(&mut self) {
        self.execution_state.total_requests += 1;
        self.execution_state.failed_requests += 1;
        self.touch();
    }

    /// Success rate
    pub fn success_rate(&self) -> f64 {
        if self.execution_state.total_requests == 0 {
            1.0
        } else {
            self.execution_state.successful_requests as f64
                / self.execution_state.total_requests as f64
        }
    }
}

/// State serializer
#[derive(Debug)]
pub struct StateSerializer {
    /// Use compression
    compression: bool,
    /// Compression level
    compression_level: i32,
}

impl StateSerializer {
    /// Create new serializer
    pub fn new(compression: bool) -> Self {
        Self {
            compression,
            compression_level: 3,
        }
    }

    /// Serialize state to bytes
    pub fn serialize(&self, state: &FunctionState) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(state)
            .map_err(|e| ServerlessError::SerializationError(e.to_string()))?;

        if self.compression {
            // In real implementation, use zstd compression
            Ok(json)
        } else {
            Ok(json)
        }
    }

    /// Deserialize state from bytes
    pub fn deserialize(&self, data: &[u8]) -> Result<FunctionState> {
        let json_data = if self.compression {
            // In real implementation, decompress with zstd
            data.to_vec()
        } else {
            data.to_vec()
        };

        serde_json::from_slice(&json_data)
            .map_err(|e| ServerlessError::DeserializationError(e.to_string()))
    }

    /// Set compression level
    pub fn with_compression_level(mut self, level: i32) -> Self {
        self.compression_level = level;
        self
    }
}

impl Default for StateSerializer {
    fn default() -> Self {
        Self::new(true)
    }
}

/// State manager for persisting function state
#[derive(Debug)]
pub struct StateManager {
    /// Current state
    state: FunctionState,
    /// Serializer
    serializer: StateSerializer,
    /// Auto-save interval (seconds)
    auto_save_interval: u64,
    /// Last save time
    last_save: Instant,
    /// State changed since last save
    dirty: bool,
}

impl StateManager {
    /// Create new state manager
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            state: FunctionState::new(function_name),
            serializer: StateSerializer::default(),
            auto_save_interval: 60,
            last_save: Instant::now(),
            dirty: false,
        }
    }

    /// Load state from bytes
    pub fn load(data: &[u8]) -> Result<Self> {
        let serializer = StateSerializer::default();
        let state = serializer.deserialize(data)?;

        Ok(Self {
            state,
            serializer,
            auto_save_interval: 60,
            last_save: Instant::now(),
            dirty: false,
        })
    }

    /// Get current state
    pub fn state(&self) -> &FunctionState {
        &self.state
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> &mut FunctionState {
        self.dirty = true;
        &mut self.state
    }

    /// Save state to bytes
    pub fn save(&mut self) -> Result<Vec<u8>> {
        let data = self.serializer.serialize(&self.state)?;
        self.last_save = Instant::now();
        self.dirty = false;
        Ok(data)
    }

    /// Check if should auto-save
    pub fn should_auto_save(&self) -> bool {
        self.dirty && self.last_save.elapsed().as_secs() >= self.auto_save_interval
    }

    /// Set auto-save interval
    pub fn set_auto_save_interval(&mut self, seconds: u64) {
        self.auto_save_interval = seconds;
    }

    /// Update model state
    pub fn update_model(&mut self, model_state: ModelState) {
        self.state.model_state = model_state;
        self.state.touch();
        self.dirty = true;
    }

    /// Update cache state
    pub fn update_cache(&mut self, cache_state: CacheState) {
        self.state.cache_state = cache_state;
        self.state.touch();
        self.dirty = true;
    }

    /// Record request
    pub fn record_request(&mut self, success: bool, latency_ms: Option<f64>) {
        if success {
            self.state.record_success(latency_ms.unwrap_or(0.0));
        } else {
            self.state.record_failure();
        }
        self.dirty = true;
    }
}

/// State diff for incremental updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    /// Changed fields
    pub changes: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: u64,
}

impl StateDiff {
    /// Create new diff
    pub fn new() -> Self {
        Self {
            changes: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Add change
    pub fn add(&mut self, field: impl Into<String>, value: serde_json::Value) {
        self.changes.insert(field.into(), value);
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

impl Default for StateDiff {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_state_creation() {
        let state = FunctionState::new("test-function");

        assert_eq!(state.function_name, "test-function");
        assert_eq!(state.version, 1);
        assert!(state.created_at > 0);
    }

    #[test]
    fn test_record_requests() {
        let mut state = FunctionState::new("test");

        state.record_success(100.0);
        state.record_success(200.0);
        state.record_failure();

        assert_eq!(state.execution_state.total_requests, 3);
        assert_eq!(state.execution_state.successful_requests, 2);
        assert_eq!(state.execution_state.failed_requests, 1);
        assert_eq!(state.execution_state.avg_latency_ms, 150.0);
    }

    #[test]
    fn test_success_rate() {
        let mut state = FunctionState::new("test");

        state.record_success(100.0);
        state.record_success(100.0);
        state.record_failure();

        assert!((state.success_rate() - 0.666666).abs() < 0.01);
    }

    #[test]
    fn test_serialization() {
        let state = FunctionState::new("test");
        let serializer = StateSerializer::new(false);

        let data = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&data).unwrap();

        assert_eq!(state.function_name, restored.function_name);
    }

    #[test]
    fn test_state_manager() {
        let mut manager = StateManager::new("test");

        manager.record_request(true, Some(50.0));
        manager.record_request(true, Some(100.0));

        let state = manager.state();
        assert_eq!(state.execution_state.successful_requests, 2);
    }

    #[test]
    fn test_state_diff() {
        let mut diff = StateDiff::new();

        diff.add("field1", serde_json::json!(42));
        diff.add("field2", serde_json::json!("value"));

        assert!(!diff.is_empty());
        assert_eq!(diff.changes.len(), 2);
    }
}

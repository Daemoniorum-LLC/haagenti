//! Hardware-aware optimization

use crate::{OptError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hardware profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: DeviceType,
    /// Memory capacity (bytes)
    pub memory_capacity: u64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,
    /// Compute capability (TFLOPS)
    pub compute_tflops: f32,
    /// Number of cores/SMs
    pub num_cores: u32,
    /// Clock speed (MHz)
    pub clock_mhz: u32,
    /// Capabilities
    pub capabilities: Vec<DeviceCapability>,
}

/// Device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Custom,
}

/// Device capability
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceCapability {
    /// FP16 support
    Fp16,
    /// BF16 support
    Bf16,
    /// INT8 support
    Int8,
    /// INT4 support
    Int4,
    /// Tensor cores
    TensorCores,
    /// Sparse operations
    Sparse,
    /// Flash attention
    FlashAttention,
    /// Fused operations
    FusedOps,
}

impl HardwareProfile {
    /// Detect current hardware
    pub fn detect() -> Self {
        // Simplified detection - in practice would use system APIs
        Self {
            name: "Generic CPU".into(),
            device_type: DeviceType::Cpu,
            memory_capacity: 16 * 1024 * 1024 * 1024, // 16GB
            memory_bandwidth: 50.0,
            compute_tflops: 0.5,
            num_cores: num_cpus(),
            clock_mhz: 3000,
            capabilities: vec![DeviceCapability::Fp16],
        }
    }

    /// Create NVIDIA GPU profile
    pub fn nvidia_gpu(name: &str, memory_gb: u64, tflops: f32) -> Self {
        Self {
            name: name.into(),
            device_type: DeviceType::Gpu,
            memory_capacity: memory_gb * 1024 * 1024 * 1024,
            memory_bandwidth: 900.0,
            compute_tflops: tflops,
            num_cores: 80, // Typical SM count
            clock_mhz: 1700,
            capabilities: vec![
                DeviceCapability::Fp16,
                DeviceCapability::Bf16,
                DeviceCapability::Int8,
                DeviceCapability::TensorCores,
                DeviceCapability::FlashAttention,
            ],
        }
    }

    /// Has capability
    pub fn has_capability(&self, cap: &DeviceCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Memory capacity in GB
    pub fn memory_gb(&self) -> f32 {
        self.memory_capacity as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Estimated tokens per second for given model size
    pub fn estimate_tps(&self, model_params_b: f32) -> f32 {
        // Simplified: memory-bound estimation
        let bytes_per_param = if self.has_capability(&DeviceCapability::Int4) {
            0.5
        } else if self.has_capability(&DeviceCapability::Int8) {
            1.0
        } else {
            2.0 // FP16
        };

        let model_size_gb = model_params_b * bytes_per_param;
        let memory_time_per_token = model_size_gb / self.memory_bandwidth;

        1.0 / memory_time_per_token
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4)
}

/// Hardware optimizer
#[derive(Debug)]
pub struct HardwareOptimizer {
    /// Hardware profile
    profile: HardwareProfile,
    /// Optimization settings
    settings: OptSettings,
}

/// Optimization settings for hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptSettings {
    /// Target latency (ms)
    pub target_latency_ms: Option<f32>,
    /// Target throughput (tokens/sec)
    pub target_throughput: Option<f32>,
    /// Maximum memory usage ratio
    pub max_memory_ratio: f32,
    /// Prefer lower precision
    pub prefer_lower_precision: bool,
    /// Enable fusion
    pub enable_fusion: bool,
}

impl Default for OptSettings {
    fn default() -> Self {
        Self {
            target_latency_ms: None,
            target_throughput: None,
            max_memory_ratio: 0.9,
            prefer_lower_precision: true,
            enable_fusion: true,
        }
    }
}

/// Recommended configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedConfig {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub max_seq_len: usize,
    /// Precision
    pub precision: String,
    /// Enable flash attention
    pub flash_attention: bool,
    /// Enable KV cache quantization
    pub kv_cache_quant: bool,
    /// Number of layers to pipeline
    pub pipeline_layers: usize,
    /// Estimated memory usage (GB)
    pub estimated_memory_gb: f32,
    /// Estimated throughput (tokens/sec)
    pub estimated_tps: f32,
}

impl HardwareOptimizer {
    /// Create new hardware optimizer
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            profile,
            settings: OptSettings::default(),
        }
    }

    /// Set optimization settings
    pub fn with_settings(mut self, settings: OptSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Recommend configuration for model
    pub fn recommend(&self, model_params_b: f32, context_len: usize) -> RecommendedConfig {
        let memory_gb = self.profile.memory_gb();
        let available = memory_gb * self.settings.max_memory_ratio;

        // Determine precision
        let precision = if self.settings.prefer_lower_precision {
            if self.profile.has_capability(&DeviceCapability::Int4) {
                "int4"
            } else if self.profile.has_capability(&DeviceCapability::Int8) {
                "int8"
            } else if self.profile.has_capability(&DeviceCapability::Fp16) {
                "fp16"
            } else {
                "fp32"
            }
        } else {
            "fp16"
        };

        let bytes_per_param = match precision {
            "int4" => 0.5,
            "int8" => 1.0,
            "fp16" => 2.0,
            _ => 4.0,
        };

        let model_size_gb = model_params_b * bytes_per_param;

        // Calculate max batch size
        let remaining = available - model_size_gb;
        let kv_bytes_per_token = model_params_b * 0.1 * bytes_per_param; // Approximate
        let max_tokens = (remaining / kv_bytes_per_token * 1024.0 * 1024.0 * 1024.0) as usize;

        let batch_size = (max_tokens / context_len).max(1).min(32);

        let flash_attention = self.profile.has_capability(&DeviceCapability::FlashAttention);

        RecommendedConfig {
            batch_size,
            max_seq_len: context_len,
            precision: precision.into(),
            flash_attention,
            kv_cache_quant: self.profile.has_capability(&DeviceCapability::Int8),
            pipeline_layers: 0,
            estimated_memory_gb: model_size_gb + (batch_size * context_len) as f32 * kv_bytes_per_token / (1024.0 * 1024.0 * 1024.0),
            estimated_tps: self.profile.estimate_tps(model_params_b) * batch_size as f32,
        }
    }

    /// Optimal batch size for given memory budget
    pub fn optimal_batch_size(&self, model_size_gb: f32, seq_len: usize) -> usize {
        let available = self.profile.memory_gb() * self.settings.max_memory_ratio - model_size_gb;
        let per_batch_gb = 0.1 * model_size_gb; // Approximate activation memory per batch

        (available / per_batch_gb) as usize
    }

    /// Get hardware profile
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }
}

/// Profile database
#[derive(Debug, Default)]
pub struct ProfileDatabase {
    profiles: HashMap<String, HardwareProfile>,
}

impl ProfileDatabase {
    /// Create new database
    pub fn new() -> Self {
        Self::default()
    }

    /// Load common profiles
    pub fn with_common_profiles() -> Self {
        let mut db = Self::new();

        db.add(HardwareProfile::nvidia_gpu("A100-80GB", 80, 312.0));
        db.add(HardwareProfile::nvidia_gpu("A100-40GB", 40, 312.0));
        db.add(HardwareProfile::nvidia_gpu("H100", 80, 990.0));
        db.add(HardwareProfile::nvidia_gpu("RTX-4090", 24, 165.0));
        db.add(HardwareProfile::nvidia_gpu("RTX-3090", 24, 142.0));

        db
    }

    /// Add profile
    pub fn add(&mut self, profile: HardwareProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Get profile by name
    pub fn get(&self, name: &str) -> Option<&HardwareProfile> {
        self.profiles.get(name)
    }

    /// List all profiles
    pub fn list(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile() {
        let profile = HardwareProfile::nvidia_gpu("A100", 80, 312.0);

        assert_eq!(profile.memory_gb(), 80.0);
        assert!(profile.has_capability(&DeviceCapability::TensorCores));
        assert!(!profile.has_capability(&DeviceCapability::Int4));
    }

    #[test]
    fn test_tps_estimation() {
        let profile = HardwareProfile::nvidia_gpu("A100", 80, 312.0);

        // 7B model
        let tps = profile.estimate_tps(7.0);
        assert!(tps > 0.0);
    }

    #[test]
    fn test_hardware_optimizer() {
        let profile = HardwareProfile::nvidia_gpu("A100", 80, 312.0);
        let optimizer = HardwareOptimizer::new(profile);

        let config = optimizer.recommend(7.0, 4096);

        assert!(config.batch_size >= 1);
        assert!(config.estimated_memory_gb < 80.0);
    }

    #[test]
    fn test_profile_database() {
        let db = ProfileDatabase::with_common_profiles();

        assert!(db.get("A100-80GB").is_some());
        assert!(db.get("H100").is_some());
        assert!(db.list().len() >= 5);
    }
}

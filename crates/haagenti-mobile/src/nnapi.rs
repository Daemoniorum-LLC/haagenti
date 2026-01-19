//! NNAPI integration for Android

use crate::{MobileError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// NNAPI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnapiConfig {
    /// Model path
    pub model_path: PathBuf,
    /// Prefer GPU accelerator
    pub prefer_gpu: bool,
    /// Prefer NPU/DSP accelerator
    pub prefer_npu: bool,
    /// Allow FP16 computation
    pub allow_fp16: bool,
    /// Execution preference
    pub execution_preference: ExecutionPreference,
    /// Deadline in nanoseconds (0 = no deadline)
    pub deadline_ns: u64,
}

impl Default for NnapiConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prefer_gpu: true,
            prefer_npu: true,
            allow_fp16: true,
            execution_preference: ExecutionPreference::SustainedSpeed,
            deadline_ns: 0,
        }
    }
}

/// NNAPI execution preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPreference {
    /// Optimize for low power consumption
    LowPower,
    /// Optimize for fast single execution
    FastSingleAnswer,
    /// Optimize for sustained speed
    SustainedSpeed,
}

impl ExecutionPreference {
    /// Convert to NNAPI constant
    pub fn to_nnapi(&self) -> i32 {
        match self {
            ExecutionPreference::LowPower => 0,
            ExecutionPreference::FastSingleAnswer => 1,
            ExecutionPreference::SustainedSpeed => 2,
        }
    }
}

/// NNAPI device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Unknown device
    Unknown,
    /// CPU
    Cpu,
    /// GPU
    Gpu,
    /// DSP/NPU accelerator
    Accelerator,
    /// Other device
    Other,
}

impl DeviceType {
    /// From NNAPI device type constant
    pub fn from_nnapi(value: i32) -> Self {
        match value {
            1 => DeviceType::Cpu,
            2 => DeviceType::Gpu,
            3 => DeviceType::Accelerator,
            4 => DeviceType::Other,
            _ => DeviceType::Unknown,
        }
    }
}

/// NNAPI device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnapiDevice {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: DeviceType,
    /// Feature level (Android API level)
    pub feature_level: i32,
    /// Device version
    pub version: String,
}

/// NNAPI model wrapper
#[derive(Debug)]
pub struct NnapiModel {
    /// Configuration
    config: NnapiConfig,
    /// Model name
    name: String,
    /// Input shape
    input_shape: Vec<usize>,
    /// Output shape
    output_shape: Vec<usize>,
    /// Whether model is loaded
    loaded: bool,
    /// Selected devices
    devices: Vec<NnapiDevice>,
}

impl NnapiModel {
    /// Create new NNAPI model
    pub fn new(name: impl Into<String>, config: NnapiConfig) -> Self {
        Self {
            config,
            name: name.into(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            loaded: false,
            devices: Vec::new(),
        }
    }

    /// Load model
    pub async fn load(&mut self) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            self.load_android().await?;
        }

        #[cfg(not(target_os = "android"))]
        {
            // Simulate loading for non-Android
            self.input_shape = vec![1, 512];
            self.output_shape = vec![1, 512];
        }

        self.loaded = true;
        Ok(())
    }

    #[cfg(target_os = "android")]
    async fn load_android(&mut self) -> Result<()> {
        // In actual implementation:
        // 1. Create ANeuralNetworksModel
        // 2. Add operations and operands
        // 3. Compile for selected devices
        unimplemented!("NNAPI loading requires Android NDK")
    }

    /// Run inference
    pub async fn predict(&self, input: &[f32]) -> Result<Vec<f32>> {
        if !self.loaded {
            return Err(MobileError::ModelLoadError("Model not loaded".into()));
        }

        #[cfg(target_os = "android")]
        {
            self.predict_android(input).await
        }

        #[cfg(not(target_os = "android"))]
        {
            // Simulate prediction
            Ok(vec![0.0; self.output_shape.iter().product()])
        }
    }

    #[cfg(target_os = "android")]
    async fn predict_android(&self, input: &[f32]) -> Result<Vec<f32>> {
        unimplemented!("NNAPI prediction requires Android NDK")
    }

    /// Get model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get selected devices
    pub fn devices(&self) -> &[NnapiDevice] {
        &self.devices
    }
}

/// NNAPI runtime for managing models
#[derive(Debug)]
pub struct NnapiRuntime {
    /// Models by name
    models: std::collections::HashMap<String, NnapiModel>,
    /// Available devices
    available_devices: Vec<NnapiDevice>,
    /// NNAPI version
    nnapi_version: i32,
}

impl NnapiRuntime {
    /// Create new runtime
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
            available_devices: Self::enumerate_devices(),
            nnapi_version: Self::get_nnapi_version(),
        }
    }

    /// Enumerate available devices
    fn enumerate_devices() -> Vec<NnapiDevice> {
        #[cfg(target_os = "android")]
        {
            // Query ANeuralNetworks_getDeviceCount and iterate
            Vec::new()
        }

        #[cfg(not(target_os = "android"))]
        {
            // Return simulated devices
            vec![
                NnapiDevice {
                    name: "CPU".into(),
                    device_type: DeviceType::Cpu,
                    feature_level: 30,
                    version: "1.0".into(),
                },
            ]
        }
    }

    /// Get NNAPI version
    fn get_nnapi_version() -> i32 {
        #[cfg(target_os = "android")]
        {
            // Check runtime feature level
            30 // Android 11
        }

        #[cfg(not(target_os = "android"))]
        {
            0
        }
    }

    /// Register a model
    pub fn register(&mut self, model: NnapiModel) {
        let name = model.name().to_string();
        self.models.insert(name, model);
    }

    /// Get model by name
    pub fn get(&self, name: &str) -> Option<&NnapiModel> {
        self.models.get(name)
    }

    /// Get mutable model by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NnapiModel> {
        self.models.get_mut(name)
    }

    /// Load all registered models
    pub async fn load_all(&mut self) -> Result<()> {
        for model in self.models.values_mut() {
            model.load().await?;
        }
        Ok(())
    }

    /// Get available devices
    pub fn devices(&self) -> &[NnapiDevice] {
        &self.available_devices
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.available_devices
            .iter()
            .any(|d| d.device_type == DeviceType::Gpu)
    }

    /// Check if NPU/accelerator is available
    pub fn has_accelerator(&self) -> bool {
        self.available_devices
            .iter()
            .any(|d| d.device_type == DeviceType::Accelerator)
    }

    /// Get NNAPI version
    pub fn version(&self) -> i32 {
        self.nnapi_version
    }

    /// Check minimum NNAPI version
    pub fn requires_version(&self, min_version: i32) -> bool {
        self.nnapi_version >= min_version
    }
}

impl Default for NnapiRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation compatibility check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSupport {
    /// Operation name
    pub name: String,
    /// Supported on CPU
    pub cpu: bool,
    /// Supported on GPU
    pub gpu: bool,
    /// Supported on NPU
    pub npu: bool,
    /// Minimum NNAPI version
    pub min_version: i32,
}

/// Common NNAPI operations
pub mod operations {
    use super::OperationSupport;

    /// Matrix multiplication support
    pub fn matmul() -> OperationSupport {
        OperationSupport {
            name: "BATCH_MATMUL".into(),
            cpu: true,
            gpu: true,
            npu: true,
            min_version: 29,
        }
    }

    /// Softmax support
    pub fn softmax() -> OperationSupport {
        OperationSupport {
            name: "SOFTMAX".into(),
            cpu: true,
            gpu: true,
            npu: true,
            min_version: 27,
        }
    }

    /// Layer normalization support
    pub fn layer_norm() -> OperationSupport {
        OperationSupport {
            name: "LAYER_NORMALIZATION".into(),
            cpu: true,
            gpu: true,
            npu: false, // Not all NPUs support
            min_version: 30,
        }
    }

    /// GELU activation support
    pub fn gelu() -> OperationSupport {
        OperationSupport {
            name: "GELU".into(),
            cpu: true,
            gpu: true,
            npu: false,
            min_version: 31,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NnapiConfig::default();
        assert!(config.prefer_gpu);
        assert!(config.prefer_npu);
        assert!(config.allow_fp16);
    }

    #[test]
    fn test_execution_preference() {
        assert_eq!(ExecutionPreference::LowPower.to_nnapi(), 0);
        assert_eq!(ExecutionPreference::FastSingleAnswer.to_nnapi(), 1);
        assert_eq!(ExecutionPreference::SustainedSpeed.to_nnapi(), 2);
    }

    #[test]
    fn test_device_type() {
        assert_eq!(DeviceType::from_nnapi(1), DeviceType::Cpu);
        assert_eq!(DeviceType::from_nnapi(2), DeviceType::Gpu);
        assert_eq!(DeviceType::from_nnapi(3), DeviceType::Accelerator);
    }

    #[test]
    fn test_model_creation() {
        let config = NnapiConfig::default();
        let model = NnapiModel::new("test_model", config);

        assert_eq!(model.name(), "test_model");
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_runtime_creation() {
        let runtime = NnapiRuntime::new();

        // On non-Android, we simulate a CPU device
        #[cfg(not(target_os = "android"))]
        {
            assert!(!runtime.devices().is_empty());
        }
    }

    #[test]
    fn test_operation_support() {
        let matmul = operations::matmul();
        assert!(matmul.cpu);
        assert!(matmul.gpu);
        assert_eq!(matmul.min_version, 29);
    }
}

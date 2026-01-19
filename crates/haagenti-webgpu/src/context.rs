//! WebGPU context and device management

use crate::{Result, WebGpuError};
use serde::{Deserialize, Serialize};

/// Configuration for WebGPU context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Prefer high-performance GPU
    pub high_performance: bool,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Enable shader debugging
    pub debug_shaders: bool,
    /// Timeout for device acquisition in ms
    pub device_timeout_ms: u64,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            high_performance: true,
            max_buffer_size: 256 * 1024 * 1024, // 256MB
            debug_shaders: false,
            device_timeout_ms: 5000,
        }
    }
}

/// Device capabilities and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Maximum buffer size
    pub max_buffer_size: u64,
    /// Maximum compute workgroup size X
    pub max_workgroup_size_x: u32,
    /// Maximum compute workgroup size Y
    pub max_workgroup_size_y: u32,
    /// Maximum compute workgroup size Z
    pub max_workgroup_size_z: u32,
    /// Maximum workgroups per dimension
    pub max_workgroups_per_dimension: u32,
    /// Maximum bind groups
    pub max_bind_groups: u32,
    /// Maximum storage buffers per shader stage
    pub max_storage_buffers: u32,
    /// Supports timestamp queries
    pub timestamp_queries: bool,
    /// Adapter description
    pub adapter_info: String,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024 * 1024,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_workgroup_size_z: 64,
            max_workgroups_per_dimension: 65535,
            max_bind_groups: 4,
            max_storage_buffers: 8,
            timestamp_queries: false,
            adapter_info: "Unknown".into(),
        }
    }
}

/// WebGPU context for compute operations
#[derive(Debug)]
pub struct WebGpuContext {
    config: ContextConfig,
    capabilities: DeviceCapabilities,
    initialized: bool,
    // In WASM, these would hold actual WebGPU handles
    // For non-WASM, we simulate the interface
}

impl WebGpuContext {
    /// Create a new context (without initialization)
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            capabilities: DeviceCapabilities::default(),
            initialized: false,
        }
    }

    /// Initialize WebGPU (async in browser)
    pub async fn initialize(&mut self) -> Result<()> {
        // In actual WASM implementation:
        // 1. Get navigator.gpu
        // 2. Request adapter with power preference
        // 3. Request device with required limits
        // 4. Query capabilities

        #[cfg(target_arch = "wasm32")]
        {
            self.initialize_wasm().await?;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Simulate initialization for non-WASM
            self.capabilities = DeviceCapabilities {
                adapter_info: "Simulated WebGPU".into(),
                ..Default::default()
            };
        }

        self.initialized = true;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    async fn initialize_wasm(&mut self) -> Result<()> {
        use wasm_bindgen::JsCast;
        use web_sys::window;

        let window = window().ok_or_else(|| {
            WebGpuError::NotAvailable("No window object".into())
        })?;

        let navigator = window.navigator();

        // Check for GPU support
        let gpu = navigator.gpu();

        // Request adapter
        let adapter_options = web_sys::GpuRequestAdapterOptions::new();
        if self.config.high_performance {
            adapter_options.set_power_preference(
                web_sys::GpuPowerPreference::HighPerformance
            );
        }

        let adapter_promise = gpu.request_adapter_with_options(&adapter_options);
        let adapter = wasm_bindgen_futures::JsFuture::from(adapter_promise)
            .await
            .map_err(|e| WebGpuError::AdapterError(format!("{:?}", e)))?
            .dyn_into::<web_sys::GpuAdapter>()
            .map_err(|_| WebGpuError::AdapterError("Invalid adapter".into()))?;

        // Get adapter info
        let info = adapter.info();
        self.capabilities.adapter_info = format!(
            "{} - {} ({})",
            info.vendor(),
            info.description(),
            info.architecture()
        );

        // Request device
        let device_descriptor = web_sys::GpuDeviceDescriptor::new();
        let device_promise = adapter.request_device_with_descriptor(&device_descriptor);
        let _device = wasm_bindgen_futures::JsFuture::from(device_promise)
            .await
            .map_err(|e| WebGpuError::DeviceError(format!("{:?}", e)))?
            .dyn_into::<web_sys::GpuDevice>()
            .map_err(|_| WebGpuError::DeviceError("Invalid device".into()))?;

        Ok(())
    }

    /// Check if context is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get device capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Get configuration
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Check if a buffer size is supported
    pub fn supports_buffer_size(&self, size: u64) -> bool {
        size <= self.capabilities.max_buffer_size
    }

    /// Compute optimal workgroup size for a given total size
    pub fn optimal_workgroup_size(&self, total_elements: u32) -> (u32, u32, u32) {
        let max_x = self.capabilities.max_workgroup_size_x;

        // For 1D compute, use single dimension
        let workgroup_size = max_x.min(256);
        let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;

        (num_workgroups, 1, 1)
    }

    /// Estimate memory usage for a model
    pub fn estimate_memory_usage(&self, model_params: u64, precision_bits: u32) -> u64 {
        let bytes_per_param = precision_bits as u64 / 8;
        model_params * bytes_per_param
    }

    /// Check if model fits in GPU memory
    pub fn model_fits(&self, model_params: u64, precision_bits: u32) -> bool {
        let required = self.estimate_memory_usage(model_params, precision_bits);
        required <= self.capabilities.max_buffer_size
    }
}

/// Browser detection utilities
pub mod browser {
    /// Check if running in browser environment
    pub fn is_browser() -> bool {
        cfg!(target_arch = "wasm32")
    }

    /// Get browser name (if detectable)
    #[cfg(target_arch = "wasm32")]
    pub fn browser_name() -> Option<String> {
        use web_sys::window;
        window()
            .and_then(|w| w.navigator().user_agent().ok())
            .map(|ua| {
                if ua.contains("Chrome") {
                    "Chrome".to_string()
                } else if ua.contains("Firefox") {
                    "Firefox".to_string()
                } else if ua.contains("Safari") {
                    "Safari".to_string()
                } else {
                    "Unknown".to_string()
                }
            })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn browser_name() -> Option<String> {
        None
    }

    /// Check WebGPU availability
    #[cfg(target_arch = "wasm32")]
    pub fn webgpu_available() -> bool {
        use web_sys::window;
        window()
            .map(|w| !w.navigator().gpu().is_undefined())
            .unwrap_or(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn webgpu_available() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = WebGpuContext::new(ContextConfig::default());
        assert!(!ctx.is_initialized());
    }

    #[test]
    fn test_workgroup_calculation() {
        let ctx = WebGpuContext::new(ContextConfig::default());
        let (x, y, z) = ctx.optimal_workgroup_size(1000);
        assert!(x > 0);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_memory_estimation() {
        let ctx = WebGpuContext::new(ContextConfig::default());

        // 1M params at FP16 = 2MB
        let usage = ctx.estimate_memory_usage(1_000_000, 16);
        assert_eq!(usage, 2_000_000);

        // 1M params at INT4 = 0.5MB
        let usage = ctx.estimate_memory_usage(1_000_000, 4);
        assert_eq!(usage, 500_000);
    }
}

//! Mobile deployment support for iOS (CoreML) and Android (NNAPI)
//!
//! This crate provides optimized inference for mobile devices with:
//! - CoreML integration for iOS/macOS
//! - NNAPI integration for Android
//! - INT4 quantization for memory efficiency
//! - Thermal management for sustained performance
//! - Battery-aware execution scheduling

mod coreml;
mod error;
mod nnapi;
mod quantization;
mod runtime;
mod thermal;

pub use coreml::{CoreMLConfig, CoreMLModel, CoreMLRuntime};
pub use error::{MobileError, Result};
pub use nnapi::{NnapiConfig, NnapiModel, NnapiRuntime};
pub use quantization::{Int4Quantizer, QuantizationConfig, QuantizedTensor};
pub use runtime::{MobileRuntime, RuntimeConfig, RuntimeStats};
pub use thermal::{ThermalManager, ThermalPolicy, ThermalState};

/// Platform detection
pub mod platform {
    /// Check if running on iOS
    pub fn is_ios() -> bool {
        cfg!(target_os = "ios")
    }

    /// Check if running on Android
    pub fn is_android() -> bool {
        cfg!(target_os = "android")
    }

    /// Check if running on mobile
    pub fn is_mobile() -> bool {
        is_ios() || is_android()
    }

    /// Get platform name
    pub fn platform_name() -> &'static str {
        if is_ios() {
            "iOS"
        } else if is_android() {
            "Android"
        } else {
            "Desktop"
        }
    }
}

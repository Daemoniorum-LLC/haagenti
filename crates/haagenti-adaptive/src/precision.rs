//! Precision types and hardware capabilities

use serde::{Deserialize, Serialize};

/// Precision levels for inference
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum Precision {
    /// 4-bit integer (most aggressive compression)
    INT4,
    /// 8-bit integer
    INT8,
    /// 16-bit floating point (brain float)
    BF16,
    /// 16-bit floating point
    #[default]
    FP16,
    /// 32-bit floating point (full precision)
    FP32,
}

impl Precision {
    /// Bits per element
    pub fn bits(&self) -> u32 {
        match self {
            Precision::INT4 => 4,
            Precision::INT8 => 8,
            Precision::BF16 => 16,
            Precision::FP16 => 16,
            Precision::FP32 => 32,
        }
    }

    /// Bytes per element
    pub fn bytes(&self) -> f32 {
        self.bits() as f32 / 8.0
    }

    /// VRAM usage relative to FP32 (0.0 - 1.0)
    pub fn vram_ratio(&self) -> f32 {
        self.bits() as f32 / 32.0
    }

    /// Approximate speedup factor relative to FP32
    pub fn speedup_factor(&self) -> f32 {
        match self {
            Precision::INT4 => 4.0,
            Precision::INT8 => 2.5,
            Precision::BF16 => 1.8,
            Precision::FP16 => 2.0,
            Precision::FP32 => 1.0,
        }
    }

    /// Quality impact (lower is more lossy)
    /// This is an approximation - actual impact depends on model and content
    pub fn quality_factor(&self) -> f32 {
        match self {
            Precision::INT4 => 0.92,
            Precision::INT8 => 0.97,
            Precision::BF16 => 0.995,
            Precision::FP16 => 0.998,
            Precision::FP32 => 1.0,
        }
    }

    /// Whether this precision is lossless (or nearly so)
    pub fn is_lossless(&self) -> bool {
        matches!(self, Precision::FP32 | Precision::FP16 | Precision::BF16)
    }

    /// Parse from string (returns None on failure)
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl std::str::FromStr for Precision {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "INT4" | "I4" | "4BIT" => Ok(Precision::INT4),
            "INT8" | "I8" | "8BIT" => Ok(Precision::INT8),
            "BF16" | "BFLOAT16" => Ok(Precision::BF16),
            "FP16" | "FLOAT16" | "F16" | "HALF" => Ok(Precision::FP16),
            "FP32" | "FLOAT32" | "F32" | "FULL" => Ok(Precision::FP32),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::INT4 => write!(f, "INT4"),
            Precision::INT8 => write!(f, "INT8"),
            Precision::BF16 => write!(f, "BF16"),
            Precision::FP16 => write!(f, "FP16"),
            Precision::FP32 => write!(f, "FP32"),
        }
    }
}

/// Hardware precision capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionCapabilities {
    /// Supported precisions (ordered by preference)
    pub supported: Vec<Precision>,
    /// Native (fastest) precision
    pub native: Precision,
    /// Whether INT4 uses tensor cores
    pub int4_tensor_cores: bool,
    /// Whether INT8 uses tensor cores
    pub int8_tensor_cores: bool,
    /// Available VRAM in MB
    pub vram_mb: u64,
    /// Compute capability (for NVIDIA GPUs)
    pub compute_capability: Option<(u32, u32)>,
}

impl PrecisionCapabilities {
    /// Create capabilities for a modern GPU (RTX 30/40 series)
    pub fn modern_gpu(vram_mb: u64) -> Self {
        Self {
            supported: vec![
                Precision::INT4,
                Precision::INT8,
                Precision::BF16,
                Precision::FP16,
                Precision::FP32,
            ],
            native: Precision::FP16,
            int4_tensor_cores: true,
            int8_tensor_cores: true,
            vram_mb,
            compute_capability: Some((8, 6)), // Ampere
        }
    }

    /// Create capabilities for an older GPU (GTX 10 series)
    pub fn legacy_gpu(vram_mb: u64) -> Self {
        Self {
            supported: vec![Precision::FP16, Precision::FP32],
            native: Precision::FP32,
            int4_tensor_cores: false,
            int8_tensor_cores: false,
            vram_mb,
            compute_capability: Some((6, 1)), // Pascal
        }
    }

    /// Create capabilities for CPU
    pub fn cpu(memory_mb: u64) -> Self {
        Self {
            supported: vec![Precision::INT8, Precision::FP32],
            native: Precision::FP32,
            int4_tensor_cores: false,
            int8_tensor_cores: false,
            vram_mb: memory_mb,
            compute_capability: None,
        }
    }

    /// Check if a precision is supported
    pub fn supports(&self, precision: Precision) -> bool {
        self.supported.contains(&precision)
    }

    /// Get the best supported precision at or below the given level
    pub fn best_supported(&self, max_precision: Precision) -> Precision {
        self.supported
            .iter()
            .filter(|&&p| p <= max_precision)
            .max()
            .copied()
            .unwrap_or(self.native)
    }

    /// Estimate VRAM usage for a model at given precision
    pub fn estimate_vram(&self, model_params: u64, precision: Precision) -> u64 {
        let base_bytes = model_params * 4; // FP32 baseline
        (base_bytes as f32 * precision.vram_ratio()) as u64
    }

    /// Check if a model fits at given precision
    pub fn fits_model(&self, model_params: u64, precision: Precision) -> bool {
        let required_mb = self.estimate_vram(model_params, precision) / (1024 * 1024);
        required_mb <= self.vram_mb
    }
}

impl Default for PrecisionCapabilities {
    fn default() -> Self {
        Self::modern_gpu(8192) // 8GB default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_ordering() {
        assert!(Precision::INT4 < Precision::INT8);
        assert!(Precision::INT8 < Precision::FP16);
        assert!(Precision::FP16 < Precision::FP32);
    }

    #[test]
    fn test_precision_bits() {
        assert_eq!(Precision::INT4.bits(), 4);
        assert_eq!(Precision::INT8.bits(), 8);
        assert_eq!(Precision::FP16.bits(), 16);
        assert_eq!(Precision::FP32.bits(), 32);
    }

    #[test]
    fn test_vram_ratio() {
        assert!((Precision::INT4.vram_ratio() - 0.125).abs() < 0.001);
        assert!((Precision::INT8.vram_ratio() - 0.25).abs() < 0.001);
        assert!((Precision::FP16.vram_ratio() - 0.5).abs() < 0.001);
        assert!((Precision::FP32.vram_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_capabilities() {
        let caps = PrecisionCapabilities::modern_gpu(12288);
        assert!(caps.supports(Precision::INT4));
        assert!(caps.supports(Precision::FP16));
        assert!(caps.int4_tensor_cores);
    }

    #[test]
    fn test_best_supported() {
        let caps = PrecisionCapabilities::legacy_gpu(4096);
        // Legacy doesn't support INT4/INT8
        assert_eq!(caps.best_supported(Precision::INT4), Precision::FP32);
        assert_eq!(caps.best_supported(Precision::FP16), Precision::FP16);
    }
}

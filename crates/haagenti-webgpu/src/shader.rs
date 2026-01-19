//! WGSL shader management

use crate::{Result, WebGpuError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// WGSL shader source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WgslSource {
    /// Shader label
    pub label: String,
    /// WGSL source code
    pub code: String,
    /// Entry points
    pub entry_points: Vec<String>,
}

impl WgslSource {
    /// Create new shader source
    pub fn new(label: impl Into<String>, code: impl Into<String>) -> Self {
        let code = code.into();
        let entry_points = Self::detect_entry_points(&code);

        Self {
            label: label.into(),
            code,
            entry_points,
        }
    }

    /// Detect entry points from source
    fn detect_entry_points(code: &str) -> Vec<String> {
        let mut entries = Vec::new();

        // Look for @compute, @vertex, @fragment
        for line in code.lines() {
            let line = line.trim();
            if line.starts_with("@compute") || line.starts_with("@vertex") || line.starts_with("@fragment") {
                // Next line should be fn name
                if let Some(fn_start) = line.find("fn ") {
                    let rest = &line[fn_start + 3..];
                    if let Some(paren) = rest.find('(') {
                        entries.push(rest[..paren].trim().to_string());
                    }
                }
            }
        }

        entries
    }

    /// Validate WGSL syntax (basic checks)
    pub fn validate(&self) -> Result<()> {
        // Check for required elements
        if !self.code.contains("@") {
            return Err(WebGpuError::ShaderError(
                "No shader entry point found".into(),
            ));
        }

        // Check balanced braces
        let open = self.code.matches('{').count();
        let close = self.code.matches('}').count();
        if open != close {
            return Err(WebGpuError::ShaderError(
                "Unbalanced braces in shader".into(),
            ));
        }

        Ok(())
    }
}

/// Compiled shader module
#[derive(Debug)]
pub struct ShaderModule {
    /// Shader source
    pub source: WgslSource,
    /// Compilation timestamp
    pub compiled_at: std::time::Instant,
    /// Whether compilation succeeded
    pub compiled: bool,
}

impl ShaderModule {
    /// Create from source (validates but doesn't compile)
    pub fn from_source(source: WgslSource) -> Result<Self> {
        source.validate()?;

        Ok(Self {
            source,
            compiled_at: std::time::Instant::now(),
            compiled: false,
        })
    }

    /// Mark as compiled
    pub fn mark_compiled(&mut self) {
        self.compiled = true;
        self.compiled_at = std::time::Instant::now();
    }
}

/// Standard WGSL shader templates
pub mod templates {
    use super::WgslSource;

    /// Matrix multiplication shader
    pub fn matmul() -> WgslSource {
        WgslSource::new(
            "matmul",
            r#"
struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16)
fn matmul_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
        sum = sum + a[row * uniforms.K + k] * b[k * uniforms.N + col];
    }
    c[row * uniforms.N + col] = sum;
}
"#,
        )
    }

    /// Element-wise addition shader
    pub fn add() -> WgslSource {
        WgslSource::new(
            "add",
            r#"
struct Uniforms {
    size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}
"#,
        )
    }

    /// GELU activation shader
    pub fn gelu() -> WgslSource {
        WgslSource::new(
            "gelu",
            r#"
struct Uniforms {
    size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;

@compute @workgroup_size(256)
fn gelu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    let x = input[idx];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#,
        )
    }

    /// Softmax shader
    pub fn softmax() -> WgslSource {
        WgslSource::new(
            "softmax",
            r#"
struct Uniforms {
    batch_size: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn softmax_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= uniforms.batch_size) {
        return;
    }

    let offset = batch_idx * uniforms.seq_len;

    // Find max for numerical stability
    var max_val: f32 = input[offset];
    for (var i: u32 = 1u; i < uniforms.seq_len; i = i + 1u) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < uniforms.seq_len; i = i + 1u) {
        let exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i: u32 = 0u; i < uniforms.seq_len; i = i + 1u) {
        output[offset + i] = output[offset + i] / sum;
    }
}
"#,
        )
    }

    /// Layer normalization shader
    pub fn layer_norm() -> WgslSource {
        WgslSource::new(
            "layer_norm",
            r#"
struct Uniforms {
    batch_size: u32,
    hidden_size: u32,
    epsilon: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn layer_norm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= uniforms.batch_size) {
        return;
    }

    let offset = batch_idx * uniforms.hidden_size;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < uniforms.hidden_size; i = i + 1u) {
        mean = mean + input[offset + i];
    }
    mean = mean / f32(uniforms.hidden_size);

    // Compute variance
    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < uniforms.hidden_size; i = i + 1u) {
        let diff = input[offset + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(uniforms.hidden_size);

    // Normalize and scale
    let std_inv = 1.0 / sqrt(variance + uniforms.epsilon);
    for (var i: u32 = 0u; i < uniforms.hidden_size; i = i + 1u) {
        let normalized = (input[offset + i] - mean) * std_inv;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"#,
        )
    }

    /// Dequantization shader (INT4 to FP32)
    pub fn dequantize_int4() -> WgslSource {
        WgslSource::new(
            "dequantize_int4",
            r#"
struct Uniforms {
    size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<u32>;  // Packed INT4
@group(0) @binding(2) var<storage, read> scales: array<f32>;      // Per-group scale
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const GROUP_SIZE: u32 = 32u;

@compute @workgroup_size(256)
fn dequantize_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    // Each u32 contains 8 INT4 values
    let packed_idx = idx / 8u;
    let bit_offset = (idx % 8u) * 4u;

    let packed = quantized[packed_idx];
    let int4_val = (packed >> bit_offset) & 0xFu;

    // Convert from unsigned to signed (-8 to 7)
    let signed_val = f32(i32(int4_val) - 8);

    // Apply scale
    let group_idx = idx / GROUP_SIZE;
    output[idx] = signed_val * scales[group_idx];
}
"#,
        )
    }
}

/// Shader library for common operations
#[derive(Debug, Default)]
pub struct ShaderLibrary {
    shaders: HashMap<String, ShaderModule>,
}

impl ShaderLibrary {
    /// Create new library with standard shaders
    pub fn with_standard() -> Self {
        let mut lib = Self::default();

        // Add standard templates
        lib.add(templates::matmul()).ok();
        lib.add(templates::add()).ok();
        lib.add(templates::gelu()).ok();
        lib.add(templates::softmax()).ok();
        lib.add(templates::layer_norm()).ok();
        lib.add(templates::dequantize_int4()).ok();

        lib
    }

    /// Add a shader to the library
    pub fn add(&mut self, source: WgslSource) -> Result<()> {
        let module = ShaderModule::from_source(source.clone())?;
        self.shaders.insert(source.label.clone(), module);
        Ok(())
    }

    /// Get a shader by label
    pub fn get(&self, label: &str) -> Option<&ShaderModule> {
        self.shaders.get(label)
    }

    /// List all shader labels
    pub fn list(&self) -> Vec<&str> {
        self.shaders.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_validation() {
        let valid = WgslSource::new("test", r#"
@compute @workgroup_size(256)
fn main() {
    // Empty
}
"#);
        assert!(valid.validate().is_ok());

        let invalid = WgslSource::new("test", "no entry point");
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_standard_shaders() {
        let lib = ShaderLibrary::with_standard();

        assert!(lib.get("matmul").is_some());
        assert!(lib.get("add").is_some());
        assert!(lib.get("gelu").is_some());
        assert!(lib.get("softmax").is_some());
        assert!(lib.get("layer_norm").is_some());
        assert!(lib.get("dequantize_int4").is_some());
    }

    #[test]
    fn test_template_shaders() {
        let matmul = templates::matmul();
        assert!(matmul.validate().is_ok());

        let gelu = templates::gelu();
        assert!(gelu.validate().is_ok());

        let dequant = templates::dequantize_int4();
        assert!(dequant.validate().is_ok());
    }
}

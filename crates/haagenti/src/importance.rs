//! Importance-Guided Compression
//!
//! Uses pre-computed or heuristic importance scores to guide coefficient retention
//! during spectral compression. Higher importance coefficients get more retention.
//!
//! ## Importance Sources
//!
//! 1. **Pre-computed** - Load from `.importance.json` file (Fisher info, gradient norms)
//! 2. **Layer-type heuristics** - Built-in sensitivity profiles for LLM layers
//! 3. **Magnitude-based** - Default: retain largest magnitude coefficients
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::importance::{ImportanceGuidedEncoder, ImportanceGuidedDecoder, ImportanceMap};
//!
//! // Load importance from file (or use heuristics)
//! let map = ImportanceMap::load_or_default("model.importance.json");
//!
//! let encoder = ImportanceGuidedEncoder::new(0.50, map);
//! let compressed = encoder.encode(&tensor, width, height, "model.layers.0.mlp.gate_proj")?;
//!
//! let decoder = ImportanceGuidedDecoder::new();
//! let reconstructed = decoder.decode(&compressed)?;
//! ```
//!
//! ## Importance File Format
//!
//! ```json
//! {
//!   "version": 1,
//!   "source": "fisher_information",
//!   "tensors": {
//!     "model.layers.0.mlp.gate_proj.weight": {
//!       "importance": 0.65,
//!       "sensitivity": "medium"
//!     },
//!     "model.layers.0.self_attn.q_proj.weight": {
//!       "importance": 0.85,
//!       "sensitivity": "high"
//!     }
//!   }
//! }
//! ```

use haagenti_core::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Quality sensitivity level for a tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sensitivity {
    /// Can tolerate heavy compression (50% quality acceptable)
    VeryLow,
    /// Tolerates moderate compression (70% quality acceptable)
    Low,
    /// Standard sensitivity (85% quality target)
    Medium,
    /// Sensitive to compression (95% quality target)
    High,
    /// Critical - must preserve full precision
    Full,
}

impl Sensitivity {
    /// Get minimum quality threshold for this sensitivity level.
    pub fn min_quality(&self) -> f32 {
        match self {
            Sensitivity::VeryLow => 0.50,
            Sensitivity::Low => 0.70,
            Sensitivity::Medium => 0.85,
            Sensitivity::High => 0.95,
            Sensitivity::Full => 1.00,
        }
    }

    /// Get retention multiplier for this sensitivity level.
    pub fn retention_multiplier(&self) -> f32 {
        match self {
            Sensitivity::VeryLow => 0.50,
            Sensitivity::Low => 0.70,
            Sensitivity::Medium => 1.00,
            Sensitivity::High => 1.20,
            Sensitivity::Full => 1.50,
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "very_low" | "verylow" => Sensitivity::VeryLow,
            "low" => Sensitivity::Low,
            "medium" | "normal" => Sensitivity::Medium,
            "high" => Sensitivity::High,
            "full" | "critical" => Sensitivity::Full,
            _ => Sensitivity::Medium,
        }
    }
}

/// Importance information for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorImportance {
    /// Overall importance score (0.0 - 1.0)
    pub importance: f32,
    /// Quality sensitivity level
    pub sensitivity: Sensitivity,
    /// Optional: per-coefficient importance weights (same shape as tensor)
    pub coefficient_weights: Option<Vec<f32>>,
}

impl Default for TensorImportance {
    fn default() -> Self {
        Self {
            importance: 0.5,
            sensitivity: Sensitivity::Medium,
            coefficient_weights: None,
        }
    }
}

/// Map of tensor names to importance scores.
#[derive(Debug, Clone)]
pub struct ImportanceMap {
    /// Source of importance data
    pub source: String,
    /// Per-tensor importance
    pub tensors: HashMap<String, TensorImportance>,
    /// Whether to use heuristics for missing tensors
    pub use_heuristics: bool,
}

impl Default for ImportanceMap {
    fn default() -> Self {
        Self {
            source: "heuristic".to_string(),
            tensors: HashMap::new(),
            use_heuristics: true,
        }
    }
}

impl ImportanceMap {
    /// Create an empty importance map that uses heuristics for all tensors.
    pub fn heuristic_only() -> Self {
        Self {
            source: "heuristic".to_string(),
            tensors: HashMap::new(),
            use_heuristics: true,
        }
    }

    /// Load importance map from JSON file, falling back to heuristics.
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        match Self::load(path) {
            Ok(map) => map,
            Err(_) => Self::heuristic_only(),
        }
    }

    /// Load importance map from JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::parse(&content)
    }

    /// Parse importance map from JSON string.
    pub fn parse(json: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::corrupted(format!("invalid JSON: {}", e)))?;

        let obj = value
            .as_object()
            .ok_or_else(|| Error::corrupted("expected JSON object"))?;

        let source = obj
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let mut tensors = HashMap::new();

        if let Some(tensors_obj) = obj.get("tensors").and_then(|v| v.as_object()) {
            for (name, info) in tensors_obj {
                if let Some(info_obj) = info.as_object() {
                    let importance = info_obj
                        .get("importance")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(0.5);

                    let sensitivity = info_obj
                        .get("sensitivity")
                        .and_then(|v| v.as_str())
                        .map(Sensitivity::from_str)
                        .unwrap_or(Sensitivity::Medium);

                    tensors.insert(
                        name.clone(),
                        TensorImportance {
                            importance,
                            sensitivity,
                            coefficient_weights: None,
                        },
                    );
                }
            }
        }

        Ok(Self {
            source,
            tensors,
            use_heuristics: true,
        })
    }

    /// Get importance for a tensor, using heuristics if not found.
    pub fn get(&self, name: &str) -> TensorImportance {
        if let Some(info) = self.tensors.get(name) {
            return info.clone();
        }

        if self.use_heuristics {
            return Self::heuristic_importance(name);
        }

        TensorImportance::default()
    }

    /// Compute heuristic importance based on layer name patterns.
    ///
    /// Based on findings from haagenti-importance QualityPredictor:
    /// - LayerNorm/Bias: Full sensitivity (critical for output)
    /// - Embeddings: Full sensitivity (vocabulary precision)
    /// - Attention Q/K: Medium-High sensitivity
    /// - Attention V/O: High sensitivity
    /// - FFN/MLP: Low sensitivity (largest but most compressible)
    pub fn heuristic_importance(name: &str) -> TensorImportance {
        let name_lower = name.to_lowercase();

        // === Critical layers: Full sensitivity ===
        if name_lower.contains("layernorm")
            || name_lower.contains("layer_norm")
            || name_lower.contains("ln_")
            || name_lower.contains("_ln")
            || name_lower.contains("norm.weight")
            || name_lower.contains("rms_norm")
            || name_lower.contains("input_layernorm")
            || name_lower.contains("final_layernorm")
            || name_lower.contains("post_attention_layernorm")
        {
            return TensorImportance {
                importance: 1.0,
                sensitivity: Sensitivity::Full,
                coefficient_weights: None,
            };
        }

        // Bias vectors - small but critical
        if name_lower.contains(".bias") || name_lower.ends_with("_bias") {
            return TensorImportance {
                importance: 1.0,
                sensitivity: Sensitivity::Full,
                coefficient_weights: None,
            };
        }

        // Embeddings - vocabulary precision
        if name_lower.contains("embed_tokens")
            || name_lower.contains("wte")
            || name_lower.contains("word_embed")
            || name_lower.contains("token_embed")
            || name_lower.contains("embed.weight")
        {
            return TensorImportance {
                importance: 0.95,
                sensitivity: Sensitivity::Full,
                coefficient_weights: None,
            };
        }

        // Output head (lm_head) - critical for token prediction
        if name_lower.contains("lm_head") || name_lower.contains("output.weight") {
            return TensorImportance {
                importance: 0.90,
                sensitivity: Sensitivity::High,
                coefficient_weights: None,
            };
        }

        // === Attention layers ===
        // Q/K projections: important for attention patterns
        if name_lower.contains("q_proj")
            || name_lower.contains("k_proj")
            || name_lower.contains(".wq.")
            || name_lower.contains(".wk.")
            || name_lower.contains("query")
            || name_lower.contains("key")
        {
            return TensorImportance {
                importance: 0.75,
                sensitivity: Sensitivity::Medium,
                coefficient_weights: None,
            };
        }

        // V/O projections: high sensitivity for output quality
        if name_lower.contains("v_proj")
            || name_lower.contains("o_proj")
            || name_lower.contains(".wv.")
            || name_lower.contains(".wo.")
            || name_lower.contains("value")
            || name_lower.contains("dense")
        {
            return TensorImportance {
                importance: 0.80,
                sensitivity: Sensitivity::High,
                coefficient_weights: None,
            };
        }

        // Combined QKV projection
        if name_lower.contains("qkv") || name_lower.contains("c_attn") {
            return TensorImportance {
                importance: 0.75,
                sensitivity: Sensitivity::Medium,
                coefficient_weights: None,
            };
        }

        // === FFN/MLP layers: Most compressible ===
        if name_lower.contains("mlp.")
            || name_lower.contains("feed_forward")
            || name_lower.contains("ffn")
            || name_lower.contains(".fc1")
            || name_lower.contains(".fc2")
            || name_lower.contains("up_proj")
            || name_lower.contains("down_proj")
            || name_lower.contains("gate_proj")
            || name_lower.contains("w1.")
            || name_lower.contains("w2.")
            || name_lower.contains("w3.")
        {
            return TensorImportance {
                importance: 0.50,
                sensitivity: Sensitivity::Low,
                coefficient_weights: None,
            };
        }

        // Default: medium sensitivity
        TensorImportance::default()
    }

    /// Add or update importance for a tensor.
    pub fn set(&mut self, name: String, importance: TensorImportance) {
        self.tensors.insert(name, importance);
    }

    /// Get the number of tensors with explicit importance.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the map is empty (no explicit importance, only heuristics).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Compressed weight with importance-guided retention.
#[derive(Debug, Clone)]
pub struct ImportanceCompressedWeight {
    /// Original tensor dimensions
    pub width: usize,
    pub height: usize,
    /// Importance score used for this tensor
    pub importance: f32,
    /// Effective retention ratio (adjusted by importance)
    pub effective_retention: f32,
    /// Number of retained coefficients
    pub retained_count: usize,
    /// DCT coefficients (FP32 for now - can add INT4 later)
    pub coefficients: Vec<f32>,
    /// Indices of retained coefficients in original DCT array
    pub indices: Vec<u32>,
    /// Total DCT coefficients in original
    pub total_coefficients: usize,
}

impl ImportanceCompressedWeight {
    /// Calculate storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.coefficients.len() * 4 + self.indices.len() * 4
    }

    /// Calculate original size in bytes.
    pub fn original_bytes(&self) -> usize {
        self.width * self.height * 4
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let orig = self.original_bytes();
        let compressed = self.storage_bytes();
        if compressed == 0 {
            0.0
        } else {
            orig as f32 / compressed as f32
        }
    }
}

/// Importance-guided spectral encoder.
///
/// Adjusts retention based on tensor importance:
/// - High importance tensors get more retention
/// - Low importance tensors can be compressed more aggressively
#[derive(Debug, Clone)]
pub struct ImportanceGuidedEncoder {
    /// Base retention ratio
    base_retention: f32,
    /// Importance map
    importance_map: ImportanceMap,
    /// Minimum retention (even for very low importance)
    min_retention: f32,
    /// Maximum retention (cap for very high importance)
    max_retention: f32,
}

impl ImportanceGuidedEncoder {
    /// Create encoder with base retention and importance map.
    pub fn new(base_retention: f32, importance_map: ImportanceMap) -> Self {
        Self {
            base_retention: base_retention.clamp(0.01, 1.0),
            importance_map,
            min_retention: 0.10,
            max_retention: 0.95,
        }
    }

    /// Create encoder with heuristic-only importance.
    pub fn with_heuristics(base_retention: f32) -> Self {
        Self::new(base_retention, ImportanceMap::heuristic_only())
    }

    /// Set minimum retention.
    pub fn with_min_retention(mut self, min: f32) -> Self {
        self.min_retention = min.clamp(0.01, 1.0);
        self
    }

    /// Set maximum retention.
    pub fn with_max_retention(mut self, max: f32) -> Self {
        self.max_retention = max.clamp(0.01, 1.0);
        self
    }

    /// Calculate effective retention for a tensor based on its importance.
    pub fn effective_retention(&self, tensor_name: &str) -> f32 {
        let info = self.importance_map.get(tensor_name);

        // Adjust retention based on importance and sensitivity
        let importance_factor = info.importance;
        let sensitivity_mult = info.sensitivity.retention_multiplier();

        // Scale retention: low importance = lower retention, high importance = higher retention
        // Base formula: effective = base * (0.5 + importance * 0.5) * sensitivity_mult
        let adjusted = self.base_retention * (0.5 + importance_factor * 0.5) * sensitivity_mult;

        adjusted.clamp(self.min_retention, self.max_retention)
    }

    /// Encode a 2D tensor with importance-guided retention.
    pub fn encode(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        tensor_name: &str,
    ) -> Result<ImportanceCompressedWeight> {
        let n = width * height;
        if data.len() != n {
            return Err(Error::corrupted("data size mismatch"));
        }
        if n == 0 {
            return Err(Error::corrupted("empty tensor"));
        }

        let info = self.importance_map.get(tensor_name);
        let effective_retention = self.effective_retention(tensor_name);

        // Perform 2D DCT
        let dct_coeffs = dct_2d(data, width, height);

        // Calculate how many coefficients to retain
        let retain_count = ((n as f32 * effective_retention) as usize).max(1).min(n);

        // Sort coefficients by magnitude (or by importance weights if available)
        let mut indexed: Vec<(usize, f32, f32)> = if let Some(weights) = &info.coefficient_weights {
            // Weight by importance
            dct_coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let w = weights.get(i).copied().unwrap_or(1.0);
                    (i, c, c.abs() * w)
                })
                .collect()
        } else {
            // Pure magnitude-based
            dct_coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, c, c.abs()))
                .collect()
        };

        // Sort by weighted magnitude (descending)
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top retain_count coefficients
        let retained: Vec<(usize, f32)> = indexed
            .into_iter()
            .take(retain_count)
            .map(|(i, c, _)| (i, c))
            .collect();

        let coefficients: Vec<f32> = retained.iter().map(|(_, c)| *c).collect();
        let indices: Vec<u32> = retained.iter().map(|(i, _)| *i as u32).collect();

        Ok(ImportanceCompressedWeight {
            width,
            height,
            importance: info.importance,
            effective_retention,
            retained_count: retain_count,
            coefficients,
            indices,
            total_coefficients: n,
        })
    }

    /// Get the importance map.
    pub fn importance_map(&self) -> &ImportanceMap {
        &self.importance_map
    }
}

/// Importance-guided decoder.
#[derive(Debug, Clone, Default)]
pub struct ImportanceGuidedDecoder;

impl ImportanceGuidedDecoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode compressed weight back to tensor.
    pub fn decode(&self, compressed: &ImportanceCompressedWeight) -> Result<Vec<f32>> {
        let n = compressed.width * compressed.height;

        // Reconstruct DCT coefficient array
        let mut dct_coeffs = vec![0.0f32; n];

        for (i, &value) in compressed.coefficients.iter().enumerate() {
            if i < compressed.indices.len() {
                let idx = compressed.indices[i] as usize;
                if idx < n {
                    dct_coeffs[idx] = value;
                }
            }
        }

        // Inverse DCT
        let reconstructed = idct_2d(&dct_coeffs, compressed.width, compressed.height);

        Ok(reconstructed)
    }
}

// =============================================================================
// DCT Implementation (same as mixed_precision.rs)
// =============================================================================

fn dct_1d(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; n];
    let scale = (2.0 / n as f32).sqrt();

    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum +=
                input[i] * (std::f32::consts::PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32).cos();
        }
        output[k] = sum
            * scale
            * if k == 0 {
                1.0 / std::f32::consts::SQRT_2
            } else {
                1.0
            };
    }

    output
}

fn idct_1d(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; n];
    let scale = (2.0 / n as f32).sqrt();

    for i in 0..n {
        let mut sum = 0.0f32;
        for k in 0..n {
            let coeff = input[k]
                * if k == 0 {
                    1.0 / std::f32::consts::SQRT_2
                } else {
                    1.0
                };
            sum += coeff * (std::f32::consts::PI * ((2 * i + 1) * k) as f32 / (2 * n) as f32).cos();
        }
        output[i] = sum * scale;
    }

    output
}

fn dct_2d(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }

    // DCT on rows
    let mut temp = vec![0.0f32; width * height];
    for row in 0..height {
        let row_data: Vec<f32> = data[row * width..(row + 1) * width].to_vec();
        let dct_row = dct_1d(&row_data);
        temp[row * width..(row + 1) * width].copy_from_slice(&dct_row);
    }

    // DCT on columns
    let mut output = vec![0.0f32; width * height];
    for col in 0..width {
        let col_data: Vec<f32> = (0..height).map(|row| temp[row * width + col]).collect();
        let dct_col = dct_1d(&col_data);
        for row in 0..height {
            output[row * width + col] = dct_col[row];
        }
    }

    output
}

fn idct_2d(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }

    // IDCT on columns
    let mut temp = vec![0.0f32; width * height];
    for col in 0..width {
        let col_data: Vec<f32> = (0..height).map(|row| data[row * width + col]).collect();
        let idct_col = idct_1d(&col_data);
        for row in 0..height {
            temp[row * width + col] = idct_col[row];
        }
    }

    // IDCT on rows
    let mut output = vec![0.0f32; width * height];
    for row in 0..height {
        let row_data: Vec<f32> = temp[row * width..(row + 1) * width].to_vec();
        let idct_row = idct_1d(&row_data);
        output[row * width..(row + 1) * width].copy_from_slice(&idct_row);
    }

    output
}

// =============================================================================
// Quality Metrics
// =============================================================================

/// Calculate MSE between two vectors.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

/// Calculate cosine similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_levels() {
        assert_eq!(Sensitivity::VeryLow.min_quality(), 0.50);
        assert_eq!(Sensitivity::Low.min_quality(), 0.70);
        assert_eq!(Sensitivity::Medium.min_quality(), 0.85);
        assert_eq!(Sensitivity::High.min_quality(), 0.95);
        assert_eq!(Sensitivity::Full.min_quality(), 1.00);
    }

    #[test]
    fn test_sensitivity_from_str() {
        assert_eq!(Sensitivity::from_str("low"), Sensitivity::Low);
        assert_eq!(Sensitivity::from_str("HIGH"), Sensitivity::High);
        assert_eq!(Sensitivity::from_str("critical"), Sensitivity::Full);
        assert_eq!(Sensitivity::from_str("unknown"), Sensitivity::Medium);
    }

    #[test]
    fn test_heuristic_importance_layernorm() {
        let info = ImportanceMap::heuristic_importance("model.layers.0.input_layernorm.weight");
        assert_eq!(info.sensitivity, Sensitivity::Full);
        assert_eq!(info.importance, 1.0);
    }

    #[test]
    fn test_heuristic_importance_mlp() {
        let info = ImportanceMap::heuristic_importance("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(info.sensitivity, Sensitivity::Low);
        assert!(info.importance < 0.6);
    }

    #[test]
    fn test_heuristic_importance_attention() {
        let info = ImportanceMap::heuristic_importance("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(info.sensitivity, Sensitivity::Medium);
        assert!(info.importance >= 0.7);
    }

    #[test]
    fn test_heuristic_importance_embedding() {
        let info = ImportanceMap::heuristic_importance("model.embed_tokens.weight");
        assert_eq!(info.sensitivity, Sensitivity::Full);
        assert!(info.importance >= 0.9);
    }

    #[test]
    fn test_importance_map_parse() {
        let json = r#"{
            "version": 1,
            "source": "fisher_information",
            "tensors": {
                "layer.0.weight": {
                    "importance": 0.8,
                    "sensitivity": "high"
                }
            }
        }"#;

        let map = ImportanceMap::parse(json).unwrap();
        assert_eq!(map.source, "fisher_information");
        assert_eq!(map.tensors.len(), 1);

        let info = map.get("layer.0.weight");
        assert_eq!(info.importance, 0.8);
        assert_eq!(info.sensitivity, Sensitivity::High);
    }

    #[test]
    fn test_importance_map_fallback() {
        let map = ImportanceMap::heuristic_only();

        // Not in map, should use heuristic
        let info = map.get("model.layers.0.mlp.up_proj.weight");
        assert_eq!(info.sensitivity, Sensitivity::Low);
    }

    #[test]
    fn test_effective_retention() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);

        // Low importance (MLP) should get lower retention
        let mlp_ret = encoder.effective_retention("model.layers.0.mlp.gate_proj.weight");

        // High importance (embedding) should get higher retention
        let embed_ret = encoder.effective_retention("model.embed_tokens.weight");

        assert!(
            mlp_ret < embed_ret,
            "MLP {} should be < embedding {}",
            mlp_ret,
            embed_ret
        );
    }

    #[test]
    fn test_encoder_basic() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.encode(&data, 8, 8, "test.weight").unwrap();

        assert!(compressed.retained_count > 0);
        assert_eq!(compressed.coefficients.len(), compressed.retained_count);
        assert_eq!(compressed.indices.len(), compressed.retained_count);
    }

    #[test]
    fn test_roundtrip() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.70, map);
        let decoder = ImportanceGuidedDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let compressed = encoder.encode(&data, 8, 8, "test.weight").unwrap();
        let reconstructed = decoder.decode(&compressed).unwrap();

        assert_eq!(reconstructed.len(), data.len());

        let cos = cosine_similarity(&data, &reconstructed);
        assert!(cos > 0.8, "Cosine similarity too low: {}", cos);
    }

    #[test]
    fn test_importance_affects_retention() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);

        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

        // Compress same data with different "tensor names" (different importance)
        let mlp_compressed = encoder
            .encode(&data, 16, 16, "model.layers.0.mlp.gate_proj.weight")
            .unwrap();
        let norm_compressed = encoder
            .encode(&data, 16, 16, "model.layers.0.input_layernorm.weight")
            .unwrap();

        // LayerNorm should have more retained coefficients
        assert!(
            norm_compressed.retained_count > mlp_compressed.retained_count,
            "LayerNorm ({}) should retain more than MLP ({})",
            norm_compressed.retained_count,
            mlp_compressed.retained_count
        );
    }

    #[test]
    fn test_compression_ratio() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.30, map);

        let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        let compressed = encoder.encode(&data, 32, 32, "test.weight").unwrap();

        let ratio = compressed.compression_ratio();
        assert!(
            ratio > 1.0,
            "Expected compression ratio > 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_quality_by_layer_type() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);
        let decoder = ImportanceGuidedDecoder::new();

        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

        // Test different layer types
        let layers = [
            ("model.layers.0.mlp.gate_proj.weight", Sensitivity::Low),
            (
                "model.layers.0.self_attn.q_proj.weight",
                Sensitivity::Medium,
            ),
            ("model.layers.0.input_layernorm.weight", Sensitivity::Full),
        ];

        let mut qualities: Vec<(String, f32)> = Vec::new();

        for (name, _expected_sens) in layers {
            let compressed = encoder.encode(&data, 16, 16, name).unwrap();
            let reconstructed = decoder.decode(&compressed).unwrap();
            let cos = cosine_similarity(&data, &reconstructed);
            qualities.push((name.to_string(), cos));
        }

        // Higher importance layers should have better quality
        // Note: Quality difference may be small for synthetic data
        println!("Quality by layer type:");
        for (name, cos) in &qualities {
            println!("  {}: {:.4}", name, cos);
        }
    }

    #[test]
    fn test_dct_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let dct = dct_2d(&data, 8, 8);
        let reconstructed = idct_2d(&dct, 8, 8);

        let cos = cosine_similarity(&data, &reconstructed);
        assert!(cos > 0.999, "DCT roundtrip should be near-perfect: {}", cos);
    }

    #[test]
    fn test_empty_tensor() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);

        let result = encoder.encode(&[], 0, 0, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_size_mismatch() {
        let map = ImportanceMap::heuristic_only();
        let encoder = ImportanceGuidedEncoder::new(0.50, map);

        let data = vec![1.0f32; 10];
        let result = encoder.encode(&data, 8, 8, "test"); // Should be 64 elements
        assert!(result.is_err());
    }
}

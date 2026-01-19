//! Create a compressed model for inference testing
//!
//! This compresses all weights using Spectral + INT4, then decompresses
//! and saves to a new safetensors file for comparison testing.
//!
//! Usage:
//! ```bash
//! RETENTION=0.30 cargo run --example create_compressed_model --features='lz4,zstd' --release
//! ```
//!
//! Environment variables:
//! - RETENTION: Base retention ratio (0.0-1.0), default 0.30
//! - TARGET_QUALITY: Target energy retention for adaptive mode (0.0-1.0), default 0.90
//! - NO_INT4: Disable INT4 quantization
//! - HYBRID: Use position-based hybrid compression mode
//! - LAYERTYPE: Use layer-type-based compression mode
//! - ADAPTIVE: Use per-tensor adaptive retention based on spectral analysis
//! - MIXED_PRECISION: Use FP16 essentials + INT4 details encoding
//! - FP16_RATIO: Fraction of retained coefficients to store as FP16 (0.0-1.0), default 0.20
//! - IMPORTANCE: Use importance-guided compression (heuristic or file-based)
//! - IMPORTANCE_FILE: Path to JSON importance map file (optional, uses heuristics if not set)
//! - MODEL: Specify model name (default: Qwen/Qwen2.5-0.5B-Instruct)
//! - QUIET: Disable progress bar (for CI/scripting)

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::adaptive::AdaptiveSpectralEncoder;
use haagenti::mixed_precision::{MixedPrecisionEncoder, MixedPrecisionDecoder};
use haagenti::importance::{ImportanceGuidedEncoder, ImportanceGuidedDecoder, ImportanceMap};
use indicatif::{ProgressBar, ProgressStyle};

/// Error types for model compression
#[derive(Debug)]
enum CompressionError {
    ModelNotFound(String),
    FileReadError(String, std::io::Error),
    FileWriteError(String, std::io::Error),
    HeaderParseError(String),
    EncodingError(String),
    InvalidDtype(String),
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionError::ModelNotFound(model) => {
                write!(f, "Model '{}' not found in HuggingFace cache.\n\
                       Download with: huggingface-cli download {}", model, model)
            }
            CompressionError::FileReadError(path, err) => {
                write!(f, "Failed to read '{}': {}", path, err)
            }
            CompressionError::FileWriteError(path, err) => {
                write!(f, "Failed to write '{}': {}", path, err)
            }
            CompressionError::HeaderParseError(msg) => {
                write!(f, "Failed to parse safetensors header: {}", msg)
            }
            CompressionError::EncodingError(msg) => {
                write!(f, "Encoding error: {}", msg)
            }
            CompressionError::InvalidDtype(dtype) => {
                write!(f, "Unsupported dtype '{}'. Supported: F32, F16, BF16", dtype)
            }
        }
    }
}

impl std::error::Error for CompressionError {}

const Q4_BLOCK_SIZE: usize = 32;
const MAX_ELEMENTS: usize = 10_000_000;

/// Compression mode
#[derive(Debug, Clone, Copy, PartialEq)]
enum CompressionMode {
    Uniform,        // Same retention for all layers
    Hybrid,         // Position-based (first/last layers protected)
    LayerType,      // Type-based (FFN aggressive, attention careful, norms full)
    Adaptive,       // Per-tensor adaptive retention based on spectral analysis
    MixedPrecision, // FP16 essentials + INT4 details
    Importance,     // Learned/heuristic importance-guided compression
}

/// Determine retention ratio based on tensor name and compression mode
///
/// Layer-type mode uses insights from haagenti-importance QualityPredictor:
/// - FFN/MLP: Can tolerate 50% quality (largest portion of model)
/// - Attention Q/K: Medium sensitivity (70%)
/// - Attention V/O: Higher sensitivity (80%)
/// - LayerNorm/Bias: Critical (100%)
/// - Embeddings: Critical (100%)
fn get_layer_retention(name: &str, base_retention: f32, mode: CompressionMode) -> f32 {
    match mode {
        CompressionMode::Uniform => base_retention,

        CompressionMode::Hybrid => {
            // Original position-based hybrid mode
            if name.contains("layernorm") || name.contains("norm.weight") || name.contains("bias") {
                return 1.0;
            }
            if name.contains("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Some(idx_str) = parts.get(2) {
                    if let Ok(layer_idx) = idx_str.parse::<usize>() {
                        if layer_idx == 0 || layer_idx >= 23 { return 1.0; }
                        if layer_idx <= 2 || layer_idx >= 21 { return 0.80; }
                        return base_retention;
                    }
                }
            }
            base_retention
        }

        CompressionMode::LayerType => {
            // === Critical layers: 100% retention ===
            // LayerNorm - critical for training stability
            if name.contains("layernorm") || name.contains("layer_norm") ||
               name.contains("ln_") || name.contains("_ln") ||
               name.contains("norm.weight") || name.contains("input_layernorm") ||
               name.contains("post_attention_layernorm") || name.contains("final_layernorm") {
                return 1.0;
            }

            // Bias vectors - small but critical for output distribution
            if name.contains(".bias") || name.ends_with("_bias") {
                return 1.0;
            }

            // Embeddings - need full vocabulary precision
            if name.contains("embed_tokens") || name.contains("wte") ||
               name.contains("word_embed") || name.contains("token_embed") {
                return 1.0;
            }

            // Output head (lm_head) - critical for token prediction
            if name.contains("lm_head") || name.contains("output.weight") {
                return 0.90; // Slightly less critical than embeddings
            }

            // === FFN/MLP: Aggressive compression (50-60% of base) ===
            // These are ~67% of model parameters and tolerate compression well
            if name.contains("mlp.") || name.contains("feed_forward") ||
               name.contains("ffn") || name.contains(".fc1") || name.contains(".fc2") ||
               name.contains("up_proj") || name.contains("down_proj") ||
               name.contains("gate_proj") || name.contains("w1.") ||
               name.contains("w2.") || name.contains("w3.") {
                // FFN can tolerate 50% quality per haagenti-importance
                return (base_retention * 0.70).max(0.40);
            }

            // === Attention Q/K: Medium compression ===
            if name.contains("q_proj") || name.contains("k_proj") ||
               name.contains(".wq.") || name.contains(".wk.") ||
               name.contains("query") || name.contains("key") {
                return base_retention; // Use base retention
            }

            // === Attention V/O: Careful compression ===
            if name.contains("v_proj") || name.contains("o_proj") ||
               name.contains(".wv.") || name.contains(".wo.") ||
               name.contains("value") || name.contains("dense") {
                return (base_retention * 1.1).min(1.0); // 10% more retention
            }

            // === Attention combined (qkv_proj) ===
            if name.contains("qkv") || name.contains("c_attn") {
                return base_retention;
            }

            // Default: base retention
            base_retention
        }

        CompressionMode::Adaptive => {
            // Adaptive mode is handled separately with AdaptiveSpectralEncoder
            // This branch should not be reached in normal operation
            base_retention
        }

        CompressionMode::MixedPrecision => {
            // Mixed precision mode uses base retention
            // Layer-type adjustments could be added here if needed
            base_retention
        }

        CompressionMode::Importance => {
            // Importance mode is handled separately with ImportanceGuidedEncoder
            // This branch should not be reached in normal operation
            base_retention
        }
    }
}

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

fn parse_safetensors_header(data: &[u8]) -> Result<(usize, HashMap<String, TensorInfo>), String> {
    if data.len() < 8 {
        return Err("File too small".to_string());
    }

    let header_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if data.len() < 8 + header_len {
        return Err("Invalid header length".to_string());
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_len])
        .map_err(|e| format!("Invalid UTF-8: {}", e))?;

    let header: serde_json::Value =
        serde_json::from_str(header_json).map_err(|e| format!("Invalid JSON: {}", e))?;

    let mut tensors = HashMap::new();

    if let serde_json::Value::Object(obj) = header {
        for (name, value) in obj {
            if name == "__metadata__" {
                continue;
            }

            if let serde_json::Value::Object(tensor_obj) = value {
                let dtype = tensor_obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("F32")
                    .to_string();

                let shape: Vec<usize> = tensor_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect()
                    })
                    .unwrap_or_default();

                let offsets = tensor_obj.get("data_offsets").and_then(|v| v.as_array());
                if let Some(offs) = offsets {
                    let start = offs[0].as_u64().unwrap_or(0) as usize;
                    let end = offs[1].as_u64().unwrap_or(0) as usize;
                    tensors.insert(name, TensorInfo { dtype, shape, data_offsets: (start, end) });
                }
            }
        }
    }

    Ok((8 + header_len, tensors))
}

fn bytes_to_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, CompressionError> {
    match dtype {
        "F32" => Ok(data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "F16" => Ok(data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        "BF16" => Ok(data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()),
        _ => Err(CompressionError::InvalidDtype(dtype.to_string())),
    }
}

fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &v in values {
        bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    bytes
}

fn quantize_int4(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * 2 + (weights.len() + 1) / 2);

    let mut scales = Vec::with_capacity(num_blocks);
    for block in weights.chunks(Q4_BLOCK_SIZE) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);
        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }

    let mut nibble_buffer = Vec::new();
    for (block_idx, block) in weights.chunks(Q4_BLOCK_SIZE).enumerate() {
        let scale = scales[block_idx];
        for &val in block {
            let q = ((val / scale).round() as i8).clamp(-8, 7);
            nibble_buffer.push((q + 8) as u8);
        }
    }

    for pair in nibble_buffer.chunks(2) {
        let byte = if pair.len() == 2 {
            (pair[0] & 0x0F) | ((pair[1] & 0x0F) << 4)
        } else {
            pair[0] & 0x0F
        };
        output.push(byte);
    }

    output
}

fn dequantize_int4(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = (num_elements + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let scales_bytes = num_blocks * 2;

    if data.len() < scales_bytes {
        return vec![];
    }

    let scales: Vec<f32> = data[..scales_bytes]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    let packed_data = &data[scales_bytes..];
    let mut output = Vec::with_capacity(num_elements);

    let mut nibble_idx = 0;
    for block_idx in 0..num_blocks {
        let scale = scales[block_idx];
        let block_size = Q4_BLOCK_SIZE.min(num_elements - block_idx * Q4_BLOCK_SIZE);

        for _ in 0..block_size {
            let byte_idx = nibble_idx / 2;
            let is_high = nibble_idx % 2 == 1;

            if byte_idx >= packed_data.len() {
                break;
            }

            let nibble = if is_high {
                (packed_data[byte_idx] >> 4) & 0x0F
            } else {
                packed_data[byte_idx] & 0x0F
            };

            let q = (nibble as i8) - 8;
            output.push(q as f32 * scale);
            nibble_idx += 1;
        }
    }

    output
}

fn compress_decompress(
    values: &[f32],
    width: usize,
    height: usize,
    encoder: &CompressiveSpectralEncoder,
    use_int4: bool,
) -> Result<Vec<f32>, String> {
    // For 1D tensors (bias vectors), preserve the mean explicitly
    // because DCT can lose DC component precision
    let is_1d = height == 1;
    let mean = if is_1d {
        values.iter().sum::<f32>() / values.len() as f32
    } else {
        0.0
    };

    // Center the data for 1D tensors
    let centered: Vec<f32> = if is_1d {
        values.iter().map(|v| v - mean).collect()
    } else {
        values.to_vec()
    };

    let mut fragments = encoder
        .encode_2d(&centered, width, height)
        .map_err(|e| format!("Encode failed: {:?}", e))?;

    if use_int4 {
        // Extract coefficients
        let frag0 = &fragments[0].data;
        let total_coeffs = u32::from_le_bytes([frag0[0], frag0[1], frag0[2], frag0[3]]) as usize;
        let essential_count = u32::from_le_bytes([frag0[4], frag0[5], frag0[6], frag0[7]]) as usize;

        let mut all_coeffs: Vec<f32> = Vec::with_capacity(total_coeffs);
        let essential_start = 20;

        for i in 0..essential_count {
            let offset = essential_start + i * 4;
            if offset + 4 <= frag0.len() {
                all_coeffs.push(f32::from_le_bytes([
                    frag0[offset], frag0[offset + 1], frag0[offset + 2], frag0[offset + 3]
                ]));
            }
        }

        for frag in &fragments[1..] {
            let data = &frag.data;
            if data.len() < 8 { continue; }
            let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            for i in 0..coeff_count {
                let offset = 8 + i * 4;
                if offset + 4 <= data.len() {
                    all_coeffs.push(f32::from_le_bytes([
                        data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
                    ]));
                }
            }
        }

        // INT4 round-trip
        let quantized = quantize_int4(&all_coeffs);
        let dequantized = dequantize_int4(&quantized, all_coeffs.len());

        // Put back
        let mut coeff_idx = 0;
        {
            let frag0 = &mut fragments[0].data;
            for i in 0..essential_count {
                if coeff_idx < dequantized.len() {
                    let offset = essential_start + i * 4;
                    let bytes = dequantized[coeff_idx].to_le_bytes();
                    if offset + 4 <= frag0.len() {
                        frag0[offset..offset+4].copy_from_slice(&bytes);
                    }
                    coeff_idx += 1;
                }
            }
        }

        for frag in &mut fragments[1..] {
            let data = &mut frag.data;
            if data.len() < 8 { continue; }
            let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            for i in 0..coeff_count {
                if coeff_idx < dequantized.len() {
                    let offset = 8 + i * 4;
                    let bytes = dequantized[coeff_idx].to_le_bytes();
                    if offset + 4 <= data.len() {
                        data[offset..offset+4].copy_from_slice(&bytes);
                    }
                    coeff_idx += 1;
                }
            }
        }
    }

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).map_err(|e| format!("{:?}", e))?;
    for frag in &fragments[1..] {
        decoder.add_detail(frag).map_err(|e| format!("{:?}", e))?;
    }
    let mut reconstructed = decoder.reconstruct().map_err(|e| format!("{:?}", e))?;

    // Add mean back for 1D tensors
    if is_1d {
        for v in &mut reconstructed {
            *v += mean;
        }
    }

    Ok(reconstructed)
}

/// Compress and decompress using adaptive per-tensor retention.
///
/// Returns (reconstructed_values, retention_used).
fn compress_decompress_adaptive(
    values: &[f32],
    width: usize,
    height: usize,
    target_quality: f32,
    use_int4: bool,
) -> Result<(Vec<f32>, f32), String> {
    // For 1D tensors (bias vectors), preserve the mean explicitly
    let is_1d = height == 1;
    let mean = if is_1d {
        values.iter().sum::<f32>() / values.len() as f32
    } else {
        0.0
    };

    // Center the data for 1D tensors
    let centered: Vec<f32> = if is_1d {
        values.iter().map(|v| v - mean).collect()
    } else {
        values.to_vec()
    };

    // Use adaptive encoder
    let encoder = AdaptiveSpectralEncoder::new(target_quality, 8);
    let (meta, mut fragments) = encoder
        .encode_2d(&centered, width, height)
        .map_err(|e| format!("Adaptive encode failed: {:?}", e))?;

    let retention_used = meta.retention_ratio;

    if use_int4 {
        // Extract and quantize coefficients (same as non-adaptive)
        let frag0 = &fragments[0].data;
        let total_coeffs = u32::from_le_bytes([frag0[0], frag0[1], frag0[2], frag0[3]]) as usize;
        let essential_count = u32::from_le_bytes([frag0[4], frag0[5], frag0[6], frag0[7]]) as usize;

        let mut all_coeffs: Vec<f32> = Vec::with_capacity(total_coeffs);
        let essential_start = 20;

        for i in 0..essential_count {
            let offset = essential_start + i * 4;
            if offset + 4 <= frag0.len() {
                all_coeffs.push(f32::from_le_bytes([
                    frag0[offset], frag0[offset + 1], frag0[offset + 2], frag0[offset + 3]
                ]));
            }
        }

        for frag in &fragments[1..] {
            let data = &frag.data;
            if data.len() < 8 { continue; }
            let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            for i in 0..coeff_count {
                let offset = 8 + i * 4;
                if offset + 4 <= data.len() {
                    all_coeffs.push(f32::from_le_bytes([
                        data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
                    ]));
                }
            }
        }

        // INT4 round-trip
        let quantized = quantize_int4(&all_coeffs);
        let dequantized = dequantize_int4(&quantized, all_coeffs.len());

        // Put back
        let mut coeff_idx = 0;
        {
            let frag0 = &mut fragments[0].data;
            for i in 0..essential_count {
                if coeff_idx < dequantized.len() {
                    let offset = essential_start + i * 4;
                    let bytes = dequantized[coeff_idx].to_le_bytes();
                    if offset + 4 <= frag0.len() {
                        frag0[offset..offset+4].copy_from_slice(&bytes);
                    }
                    coeff_idx += 1;
                }
            }
        }

        for frag in &mut fragments[1..] {
            let data = &mut frag.data;
            if data.len() < 8 { continue; }
            let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            for i in 0..coeff_count {
                if coeff_idx < dequantized.len() {
                    let offset = 8 + i * 4;
                    let bytes = dequantized[coeff_idx].to_le_bytes();
                    if offset + 4 <= data.len() {
                        data[offset..offset+4].copy_from_slice(&bytes);
                    }
                    coeff_idx += 1;
                }
            }
        }
    }

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).map_err(|e| format!("{:?}", e))?;
    for frag in &fragments[1..] {
        decoder.add_detail(frag).map_err(|e| format!("{:?}", e))?;
    }
    let mut reconstructed = decoder.reconstruct().map_err(|e| format!("{:?}", e))?;

    // Add mean back for 1D tensors
    if is_1d {
        for v in &mut reconstructed {
            *v += mean;
        }
    }

    Ok((reconstructed, retention_used))
}

/// Compress and decompress using mixed precision (FP16 essentials + INT4 details).
///
/// Uses a simpler DCT-based approach suitable for moderate tensor sizes.
fn compress_decompress_mixed_precision(
    values: &[f32],
    width: usize,
    height: usize,
    retention: f32,
    fp16_ratio: f32,
) -> Result<Vec<f32>, String> {
    // For 1D tensors (bias vectors), preserve the mean explicitly
    let is_1d = height == 1;
    let mean = if is_1d {
        values.iter().sum::<f32>() / values.len() as f32
    } else {
        0.0
    };

    // Center the data for 1D tensors
    let centered: Vec<f32> = if is_1d {
        values.iter().map(|v| v - mean).collect()
    } else {
        values.to_vec()
    };

    // Use mixed precision encoder
    let encoder = MixedPrecisionEncoder::new(retention, fp16_ratio);
    let decoder = MixedPrecisionDecoder::new();

    let compressed = encoder.encode(&centered, width, height)
        .map_err(|e| format!("MixedPrecision encode failed: {:?}", e))?;

    let mut reconstructed = decoder.decode(&compressed)
        .map_err(|e| format!("MixedPrecision decode failed: {:?}", e))?;

    // Add mean back for 1D tensors
    if is_1d {
        for v in &mut reconstructed {
            *v += mean;
        }
    }

    Ok(reconstructed)
}

/// Compress and decompress using importance-guided compression.
///
/// Uses heuristic layer-type importance or loaded importance map.
/// Returns (reconstructed_values, retention_used).
fn compress_decompress_importance(
    values: &[f32],
    width: usize,
    height: usize,
    tensor_name: &str,
    encoder: &ImportanceGuidedEncoder,
    use_int4: bool,
) -> Result<(Vec<f32>, f32), String> {
    // For 1D tensors (bias vectors), preserve the mean explicitly
    let is_1d = height == 1;
    let mean = if is_1d {
        values.iter().sum::<f32>() / values.len() as f32
    } else {
        0.0
    };

    // Center the data for 1D tensors
    let centered: Vec<f32> = if is_1d {
        values.iter().map(|v| v - mean).collect()
    } else {
        values.to_vec()
    };

    // Get tensor-specific retention from importance encoder
    let tensor_retention = encoder.effective_retention(tensor_name);

    // Encode using importance-guided encoder
    let compressed = encoder.encode(&centered, width, height, tensor_name)
        .map_err(|e| format!("Importance encode failed: {:?}", e))?;

    // Decode
    let decoder = ImportanceGuidedDecoder::new();
    let mut reconstructed = decoder.decode(&compressed)
        .map_err(|e| format!("Importance decode failed: {:?}", e))?;

    // If INT4 is enabled, apply additional quantization to the output
    // (importance-guided encoding uses its own quantization, but we can apply INT4 on top)
    if use_int4 {
        // Apply INT4 round-trip to reconstructed values
        let quantized = quantize_int4(&reconstructed);
        reconstructed = dequantize_int4(&quantized, reconstructed.len());
    }

    // Add mean back for 1D tensors
    if is_1d {
        for v in &mut reconstructed {
            *v += mean;
        }
    }

    Ok((reconstructed, tensor_retention))
}

fn find_model_in_cache(model_name: &str) -> Option<Vec<PathBuf>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/crook".to_string());
    let cache_dir = PathBuf::from(home).join(".cache/huggingface/hub");
    let model_dir_name = format!("models--{}", model_name.replace('/', "--"));
    let model_dir = cache_dir.join(&model_dir_name);

    if !model_dir.exists() {
        return None;
    }

    let snapshots = model_dir.join("snapshots");
    if !snapshots.exists() {
        return None;
    }

    // Find the first snapshot directory
    let mut snapshot_dir: Option<PathBuf> = None;
    if let Ok(entries) = fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                snapshot_dir = Some(path);
                break;
            }
        }
    }

    let snapshot = snapshot_dir?;

    // Find all safetensor files
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(&snapshot) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                // Skip index files
                if !path.file_name()
                    .map(|n| n.to_string_lossy().contains("index"))
                    .unwrap_or(false)
                {
                    files.push(path);
                }
            }
        }
    }

    // Sort files for consistent ordering (model-00001, model-00002, etc.)
    files.sort();

    if files.is_empty() { None } else { Some(files) }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\nError: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), CompressionError> {
    let base_retention: f32 = std::env::var("RETENTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);

    let use_int4 = std::env::var("NO_INT4").is_err();

    // Determine compression mode
    let mode = if std::env::var("IMPORTANCE").is_ok() {
        CompressionMode::Importance
    } else if std::env::var("MIXED_PRECISION").is_ok() {
        CompressionMode::MixedPrecision
    } else if std::env::var("ADAPTIVE").is_ok() {
        CompressionMode::Adaptive
    } else if std::env::var("LAYERTYPE").is_ok() {
        CompressionMode::LayerType
    } else if std::env::var("HYBRID").is_ok() {
        CompressionMode::Hybrid
    } else {
        CompressionMode::Uniform
    };

    // Importance map file (optional - defaults to heuristics)
    let importance_file = std::env::var("IMPORTANCE_FILE").ok();

    // FP16 ratio for mixed precision mode (fraction of retained coefficients stored as FP16)
    let fp16_ratio: f32 = std::env::var("FP16_RATIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.20); // Default 20% FP16 essentials

    // Target quality for adaptive mode
    let target_quality: f32 = std::env::var("TARGET_QUALITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.90);

    // Get model name from environment or use default
    let model_name = std::env::var("MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".to_string());

    // MAX_TENSORS limit for testing
    let max_tensors: Option<usize> = std::env::var("MAX_TENSORS")
        .ok()
        .and_then(|s| s.parse().ok());

    let input_paths = find_model_in_cache(&model_name)
        .ok_or_else(|| CompressionError::ModelNotFound(model_name.clone()))?;

    // Use first path for display purposes
    let display_path = input_paths.first()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mode_prefix = match mode {
        CompressionMode::Uniform => "",
        CompressionMode::Hybrid => "hybrid-",
        CompressionMode::LayerType => "layertype-",
        CompressionMode::Adaptive => "adaptive-",
        CompressionMode::MixedPrecision => "mixedprec-",
        CompressionMode::Importance => "importance-",
    };

    let output_path = PathBuf::from(format!(
        "/tmp/qwen-compressed-{}{:.0}pct{}.safetensors",
        mode_prefix,
        if mode == CompressionMode::Adaptive { target_quality * 100.0 } else { base_retention * 100.0 },
        if use_int4 && mode != CompressionMode::MixedPrecision && mode != CompressionMode::Importance { "-int4" }
        else if use_int4 && mode == CompressionMode::Importance { "-int4pp" }  // INT4 post-processing
        else { "" }
    ));

    println!("\n=== Creating Compressed Model ===\n");
    println!("Input:  {} ({} shard{})", display_path, input_paths.len(), if input_paths.len() > 1 { "s" } else { "" });
    println!("Output: {}", output_path.display());
    if let Some(max) = max_tensors {
        println!("Max tensors: {}", max);
    }
    println!("Base Retention: {:.0}%", base_retention * 100.0);
    println!("Mode: {:?}", mode);
    match mode {
        CompressionMode::Uniform => {
            println!("  - All layers: {:.0}%", base_retention * 100.0);
        }
        CompressionMode::Hybrid => {
            println!("  - Layers 0, 23: 100% (critical)");
            println!("  - Layers 1-2, 21-22: 80% (near-critical)");
            println!("  - Layers 3-20: {:.0}% (aggressive)", base_retention * 100.0);
            println!("  - Layernorms/biases: 100%");
        }
        CompressionMode::LayerType => {
            println!("  - LayerNorm/Bias/Embed: 100% (critical)");
            println!("  - lm_head: 90%");
            println!("  - Attention Q/K: {:.0}% (base)", base_retention * 100.0);
            println!("  - Attention V/O: {:.0}% (+10%)", (base_retention * 1.1).min(1.0) * 100.0);
            println!("  - FFN/MLP: {:.0}% (aggressive)", (base_retention * 0.70).max(0.40) * 100.0);
        }
        CompressionMode::Adaptive => {
            println!("  - Target quality: {:.0}%", target_quality * 100.0);
            println!("  - Per-tensor retention based on spectral energy analysis");
            println!("  - Optimizes each tensor independently for target quality");
        }
        CompressionMode::MixedPrecision => {
            println!("  - Retention: {:.0}%", base_retention * 100.0);
            println!("  - FP16 essentials: {:.0}% of retained coefficients", fp16_ratio * 100.0);
            println!("  - INT4 details: {:.0}% of retained coefficients", (1.0 - fp16_ratio) * 100.0);
            println!("  - Progressive decode: load FP16 first, then INT4");
        }
        CompressionMode::Importance => {
            println!("  - Base retention: {:.0}%", base_retention * 100.0);
            if let Some(ref file) = importance_file {
                println!("  - Importance map: {}", file);
            } else {
                println!("  - Importance: Heuristic (layer-type based)");
            }
            println!("  - Higher retention for: embeddings, v_proj, o_proj, LayerNorm");
            println!("  - Lower retention for: MLP/FFN, q_proj, k_proj");
        }
    }
    if mode != CompressionMode::MixedPrecision && mode != CompressionMode::Importance {
        println!("INT4: {}", use_int4);
    } else if mode == CompressionMode::Importance {
        println!("INT4 post-processing: {}", use_int4);
    }
    println!();

    let start = Instant::now();

    // Check if we should show progress bar
    let quiet = std::env::var("QUIET").is_ok();

    // Process all tensors and collect results
    let mut processed_tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut skipped = 0;
    let mut processed = 0;
    let mut retention_stats: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_input_size: u64 = 0;

    // Create importance encoder if in importance mode
    let importance_encoder = if mode == CompressionMode::Importance {
        let importance_map = if let Some(ref path) = importance_file {
            ImportanceMap::load(path).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to load importance map: {}. Using heuristics.", e);
                ImportanceMap::heuristic_only()
            })
        } else {
            ImportanceMap::heuristic_only()
        };
        Some(ImportanceGuidedEncoder::new(base_retention, importance_map))
    } else {
        None
    };

    // Count total tensors across all shards for progress bar
    let mut total_tensors = 0usize;
    for input_path in &input_paths {
        let data = fs::read(input_path)
            .map_err(|e| CompressionError::FileReadError(input_path.display().to_string(), e))?;
        let (_, tensors) = parse_safetensors_header(&data)
            .map_err(CompressionError::HeaderParseError)?;
        total_tensors += tensors.len();
        total_input_size += data.len() as u64;
    }

    let effective_total = max_tensors.map(|m| m.min(total_tensors)).unwrap_or(total_tensors);
    println!("Found {} tensors across {} shard(s){}\n",
        total_tensors,
        input_paths.len(),
        if max_tensors.is_some() { format!(" (processing {})", effective_total) } else { String::new() }
    );

    // Create progress bar
    let progress = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(effective_total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .expect("valid template")
                .progress_chars("=>-")
        );
        pb
    };

    // Track total processed across all shards
    let mut tensors_processed_total = 0usize;
    let max_tensor_limit = max_tensors.unwrap_or(usize::MAX);

    // Process each shard file
    for (shard_idx, input_path) in input_paths.iter().enumerate() {
        if tensors_processed_total >= max_tensor_limit {
            break;
        }

        if !quiet && input_paths.len() > 1 {
            progress.println(format!("Processing shard {}/{}: {}",
                shard_idx + 1, input_paths.len(),
                input_path.file_name().unwrap_or_default().to_string_lossy()));
        }

        let data = fs::read(input_path)
            .map_err(|e| CompressionError::FileReadError(input_path.display().to_string(), e))?;
        let (data_start, tensors) = parse_safetensors_header(&data)
            .map_err(CompressionError::HeaderParseError)?;

        // Sort tensors by name for consistent ordering
        let mut tensor_items: Vec<_> = tensors.iter().collect();
        tensor_items.sort_by(|a, b| a.0.cmp(b.0));

        for (name, info) in tensor_items {
            if tensors_processed_total >= max_tensor_limit {
                break;
            }

            let num_elements: usize = info.shape.iter().product();

            // Get original data
            let (start_off, end_off) = info.data_offsets;
            let tensor_data = &data[data_start + start_off..data_start + end_off];

            // Update progress bar with current tensor name
            progress.set_message(name.clone());

            if num_elements < 256 || num_elements > MAX_ELEMENTS {
                // Keep original (just convert to F16 for consistency)
                let values = bytes_to_f32(tensor_data, &info.dtype)?;
                let output_bytes = f32_to_f16_bytes(&values);
                processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                if num_elements > MAX_ELEMENTS && !quiet {
                    progress.println(format!("  SKIP {} (too large)", name));
                }
                skipped += 1;
                tensors_processed_total += 1;
                progress.inc(1);
                continue;
            }

            let values = bytes_to_f32(tensor_data, &info.dtype)?;
            if values.is_empty() {
                skipped += 1;
                tensors_processed_total += 1;
                progress.inc(1);
                continue;
            }

            let (width, height) = if info.shape.len() == 2 {
                (info.shape[1], info.shape[0])
            } else if info.shape.len() == 1 {
                (info.shape[0], 1)
            } else {
                (num_elements, 1)
            };

            // Handle different compression modes
            if mode == CompressionMode::Adaptive {
                match compress_decompress_adaptive(&values, width, height, target_quality, use_int4) {
                    Ok((reconstructed, retention_used)) => {
                        let retention_key = format!("{:.0}%", retention_used * 100.0);
                        *retention_stats.entry(retention_key).or_insert(0) += 1;
                        let output_bytes = f32_to_f16_bytes(&reconstructed);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        processed += 1;
                    }
                    Err(e) => {
                        if !quiet {
                            progress.println(format!("  FAIL {}: {}", name, e));
                        }
                        // Keep original on compression failure (graceful degradation)
                        let fallback_values = bytes_to_f32(tensor_data, &info.dtype)?;
                        let output_bytes = f32_to_f16_bytes(&fallback_values);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        skipped += 1;
                    }
                }
            } else if mode == CompressionMode::MixedPrecision {
                // Mixed precision: FP16 essentials + INT4 details
                // Limit tensor size to avoid slow DCT on large tensors
                if num_elements > 100_000 {
                    // Too large for mixed precision DCT - fall back to compressive spectral
                    let tensor_retention = get_layer_retention(name, base_retention, mode);
                    let encoder = CompressiveSpectralEncoder::new(8, tensor_retention);
                    match compress_decompress(&values, width, height, &encoder, use_int4) {
                        Ok(reconstructed) => {
                            let output_bytes = f32_to_f16_bytes(&reconstructed);
                            processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                            processed += 1;
                        }
                        Err(e) => {
                            if !quiet {
                                progress.println(format!("  FAIL {}: {}", name, e));
                            }
                            let fallback_values = bytes_to_f32(tensor_data, &info.dtype)?;
                            let output_bytes = f32_to_f16_bytes(&fallback_values);
                            processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                            skipped += 1;
                        }
                    }
                } else {
                    // Use mixed precision for smaller tensors
                    match compress_decompress_mixed_precision(&values, width, height, base_retention, fp16_ratio) {
                        Ok(reconstructed) => {
                            let retention_key = format!("{:.0}%", base_retention * 100.0);
                            *retention_stats.entry(retention_key).or_insert(0) += 1;
                            let output_bytes = f32_to_f16_bytes(&reconstructed);
                            processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                            processed += 1;
                        }
                        Err(e) => {
                            if !quiet {
                                progress.println(format!("  FAIL {}: {}", name, e));
                            }
                            // Keep original on compression failure (graceful degradation)
                            let fallback_values = bytes_to_f32(tensor_data, &info.dtype)?;
                            let output_bytes = f32_to_f16_bytes(&fallback_values);
                            processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                            skipped += 1;
                        }
                    }
                }
            } else if mode == CompressionMode::Importance {
                // Importance-guided compression
                let encoder = importance_encoder.as_ref().expect("importance_encoder should be initialized");
                match compress_decompress_importance(&values, width, height, name, encoder, use_int4) {
                    Ok((reconstructed, retention_used)) => {
                        let retention_key = format!("{:.0}%", retention_used * 100.0);
                        *retention_stats.entry(retention_key).or_insert(0) += 1;
                        let output_bytes = f32_to_f16_bytes(&reconstructed);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        processed += 1;
                    }
                    Err(e) => {
                        if !quiet {
                            progress.println(format!("  FAIL {}: {}", name, e));
                        }
                        // Keep original on compression failure (graceful degradation)
                        let fallback_values = bytes_to_f32(tensor_data, &info.dtype)?;
                        let output_bytes = f32_to_f16_bytes(&fallback_values);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        skipped += 1;
                    }
                }
            } else {
                // Other modes: use compressive spectral encoder
                let tensor_retention = get_layer_retention(name, base_retention, mode);
                let retention_key = format!("{:.0}%", tensor_retention * 100.0);
                *retention_stats.entry(retention_key).or_insert(0) += 1;

                let encoder = CompressiveSpectralEncoder::new(8, tensor_retention);

                match compress_decompress(&values, width, height, &encoder, use_int4) {
                    Ok(reconstructed) => {
                        let output_bytes = f32_to_f16_bytes(&reconstructed);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        processed += 1;
                    }
                    Err(e) => {
                        if !quiet {
                            progress.println(format!("  FAIL {}: {}", name, e));
                        }
                        // Keep original on compression failure (graceful degradation)
                        let fallback_values = bytes_to_f32(tensor_data, &info.dtype)?;
                        let output_bytes = f32_to_f16_bytes(&fallback_values);
                        processed_tensors.push((name.clone(), info.shape.clone(), output_bytes));
                        skipped += 1;
                    }
                }
            }
            tensors_processed_total += 1;
            progress.inc(1);
        }
    }

    progress.finish_with_message("done");
    println!("\nProcessed {} tensors, skipped {}", processed, skipped);

    if mode != CompressionMode::Uniform {
        println!("\nRetention distribution:");
        let mut stats: Vec<_> = retention_stats.into_iter().collect();
        stats.sort_by(|a, b| b.0.cmp(&a.0));
        for (retention, count) in stats {
            println!("  {}: {} tensors", retention, count);
        }
    }

    // Write safetensors file
    println!("\nWriting output...");

    // Build header
    let mut header_map = serde_json::Map::new();
    let mut current_offset: u64 = 0;

    for (name, shape, bytes) in &processed_tensors {
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert("dtype".to_string(), serde_json::Value::String("F16".to_string()));
        tensor_info.insert("shape".to_string(), serde_json::Value::Array(
            shape.iter().map(|&s| serde_json::Value::Number(s.into())).collect()
        ));
        let end_offset = current_offset + bytes.len() as u64;
        tensor_info.insert("data_offsets".to_string(), serde_json::Value::Array(vec![
            serde_json::Value::Number(current_offset.into()),
            serde_json::Value::Number(end_offset.into()),
        ]));
        header_map.insert(name.clone(), serde_json::Value::Object(tensor_info));
        current_offset = end_offset;
    }

    let header_json = serde_json::to_string(&serde_json::Value::Object(header_map))
        .map_err(|e| CompressionError::EncodingError(format!("JSON serialization failed: {}", e)))?;
    let header_bytes = header_json.as_bytes();

    // Pad header to 8-byte alignment
    let padding = (8 - (header_bytes.len() % 8)) % 8;
    let padded_header_len = header_bytes.len() + padding;

    let file = File::create(&output_path)
        .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;
    let mut writer = BufWriter::new(file);

    // Write header length
    writer.write_all(&(padded_header_len as u64).to_le_bytes())
        .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;
    // Write header
    writer.write_all(header_bytes)
        .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;
    // Write padding
    writer.write_all(&vec![0x20u8; padding])
        .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;

    // Write tensor data
    for (_, _, bytes) in &processed_tensors {
        writer.write_all(bytes)
            .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;
    }

    writer.flush()
        .map_err(|e| CompressionError::FileWriteError(output_path.display().to_string(), e))?;

    let elapsed = start.elapsed();
    let output_size = fs::metadata(&output_path)
        .map_err(|e| CompressionError::FileReadError(output_path.display().to_string(), e))?
        .len();

    println!("\n=== Done ===\n");
    println!("Time: {:.1}s", elapsed.as_secs_f64());
    println!("Input:  {:.2} MB", total_input_size as f64 / 1024.0 / 1024.0);
    println!("Output: {:.2} MB", output_size as f64 / 1024.0 / 1024.0);
    println!("\nCompressed model saved to: {}", output_path.display());
    println!("\nTest with Python:");
    println!("  python3 test_inference.py --original {} --compressed {}",
             display_path, output_path.display());

    Ok(())
}

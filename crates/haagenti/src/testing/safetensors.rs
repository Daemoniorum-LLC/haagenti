//! Safetensors file parsing utilities.
//!
//! Provides functions to parse safetensors model files:
//! - Header parsing to extract tensor metadata
//! - Dtype conversion (F32, F16, BF16) to f32
//! - HuggingFace cache model discovery

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Tensor metadata from safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type string (F32, F16, BF16, etc.)
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offsets (start, end) in the data section
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    /// Returns the number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the byte size based on dtype.
    pub fn byte_size(&self) -> usize {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// Parse safetensors file header to extract tensor metadata.
///
/// # Arguments
///
/// * `data` - Raw bytes of the safetensors file
///
/// # Returns
///
/// A tuple of (data_offset, tensors_map) where:
/// - `data_offset` is where tensor data starts (after header)
/// - `tensors_map` maps tensor names to their metadata
///
/// # Errors
///
/// Returns an error string if the header is invalid or malformed.
pub fn parse_safetensors_header(
    data: &[u8],
) -> Result<(usize, HashMap<String, TensorInfo>), String> {
    if data.len() < 8 {
        return Err("File too small for safetensors header".to_string());
    }

    let header_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if data.len() < 8 + header_len {
        return Err(format!(
            "Invalid header length: {} (file size: {})",
            header_len,
            data.len()
        ));
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_len])
        .map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

    let header: serde_json::Value =
        serde_json::from_str(header_json).map_err(|e| format!("Invalid JSON in header: {}", e))?;

    let mut tensors = HashMap::new();

    if let serde_json::Value::Object(obj) = header {
        for (name, value) in obj {
            // Skip metadata entry
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
                    let start = offs.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let end = offs.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    tensors.insert(
                        name,
                        TensorInfo {
                            dtype,
                            shape,
                            data_offsets: (start, end),
                        },
                    );
                }
            }
        }
    }

    Ok((8 + header_len, tensors))
}

/// Convert raw bytes to f32 values based on dtype.
///
/// # Arguments
///
/// * `data` - Raw byte data from safetensors file
/// * `dtype` - Data type string: "F32", "F16", or "BF16"
///
/// # Returns
///
/// Vector of f32 values. Returns empty vector for unsupported dtypes.
///
/// # Note
///
/// Requires the `half` crate for F16 support.
pub fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => {
            #[cfg(feature = "half")]
            {
                data.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect()
            }
            #[cfg(not(feature = "half"))]
            {
                // Fallback: manual F16 conversion
                data.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f16_to_f32_manual(bits)
                    })
                    .collect()
            }
        }
        "BF16" => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        _ => {
            // Unsupported dtype
            vec![]
        }
    }
}

/// Manual F16 to F32 conversion (when half crate unavailable).
#[allow(dead_code)]
fn f16_to_f32_manual(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let new_exp = ((e + 127) as u32) & 0xff;
            f32::from_bits((sign << 31) | (new_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
    } else {
        // Normal
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Find a model in the HuggingFace cache directory.
///
/// # Arguments
///
/// * `model_name` - Model identifier (e.g., "Qwen/Qwen2.5-0.5B")
///
/// # Returns
///
/// Path to `model.safetensors` if found, None otherwise.
///
/// # Search Locations
///
/// Searches in `~/.cache/huggingface/hub/models--{owner}--{name}/snapshots/*/model.safetensors`
pub fn find_model_in_cache(model_name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
    let cache_base = format!("{}/.cache/huggingface/hub", home);
    let pattern = format!("{}/models--{}", cache_base, model_name.replace('/', "--"));

    if let Ok(entries) = fs::read_dir(&pattern) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy() == "snapshots" {
                if let Ok(snapshots) = fs::read_dir(entry.path()) {
                    for snap in snapshots.flatten() {
                        // Try model.safetensors first
                        let model_path = snap.path().join("model.safetensors");
                        if model_path.exists() {
                            return Some(model_path);
                        }
                        // Also check for sharded models (model-00001-of-00002.safetensors)
                        if let Ok(files) = fs::read_dir(snap.path()) {
                            for file in files.flatten() {
                                let name = file.file_name().to_string_lossy().to_string();
                                if name.starts_with("model-") && name.ends_with(".safetensors") {
                                    return Some(file.path());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// List all safetensors files for a model (handles sharded models).
///
/// # Arguments
///
/// * `model_name` - Model identifier (e.g., "meta-llama/Llama-3.1-8B")
///
/// # Returns
///
/// Vector of paths to all safetensors files, sorted by name.
pub fn find_all_model_shards(model_name: &str) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
    let cache_base = format!("{}/.cache/huggingface/hub", home);
    let pattern = format!("{}/models--{}", cache_base, model_name.replace('/', "--"));

    let mut shards = Vec::new();

    if let Ok(entries) = fs::read_dir(&pattern) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy() == "snapshots" {
                if let Ok(snapshots) = fs::read_dir(entry.path()) {
                    for snap in snapshots.flatten() {
                        if let Ok(files) = fs::read_dir(snap.path()) {
                            for file in files.flatten() {
                                let name = file.file_name().to_string_lossy().to_string();
                                if name.ends_with(".safetensors") {
                                    shards.push(file.path());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    shards.sort();
    shards
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_f32_f32() {
        let val: f32 = 3.14159;
        let bytes = val.to_le_bytes();
        let result = bytes_to_f32(&bytes, "F32");
        assert_eq!(result.len(), 1);
        assert!((result[0] - val).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_bf16() {
        // BF16 of 1.0 is 0x3F80
        let bytes = [0x80, 0x3F];
        let result = bytes_to_f32(&bytes, "BF16");
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_unsupported() {
        let bytes = [0u8; 8];
        let result = bytes_to_f32(&bytes, "I8");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tensor_info_num_elements() {
        let info = TensorInfo {
            dtype: "F16".to_string(),
            shape: vec![2, 3, 4],
            data_offsets: (0, 48),
        };
        assert_eq!(info.num_elements(), 24);
    }

    #[test]
    fn test_parse_empty_file() {
        let result = parse_safetensors_header(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_f16_manual_conversion() {
        // 1.0 in F16 is 0x3C00
        let result = f16_to_f32_manual(0x3C00);
        assert!((result - 1.0).abs() < 1e-6);

        // 0.0 in F16
        let result = f16_to_f32_manual(0x0000);
        assert_eq!(result, 0.0);

        // -0.0 in F16
        let result = f16_to_f32_manual(0x8000);
        assert!(result.is_sign_negative() && result == 0.0);
    }
}

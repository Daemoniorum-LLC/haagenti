//! Full compression pipeline: Safetensors → CompressiveSpectral → INT4 → Zstd
//!
//! This combines three compression stages for maximum compression:
//! 1. Compressive spectral encoding: ~5x (at 10% retention, on original floats)
//! 2. INT4 quantization: ~4x (on spectral coefficients)
//! 3. Zstd compression: ~2x
//!
//! Target: 405B model from ~810GB (FP16) → ~20GB (~40x total)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example safetensors_to_compressive_hct --features="lz4,zstd" -- \
//!     --input ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

use haagenti::compressive::CompressiveSpectralEncoder;
use haagenti::{ZstdCodec, Codec, Compressor};

/// Block size for INT4 quantization.
const Q4_BLOCK_SIZE: usize = 32;

/// Tensor metadata from safetensors.
#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Compressed tensor output.
struct CompressedTensor {
    name: String,
    original_shape: Vec<usize>,
    original_elements: usize,
    quantized_bytes: usize,
    spectral_bytes: usize,
    final_bytes: usize,
}

/// Parse safetensors header.
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

    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| format!("Invalid JSON: {}", e))?;

    let mut tensors = HashMap::new();

    if let serde_json::Value::Object(obj) = header {
        for (name, value) in obj {
            if name == "__metadata__" {
                continue;
            }

            if let serde_json::Value::Object(tensor_obj) = value {
                let dtype = tensor_obj.get("dtype").and_then(|v| v.as_str())
                    .unwrap_or("F32").to_string();

                let shape: Vec<usize> = tensor_obj.get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
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

/// Convert bytes to f32 based on dtype.
fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        "BF16" => data.chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        _ => vec![],
    }
}

/// Quantize f32 weights to INT4 with per-block FP16 scaling.
/// Returns packed bytes: [scales_f16...][packed_int4...]
fn quantize_int4(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * 2 + (weights.len() + 1) / 2);

    // First pass: compute and store scales
    let mut scales = Vec::with_capacity(num_blocks);
    for block in weights.chunks(Q4_BLOCK_SIZE) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };
        scales.push(scale);
        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }

    // Second pass: quantize and pack nibbles
    let mut nibble_buffer = Vec::new();
    for (block_idx, block) in weights.chunks(Q4_BLOCK_SIZE).enumerate() {
        let scale = scales[block_idx];
        for &val in block {
            let q = ((val / scale).round() as i8).clamp(-8, 7);
            nibble_buffer.push((q + 8) as u8);
        }
    }

    // Pack nibbles into bytes
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

/// Find model in HuggingFace cache.
fn find_model_in_cache(model_name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/crook".to_string());
    let cache_base = format!("{}/.cache/huggingface/hub", home);
    let pattern = format!("{}/models--{}", cache_base, model_name.replace('/', "--"));

    if let Ok(entries) = fs::read_dir(&pattern) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy() == "snapshots" {
                if let Ok(snapshots) = fs::read_dir(entry.path()) {
                    for snap in snapshots.flatten() {
                        let model_path = snap.path().join("model.safetensors");
                        if model_path.exists() {
                            return Some(model_path);
                        }
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let input_path = if args.len() > 2 && args[1] == "--input" {
        PathBuf::from(&args[2])
    } else {
        // Try to find a model
        let models = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM2-135M",
            "meta-llama/Llama-3.2-1B",
        ];

        let mut found = None;
        for model in &models {
            if let Some(path) = find_model_in_cache(model) {
                println!("Found {} at: {}", model, path.display());
                found = Some(path);
                break;
            }
        }

        match found {
            Some(p) => p,
            None => {
                eprintln!("No model found. Usage: cargo run --example safetensors_to_compressive_hct -- --input <model.safetensors>");
                std::process::exit(1);
            }
        }
    };

    let retention_ratio = 0.10; // 10% retention = ~5x compression from spectral stage
    let num_fragments = 8u16;

    println!("\n=== Full Compression Pipeline ===\n");
    println!("Input: {}", input_path.display());
    println!("Pipeline: FP16/BF16 → Spectral@{}% (~5x) → INT4 (~4x) → Zstd (~2x)",
             (retention_ratio * 100.0) as u32);
    println!("Target: ~40x total compression\n");

    let start = Instant::now();

    // Read safetensors
    let data = fs::read(&input_path).expect("Failed to read file");
    let file_size = data.len();
    println!("File size: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);

    // Parse header
    let (data_start, tensors) = parse_safetensors_header(&data).expect("Failed to parse header");
    println!("Tensors: {}\n", tensors.len());

    // Initialize encoders
    let spectral_encoder = CompressiveSpectralEncoder::new(num_fragments, retention_ratio);
    let zstd = ZstdCodec::new();

    // Track statistics
    let mut total_original_bytes = 0usize;
    let mut total_quantized_bytes = 0usize;
    let mut total_spectral_bytes = 0usize;
    let mut total_final_bytes = 0usize;
    let mut tensor_count = 0usize;
    let mut results: Vec<CompressedTensor> = Vec::new();

    // Process each tensor
    println!("Processing tensors...");
    for (name, info) in tensors.iter() {
        let num_elements: usize = info.shape.iter().product();
        if num_elements < 256 {
            continue; // Skip tiny tensors
        }

        // Get tensor data
        let (start_off, end_off) = info.data_offsets;
        let tensor_data = &data[data_start + start_off..data_start + end_off];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let original_bytes = match info.dtype.as_str() {
            "F32" => num_elements * 4,
            _ => num_elements * 2, // F16/BF16
        };
        total_original_bytes += original_bytes;

        // Determine dimensions for 2D encoding
        let (width, height) = if info.shape.len() == 2 {
            (info.shape[1], info.shape[0])
        } else if info.shape.len() == 1 {
            (info.shape[0], 1)
        } else {
            // Flatten higher dimensions
            let total = values.len();
            (total, 1)
        };

        // Stage 1: Compressive spectral encoding on ORIGINAL float values
        // This is where the 5x compression happens (at 10% retention)
        let (spectral_data, spectral_bytes): (Vec<f32>, usize) = if let Ok(fragments) = spectral_encoder.encode_2d(&values, width, height) {
            let bytes: usize = fragments.iter().map(|f| f.data.len()).sum();
            total_spectral_bytes += bytes;
            // Collect all fragment data as f32 for next stage
            let data: Vec<f32> = fragments.iter()
                .flat_map(|f| f.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            (data, bytes)
        } else {
            // Fallback: use original values
            let bytes = values.len() * 4;
            total_spectral_bytes += bytes;
            (values.clone(), bytes)
        };

        // Stage 2: INT4 quantization on spectral coefficients
        let quantized = quantize_int4(&spectral_data);
        let quantized_bytes = quantized.len();
        total_quantized_bytes += quantized_bytes;

        // Stage 3: Zstd compression on the quantized spectral data
        let final_bytes = if let Ok(compressed) = zstd.compress(&quantized) {
            total_final_bytes += compressed.len();
            compressed.len()
        } else {
            total_final_bytes += quantized_bytes;
            quantized_bytes
        };

        results.push(CompressedTensor {
            name: name.clone(),
            original_shape: info.shape.clone(),
            original_elements: num_elements,
            quantized_bytes,
            spectral_bytes,
            final_bytes,
        });

        tensor_count += 1;
        if tensor_count % 50 == 0 {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }

    let elapsed = start.elapsed();
    println!("\n\nProcessed {} tensors in {:.2}s\n", tensor_count, elapsed.as_secs_f64());

    // Print results
    println!("=== Compression Results ===\n");
    println!("{:<20} {:>12} {:>10}", "Stage", "Size", "Ratio");
    println!("{}", "-".repeat(45));

    println!("{:<20} {:>12} {:>10}",
             "Original (FP16/BF16)",
             format!("{:.2} MB", total_original_bytes as f64 / 1024.0 / 1024.0),
             "1.00x");

    let spectral_ratio = total_original_bytes as f64 / total_spectral_bytes as f64;
    println!("{:<20} {:>12} {:>10}",
             "After Spectral@10%",
             format!("{:.2} MB", total_spectral_bytes as f64 / 1024.0 / 1024.0),
             format!("{:.1}x", spectral_ratio));

    let q4_ratio = total_original_bytes as f64 / total_quantized_bytes as f64;
    println!("{:<20} {:>12} {:>10}",
             "After INT4",
             format!("{:.2} MB", total_quantized_bytes as f64 / 1024.0 / 1024.0),
             format!("{:.1}x", q4_ratio));

    let final_ratio = total_original_bytes as f64 / total_final_bytes as f64;
    println!("{:<20} {:>12} {:>10}",
             "After Zstd",
             format!("{:.2} MB", total_final_bytes as f64 / 1024.0 / 1024.0),
             format!("{:.1}x", final_ratio));

    println!("\n=== Summary ===\n");
    println!("Input:  {:.2} MB ({} tensors)", total_original_bytes as f64 / 1024.0 / 1024.0, tensor_count);
    println!("Output: {:.2} MB", total_final_bytes as f64 / 1024.0 / 1024.0);
    println!("Total compression: {:.1}x", final_ratio);
    println!();

    // Projection for 405B
    let input_405b_gb = 810.0; // 405B params at FP16
    let projected_gb = input_405b_gb / final_ratio;
    println!("Projected for 405B model:");
    println!("  Input:  {:.0} GB (FP16)", input_405b_gb);
    println!("  Output: {:.1} GB", projected_gb);
    println!("  Target: 19 GB {}", if projected_gb <= 25.0 { "✓" } else { "✗" });
}

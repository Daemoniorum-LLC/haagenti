//! Test CompressiveSpectralEncoder on real model weights.
//!
//! Compares compression ratios between:
//! - Original SpectralEncoder (progressive streaming, storage expansion)
//! - CompressiveSpectralEncoder (storage compression, truncation-based)
//!
//! Usage:
//! ```bash
//! cargo run --example test_compressive_encoder --features="lz4,zstd" -- \
//!     --input ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/model.safetensors
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::holotensor::SpectralEncoder;

/// Tensor metadata from safetensors.
#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
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

/// Convert bytes to f32 based on dtype.
fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" | "BF16" => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                if dtype == "F16" {
                    half::f16::from_bits(bits).to_f32()
                } else {
                    half::bf16::from_bits(bits).to_f32()
                }
            })
            .collect(),
        _ => vec![],
    }
}

/// Calculate MSE between two slices.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

/// Calculate PSNR from MSE.
fn psnr_from_mse(mse: f32, max_val: f32) -> f32 {
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    10.0 * (max_val.powi(2) / mse).log10()
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

    // Find input file
    let input_path = if args.len() > 2 && args[1] == "--input" {
        PathBuf::from(&args[2])
    } else {
        // Try common models (in order of size for progressive testing)
        let models = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen3-0.6B",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "HuggingFaceTB/SmolLM2-135M",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
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
                eprintln!("No model found. Usage: cargo run --example test_compressive_encoder -- --input <model.safetensors>");
                std::process::exit(1);
            }
        }
    };

    println!("\n=== CompressiveSpectralEncoder Test ===\n");
    println!("Input: {}", input_path.display());

    // Read safetensors file
    let data = fs::read(&input_path).expect("Failed to read file");
    let file_size = data.len();
    println!("File size: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);

    // Parse header
    let (data_start, tensors) = parse_safetensors_header(&data).expect("Failed to parse header");
    println!("Tensors: {}\n", tensors.len());

    // Test different retention ratios
    let retention_ratios = [1.0, 0.5, 0.2, 0.1, 0.05];
    let num_fragments = 8u16;

    println!("Testing retention ratios: {:?}", retention_ratios);
    println!("Fragments: {}\n", num_fragments);

    // Track totals
    let mut total_input_bytes = 0usize;
    let mut total_spectral_bytes = 0usize;
    let mut total_compressive_bytes: Vec<usize> = vec![0; retention_ratios.len()];
    let mut tensor_count = 0usize;

    // Process each tensor
    for (name, info) in tensors.iter() {
        // Skip small tensors and non-weight tensors
        let num_elements: usize = info.shape.iter().product();
        if num_elements < 1024 {
            continue;
        }

        // Get tensor data
        let (start, end) = info.data_offsets;
        let tensor_data = &data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let input_bytes = values.len() * 4;
        total_input_bytes += input_bytes;
        tensor_count += 1;

        // Determine dimensions for 2D encoding
        let (width, height) = if info.shape.len() == 2 {
            (info.shape[1], info.shape[0])
        } else if info.shape.len() == 1 {
            (info.shape[0], 1)
        } else {
            // Flatten higher dimensions
            let total = values.len();
            let side = (total as f64).sqrt() as usize;
            if side * side == total {
                (side, side)
            } else {
                (total, 1)
            }
        };

        // Test original SpectralEncoder
        let spectral_encoder = SpectralEncoder::new(num_fragments);
        if let Ok(fragments) = spectral_encoder.encode_2d(&values, width, height) {
            let spectral_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();
            total_spectral_bytes += spectral_bytes;
        }

        // Test CompressiveSpectralEncoder with different retention ratios
        for (i, &ratio) in retention_ratios.iter().enumerate() {
            let compressive_encoder = CompressiveSpectralEncoder::new(num_fragments, ratio);
            if let Ok(fragments) = compressive_encoder.encode_2d(&values, width, height) {
                let compressive_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();
                total_compressive_bytes[i] += compressive_bytes;
            }
        }

        // Print progress every 20 tensors
        if tensor_count % 20 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!(
        "\n\nProcessed {} tensors ({:.2} MB input)\n",
        tensor_count,
        total_input_bytes as f64 / 1024.0 / 1024.0
    );

    // Print results
    println!("=== Compression Results ===\n");
    println!(
        "{:<25} {:>12} {:>12} {:>10}",
        "Method", "Output", "Ratio", "Note"
    );
    println!("{}", "-".repeat(65));

    println!(
        "{:<25} {:>12} {:>12} {:>10}",
        "Original (f32)",
        format!("{:.2} MB", total_input_bytes as f64 / 1024.0 / 1024.0),
        "1.00x",
        "baseline"
    );

    let spectral_ratio = total_input_bytes as f64 / total_spectral_bytes as f64;
    println!(
        "{:<25} {:>12} {:>12} {:>10}",
        "SpectralEncoder",
        format!("{:.2} MB", total_spectral_bytes as f64 / 1024.0 / 1024.0),
        format!("{:.2}x", spectral_ratio),
        if spectral_ratio < 1.0 {
            "EXPANSION"
        } else {
            ""
        }
    );

    for (i, &ratio) in retention_ratios.iter().enumerate() {
        let comp_ratio = total_input_bytes as f64 / total_compressive_bytes[i] as f64;
        let label = format!("Compressive({:.0}%)", ratio * 100.0);
        println!(
            "{:<25} {:>12} {:>12} {:>10}",
            label,
            format!(
                "{:.2} MB",
                total_compressive_bytes[i] as f64 / 1024.0 / 1024.0
            ),
            format!("{:.2}x", comp_ratio),
            if comp_ratio > 1.0 {
                "compression"
            } else {
                "expansion"
            }
        );
    }

    // Quality analysis on sample tensors
    println!("\n=== Quality Analysis (sample tensors) ===\n");

    // Collect tensors for quality analysis (reasonable size range for fast analysis)
    let mut sorted_tensors: Vec<_> = tensors
        .iter()
        .filter_map(|(name, info)| {
            let num_elements: usize = info.shape.iter().product();
            // Skip very small and very large tensors (too slow for quality analysis)
            if num_elements < 10_000 || num_elements > 1_000_000 {
                return None;
            }
            Some((name.clone(), info, num_elements))
        })
        .collect();
    // Sort by size descending to get largest tensors within the range
    sorted_tensors.sort_by_key(|(_, _, n)| std::cmp::Reverse(*n));

    // Analyze up to 3 tensors (already filtered to reasonable size range)
    let mut analyzed = 0;

    for (name, info, num_elements) in sorted_tensors.iter().take(3) {
        let (start, end) = info.data_offsets;
        let tensor_data = &data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let max_val = values.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        if max_val == 0.0 {
            continue;
        }

        let (width, height) = if info.shape.len() == 2 {
            (info.shape[1], info.shape[0])
        } else {
            (values.len(), 1)
        };

        if analyzed > 0 {
            println!();
        }
        println!(
            "Tensor: {} (shape: {:?}, {} elements)",
            name, info.shape, num_elements
        );
        println!("Max absolute value: {:.6}\n", max_val);

        println!(
            "{:<20} {:>12} {:>12} {:>12}",
            "Retention", "Output", "MSE", "PSNR"
        );
        println!("{}", "-".repeat(60));

        for &ratio in &retention_ratios {
            let encoder = CompressiveSpectralEncoder::new(num_fragments, ratio);
            if let Ok(fragments) = encoder.encode_2d(&values, width, height) {
                let output_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

                // Decode and measure quality
                let mut decoder = CompressiveSpectralDecoder::new();
                decoder.add_essentials(&fragments[0]).ok();
                for frag in &fragments[1..] {
                    decoder.add_detail(frag).ok();
                }

                if let Ok(reconstructed) = decoder.reconstruct() {
                    let error = mse(&values, &reconstructed);
                    let psnr = psnr_from_mse(error, max_val);

                    println!(
                        "{:<20} {:>12} {:>12} {:>12}",
                        format!("{:.0}%", ratio * 100.0),
                        format!("{:.2} KB", output_bytes as f64 / 1024.0),
                        format!("{:.2e}", error),
                        format!("{:.1} dB", psnr)
                    );
                }
            }
        }

        analyzed += 1;
    }

    println!("\n=== Summary ===\n");
    println!("SpectralEncoder: Designed for progressive streaming (any fragment works)");
    println!(
        "                 Results in {:.1}x storage EXPANSION",
        1.0 / spectral_ratio
    );
    println!();
    println!("CompressiveSpectralEncoder: Designed for storage compression");
    println!(
        "                            With 10% retention: {:.1}x COMPRESSION",
        total_input_bytes as f64 / total_compressive_bytes[3] as f64
    );
    println!();
    println!("For 405B → 19GB target, combine:");
    println!("  - 4-bit quantization: 8x");
    println!(
        "  - Compressive(10%): ~{:.1}x",
        total_input_bytes as f64 / total_compressive_bytes[3] as f64
    );
    println!("  - Zstd on coefficients: ~2x");
    println!(
        "  - Total: ~{:.0}x",
        8.0 * (total_input_bytes as f64 / total_compressive_bytes[3] as f64) * 2.0
    );
}

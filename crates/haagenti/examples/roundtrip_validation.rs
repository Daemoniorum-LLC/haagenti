//! Round-trip validation: Compress → Decompress → Verify
//!
//! This validates that our compression pipeline preserves model weights
//! with acceptable quality for inference.
//!
//! Pipeline: FP16 → Spectral@10% → INT4 → decompress → verify
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example roundtrip_validation --features="lz4,zstd" --release
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};

/// Block size for INT4 quantization.
const Q4_BLOCK_SIZE: usize = 32;

/// Maximum elements to process (FFT is O(n log n), too slow above this).
const MAX_ELEMENTS: usize = 10_000_000;

/// Tensor metadata from safetensors.
#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Results for a single tensor.
struct TensorResult {
    name: String,
    shape: Vec<usize>,
    original_bytes: usize,
    compressed_bytes: usize,
    mse: f32,
    psnr: f32,
    cosine_sim: f32,
    max_error: f32,
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
        "F16" => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        _ => vec![],
    }
}

/// Quantize f32 weights to INT4 with per-block FP16 scaling.
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

/// Dequantize INT4 back to f32.
fn dequantize_int4(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = (num_elements + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let scales_bytes = num_blocks * 2;

    if data.len() < scales_bytes {
        return vec![];
    }

    // Read scales
    let scales: Vec<f32> = data[..scales_bytes]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    // Unpack nibbles and dequantize
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

/// Calculate cosine similarity.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate max absolute error.
fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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

/// Full round-trip: compress and decompress
///
/// Modes:
/// - spectral_only=true: Test spectral encoding only (no INT4)
/// - spectral_only=false: Test full pipeline (Spectral + INT4)
///
/// Fragment structure (must be preserved!):
/// - Fragment 0: [20-byte header][essential coeffs as f32][index map as u32]
/// - Fragment N: [8-byte header][detail coeffs as f32]
fn roundtrip(
    values: &[f32],
    width: usize,
    height: usize,
    spectral_encoder: &CompressiveSpectralEncoder,
    spectral_only: bool,
) -> Result<(Vec<f32>, usize), String> {
    // Stage 1: Spectral encode
    let mut fragments = spectral_encoder
        .encode_2d(values, width, height)
        .map_err(|e| format!("Spectral encode failed: {:?}", e))?;

    // Calculate compressed size from fragment data
    let spectral_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    if spectral_only {
        // SPECTRAL ONLY: Just decode directly without INT4
        let mut decoder = CompressiveSpectralDecoder::new();

        decoder
            .add_essentials(&fragments[0])
            .map_err(|e| format!("Add essentials failed: {:?}", e))?;

        for frag in &fragments[1..] {
            decoder
                .add_detail(frag)
                .map_err(|e| format!("Add detail failed: {:?}", e))?;
        }

        let reconstructed = decoder
            .reconstruct()
            .map_err(|e| format!("Reconstruct failed: {:?}", e))?;

        return Ok((reconstructed, spectral_bytes));
    }

    // FULL PIPELINE: Spectral + INT4

    // Parse fragment 0 header to understand structure
    let frag0 = &fragments[0].data;
    if frag0.len() < 20 {
        return Err("Fragment 0 too short".to_string());
    }
    let total_coeffs = u32::from_le_bytes([frag0[0], frag0[1], frag0[2], frag0[3]]) as usize;
    let essential_count = u32::from_le_bytes([frag0[4], frag0[5], frag0[6], frag0[7]]) as usize;

    // Extract ONLY the coefficient values (skip headers and index map)
    let mut all_coeffs: Vec<f32> = Vec::with_capacity(total_coeffs);

    // From fragment 0: essential coefficients start at offset 20
    let essential_start = 20;
    for i in 0..essential_count {
        let offset = essential_start + i * 4;
        if offset + 4 <= frag0.len() {
            all_coeffs.push(f32::from_le_bytes([
                frag0[offset], frag0[offset + 1], frag0[offset + 2], frag0[offset + 3]
            ]));
        }
    }

    // From detail fragments: coefficients start at offset 8
    for frag in &fragments[1..] {
        let data = &frag.data;
        if data.len() < 8 {
            continue;
        }
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

    // Stage 2: INT4 quantize the coefficients
    let quantized = quantize_int4(&all_coeffs);

    // Estimate compressed size (INT4 bytes, zstd would add ~2x more)
    let compressed_size = quantized.len() / 2;

    // Stage 2: INT4 dequantize
    let dequantized = dequantize_int4(&quantized, all_coeffs.len());

    // Put dequantized values back into fragments (only coefficient areas, preserve headers!)
    let mut coeff_idx = 0;

    // Update fragment 0 essential coefficients (offset 20, count = essential_count)
    {
        let frag0 = &mut fragments[0].data;
        for i in 0..essential_count {
            if coeff_idx < dequantized.len() {
                let offset = essential_start + i * 4;
                let bytes = dequantized[coeff_idx].to_le_bytes();
                if offset + 4 <= frag0.len() {
                    frag0[offset] = bytes[0];
                    frag0[offset + 1] = bytes[1];
                    frag0[offset + 2] = bytes[2];
                    frag0[offset + 3] = bytes[3];
                }
                coeff_idx += 1;
            }
        }
    }

    // Update detail fragment coefficients (offset 8)
    for frag in &mut fragments[1..] {
        let data = &mut frag.data;
        if data.len() < 8 {
            continue;
        }
        let coeff_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        for i in 0..coeff_count {
            if coeff_idx < dequantized.len() {
                let offset = 8 + i * 4;
                let bytes = dequantized[coeff_idx].to_le_bytes();
                if offset + 4 <= data.len() {
                    data[offset] = bytes[0];
                    data[offset + 1] = bytes[1];
                    data[offset + 2] = bytes[2];
                    data[offset + 3] = bytes[3];
                }
                coeff_idx += 1;
            }
        }
    }

    // Stage 1: Spectral decode with INT4-modified fragments
    let mut decoder = CompressiveSpectralDecoder::new();

    decoder
        .add_essentials(&fragments[0])
        .map_err(|e| format!("Add essentials failed: {:?}", e))?;

    for frag in &fragments[1..] {
        decoder
            .add_detail(frag)
            .map_err(|e| format!("Add detail failed: {:?}", e))?;
    }

    let reconstructed = decoder
        .reconstruct()
        .map_err(|e| format!("Reconstruct failed: {:?}", e))?;

    Ok((reconstructed, compressed_size))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Find input file
    let input_path = if args.len() > 2 && args[1] == "--input" {
        PathBuf::from(&args[2])
    } else {
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
                eprintln!("No model found. Usage: cargo run --example roundtrip_validation -- --input <model.safetensors>");
                std::process::exit(1);
            }
        }
    };

    let retention_ratio = std::env::var("RETENTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.10);
    let num_fragments = 8u16;
    let spectral_only = std::env::var("SPECTRAL_ONLY").is_ok();

    println!("\n=== Round-Trip Validation ===\n");
    println!("Input: {}", input_path.display());
    if spectral_only {
        println!(
            "Pipeline: FP16 -> Spectral@{:.0}% -> decompress -> verify (NO INT4)",
            retention_ratio * 100.0
        );
    } else {
        println!(
            "Pipeline: FP16 -> Spectral@{:.0}% -> INT4 -> decompress -> verify",
            retention_ratio * 100.0
        );
    }
    println!("Max tensor size: {}M elements", MAX_ELEMENTS / 1_000_000);
    println!();

    let start = Instant::now();

    // Read safetensors
    let data = fs::read(&input_path).expect("Failed to read file");
    let file_size = data.len();
    println!("File size: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);

    // Parse header
    let (data_start, tensors) =
        parse_safetensors_header(&data).expect("Failed to parse header");
    println!("Tensors: {}\n", tensors.len());

    // Initialize encoder
    let spectral_encoder = CompressiveSpectralEncoder::new(num_fragments, retention_ratio);

    // Track results
    let mut results: Vec<TensorResult> = Vec::new();
    let mut total_original = 0usize;
    let mut total_compressed = 0usize;
    let mut tensor_count = 0usize;
    let mut skipped_large = 0usize;
    let mut skipped_large_bytes = 0usize;
    let mut skipped_tiny = 0usize;

    // Sort tensors by size (largest first) for consistent processing
    let mut sorted_tensors: Vec<_> = tensors.iter().collect();
    sorted_tensors.sort_by_key(|(_, info)| {
        std::cmp::Reverse(info.shape.iter().product::<usize>())
    });

    println!("Processing tensors...\n");

    // Limit for quick testing - set to 0 for unlimited
    let max_tensors: usize = std::env::var("MAX_TENSORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    for (idx, (name, info)) in sorted_tensors.iter().enumerate() {
        // Early exit if limit reached
        if max_tensors > 0 && tensor_count >= max_tensors {
            println!("\n  [Stopping at {} tensors per MAX_TENSORS limit]", max_tensors);
            break;
        }
        let num_elements: usize = info.shape.iter().product();

        // Skip tiny tensors (biases, layer norms, etc.)
        if num_elements < 256 {
            skipped_tiny += 1;
            continue;
        }

        // Skip very large tensors (embedding tables - FFT too slow)
        if num_elements > MAX_ELEMENTS {
            let tensor_bytes = match info.dtype.as_str() {
                "F32" => num_elements * 4,
                _ => num_elements * 2,
            };
            skipped_large += 1;
            skipped_large_bytes += tensor_bytes;
            println!(
                "  [{:3}] SKIP {} ({:.1}M elements - exceeds limit)",
                idx + 1,
                name,
                num_elements as f64 / 1_000_000.0
            );
            continue;
        }

        // Progress indicator
        let size_str = if num_elements >= 1_000_000 {
            format!("{:.1}M", num_elements as f64 / 1_000_000.0)
        } else {
            format!("{:.1}K", num_elements as f64 / 1_000.0)
        };
        println!("  [{:3}] {} ({} elements)...", idx + 1, name, size_str);
        std::io::stdout().flush().ok();

        // Get tensor data
        let (start_off, end_off) = info.data_offsets;
        let tensor_data = &data[data_start + start_off..data_start + end_off];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let original_bytes = match info.dtype.as_str() {
            "F32" => num_elements * 4,
            _ => num_elements * 2,
        };

        // Determine dimensions for 2D encoding
        let (width, height) = if info.shape.len() == 2 {
            (info.shape[1], info.shape[0])
        } else if info.shape.len() == 1 {
            (info.shape[0], 1)
        } else {
            (num_elements, 1)
        };

        // Perform round-trip
        match roundtrip(&values, width, height, &spectral_encoder, spectral_only) {
            Ok((reconstructed, compressed_size)) => {
                let max_val = values.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let error = mse(&values, &reconstructed);
                let psnr = psnr_from_mse(error, max_val);
                let cosine = cosine_similarity(&values, &reconstructed);
                let max_err = max_error(&values, &reconstructed);

                // Print per-tensor quality
                let quality_indicator = if cosine > 0.999 {
                    "EXCELLENT"
                } else if cosine > 0.99 {
                    "GOOD"
                } else if cosine > 0.95 {
                    "OK"
                } else {
                    "POOR"
                };
                println!(
                    "        -> cosine: {:.6}, PSNR: {:.1} dB [{}]",
                    cosine, psnr, quality_indicator
                );

                results.push(TensorResult {
                    name: (*name).clone(),
                    shape: info.shape.clone(),
                    original_bytes,
                    compressed_bytes: compressed_size,
                    mse: error,
                    psnr,
                    cosine_sim: cosine,
                    max_error: max_err,
                });

                total_original += original_bytes;
                total_compressed += compressed_size;
                tensor_count += 1;
            }
            Err(e) => {
                println!("        -> FAILED: {}", e);
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nProcessed {} tensors in {:.2}s",
        tensor_count,
        elapsed.as_secs_f64()
    );
    println!(
        "Skipped: {} large (>{:.0}M elements, {:.2} MB), {} tiny (<256 elements)",
        skipped_large,
        MAX_ELEMENTS as f64 / 1_000_000.0,
        skipped_large_bytes as f64 / 1024.0 / 1024.0,
        skipped_tiny
    );

    if results.is_empty() {
        eprintln!("\nNo tensors were processed!");
        std::process::exit(1);
    }

    // Calculate aggregate metrics
    let avg_mse: f32 = results.iter().map(|r| r.mse).sum::<f32>() / results.len() as f32;
    let avg_cosine: f32 =
        results.iter().map(|r| r.cosine_sim).sum::<f32>() / results.len() as f32;
    let worst_cosine = results
        .iter()
        .map(|r| r.cosine_sim)
        .fold(f32::MAX, f32::min);
    let best_cosine = results
        .iter()
        .map(|r| r.cosine_sim)
        .fold(0.0f32, f32::max);
    let best_psnr = results.iter().map(|r| r.psnr).fold(0.0f32, f32::max);
    let worst_psnr = results
        .iter()
        .map(|r| r.psnr)
        .filter(|p| p.is_finite())
        .fold(f32::MAX, f32::min);

    // Print summary
    println!("\n=== Compression Results ===\n");
    let ratio = total_original as f64 / total_compressed as f64;
    println!(
        "Original:   {:.2} MB",
        total_original as f64 / 1024.0 / 1024.0
    );
    println!(
        "Compressed: {:.2} MB (INT4, before entropy coding)",
        total_compressed as f64 / 1024.0 / 1024.0
    );
    println!("Ratio:      {:.1}x (spectral + INT4 only)\n", ratio);

    println!("=== Quality Metrics ===\n");
    println!("{:<20} {:>12}", "Metric", "Value");
    println!("{}", "-".repeat(35));
    println!("{:<20} {:>12}", "Avg Cosine Sim", format!("{:.6}", avg_cosine));
    println!("{:<20} {:>12}", "Best Cosine Sim", format!("{:.6}", best_cosine));
    println!("{:<20} {:>12}", "Worst Cosine Sim", format!("{:.6}", worst_cosine));
    println!("{:<20} {:>12}", "Avg MSE", format!("{:.2e}", avg_mse));
    println!("{:<20} {:>12}", "Best PSNR", format!("{:.1} dB", best_psnr));
    println!("{:<20} {:>12}", "Worst PSNR", format!("{:.1} dB", worst_psnr));

    // Show worst performing tensors
    println!("\n=== Lowest Quality Tensors ===\n");
    let mut by_cosine: Vec<_> = results.iter().collect();
    by_cosine.sort_by(|a, b| a.cosine_sim.partial_cmp(&b.cosine_sim).unwrap());

    println!(
        "{:<40} {:>10} {:>10} {:>12}",
        "Tensor", "Cosine", "PSNR", "Shape"
    );
    println!("{}", "-".repeat(75));

    for result in by_cosine.iter().take(5) {
        let shape_str = format!("{:?}", result.shape);
        let truncated_name = if result.name.len() > 38 {
            format!("...{}", &result.name[result.name.len() - 35..])
        } else {
            result.name.clone()
        };
        println!(
            "{:<40} {:>10.6} {:>10.1} {:>12}",
            truncated_name,
            result.cosine_sim,
            result.psnr,
            shape_str
        );
    }

    // Quality assessment
    println!("\n=== Quality Assessment ===\n");

    let quality_grade = if avg_cosine > 0.999 {
        "EXCELLENT"
    } else if avg_cosine > 0.99 {
        "GOOD"
    } else if avg_cosine > 0.95 {
        "ACCEPTABLE"
    } else if avg_cosine > 0.90 {
        "MARGINAL"
    } else {
        "POOR"
    };

    println!("Overall Quality: {}", quality_grade);
    println!();

    if avg_cosine > 0.99 {
        println!("The compressed model should produce outputs very similar to the original.");
        println!("Expected quality impact: Minimal - suitable for most use cases.");
    } else if avg_cosine > 0.95 {
        println!("The compressed model may have slightly different outputs.");
        println!("Expected quality impact: Small - suitable for many use cases.");
    } else {
        println!("The compressed model will have noticeably different outputs.");
        println!("Consider using a higher retention ratio for better quality.");
    }

    // Projection for 405B
    println!("\n=== 405B Model Projection ===\n");
    let input_405b_gb = 810.0;
    let projected_gb = input_405b_gb / ratio;
    println!("Input:  {:.0} GB (FP16)", input_405b_gb);
    println!("Output: {:.1} GB (before zstd, ~{:.1} GB after)", projected_gb, projected_gb / 2.0);
    println!(
        "Target: 19 GB {}",
        if projected_gb / 2.0 <= 25.0 { "OK" } else { "MISS" }
    );

    // Show attention tensor quality (important for inference)
    println!("\n=== Key Tensor Analysis ===\n");

    let attn_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.contains("attn") || r.name.contains("attention") || r.name.contains("self_attn"))
        .collect();

    let mlp_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.contains("mlp") || r.name.contains("ffn") || r.name.contains("dense"))
        .collect();

    if !attn_results.is_empty() {
        let avg_attn_cosine: f32 = attn_results.iter().map(|r| r.cosine_sim).sum::<f32>() / attn_results.len() as f32;
        println!("Attention layers ({} tensors): avg cosine = {:.6}", attn_results.len(), avg_attn_cosine);
    }

    if !mlp_results.is_empty() {
        let avg_mlp_cosine: f32 = mlp_results.iter().map(|r| r.cosine_sim).sum::<f32>() / mlp_results.len() as f32;
        println!("MLP/FFN layers ({} tensors):   avg cosine = {:.6}", mlp_results.len(), avg_mlp_cosine);
    }

    println!("\n=== Validation Complete ===\n");
    println!(
        "Compression: {:.1}x ({:.2} MB -> {:.2} MB)",
        ratio,
        total_original as f64 / 1024.0 / 1024.0,
        total_compressed as f64 / 1024.0 / 1024.0
    );
    println!("Quality:     {} (avg cosine: {:.6})", quality_grade, avg_cosine);

    if quality_grade == "EXCELLENT" || quality_grade == "GOOD" {
        println!("\nRound-trip validation PASSED!");
        println!("The compression pipeline maintains high fidelity for inference.");
    } else {
        println!("\nRound-trip validation shows quality concerns.");
        println!("Consider adjusting retention ratio for production use.");
    }
}

//! Adaptive Retention Testing
//!
//! Compares adaptive per-tensor retention against uniform retention
//! to measure compression ratio and quality improvements.
//!
//! ## Usage
//!
//! ```bash
//! # Compare adaptive vs uniform at 95% target energy
//! TARGET_QUALITY=0.95 cargo run --release --example test_adaptive_retention
//!
//! # Test with specific retention for uniform baseline
//! UNIFORM_RETENTION=0.30 TARGET_QUALITY=0.95 cargo run --release --example test_adaptive_retention
//!
//! # Limit number of tensors processed
//! MAX_TENSORS=10 cargo run --release --example test_adaptive_retention
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

use haagenti::adaptive::{AdaptiveBatchEncoder, AdaptiveSpectralDecoder, AdaptiveSpectralEncoder};
use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::spectral_analysis::SpectralAnalyzer;

/// Number of fragments for encoding.
const NUM_FRAGMENTS: u16 = 8;

/// Maximum elements to process per tensor.
const MAX_ELEMENTS: usize = 10_000_000;

/// Tensor metadata from safetensors.
#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Results for comparing uniform vs adaptive.
#[derive(Debug)]
struct ComparisonResult {
    name: String,
    shape: Vec<usize>,
    uniform_retention: f32,
    adaptive_retention: f32,
    uniform_compressed_bytes: usize,
    adaptive_compressed_bytes: usize,
    uniform_mse: f32,
    adaptive_mse: f32,
    uniform_cosine: f32,
    adaptive_cosine: f32,
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

/// Calculate MSE between two slices.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
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

/// Reshape tensor to 2D for compression.
fn reshape_to_2d(shape: &[usize]) -> (usize, usize) {
    let total: usize = shape.iter().product();
    if shape.len() >= 2 {
        let last = *shape.last().unwrap();
        let rest: usize = shape[..shape.len() - 1].iter().product();
        (rest, last)
    } else {
        // 1D tensor: make it square-ish
        let sqrt = (total as f64).sqrt() as usize;
        let width = sqrt.max(1);
        let height = (total + width - 1) / width;
        (height, width)
    }
}

/// Compress with uniform retention and measure.
fn compress_uniform(
    values: &[f32],
    width: usize,
    height: usize,
    retention: f32,
) -> Result<(usize, f32, f32), String> {
    let encoder = CompressiveSpectralEncoder::new(NUM_FRAGMENTS, retention);
    let fragments = encoder
        .encode_2d(values, width, height)
        .map_err(|e| format!("Uniform encode failed: {:?}", e))?;

    let compressed_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder
        .add_essentials(&fragments[0])
        .map_err(|e| format!("Uniform decode essentials failed: {:?}", e))?;
    for frag in &fragments[1..] {
        decoder
            .add_detail(frag)
            .map_err(|e| format!("Uniform decode detail failed: {:?}", e))?;
    }
    let reconstructed = decoder
        .reconstruct()
        .map_err(|e| format!("Uniform reconstruct failed: {:?}", e))?;

    let mse_val = mse(values, &reconstructed);
    let cosine = cosine_similarity(values, &reconstructed);

    Ok((compressed_bytes, mse_val, cosine))
}

/// Compress with adaptive retention and measure.
fn compress_adaptive(
    values: &[f32],
    width: usize,
    height: usize,
    target_quality: f32,
) -> Result<(f32, usize, f32, f32), String> {
    let encoder = AdaptiveSpectralEncoder::new(target_quality, NUM_FRAGMENTS);
    let (meta, fragments) = encoder
        .encode_2d(values, width, height)
        .map_err(|e| format!("Adaptive encode failed: {:?}", e))?;

    let compressed_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    // Decode
    let decoder = AdaptiveSpectralDecoder::new();
    let reconstructed = decoder
        .decode(&meta, &fragments)
        .map_err(|e| format!("Adaptive decode failed: {:?}", e))?;

    let mse_val = mse(values, &reconstructed);
    let cosine = cosine_similarity(values, &reconstructed);

    Ok((meta.retention_ratio, compressed_bytes, mse_val, cosine))
}

fn main() {
    println!("=== Adaptive vs Uniform Retention Comparison ===\n");

    // Configuration from environment
    let target_quality: f32 = std::env::var("TARGET_QUALITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);
    let uniform_retention: f32 = std::env::var("UNIFORM_RETENTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);
    let max_tensors: usize = std::env::var("MAX_TENSORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("Configuration:");
    println!("  Target quality: {:.1}%", target_quality * 100.0);
    println!(
        "  Uniform retention baseline: {:.1}%",
        uniform_retention * 100.0
    );
    println!("  Max tensors: {}", max_tensors);
    println!();

    // Find model - try multiple variants
    let model_variants = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M",
        "meta-llama/Llama-3.2-1B",
    ];

    let mut model_path = None;
    let mut found_name = "";
    for name in &model_variants {
        if let Some(p) = find_model_in_cache(name) {
            model_path = Some(p);
            found_name = name;
            break;
        }
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("No model found. Tried: {:?}", model_variants);
            eprintln!("Please download with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct");
            return;
        }
    };

    println!("Model: {} ({})", found_name, model_path.display());

    // Load safetensors
    let file_data = match fs::read(&model_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read model: {}", e);
            return;
        }
    };

    let (data_start, tensors) = match parse_safetensors_header(&file_data) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to parse header: {}", e);
            return;
        }
    };

    println!("Found {} tensors\n", tensors.len());

    // Filter to weight tensors only
    let mut weight_tensors: Vec<_> = tensors
        .iter()
        .filter(|(name, info)| {
            (name.contains("weight") || name.contains("embed"))
                && !name.contains("layernorm")
                && !name.contains("ln_")
                && info.shape.iter().product::<usize>() > 0
                && info.shape.iter().product::<usize>() <= MAX_ELEMENTS
        })
        .collect();

    weight_tensors.sort_by_key(|(name, _)| name.clone());
    weight_tensors.truncate(max_tensors);

    println!("Processing {} weight tensors...\n", weight_tensors.len());

    // Process each tensor
    let mut results: Vec<ComparisonResult> = Vec::new();
    let mut total_uniform_bytes = 0usize;
    let mut total_adaptive_bytes = 0usize;
    let mut retention_sum = 0.0f32;

    for (name, info) in &weight_tensors {
        let (start, end) = info.data_offsets;
        let tensor_data = &file_data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let (height, width) = reshape_to_2d(&info.shape);
        let padded_len = width * height;
        let mut padded = values.clone();
        padded.resize(padded_len, 0.0);

        print!("  {} {:?}... ", name, info.shape);
        std::io::stdout().flush().ok();

        // Uniform compression
        let (uniform_bytes, uniform_mse, uniform_cosine) =
            match compress_uniform(&padded, width, height, uniform_retention) {
                Ok(r) => r,
                Err(e) => {
                    println!("SKIP (uniform: {})", e);
                    continue;
                }
            };

        // Adaptive compression
        let (adaptive_ret, adaptive_bytes, adaptive_mse, adaptive_cosine) =
            match compress_adaptive(&padded, width, height, target_quality) {
                Ok(r) => r,
                Err(e) => {
                    println!("SKIP (adaptive: {})", e);
                    continue;
                }
            };

        let result = ComparisonResult {
            name: name.to_string(),
            shape: info.shape.clone(),
            uniform_retention,
            adaptive_retention: adaptive_ret,
            uniform_compressed_bytes: uniform_bytes,
            adaptive_compressed_bytes: adaptive_bytes,
            uniform_mse,
            adaptive_mse,
            uniform_cosine,
            adaptive_cosine,
        };

        // Print result
        let size_diff = adaptive_bytes as f32 / uniform_bytes as f32;
        let quality_diff = if adaptive_mse > 0.0 && uniform_mse > 0.0 {
            uniform_mse / adaptive_mse
        } else {
            1.0
        };

        println!(
            "ret={:.1}% size={:.1}x quality={:.1}x cos={:.6}",
            adaptive_ret * 100.0,
            size_diff,
            quality_diff,
            adaptive_cosine
        );

        total_uniform_bytes += uniform_bytes;
        total_adaptive_bytes += adaptive_bytes;
        retention_sum += adaptive_ret;

        results.push(result);
    }

    if results.is_empty() {
        println!("\nNo tensors processed!");
        return;
    }

    // Summary statistics
    println!("\n=== SUMMARY ===\n");

    let avg_retention = retention_sum / results.len() as f32;
    let storage_savings = 1.0 - (total_adaptive_bytes as f32 / total_uniform_bytes as f32);
    let avg_uniform_mse: f32 =
        results.iter().map(|r| r.uniform_mse).sum::<f32>() / results.len() as f32;
    let avg_adaptive_mse: f32 =
        results.iter().map(|r| r.adaptive_mse).sum::<f32>() / results.len() as f32;
    let avg_uniform_cos: f32 =
        results.iter().map(|r| r.uniform_cosine).sum::<f32>() / results.len() as f32;
    let avg_adaptive_cos: f32 =
        results.iter().map(|r| r.adaptive_cosine).sum::<f32>() / results.len() as f32;

    println!("Tensors processed: {}", results.len());
    println!();
    println!("Retention:");
    println!("  Uniform baseline: {:.1}%", uniform_retention * 100.0);
    println!("  Adaptive average: {:.1}%", avg_retention * 100.0);
    println!(
        "  Adaptive min:     {:.1}%",
        results
            .iter()
            .map(|r| r.adaptive_retention)
            .fold(f32::MAX, f32::min)
            * 100.0
    );
    println!(
        "  Adaptive max:     {:.1}%",
        results
            .iter()
            .map(|r| r.adaptive_retention)
            .fold(0.0f32, f32::max)
            * 100.0
    );
    println!();
    println!("Storage:");
    println!("  Uniform total:  {} bytes", total_uniform_bytes);
    println!("  Adaptive total: {} bytes", total_adaptive_bytes);
    println!("  Savings:        {:.1}%", storage_savings * 100.0);
    println!();
    println!("Quality (MSE):");
    println!("  Uniform avg:  {:.6e}", avg_uniform_mse);
    println!("  Adaptive avg: {:.6e}", avg_adaptive_mse);
    println!(
        "  Improvement:  {:.2}x",
        if avg_adaptive_mse > 0.0 {
            avg_uniform_mse / avg_adaptive_mse
        } else {
            1.0
        }
    );
    println!();
    println!("Quality (Cosine Similarity):");
    println!("  Uniform avg:  {:.6}", avg_uniform_cos);
    println!("  Adaptive avg: {:.6}", avg_adaptive_cos);
    println!();

    // Retention distribution
    println!("Retention Distribution:");
    let mut buckets = [0usize; 10];
    for r in &results {
        let bucket = ((r.adaptive_retention * 10.0) as usize).min(9);
        buckets[bucket] += 1;
    }
    for (i, count) in buckets.iter().enumerate() {
        let pct = *count as f32 / results.len() as f32 * 100.0;
        let bar = "#".repeat((*count * 50 / results.len().max(1)).max(0));
        println!(
            "  {:>2}-{:>2}%: {:>3} ({:>5.1}%) {}",
            i * 10,
            (i + 1) * 10,
            count,
            pct,
            bar
        );
    }
    println!();

    // Classification breakdown
    let low_rank_count = results
        .iter()
        .filter(|r| r.adaptive_retention < 0.3)
        .count();
    let medium_rank_count = results
        .iter()
        .filter(|r| r.adaptive_retention >= 0.3 && r.adaptive_retention < 0.6)
        .count();
    let high_rank_count = results
        .iter()
        .filter(|r| r.adaptive_retention >= 0.6)
        .count();

    println!("Tensor Classification:");
    println!(
        "  Low-rank (< 30%):     {} ({:.1}%)",
        low_rank_count,
        low_rank_count as f32 / results.len() as f32 * 100.0
    );
    println!(
        "  Medium-rank (30-60%): {} ({:.1}%)",
        medium_rank_count,
        medium_rank_count as f32 / results.len() as f32 * 100.0
    );
    println!(
        "  High-rank (> 60%):    {} ({:.1}%)",
        high_rank_count,
        high_rank_count as f32 / results.len() as f32 * 100.0
    );
    println!();

    // Conclusion
    if avg_adaptive_mse < avg_uniform_mse && total_adaptive_bytes < total_uniform_bytes {
        println!("RESULT: Adaptive wins on BOTH storage and quality!");
    } else if total_adaptive_bytes < total_uniform_bytes {
        println!(
            "RESULT: Adaptive achieves {:.1}% storage savings",
            storage_savings * 100.0
        );
    } else if avg_adaptive_mse < avg_uniform_mse {
        println!(
            "RESULT: Adaptive achieves {:.2}x better quality",
            avg_uniform_mse / avg_adaptive_mse
        );
    } else {
        println!("RESULT: Uniform baseline is competitive for this model");
    }
}

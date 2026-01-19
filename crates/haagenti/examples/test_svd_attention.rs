//! SVD vs DCT Comparison for Attention Weights
//!
//! Compares SVD-based compression against DCT-based compression
//! specifically for attention projection matrices (Q, K, V, O).
//!
//! ## Usage
//!
//! ```bash
//! # Compare at various compression ratios
//! cargo run --release --example test_svd_attention
//!
//! # Specify rank ratio for SVD
//! SVD_RANK=0.30 DCT_RETENTION=0.30 cargo run --release --example test_svd_attention
//!
//! # Limit tensors
//! MAX_TENSORS=20 cargo run --release --example test_svd_attention
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::Instant;

use haagenti::svd_compression::{SvdEncoder, SvdDecoder, mse, cosine_similarity};
use haagenti::compressive::{CompressiveSpectralEncoder, CompressiveSpectralDecoder};

/// Tensor metadata from safetensors.
#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Comparison results for a single tensor.
#[derive(Debug)]
struct ComparisonResult {
    name: String,
    shape: Vec<usize>,
    original_bytes: usize,
    // SVD results
    svd_compressed_bytes: usize,
    svd_mse: f32,
    svd_cosine: f32,
    svd_rank: usize,
    svd_time_us: u64,
    // DCT results
    dct_compressed_bytes: usize,
    dct_mse: f32,
    dct_cosine: f32,
    dct_time_us: u64,
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

/// Check if tensor is an attention projection weight.
fn is_attention_weight(name: &str) -> bool {
    name.contains("q_proj") || name.contains("k_proj") ||
    name.contains("v_proj") || name.contains("o_proj") ||
    name.contains(".wq.") || name.contains(".wk.") ||
    name.contains(".wv.") || name.contains(".wo.") ||
    name.contains("query") || name.contains("key") ||
    name.contains("value") || name.contains("qkv")
}

/// Check if tensor is an MLP/FFN weight.
fn is_mlp_weight(name: &str) -> bool {
    name.contains("mlp.") || name.contains("feed_forward") ||
    name.contains("ffn") || name.contains(".fc1") || name.contains(".fc2") ||
    name.contains("up_proj") || name.contains("down_proj") ||
    name.contains("gate_proj") || name.contains("w1.") ||
    name.contains("w2.") || name.contains("w3.")
}

/// Compress with SVD and measure.
fn compress_svd(
    values: &[f32],
    rows: usize,
    cols: usize,
    rank_ratio: f32,
) -> Result<(usize, f32, f32, usize, u64), String> {
    let start = Instant::now();

    let encoder = SvdEncoder::new(rank_ratio);
    let compressed = encoder
        .compress(values, rows, cols)
        .map_err(|e| format!("SVD encode failed: {:?}", e))?;

    let decoder = SvdDecoder::new();
    let reconstructed = decoder
        .decompress(&compressed)
        .map_err(|e| format!("SVD decode failed: {:?}", e))?;

    let elapsed = start.elapsed().as_micros() as u64;

    let mse_val = mse(values, &reconstructed);
    let cosine = cosine_similarity(values, &reconstructed);
    let compressed_bytes = compressed.storage_bytes();

    Ok((compressed_bytes, mse_val, cosine, compressed.rank, elapsed))
}

/// Compress with DCT (spectral) and measure.
fn compress_dct(
    values: &[f32],
    width: usize,
    height: usize,
    retention: f32,
) -> Result<(usize, f32, f32, u64), String> {
    let start = Instant::now();

    let encoder = CompressiveSpectralEncoder::new(8, retention);
    let fragments = encoder
        .encode_2d(values, width, height)
        .map_err(|e| format!("DCT encode failed: {:?}", e))?;

    let compressed_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    let mut decoder = CompressiveSpectralDecoder::new();
    decoder
        .add_essentials(&fragments[0])
        .map_err(|e| format!("DCT decode essentials failed: {:?}", e))?;
    for frag in &fragments[1..] {
        decoder
            .add_detail(frag)
            .map_err(|e| format!("DCT decode detail failed: {:?}", e))?;
    }
    let reconstructed = decoder
        .reconstruct()
        .map_err(|e| format!("DCT reconstruct failed: {:?}", e))?;

    let elapsed = start.elapsed().as_micros() as u64;

    let mse_val = mse(values, &reconstructed);
    let cosine = cosine_similarity(values, &reconstructed);

    Ok((compressed_bytes, mse_val, cosine, elapsed))
}

fn main() {
    println!("=== SVD vs DCT Compression Comparison ===\n");

    // Configuration
    let svd_rank: f32 = std::env::var("SVD_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);
    let dct_retention: f32 = std::env::var("DCT_RETENTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);
    let max_tensors: usize = std::env::var("MAX_TENSORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let max_elements: usize = std::env::var("MAX_ELEMENTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    println!("Configuration:");
    println!("  SVD rank ratio:   {:.1}%", svd_rank * 100.0);
    println!("  DCT retention:    {:.1}%", dct_retention * 100.0);
    println!("  Max tensors:      {}", max_tensors);
    println!("  Max elements:     {}", max_elements);
    println!();

    // Find model
    let model_variants = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
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
            return;
        }
    };

    println!("Model: {} ({})\n", found_name, model_path.display());

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

    // Filter to attention weights only
    let mut attention_tensors: Vec<_> = tensors
        .iter()
        .filter(|(name, info)| {
            is_attention_weight(name)
                && info.shape.len() == 2
                && info.shape.iter().product::<usize>() > 0
                && info.shape.iter().product::<usize>() <= max_elements
        })
        .collect();

    // Also collect MLP tensors for comparison
    let mut mlp_tensors: Vec<_> = tensors
        .iter()
        .filter(|(name, info)| {
            is_mlp_weight(name)
                && info.shape.len() == 2
                && info.shape.iter().product::<usize>() > 0
                && info.shape.iter().product::<usize>() <= max_elements
        })
        .collect();

    attention_tensors.sort_by_key(|(name, _)| name.to_string());
    mlp_tensors.sort_by_key(|(name, _)| name.to_string());

    attention_tensors.truncate(max_tensors);
    mlp_tensors.truncate(max_tensors);

    println!("Found {} attention tensors, {} MLP tensors\n",
             attention_tensors.len(), mlp_tensors.len());

    // Process attention tensors
    println!("=== ATTENTION TENSORS ===\n");

    let mut attn_results: Vec<ComparisonResult> = Vec::new();

    for (name, info) in &attention_tensors {
        let (start, end) = info.data_offsets;
        let tensor_data = &file_data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let rows = info.shape[0];
        let cols = info.shape[1];

        print!("  {} [{}, {}]... ", name, rows, cols);
        std::io::stdout().flush().ok();

        // SVD compression
        let (svd_bytes, svd_mse, svd_cos, svd_rank, svd_time) =
            match compress_svd(&values, rows, cols, svd_rank) {
                Ok(r) => r,
                Err(e) => {
                    println!("SVD FAIL: {}", e);
                    continue;
                }
            };

        // DCT compression (width, height swapped for row-major)
        let (dct_bytes, dct_mse, dct_cos, dct_time) =
            match compress_dct(&values, cols, rows, dct_retention) {
                Ok(r) => r,
                Err(e) => {
                    println!("DCT FAIL: {}", e);
                    continue;
                }
            };

        let original_bytes = values.len() * 4;
        let svd_ratio = original_bytes as f32 / svd_bytes as f32;
        let dct_ratio = original_bytes as f32 / dct_bytes as f32;

        // Determine winner
        let quality_winner = if svd_cos > dct_cos { "SVD" } else { "DCT" };
        let size_winner = if svd_bytes < dct_bytes { "SVD" } else { "DCT" };

        println!(
            "SVD: {:.4} ({:.1}x) | DCT: {:.4} ({:.1}x) | Q:{} S:{}",
            svd_cos, svd_ratio, dct_cos, dct_ratio, quality_winner, size_winner
        );

        attn_results.push(ComparisonResult {
            name: name.to_string(),
            shape: info.shape.clone(),
            original_bytes,
            svd_compressed_bytes: svd_bytes,
            svd_mse,
            svd_cosine: svd_cos,
            svd_rank,
            svd_time_us: svd_time,
            dct_compressed_bytes: dct_bytes,
            dct_mse,
            dct_cosine: dct_cos,
            dct_time_us: dct_time,
        });
    }

    // Process MLP tensors
    println!("\n=== MLP/FFN TENSORS ===\n");

    let mut mlp_results: Vec<ComparisonResult> = Vec::new();

    for (name, info) in &mlp_tensors {
        let (start, end) = info.data_offsets;
        let tensor_data = &file_data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);

        if values.is_empty() {
            continue;
        }

        let rows = info.shape[0];
        let cols = info.shape[1];

        print!("  {} [{}, {}]... ", name, rows, cols);
        std::io::stdout().flush().ok();

        // SVD compression
        let (svd_bytes, svd_mse, svd_cos, svd_rank, svd_time) =
            match compress_svd(&values, rows, cols, svd_rank) {
                Ok(r) => r,
                Err(e) => {
                    println!("SVD FAIL: {}", e);
                    continue;
                }
            };

        // DCT compression
        let (dct_bytes, dct_mse, dct_cos, dct_time) =
            match compress_dct(&values, cols, rows, dct_retention) {
                Ok(r) => r,
                Err(e) => {
                    println!("DCT FAIL: {}", e);
                    continue;
                }
            };

        let original_bytes = values.len() * 4;
        let svd_ratio = original_bytes as f32 / svd_bytes as f32;
        let dct_ratio = original_bytes as f32 / dct_bytes as f32;

        let quality_winner = if svd_cos > dct_cos { "SVD" } else { "DCT" };
        let size_winner = if svd_bytes < dct_bytes { "SVD" } else { "DCT" };

        println!(
            "SVD: {:.4} ({:.1}x) | DCT: {:.4} ({:.1}x) | Q:{} S:{}",
            svd_cos, svd_ratio, dct_cos, dct_ratio, quality_winner, size_winner
        );

        mlp_results.push(ComparisonResult {
            name: name.to_string(),
            shape: info.shape.clone(),
            original_bytes,
            svd_compressed_bytes: svd_bytes,
            svd_mse,
            svd_cosine: svd_cos,
            svd_rank,
            svd_time_us: svd_time,
            dct_compressed_bytes: dct_bytes,
            dct_mse,
            dct_cosine: dct_cos,
            dct_time_us: dct_time,
        });
    }

    // Summary
    println!("\n=== SUMMARY ===\n");

    if !attn_results.is_empty() {
        let attn_svd_wins_quality = attn_results.iter().filter(|r| r.svd_cosine > r.dct_cosine).count();
        let attn_svd_wins_size = attn_results.iter().filter(|r| r.svd_compressed_bytes < r.dct_compressed_bytes).count();
        let avg_attn_svd_cos: f32 = attn_results.iter().map(|r| r.svd_cosine).sum::<f32>() / attn_results.len() as f32;
        let avg_attn_dct_cos: f32 = attn_results.iter().map(|r| r.dct_cosine).sum::<f32>() / attn_results.len() as f32;
        let avg_attn_svd_mse: f32 = attn_results.iter().map(|r| r.svd_mse).sum::<f32>() / attn_results.len() as f32;
        let avg_attn_dct_mse: f32 = attn_results.iter().map(|r| r.dct_mse).sum::<f32>() / attn_results.len() as f32;
        let total_attn_svd_bytes: usize = attn_results.iter().map(|r| r.svd_compressed_bytes).sum();
        let total_attn_dct_bytes: usize = attn_results.iter().map(|r| r.dct_compressed_bytes).sum();
        let total_attn_original: usize = attn_results.iter().map(|r| r.original_bytes).sum();

        println!("ATTENTION TENSORS ({} total):", attn_results.len());
        println!("  Quality wins:  SVD {} vs DCT {}", attn_svd_wins_quality, attn_results.len() - attn_svd_wins_quality);
        println!("  Size wins:     SVD {} vs DCT {}", attn_svd_wins_size, attn_results.len() - attn_svd_wins_size);
        println!("  Avg cosine:    SVD {:.6} vs DCT {:.6}", avg_attn_svd_cos, avg_attn_dct_cos);
        println!("  Avg MSE:       SVD {:.6e} vs DCT {:.6e}", avg_attn_svd_mse, avg_attn_dct_mse);
        println!("  Total bytes:   SVD {} vs DCT {} (original {})",
                 total_attn_svd_bytes, total_attn_dct_bytes, total_attn_original);
        println!("  Compression:   SVD {:.2}x vs DCT {:.2}x",
                 total_attn_original as f32 / total_attn_svd_bytes as f32,
                 total_attn_original as f32 / total_attn_dct_bytes as f32);
        println!();
    }

    if !mlp_results.is_empty() {
        let mlp_svd_wins_quality = mlp_results.iter().filter(|r| r.svd_cosine > r.dct_cosine).count();
        let mlp_svd_wins_size = mlp_results.iter().filter(|r| r.svd_compressed_bytes < r.dct_compressed_bytes).count();
        let avg_mlp_svd_cos: f32 = mlp_results.iter().map(|r| r.svd_cosine).sum::<f32>() / mlp_results.len() as f32;
        let avg_mlp_dct_cos: f32 = mlp_results.iter().map(|r| r.dct_cosine).sum::<f32>() / mlp_results.len() as f32;
        let avg_mlp_svd_mse: f32 = mlp_results.iter().map(|r| r.svd_mse).sum::<f32>() / mlp_results.len() as f32;
        let avg_mlp_dct_mse: f32 = mlp_results.iter().map(|r| r.dct_mse).sum::<f32>() / mlp_results.len() as f32;
        let total_mlp_svd_bytes: usize = mlp_results.iter().map(|r| r.svd_compressed_bytes).sum();
        let total_mlp_dct_bytes: usize = mlp_results.iter().map(|r| r.dct_compressed_bytes).sum();
        let total_mlp_original: usize = mlp_results.iter().map(|r| r.original_bytes).sum();

        println!("MLP/FFN TENSORS ({} total):", mlp_results.len());
        println!("  Quality wins:  SVD {} vs DCT {}", mlp_svd_wins_quality, mlp_results.len() - mlp_svd_wins_quality);
        println!("  Size wins:     SVD {} vs DCT {}", mlp_svd_wins_size, mlp_results.len() - mlp_svd_wins_size);
        println!("  Avg cosine:    SVD {:.6} vs DCT {:.6}", avg_mlp_svd_cos, avg_mlp_dct_cos);
        println!("  Avg MSE:       SVD {:.6e} vs DCT {:.6e}", avg_mlp_svd_mse, avg_mlp_dct_mse);
        println!("  Total bytes:   SVD {} vs DCT {} (original {})",
                 total_mlp_svd_bytes, total_mlp_dct_bytes, total_mlp_original);
        println!("  Compression:   SVD {:.2}x vs DCT {:.2}x",
                 total_mlp_original as f32 / total_mlp_svd_bytes as f32,
                 total_mlp_original as f32 / total_mlp_dct_bytes as f32);
        println!();
    }

    // Final recommendation
    println!("=== RECOMMENDATION ===\n");

    if !attn_results.is_empty() && !mlp_results.is_empty() {
        let avg_attn_svd_cos: f32 = attn_results.iter().map(|r| r.svd_cosine).sum::<f32>() / attn_results.len() as f32;
        let avg_attn_dct_cos: f32 = attn_results.iter().map(|r| r.dct_cosine).sum::<f32>() / attn_results.len() as f32;
        let avg_mlp_svd_cos: f32 = mlp_results.iter().map(|r| r.svd_cosine).sum::<f32>() / mlp_results.len() as f32;
        let avg_mlp_dct_cos: f32 = mlp_results.iter().map(|r| r.dct_cosine).sum::<f32>() / mlp_results.len() as f32;

        if avg_attn_svd_cos > avg_attn_dct_cos {
            println!("Attention: SVD wins by {:.4} cosine similarity",
                     avg_attn_svd_cos - avg_attn_dct_cos);
        } else {
            println!("Attention: DCT wins by {:.4} cosine similarity",
                     avg_attn_dct_cos - avg_attn_svd_cos);
        }

        if avg_mlp_svd_cos > avg_mlp_dct_cos {
            println!("MLP/FFN:   SVD wins by {:.4} cosine similarity",
                     avg_mlp_svd_cos - avg_mlp_dct_cos);
        } else {
            println!("MLP/FFN:   DCT wins by {:.4} cosine similarity",
                     avg_mlp_dct_cos - avg_mlp_svd_cos);
        }

        println!();
        println!("Suggested hybrid strategy:");
        if avg_attn_svd_cos > avg_attn_dct_cos {
            println!("  - Use SVD for attention weights (Q, K, V, O projections)");
        } else {
            println!("  - Use DCT for attention weights (Q, K, V, O projections)");
        }
        if avg_mlp_svd_cos > avg_mlp_dct_cos {
            println!("  - Use SVD for MLP/FFN weights");
        } else {
            println!("  - Use DCT for MLP/FFN weights");
        }
    }
}

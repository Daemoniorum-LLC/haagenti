//! Importance-Guided Compression Test
//!
//! Compares importance-guided compression against uniform compression:
//! - Heuristic-based importance (layer type)
//! - File-loaded importance (pre-computed)
//! - Uniform retention (baseline)
//!
//! Usage:
//!   cargo run --release --example test_importance_compression
//!   MODEL=Qwen/Qwen2.5-0.5B-Instruct IMPORTANCE_FILE=model.importance.json cargo run --release --example test_importance_compression

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::importance::{
    cosine_similarity, mse, ImportanceGuidedDecoder, ImportanceGuidedEncoder, ImportanceMap,
    Sensitivity,
};

// ============================================================================
// Safetensors Utilities
// ============================================================================

fn find_model_in_cache(model_name: &str) -> Option<Vec<PathBuf>> {
    let home = env::var("HOME").ok()?;
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

    let mut snapshot_dir: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                snapshot_dir = Some(path);
            }
        }
    }

    let snapshot = snapshot_dir?;

    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&snapshot) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }

    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

fn parse_safetensors_header(data: &[u8]) -> Option<(u64, HashMap<String, TensorInfo>)> {
    if data.len() < 8 {
        return None;
    }

    let header_size = u64::from_le_bytes(data[0..8].try_into().ok()?);
    if header_size > 100_000_000 {
        return None;
    }

    let header_end = 8 + header_size as usize;
    if data.len() < header_end {
        return None;
    }

    let header_json = std::str::from_utf8(&data[8..header_end]).ok()?;
    let header: serde_json::Value = serde_json::from_str(header_json).ok()?;

    let obj = header.as_object()?;
    let mut tensors = HashMap::new();

    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }

        let info_obj = info.as_object()?;
        let dtype = info_obj.get("dtype")?.as_str()?;
        let shape: Vec<usize> = info_obj
            .get("shape")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let data_offsets: Vec<usize> = info_obj
            .get("data_offsets")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();

        if data_offsets.len() != 2 {
            continue;
        }

        tensors.insert(
            name.clone(),
            TensorInfo {
                dtype: dtype.to_string(),
                shape,
                data_offsets: (data_offsets[0], data_offsets[1]),
            },
        );
    }

    Some((header_size, tensors))
}

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

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
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        _ => vec![],
    }
}

// ============================================================================
// Test Results
// ============================================================================

#[derive(Debug)]
struct CompressionResult {
    name: String,
    original_bytes: usize,
    compressed_bytes: usize,
    ratio: f32,
    cosine_sim: f32,
    mse: f32,
    retention_used: f32,
}

impl CompressionResult {
    fn print(&self) {
        println!("  {} compression:", self.name);
        println!("    Retention: {:.1}%", self.retention_used * 100.0);
        println!(
            "    Compression ratio: {:.2}x ({} -> {} bytes)",
            self.ratio, self.original_bytes, self.compressed_bytes
        );
        println!("    Cosine similarity: {:.4}", self.cosine_sim);
        println!("    MSE: {:.6}", self.mse);
    }
}

// ============================================================================
// Compression Methods
// ============================================================================

fn test_importance_guided(
    data: &[f32],
    width: usize,
    height: usize,
    base_retention: f32,
    tensor_name: &str,
    importance_map: &ImportanceMap,
) -> CompressionResult {
    let encoder = ImportanceGuidedEncoder::new(base_retention, importance_map.clone());
    let decoder = ImportanceGuidedDecoder::new();

    let effective_ret = encoder.effective_retention(tensor_name);
    let compressed = encoder.encode(data, width, height, tensor_name).unwrap();
    let reconstructed = decoder.decode(&compressed).unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes = compressed.storage_bytes();

    CompressionResult {
        name: format!("Importance-Guided ({})", importance_map.source),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
        retention_used: effective_ret,
    }
}

fn test_uniform_compressive(
    data: &[f32],
    width: usize,
    height: usize,
    retention: f32,
) -> CompressionResult {
    let encoder = CompressiveSpectralEncoder::new(8, retention);

    let fragments = encoder.encode_2d(data, width, height).unwrap();

    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    CompressionResult {
        name: "Uniform Compressive".to_string(),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
        retention_used: retention,
    }
}

// ============================================================================
// Importance Analysis
// ============================================================================

fn analyze_importance_distribution(importance_map: &ImportanceMap, tensor_names: &[String]) {
    println!("\n=== Importance Distribution Analysis ===");

    let mut by_sensitivity: HashMap<String, Vec<String>> = HashMap::new();

    for name in tensor_names {
        let info = importance_map.get(name);
        let sens_name = format!("{:?}", info.sensitivity);
        by_sensitivity
            .entry(sens_name)
            .or_default()
            .push(name.clone());
    }

    println!("Tensors by sensitivity level:");
    for (sens, tensors) in &by_sensitivity {
        println!("  {}: {} tensors", sens, tensors.len());
        for t in tensors.iter().take(3) {
            let info = importance_map.get(t);
            println!("    - {} (importance: {:.2})", t, info.importance);
        }
        if tensors.len() > 3 {
            println!("    ... and {} more", tensors.len() - 3);
        }
    }
}

fn compare_retention_strategies(
    tensors: &[(String, Vec<f32>, usize, usize)],
    base_retention: f32,
    importance_map: &ImportanceMap,
) {
    println!("\n=== Retention Strategy Comparison ===");
    println!("Base retention: {:.0}%\n", base_retention * 100.0);

    let mut total_importance_quality = 0.0f32;
    let mut total_uniform_quality = 0.0f32;
    let mut total_importance_ratio = 0.0f32;
    let mut total_uniform_ratio = 0.0f32;
    let mut count = 0;

    for (name, data, width, height) in tensors {
        // Importance-guided
        let imp_result =
            test_importance_guided(data, *width, *height, base_retention, name, importance_map);

        // Uniform
        let uniform_result = test_uniform_compressive(data, *width, *height, base_retention);

        total_importance_quality += imp_result.cosine_sim;
        total_uniform_quality += uniform_result.cosine_sim;
        total_importance_ratio += imp_result.ratio;
        total_uniform_ratio += uniform_result.ratio;
        count += 1;

        // Show details for first few tensors
        if count <= 5 {
            println!("--- Tensor: {} ---", name);
            let info = importance_map.get(name);
            println!(
                "  Heuristic: importance={:.2}, sensitivity={:?}",
                info.importance, info.sensitivity
            );
            imp_result.print();
            uniform_result.print();

            let quality_diff = imp_result.cosine_sim - uniform_result.cosine_sim;
            println!(
                "  Quality difference: {:+.4} (positive = importance-guided better)",
                quality_diff
            );
            println!();
        }
    }

    let avg_imp_quality = total_importance_quality / count as f32;
    let avg_uniform_quality = total_uniform_quality / count as f32;
    let avg_imp_ratio = total_importance_ratio / count as f32;
    let avg_uniform_ratio = total_uniform_ratio / count as f32;

    println!("=== Summary over {} tensors ===", count);
    println!(
        "  Importance-Guided: avg quality={:.4}, avg ratio={:.2}x",
        avg_imp_quality, avg_imp_ratio
    );
    println!(
        "  Uniform:           avg quality={:.4}, avg ratio={:.2}x",
        avg_uniform_quality, avg_uniform_ratio
    );
    println!();
    println!(
        "  Quality difference: {:+.4} (positive = importance-guided better)",
        avg_imp_quality - avg_uniform_quality
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Importance-Guided Compression Test ===\n");

    let model_name = env::var("MODEL").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".to_string());
    let importance_file = env::var("IMPORTANCE_FILE").ok();
    let max_tensors: usize = env::var("MAX_TENSORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let base_retention: f32 = env::var("RETENTION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.50);

    println!("Model: {}", model_name);
    println!("Max tensors: {}", max_tensors);
    println!("Base retention: {:.0}%", base_retention * 100.0);

    // Load importance map
    let importance_map = if let Some(path) = &importance_file {
        println!("Importance file: {}", path);
        ImportanceMap::load_or_default(path)
    } else {
        println!("Using heuristic importance (no file specified)");
        ImportanceMap::heuristic_only()
    };

    // Find model
    let safetensor_files = match find_model_in_cache(&model_name) {
        Some(files) => files,
        None => {
            eprintln!("Model not found in cache: {}", model_name);
            eprintln!("Run: huggingface-cli download {}", model_name);

            println!("\nFalling back to synthetic tensor test...\n");
            run_synthetic_test(base_retention);
            return;
        }
    };

    println!("Found {} safetensors files\n", safetensor_files.len());

    // Collect tensors
    let mut tensors_data: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();
    let mut all_tensor_names: Vec<String> = Vec::new();

    for file_path in &safetensor_files {
        if tensors_data.len() >= max_tensors {
            break;
        }

        let mut file = match File::open(file_path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let mut data = Vec::new();
        if file.read_to_end(&mut data).is_err() {
            continue;
        }

        let (header_size, tensors) = match parse_safetensors_header(&data) {
            Some(h) => h,
            None => continue,
        };

        let data_offset = 8 + header_size as usize;

        for (name, info) in &tensors {
            all_tensor_names.push(name.clone());

            if tensors_data.len() >= max_tensors {
                break;
            }

            if info.shape.len() != 2 {
                continue;
            }
            let (height, width) = (info.shape[0], info.shape[1]);
            let n = width * height;

            // Limit to moderate tensor sizes for reasonable test time
            if n < 1024 || n > 100_000 {
                continue;
            }

            // Skip layernorm and biases (always full precision in real usage)
            if name.contains("layernorm") || name.contains("ln") || name.contains("bias") {
                continue;
            }

            let start = data_offset + info.data_offsets.0;
            let end = data_offset + info.data_offsets.1;
            if end > data.len() {
                continue;
            }

            let tensor_bytes = &data[start..end];
            let tensor_f32 = bytes_to_f32(tensor_bytes, &info.dtype);

            if tensor_f32.len() != n {
                continue;
            }

            tensors_data.push((name.clone(), tensor_f32, width, height));
        }
    }

    if tensors_data.is_empty() {
        println!("No suitable tensors found, running synthetic test instead...\n");
        run_synthetic_test(base_retention);
        return;
    }

    // Analyze importance distribution
    analyze_importance_distribution(&importance_map, &all_tensor_names);

    // Compare strategies
    compare_retention_strategies(&tensors_data, base_retention, &importance_map);
}

fn run_synthetic_test(base_retention: f32) {
    println!("=== Synthetic Tensor Test ===\n");

    let importance_map = ImportanceMap::heuristic_only();

    // Create test tensors with different "layer types"
    let width = 128;
    let height = 128;
    let n = width * height;

    let test_cases: Vec<(&str, Vec<f32>)> = vec![
        (
            "model.layers.0.mlp.gate_proj.weight",
            (0..n).map(|i| (i as f32 * 0.01).sin()).collect(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            (0..n).map(|i| (i as f32 * 0.02).cos()).collect(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            (0..n).map(|i| (i as f32 * 0.015).sin() * 0.5).collect(),
        ),
        (
            "model.embed_tokens.weight",
            (0..n)
                .map(|i| ((i as f32 * 1.618).sin() * 1000.0) % 1.0)
                .collect(),
        ),
    ];

    let tensors_data: Vec<(String, Vec<f32>, usize, usize)> = test_cases
        .into_iter()
        .map(|(name, data)| (name.to_string(), data, width, height))
        .collect();

    compare_retention_strategies(&tensors_data, base_retention, &importance_map);

    // Show expected behavior
    println!("\n=== Expected Behavior ===");
    println!("Layer type sensitivities:");
    for name in [
        "mlp.gate_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
        "embed_tokens",
    ] {
        let full_name = format!("model.layers.0.{}.weight", name);
        let info = importance_map.get(&full_name);
        println!(
            "  {}: {:?} (importance={:.2})",
            name, info.sensitivity, info.importance
        );
    }
    println!();
    println!("Expected: High-importance layers (embed, v_proj) get better quality");
    println!("          Low-importance layers (MLP) get more compression");
}

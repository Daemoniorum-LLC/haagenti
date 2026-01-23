//! Mixed Precision Compression Test
//!
//! Compares mixed precision (FP16 essentials + INT4 details) against:
//! - All-INT4 compression
//! - Uniform DCT compression
//! - Pure FP16 compression
//!
//! Usage:
//!   MODEL=Qwen/Qwen2.5-0.5B-Instruct cargo run --release --example test_mixed_precision

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::mixed_precision::{
    cosine_similarity, mse, MixedPrecisionDecoder, MixedPrecisionEncoder,
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

    // Find snapshot directory
    let snapshots = model_dir.join("snapshots");
    if !snapshots.exists() {
        return None;
    }

    // Get latest snapshot
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

    // Find safetensors files
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
}

impl CompressionResult {
    fn print(&self) {
        println!("  {} compression:", self.name);
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

fn test_mixed_precision(
    data: &[f32],
    width: usize,
    height: usize,
    retention: f32,
    fp16_ratio: f32,
) -> CompressionResult {
    let encoder = MixedPrecisionEncoder::new(retention, fp16_ratio);
    let decoder = MixedPrecisionDecoder::new();

    let compressed = encoder.encode(data, width, height).unwrap();
    let reconstructed = decoder.decode(&compressed).unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes = compressed.storage_bytes();

    CompressionResult {
        name: format!("Mixed Precision ({:.0}% FP16)", fp16_ratio * 100.0),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
    }
}

fn test_all_int4(data: &[f32], width: usize, height: usize, retention: f32) -> CompressionResult {
    // FP16 ratio = 0 means all INT4
    let encoder = MixedPrecisionEncoder::new(retention, 0.0);
    let decoder = MixedPrecisionDecoder::new();

    let compressed = encoder.encode(data, width, height).unwrap();
    let reconstructed = decoder.decode(&compressed).unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes = compressed.storage_bytes();

    CompressionResult {
        name: "All INT4".to_string(),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
    }
}

fn test_all_fp16(data: &[f32], width: usize, height: usize, retention: f32) -> CompressionResult {
    // FP16 ratio = 1.0 means all FP16
    let encoder = MixedPrecisionEncoder::new(retention, 1.0);
    let decoder = MixedPrecisionDecoder::new();

    let compressed = encoder.encode(data, width, height).unwrap();
    let reconstructed = decoder.decode(&compressed).unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes = compressed.storage_bytes();

    CompressionResult {
        name: "All FP16".to_string(),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
    }
}

fn test_compressive_spectral(
    data: &[f32],
    width: usize,
    height: usize,
    retention: f32,
) -> CompressionResult {
    // CompressiveSpectralEncoder takes num_fragments, not retention
    // Calculate fragments based on retention: more fragments = higher retention
    let num_fragments = ((retention * 10.0) as u16).clamp(2, 10);
    let encoder = CompressiveSpectralEncoder::new(num_fragments, retention);

    let fragments = encoder.encode_2d(data, width, height).unwrap();

    // Decode with all fragments
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    let original_bytes = data.len() * 4;
    let compressed_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();

    CompressionResult {
        name: "Compressive Spectral (INT4)".to_string(),
        original_bytes,
        compressed_bytes,
        ratio: original_bytes as f32 / compressed_bytes as f32,
        cosine_sim: cosine_similarity(data, &reconstructed),
        mse: mse(data, &reconstructed),
    }
}

// ============================================================================
// Progressive Decode Test
// ============================================================================

fn test_progressive_decode(
    data: &[f32],
    width: usize,
    height: usize,
    retention: f32,
    fp16_ratio: f32,
) {
    println!("\n=== Progressive Decoding Test ===");
    println!("Testing FP16 essentials-only vs full decode");

    let encoder = MixedPrecisionEncoder::new(retention, fp16_ratio);
    let decoder = MixedPrecisionDecoder::new();

    let compressed = encoder.encode(data, width, height).unwrap();

    // Essentials only
    let essentials = decoder.decode_essentials_only(&compressed).unwrap();
    let full = decoder.decode(&compressed).unwrap();

    let cos_essentials = cosine_similarity(data, &essentials);
    let cos_full = cosine_similarity(data, &full);

    let fp16_bytes = compressed.fp16_data.len();
    let int4_bytes = compressed.int4_data.len();
    let index_bytes = compressed.index_map.len() * 4;
    let total_bytes = compressed.storage_bytes();

    println!("  Storage breakdown:");
    println!(
        "    FP16 essentials: {} bytes ({:.1}%)",
        fp16_bytes,
        100.0 * fp16_bytes as f32 / total_bytes as f32
    );
    println!(
        "    INT4 details: {} bytes ({:.1}%)",
        int4_bytes,
        100.0 * int4_bytes as f32 / total_bytes as f32
    );
    println!(
        "    Index map: {} bytes ({:.1}%)",
        index_bytes,
        100.0 * index_bytes as f32 / total_bytes as f32
    );

    println!("\n  Quality comparison:");
    println!("    Essentials only (FP16): cosine = {:.4}", cos_essentials);
    println!("    Full decode (FP16+INT4): cosine = {:.4}", cos_full);
    println!(
        "    Improvement: +{:.2}%",
        100.0 * (cos_full - cos_essentials) / cos_essentials
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Mixed Precision Compression Test ===\n");

    // Default model
    let model_name = env::var("MODEL").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".to_string());
    let max_tensors: usize = env::var("MAX_TENSORS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let retention: f32 = env::var("RETENTION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.50);

    println!("Model: {}", model_name);
    println!("Max tensors: {}", max_tensors);
    println!("Retention: {:.0}%", retention * 100.0);

    // Find model
    let safetensor_files = match find_model_in_cache(&model_name) {
        Some(files) => files,
        None => {
            eprintln!("Model not found in cache: {}", model_name);
            eprintln!("Run: huggingface-cli download {}", model_name);

            // Fall back to synthetic test
            println!("\nFalling back to synthetic tensor test...\n");
            run_synthetic_test(retention);
            return;
        }
    };

    println!("Found {} safetensors files", safetensor_files.len());

    // Process each file
    let mut tensors_processed = 0;
    let mut total_results: Vec<Vec<CompressionResult>> = Vec::new();

    for file_path in &safetensor_files {
        if tensors_processed >= max_tensors {
            break;
        }

        let mut file = match File::open(file_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open {:?}: {}", file_path, e);
                continue;
            }
        };

        let mut data = Vec::new();
        if let Err(e) = file.read_to_end(&mut data) {
            eprintln!("Failed to read {:?}: {}", file_path, e);
            continue;
        }

        let (header_size, tensors) = match parse_safetensors_header(&data) {
            Some(h) => h,
            None => {
                eprintln!("Failed to parse header for {:?}", file_path);
                continue;
            }
        };

        let data_offset = 8 + header_size as usize;

        // Process suitable tensors
        for (name, info) in &tensors {
            if tensors_processed >= max_tensors {
                break;
            }

            // Skip small tensors and biases
            if info.shape.len() != 2 {
                continue;
            }
            let (height, width) = (info.shape[0], info.shape[1]);
            let n = width * height;
            // Limit to 256K elements max due to O(n²) DCT complexity
            if !(1024..=256_000).contains(&n) {
                continue;
            }

            // Skip layernorm and biases
            if name.contains("layernorm") || name.contains("ln") || name.contains("bias") {
                continue;
            }

            // Extract tensor data
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

            println!("\n--- Tensor: {} ---", name);
            println!("Shape: {:?}, DType: {}", info.shape, info.dtype);

            // Run compression tests
            let results = vec![
                // Mixed precision with different FP16 ratios
                test_mixed_precision(&tensor_f32, width, height, retention, 0.10),
                test_mixed_precision(&tensor_f32, width, height, retention, 0.20),
                test_mixed_precision(&tensor_f32, width, height, retention, 0.30),
                // Baselines
                test_all_int4(&tensor_f32, width, height, retention),
                test_all_fp16(&tensor_f32, width, height, retention),
                test_compressive_spectral(&tensor_f32, width, height, retention),
            ];

            for r in &results {
                r.print();
            }

            // Progressive decode test for first tensor
            if tensors_processed == 0 {
                test_progressive_decode(&tensor_f32, width, height, retention, 0.20);
            }

            total_results.push(results);
            tensors_processed += 1;
        }
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Processed {} tensors", tensors_processed);

    if !total_results.is_empty() {
        // Calculate averages by method
        let num_methods = total_results[0].len();
        for method_idx in 0..num_methods {
            let avg_ratio: f32 = total_results
                .iter()
                .map(|r| r[method_idx].ratio)
                .sum::<f32>()
                / total_results.len() as f32;
            let avg_cos: f32 = total_results
                .iter()
                .map(|r| r[method_idx].cosine_sim)
                .sum::<f32>()
                / total_results.len() as f32;

            println!(
                "  {}: avg ratio {:.2}x, avg cosine {:.4}",
                total_results[0][method_idx].name, avg_ratio, avg_cos
            );
        }
    }
}

fn run_synthetic_test(retention: f32) {
    println!("=== Synthetic Tensor Test ===\n");

    let width = 512;
    let height = 512;
    let n = width * height;

    // Generate various test patterns
    let patterns: Vec<(&str, Vec<f32>)> = vec![
        (
            "Sine pattern",
            (0..n).map(|i| (i as f32 * 0.01).sin()).collect(),
        ),
        (
            "Random-like",
            (0..n)
                .map(|i| ((i as f32 * 1.618).sin() * 1000.0) % 1.0)
                .collect(),
        ),
        (
            "Low frequency",
            (0..n).map(|i| (i as f32 * 0.001).cos() * 0.5).collect(),
        ),
        ("Sparse", {
            let mut v = vec![0.0f32; n];
            for i in (0..n).step_by(100) {
                v[i] = 1.0;
            }
            v
        }),
    ];

    for (pattern_name, tensor) in patterns {
        println!("\n--- Pattern: {} ---", pattern_name);
        println!("Shape: {}x{}", width, height);

        // Run tests
        let results = vec![
            test_mixed_precision(&tensor, width, height, retention, 0.10),
            test_mixed_precision(&tensor, width, height, retention, 0.20),
            test_mixed_precision(&tensor, width, height, retention, 0.30),
            test_all_int4(&tensor, width, height, retention),
            test_all_fp16(&tensor, width, height, retention),
            test_compressive_spectral(&tensor, width, height, retention),
        ];

        for r in &results {
            r.print();
        }

        // Progressive decode
        test_progressive_decode(&tensor, width, height, retention, 0.20);
    }
}

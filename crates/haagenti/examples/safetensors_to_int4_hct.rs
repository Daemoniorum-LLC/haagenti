//! Convert safetensors model weights to INT4-quantized HCT format.
//!
//! This tool reads BF16/FP16 safetensors and quantizes to INT4 with per-block
//! FP16 scaling, producing compressed .hct files for Infernum inference.
//!
//! ## INT4 Format Specification
//!
//! - **Block size**: 32 elements per FP16 scale factor
//! - **Layout**: [FP16 scales][packed INT4 nibbles] - scales first
//! - **Quantization**: Symmetric, centered at 8
//!   - nibble range 0-15 maps to values -8 to +7
//!   - dequantize: value = (nibble - 8) * scale
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example safetensors_to_int4_hct --features="lz4" -- \
//!     --input-dir /path/to/model \
//!     --output-dir /path/to/output
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::time::Instant;

use haagenti::tensor::{CompressionAlgorithm, DType, HctWriter};
use haagenti::Lz4Compressor;
use haagenti_core::CompressionLevel;

/// Block size for INT4 quantization (matches Infernum's expected format).
const Q4_BLOCK_SIZE: usize = 32;

/// Tensor metadata from safetensors.
#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Parse safetensors header to get tensor metadata.
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
                    .ok_or_else(|| format!("Missing dtype for {}", name))?
                    .to_string();

                let shape: Vec<usize> = tensor_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Missing shape for {}", name))?
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();

                let offsets = tensor_obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Missing offsets for {}", name))?;

                let start = offsets[0].as_u64().ok_or("Invalid offset")? as usize;
                let end = offsets[1].as_u64().ok_or("Invalid offset")? as usize;

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

    Ok((8 + header_len, tensors))
}

/// Convert BF16 bytes to f32 values.
fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // BF16 to F32: just shift left by 16 bits
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Convert F16 bytes to f32 values.
fn f16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert f32 to F16 bytes.
fn f32_to_f16_bytes(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_le_bytes()
}

/// Quantize f32 weights to INT4 with per-block FP16 scaling.
///
/// Uses symmetric quantization centered at 8:
/// - nibble range 0-15 maps to -8 to +7
/// - scale = max(abs(block)) / 7.0
/// - quantized = round(value / scale) + 8
///
/// Returns (scales_f16_bytes, packed_int4_bytes).
fn quantize_int4_symmetric(weights: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let num_blocks = (weights.len() + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    let mut scales_bytes = Vec::with_capacity(num_blocks * 2);
    let mut packed = Vec::with_capacity((weights.len() + 1) / 2);

    for block in weights.chunks(Q4_BLOCK_SIZE) {
        // Find max absolute value for symmetric quantization
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Compute scale: we map [-max_abs, +max_abs] to [-8, +7] (nibbles 0-15)
        // So scale = max_abs / 7.0 (we use 7 since +7 is the max positive)
        let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };

        // Store scale as FP16
        scales_bytes.extend_from_slice(&f32_to_f16_bytes(scale));

        // Quantize block values using symmetric formula:
        // q = round(value / scale) + 8, clamped to [0, 15]
        let quantized: Vec<u8> = block
            .iter()
            .map(|&w| {
                if scale > 1e-10 {
                    let q = (w / scale).round() as i32 + 8;
                    q.clamp(0, 15) as u8
                } else {
                    8 // Zero maps to center (8)
                }
            })
            .collect();

        // Pack 2 INT4 values per byte (low nibble first)
        for chunk in quantized.chunks(2) {
            let low = chunk[0] & 0x0F;
            let high = if chunk.len() > 1 {
                (chunk[1] & 0x0F) << 4
            } else {
                0
            };
            packed.push(low | high);
        }
    }

    (scales_bytes, packed)
}

/// Statistics from conversion.
#[derive(Debug, Default)]
struct ConversionStats {
    total_tensors: usize,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    elapsed_ms: u64,
}

fn convert_safetensors_to_int4(input: &Path, output_dir: &Path) -> Result<ConversionStats, String> {
    let start = Instant::now();

    let data = fs::read(input).map_err(|e| format!("Failed to read: {}", e))?;
    println!(
        "Read {} ({:.2} MB)",
        input.display(),
        data.len() as f64 / 1_000_000.0
    );

    let (data_start, tensors) = parse_safetensors_header(&data)?;
    println!("Found {} tensors", tensors.len());

    let compressor = Lz4Compressor::with_level(CompressionLevel::Fast);
    let mut stats = ConversionStats::default();

    for (name, info) in &tensors {
        let tensor_data = &data[data_start + info.data_offsets.0..data_start + info.data_offsets.1];

        // Convert to f32
        let f32_weights = match info.dtype.as_str() {
            "BF16" => bf16_to_f32(tensor_data),
            "F16" => f16_to_f32(tensor_data),
            "F32" => tensor_data
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => {
                println!("  Skipping {} (unsupported dtype: {})", name, other);
                continue;
            }
        };

        // Verify element count matches shape
        let expected_elements: usize = info.shape.iter().product();
        if f32_weights.len() != expected_elements {
            println!(
                "  Warning: {} element count mismatch: {} vs expected {}",
                name,
                f32_weights.len(),
                expected_elements
            );
        }

        // Quantize to INT4 with symmetric scaling
        let (scales_bytes, packed_int4) = quantize_int4_symmetric(&f32_weights);

        // Create output file
        let safe_name = name.replace('/', "_").replace('.', "_");
        let output_path = output_dir.join(format!("{}.hct", safe_name));

        // Shape remains the same as original (element count, not packed)
        let shape: Vec<u64> = info.shape.iter().map(|&s| s as u64).collect();

        let output_file =
            File::create(&output_path).map_err(|e| format!("Failed to create file: {}", e))?;

        // Write as I4 HCT
        let mut writer = HctWriter::new(output_file, CompressionAlgorithm::Lz4, DType::I4, shape)
            .with_block_size(16384);

        // Data layout: [FP16 scales][packed INT4 data]
        let mut quant_data = Vec::with_capacity(scales_bytes.len() + packed_int4.len());
        quant_data.extend_from_slice(&scales_bytes);
        quant_data.extend_from_slice(&packed_int4);

        writer
            .compress_data(&quant_data, &compressor)
            .map_err(|e| format!("Compression failed: {}", e))?;
        writer
            .finish()
            .map_err(|e| format!("Failed to finish: {}", e))?;

        let compressed_size = fs::metadata(&output_path)
            .map_err(|e| format!("Failed to get size: {}", e))?
            .len();

        let original_size = tensor_data.len() as u64;
        let ratio = original_size as f64 / compressed_size as f64;

        // Print progress for larger tensors
        if original_size > 1_000_000 {
            let num_blocks = (f32_weights.len() + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
            println!(
                "  {} ({:?} {:?}): {:.1}MB -> {:.1}MB ({:.2}x), {} blocks",
                name,
                info.dtype,
                info.shape,
                original_size as f64 / 1_000_000.0,
                compressed_size as f64 / 1_000_000.0,
                ratio,
                num_blocks
            );
        }

        stats.total_tensors += 1;
        stats.total_original_bytes += original_size;
        stats.total_compressed_bytes += compressed_size;
    }

    stats.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(stats)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut input_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input-dir" | "-i" => {
                input_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output-dir" | "-o" => {
                output_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--help" | "-h" => {
                println!("safetensors_to_int4_hct - Quantize safetensors to INT4 HCT");
                println!();
                println!("Usage: safetensors_to_int4_hct [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -i, --input-dir <DIR>    Input directory with .safetensors files");
                println!("  -o, --output-dir <DIR>   Output directory for .hct files");
                println!("  -h, --help               Show help");
                println!();
                println!("INT4 Format:");
                println!("  - Block size: {} elements per FP16 scale", Q4_BLOCK_SIZE);
                println!("  - Layout: [scales_f16][packed_int4]");
                println!("  - Symmetric quantization: value = (nibble - 8) * scale");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    let input_dir = input_dir.expect("--input-dir is required");
    let output_dir = output_dir.unwrap_or_else(|| input_dir.join("int4_hct"));

    println!("=== SafeTensors to INT4 HCT Converter ===\n");
    println!("Input:  {}", input_dir.display());
    println!("Output: {}", output_dir.display());
    println!("Block size: {} elements per FP16 scale", Q4_BLOCK_SIZE);
    println!("Quantization: Symmetric, centered at 8");
    println!();

    // Create output directory
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Find all safetensors files
    let safetensors_files: Vec<_> = fs::read_dir(&input_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if safetensors_files.is_empty() {
        eprintln!("No .safetensors files found in {}", input_dir.display());
        std::process::exit(1);
    }

    println!("Found {} safetensors files\n", safetensors_files.len());

    // Copy config.json and tokenizer files
    for filename in &["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let src = input_dir.join(filename);
        let dst = output_dir.join(filename);
        if src.exists() {
            fs::copy(&src, &dst).expect(&format!("Failed to copy {}", filename));
            println!("Copied {}", filename);
        }
    }
    println!();

    // Convert each safetensors file
    let mut total_stats = ConversionStats::default();

    for safetensors_path in &safetensors_files {
        println!(
            "Processing: {}",
            safetensors_path.file_name().unwrap().to_string_lossy()
        );

        match convert_safetensors_to_int4(safetensors_path, &output_dir) {
            Ok(stats) => {
                total_stats.total_tensors += stats.total_tensors;
                total_stats.total_original_bytes += stats.total_original_bytes;
                total_stats.total_compressed_bytes += stats.total_compressed_bytes;
                total_stats.elapsed_ms += stats.elapsed_ms;
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
        println!();
    }

    let ratio =
        total_stats.total_original_bytes as f64 / total_stats.total_compressed_bytes.max(1) as f64;
    let throughput = total_stats.total_original_bytes as f64
        / (total_stats.elapsed_ms as f64 / 1000.0)
        / 1_000_000.0;

    println!("=== Summary ===");
    println!("Tensors:    {}", total_stats.total_tensors);
    println!(
        "Original:   {:.2} GB",
        total_stats.total_original_bytes as f64 / 1_000_000_000.0
    );
    println!(
        "Compressed: {:.2} GB",
        total_stats.total_compressed_bytes as f64 / 1_000_000_000.0
    );
    println!("Ratio:      {:.2}x", ratio);
    println!("Time:       {:.2}s", total_stats.elapsed_ms as f64 / 1000.0);
    println!("Throughput: {:.0} MB/s", throughput);
    println!("\nOutput: {}", output_dir.display());
}

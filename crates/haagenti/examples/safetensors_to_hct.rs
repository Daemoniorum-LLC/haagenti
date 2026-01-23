//! Convert safetensors model weights to HCT compressed format.
//!
//! This tool reads a safetensors file and converts each tensor to a compressed
//! .hct file, suitable for loading with Infernum's compressed weight loader.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example safetensors_to_hct --features="lz4,zstd" -- \
//!     --input model.safetensors \
//!     --output-dir compressed/ \
//!     --algorithm zstd
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

use haagenti::tensor::{CompressionAlgorithm, DType, HctWriter, DEFAULT_BLOCK_SIZE};
use haagenti::{Lz4Compressor, ZstdCompressor};
use haagenti_core::CompressionLevel;

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

    // First 8 bytes are header length (little-endian u64)
    let header_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if data.len() < 8 + header_len {
        return Err("Invalid header length".to_string());
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_len])
        .map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

    // Parse JSON
    let header: serde_json::Value =
        serde_json::from_str(header_json).map_err(|e| format!("Invalid JSON header: {}", e))?;

    let mut tensors = HashMap::new();

    if let serde_json::Value::Object(obj) = header {
        for (name, value) in obj {
            // Skip __metadata__ key
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
                    .ok_or_else(|| format!("Missing data_offsets for {}", name))?;

                let start = offsets[0].as_u64().ok_or("Invalid offset")? as usize;
                let end = offsets[1].as_u64().ok_or("Invalid offset")? as usize;

                tensors.insert(
                    name.clone(),
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

/// Map safetensors dtype string to HCT DType.
fn map_dtype(dtype: &str) -> Result<DType, String> {
    match dtype {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I8" => Ok(DType::I8),
        // I4 isn't standard in safetensors, but we support it
        _ => Err(format!("Unsupported dtype: {}", dtype)),
    }
}

/// Configuration for the converter.
struct ConverterConfig {
    input: PathBuf,
    output_dir: PathBuf,
    algorithm: CompressionAlgorithm,
    level: CompressionLevel,
    block_size: u32,
}

/// Statistics from conversion.
#[derive(Debug, Default)]
struct ConversionStats {
    total_tensors: usize,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    elapsed_ms: u64,
}

fn convert_safetensors_to_hct(config: &ConverterConfig) -> Result<ConversionStats, String> {
    let start = Instant::now();

    // Read entire file
    let data = fs::read(&config.input).map_err(|e| format!("Failed to read input: {}", e))?;

    println!(
        "Read {} ({:.2} MB)",
        config.input.display(),
        data.len() as f64 / 1_000_000.0
    );

    // Parse header
    let (data_start, tensors) = parse_safetensors_header(&data)?;
    println!("Found {} tensors", tensors.len());

    // Create output directory
    fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    let mut stats = ConversionStats::default();

    // Process each tensor
    for (name, info) in &tensors {
        let dtype = map_dtype(&info.dtype)?;
        let shape: Vec<u64> = info.shape.iter().map(|&s| s as u64).collect();

        // Extract tensor data
        let tensor_data = &data[data_start + info.data_offsets.0..data_start + info.data_offsets.1];

        // Create output file
        let safe_name = name.replace(['/', '.'], "_");
        let output_path = config.output_dir.join(format!("{}.hct", safe_name));

        let output_file = File::create(&output_path)
            .map_err(|e| format!("Failed to create output file: {}", e))?;

        // Compress based on algorithm
        let compressed_size = match config.algorithm {
            CompressionAlgorithm::Lz4 => {
                let compressor = Lz4Compressor::with_level(config.level);
                let mut writer =
                    HctWriter::new(output_file, CompressionAlgorithm::Lz4, dtype, shape.clone())
                        .with_block_size(config.block_size);
                writer
                    .compress_data(tensor_data, &compressor)
                    .map_err(|e| format!("Compression failed: {}", e))?;
                writer
                    .finish()
                    .map_err(|e| format!("Failed to finish: {}", e))?;

                fs::metadata(&output_path)
                    .map_err(|e| format!("Failed to get output size: {}", e))?
                    .len()
            }
            CompressionAlgorithm::Zstd => {
                let compressor = ZstdCompressor::with_level(config.level);
                let mut writer = HctWriter::new(
                    output_file,
                    CompressionAlgorithm::Zstd,
                    dtype,
                    shape.clone(),
                )
                .with_block_size(config.block_size);
                writer
                    .compress_data(tensor_data, &compressor)
                    .map_err(|e| format!("Compression failed: {}", e))?;
                writer
                    .finish()
                    .map_err(|e| format!("Failed to finish: {}", e))?;

                fs::metadata(&output_path)
                    .map_err(|e| format!("Failed to get output size: {}", e))?
                    .len()
            }
        };

        let original_size = tensor_data.len() as u64;
        let ratio = original_size as f64 / compressed_size as f64;

        println!(
            "  {} ({:?} {:?}): {} -> {} bytes ({:.2}x)",
            name, info.dtype, info.shape, original_size, compressed_size, ratio
        );

        stats.total_tensors += 1;
        stats.total_original_bytes += original_size;
        stats.total_compressed_bytes += compressed_size;
    }

    stats.elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(stats)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Simple argument parsing
    let mut input: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut algorithm = CompressionAlgorithm::Zstd;
    let mut level = CompressionLevel::Default;
    let mut block_size = DEFAULT_BLOCK_SIZE;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--output-dir" | "-o" => {
                output_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--algorithm" | "-a" => {
                algorithm = match args[i + 1].as_str() {
                    "lz4" => CompressionAlgorithm::Lz4,
                    "zstd" => CompressionAlgorithm::Zstd,
                    other => {
                        eprintln!("Unknown algorithm: {}", other);
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--level" | "-l" => {
                level = match args[i + 1].as_str() {
                    "fast" => CompressionLevel::Fast,
                    "default" => CompressionLevel::Default,
                    "best" => CompressionLevel::Best,
                    other => {
                        eprintln!("Unknown level: {}", other);
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--block-size" | "-b" => {
                block_size = args[i + 1].parse().expect("Invalid block size");
                i += 2;
            }
            "--help" | "-h" => {
                println!("safetensors_to_hct - Convert safetensors to HCT compressed format");
                println!();
                println!("Usage: safetensors_to_hct [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -i, --input <FILE>       Input safetensors file");
                println!("  -o, --output-dir <DIR>   Output directory for .hct files");
                println!(
                    "  -a, --algorithm <ALG>    Compression algorithm (lz4, zstd) [default: zstd]"
                );
                println!("  -l, --level <LEVEL>      Compression level (fast, default, best)");
                println!("  -b, --block-size <SIZE>  Block size in bytes [default: 16384]");
                println!("  -h, --help               Show this help");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    let input = input.expect("--input is required");
    let output_dir = output_dir.unwrap_or_else(|| {
        input
            .parent()
            .unwrap_or(Path::new("."))
            .join("hct_compressed")
    });

    let config = ConverterConfig {
        input,
        output_dir,
        algorithm,
        level,
        block_size,
    };

    println!("=== SafeTensors to HCT Converter ===\n");
    println!("Input: {}", config.input.display());
    println!("Output: {}", config.output_dir.display());
    println!("Algorithm: {:?}", config.algorithm);
    println!("Level: {:?}", config.level);
    println!("Block Size: {} bytes", config.block_size);
    println!();

    match convert_safetensors_to_hct(&config) {
        Ok(stats) => {
            let overall_ratio =
                stats.total_original_bytes as f64 / stats.total_compressed_bytes as f64;
            let throughput = stats.total_original_bytes as f64
                / (stats.elapsed_ms as f64 / 1000.0)
                / 1_000_000.0;

            println!("\n=== Summary ===");
            println!("Tensors: {}", stats.total_tensors);
            println!(
                "Original: {:.2} MB",
                stats.total_original_bytes as f64 / 1_000_000.0
            );
            println!(
                "Compressed: {:.2} MB",
                stats.total_compressed_bytes as f64 / 1_000_000.0
            );
            println!("Ratio: {:.2}x", overall_ratio);
            println!("Time: {:.2}s", stats.elapsed_ms as f64 / 1000.0);
            println!("Throughput: {:.0} MB/s", throughput);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

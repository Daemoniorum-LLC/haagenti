//! Convert safetensors model weights to Holographic HCT format.
//!
//! This tool uses Haagenti's HoloTensorEncoder with Spectral (DCT) encoding
//! to create progressive, quality-controlled compressed model weights.
//!
//! ## Encoding Schemes
//!
//! - **Spectral (DCT)**: Default, optimized for neural network weights.
//!   Uses Discrete Cosine Transform with `essential_ratio` for quality control.
//! - **LRDF**: Low-Rank Distributed Fragments using SVD decomposition.
//! - **RandomProjection**: Johnson-Lindenstrauss random projections.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example safetensors_to_holographic --features="lz4,zstd" -- \
//!     --input model.safetensors \
//!     --output-dir holographic/ \
//!     --encoding spectral \
//!     --essential-ratio 0.15 \
//!     --fragments 16
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use haagenti::holotensor::{HoloTensorEncoder, HoloTensorWriter, HolographicEncoding};
use haagenti::tensor::{CompressionAlgorithm, DType};

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
        .map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

    let header: serde_json::Value =
        serde_json::from_str(header_json).map_err(|e| format!("Invalid JSON header: {}", e))?;

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

/// Convert bytes to f32 based on dtype.
fn bytes_to_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => {
            if data.len() % 4 != 0 {
                return Err("Invalid F32 data length".to_string());
            }
            Ok(data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        "F16" => {
            if data.len() % 2 != 0 {
                return Err("Invalid F16 data length".to_string());
            }
            Ok(data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        "BF16" => {
            if data.len() % 2 != 0 {
                return Err("Invalid BF16 data length".to_string());
            }
            Ok(data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect())
        }
        _ => Err(format!("Unsupported dtype: {}", dtype)),
    }
}

/// Map safetensors dtype to HCT DType.
fn map_dtype(dtype: &str) -> Result<DType, String> {
    match dtype {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        _ => Err(format!("Unsupported dtype: {}", dtype)),
    }
}

/// Configuration for holographic conversion.
struct HolographicConfig {
    input: PathBuf,
    output_dir: PathBuf,
    encoding: HolographicEncoding,
    num_fragments: u16,
    essential_ratio: f32,
    max_rank: usize,
    compression: CompressionAlgorithm,
    seed: u64,
    skip_small: usize, // Skip tensors smaller than this (use raw)
}

/// Statistics from conversion.
#[derive(Debug, Default)]
struct ConversionStats {
    total_tensors: usize,
    holographic_tensors: usize,
    raw_tensors: usize,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
    elapsed_ms: u64,
}

fn convert_to_holographic(config: &HolographicConfig) -> Result<ConversionStats, String> {
    let start = Instant::now();

    let data = fs::read(&config.input).map_err(|e| format!("Failed to read input: {}", e))?;

    println!(
        "Read {} ({:.2} GB)",
        config.input.display(),
        data.len() as f64 / 1_000_000_000.0
    );

    let (data_start, tensors) = parse_safetensors_header(&data)?;
    println!("Found {} tensors", tensors.len());

    fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    let mut stats = ConversionStats::default();

    // Sort tensors by name for consistent ordering
    let mut tensor_names: Vec<_> = tensors.keys().collect();
    tensor_names.sort();

    for (idx, name) in tensor_names.iter().enumerate() {
        let info = &tensors[*name];
        let tensor_data = &data[data_start + info.data_offsets.0..data_start + info.data_offsets.1];
        let original_size = tensor_data.len() as u64;

        // Convert tensor name to safe filename
        let safe_name = name.replace('/', "_").replace('.', "_");
        let output_path = config.output_dir.join(format!("{}.hct", safe_name));

        // Determine if tensor is suitable for holographic encoding
        let num_elements: usize = info.shape.iter().product();
        let is_matrix = info.shape.len() >= 2;
        let is_large_enough = num_elements >= config.skip_small;

        let compressed_size = if is_matrix && is_large_enough {
            // Use holographic encoding for matrices
            let f32_data = bytes_to_f32(tensor_data, &info.dtype)?;

            // Get matrix dimensions (treat as 2D)
            let (rows, cols) = if info.shape.len() == 2 {
                (info.shape[0], info.shape[1])
            } else {
                // Flatten multi-dimensional to 2D
                let last_dim = *info.shape.last().unwrap();
                let first_dims: usize = info.shape[..info.shape.len() - 1].iter().product();
                (first_dims, last_dim)
            };

            // Create encoder
            let encoder = HoloTensorEncoder::new(config.encoding)
                .with_fragments(config.num_fragments)
                .with_seed(config.seed + idx as u64)
                .with_compression(config.compression)
                .with_essential_ratio(config.essential_ratio)
                .with_max_rank(config.max_rank);

            // Encode to holographic fragments
            let (header, fragments) = encoder
                .encode_2d(&f32_data, rows, cols)
                .map_err(|e| format!("Encoding failed for {}: {}", name, e))?;

            // Write to file
            let file = File::create(&output_path)
                .map_err(|e| format!("Failed to create output file: {}", e))?;
            let mut writer = HoloTensorWriter::new(BufWriter::new(file));
            writer
                .write(&header, &fragments)
                .map_err(|e| format!("Failed to write holographic tensor: {}", e))?;

            stats.holographic_tensors += 1;

            fs::metadata(&output_path)
                .map_err(|e| format!("Failed to get output size: {}", e))?
                .len()
        } else {
            // Use raw storage for small/1D tensors (bias, layernorm, etc.)
            // Just copy the raw bytes with a simple header
            let dtype = map_dtype(&info.dtype)?;
            let shape: Vec<u64> = info.shape.iter().map(|&s| s as u64).collect();

            // Write raw HCT (no holographic encoding)
            let file = File::create(&output_path)
                .map_err(|e| format!("Failed to create output file: {}", e))?;

            let mut writer = haagenti::tensor::HctWriter::new(
                file,
                haagenti::tensor::CompressionAlgorithm::Lz4,
                dtype,
                shape,
            );

            let compressor = haagenti::Lz4Compressor::new();
            writer
                .compress_data(tensor_data, &compressor)
                .map_err(|e| format!("Compression failed: {}", e))?;
            writer
                .finish()
                .map_err(|e| format!("Failed to finish: {}", e))?;

            stats.raw_tensors += 1;

            fs::metadata(&output_path)
                .map_err(|e| format!("Failed to get output size: {}", e))?
                .len()
        };

        let ratio = original_size as f64 / compressed_size as f64;
        let encoding_type = if stats.holographic_tensors > stats.raw_tensors {
            "HOLO"
        } else {
            "RAW"
        };

        if idx < 10 || idx % 50 == 0 {
            println!(
                "  [{}/{}] {} ({:?} {:?}): {:.2} MB -> {:.2} MB ({:.2}x) [{}]",
                idx + 1,
                tensors.len(),
                name,
                info.dtype,
                info.shape,
                original_size as f64 / 1_000_000.0,
                compressed_size as f64 / 1_000_000.0,
                ratio,
                encoding_type
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

    let mut input: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut encoding = HolographicEncoding::Spectral;
    let mut num_fragments: u16 = 16;
    let mut essential_ratio: f32 = 0.15; // 15% essential coefficients
    let mut max_rank: usize = 256;
    let mut compression = CompressionAlgorithm::Lz4;
    let mut seed: u64 = 42;
    let mut skip_small: usize = 4096; // Skip tensors smaller than 4K elements

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
            "--encoding" | "-e" => {
                encoding = match args[i + 1].as_str() {
                    "spectral" | "dct" => HolographicEncoding::Spectral,
                    "lrdf" | "svd" => HolographicEncoding::LowRankDistributed,
                    "rph" | "random" => HolographicEncoding::RandomProjection,
                    other => {
                        eprintln!("Unknown encoding: {} (use: spectral, lrdf, rph)", other);
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--fragments" | "-f" => {
                num_fragments = args[i + 1].parse().expect("Invalid fragments");
                i += 2;
            }
            "--essential-ratio" | "-r" => {
                essential_ratio = args[i + 1].parse().expect("Invalid essential ratio");
                i += 2;
            }
            "--max-rank" | "-m" => {
                max_rank = args[i + 1].parse().expect("Invalid max rank");
                i += 2;
            }
            "--compression" | "-c" => {
                compression = match args[i + 1].as_str() {
                    "lz4" => CompressionAlgorithm::Lz4,
                    "zstd" => CompressionAlgorithm::Zstd,
                    other => {
                        eprintln!("Unknown compression: {} (use: lz4, zstd)", other);
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--seed" | "-s" => {
                seed = args[i + 1].parse().expect("Invalid seed");
                i += 2;
            }
            "--skip-small" => {
                skip_small = args[i + 1].parse().expect("Invalid skip-small");
                i += 2;
            }
            "--help" | "-h" => {
                println!("safetensors_to_holographic - Convert safetensors to Holographic HCT");
                println!();
                println!("Usage: safetensors_to_holographic [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -i, --input <FILE>          Input safetensors file");
                println!("  -o, --output-dir <DIR>      Output directory for .hct files");
                println!("  -e, --encoding <ENC>        Encoding: spectral (default), lrdf, rph");
                println!("  -f, --fragments <N>         Number of fragments [default: 16]");
                println!(
                    "  -r, --essential-ratio <R>   Essential ratio for spectral [default: 0.15]"
                );
                println!("  -m, --max-rank <R>          Max rank for LRDF [default: 256]");
                println!(
                    "  -c, --compression <ALG>     Fragment compression: lz4, zstd [default: lz4]"
                );
                println!("  -s, --seed <N>              Random seed [default: 42]");
                println!(
                    "  --skip-small <N>            Skip tensors smaller than N [default: 4096]"
                );
                println!("  -h, --help                  Show this help");
                println!();
                println!("Encoding Schemes:");
                println!("  spectral  DCT-based, optimized for neural network weights");
                println!(
                    "            Use --essential-ratio to control quality (0.1-0.3 recommended)"
                );
                println!("  lrdf      SVD-based low-rank decomposition");
                println!("            Use --max-rank to control quality (64-512 recommended)");
                println!("  rph       Random projection (Johnson-Lindenstrauss)");
                println!();
                println!("Quality Guidelines:");
                println!("  High quality:   --essential-ratio 0.20 or --max-rank 512");
                println!("  Medium quality: --essential-ratio 0.15 or --max-rank 256");
                println!("  Aggressive:     --essential-ratio 0.10 or --max-rank 128");
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
            .join("holographic_hct")
    });

    let config = HolographicConfig {
        input,
        output_dir,
        encoding,
        num_fragments,
        essential_ratio,
        max_rank,
        compression,
        seed,
        skip_small,
    };

    println!("=== SafeTensors to Holographic HCT Converter ===\n");
    println!("Input: {}", config.input.display());
    println!("Output: {}", config.output_dir.display());
    println!("Encoding: {:?}", config.encoding);
    println!("Fragments: {}", config.num_fragments);
    match config.encoding {
        HolographicEncoding::Spectral => {
            println!("Essential Ratio: {:.1}%", config.essential_ratio * 100.0);
        }
        HolographicEncoding::LowRankDistributed => {
            println!("Max Rank: {}", config.max_rank);
        }
        HolographicEncoding::RandomProjection => {
            println!("Projection Seed: {}", config.seed);
        }
    }
    println!("Compression: {:?}", config.compression);
    println!();

    match convert_to_holographic(&config) {
        Ok(stats) => {
            let overall_ratio =
                stats.total_original_bytes as f64 / stats.total_compressed_bytes as f64;
            let throughput = stats.total_original_bytes as f64
                / (stats.elapsed_ms as f64 / 1000.0)
                / 1_000_000.0;

            println!("\n=== Summary ===");
            println!("Total Tensors: {}", stats.total_tensors);
            println!(
                "  Holographic: {} ({:.1}%)",
                stats.holographic_tensors,
                stats.holographic_tensors as f64 / stats.total_tensors as f64 * 100.0
            );
            println!(
                "  Raw (small): {} ({:.1}%)",
                stats.raw_tensors,
                stats.raw_tensors as f64 / stats.total_tensors as f64 * 100.0
            );
            println!(
                "Original: {:.2} GB",
                stats.total_original_bytes as f64 / 1_000_000_000.0
            );
            println!(
                "Compressed: {:.2} GB",
                stats.total_compressed_bytes as f64 / 1_000_000_000.0
            );
            println!("Compression Ratio: {:.2}x", overall_ratio);
            println!("Time: {:.1}s", stats.elapsed_ms as f64 / 1000.0);
            println!("Throughput: {:.0} MB/s", throughput);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

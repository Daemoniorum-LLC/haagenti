//! Test compression on very large tensors (405B model scale)
//!
//! Usage:
//! ```bash
//! TENSOR_FILE=/tmp/llama405b-safetensors/model_layers_0_mlp_gate_proj_weight.safetensors \
//!   cargo run --release --example test_large_tensor
//! ```

use std::collections::HashMap;
use std::env;
use std::fs;
use std::time::Instant;

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};

#[derive(Debug)]
struct TensorInfo {
    shape: Vec<usize>,
    dtype: String,
    data_offsets: (usize, usize),
}

fn parse_safetensors_header(data: &[u8]) -> Option<(usize, HashMap<String, TensorInfo>)> {
    if data.len() < 8 {
        return None;
    }

    let header_size = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
    if header_size > 100_000_000 {
        return None;
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_size]).ok()?;
    let header: serde_json::Value = serde_json::from_str(header_json).ok()?;

    let mut tensors = HashMap::new();
    if let Some(obj) = header.as_object() {
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            if let Some(info_obj) = info.as_object() {
                let shape: Vec<usize> = info_obj
                    .get("shape")?
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();

                let dtype = info_obj.get("dtype")?.as_str()?.to_string();

                let offsets = info_obj.get("data_offsets")?.as_array()?;
                let start = offsets.get(0)?.as_u64()? as usize;
                let end = offsets.get(1)?.as_u64()? as usize;

                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        shape,
                        dtype,
                        data_offsets: (start, end),
                    },
                );
            }
        }
    }

    Some((8 + header_size, tensors))
}

fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        "F16" => data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        _ => vec![],
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

fn main() {
    let tensor_file = env::var("TENSOR_FILE").unwrap_or_else(|_| {
        "/tmp/llama405b-safetensors/model_layers_0_mlp_gate_proj_weight.safetensors".to_string()
    });

    let retention: f32 = env::var("RETENTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.30);

    println!("=== Large Tensor Compression Test ===\n");
    println!("File: {}", tensor_file);
    println!("Retention: {:.0}%\n", retention * 100.0);

    // Load tensor
    println!("Loading tensor...");
    let load_start = Instant::now();
    let data = fs::read(&tensor_file).expect("Failed to read tensor file");
    let load_time = load_start.elapsed();
    println!(
        "  Loaded {:.2} MB in {:.2}s\n",
        data.len() as f64 / 1024.0 / 1024.0,
        load_time.as_secs_f64()
    );

    let (data_start, tensors) = parse_safetensors_header(&data).expect("Failed to parse header");

    for (name, info) in &tensors {
        println!("Tensor: {}", name);
        println!("  Shape: {:?}", info.shape);
        println!("  Dtype: {}", info.dtype);
        let num_elements: usize = info.shape.iter().product();
        println!(
            "  Elements: {} ({:.2}M)",
            num_elements,
            num_elements as f64 / 1_000_000.0
        );

        // Convert to f32
        println!("\nConverting to f32...");
        let convert_start = Instant::now();
        let (start, end) = info.data_offsets;
        let tensor_data = &data[data_start + start..data_start + end];
        let values = bytes_to_f32(tensor_data, &info.dtype);
        let convert_time = convert_start.elapsed();
        println!("  Converted in {:.2}s", convert_time.as_secs_f64());

        // Get dimensions
        let (width, height) = if info.shape.len() == 2 {
            (info.shape[1], info.shape[0])
        } else {
            (num_elements, 1)
        };

        // For very large tensors, we'll process in chunks
        let chunk_height = 1024.min(height);
        let num_chunks = (height + chunk_height - 1) / chunk_height;

        println!(
            "\nCompressing {} chunks of {}x{} each...",
            num_chunks, width, chunk_height
        );

        let encoder = CompressiveSpectralEncoder::new(8, retention);

        let compress_start = Instant::now();
        let mut reconstructed = Vec::with_capacity(values.len());
        let mut total_encoded_size = 0usize;

        for chunk_idx in 0..num_chunks {
            let row_start = chunk_idx * chunk_height;
            let row_end = (row_start + chunk_height).min(height);
            let actual_chunk_height = row_end - row_start;

            let chunk_start = row_start * width;
            let chunk_end = row_end * width;
            let chunk_values = &values[chunk_start..chunk_end];

            // Encode
            let encoded = encoder
                .encode_2d(chunk_values, width, actual_chunk_height)
                .expect("Encoding failed");
            total_encoded_size += encoded.iter().map(|f| f.data_size()).sum::<usize>();

            // Decode using the progressive decoder API
            let mut chunk_decoder = CompressiveSpectralDecoder::new();

            // Add fragment 0 (essentials) first
            for frag in &encoded {
                if frag.index == 0 {
                    chunk_decoder
                        .add_essentials(frag)
                        .expect("Failed to add essentials");
                } else {
                    chunk_decoder
                        .add_detail(frag)
                        .expect("Failed to add detail");
                }
            }

            let mut decoded = chunk_decoder.reconstruct().expect("Reconstruction failed");

            // Add back mean (DCT removes DC component)
            let mean: f32 = chunk_values.iter().sum::<f32>() / chunk_values.len() as f32;
            for v in &mut decoded {
                *v += mean;
            }

            reconstructed.extend(decoded);

            if (chunk_idx + 1) % 10 == 0 || chunk_idx == num_chunks - 1 {
                print!("\r  Progress: {}/{} chunks", chunk_idx + 1, num_chunks);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
        println!();

        let compress_time = compress_start.elapsed();

        // Calculate quality metrics
        let cos_sim = cosine_similarity(&values, &reconstructed);
        let mse_val = mse(&values, &reconstructed);

        let original_size = values.len() * 4; // f32
        let compression_ratio = original_size as f64 / total_encoded_size as f64;

        println!("\n=== Results ===\n");
        println!("Compression time: {:.2}s", compress_time.as_secs_f64());
        println!(
            "Throughput: {:.2} MB/s",
            (original_size as f64 / 1024.0 / 1024.0) / compress_time.as_secs_f64()
        );
        println!("\nQuality:");
        println!("  Cosine similarity: {:.6}", cos_sim);
        println!("  MSE: {:.6}", mse_val);
        println!("\nSize:");
        println!(
            "  Original (f32): {:.2} MB",
            original_size as f64 / 1024.0 / 1024.0
        );
        println!(
            "  Encoded: {:.2} MB",
            total_encoded_size as f64 / 1024.0 / 1024.0
        );
        println!("  Compression ratio: {:.2}x", compression_ratio);
    }
}

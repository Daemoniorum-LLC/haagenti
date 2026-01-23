//! Validate that compressed models produce correct inference results.
//!
//! This test compares tensors from original and compressed Qwen model.
//!
//! Run with:
//! ```bash
//! cargo run --release --example validate_compressed_inference -p haagenti-cuda
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn find_huggingface_model(model_name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let cache_dir = PathBuf::from(home).join(".cache/huggingface/hub");
    let model_dir_name = format!("models--{}", model_name.replace('/', "--"));
    let model_dir = cache_dir.join(&model_dir_name);

    if model_dir.exists() {
        let snapshots_dir = model_dir.join("snapshots");
        if snapshots_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let single_shard = path.join("model.safetensors");
                        if single_shard.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Simple safetensors header parser - extracts tensor info without full JSON parsing
fn parse_tensor_info(
    header_str: &str,
    tensor_name: &str,
) -> Option<(String, Vec<usize>, usize, usize)> {
    // Find the tensor entry in JSON
    let search = format!("\"{}\"", tensor_name);
    let pos = header_str.find(&search)?;

    // Find the end of this tensor's entry (next tensor or end of object)
    let tensor_section = &header_str[pos..];
    let section_end = tensor_section[1..]
        .find("},")
        .map(|p| p + 2)
        .unwrap_or(tensor_section.len());
    let tensor_section = &tensor_section[..section_end];

    // Find dtype - look for "dtype":"<type>"
    let dtype = if let Some(dtype_start) = tensor_section.find("\"dtype\":") {
        let rest = &tensor_section[dtype_start + 8..];
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            let dtype_end = rest[1..].find('"').map(|p| p + 1)?;
            rest[1..dtype_end].to_string()
        } else {
            return None;
        }
    } else {
        return None;
    };

    // Find shape - look for "shape":[X,Y,...]
    let shape = if let Some(shape_start) = tensor_section.find("\"shape\":") {
        let rest = &tensor_section[shape_start + 8..];
        let rest = rest.trim_start();
        if rest.starts_with('[') {
            let shape_end = rest.find(']')?;
            let shape_str = &rest[1..shape_end];
            shape_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        } else {
            return None;
        }
    } else {
        return None;
    };

    // Find data_offsets - look for "data_offsets":[start,end]
    let (start, end) = if let Some(off_start) = tensor_section.find("\"data_offsets\":") {
        let rest = &tensor_section[off_start + 15..];
        let rest = rest.trim_start();
        if rest.starts_with('[') {
            let off_end = rest.find(']')?;
            let off_str = &rest[1..off_end];
            let nums: Vec<usize> = off_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if nums.len() >= 2 {
                (nums[0], nums[1])
            } else {
                return None;
            }
        } else {
            return None;
        }
    } else {
        return None;
    };

    Some((dtype, shape, start, end))
}

fn read_safetensors_header(path: &std::path::Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| format!("Failed to read header length: {}", e))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {}", e))?;

    Ok(String::from_utf8_lossy(&header_bytes).to_string())
}

fn read_tensor_data(path: &std::path::Path, offset: usize, len: usize) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| format!("Failed to read header length: {}", e))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    file.seek(SeekFrom::Start((8 + header_len + offset) as u64))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    let mut data = vec![0u8; len];
    file.read_exact(&mut data)
        .map_err(|e| format!("Failed to read tensor: {}", e))?;

    Ok(data)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn bytes_to_f16_as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn bytes_to_bf16_as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::bf16::from_bits(bits).to_f32()
        })
        .collect()
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Compressed Model Quality Validation");
    println!("====================================\n");

    // Find original model
    let model_name = "Qwen/Qwen2.5-0.5B-Instruct";
    let model_path = find_huggingface_model(model_name)
        .ok_or_else(|| format!("Model {} not found in HuggingFace cache", model_name))?;

    let original_file = model_path.join("model.safetensors");
    if !original_file.exists() {
        return Err(format!("Original model file not found: {}", original_file.display()).into());
    }

    // Find compressed model
    let compressed_file = PathBuf::from("/tmp/qwen-compressed-70pct-int4.safetensors");
    if !compressed_file.exists() {
        return Err(format!(
            "Compressed model not found: {}\n\
             Create with: RETENTION=0.70 cargo run --release --example create_compressed_model -p haagenti",
            compressed_file.display()
        ).into());
    }

    println!("Original:   {}", original_file.display());
    println!("Compressed: {}\n", compressed_file.display());

    // Read headers
    let orig_header = read_safetensors_header(&original_file)?;
    let comp_header = read_safetensors_header(&compressed_file)?;

    // Test tensors covering different types
    let test_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.11.mlp.up_proj.weight",
        "model.layers.23.self_attn.o_proj.weight",
        "lm_head.weight",
    ];

    println!(
        "{:50} {:>10} {:>10} {:>10}",
        "Tensor", "Elements", "Cosine", "Status"
    );
    println!("{}", "-".repeat(85));

    let mut total_cosine = 0.0;
    let mut count = 0;

    for tensor_name in test_tensors.iter() {
        let orig_info = match parse_tensor_info(&orig_header, tensor_name) {
            Some(info) => info,
            None => {
                println!(
                    "{:50} {:>10} {:>10} {:>10}",
                    tensor_name, "N/A", "N/A", "NOT FOUND"
                );
                continue;
            }
        };

        let comp_info = match parse_tensor_info(&comp_header, tensor_name) {
            Some(info) => info,
            None => {
                println!(
                    "{:50} {:>10} {:>10} {:>10}",
                    tensor_name, "N/A", "N/A", "NOT FOUND"
                );
                continue;
            }
        };

        let (orig_dtype, orig_shape, orig_start, orig_end) = orig_info;
        let (comp_dtype, _comp_shape, comp_start, comp_end) = comp_info;

        let elements: usize = orig_shape.iter().product();
        let orig_bytes = read_tensor_data(&original_file, orig_start, orig_end - orig_start)?;
        let comp_bytes = read_tensor_data(&compressed_file, comp_start, comp_end - comp_start)?;

        // Convert to f32
        let orig_f32 = match orig_dtype.as_str() {
            "F32" => bytes_to_f32(&orig_bytes),
            "F16" => bytes_to_f16_as_f32(&orig_bytes),
            "BF16" => bytes_to_bf16_as_f32(&orig_bytes),
            _ => {
                println!(
                    "{:50} {:>10} {:>10} {:>10}",
                    tensor_name, elements, "N/A", &orig_dtype
                );
                continue;
            }
        };

        let comp_f32 = match comp_dtype.as_str() {
            "F32" => bytes_to_f32(&comp_bytes),
            "F16" => bytes_to_f16_as_f32(&comp_bytes),
            "BF16" => bytes_to_bf16_as_f32(&comp_bytes),
            _ => {
                println!(
                    "{:50} {:>10} {:>10} {:>10}",
                    tensor_name, elements, "N/A", &comp_dtype
                );
                continue;
            }
        };

        // Compute cosine similarity
        let cos = cosine_similarity(&orig_f32, &comp_f32);
        let status = if cos >= 0.99 {
            "EXCELLENT"
        } else if cos >= 0.95 {
            "GOOD"
        } else if cos >= 0.90 {
            "OK"
        } else {
            "DEGRADED"
        };

        println!(
            "{:50} {:>10} {:>10.4} {:>10}",
            tensor_name, elements, cos, status
        );
        total_cosine += cos;
        count += 1;
    }

    if count > 0 {
        let avg_cosine = total_cosine / count as f32;
        println!("\n{:50} {:>10} {:>10.4}", "AVERAGE", "", avg_cosine);

        println!("\nConclusion:");
        if avg_cosine >= 0.99 {
            println!("  Model quality: EXCELLENT - Ready for production");
        } else if avg_cosine >= 0.95 {
            println!("  Model quality: GOOD - Suitable for most uses");
        } else if avg_cosine >= 0.90 {
            println!("  Model quality: ACCEPTABLE - Minor degradation");
        } else {
            println!("  Model quality: DEGRADED - Consider higher retention");
        }
    }

    println!("\nValidation complete.");
    Ok(())
}

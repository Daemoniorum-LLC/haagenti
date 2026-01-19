//! Test HCT format decompression from compressed model files.
//!
//! This validates that we can decompress actual HCT-compressed tensors.
//!
//! Usage:
//! ```bash
//! # First create a compressed model:
//! CARGO_INCREMENTAL=0 cargo run --release --example create_compressed_model -p haagenti
//!
//! # Then test decompression:
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib CARGO_INCREMENTAL=0 cargo run --release --example test_hct_decompress -p haagenti-cuda
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use haagenti_cuda::decompress::{decompress_cpu, GpuDecompressor};

    println!("HCT Format Decompression Test");
    println!("==============================\n");

    // Find compressed model
    let compressed_paths = [
        PathBuf::from("/tmp/SmolLM2-135M-compressed/model.safetensors"),
        PathBuf::from("/tmp/qwen-compressed-70pct-int4.safetensors"),
    ];

    let compressed_path = compressed_paths
        .iter()
        .find(|p| p.exists())
        .ok_or("No compressed model found. Run create_compressed_model first.")?;

    println!("Compressed model: {}\n", compressed_path.display());

    // Read safetensors header
    let mut file = File::open(compressed_path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header_str = String::from_utf8_lossy(&header_bytes);

    // Simple JSON parsing to extract tensor info
    let tensors = parse_safetensors_header(&header_str);

    if tensors.is_empty() {
        return Err("No tensors found in compressed file".into());
    }

    println!("Found {} tensors, testing decompression on first 5...\n", tensors.len());

    let test_tensors: Vec<_> = tensors.into_iter().take(5).collect();

    // Test CPU decompression
    println!("CPU Decompression:");
    println!("{:50} {:>12} {:>14}", "Tensor", "Elements", "Status");
    println!("{}", "-".repeat(80));

    let data_start = 8 + header_len;

    for (name, shape, start, end) in &test_tensors {
        let compressed_size = end - start;

        // Read compressed data
        file.seek(SeekFrom::Start((data_start + start) as u64))?;
        let mut compressed = vec![0u8; compressed_size];
        file.read_exact(&mut compressed)?;

        let elements: usize = shape.iter().product();

        // Test CPU decompression
        let result = decompress_cpu(&compressed, shape);
        let status = match &result {
            Ok(data) => format!("OK ({} values)", data.len()),
            Err(e) => format!("FAIL: {}", e),
        };

        let name_short: String = if name.len() > 48 {
            format!("{}...", &name[..45])
        } else {
            name.clone()
        };
        println!("{:50} {:>12} {:>14}", name_short, elements, status);
    }

    println!();

    // Test GPU decompression (if available)
    println!("GPU Decompression:");
    match GpuDecompressor::new(0) {
        Ok(mut gpu) => {
            println!("{:50} {:>12} {:>14}", "Tensor", "Elements", "Status");
            println!("{}", "-".repeat(80));

            for (name, shape, start, end) in &test_tensors {
                let compressed_size = end - start;

                file.seek(SeekFrom::Start((data_start + start) as u64))?;
                let mut compressed = vec![0u8; compressed_size];
                file.read_exact(&mut compressed)?;

                let elements: usize = shape.iter().product();

                let result = gpu.decompress(&compressed, shape);
                let status = match &result {
                    Ok(data) => format!("OK ({} values)", data.len()),
                    Err(e) => format!("FAIL: {}", e),
                };

                let name_short: String = if name.len() > 48 {
                    format!("{}...", &name[..45])
                } else {
                    name.clone()
                };
                println!("{:50} {:>12} {:>14}", name_short, elements, status);
            }
        }
        Err(e) => {
            println!("GPU not available: {}", e);
            println!("Skipping GPU tests.");
        }
    }

    println!("\nDecompression test complete.");
    Ok(())
}

/// Simple parser for safetensors header JSON.
/// Returns Vec<(name, shape, start_offset, end_offset)>
fn parse_safetensors_header(header: &str) -> Vec<(String, Vec<usize>, usize, usize)> {
    let mut tensors = Vec::new();

    // Find each tensor entry by looking for patterns like:
    // "tensor_name":{"dtype":"...", "shape":[...], "data_offsets":[start, end]}

    let mut pos = 0;
    while let Some(name_start) = header[pos..].find('"') {
        let name_start = pos + name_start + 1;
        if let Some(name_end) = header[name_start..].find('"') {
            let name_end = name_start + name_end;
            let name = &header[name_start..name_end];

            // Skip metadata
            if name == "__metadata__" || name.starts_with("__") {
                pos = name_end + 1;
                continue;
            }

            // Look for the tensor's data after the name
            let tensor_start = name_end + 1;
            if let Some(entry_start) = header[tensor_start..].find('{') {
                let entry_start = tensor_start + entry_start;
                if let Some(entry_end) = find_matching_brace(&header[entry_start..]) {
                    let entry = &header[entry_start..entry_start + entry_end + 1];

                    // Extract shape
                    let shape = extract_array(entry, "shape")
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>();

                    // Extract data_offsets
                    let offsets = extract_array(entry, "data_offsets").unwrap_or_default();
                    if offsets.len() >= 2 {
                        if let (Ok(start), Ok(end)) = (
                            offsets[0].trim().parse::<usize>(),
                            offsets[1].trim().parse::<usize>(),
                        ) {
                            if !shape.is_empty() {
                                tensors.push((name.to_string(), shape, start, end));
                            }
                        }
                    }

                    pos = entry_start + entry_end + 1;
                    continue;
                }
            }
        }
        pos += 1;
    }

    tensors
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.chars().enumerate() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn extract_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{}\"", key);
    let key_pos = json.find(&pattern)?;
    let after_key = &json[key_pos + pattern.len()..];

    let arr_start = after_key.find('[')?;
    let arr_end = after_key[arr_start..].find(']')?;
    let arr_content = &after_key[arr_start + 1..arr_start + arr_end];

    Some(arr_content.split(',').map(|s| s.to_string()).collect())
}

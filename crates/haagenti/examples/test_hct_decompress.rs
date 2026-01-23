//! Test HCT file decompression.
//!
//! Run with:
//! ```bash
//! cargo run --example test_hct_decompress --features="lz4,zstd" -- /tmp/smol_zstd/model_layers_0_mlp_down_proj_weight.hct
//! ```

use std::fs::File;
use std::path::PathBuf;

use haagenti::tensor::{CompressionAlgorithm, HctReader};
use haagenti::{Lz4Decompressor, ZstdDecompressor};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: test_hct_decompress <hct_file>");
        std::process::exit(1);
    }

    let path = PathBuf::from(&args[1]);
    println!("=== HCT Decompression Test ===\n");
    println!("File: {}", path.display());

    // Open the file
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file: {}", e);
            std::process::exit(1);
        }
    };

    // Read HCT header
    let mut reader = match HctReader::new(file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read HCT header: {}", e);
            std::process::exit(1);
        }
    };

    // Copy header values before mutable borrow
    let algorithm = reader.header().algorithm;
    let dtype = reader.header().dtype;
    let shape = reader.header().shape.clone();
    let original_size = reader.header().original_size;
    let compressed_size = reader.header().compressed_size;
    let block_size = reader.header().block_size;
    let num_blocks_header = reader.header().num_blocks;

    println!("Algorithm: {:?}", algorithm);
    println!("Dtype: {:?}", dtype);
    println!("Shape: {:?}", shape);
    println!("Original size: {} bytes", original_size);
    println!("Compressed size: {} bytes", compressed_size);
    println!("Block size: {} bytes", block_size);
    println!("Num blocks: {}", num_blocks_header);
    println!();

    // Try to decompress
    println!("Attempting decompression...");

    let result = match algorithm {
        CompressionAlgorithm::Lz4 => {
            let decompressor = Lz4Decompressor::new();
            reader.decompress_all(&decompressor)
        }
        CompressionAlgorithm::Zstd => {
            let decompressor = ZstdDecompressor::new();
            reader.decompress_all(&decompressor)
        }
    };

    match result {
        Ok(data) => {
            println!("SUCCESS! Decompressed {} bytes", data.len());
            if data.len() != original_size as usize {
                println!(
                    "WARNING: Size mismatch! Expected {}, got {}",
                    original_size,
                    data.len()
                );
            }

            // Show first few bytes
            if data.len() >= 16 {
                println!("First 16 bytes: {:02x?}", &data[..16]);
            }
        }
        Err(e) => {
            eprintln!("DECOMPRESSION FAILED: {}", e);

            // Try block by block to find the failing block
            println!("\nTrying block-by-block decompression...");

            // Re-open file
            let file = File::open(&path).expect("reopen");
            let mut reader = HctReader::new(file).expect("re-read");
            let num_blocks = reader.num_blocks();

            match algorithm {
                CompressionAlgorithm::Zstd => {
                    let decompressor = ZstdDecompressor::new();
                    for i in 0..num_blocks {
                        match reader.decompress_block(i, &decompressor) {
                            Ok(block) => println!("  Block {}: {} bytes OK", i, block.len()),
                            Err(e) => {
                                eprintln!("  Block {}: FAILED - {}", i, e);

                                // Read the raw compressed block to inspect
                                let file = File::open(&path).expect("reopen2");
                                let mut reader2 = HctReader::new(file).expect("re-read2");
                                if let Ok(raw) = reader2.read_block(i) {
                                    println!("    Raw compressed size: {} bytes", raw.len());
                                    if raw.len() >= 16 {
                                        println!("    First 16 bytes: {:02x?}", &raw[..16]);
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
                CompressionAlgorithm::Lz4 => {
                    let decompressor = Lz4Decompressor::new();
                    for i in 0..num_blocks {
                        match reader.decompress_block(i, &decompressor) {
                            Ok(block) => println!("  Block {}: {} bytes OK", i, block.len()),
                            Err(e) => {
                                eprintln!("  Block {}: FAILED - {}", i, e);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

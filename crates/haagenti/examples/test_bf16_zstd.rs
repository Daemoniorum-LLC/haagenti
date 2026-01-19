//! Test Zstd roundtrip with BF16-like random data (simulating model weights).
//!
//! Run with:
//! ```bash
//! cargo run --example test_bf16_zstd
//! ```

use std::io::Cursor;

use haagenti::tensor::{CompressionAlgorithm, DType, HctReader, HctWriter};
use haagenti::{ZstdCompressor, ZstdDecompressor};
use haagenti_core::CompressionLevel;

fn main() {
    println!("=== BF16 Zstd Roundtrip Test ===\n");

    // Test 1: Small BF16 tensor (1KB)
    test_roundtrip("Small (1KB)", 512, CompressionLevel::Default);

    // Test 2: Medium BF16 tensor (64KB)
    test_roundtrip("Medium (64KB)", 32768, CompressionLevel::Default);

    // Test 3: Large BF16 tensor (1MB)
    test_roundtrip("Large (1MB)", 524288, CompressionLevel::Default);

    // Test 4: Very large - embedding size (~8MB)
    test_roundtrip("Embedding (8MB)", 4194304, CompressionLevel::Default);

    // Test 5: With different block sizes
    test_roundtrip_block_size("16KB blocks", 65536, 16384);
    test_roundtrip_block_size("32KB blocks", 65536, 32768);
    test_roundtrip_block_size("64KB blocks", 65536, 65536);

    println!("\n=== All Tests Complete ===");
}

fn test_roundtrip(name: &str, num_bf16_values: usize, level: CompressionLevel) {
    print!("Test {}: ", name);

    // Generate pseudo-random BF16 data (similar to model weights)
    // BF16 is 2 bytes per value
    let original_data: Vec<u8> = (0..num_bf16_values * 2)
        .map(|i| {
            // Mix of patterns that look like BF16 weights
            let x = i as u32;
            let hash = x.wrapping_mul(0x9E3779B9).wrapping_add(x >> 16);
            (hash & 0xFF) as u8
        })
        .collect();

    let original_size = original_data.len();

    // Compress
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let compressor = ZstdCompressor::with_level(level);

        let shape = vec![num_bf16_values as u64];
        let mut writer = HctWriter::new(cursor, CompressionAlgorithm::Zstd, DType::BF16, shape);

        if let Err(e) = writer.compress_data(&original_data, &compressor) {
            println!("COMPRESS FAILED: {}", e);
            return;
        }

        if let Err(e) = writer.finish() {
            println!("FINISH FAILED: {}", e);
            return;
        }
    }

    let compressed_size = buffer.len();
    let ratio = original_size as f64 / compressed_size as f64;

    // Decompress
    let cursor = Cursor::new(&buffer);
    let mut reader = match HctReader::new(cursor) {
        Ok(r) => r,
        Err(e) => {
            println!("READER FAILED: {}", e);
            return;
        }
    };

    let decompressor = ZstdDecompressor::new();
    let decompressed = match reader.decompress_all(&decompressor) {
        Ok(d) => d,
        Err(e) => {
            println!("DECOMPRESS FAILED: {}", e);
            println!("  Original size: {} bytes", original_size);
            println!("  Compressed size: {} bytes", compressed_size);
            println!("  Num blocks: {}", reader.num_blocks());
            return;
        }
    };

    // Verify
    if decompressed == original_data {
        println!("OK ({} -> {} bytes, {:.2}x)", original_size, compressed_size, ratio);
    } else {
        println!("DATA MISMATCH!");
        println!("  Expected {} bytes, got {} bytes", original_data.len(), decompressed.len());
        if decompressed.len() >= 16 && original_data.len() >= 16 {
            println!("  First 16 original: {:?}", &original_data[..16]);
            println!("  First 16 decompressed: {:?}", &decompressed[..16]);
        }
    }
}

fn test_roundtrip_block_size(name: &str, num_bf16_values: usize, block_size: u32) {
    print!("Test {}: ", name);

    let original_data: Vec<u8> = (0..num_bf16_values * 2)
        .map(|i| {
            let x = i as u32;
            let hash = x.wrapping_mul(0x9E3779B9).wrapping_add(x >> 16);
            (hash & 0xFF) as u8
        })
        .collect();

    let original_size = original_data.len();

    // Compress with specific block size
    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let compressor = ZstdCompressor::new();

        let shape = vec![num_bf16_values as u64];
        let mut writer = HctWriter::new(cursor, CompressionAlgorithm::Zstd, DType::BF16, shape)
            .with_block_size(block_size);

        if let Err(e) = writer.compress_data(&original_data, &compressor) {
            println!("COMPRESS FAILED: {}", e);
            return;
        }

        if let Err(e) = writer.finish() {
            println!("FINISH FAILED: {}", e);
            return;
        }
    }

    let compressed_size = buffer.len();
    let ratio = original_size as f64 / compressed_size as f64;

    // Decompress
    let cursor = Cursor::new(&buffer);
    let mut reader = match HctReader::new(cursor) {
        Ok(r) => r,
        Err(e) => {
            println!("READER FAILED: {}", e);
            return;
        }
    };

    let decompressor = ZstdDecompressor::new();
    let decompressed = match reader.decompress_all(&decompressor) {
        Ok(d) => d,
        Err(e) => {
            println!("DECOMPRESS FAILED: {}", e);
            println!("  Num blocks: {}", reader.num_blocks());
            return;
        }
    };

    if decompressed == original_data {
        println!("OK ({} blocks, {:.2}x)", reader.num_blocks(), ratio);
    } else {
        println!("DATA MISMATCH!");
    }
}

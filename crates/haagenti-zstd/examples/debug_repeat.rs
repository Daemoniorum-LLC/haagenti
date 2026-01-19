//! Debug repeat offset match finder

use haagenti_core::{CompressionLevel, Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};
use std::io::Cursor;

fn main() {
    // Simple test pattern
    let mut data = Vec::new();
    for i in 0..100 {
        let line = format!("[2024-01-{:02} 10:{:02}:00] INFO Processing request #{}\n",
                          (i % 28) + 1, i % 60, i * 1000);
        data.extend_from_slice(line.as_bytes());
    }

    println!("Input: {} bytes", data.len());

    // Test with Default level (Lazy) - this works
    let default_compressor = ZstdCompressor::with_level(CompressionLevel::Default);
    let default_compressed = default_compressor.compress(&data).unwrap();
    println!("Default compressed: {} bytes", default_compressed.len());

    let decompressor = ZstdDecompressor;
    let default_dec = decompressor.decompress(&default_compressed).unwrap();
    assert_eq!(default_dec, data, "Default decompression failed");
    println!("Default decompression: OK");

    // Verify with reference
    let ref_dec = zstd::decode_all(Cursor::new(&default_compressed)).unwrap();
    assert_eq!(ref_dec, data, "Reference decode of Default failed");
    println!("Reference decode of Default: OK");

    // Test with Best level (RepeatOffsetMatchFinder)
    let best_compressor = ZstdCompressor::with_level(CompressionLevel::Best);
    let best_compressed = best_compressor.compress(&data).unwrap();
    println!("\nBest compressed: {} bytes", best_compressed.len());

    // Try reference decoder first
    match zstd::decode_all(Cursor::new(&best_compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("Reference decode of Best: OK");
            } else {
                println!("Reference decode of Best: MISMATCH");
                println!("  Expected: {} bytes", data.len());
                println!("  Got: {} bytes", dec.len());
            }
        }
        Err(e) => {
            println!("Reference decode of Best: FAILED - {:?}", e);

            // Compare bytes
            println!("\nComparing Default vs Best output:");
            let max = default_compressed.len().max(best_compressed.len()).min(100);
            for i in 0..max {
                let d = default_compressed.get(i);
                let b = best_compressed.get(i);
                if d != b {
                    println!("  Diff at {}: default={:02x?} best={:02x?}", i, d, b);
                }
            }
        }
    }
}

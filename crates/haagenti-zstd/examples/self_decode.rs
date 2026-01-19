//! Test if our own decoder can decompress our output

use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    let input = b"ABCDABCDABCDABCD";
    println!("Input: {:?} ({} bytes)", String::from_utf8_lossy(input), input.len());

    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(input).unwrap();

    println!("\nCompressed: {} bytes", compressed.len());
    print!("Hex: ");
    for b in &compressed {
        print!("{:02x} ", b);
    }
    println!();

    // Try our own decoder
    println!("\n=== Our decoder ===");
    let decompressor = ZstdDecompressor;
    match decompressor.decompress(&compressed) {
        Ok(decompressed) => {
            println!("SUCCESS: {} bytes", decompressed.len());
            if decompressed == input {
                println!("Content matches!");
            } else {
                println!("Content mismatch!");
                println!("Expected: {:?}", String::from_utf8_lossy(input));
                println!("Got: {:?}", String::from_utf8_lossy(&decompressed));
            }
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Try reference decoder
    println!("\n=== Reference decoder ===");
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(decompressed) => {
            println!("SUCCESS: {} bytes", decompressed.len());
        }
        Err(e) => println!("FAILED: {:?}", e),
    }
}

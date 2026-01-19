//! Test with larger input

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Create a larger repeating pattern that definitely compresses
    let input = b"The quick brown fox jumps over the lazy dog. ";
    let mut large_input = Vec::new();
    for _ in 0..10 {
        large_input.extend_from_slice(input);
    }
    println!("Input: {} bytes", large_input.len());

    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(&large_input).unwrap();
    println!("Compressed: {} bytes", compressed.len());

    // Our decoder
    println!("\n=== Our decoder ===");
    use haagenti_core::Decompressor;
    use haagenti_zstd::ZstdDecompressor;
    let decompressor = ZstdDecompressor;
    match decompressor.decompress(&compressed) {
        Ok(dec) => {
            println!("SUCCESS: {} bytes", dec.len());
            if dec == large_input {
                println!("Content matches!");
            } else {
                println!("Content MISMATCH!");
            }
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Reference decoder
    println!("\n=== Reference decoder ===");
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(dec) => {
            println!("SUCCESS: {} bytes", dec.len());
            if dec == large_input {
                println!("Content matches!");
            } else {
                println!("Content MISMATCH!");
            }
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Reference encoder for comparison
    println!("\n=== Reference encoder ===");
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&large_input), 1).unwrap();
    println!("Reference compressed: {} bytes", ref_compressed.len());
}

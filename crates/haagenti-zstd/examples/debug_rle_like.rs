//! Debug the RLE-like pattern issue

use haagenti_core::Compressor;
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    // The failing pattern
    let mut data = Vec::new();
    data.extend(vec![b'X'; 100]);
    data.extend(vec![b'Y'; 100]);
    data.extend(vec![b'Z'; 100]);
    data.extend(vec![b'X'; 100]);

    println!("Input size: {} bytes", data.len());
    println!("Input pattern: X*100 Y*100 Z*100 X*100");

    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(&data).unwrap();
    println!("Compressed size: {} bytes", compressed.len());
    println!(
        "Compressed hex (first 64 bytes): {:02x?}",
        &compressed[..64.min(compressed.len())]
    );

    // Try reference decompression
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(ref_decompressed) => {
            println!(
                "Reference decompression: OK, {} bytes",
                ref_decompressed.len()
            );
            if ref_decompressed == data {
                println!("Data matches original!");
            } else {
                println!("Data MISMATCH!");
            }
        }
        Err(e) => {
            println!("Reference decompression FAILED: {}", e);
        }
    }

    // Try our decompression
    let decompressor = ZstdDecompressor::new();
    match haagenti_core::Decompressor::decompress(&decompressor, &compressed) {
        Ok(our_decompressed) => {
            println!("Our decompression: OK, {} bytes", our_decompressed.len());
            if our_decompressed == data {
                println!("Data matches original!");
            } else {
                println!("Data MISMATCH!");
            }
        }
        Err(e) => {
            println!("Our decompression FAILED: {:?}", e);
        }
    }
}

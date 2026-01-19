//! Verify our compression of repeating text works with reference zstd

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    for size in [1024, 4096, 16384, 65536] {
        let data: Vec<u8> = sample.iter().cycle().take(size).copied().collect();

        // Compress with our implementation
        let compressor = ZstdCompressor::new();
        let compressed = compressor.compress(&data).unwrap();

        // Try to decompress with reference zstd
        match zstd::decode_all(std::io::Cursor::new(&compressed)) {
            Ok(decompressed) => {
                if decompressed == data {
                    println!("Size {}: PASS (compressed {} -> {} bytes)",
                             size, data.len(), compressed.len());
                } else {
                    println!("Size {}: FAIL - data mismatch", size);
                    println!("  Expected: {} bytes", data.len());
                    println!("  Got: {} bytes", decompressed.len());
                    // Find first difference
                    for i in 0..data.len().min(decompressed.len()) {
                        if data[i] != decompressed[i] {
                            println!("  First diff at {}: expected 0x{:02x}, got 0x{:02x}",
                                     i, data[i], decompressed[i]);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                println!("Size {}: FAIL - decode error: {}", size, e);
                println!("  Compressed: {} bytes", compressed.len());
                println!("  First 32 bytes: {:02x?}", &compressed[..32.min(compressed.len())]);
            }
        }
    }
}

//! Debug the RLE-like pattern issue - compare slow vs fast path

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn main() {
    // The failing pattern
    let mut data = Vec::new();
    data.extend(vec![b'X'; 100]);
    data.extend(vec![b'Y'; 100]);
    data.extend(vec![b'Z'; 100]);
    data.extend(vec![b'X'; 100]);

    println!("Input size: {} bytes", data.len());

    // Compress with our compressor
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let compressed = ctx.compress(&data).unwrap();
    println!("Compressed size: {} bytes", compressed.len());

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
                // Show difference
                let min_len = ref_decompressed.len().min(data.len());
                for i in 0..min_len {
                    if ref_decompressed[i] != data[i] {
                        println!(
                            "First diff at byte {}: got {}, expected {}",
                            i, ref_decompressed[i], data[i]
                        );
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!("Reference decompression FAILED: {}", e);
        }
    }

    // Now test with smaller input (uses slow path)
    let small_data: Vec<u8> = data.iter().take(1000).copied().collect();
    println!("\n--- Small test (slow path) ---");
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let small_compressed = ctx.compress(&small_data).unwrap();
    println!(
        "Small input {} bytes -> {} bytes",
        small_data.len(),
        small_compressed.len()
    );

    match zstd::decode_all(std::io::Cursor::new(&small_compressed)) {
        Ok(decompressed) => {
            if decompressed == small_data {
                println!("Small test: PASS");
            } else {
                println!("Small test: MISMATCH");
            }
        }
        Err(e) => {
            println!("Small test decompression FAILED: {}", e);
        }
    }
}

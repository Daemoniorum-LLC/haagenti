//! Cross-decode test

use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    // Simple pattern
    let mut data = Vec::new();
    data.extend(vec![b'X'; 100]);
    data.extend(vec![b'Y'; 100]);

    println!("Test 1: Reference compress -> Our decompress");
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();
    let our_decompressor = ZstdDecompressor::new();
    match our_decompressor.decompress(&ref_compressed) {
        Ok(dec) => {
            if dec == data {
                println!("  PASS: Our decompressor handles reference output");
            } else {
                println!("  FAIL: Decompressed but data mismatch");
            }
        }
        Err(e) => {
            println!("  FAIL: {:?}", e);
        }
    }

    println!("\nTest 2: Our compress -> Our decompress");
    let our_compressor = ZstdCompressor::new();
    let our_compressed = our_compressor.compress(&data).unwrap();
    match our_decompressor.decompress(&our_compressed) {
        Ok(dec) => {
            if dec == data {
                println!("  PASS: Our roundtrip works");
            } else {
                println!("  FAIL: Data mismatch");
            }
        }
        Err(e) => {
            println!("  FAIL: {:?}", e);
        }
    }

    println!("\nTest 3: Our compress -> Reference decompress");
    match zstd::decode_all(std::io::Cursor::new(&our_compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("  PASS: Reference handles our output");
            } else {
                println!("  FAIL: Data mismatch");
            }
        }
        Err(e) => {
            println!("  FAIL: {}", e);
        }
    }

    println!("\nTest 4: Same tests with uniform data (should work)");
    let uniform = vec![b'A'; 200];

    let ref_comp = zstd::encode_all(std::io::Cursor::new(&uniform), 1).unwrap();
    let our_comp = our_compressor.compress(&uniform).unwrap();

    println!("  Ref compress size: {} bytes", ref_comp.len());
    println!("  Our compress size: {} bytes", our_comp.len());

    match zstd::decode_all(std::io::Cursor::new(&our_comp)) {
        Ok(dec) => println!(
            "  Ref decode our: {}",
            if dec == uniform { "PASS" } else { "FAIL" }
        ),
        Err(e) => println!("  Ref decode our: FAIL - {}", e),
    }
}

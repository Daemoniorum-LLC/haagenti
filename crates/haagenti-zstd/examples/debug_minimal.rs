//! Find minimal failing case

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // Binary search for minimal failing case
    for lines in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] {
        let mut data = Vec::new();
        for i in 0..lines {
            let line = format!(
                "[2024-01-{:02} 10:{:02}:00] INFO Request #{}\n",
                (i % 28) + 1,
                i % 60,
                i * 1000
            );
            data.extend_from_slice(line.as_bytes());
        }

        let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
        let compressed = compressor.compress(&data).unwrap();

        match zstd::decode_all(Cursor::new(&compressed)) {
            Ok(dec) if dec == data => {
                println!(
                    "{:2} lines ({:4} bytes): OK ({:3} bytes compressed)",
                    lines,
                    data.len(),
                    compressed.len()
                );
            }
            Ok(_) => {
                println!(
                    "{:2} lines ({:4} bytes): CONTENT MISMATCH",
                    lines,
                    data.len()
                );
            }
            Err(e) => {
                println!(
                    "{:2} lines ({:4} bytes): FAILED - {:?}",
                    lines,
                    data.len(),
                    e
                );

                // Show the data that fails
                if lines <= 5 {
                    println!("    Data: {:?}", String::from_utf8_lossy(&data));
                }
            }
        }
    }

    // Now test with simpler patterns
    println!("\n=== Simpler patterns ===");
    for len in [100, 200, 300, 400, 500] {
        let data: Vec<u8> = (0..len).map(|i| b'a' + (i % 26) as u8).collect();

        let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
        let compressed = compressor.compress(&data).unwrap();

        match zstd::decode_all(Cursor::new(&compressed)) {
            Ok(dec) if dec == data => {
                println!(
                    "{} bytes alpha: OK ({} bytes compressed)",
                    len,
                    compressed.len()
                );
            }
            Ok(_) => {
                println!("{} bytes alpha: CONTENT MISMATCH", len);
            }
            Err(e) => {
                println!("{} bytes alpha: FAILED - {:?}", len, e);
            }
        }
    }
}

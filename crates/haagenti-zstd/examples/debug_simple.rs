//! Debug simple compression

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // Test patterns to find which fails
    let test_patterns: Vec<(&str, Vec<u8>)> = vec![
        ("simple repeat 5K", {
            let input = b"The quick brown fox jumps over the lazy dog. ";
            let mut v = Vec::new();
            for _ in 0..120 {
                v.extend_from_slice(input);
            }
            v
        }),
        ("log-like 100 lines", {
            let mut v = Vec::new();
            for i in 0..100 {
                let line = format!(
                    "[2024-01-{:02} 10:{:02}:00] INFO Processing request #{}\n",
                    (i % 28) + 1,
                    i % 60,
                    i * 1000
                );
                v.extend_from_slice(line.as_bytes());
            }
            v
        }),
        ("log-like 50 lines", {
            let mut v = Vec::new();
            for i in 0..50 {
                let line = format!(
                    "[2024-01-{:02} 10:{:02}:00] INFO Processing request #{}\n",
                    (i % 28) + 1,
                    i % 60,
                    i * 1000
                );
                v.extend_from_slice(line.as_bytes());
            }
            v
        }),
        ("sequential numbers", {
            let mut v = Vec::new();
            for i in 0..1000 {
                v.extend_from_slice(format!("Line {} of content here.\n", i).as_bytes());
            }
            v
        }),
    ];

    for (name, data) in test_patterns {
        // Try Fast level (greedy) to see if it's a lazy matching issue
        let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
        let compressed = compressor.compress(&data).unwrap();

        // Try reference decoder
        match zstd::decode_all(Cursor::new(&compressed)) {
            Ok(dec) => {
                if dec == data {
                    println!(
                        "{:25} ({:5} bytes): OK ({} bytes compressed)",
                        name,
                        data.len(),
                        compressed.len()
                    );
                } else {
                    println!("{:25} ({:5} bytes): CONTENT MISMATCH", name, data.len());
                }
            }
            Err(e) => {
                println!("{:25} ({:5} bytes): FAILED - {:?}", name, data.len(), e);
            }
        }
    }
}

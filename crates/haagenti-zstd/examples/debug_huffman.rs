//! Debug Huffman encoding

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // Test patterns with increasing entropy
    let test_patterns: Vec<(&str, Vec<u8>)> = vec![
        ("all same byte", vec![b'A'; 3000]),
        ("2 unique bytes", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push(if i % 2 == 0 { b'A' } else { b'B' });
            }
            v
        }),
        ("10 unique bytes", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push(b'0' + (i % 10) as u8);
            }
            v
        }),
        ("26 unique bytes (alpha)", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push(b'a' + (i % 26) as u8);
            }
            v
        }),
        ("64 unique bytes", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push(b' ' + (i % 64) as u8);
            }
            v
        }),
        ("128 unique bytes", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push((i % 128) as u8);
            }
            v
        }),
        ("256 unique bytes", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push((i % 256) as u8);
            }
            v
        }),
        ("random-looking but compressible", {
            let mut v = Vec::new();
            for i in 0..3000 {
                v.push(((i * 7 + 13) % 256) as u8);
            }
            v
        }),
        (
            "short log line (60 bytes)",
            b"[2024-01-01 10:00:00] INFO Processing request #12345\n".to_vec(),
        ),
        ("short log repeated", {
            let line = b"[2024-01-01 10:00:00] INFO Processing request #12345\n";
            let mut v = Vec::new();
            for _ in 0..50 {
                v.extend_from_slice(line);
            }
            v
        }),
        ("varying log lines 10", {
            let mut v = Vec::new();
            for i in 0..10 {
                let line = format!(
                    "[2024-01-{:02} 10:{:02}:00] INFO Request #{}\n",
                    (i % 28) + 1,
                    i % 60,
                    i * 1000
                );
                v.extend_from_slice(line.as_bytes());
            }
            v
        }),
        ("varying log lines 20", {
            let mut v = Vec::new();
            for i in 0..20 {
                let line = format!(
                    "[2024-01-{:02} 10:{:02}:00] INFO Request #{}\n",
                    (i % 28) + 1,
                    i % 60,
                    i * 1000
                );
                v.extend_from_slice(line.as_bytes());
            }
            v
        }),
    ];

    for (name, data) in test_patterns {
        let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
        let compressed = compressor.compress(&data).unwrap();

        match zstd::decode_all(Cursor::new(&compressed)) {
            Ok(dec) => {
                if dec == data {
                    println!(
                        "{:30} ({:5} bytes): OK ({:4} bytes)",
                        name,
                        data.len(),
                        compressed.len()
                    );
                } else {
                    println!("{:30} ({:5} bytes): CONTENT MISMATCH", name, data.len());
                }
            }
            Err(e) => {
                println!("{:30} ({:5} bytes): FAILED - {:?}", name, data.len(), e);
            }
        }
    }
}

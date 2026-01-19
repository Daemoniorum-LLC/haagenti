//! Compare compression ratios across compression levels

use haagenti_core::{CompressionLevel, Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};
use std::io::Cursor;

fn main() {
    println!("=== Compression Level Comparison ===\n");

    // Test patterns
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("repeating text", {
            let text = b"The quick brown fox jumps over the lazy dog. ";
            let mut v = Vec::new();
            for _ in 0..50 {
                v.extend_from_slice(text);
            }
            v
        }),
        ("log-like data", {
            let mut v = Vec::new();
            for i in 0..100 {
                let line = format!("[2024-01-{:02} 10:{:02}:00] INFO Processing request #{}\n",
                                  (i % 28) + 1, i % 60, i * 1000);
                v.extend_from_slice(line.as_bytes());
            }
            v
        }),
        ("JSON data", {
            let mut v = Vec::new();
            v.extend_from_slice(b"[");
            for i in 0..50 {
                if i > 0 { v.extend_from_slice(b","); }
                let obj = format!(r#"{{"id":{},"name":"user{}","email":"user{}@example.com"}}"#, i, i, i);
                v.extend_from_slice(obj.as_bytes());
            }
            v.extend_from_slice(b"]");
            v
        }),
        ("HTML snippet", {
            let mut v = Vec::new();
            for _ in 0..20 {
                v.extend_from_slice(b"<div class=\"container\"><h1>Title</h1><p>Content here</p></div>\n");
            }
            v
        }),
    ];

    let levels = [
        ("Fast", CompressionLevel::Fast),
        ("Default", CompressionLevel::Default),
        ("Best", CompressionLevel::Best),
    ];

    println!("{:20} {:>8} {:>10} {:>10} {:>10} {:>10}",
             "Pattern", "Size", "Fast", "Default", "Best", "Ref");
    println!("{:-<70}", "");

    let decompressor = ZstdDecompressor;

    for (name, data) in &patterns {
        print!("{:20} {:8}", name, data.len());

        let ref_size = zstd::encode_all(Cursor::new(data), 1).unwrap().len();

        for (level_name, level) in &levels {
            let compressor = ZstdCompressor::with_level(*level);
            let compressed = compressor.compress(data).unwrap();

            // Verify decompression works
            let decompressed = decompressor.decompress(&compressed).unwrap();
            assert_eq!(&decompressed, data, "{} {} decompression mismatch", name, level_name);

            let ratio = compressed.len() as f64 / data.len() as f64 * 100.0;
            print!(" {:9.1}%", ratio);
        }

        let ref_ratio = ref_size as f64 / data.len() as f64 * 100.0;
        print!(" {:9.1}%", ref_ratio);
        println!();
    }

    println!("\n=== Summary ===");
    println!("- Fast: Greedy matching, fastest speed");
    println!("- Default: Lazy matching, balanced");
    println!("- Best: Repeat offset-aware matching, best ratio");
    println!("- Ref: Reference zstd at level 1");
}

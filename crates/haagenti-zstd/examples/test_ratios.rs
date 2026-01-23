use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    println!("\n=== Compression Ratio Test ===\n");

    // Test cases
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("Uniform A*1000", vec![b'A'; 1000]),
        (
            "Pattern ABCD*250",
            (0..250).flat_map(|_| b"ABCD".iter().copied()).collect(),
        ),
        ("RLE-like", {
            let mut v = Vec::new();
            v.extend(vec![b'X'; 100]);
            v.extend(vec![b'Y'; 100]);
            v.extend(vec![b'Z'; 100]);
            v.extend(vec![b'X'; 100]);
            v
        }),
        (
            "Text-like",
            "The quick brown fox jumps over the lazy dog. "
                .repeat(20)
                .into_bytes(),
        ),
        ("Binary seq", (0u8..=255).cycle().take(1000).collect()),
        ("Mixed X*20 Y*20", {
            let mut v = Vec::new();
            for _ in 0..10 {
                v.extend(vec![b'X'; 20]);
                v.extend(vec![b'Y'; 20]);
            }
            v
        }),
    ];

    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor::new();

    for (name, data) in test_cases {
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = decompressor.decompress(&compressed).unwrap();

        let ratio = data.len() as f64 / compressed.len() as f64;
        let correct = decompressed == data;

        println!(
            "{:20} | Input: {:5} | Compressed: {:5} | Ratio: {:6.2}x | Correct: {}",
            name,
            data.len(),
            compressed.len(),
            ratio,
            if correct { "✓" } else { "✗" }
        );
    }

    println!();
}

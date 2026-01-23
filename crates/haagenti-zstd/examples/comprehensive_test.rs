//! Comprehensive compression test

use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    println!("=== Comprehensive Compression Test ===\n");

    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("single byte", vec![b'A']),
        ("small repeating", b"ABCDABCDABCDABCD".to_vec()),
        (
            "lorem ipsum",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".to_vec(),
        ),
        ("binary zeros", vec![0u8; 1000]),
        (
            "random-ish",
            (0..256).cycle().take(1000).map(|x| x as u8).collect(),
        ),
        ("long text pattern", {
            let input = b"The quick brown fox jumps over the lazy dog. ";
            let mut v = Vec::new();
            for _ in 0..50 {
                v.extend_from_slice(input);
            }
            v
        }),
        ("64KB zeros", vec![0u8; 65536]),
        ("64KB pattern", {
            let mut v = Vec::with_capacity(65536);
            for i in 0..65536 {
                v.push((i % 256) as u8);
            }
            v
        }),
    ];

    let compressor = ZstdCompressor::new();
    let decompressor = ZstdDecompressor;

    let mut all_passed = true;

    for (name, input) in &test_cases {
        print!("{:25} ({:6} bytes): ", name, input.len());

        // Compress
        let compressed = match compressor.compress(input) {
            Ok(c) => c,
            Err(e) => {
                println!("COMPRESS FAILED: {:?}", e);
                all_passed = false;
                continue;
            }
        };

        let ratio = if input.is_empty() {
            0.0
        } else {
            (compressed.len() as f64 / input.len() as f64) * 100.0
        };

        // Decompress with our decoder
        let our_result = decompressor.decompress(&compressed);

        // Decompress with reference decoder
        let ref_result = zstd::decode_all(std::io::Cursor::new(&compressed));

        match (&our_result, &ref_result) {
            (Ok(ours), Ok(refs)) if ours == input && refs == input => {
                print!("OK (ratio: {:5.1}%)", ratio);

                // Also check reference encoder produces decodable output
                if let Ok(ref_compressed) = zstd::encode_all(std::io::Cursor::new(input), 1) {
                    let ref_ratio = if input.is_empty() {
                        0.0
                    } else {
                        (ref_compressed.len() as f64 / input.len() as f64) * 100.0
                    };
                    let gap = ratio - ref_ratio;
                    print!(", ref: {:5.1}%, gap: {:+.1}%", ref_ratio, gap);
                }
                println!();
            }
            (Ok(ours), Ok(refs)) => {
                println!("CONTENT MISMATCH!");
                if ours != input {
                    println!("  Our decoder: wrong content");
                }
                if refs != input {
                    println!("  Ref decoder: wrong content");
                }
                all_passed = false;
            }
            (Err(e), _) => {
                println!("OUR DECODER FAILED: {:?}", e);
                all_passed = false;
            }
            (_, Err(e)) => {
                println!("REF DECODER FAILED: {:?}", e);
                all_passed = false;
            }
        }
    }

    println!(
        "\n{}",
        if all_passed {
            "ALL TESTS PASSED!"
        } else {
            "SOME TESTS FAILED!"
        }
    );
}

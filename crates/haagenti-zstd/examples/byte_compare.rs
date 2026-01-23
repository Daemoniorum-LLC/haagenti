//! Byte-by-byte comparison of our output vs reference

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Use same input
    let input = b"The quick brown fox jumps over the lazy dog. ";
    let mut large_input = Vec::new();
    for _ in 0..10 {
        large_input.extend_from_slice(input);
    }

    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(&large_input).unwrap();
    let reference = zstd::encode_all(std::io::Cursor::new(&large_input), 1).unwrap();

    println!("Ours: {} bytes", ours.len());
    println!("Ref:  {} bytes", reference.len());

    println!("\n=== Byte comparison (first difference highlighted) ===\n");

    let max_len = ours.len().max(reference.len());
    let mut first_diff = None;

    for i in 0..max_len {
        let o = ours.get(i);
        let r = reference.get(i);

        let marker = if o != r && first_diff.is_none() {
            first_diff = Some(i);
            " <-- FIRST DIFF"
        } else if o != r {
            " <-- diff"
        } else {
            ""
        };

        match (o, r) {
            (Some(ob), Some(rb)) => {
                println!("{:04x}: ours={:02x}  ref={:02x}{}", i, ob, rb, marker);
            }
            (Some(ob), None) => {
                println!("{:04x}: ours={:02x}  ref=--{}", i, ob, marker);
            }
            (None, Some(rb)) => {
                println!("{:04x}: ours=--  ref={:02x}{}", i, rb, marker);
            }
            (None, None) => break,
        }
    }

    if let Some(idx) = first_diff {
        println!(
            "\n=== First difference at offset 0x{:04x} ({}) ===",
            idx, idx
        );

        // Context around the diff
        let start = idx.saturating_sub(5);
        let end = (idx + 10).min(max_len);

        println!("\nOurs around diff:");
        for i in start..end {
            if let Some(b) = ours.get(i) {
                print!("{:02x} ", b);
            }
        }
        println!();

        println!("\nRef around diff:");
        for i in start..end {
            if let Some(b) = reference.get(i) {
                print!("{:02x} ", b);
            }
        }
        println!();
    }
}

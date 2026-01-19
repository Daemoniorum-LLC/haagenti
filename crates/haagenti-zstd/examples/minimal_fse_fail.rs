//! Minimal test case for FSE encoding failure with non-uniform sequences

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;
use haagenti_zstd::compress::{LazyMatchFinder, block, analyze_for_rle, EncodedSequence};

fn main() {
    // Create data that produces NON-uniform sequences
    // We need different literal lengths, offsets, or match lengths
    let mut data = Vec::new();

    // Pattern 1: "ABCD" repeated (offset 4)
    for _ in 0..10 {
        data.extend_from_slice(b"ABCD");
    }
    // Add some different pattern
    data.extend_from_slice(b"XYZ");
    // Pattern 2: "EFGH" repeated (offset 4 but different content)
    for _ in 0..10 {
        data.extend_from_slice(b"EFGH");
    }

    println!("Input: {} bytes", data.len());

    // Get matches
    let mut mf = LazyMatchFinder::new(24);
    let matches = mf.find_matches_auto(&data);
    println!("Matches: {}", matches.len());

    for (i, m) in matches.iter().enumerate() {
        println!("  Match {}: pos={}, offset={}, len={}", i, m.position, m.offset, m.length);
    }

    // Convert to sequences
    let (literals, seqs) = block::matches_to_sequences(&data, &matches);
    println!("\nLiterals: {} bytes", literals.len());
    println!("Sequences: {}", seqs.len());

    for (i, s) in seqs.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(s);
        println!("  Seq {}: ll={}, offset={}, ml={} -> ll_code={}, of_code={}, ml_code={}",
                 i, s.literal_length, s.offset, s.match_length,
                 enc.ll_code, enc.of_code, enc.ml_code);
    }

    // Check uniformity
    let suitability = analyze_for_rle(&seqs);
    println!("\nRLE suitability:");
    println!("  LL uniform: {} (code {})", suitability.ll_uniform, suitability.ll_code);
    println!("  OF uniform: {} (code {})", suitability.of_uniform, suitability.of_code);
    println!("  ML uniform: {} (code {})", suitability.ml_uniform, suitability.ml_code);
    println!("  All uniform: {}", suitability.all_uniform());

    // Compress
    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(&data).unwrap();
    println!("\nCompressed: {} bytes", compressed.len());
    println!("Bytes: {:02x?}", &compressed[..50.min(compressed.len())]);

    // Try to decode with reference
    println!("\nReference decode:");
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(decompressed) => {
            if decompressed == data {
                println!("  SUCCESS!");
            } else {
                println!("  MISMATCH: {} vs {} bytes", decompressed.len(), data.len());
            }
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }
}

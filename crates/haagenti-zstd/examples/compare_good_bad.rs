//! Compare working vs failing case

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::MatchFinder;
use std::io::Cursor;

fn test_pattern(name: &str, data: &[u8]) {
    println!("\n=== {} ({} bytes) ===", name, data.len());

    // Find matches
    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    println!("Matches: {}", matches.len());

    // Convert to sequences
    let (literals, sequences) = matches_to_sequences(data, &matches);
    println!("Literals: {}, Sequences: {}", literals.len(), sequences.len());

    // Compress
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();

    // Decode
    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) if dec == data => println!("Result: OK ({} -> {} bytes)", data.len(), compressed.len()),
        Ok(_) => println!("Result: MISMATCH"),
        Err(e) => println!("Result: FAILED - {:?}", e),
    }
}

fn main() {
    // Working: simple repeating pattern
    let working = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. ";
    test_pattern("Simple repeat (WORKS)", working);

    // Failing: 2 log lines
    let failing = b"[2024-01-01 10:00:00] INFO Request #0\n[2024-01-02 10:01:00] INFO Request #1000\n";
    test_pattern("Log lines (FAILS)", failing);

    // What's different about log lines?
    // 1. More varied offsets
    // 2. Repeat offsets (offset=1 encoded)
    // 3. Different literal lengths

    // Test with 1 log line (should work)
    let one_line = b"[2024-01-01 10:00:00] INFO Request #0\n";
    test_pattern("One log line", one_line);

    // Test with 2 similar lines (same length, different content)
    let two_same_len = b"AAAAABBBBBCCCCCDDDDDEEEEE\nAAAABBBBBCCCCCDDDDEEEEEE\n";
    test_pattern("Two similar lines", two_same_len);

    // Test with pattern that has offset 1 (RLE-like)
    let rle_like = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    test_pattern("RLE pattern", rle_like);

    // Test with varied short matches
    let short_matches = b"abcabc123abc456abcxyzabc";
    test_pattern("Short matches", short_matches);
}

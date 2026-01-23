//! Test with exactly 2 sequences

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::MatchFinder;
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn test_pattern(name: &str, data: &[u8]) {
    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    let (literals, sequences) = matches_to_sequences(data, &matches);

    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();

    let result = match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) if dec == data => "OK".to_string(),
        Ok(_) => "MISMATCH".to_string(),
        Err(e) => format!("FAILED"),
    };

    println!(
        "{:40} matches={}, seqs={}, result={}",
        name,
        matches.len(),
        sequences.len(),
        result
    );
}

fn main() {
    println!(
        "{:40} {:>8} {:>5} {:>10}",
        "Pattern", "matches", "seqs", "result"
    );
    println!("{:-<70}", "");

    // Need to craft patterns with exact number of matches
    // Pattern with 2 matches: "abcdabcdXYZabcd" - matches at pos 4 and 11
    test_pattern("2 matches: abcdabcdXYZabcd", b"abcdabcdXYZabcd");

    // Pattern with 2 matches different lengths
    test_pattern(
        "2 matches: abcdefghXabcdefghYabcd",
        b"abcdefghXabcdefghYabcd",
    );

    // Try progressively more matches
    for n in 1..=5 {
        let mut data = Vec::new();
        for i in 0..=n {
            data.extend_from_slice(b"PATTERN");
            if i < n {
                data.push(b'0' + i as u8);
            }
        }
        test_pattern(&format!("{} repeats of PATTERN", n + 1), &data);
    }

    // Try pattern that we know works with 1 match
    test_pattern("1 match: foofoo", b"foofoo");

    // Extend to 2 matches
    test_pattern("2 matches: foofoofoo", b"foofoofoo");
    test_pattern("3 matches: foofoofoofoo", b"foofoofoofoo");
}

//! Compare working vs broken cases

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::MatchFinder;
use haagenti_zstd::compress::EncodedSequence;
use std::io::Cursor;

fn test_case(name: &str, data: &[u8]) {
    println!("\n{}", "=".repeat(60));
    println!("=== {} ({} bytes) ===", name, data.len());
    println!("{}", "=".repeat(60));

    // Find matches
    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    
    // Convert to sequences
    let (literals, sequences) = matches_to_sequences(data, &matches);
    println!("Matches: {}, Sequences: {}, Literals: {}", 
             matches.len(), sequences.len(), literals.len());

    // Show encoded sequences
    for (i, seq) in sequences.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(seq);
        println!("  Seq {}: LL={} (code={}), OF={} (code={}, {}b), ML={} (code={})",
                 i, seq.literal_length, enc.ll_code, 
                 seq.offset, enc.of_code, enc.of_bits,
                 seq.match_length, enc.ml_code);
    }

    // Compress
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();
    println!("\nCompressed: {} bytes", compressed.len());
    
    // Decode
    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) if dec == data => println!("Result: OK"),
        Ok(dec) => println!("Result: MISMATCH (got {} bytes)", dec.len()),
        Err(e) => {
            println!("Result: FAILED");
            
            // Show compressed hex for debugging
            println!("\nCompressed hex:");
            for (i, chunk) in compressed.chunks(16).enumerate() {
                print!("  {:04x}: ", i * 16);
                for b in chunk {
                    print!("{:02x} ", b);
                }
                println!();
            }
        }
    }
}

fn main() {
    // Working: 2 sequences with different offset codes, no repeat offset
    test_case("WORKS: 2 seq, no repeat", b"abcdabcdXYZabcd");
    
    // Broken: 2 sequences where seq[1] uses repeat offset  
    test_case("FAILS: 2 seq, with repeat", b"abcdefghXabcdefghYabcd");
    
    // Working: 1 sequence only
    test_case("WORKS: 1 seq only", b"PATTERN0PATTERN");
    
    // Broken: 2 sequences with repeat offset
    test_case("FAILS: 2 seq, repeat offset", b"PATTERN0PATTERN1PATTERN");
    
    // Let's try something similar to working case but adjusted
    // Working case has offsets that don't match repeat offsets
    test_case("TEST: Force different offsets", b"abcdXXXXabcdYYYYYYabcd");
}

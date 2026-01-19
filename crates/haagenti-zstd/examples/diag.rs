use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::{
    CompressContext, CompressibilityFingerprint, MatchFinder,
    analyze_for_rle, encode_sequences_rle, encode_sequences_fse,
};
use haagenti_zstd::compress::block::{matches_to_sequences, encode_literals, encode_sequences};

fn main() {
    // Generate test data
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::new();
    while data.len() < 4096 {  // Use 4KB for clearer output
        data.extend_from_slice(pattern);
    }
    data.truncate(4096);
    
    println!("=== Input Analysis ===");
    println!("Size: {} bytes", data.len());
    println!("Pattern repeats: {}", data.len() / pattern.len());
    
    // Step 1: Fingerprint analysis
    let fingerprint = CompressibilityFingerprint::analyze(&data);
    println!("\n=== Fingerprint ===");
    println!("{:?}", fingerprint);
    
    // Step 2: Match finding
    let mut mf = MatchFinder::new(16);
    let matches = mf.find_matches(&data);
    println!("\n=== Match Finding ===");
    println!("Matches found: {}", matches.len());
    if !matches.is_empty() {
        println!("First 5 matches:");
        for m in matches.iter().take(5) {
            println!("  pos={} offset={} len={}", m.position, m.offset, m.length);
        }
        let total_match_bytes: usize = matches.iter().map(|m| m.length).sum();
        println!("Total match bytes: {} ({:.1}% of input)", 
            total_match_bytes, 
            total_match_bytes as f64 / data.len() as f64 * 100.0);
    }
    
    // Step 3: Convert to sequences
    let (literals, sequences) = matches_to_sequences(&data, &matches);
    println!("\n=== Sequences ===");
    println!("Literals: {} bytes", literals.len());
    println!("Sequences: {} count", sequences.len());
    if !sequences.is_empty() {
        println!("First 5 sequences:");
        for s in sequences.iter().take(5) {
            println!("  ll={} offset={} ml={}", s.literal_length, s.offset, s.match_length);
        }
    }
    
    // Step 4: Check RLE suitability
    let rle_suit = analyze_for_rle(&sequences);
    println!("\n=== RLE Analysis ===");
    println!("{:?}", rle_suit);
    println!("All uniform: {}", rle_suit.all_uniform());
    
    // Step 5: Try encoding literals
    let mut literals_encoded = Vec::new();
    encode_literals(&literals, &mut literals_encoded).unwrap();
    println!("\n=== Literals Encoding ===");
    println!("Original: {} bytes", literals.len());
    println!("Encoded: {} bytes", literals_encoded.len());
    
    // Step 6: Try encoding sequences with RLE
    let mut seq_encoded = Vec::new();
    if rle_suit.all_uniform() && !sequences.is_empty() {
        if let Ok(()) = encode_sequences_rle(&sequences, &rle_suit, &mut seq_encoded) {
            println!("\n=== RLE Sequence Encoding ===");
            println!("Sequences encoded: {} bytes", seq_encoded.len());
        } else {
            println!("\nRLE sequence encoding failed");
        }
    } else {
        // Try FSE
        if encode_sequences_fse(&sequences, &mut seq_encoded).is_ok() {
            println!("\n=== FSE Sequence Encoding ===");
            println!("Sequences encoded: {} bytes", seq_encoded.len());
        } else {
            println!("\nFSE sequence encoding failed");
        }
    }
    
    // Step 7: Total compressed size
    let total_compressed = literals_encoded.len() + seq_encoded.len();
    println!("\n=== Summary ===");
    println!("Input size: {} bytes", data.len());
    println!("Total compressed: {} bytes", total_compressed);
    println!("Expected ratio: {:.2}x", data.len() as f64 / total_compressed as f64);
    
    if total_compressed >= data.len() {
        println!("\n>>> PROBLEM: Compressed size >= input size, will fall back to raw!");
    }
}

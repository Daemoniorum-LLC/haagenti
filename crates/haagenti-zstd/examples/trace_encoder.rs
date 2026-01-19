//! Trace encoder output for 500-byte pattern

use haagenti_zstd::compress::SpeculativeCompressor;

fn main() {
    let input: Vec<u8> = b"ABCD".iter().cycle().take(500).copied().collect();
    println!("Input: {} bytes", input.len());
    
    // Reference compression
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("\nReference compressed: {} bytes", ref_compressed.len());
    println!("Hex: {:02x?}", ref_compressed);
    
    // Our compression
    let compressor = SpeculativeCompressor::new();
    let our_compressed = compressor.compress(&input).unwrap();
    println!("\nOur compressed: {} bytes", our_compressed.len());
    println!("Hex: {:02x?}", our_compressed);
    
    // Compare FSE bitstreams
    let ref_fse = &ref_compressed[16..];
    let our_fse = &our_compressed[16..];
    println!("\nFSE comparison:");
    println!("  Reference: {:02x?}", ref_fse);
    println!("  Ours:      {:02x?}", our_fse);
    
    // Parse both as u32 LE
    let ref_u32 = u32::from_le_bytes([ref_fse[0], ref_fse[1], ref_fse[2], ref_fse[3]]);
    let our_u32 = u32::from_le_bytes([our_fse[0], our_fse[1], our_fse[2], our_fse[3]]);
    println!("\n  Reference bits: {:032b}", ref_u32);
    println!("  Our bits:       {:032b}", our_u32);
    
    // Find sentinels
    let ref_sentinel = 31 - ref_u32.leading_zeros();
    let our_sentinel = 31 - our_u32.leading_zeros();
    println!("\n  Reference sentinel: bit {}", ref_sentinel);
    println!("  Our sentinel: bit {}", our_sentinel);
    
    // What does our encoder think it's encoding?
    // Check the match finder output
    let mut mf = haagenti_zstd::compress::LazyMatchFinder::new(16);
    let matches = mf.find_matches(&input);
    println!("\n=== Match Finder Output ===");
    for m in &matches {
        println!("  Match: pos={}, len={}, offset={}", m.position, m.length, m.offset);
    }
    
    // Check sequences
    let (literals, seqs) = haagenti_zstd::compress::block::matches_to_sequences(&input, &matches);
    println!("\n=== Sequences ===");
    println!("  Literals: {} bytes", literals.len());
    for (i, s) in seqs.iter().enumerate() {
        println!("  Seq[{}]: ll={}, ml={}, offset={}", i, s.literal_length, s.match_length, s.offset);
    }
    
    // Check encoded sequences
    use haagenti_zstd::compress::EncodedSequence;
    for (i, s) in seqs.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(s);
        println!("\n  Encoded[{}]:", i);
        println!("    LL: code={}, extra={} ({} bits)", enc.ll_code, enc.ll_extra, enc.ll_bits);
        println!("    OF: code={}, extra={} ({} bits)", enc.of_code, enc.of_extra, enc.of_bits);
        println!("    ML: code={}, extra={} ({} bits)", enc.ml_code, enc.ml_extra, enc.ml_bits);
    }
}

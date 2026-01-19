//! Debug FSE encoding for failing case

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::MatchFinder;
use haagenti_zstd::compress::{encode_sequences_fse, EncodedSequence};
use std::io::Cursor;

fn main() {
    // Minimal failing case: 2 log lines
    let data = b"[2024-01-01 10:00:00] INFO Request #0\n[2024-01-02 10:01:00] INFO Request #1000\n";
    println!("Input: {} bytes", data.len());

    // Find matches
    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    println!("\nMatches: {}", matches.len());
    for m in &matches {
        println!("  pos={}, offset={}, len={}", m.position, m.offset, m.length);
    }

    // Convert to sequences
    let (literals, sequences) = matches_to_sequences(data, &matches);
    println!("\nLiterals: {} bytes", literals.len());
    println!("Sequences: {}", sequences.len());

    // Encode each sequence
    println!("\nEncoded sequences:");
    for (i, seq) in sequences.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(seq);
        println!("  Seq {}: LL={} (code={}, extra={}, bits={}), OF={} (code={}, extra={}, bits={}), ML={} (code={}, extra={}, bits={})",
                 i,
                 seq.literal_length, enc.ll_code, enc.ll_extra, enc.ll_bits,
                 seq.offset, enc.of_code, enc.of_extra, enc.of_bits,
                 seq.match_length, enc.ml_code, enc.ml_extra, enc.ml_bits);
    }

    // Encode with FSE
    let mut output = Vec::new();
    encode_sequences_fse(&sequences, &mut output).unwrap();
    println!("\nFSE encoded: {} bytes", output.len());
    println!("Hex: {:02x?}", &output);

    // Parse the output
    if !output.is_empty() {
        let count = output[0];
        println!("\nParsed:");
        println!("  Sequence count: {}", count);
        if output.len() > 1 {
            let mode = output[1];
            println!("  Mode byte: 0x{:02x}", mode);
            println!("    LL mode: {}", mode & 3);
            println!("    OF mode: {}", (mode >> 2) & 3);
            println!("    ML mode: {}", (mode >> 4) & 3);
        }
    }

    // Now test full compression
    println!("\n=== Full Compression Test ===");
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();
    println!("Compressed: {} bytes", compressed.len());

    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("Reference decoder: OK");
            } else {
                println!("Reference decoder: CONTENT MISMATCH");
            }
        }
        Err(e) => {
            println!("Reference decoder: FAILED - {:?}", e);
        }
    }

    // Compare with reference encoder
    println!("\n=== Reference Encoder ===");
    let ref_compressed = zstd::encode_all(Cursor::new(data.as_slice()), 1).unwrap();
    println!("Reference compressed: {} bytes", ref_compressed.len());
    println!("Reference hex: {:02x?}", &ref_compressed);

    // Both should decode to same data
    let ref_dec = zstd::decode_all(Cursor::new(&ref_compressed)).unwrap();
    assert_eq!(ref_dec.as_slice(), data);
    println!("Reference round-trip: OK");
}

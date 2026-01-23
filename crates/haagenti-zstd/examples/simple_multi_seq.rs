//! Simple multi-sequence tests

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // Test 1: Small input that produces 1 sequence
    let input1 = b"ABCDABCDABCDABCD";
    test_encoding("1-seq (16 bytes)", input1);

    // Test 2: Slightly larger input
    let input2 = b"ABCDEFGHABCDEFGHABCDEFGHABCDEFGH";
    test_encoding("1-seq (32 bytes)", input2);

    // Test 3: 2 sequences
    let mut input3 = Vec::new();
    for _ in 0..10 {
        input3.extend_from_slice(b"ABCD");
    }
    input3.extend_from_slice(b"XXXX");
    for _ in 0..5 {
        input3.extend_from_slice(b"EFGH");
    }
    test_encoding("2-seq (64 bytes)", &input3);

    // Test 4: 3 sequences (like our failing case but smaller)
    let mut input4 = Vec::new();
    for _ in 0..10 {
        input4.extend_from_slice(b"ABCD");
    }
    input4.extend_from_slice(b"XX");
    for _ in 0..8 {
        input4.extend_from_slice(b"EFGH");
    }
    input4.extend_from_slice(b"YY");
    for _ in 0..5 {
        input4.extend_from_slice(b"IJKL");
    }
    test_encoding("3-seq (96 bytes)", &input4);
}

fn test_encoding(name: &str, input: &[u8]) {
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(input).unwrap();

    // Get reference compression for comparison
    let ref_compressed = zstd::encode_all(Cursor::new(input), 1).unwrap();

    // Parse block info
    let block_header =
        compressed[6] as u32 | ((compressed[7] as u32) << 8) | ((compressed[8] as u32) << 16);
    let block_type = (block_header >> 1) & 0x03;
    let block_size = block_header >> 3;

    let seq_count = if block_type == 2 {
        // Find sequence section
        let block_start = 9;
        let lit_header = compressed[block_start];
        let lit_size_type = (lit_header >> 2) & 0x03;
        let (lit_header_size, lit_size) = match lit_size_type {
            0 => (1, (lit_header >> 3) as usize),
            1 => (
                2,
                ((lit_header >> 4) as usize) | ((compressed[block_start + 1] as usize) << 4),
            ),
            _ => (3, 0),
        };
        let seq_start = block_start + lit_header_size + lit_size;
        compressed[seq_start] as usize
    } else {
        0
    };

    // Try reference decode
    let result = match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(decoded) if decoded == input => "OK",
        Ok(_) => "MISMATCH",
        Err(_) => "FAILED",
    };

    println!(
        "{}: {} ({} seqs, block_type={}, {} bytes compressed)",
        name,
        result,
        seq_count,
        block_type,
        compressed.len()
    );

    // On failure, show detailed comparison
    if result == "FAILED" && seq_count >= 2 {
        println!("  Our bitstream:  {:02x?}", &compressed[9..]);
        println!("  Ref bitstream:  {:02x?}", &ref_compressed[9..]);

        // Show sequence section specifically
        let block_start = 9;
        let lit_header = compressed[block_start];
        let lit_size_type = (lit_header >> 2) & 0x03;
        let (lit_header_size, lit_size) = match lit_size_type {
            0 => (1, (lit_header >> 3) as usize),
            1 => (
                2,
                ((lit_header >> 4) as usize) | ((compressed[block_start + 1] as usize) << 4),
            ),
            _ => (3, 0),
        };
        let seq_start = block_start + lit_header_size + lit_size;
        let seq_section = &compressed[seq_start..];
        println!("  Our seq section: {:02x?}", seq_section);

        // Same for reference
        let ref_lit_header = ref_compressed[block_start];
        let ref_lit_size_type = (ref_lit_header >> 2) & 0x03;
        let (ref_lit_header_size, ref_lit_size) = match ref_lit_size_type {
            0 => (1, (ref_lit_header >> 3) as usize),
            1 => (
                2,
                ((ref_lit_header >> 4) as usize)
                    | ((ref_compressed[block_start + 1] as usize) << 4),
            ),
            _ => (3, 0),
        };
        let ref_seq_start = block_start + ref_lit_header_size + ref_lit_size;
        let ref_block_header = ref_compressed[6] as u32
            | ((ref_compressed[7] as u32) << 8)
            | ((ref_compressed[8] as u32) << 16);
        let ref_block_type = (ref_block_header >> 1) & 0x03;
        if ref_block_type == 2 {
            let ref_seq_section = &ref_compressed[ref_seq_start..];
            println!("  Ref seq section: {:02x?}", ref_seq_section);
        } else {
            println!("  Ref uses block_type={}", ref_block_type);
        }
    }
}

//! Full debug of FSE encoding/decoding for failing case

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::compress::MatchFinder;
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // The minimal failing case: 2 sequences where seq[1] uses repeat offset
    let data = b"PATTERN0PATTERN1PATTERN";
    println!(
        "=== Input: {:?} ({} bytes) ===\n",
        String::from_utf8_lossy(data),
        data.len()
    );

    // Find matches
    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    println!("Matches:");
    for m in &matches {
        println!(
            "  pos={}, offset={}, len={}",
            m.position, m.offset, m.length
        );
    }

    // Convert to sequences
    let (literals, sequences) = matches_to_sequences(data, &matches);
    println!("\nSequences ({}):", sequences.len());
    for (i, seq) in sequences.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(seq);
        println!("  Seq {}: LL={} (code={}, extra={}, bits={}), OF={} (code={}, extra={}, bits={}), ML={} (code={}, extra={}, bits={})",
                 i,
                 seq.literal_length, enc.ll_code, enc.ll_extra, enc.ll_bits,
                 seq.offset, enc.of_code, enc.of_extra, enc.of_bits,
                 seq.match_length, enc.ml_code, enc.ml_extra, enc.ml_bits);
    }

    println!("\n=== Compressing ===");
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();
    println!("Compressed size: {} bytes", compressed.len());
    println!("Compressed hex: {:02x?}", &compressed);

    // Parse the frame manually
    println!("\n=== Parsing frame ===");
    if compressed.len() >= 4 {
        let magic =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        println!("Magic: 0x{:08x}", magic);

        // Frame header
        if compressed.len() >= 5 {
            let fhd = compressed[4];
            println!("Frame_Header_Descriptor: 0x{:02x}", fhd);
            let fcs_flag = (fhd >> 6) & 3;
            let single_segment = (fhd >> 5) & 1;
            let content_checksum = (fhd >> 2) & 1;
            let dict_id_flag = fhd & 3;
            println!("  Frame_Content_Size_flag: {}", fcs_flag);
            println!("  Single_Segment_flag: {}", single_segment);
            println!("  Content_Checksum_flag: {}", content_checksum);
            println!("  Dictionary_ID_flag: {}", dict_id_flag);

            // Calculate frame header size
            let mut header_size = 1; // FHD
            if single_segment == 0 {
                header_size += 1;
            } // Window_Descriptor
            header_size += match dict_id_flag {
                0 => 0,
                1 => 1,
                2 => 2,
                _ => 4,
            };
            header_size += match (fcs_flag, single_segment) {
                (0, 1) => 1,
                (1, _) => 2,
                (2, _) => 4,
                (3, _) => 8,
                _ => 0,
            };

            let block_start = 4 + header_size;
            if compressed.len() > block_start + 3 {
                let block_header = u32::from_le_bytes([
                    compressed[block_start],
                    compressed[block_start + 1],
                    compressed[block_start + 2],
                    0,
                ]);
                let last_block = block_header & 1;
                let block_type = (block_header >> 1) & 3;
                let block_size = block_header >> 3;
                println!(
                    "\nBlock header at offset {}: 0x{:06x}",
                    block_start, block_header
                );
                println!("  Last_Block: {}", last_block);
                println!(
                    "  Block_Type: {} ({})",
                    block_type,
                    match block_type {
                        0 => "Raw",
                        1 => "RLE",
                        2 => "Compressed",
                        _ => "Reserved",
                    }
                );
                println!("  Block_Size: {}", block_size);

                // Block content starts after 3-byte header
                let content_start = block_start + 3;
                if block_type == 2 && compressed.len() > content_start {
                    println!("\nBlock content hex: {:02x?}", &compressed[content_start..]);

                    // Parse literals header
                    if compressed.len() > content_start {
                        let lit_header0 = compressed[content_start];
                        let lit_type = lit_header0 & 3;
                        let size_format = (lit_header0 >> 2) & 3;
                        println!("\nLiterals header byte 0: 0x{:02x}", lit_header0);
                        println!(
                            "  Literals_Block_Type: {} ({})",
                            lit_type,
                            match lit_type {
                                0 => "Raw",
                                1 => "RLE",
                                2 => "Compressed",
                                _ => "Treeless",
                            }
                        );
                        println!("  Size_Format: {}", size_format);
                    }
                }
            }
        }
    }

    // Try decoding
    println!("\n=== Decoding ===");
    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("SUCCESS: Decoded matches input!");
            } else {
                println!("MISMATCH!");
                println!("  Input:  {:?}", String::from_utf8_lossy(data));
                println!("  Output: {:?}", String::from_utf8_lossy(&dec));
            }
        }
        Err(e) => {
            println!("FAILED: {:?}", e);
        }
    }

    // Also test with reference encoder for comparison
    println!("\n=== Reference encoder ===");
    let ref_compressed = zstd::encode_all(Cursor::new(data.as_slice()), 1).unwrap();
    println!("Reference compressed size: {} bytes", ref_compressed.len());
    println!("Reference hex: {:02x?}", &ref_compressed);
}

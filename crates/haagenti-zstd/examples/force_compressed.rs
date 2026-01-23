//! Find an input where reference zstd uses a Compressed block.

fn main() {
    // Try various inputs to find one where reference uses Compressed block
    let inputs: Vec<&[u8]> = vec![
        b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD",
        b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        b"aaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccdddd",
    ];

    for input in inputs {
        println!(
            "\n=== Testing: \"{}...\" ({} bytes) ===",
            String::from_utf8_lossy(&input[..30.min(input.len())]),
            input.len()
        );

        // Get reference compression
        let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();

        // Check block type
        let fhd = ref_compressed[4];
        let single_segment = (fhd & 0x20) != 0;
        let fcs_field = if single_segment { 1 } else { 0 };
        let mut pos = 5 + fcs_field;
        if !single_segment {
            pos += 1;
        }

        // For single segment frames, FCS comes right after FHD
        // Skip FCS field
        let fcs_size = match fhd >> 6 {
            0 => {
                if single_segment {
                    1
                } else {
                    0
                }
            }
            1 => 2,
            2 => 4,
            3 => 8,
            _ => 0,
        };
        if single_segment {
            pos = 5 + fcs_size;
        } else {
            pos = 5 + 1 + fcs_size; // FHD + window + FCS
        }

        if pos + 3 > ref_compressed.len() {
            println!("  Frame too short");
            continue;
        }

        let bh = u32::from_le_bytes([
            ref_compressed[pos],
            ref_compressed[pos + 1],
            ref_compressed[pos + 2],
            0,
        ]);
        let block_type = (bh >> 1) & 0x3;
        let block_size = (bh >> 3) as usize;
        let block_type_name = match block_type {
            0 => "Raw",
            1 => "RLE",
            2 => "Compressed",
            _ => "Reserved",
        };

        println!(
            "  Reference uses {} block ({} bytes)",
            block_type_name, block_size
        );

        if block_type == 2 {
            // Found a Compressed block!
            println!("  FOUND Compressed block from reference!");
            println!("  Reference hex: {:02x?}", &ref_compressed[..]);

            // Now compare with our compression
            let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
            let our_compressed = compressor.compress(input).unwrap();
            println!("  Ours hex: {:02x?}", &our_compressed[..]);

            // Test cross-decoding
            println!("\n  Cross-decode tests:");
            match zstd::decode_all(&our_compressed[..]) {
                Ok(d) if d == input => println!("    Ref decodes ours: ✓"),
                Ok(_) => println!("    Ref decodes ours: WRONG DATA"),
                Err(e) => println!("    Ref decodes ours: ✗ {}", e),
            }
            match haagenti_zstd::decompress::decompress_frame(&ref_compressed) {
                Ok(d) if d == input => println!("    We decode ref: ✓"),
                Ok(_) => println!("    We decode ref: WRONG DATA"),
                Err(e) => println!("    We decode ref: ✗ {:?}", e),
            }
            match haagenti_zstd::decompress::decompress_frame(&our_compressed) {
                Ok(d) if d == input => println!("    We decode ours: ✓"),
                Ok(_) => println!("    We decode ours: WRONG DATA"),
                Err(e) => println!("    We decode ours: ✗ {:?}", e),
            }
        }
    }
}

//! Test FSE with a minimal 1-sequence case (which works) vs 2-sequence (which fails).
//! This should help isolate exactly where the problem is.

fn main() {
    println!("=== Testing Minimal FSE Cases ===\n");

    // Case 1: Input that produces exactly 1 sequence
    // "abcdabcd" = 4 literals + match(offset=4, length=4)
    let input1 = b"abcdabcd";
    test_case("1-sequence", input1);

    // Case 2: Input that produces exactly 2 sequences
    // "abcdabcdabcd" = 4 literals + match(4,4) + 4 literals... no wait that's wrong
    // Let me think about this more carefully:
    // "abcdXabcd" = "abcdX" (5 literals) + match(offset=5, length=4)
    // That's still 1 sequence.
    //
    // For 2 sequences, we need:
    // - First match
    // - Some literals
    // - Second match
    //
    // "abcdabcdXabcd" = 4 literals + match(4,4) + 1 literal "X" + match(offset=5, length=4)
    let input2 = b"abcdabcdXabcd";
    test_case("2-sequence (simple)", input2);

    // Case 3: The failing case from earlier
    let input3 = b"abcdefghXabcdefghYabcd";
    test_case("2-sequence (complex)", input3);

    // Case 4: Try 3 sequences
    let input4 = b"abcdabcdXabcdYabcd";
    test_case("3-sequence", input4);
}

fn test_case(name: &str, input: &[u8]) {
    println!("=== {} ===", name);
    println!(
        "Input: {:?} ({} bytes)",
        std::str::from_utf8(input).unwrap(),
        input.len()
    );

    // Use the zstd crate to compress and see what it produces
    let ref_compressed = match zstd::encode_all(&input[..], 1) {
        Ok(c) => c,
        Err(e) => {
            println!("Reference compression failed: {}", e);
            return;
        }
    };
    println!("Reference compressed: {} bytes", ref_compressed.len());

    // Decode reference output
    let ref_decoded = zstd::decode_all(&ref_compressed[..]).unwrap();
    assert_eq!(&ref_decoded[..], input, "Reference roundtrip failed");
    println!("Reference roundtrip: OK");

    // Now use our compression via the speculative compressor
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = match compressor.compress(input) {
        Ok(c) => c,
        Err(e) => {
            println!("Our compression failed: {}", e);
            return;
        }
    };
    println!("Our compressed: {} bytes", our_compressed.len());

    // Try to decode our output with reference zstd
    match zstd::decode_all(&our_compressed[..]) {
        Ok(decoded) => {
            if decoded == input {
                println!("Reference decoded our output: OK ✓");
            } else {
                println!("Reference decoded our output: WRONG DATA");
                println!("  Expected: {:?}", std::str::from_utf8(input));
                println!("  Got: {:?}", std::str::from_utf8(&decoded).ok());
            }
        }
        Err(e) => {
            println!("Reference FAILED to decode our output: {}", e);

            // Analyze the frame
            analyze_block_type(&our_compressed);
        }
    }

    // Try our decoder
    match haagenti_zstd::decompress::decompress_frame(&our_compressed) {
        Ok(decoded) => {
            if decoded == input {
                println!("Our decoder: OK ✓");
            } else {
                println!("Our decoder: WRONG DATA");
            }
        }
        Err(e) => {
            println!("Our decoder FAILED: {}", e);
        }
    }

    println!();
}

fn analyze_block_type(frame: &[u8]) {
    if frame.len() < 7 {
        println!("  Frame too short to analyze");
        return;
    }

    // Skip magic (4) + FHD (1) + optional window desc (1)
    let pos = if (frame[4] & 0x20) != 0 { 5 } else { 6 };

    if pos + 3 > frame.len() {
        println!("  No block header");
        return;
    }

    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;

    let type_name = match block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Reserved",
        _ => "Unknown",
    };

    println!(
        "  Block type: {} ({}), size: {}",
        block_type, type_name, block_size
    );

    if block_type == 2 && pos + 3 + block_size <= frame.len() {
        let block_data = &frame[pos + 3..pos + 3 + block_size];

        // Parse literals header
        if !block_data.is_empty() {
            let lit_type = block_data[0] & 0x03;
            println!(
                "  Literals type: {}",
                match lit_type {
                    0 => "Raw",
                    1 => "RLE",
                    2 => "Compressed",
                    _ => "Treeless",
                }
            );

            // For raw literals, find sequence section
            if lit_type == 0 {
                let size_format = (block_data[0] >> 2) & 0x3;
                let (lit_size, header_size) = match size_format {
                    0 => ((block_data[0] >> 3) as usize, 1),
                    1 => {
                        let s = ((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4))
                            & 0xFFF;
                        (s, 2)
                    }
                    _ => (0, 1),
                };

                let lit_end = header_size + lit_size;
                if lit_end <= block_data.len() {
                    let seq_section = &block_data[lit_end..];
                    if !seq_section.is_empty() {
                        let seq_count = seq_section[0] as usize;
                        println!("  Sequence count: {}", seq_count);

                        if seq_section.len() > 1 {
                            let mode = seq_section[1];
                            println!("  Mode byte: 0x{:02x}", mode);

                            let ll_mode = mode & 0x3;
                            let of_mode = (mode >> 2) & 0x3;
                            let ml_mode = (mode >> 4) & 0x3;

                            let mode_name = |m| match m {
                                0 => "Predefined",
                                1 => "RLE",
                                2 => "FSE",
                                3 => "Repeat",
                                _ => "?",
                            };
                            println!(
                                "    LL: {}, OF: {}, ML: {}",
                                mode_name(ll_mode),
                                mode_name(of_mode),
                                mode_name(ml_mode)
                            );

                            if seq_section.len() > 2 {
                                println!("  Bitstream: {:02x?}", &seq_section[2..]);
                            }
                        }
                    }
                }
            }
        }
    }
}

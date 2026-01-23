//! Trace offset values produced by our compressor vs reference.

use haagenti_zstd::compress::SpeculativeCompressor;

fn main() {
    let test_cases: Vec<(&str, &[u8])> = vec![
        ("ABCD x 25", b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD"),
        ("abcdabcd", b"abcdabcd"),
        ("simple repeat", b"HelloHello"),
    ];

    for (name, input) in test_cases {
        println!("\n=== {} ({} bytes) ===", name, input.len());

        // Get reference compression
        let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
        println!("Reference: {} bytes", ref_compressed.len());

        // Get our compression
        let compressor = SpeculativeCompressor::new();
        let our_compressed = compressor.compress(input).unwrap();
        println!("Ours: {} bytes", our_compressed.len());

        // Parse both frames to extract sequence info
        println!("\nReference frame:");
        parse_sequences(&ref_compressed);

        println!("\nOur frame:");
        parse_sequences(&our_compressed);

        // Cross-decode test
        println!("\nCross-decode:");
        match zstd::decode_all(&our_compressed[..]) {
            Ok(d) if d == input => println!("  Ref decodes ours: OK"),
            Ok(d) => println!("  Ref decodes ours: WRONG DATA (got {} bytes)", d.len()),
            Err(e) => println!("  Ref decodes ours: FAIL - {}", e),
        }
        match haagenti_zstd::decompress::decompress_frame(&ref_compressed) {
            Ok(d) if d == input => println!("  We decode ref: OK"),
            Ok(d) => println!("  We decode ref: WRONG DATA (got {} bytes)", d.len()),
            Err(e) => println!("  We decode ref: FAIL - {:?}", e),
        }
    }
}

fn parse_sequences(frame: &[u8]) {
    if frame.len() < 7 {
        println!("  Frame too short");
        return;
    }

    // Parse frame header
    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
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

    let mut pos = 5;
    if !single_segment {
        pos += 1;
    }
    pos += fcs_size;

    if pos + 3 > frame.len() {
        println!("  Cannot read block header");
        return;
    }

    // Parse block header
    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    let block_type_name = match block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        _ => "Reserved",
    };
    println!("  Block: {} ({} bytes)", block_type_name, block_size);

    if block_type != 2 {
        return;
    }

    if pos + block_size > frame.len() {
        println!("  Block extends past frame");
        return;
    }

    let block_data = &frame[pos..pos + block_size];

    // Parse literals header
    let lit_type = block_data[0] & 0x03;
    let lit_type_name = match lit_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Treeless",
        _ => "?",
    };

    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => ((block_data[0] >> 3) as usize, 1),
            2 => (
                ((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF,
                2,
            ),
            _ => (0, 1),
        }
    } else {
        // Compressed literals have more complex headers
        (0, 1) // Simplified
    };

    println!("  Literals: {} ({} bytes)", lit_type_name, lit_size);

    // Parse sequences section
    let seq_section = &block_data[lit_header_size + lit_size..];
    if seq_section.is_empty() {
        println!("  No sequences section");
        return;
    }

    let seq_count = seq_section[0];
    if seq_section.len() < 2 {
        println!("  Sequences: {} (no mode byte)", seq_count);
        return;
    }

    let mode = seq_section[1];
    let ll_mode = mode & 0x03;
    let of_mode = (mode >> 2) & 0x03;
    let ml_mode = (mode >> 4) & 0x03;

    let mode_name = |m: u8| match m {
        0 => "Predefined",
        1 => "RLE",
        2 => "FSE",
        3 => "Repeat",
        _ => "?",
    };

    println!("  Sequences: {}", seq_count);
    println!(
        "    Mode: LL={}, OF={}, ML={}",
        mode_name(ll_mode),
        mode_name(of_mode),
        mode_name(ml_mode)
    );

    let bitstream = &seq_section[2..];
    println!(
        "    Bitstream ({} bytes): {:02x?}",
        bitstream.len(),
        bitstream
    );

    // Try to decode the FSE states from bitstream
    if mode == 0x00 && !bitstream.is_empty() {
        decode_initial_states(bitstream);
    }
}

fn decode_initial_states(bitstream: &[u8]) {
    // Find the sentinel bit (highest 1-bit in last byte)
    let last_byte = bitstream.last().copied().unwrap_or(0);
    if last_byte == 0 {
        println!("    Invalid bitstream (no sentinel)");
        return;
    }

    let sentinel_pos = 7 - last_byte.leading_zeros() as usize;
    let total_bits = (bitstream.len() - 1) * 8 + sentinel_pos;

    println!(
        "    Total bits: {} (sentinel at bit {})",
        total_bits, sentinel_pos
    );

    // Predefined accuracy logs: LL=6, OF=5, ML=6
    // Initial states are at MSB end: LL (6 bits), OF (5 bits), ML (6 bits) = 17 bits
    if total_bits < 17 {
        println!("    Not enough bits for initial states");
        return;
    }

    // Read bits from MSB end (after sentinel)
    // This is a simplified bit reader - read from the high end
    let mut bit_pos = total_bits;
    let mut accumulated: u64 = 0;

    // Convert bitstream to u64 for easier bit manipulation (handle up to 8 bytes)
    for (i, &b) in bitstream.iter().enumerate() {
        accumulated |= (b as u64) << (i * 8);
    }

    // Remove sentinel bit
    bit_pos -= 1;

    // Read LL state (6 bits)
    let ll_state = (accumulated >> (bit_pos - 6)) & 0x3F;
    bit_pos -= 6;

    // Read OF state (5 bits)
    let of_state = (accumulated >> (bit_pos - 5)) & 0x1F;
    bit_pos -= 5;

    // Read ML state (6 bits)
    let ml_state = (accumulated >> (bit_pos - 6)) & 0x3F;
    bit_pos -= 6;

    println!(
        "    Initial states: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );

    // Map states to codes using predefined tables
    let ll_code = get_ll_code(ll_state as usize);
    let of_code = get_of_code(of_state as usize);
    let ml_code = get_ml_code(ml_state as usize);

    println!("    Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

    // Remaining bits are for extra bits
    println!("    Remaining bits for extras: {}", bit_pos);
}

// Lookup codes from predefined table states
fn get_ll_code(state: usize) -> u8 {
    // LL predefined table: state -> symbol mapping
    const LL_SYMBOLS: [u8; 64] = [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        14, 14, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20,
        20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
    ];
    if state < 64 {
        LL_SYMBOLS[state]
    } else {
        0
    }
}

fn get_of_code(state: usize) -> u8 {
    // OF predefined table: state -> symbol mapping
    const OF_SYMBOLS: [u8; 32] = [
        0, 6, 9, 15, 21, 3, 7, 12, 18, 23, 5, 8, 14, 20, 2, 7, 11, 17, 22, 4, 8, 13, 19, 1, 6, 10,
        16, 28, 27, 26, 25, 24,
    ];
    if state < 32 {
        OF_SYMBOLS[state]
    } else {
        0
    }
}

fn get_ml_code(state: usize) -> u8 {
    // ML predefined table: state -> symbol mapping
    const ML_SYMBOLS: [u8; 64] = [
        0, 1, 2, 3, 5, 6, 8, 10, 13, 5, 6, 8, 10, 13, 6, 8, 10, 13, 7, 9, 11, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    ];
    if state < 64 {
        ML_SYMBOLS[state]
    } else {
        0
    }
}

//! Precise bit-level analysis of FSE bitstreams.

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== Precise Bitstream Analysis ===\n");

    // Test case: ABCD x 25
    let input = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD";
    println!("Input: \"ABCD\" x 25 ({} bytes)", input.len());
    println!("Expected: LL=4, offset=4, match_length=96\n");

    // Get both compressions
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    // Extract bitstreams
    let ref_bitstream = extract_bitstream(&ref_compressed).expect("ref bitstream");
    let our_bitstream = extract_bitstream(&our_compressed).expect("our bitstream");

    println!("Reference bitstream: {:02x?}", ref_bitstream);
    println!("Our bitstream:       {:02x?}\n", our_bitstream);

    println!("=== Reference Decoding ===");
    decode_and_trace(&ref_bitstream, 1);

    println!("\n=== Our Decoding ===");
    decode_and_trace(&our_bitstream, 1);

    // Cross decode test
    println!("\n=== Cross-Decode Test ===");
    match zstd::decode_all(&our_compressed[..]) {
        Ok(d) if d == input => println!("Reference decodes ours: OK"),
        Ok(d) => println!(
            "Reference decodes ours: WRONG DATA ({} vs {} bytes)",
            d.len(),
            input.len()
        ),
        Err(e) => println!("Reference decodes ours: FAIL - {}", e),
    }
}

fn extract_bitstream(frame: &[u8]) -> Option<Vec<u8>> {
    if frame.len() < 7 {
        return None;
    }

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
        return None;
    }

    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        return None;
    }
    if pos + block_size > frame.len() {
        return None;
    }

    let block_data = &frame[pos..pos + block_size];
    let lit_type = block_data[0] & 0x03;

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
        (0, 1)
    };

    let seq_section = &block_data[lit_header_size + lit_size..];
    if seq_section.len() < 2 {
        return None;
    }

    Some(seq_section[2..].to_vec())
}

fn decode_and_trace(bitstream: &[u8], seq_count: usize) {
    // Build tables
    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    )
    .unwrap();
    let of_table =
        FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    )
    .unwrap();

    // Print binary representation
    print!("Binary (MSB first): ");
    for b in bitstream.iter().rev() {
        print!("{:08b} ", b);
    }
    println!();

    // Use our BitReader
    let mut bits = BitReader::new(bitstream);
    if bits.init_from_end().is_err() {
        println!("Failed to init BitReader");
        return;
    }

    println!("Total bits after sentinel: {}", bits.bits_remaining());

    // Init decoders
    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    // Read initial states (LL first, then OF, then ML per zstd spec)
    ll_decoder.init_state(&mut bits).expect("LL init");
    of_decoder.init_state(&mut bits).expect("OF init");
    ml_decoder.init_state(&mut bits).expect("ML init");

    let ll_state = ll_decoder.state();
    let of_state = of_decoder.state();
    let ml_state = ml_decoder.state();

    println!(
        "Initial states: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );

    // Get codes from states
    let ll_entry = ll_table.decode(ll_state);
    let of_entry = of_table.decode(of_state);
    let ml_entry = ml_table.decode(ml_state);

    println!(
        "Codes: LL={}, OF={}, ML={}",
        ll_entry.symbol, of_entry.symbol, ml_entry.symbol
    );
    println!("Bits remaining after states: {}", bits.bits_remaining());

    // Switch to LSB mode for extra bits
    bits.switch_to_lsb_mode().expect("switch to LSB");

    // For each sequence, decode
    for i in 0..seq_count {
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        // Calculate extra bit counts
        let of_extra_bits = of_code as usize;
        let ml_extra_bits = get_ml_extra_bits(ml_code) as usize;
        let ll_extra_bits = get_ll_extra_bits(ll_code) as usize;

        println!("\nSequence {}:", i);
        println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);
        println!(
            "  Extra bits needed: LL={}, OF={}, ML={}",
            ll_extra_bits, of_extra_bits, ml_extra_bits
        );

        // Read extra bits in zstd order: OF, ML, LL
        let of_extra = if of_extra_bits > 0 {
            bits.read_bits(of_extra_bits).unwrap_or(0)
        } else {
            0
        };
        let ml_extra = if ml_extra_bits > 0 {
            bits.read_bits(ml_extra_bits).unwrap_or(0)
        } else {
            0
        };
        let ll_extra = if ll_extra_bits > 0 {
            bits.read_bits(ll_extra_bits).unwrap_or(0)
        } else {
            0
        };

        // Calculate values
        let ll_value = get_ll_baseline(ll_code) + ll_extra as u32;
        let offset_value = (1u32 << of_code) + of_extra;
        let ml_baseline = get_ml_baseline(ml_code);
        let ml_value = ml_baseline + ml_extra as u32;

        // Calculate actual offset from offset_value
        let actual_offset = if offset_value <= 3 {
            // Repeat offset - would need history to resolve
            format!("repeat_offset_{}", offset_value)
        } else {
            format!("{}", offset_value - 3)
        };

        println!(
            "  Extra values: LL_extra={}, OF_extra={}, ML_extra={}",
            ll_extra, of_extra, ml_extra
        );
        println!(
            "  Decoded: LL={}, offset_value={} ({}), ML={}",
            ll_value, offset_value, actual_offset, ml_value
        );
        println!("  Bits remaining: {}", bits.bits_remaining());
    }
}

fn get_ll_extra_bits(code: u8) -> u8 {
    match code {
        0..=15 => 0,
        16..=17 => 1,
        18..=19 => 2,
        20..=21 => 3,
        22..=23 => 4,
        24..=25 => 5,
        26..=27 => 6,
        28..=29 => 7,
        30..=31 => 8,
        32..=33 => 9,
        34..=35 => 10,
        _ => 0,
    }
}

fn get_ll_baseline(code: u8) -> u32 {
    match code {
        0..=15 => code as u32,
        16 => 16,
        17 => 18,
        18 => 20,
        19 => 24,
        20 => 28,
        21 => 36,
        22 => 44,
        23 => 60,
        24 => 76,
        25 => 108,
        26 => 140,
        27 => 204,
        28 => 268,
        29 => 396,
        30 => 524,
        31 => 780,
        32 => 1036,
        33 => 1548,
        34 => 2060,
        35 => 3084,
        _ => 0,
    }
}

fn get_ml_extra_bits(code: u8) -> u8 {
    // Using zstd's predefined table values (differ from RFC for codes 43+)
    match code {
        0..=31 => 0,  // codes 0-31: no extra bits
        32..=35 => 1, // codes 32-35: 1 extra bit
        36..=37 => 2, // codes 36-37: 2 extra bits
        38..=39 => 3, // codes 38-39: 3 extra bits
        40..=41 => 4, // codes 40-41: 4 extra bits
        42 => 5,      // code 42: 5 extra bits
        43 => 7,      // code 43: 7 extra bits (zstd predefined differs from RFC!)
        44 => 8,      // code 44: 8 extra bits
        45 => 9,      // code 45: 9 extra bits
        46 => 10,     // code 46: 10 extra bits
        47 => 11,     // code 47: 11 extra bits
        48 => 12,     // code 48: 12 extra bits
        49 => 13,     // code 49: 13 extra bits
        50 => 14,     // code 50: 14 extra bits
        51 => 15,     // code 51: 15 extra bits
        52 => 16,     // code 52: 16 extra bits
        _ => 0,
    }
}

fn get_ml_baseline(code: u8) -> u32 {
    // Using zstd's actual predefined values (differ from RFC for codes 43+)
    match code {
        0..=31 => (code as u32) + 3,
        32 => 35,
        33 => 37,
        34 => 39,
        35 => 41,
        36 => 43,
        37 => 47,
        38 => 51,
        39 => 59,
        40 => 67,
        41 => 83,
        42 => 99,
        43 => 131,
        44 => 259,
        45 => 515,
        46 => 1027,
        47 => 2051,
        48 => 4099,
        49 => 8195,
        50 => 16387,
        51 => 32771,
        52 => 65539,
        _ => 3,
    }
}

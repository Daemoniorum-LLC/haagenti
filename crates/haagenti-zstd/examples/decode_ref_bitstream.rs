//! Decode reference zstd bitstream bit-by-bit

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    // Test with 2-sequence case - use the same input as simple_multi_seq
    let mut input = Vec::new();
    for _ in 0..4 {
        input.extend_from_slice(b"ABCDXXXX");
    }
    input.extend_from_slice(b"EFGHEFGHEFGHEFGHEFGHEFGHEFGHEFGH"); // 32 more bytes to trigger second sequence

    println!("Input: {} bytes", input.len());
    println!("Pattern: ABCDXXXX * 4 + EFGH * 8");

    // Compress with reference
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&input), 1).unwrap();
    println!("\nReference compressed: {} bytes", ref_compressed.len());

    // Parse frame header
    let fhd = ref_compressed[4];
    let single_segment = (fhd >> 5) & 1;
    let mut pos = 5;
    if single_segment == 0 {
        pos += 1; // window descriptor
    }

    // Skip FCS if present
    let fcs_size = (fhd >> 6) & 3;
    let fcs_bytes = match fcs_size {
        0 => {
            if single_segment == 1 {
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
    pos += fcs_bytes;

    // Parse block header
    let bh = (ref_compressed[pos] as u32)
        | ((ref_compressed[pos + 1] as u32) << 8)
        | ((ref_compressed[pos + 2] as u32) << 16);
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        println!("Not a compressed block");
        return;
    }

    let block = &ref_compressed[pos..pos + block_size];
    println!("Block: {} bytes", block.len());

    // Parse literals
    let lh = block[0];
    let lit_type = lh & 3;
    let size_format = (lh >> 2) & 3;
    let (lit_size, lit_header) = if lit_type <= 1 {
        match size_format {
            0 => ((lh >> 3) as usize, 1),
            1 => (((lh >> 4) as usize) | ((block[1] as usize) << 4), 2),
            _ => (0, 1),
        }
    } else {
        (0, 1)
    };

    let seq_start = lit_header + lit_size;
    let seq_section = &block[seq_start..];

    let seq_count = seq_section[0] as usize;
    let mode_byte = seq_section[1];
    let fse_bitstream = &seq_section[2..];

    println!("\nSequences: {} count, mode {:02x}", seq_count, mode_byte);
    println!(
        "FSE bitstream: {} bytes = {:02x?}",
        fse_bitstream.len(),
        fse_bitstream
    );

    // Show binary
    print!("Binary (LSB first per byte): ");
    for b in fse_bitstream {
        for i in 0..8 {
            print!("{}", (b >> i) & 1);
        }
        print!(" ");
    }
    println!();

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

    // Decode with our decoder
    let mut bits = BitReader::new(fse_bitstream);
    bits.init_from_end().unwrap();

    println!("\n=== Reading from MSB end ===");
    println!("Bits available: {}", bits.bits_remaining());

    // Read initial states
    let ll_state = bits
        .read_bits(LITERAL_LENGTH_ACCURACY_LOG as usize)
        .unwrap();
    let of_state = bits.read_bits(OFFSET_ACCURACY_LOG as usize).unwrap();
    let ml_state = bits.read_bits(MATCH_LENGTH_ACCURACY_LOG as usize).unwrap();

    println!(
        "Initial states: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );

    let ll_sym = ll_table.decode(ll_state as usize).symbol;
    let of_sym = of_table.decode(of_state as usize).symbol;
    let ml_sym = ml_table.decode(ml_state as usize).symbol;

    println!(
        "Initial symbols: LL={}, OF={}, ML={}",
        ll_sym, of_sym, ml_sym
    );
    println!("Bits remaining: {}", bits.bits_remaining());

    // Switch to LSB mode
    bits.switch_to_lsb_mode().unwrap();
    println!("\n=== Switched to LSB mode ===");
    println!("Bits remaining: {}", bits.bits_remaining());

    // Reinit bit reader and decode properly using our decoder
    let mut bits2 = BitReader::new(fse_bitstream);
    bits2.init_from_end().unwrap();

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    ll_decoder.init_state(&mut bits2).unwrap();
    of_decoder.init_state(&mut bits2).unwrap();
    ml_decoder.init_state(&mut bits2).unwrap();

    bits2.switch_to_lsb_mode().unwrap();

    for i in 0..seq_count {
        let is_last = i == seq_count - 1;
        println!(
            "\n--- Seq {} {} ---",
            i,
            if is_last { "(LAST)" } else { "" }
        );

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!("Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

        // RFC 8878 says: read in Offset, Match_Length, Literals_Length order
        let ll_extra_bits = get_ll_extra_bits(ll_code);
        let ml_extra_bits = get_ml_extra_bits(ml_code);
        let of_extra_bits = of_code; // OF extra bits = of_code

        println!(
            "Extra bits needed: OF={}, ML={}, LL={}",
            of_extra_bits, ml_extra_bits, ll_extra_bits
        );
        println!("Bits available: {}", bits2.bits_remaining());

        // RFC order: OF, ML, LL
        let of_extra = if of_extra_bits > 0 {
            bits2.read_bits(of_extra_bits as usize).unwrap_or(999)
        } else {
            0
        };
        let ml_extra = if ml_extra_bits > 0 {
            bits2.read_bits(ml_extra_bits as usize).unwrap_or(999)
        } else {
            0
        };
        let ll_extra = if ll_extra_bits > 0 {
            bits2.read_bits(ll_extra_bits as usize).unwrap_or(999)
        } else {
            0
        };

        println!(
            "Extras (OF,ML,LL order): OF={}, ML={}, LL={}",
            of_extra, ml_extra, ll_extra
        );

        // Compute values
        let ll_value = get_ll_baseline(ll_code) + ll_extra;
        let of_value = if of_code > 0 {
            (1u32 << of_code) + of_extra
        } else {
            of_extra
        };
        let ml_value = get_ml_baseline(ml_code) + ml_extra;

        println!("Values: LL={}, OF={}, ML={}", ll_value, of_value, ml_value);

        // Update states (skip for last)
        if !is_last {
            let ll_entry = ll_table.decode(ll_decoder.state());
            let ml_entry = ml_table.decode(ml_decoder.state());
            let of_entry = of_table.decode(of_decoder.state());

            println!(
                "FSE bits needed: LL={}, ML={}, OF={}",
                ll_entry.num_bits, ml_entry.num_bits, of_entry.num_bits
            );
            println!("Bits available: {}", bits2.bits_remaining());

            // Try all permutations to find the right order
            // Create a backup of bit position
            let before = bits2.bits_remaining();

            // Try different orders and print results
            println!("\n  Testing FSE update orders:");

            // Reinit and try different orders
            for order in [
                "LL,ML,OF", "LL,OF,ML", "ML,LL,OF", "ML,OF,LL", "OF,LL,ML", "OF,ML,LL",
            ] {
                // Reinitialize decoders for each test
                let mut bits_test = BitReader::new(fse_bitstream);
                bits_test.init_from_end().unwrap();

                let mut ll_test = FseDecoder::new(&ll_table);
                let mut of_test = FseDecoder::new(&of_table);
                let mut ml_test = FseDecoder::new(&ml_table);

                ll_test.init_state(&mut bits_test).unwrap();
                of_test.init_state(&mut bits_test).unwrap();
                ml_test.init_state(&mut bits_test).unwrap();
                bits_test.switch_to_lsb_mode().unwrap();

                // Read extras in OF, ML, LL order (per RFC)
                let of_bits = of_test.peek_symbol();
                if of_bits > 0 {
                    bits_test.read_bits(of_bits as usize).ok();
                }
                // Skip ML and LL extras (0 bits each for this test case)

                // Now read FSE update bits in different order
                match order {
                    "LL,ML,OF" => {
                        ll_test.update_state(&mut bits_test).ok();
                        ml_test.update_state(&mut bits_test).ok();
                        of_test.update_state(&mut bits_test).ok();
                    }
                    "LL,OF,ML" => {
                        ll_test.update_state(&mut bits_test).ok();
                        of_test.update_state(&mut bits_test).ok();
                        ml_test.update_state(&mut bits_test).ok();
                    }
                    "ML,LL,OF" => {
                        ml_test.update_state(&mut bits_test).ok();
                        ll_test.update_state(&mut bits_test).ok();
                        of_test.update_state(&mut bits_test).ok();
                    }
                    "ML,OF,LL" => {
                        ml_test.update_state(&mut bits_test).ok();
                        of_test.update_state(&mut bits_test).ok();
                        ll_test.update_state(&mut bits_test).ok();
                    }
                    "OF,LL,ML" => {
                        of_test.update_state(&mut bits_test).ok();
                        ll_test.update_state(&mut bits_test).ok();
                        ml_test.update_state(&mut bits_test).ok();
                    }
                    "OF,ML,LL" => {
                        of_test.update_state(&mut bits_test).ok();
                        ml_test.update_state(&mut bits_test).ok();
                        ll_test.update_state(&mut bits_test).ok();
                    }
                    _ => {}
                }

                let ll_sym = ll_test.peek_symbol();
                let of_sym = of_test.peek_symbol();
                let ml_sym = ml_test.peek_symbol();
                println!(
                    "    {}: LL={} (sym {}), OF={} (sym {}), ML={} (sym {}), remaining={}",
                    order,
                    ll_test.state(),
                    ll_sym,
                    of_test.state(),
                    of_sym,
                    ml_test.state(),
                    ml_sym,
                    bits_test.bits_remaining()
                );
            }

            // Actually do the update for the main loop (using LL,ML,OF as fallback)
            ll_decoder.update_state(&mut bits2).ok();
            ml_decoder.update_state(&mut bits2).ok();
            of_decoder.update_state(&mut bits2).ok();
        }
    }

    println!("\nFinal bits remaining: {}", bits2.bits_remaining());

    // Now let's verify with actual decompression
    println!("\n=== Verification ===");
    match zstd::decode_all(std::io::Cursor::new(&ref_compressed)) {
        Ok(decoded) => {
            println!("Decompressed: {} bytes", decoded.len());
            if decoded == input {
                println!("Content: OK");
            } else {
                println!("Content: MISMATCH");
            }
        }
        Err(e) => println!("Decompression failed: {}", e),
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
    match code {
        0..=31 => 0,
        32..=33 => 1,
        34..=35 => 2,
        36..=37 => 3,
        38..=39 => 4,
        40..=41 => 5,
        42 => 6,
        43 => 7,
        44 => 8,
        45 => 9,
        46 => 10,
        47 => 11,
        48 => 12,
        49 => 13,
        50 => 14,
        51 => 15,
        52 => 16,
        _ => 0,
    }
}

fn get_ml_baseline(code: u8) -> u32 {
    if code <= 31 {
        (code as u32) + 3
    } else {
        match code {
            32 => 35,
            33 => 37,
            34 => 39,
            35 => 43,
            36 => 47,
            37 => 51,
            38 => 59,
            39 => 67,
            40 => 83,
            41 => 99,
            42 => 131,
            43 => 259,
            44 => 515,
            45 => 1027,
            46 => 2051,
            47 => 4099,
            48 => 8195,
            49 => 16387,
            50 => 32771,
            51 => 65539,
            52 => 131075,
            _ => 0,
        }
    }
}

//! Trace the exact FSE bitstream for failing 2-sequence case.
#![allow(unused_variables)]
#![allow(clippy::unnecessary_cast)]

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    let input = b"abcdefghXabcdefghYabcd";
    println!(
        "Input: {:?} ({} bytes)",
        std::str::from_utf8(input).unwrap(),
        input.len()
    );

    // Get our compressed frame
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    // Parse the sequence section
    let (seq_section, expected_seqs) = extract_seq_section(&our_compressed);
    println!("\nSequence section: {:02x?}", seq_section);
    println!("Expected sequences based on input analysis:");
    println!("  Seq 0: 9 literals, offset=9, match=8");
    println!("  Seq 1: 1 literal, offset=?, match=4");

    // Parse and decode
    decode_with_trace(&seq_section);

    // Also check what reference expects
    println!("\n=== Testing with reference decoder ===");
    match zstd::decode_all(&our_compressed[..]) {
        Ok(decoded) => {
            println!("Reference decoded {} bytes", decoded.len());
            if decoded == input {
                println!("Content matches!");
            } else {
                println!("CONTENT MISMATCH!");
                println!("Got: {:?}", String::from_utf8_lossy(&decoded));
            }
        }
        Err(e) => println!("Reference FAILED: {}", e),
    }
}

fn extract_seq_section(frame: &[u8]) -> (Vec<u8>, usize) {
    // Skip magic (4) + FHD (1) + window (maybe 1)
    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment {
        pos += 1;
    }

    // Block header
    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        return (vec![], 0);
    }

    let block_data = &frame[pos..pos + block_size];

    // Parse literals
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
    let seq_count = seq_section[0] as usize;
    (seq_section.to_vec(), seq_count)
}

fn decode_with_trace(seq_section: &[u8]) {
    if seq_section.len() < 2 {
        return;
    }

    let seq_count = seq_section[0] as usize;
    let mode = seq_section[1];
    let bitstream = &seq_section[2..];

    println!("\n=== Decoding {} sequences ===", seq_count);
    println!("Mode: 0x{:02x}", mode);
    println!("Bitstream ({} bytes): {:02x?}", bitstream.len(), bitstream);

    // Binary view
    print!("Binary: ");
    for b in bitstream {
        print!("{:08b} ", b);
    }
    println!();

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

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    let mut bits = BitReader::new(bitstream);
    if bits.init_from_end().is_err() {
        println!("Failed to init bitreader");
        return;
    }

    let total_bits = bits.bits_remaining();
    println!("\nTotal bits after init: {}", total_bits);

    // Read initial states (MSB mode)
    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    let init_ll = ll_decoder.state();
    let init_of = of_decoder.state();
    let init_ml = ml_decoder.state();

    println!(
        "Initial states: LL={}, OF={}, ML={}",
        init_ll, init_of, init_ml
    );
    println!(
        "Bits after states: {} (consumed {} for states)",
        bits.bits_remaining(),
        total_bits - bits.bits_remaining()
    );

    // Switch to LSB mode
    bits.switch_to_lsb_mode().unwrap();
    println!(
        "After LSB switch: {} bits at bit position",
        bits.bits_remaining()
    );

    for i in 0..seq_count {
        let is_last = i == seq_count - 1;
        println!(
            "\n--- Sequence {} {} ---",
            i,
            if is_last { "(LAST)" } else { "" }
        );

        // Get codes
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();
        println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

        let ll_entry = ll_table.decode(ll_decoder.state());
        let ml_entry = ml_table.decode(ml_decoder.state());
        let of_entry = of_table.decode(of_decoder.state());

        // Read extra bits in OF, ML, LL order
        let of_extra_bits = of_code as usize;
        let ml_extra_bits = get_ml_extra_bits(ml_code) as usize;
        let ll_extra_bits = get_ll_extra_bits(ll_code) as usize;

        let before_extras = bits.bits_remaining();
        println!(
            "  Extra bits needed: OF={}, ML={}, LL={}",
            of_extra_bits, ml_extra_bits, ll_extra_bits
        );

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

        println!(
            "  Extras read: OF={}, ML={}, LL={}",
            of_extra, ml_extra, ll_extra
        );
        println!(
            "  Bits after extras: {} (consumed {} for extras)",
            bits.bits_remaining(),
            before_extras - bits.bits_remaining()
        );

        // Compute values
        let ll_value = get_ll_baseline(ll_code) + ll_extra as u32;
        let of_value = if of_code > 0 {
            (1u32 << of_code) + of_extra as u32
        } else {
            of_extra as u32
        };
        let ml_value = get_ml_baseline(ml_code) + ml_extra as u32;

        println!(
            "  Values: LL={}, OF={}, ML={}",
            ll_value, of_value, ml_value
        );

        // Update states (unless last)
        if !is_last {
            let before_update = bits.bits_remaining();
            println!(
                "  FSE update bits needed: LL={}, ML={}, OF={}",
                ll_entry.num_bits, ml_entry.num_bits, of_entry.num_bits
            );

            ll_decoder.decode_symbol(&mut bits).unwrap();
            ml_decoder.decode_symbol(&mut bits).unwrap();
            of_decoder.decode_symbol(&mut bits).unwrap();

            println!(
                "  New states: LL={}, OF={}, ML={}",
                ll_decoder.state(),
                of_decoder.state(),
                ml_decoder.state()
            );
            println!(
                "  Bits after FSE update: {} (consumed {} for FSE)",
                bits.bits_remaining(),
                before_update - bits.bits_remaining()
            );
        }
    }

    println!("\nFinal bits remaining: {}", bits.bits_remaining());
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
    (code as u32) + 3
}

//! Deep comparison of our FSE encoding vs reference zstd.
//! Trace the exact sequence values and their encoding.

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    // The complex 2-sequence case that fails
    let input = b"abcdefghXabcdefghYabcd";
    println!("=== Deep Sequence Comparison ===");
    println!(
        "Input: {:?} ({} bytes)\n",
        std::str::from_utf8(input).unwrap(),
        input.len()
    );

    // Get reference compression
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("Reference compressed: {} bytes", ref_compressed.len());
    print_frame_hex("Reference", &ref_compressed);

    // Get our compression
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();
    println!("\nOur compressed: {} bytes", our_compressed.len());
    print_frame_hex("Ours", &our_compressed);

    // Parse both compressed blocks
    println!("\n=== Parsing Reference Frame ===");
    let ref_seqs = parse_compressed_block(&ref_compressed);

    println!("\n=== Parsing Our Frame ===");
    let our_seqs = parse_compressed_block(&our_compressed);

    // Compare
    println!("\n=== Sequence Comparison ===");
    println!("Reference sequences: {:?}", ref_seqs);
    println!("Our sequences: {:?}", our_seqs);

    // Now let's also trace what match finder produces
    println!("\n=== Our Match Finder Output ===");
    trace_match_finder(input);
}

fn print_frame_hex(label: &str, data: &[u8]) {
    print!("{} hex: ", label);
    for b in data {
        print!("{:02x} ", b);
    }
    println!();
}

#[derive(Debug, Clone)]
struct DecodedSequence {
    ll_code: u8,
    ll_value: u32,
    of_code: u8,
    of_value: u32,
    ml_code: u8,
    ml_value: u32,
}

fn parse_compressed_block(frame: &[u8]) -> Vec<DecodedSequence> {
    if frame.len() < 7 {
        println!("  Frame too short");
        return vec![];
    }

    // Skip magic (4 bytes)
    println!(
        "  Magic: {:02x} {:02x} {:02x} {:02x}",
        frame[0], frame[1], frame[2], frame[3]
    );

    // Frame header descriptor
    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let fcs_field_size = match fhd >> 6 {
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
    let has_checksum = (fhd & 0x04) != 0;
    let has_dict_id = fhd & 0x03;

    println!(
        "  FHD: 0x{:02x} (single_segment={}, fcs_size={}, checksum={}, dict_id_flag={})",
        fhd, single_segment, fcs_field_size, has_checksum, has_dict_id
    );

    let mut pos = 5;

    // Window descriptor (if not single segment)
    if !single_segment {
        let wd = frame[pos];
        println!("  Window desc: 0x{:02x}", wd);
        pos += 1;
    }

    // Dictionary ID (variable size based on dict_id_flag)
    let dict_id_size = match has_dict_id {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => 0,
    };
    if dict_id_size > 0 {
        println!("  Dict ID: {} bytes", dict_id_size);
        pos += dict_id_size;
    }

    // Frame content size (variable)
    if fcs_field_size > 0 {
        println!("  FCS: {} bytes at pos {}", fcs_field_size, pos);
        pos += fcs_field_size as usize;
    }

    // Block header (3 bytes)
    if pos + 3 > frame.len() {
        println!("  No block header");
        return vec![];
    }

    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let is_last = (bh & 1) != 0;
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;

    println!(
        "  Block header: 0x{:02x}{:02x}{:02x} (last={}, type={}, size={})",
        frame[pos],
        frame[pos + 1],
        frame[pos + 2],
        is_last,
        block_type,
        block_size
    );
    pos += 3;

    if block_type != 2 {
        println!("  Not a compressed block");
        return vec![];
    }

    if pos + block_size > frame.len() {
        println!("  Block extends past frame");
        return vec![];
    }

    let block_data = &frame[pos..pos + block_size];
    println!("  Block data ({} bytes): {:02x?}", block_size, block_data);

    // Parse literals section
    let lit_type = block_data[0] & 0x03;
    let lit_type_name = match lit_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        _ => "Treeless",
    };
    println!("  Literals type: {} ({})", lit_type, lit_type_name);

    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => {
                // 5-bit size
                let s = (block_data[0] >> 3) as usize;
                (s, 1)
            }
            2 => {
                // 12-bit size
                if block_data.len() < 2 {
                    (0, 1)
                } else {
                    let s =
                        ((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF;
                    (s, 2)
                }
            }
            _ => {
                // 20-bit size
                if block_data.len() < 3 {
                    (0, 1)
                } else {
                    let s = ((block_data[0] as usize >> 4)
                        | ((block_data[1] as usize) << 4)
                        | ((block_data[2] as usize) << 12))
                        & 0xFFFFF;
                    (s, 3)
                }
            }
        }
    } else {
        // Compressed/treeless - more complex parsing
        println!("  Compressed literals - skipping for now");
        return vec![];
    };

    println!(
        "  Literals: {} bytes (header {} bytes)",
        lit_size, lit_header_size
    );

    let seq_section_start = lit_header_size + lit_size;
    if seq_section_start >= block_data.len() {
        println!("  No sequence section");
        return vec![];
    }

    let seq_section = &block_data[seq_section_start..];
    println!(
        "  Sequence section starts at offset {}, {} bytes",
        seq_section_start,
        seq_section.len()
    );
    println!("  Sequence section: {:02x?}", seq_section);

    // Parse sequence count
    let (seq_count, seq_header_size) = if seq_section[0] < 128 {
        (seq_section[0] as usize, 1)
    } else if seq_section[0] < 255 {
        if seq_section.len() < 2 {
            return vec![];
        }
        let count = ((seq_section[0] as usize - 128) << 8) + seq_section[1] as usize;
        (count, 2)
    } else {
        if seq_section.len() < 3 {
            return vec![];
        }
        let count = seq_section[1] as usize + ((seq_section[2] as usize) << 8) + 0x7F00;
        (count, 3)
    };

    println!("  Sequence count: {}", seq_count);

    if seq_count == 0 {
        return vec![];
    }

    // Parse mode byte
    let mode = seq_section[seq_header_size];
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
        "    LL: {} ({}), OF: {} ({}), ML: {} ({})",
        ll_mode,
        mode_name(ll_mode),
        of_mode,
        mode_name(of_mode),
        ml_mode,
        mode_name(ml_mode)
    );

    // Only handle predefined for now
    if ll_mode != 0 || of_mode != 0 || ml_mode != 0 {
        println!("  Non-predefined mode, skipping detailed decode");
        return vec![];
    }

    let bitstream = &seq_section[seq_header_size + 1..];
    println!(
        "  FSE bitstream: {:02x?} ({} bytes, {} bits)",
        bitstream,
        bitstream.len(),
        bitstream.len() * 8
    );

    // Decode sequences using FSE
    decode_fse_sequences(bitstream, seq_count)
}

fn decode_fse_sequences(bitstream: &[u8], seq_count: usize) -> Vec<DecodedSequence> {
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
        println!("  Failed to init bitreader");
        return vec![];
    }

    println!("  Bits after init: {}", bits.bits_remaining());

    // Read initial states (MSB mode)
    if ll_decoder.init_state(&mut bits).is_err() {
        println!("  Failed to init LL state");
        return vec![];
    }
    if of_decoder.init_state(&mut bits).is_err() {
        println!("  Failed to init OF state");
        return vec![];
    }
    if ml_decoder.init_state(&mut bits).is_err() {
        println!("  Failed to init ML state");
        return vec![];
    }

    println!(
        "  Initial states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );
    println!("  Bits after states: {}", bits.bits_remaining());

    // Switch to LSB mode for extras
    if bits.switch_to_lsb_mode().is_err() {
        println!("  Failed to switch to LSB mode");
        return vec![];
    }

    let mut sequences = Vec::new();

    for i in 0..seq_count {
        println!("\n  --- Decoding sequence {} ---", i);

        // Get codes from current states
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!("    Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

        // Read extra bits
        let ll_extra_bits = get_ll_extra_bits(ll_code);
        let ml_extra_bits = get_ml_extra_bits(ml_code);

        let ll_extra = if ll_extra_bits > 0 {
            bits.read_bits(ll_extra_bits as usize).unwrap_or(0)
        } else {
            0
        };

        let ml_extra = if ml_extra_bits > 0 {
            bits.read_bits(ml_extra_bits as usize).unwrap_or(0)
        } else {
            0
        };

        let of_extra = if of_code > 0 {
            bits.read_bits(of_code as usize).unwrap_or(0)
        } else {
            0
        };

        println!(
            "    Extras: LL={}({} bits), ML={}({} bits), OF={}({} bits)",
            ll_extra, ll_extra_bits, ml_extra, ml_extra_bits, of_extra, of_code
        );

        // Compute values
        let ll_value = get_ll_baseline(ll_code) + ll_extra as u32;
        let ml_value = get_ml_baseline(ml_code) + ml_extra as u32;
        let of_value = if of_code > 0 {
            (1u32 << of_code) + of_extra as u32
        } else {
            of_extra as u32
        };

        println!(
            "    Values: LL={}, OF={}, ML={}",
            ll_value, of_value, ml_value
        );
        println!("    Bits remaining: {}", bits.bits_remaining());

        sequences.push(DecodedSequence {
            ll_code,
            ll_value,
            of_code,
            of_value,
            ml_code,
            ml_value,
        });

        // Update states for next sequence (unless this is the last one)
        if i < seq_count - 1 {
            println!("    Updating states for next sequence...");
            let _ = ll_decoder.decode_symbol(&mut bits);
            let _ = ml_decoder.decode_symbol(&mut bits);
            let _ = of_decoder.decode_symbol(&mut bits);
            println!(
                "    New states: LL={}, OF={}, ML={}",
                ll_decoder.state(),
                of_decoder.state(),
                ml_decoder.state()
            );
            println!("    Bits remaining after update: {}", bits.bits_remaining());
        }
    }

    sequences
}

fn trace_match_finder(input: &[u8]) {
    // Can't access private module, just trace expected sequences
    println!("  Expected sequences for \"abcdefghXabcdefghYabcd\":");
    println!("    Seq 0: 9 literals (\"abcdefghX\"), match(offset=9, length=8)");
    println!("    Seq 1: 1 literal (\"Y\"), match(offset=?, length=4)");
    println!();
    println!("  OR with repeat offsets:");
    println!("    Seq 0: 9 literals, offset=9, ml=8");
    println!("    Seq 1: 1 literal, repeat offset (if offset=9 was used before), ml=4");
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
    // ML code 0 = length 3, code 1 = length 4, etc.
    (code as u32) + 3
}

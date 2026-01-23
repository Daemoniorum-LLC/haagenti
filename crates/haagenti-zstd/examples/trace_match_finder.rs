//! Trace what the compressor produces - analyze the FSE bitstream.

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, TansEncoder, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    let input = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD";
    println!("Input: \"ABCD\" x 25 ({} bytes)\n", input.len());

    // Get reference compression
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("Reference: {} bytes", ref_compressed.len());

    // Get our compression
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();
    println!("Ours: {} bytes\n", our_compressed.len());

    // Parse both and compare the FSE bitstreams
    println!("=== Reference FSE Bitstream ===");
    let ref_bitstream = extract_fse_bitstream(&ref_compressed);
    if let Some(bits) = ref_bitstream {
        analyze_bitstream(&bits);
    }

    println!("\n=== Our FSE Bitstream ===");
    let our_bitstream = extract_fse_bitstream(&our_compressed);
    if let Some(bits) = our_bitstream {
        analyze_bitstream(&bits);
    }

    // Test what OF state we get for various codes
    println!("\n=== OF Encoder State Mapping ===");
    let of_table =
        FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let mut encoder = TansEncoder::from_decode_table(&of_table);

    for code in 0..6 {
        encoder.init_state(code);
        let state = encoder.get_state();
        let decoded = of_table.decode(state as usize).symbol;
        println!(
            "OF code {} -> init_state -> state {} -> decodes to {}",
            code, state, decoded
        );
    }
}

fn extract_fse_bitstream(frame: &[u8]) -> Option<Vec<u8>> {
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
        println!("  Block type: {} (not compressed)", block_type);
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

    let seq_count = seq_section[0];
    let mode = seq_section[1];
    let bitstream = &seq_section[2..];

    println!("  Sequences: {}, Mode: 0x{:02x}", seq_count, mode);
    println!(
        "  Bitstream ({} bytes): {:02x?}",
        bitstream.len(),
        bitstream
    );

    Some(bitstream.to_vec())
}

fn analyze_bitstream(bitstream: &[u8]) {
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
        return;
    }

    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    let ll_state = ll_decoder.state();
    let of_state = of_decoder.state();
    let ml_state = ml_decoder.state();

    let ll_code = ll_table.decode(ll_state).symbol;
    let of_code = of_table.decode(of_state).symbol;
    let ml_code = ml_table.decode(ml_state).symbol;

    println!(
        "  States: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );
    println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

    // Read extras
    bits.switch_to_lsb_mode().unwrap();
    let of_extra = if of_code > 0 {
        bits.read_bits(of_code as usize).unwrap_or(0)
    } else {
        0
    };
    let ml_extra_bits = get_ml_extra_bits(ml_code) as usize;
    let ml_extra = if ml_extra_bits > 0 {
        bits.read_bits(ml_extra_bits).unwrap_or(0)
    } else {
        0
    };

    let of_value = (1u32 << of_code) + of_extra as u32;
    let ml_value = get_ml_baseline(ml_code) + ml_extra as u32;

    println!(
        "  Values: LL={}, OF={} (code {} + {}), ML={} (code {} + {})",
        ll_code, of_value, of_code, of_extra, ml_value, ml_code, ml_extra
    );
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
        _ => 3,
    }
}

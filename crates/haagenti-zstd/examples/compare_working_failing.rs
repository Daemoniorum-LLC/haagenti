//! Compare working vs failing 2-sequence cases.

use haagenti_zstd::fse::{
    FseTable, BitReader, FseDecoder,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
};

fn main() {
    println!("=== Working Case: abcdabcdXabcd ===");
    let working = b"abcdabcdXabcd";
    analyze_and_compare(working);

    println!("\n\n=== Failing Case: abcdefghXabcdefghYabcd ===");
    let failing = b"abcdefghXabcdefghYabcd";
    analyze_and_compare(failing);
}

fn analyze_and_compare(input: &[u8]) {
    println!("Input: {:?} ({} bytes)", std::str::from_utf8(input).unwrap(), input.len());

    // Get our compression
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    // Try reference decode
    let ref_result = zstd::decode_all(&our_compressed[..]);
    match &ref_result {
        Ok(decoded) if decoded.as_slice() == input => println!("Reference decode: ✓ SUCCESS"),
        Ok(decoded) => println!("Reference decode: WRONG DATA ({:?})", String::from_utf8_lossy(decoded)),
        Err(e) => println!("Reference decode: ✗ FAILED ({})", e),
    }

    // Parse our compressed frame
    let fhd = our_compressed[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment { pos += 1; }

    let bh = u32::from_le_bytes([our_compressed[pos], our_compressed[pos+1], our_compressed[pos+2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        println!("  Not a compressed block (type={})", block_type);
        return;
    }

    let block_data = &our_compressed[pos..pos+block_size];

    // Parse literals section
    let lit_type = block_data[0] & 0x03;
    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => ((block_data[0] >> 3) as usize, 1),
            2 => (((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF, 2),
            _ => (0, 1),
        }
    } else { (0, 1) };

    let seq_section = &block_data[lit_header_size + lit_size..];
    let seq_count = seq_section[0] as usize;
    let mode = seq_section[1];
    let bitstream = &seq_section[2..];

    println!("  Literals: {} bytes", lit_size);
    println!("  Sequences: {}", seq_count);
    println!("  Mode: 0x{:02x}", mode);
    println!("  Bitstream ({} bytes): {:02x?}", bitstream.len(), bitstream);

    // Decode and show states
    let ll_table = FseTable::from_predefined(&LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG).unwrap();
    let of_table = FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(&MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG).unwrap();

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    let mut bits = BitReader::new(bitstream);
    if bits.init_from_end().is_err() { return; }

    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    println!("  Init states: LL={}, OF={}, ML={}", ll_decoder.state(), of_decoder.state(), ml_decoder.state());

    bits.switch_to_lsb_mode().unwrap();

    for i in 0..seq_count {
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        let of_extra_bits = of_code as usize;
        let ml_extra_bits = get_ml_extra_bits(ml_code) as usize;
        let ll_extra_bits = get_ll_extra_bits(ll_code) as usize;

        let of_extra = if of_extra_bits > 0 { bits.read_bits(of_extra_bits).unwrap_or(0) } else { 0 };
        let ml_extra = if ml_extra_bits > 0 { bits.read_bits(ml_extra_bits).unwrap_or(0) } else { 0 };
        let ll_extra = if ll_extra_bits > 0 { bits.read_bits(ll_extra_bits).unwrap_or(0) } else { 0 };

        let ll_value = get_ll_baseline(ll_code) + ll_extra as u32;
        let of_value = (1u32 << of_code) + of_extra as u32;
        let ml_value = (ml_code as u32) + 3 + ml_extra as u32;

        println!("  Seq {}: LL={}(code {}), OF={}(code {}+{}), ML={}(code {})", 
                 i, ll_value, ll_code, of_value, of_code, of_extra, ml_value, ml_code);

        if i < seq_count - 1 {
            ll_decoder.decode_symbol(&mut bits).unwrap();
            ml_decoder.decode_symbol(&mut bits).unwrap();
            of_decoder.decode_symbol(&mut bits).unwrap();
        }
    }
}

fn get_ll_extra_bits(code: u8) -> u8 {
    match code {
        0..=15 => 0, 16..=17 => 1, 18..=19 => 2, 20..=21 => 3, 22..=23 => 4,
        24..=25 => 5, 26..=27 => 6, 28..=29 => 7, 30..=31 => 8, 32..=33 => 9, 34..=35 => 10, _ => 0,
    }
}

fn get_ll_baseline(code: u8) -> u32 {
    match code {
        0..=15 => code as u32, 16 => 16, 17 => 18, 18 => 20, 19 => 24, 20 => 28, 21 => 36,
        22 => 44, 23 => 60, 24 => 76, 25 => 108, 26 => 140, 27 => 204, 28 => 268, 29 => 396,
        30 => 524, 31 => 780, 32 => 1036, 33 => 1548, 34 => 2060, 35 => 3084, _ => 0,
    }
}

fn get_ml_extra_bits(code: u8) -> u8 {
    match code {
        0..=31 => 0, 32..=33 => 1, 34..=35 => 2, 36..=37 => 3, 38..=39 => 4,
        40..=41 => 5, 42 => 6, 43 => 7, 44 => 8, 45 => 9, 46 => 10, 47 => 11,
        48 => 12, 49 => 13, 50 => 14, 51 => 15, 52 => 16, _ => 0,
    }
}

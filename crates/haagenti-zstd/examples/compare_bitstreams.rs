//! Compare our FSE bitstream vs reference FSE bitstream bit-by-bit

use haagenti_zstd::fse::{
    FseTable, FseDecoder, BitReader,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
};

fn main() {
    // 2-sequence input
    let mut input = Vec::new();
    for _ in 0..10 { input.extend_from_slice(b"ABCD"); }
    input.extend_from_slice(b"XXXX");
    for _ in 0..5 { input.extend_from_slice(b"EFGH"); }

    println!("Input: {} bytes", input.len());

    // Compress with reference
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&input), 1).unwrap();

    // Compress with ours
    use haagenti_core::CompressionLevel;
    use haagenti_zstd::compress::CompressContext;
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let our_compressed = ctx.compress(&input).unwrap();

    // Extract FSE bitstreams
    let our_fse = extract_fse_bitstream(&our_compressed);
    let ref_fse = extract_fse_bitstream(&ref_compressed);

    println!("\n=== FSE Bitstreams ===");
    println!("Our FSE: {} bytes = {:02x?}", our_fse.len(), our_fse);
    println!("Ref FSE: {} bytes = {:02x?}", ref_fse.len(), ref_fse);

    // Show in binary (LSB first per byte, as zstd reads them)
    println!("\nOur binary (LSB first per byte, MSB of stream at end):");
    for b in &our_fse {
        for i in 0..8 {
            print!("{}", (b >> i) & 1);
        }
        print!(" ");
    }
    println!();

    println!("Ref binary (LSB first per byte, MSB of stream at end):");
    for b in &ref_fse {
        for i in 0..8 {
            print!("{}", (b >> i) & 1);
        }
        print!(" ");
    }
    println!();

    // Build FSE tables
    let ll_table = FseTable::from_predefined(&LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG).unwrap();
    let of_table = FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(&MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG).unwrap();

    // Decode our bitstream
    println!("\n=== Decoding OUR Bitstream (using OUR decoder) ===");
    decode_bitstream(&our_fse, &ll_table, &of_table, &ml_table, 2);

    // Decode reference bitstream
    println!("\n=== Decoding REFERENCE Bitstream (using OUR decoder) ===");
    decode_bitstream(&ref_fse, &ll_table, &of_table, &ml_table, 2);

    // Try decoding reference with RFC order (OF, ML, LL for extras)
    println!("\n=== Decoding REFERENCE with RFC order (OF, ML, LL extras) ===");
    decode_bitstream_rfc_order(&ref_fse, &ll_table, &of_table, &ml_table, 2);
}

fn decode_bitstream_rfc_order(fse_bits: &[u8], ll_table: &FseTable, of_table: &FseTable, ml_table: &FseTable, seq_count: usize) {
    if fse_bits.is_empty() {
        println!("  Empty bitstream");
        return;
    }

    let mut bits = BitReader::new(fse_bits);
    if bits.init_from_end().is_err() {
        println!("  Failed to init from end");
        return;
    }

    println!("  Total bits: {}", bits.bits_remaining());

    // Read initial states (LL, OF, ML order per RFC)
    let ll_state = bits.read_bits(LITERAL_LENGTH_ACCURACY_LOG as usize).unwrap_or(999);
    let of_state = bits.read_bits(OFFSET_ACCURACY_LOG as usize).unwrap_or(999);
    let ml_state = bits.read_bits(MATCH_LENGTH_ACCURACY_LOG as usize).unwrap_or(999);

    println!("  Initial states: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    // Get initial symbols from states
    let ll_sym = ll_table.decode(ll_state as usize).symbol;
    let of_sym = of_table.decode(of_state as usize).symbol;
    let ml_sym = ml_table.decode(ml_state as usize).symbol;

    println!("  Initial symbols: LL={}, OF={}, ML={}", ll_sym, of_sym, ml_sym);

    // Switch to LSB mode
    bits.switch_to_lsb_mode().unwrap();

    // Set up decoders
    let mut ll_decoder = FseDecoder::new(ll_table);
    let mut of_decoder = FseDecoder::new(of_table);
    let mut ml_decoder = FseDecoder::new(ml_table);

    let mut bits2 = BitReader::new(fse_bits);
    bits2.init_from_end().unwrap();
    ll_decoder.init_state(&mut bits2).unwrap();
    of_decoder.init_state(&mut bits2).unwrap();
    ml_decoder.init_state(&mut bits2).unwrap();
    bits2.switch_to_lsb_mode().unwrap();

    println!("  Bits remaining for sequences: {}", bits2.bits_remaining());

    // Decode sequences
    for i in 0..seq_count {
        let is_last = i == seq_count - 1;
        println!("\n  --- Seq {} {} ---", i, if is_last { "(LAST)" } else { "" });

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

        // Calculate extra bits needed
        let ll_extra_bits = get_ll_extra_bits(ll_code);
        let ml_extra_bits = get_ml_extra_bits(ml_code);
        let of_extra_bits = of_code;

        println!("  Extra bits: OF={}, ML={}, LL={}", of_extra_bits, ml_extra_bits, ll_extra_bits);
        println!("  Bits remaining: {}", bits2.bits_remaining());

        // Read extra bits in RFC order: OF, ML, LL
        let of_extra = if of_extra_bits > 0 { bits2.read_bits(of_extra_bits as usize).unwrap_or(999) } else { 0 };
        let ml_extra = if ml_extra_bits > 0 { bits2.read_bits(ml_extra_bits as usize).unwrap_or(999) } else { 0 };
        let ll_extra = if ll_extra_bits > 0 { bits2.read_bits(ll_extra_bits as usize).unwrap_or(999) } else { 0 };

        println!("  Extras read (OF,ML,LL): OF={}, ML={}, LL={}", of_extra, ml_extra, ll_extra);

        // Compute actual values
        let ll_value = get_ll_baseline(ll_code) + ll_extra;
        let of_value = (1u32 << of_code) + of_extra;  // Fixed formula!
        let ml_value = get_ml_baseline(ml_code) + ml_extra;

        println!("  Values: literal_length={}, offset={}, match_length={}", ll_value, of_value, ml_value);

        // Update FSE states (skip for last sequence)
        // RFC order for update: LL, ML, OF
        if !is_last {
            ll_decoder.update_state(&mut bits2).ok();
            ml_decoder.update_state(&mut bits2).ok();
            of_decoder.update_state(&mut bits2).ok();
            println!("  After FSE update, bits remaining: {}", bits2.bits_remaining());
        }
    }

    println!("\n  Final bits remaining: {}", bits2.bits_remaining());
}

fn extract_fse_bitstream(compressed: &[u8]) -> Vec<u8> {
    // Frame header at pos 4
    let fhd = compressed[4];
    let single_segment = (fhd >> 5) & 1;
    let mut pos = 5;
    if single_segment == 0 {
        pos += 1; // window descriptor
    }

    // FCS
    let fcs_size = (fhd >> 6) & 3;
    let fcs_bytes = match fcs_size {
        0 => if single_segment == 1 { 1 } else { 0 },
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    pos += fcs_bytes;

    // Block header
    let bh = (compressed[pos] as u32)
        | ((compressed[pos + 1] as u32) << 8)
        | ((compressed[pos + 2] as u32) << 16);
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        return vec![];
    }

    let block = &compressed[pos..pos + block_size];

    // Parse literals header
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

    // Skip seq count and mode byte
    let seq_count_byte = seq_section[0];
    let header_len = if seq_count_byte < 128 { 1 } else if seq_count_byte < 255 { 2 } else { 3 };

    // Return the FSE bitstream (after seq count + mode byte)
    seq_section[header_len + 1..].to_vec()
}

fn decode_bitstream(fse_bits: &[u8], ll_table: &FseTable, of_table: &FseTable, ml_table: &FseTable, seq_count: usize) {
    if fse_bits.is_empty() {
        println!("  Empty bitstream");
        return;
    }

    let mut bits = BitReader::new(fse_bits);
    if bits.init_from_end().is_err() {
        println!("  Failed to init from end");
        return;
    }

    println!("  Total bits: {}", bits.bits_remaining());

    // Read initial states (LL, OF, ML order)
    let ll_state = bits.read_bits(LITERAL_LENGTH_ACCURACY_LOG as usize).unwrap_or(999);
    let of_state = bits.read_bits(OFFSET_ACCURACY_LOG as usize).unwrap_or(999);
    let ml_state = bits.read_bits(MATCH_LENGTH_ACCURACY_LOG as usize).unwrap_or(999);

    println!("  Initial states: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    // Get initial symbols from states
    let ll_sym = ll_table.decode(ll_state as usize).symbol;
    let of_sym = of_table.decode(of_state as usize).symbol;
    let ml_sym = ml_table.decode(ml_state as usize).symbol;

    println!("  Initial symbols: LL={}, OF={}, ML={}", ll_sym, of_sym, ml_sym);
    println!("  Bits after states: {}", bits.bits_remaining());

    // Switch to LSB mode for extra bits and state updates
    if bits.switch_to_lsb_mode().is_err() {
        println!("  Failed to switch to LSB mode");
        return;
    }

    // Set up decoders
    let mut ll_decoder = FseDecoder::new(ll_table);
    let mut of_decoder = FseDecoder::new(of_table);
    let mut ml_decoder = FseDecoder::new(ml_table);

    // Manually set states using init
    let mut bits2 = BitReader::new(fse_bits);
    bits2.init_from_end().unwrap();
    ll_decoder.init_state(&mut bits2).unwrap();
    of_decoder.init_state(&mut bits2).unwrap();
    ml_decoder.init_state(&mut bits2).unwrap();
    bits2.switch_to_lsb_mode().unwrap();

    println!("  Bits remaining for sequences: {}", bits2.bits_remaining());

    // Decode sequences
    for i in 0..seq_count {
        let is_last = i == seq_count - 1;
        println!("\n  --- Seq {} {} ---", i, if is_last { "(LAST)" } else { "" });

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

        // Calculate extra bits needed
        let ll_extra_bits = get_ll_extra_bits(ll_code);
        let ml_extra_bits = get_ml_extra_bits(ml_code);
        let of_extra_bits = of_code;

        println!("  Extra bits: LL={}, ML={}, OF={}", ll_extra_bits, ml_extra_bits, of_extra_bits);
        println!("  Bits remaining: {}", bits2.bits_remaining());

        // Read extra bits in LL, ML, OF order (our decoder's order)
        let ll_extra = if ll_extra_bits > 0 { bits2.read_bits(ll_extra_bits as usize).unwrap_or(999) } else { 0 };
        let ml_extra = if ml_extra_bits > 0 { bits2.read_bits(ml_extra_bits as usize).unwrap_or(999) } else { 0 };
        let of_extra = if of_extra_bits > 0 { bits2.read_bits(of_extra_bits as usize).unwrap_or(999) } else { 0 };

        println!("  Extras read: LL={}, ML={}, OF={}", ll_extra, ml_extra, of_extra);

        // Compute actual values
        let ll_value = get_ll_baseline(ll_code) + ll_extra;
        let of_value = if of_code > 0 { (1u32 << of_code) + of_extra } else { of_extra };
        let ml_value = get_ml_baseline(ml_code) + ml_extra;

        println!("  Values: literal_length={}, offset={}, match_length={}", ll_value, of_value, ml_value);

        // Update FSE states (skip for last sequence)
        if !is_last {
            ll_decoder.update_state(&mut bits2).ok();
            ml_decoder.update_state(&mut bits2).ok();
            of_decoder.update_state(&mut bits2).ok();
            println!("  After FSE update, bits remaining: {}", bits2.bits_remaining());
        }
    }

    println!("\n  Final bits remaining: {}", bits2.bits_remaining());
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
        16 => 16, 17 => 18, 18 => 20, 19 => 24, 20 => 28, 21 => 36,
        22 => 44, 23 => 60, 24 => 76, 25 => 108, 26 => 140, 27 => 204,
        28 => 268, 29 => 396, 30 => 524, 31 => 780, 32 => 1036, 33 => 1548,
        34 => 2060, 35 => 3084, _ => 0,
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
            32 => 35, 33 => 37, 34 => 39, 35 => 43, 36 => 47, 37 => 51,
            38 => 59, 39 => 67, 40 => 83, 41 => 99, 42 => 131, 43 => 259,
            44 => 515, 45 => 1027, 46 => 2051, 47 => 4099, 48 => 8195,
            49 => 16387, 50 => 32771, 51 => 65539, 52 => 131075, _ => 0,
        }
    }
}

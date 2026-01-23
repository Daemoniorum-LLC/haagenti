//! Decode both reference and our bitstream for "ABCD"x25.

use haagenti_zstd::fse::{
    BitReader, FseDecoder, FseTable, LITERAL_LENGTH_ACCURACY_LOG,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== Decoding Both Bitstreams for ABCD x 25 ===\n");
    println!("Expected sequence: LL=4, OF=7 (offset_value), ML=96");
    println!("Expected codes: LL=4, OF=2 (with 3 extra), ML=41 (with 13 extra)");

    println!("\n--- Reference Bitstream [fd, e4, 88] ---");
    decode_bitstream(&[0xfd, 0xe4, 0x88], 1);

    println!("\n--- Our Bitstream [7a, ba, 44] ---");
    decode_bitstream(&[0x7a, 0xba, 0x44], 1);
}

fn decode_bitstream(bitstream: &[u8], seq_count: usize) {
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
        return;
    }

    let total_bits = bits.bits_remaining();
    println!("  Total bits: {}", total_bits);

    // Print binary
    print!("  Binary (MSB first): ");
    for b in bitstream.iter().rev() {
        print!("{:08b} ", b);
    }
    println!();

    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    let ll_state = ll_decoder.state();
    let of_state = of_decoder.state();
    let ml_state = ml_decoder.state();

    println!(
        "  Initial states: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );
    println!("  Bits after states: {}", bits.bits_remaining());

    // Get codes
    let ll_code = ll_table.decode(ll_state).symbol;
    let of_code = of_table.decode(of_state).symbol;
    let ml_code = ml_table.decode(ml_state).symbol;

    println!("  Codes: LL={}, OF={}, ML={}", ll_code, of_code, ml_code);

    // Read extra bits
    bits.switch_to_lsb_mode().unwrap();

    for i in 0..seq_count {
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        let of_extra_bits = of_code as usize;
        let ml_extra_bits = get_ml_extra_bits(ml_code) as usize;
        let ll_extra_bits = get_ll_extra_bits(ll_code) as usize;

        println!(
            "  Seq {}: reading extras OF({} bits), ML({} bits), LL({} bits)",
            i, of_extra_bits, ml_extra_bits, ll_extra_bits
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

        let ll_value = get_ll_baseline(ll_code) + ll_extra as u32;
        let of_value = (1u32 << of_code) + of_extra;
        let ml_baseline = get_ml_baseline(ml_code);
        let ml_value = ml_baseline + ml_extra as u32;

        println!(
            "  Seq {}: LL={} (code={}), OF={} (code={}+{}), ML={} (code={} base {} +{})",
            i,
            ll_value,
            ll_code,
            of_value,
            of_code,
            of_extra,
            ml_value,
            ml_code,
            ml_baseline,
            ml_extra
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
        41 => 83, // ML 41 has baseline 83
        42 => 99,
        43 => 131,
        44 => 259,
        45 => 515,
        _ => 3,
    }
}

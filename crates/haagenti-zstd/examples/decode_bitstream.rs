//! Decode and compare bitstreams bit by bit

use haagenti_zstd::block::{LITERAL_LENGTH_BASELINE, MATCH_LENGTH_BASELINE};
use haagenti_zstd::fse::{BitReader, FseDecoder, FseTable};

// Predefined distributions (from RFC 8878)
const PREDEFINED_LL_DISTRIBUTION: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const PREDEFINED_LL_ACCURACY_LOG: u8 = 6;

const PREDEFINED_OF_DISTRIBUTION: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
const PREDEFINED_OF_ACCURACY_LOG: u8 = 5;

const PREDEFINED_ML_DISTRIBUTION: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const PREDEFINED_ML_ACCURACY_LOG: u8 = 6;

fn decode_literal_length(code: u8, extra_bits: u32) -> u32 {
    if code as usize >= LITERAL_LENGTH_BASELINE.len() {
        return 0;
    }
    let (bits, baseline) = LITERAL_LENGTH_BASELINE[code as usize];
    if bits == 0 {
        baseline
    } else {
        baseline + (extra_bits & ((1 << bits) - 1))
    }
}

fn decode_match_length(code: u8, extra_bits: u32) -> u32 {
    if code as usize >= MATCH_LENGTH_BASELINE.len() {
        return 3;
    }
    let (bits, baseline) = MATCH_LENGTH_BASELINE[code as usize];
    if bits == 0 {
        baseline
    } else {
        baseline + (extra_bits & ((1 << bits) - 1))
    }
}

fn decode_offset(code: u8, extra_bits: u32) -> u32 {
    let code = code.min(31);
    (1u32 << code) + extra_bits
}

fn offset_code_extra_bits(code: u8) -> u8 {
    code.min(31)
}

fn main() {
    // Our bitstream from the 806-byte test
    // Expected: LL=28, OF=23, ML=42 (final encoder states)
    let our_bitstream = [0x85u8, 0x3d, 0x05, 0xca, 0x9e, 0x02, 0x15, 0xf5, 0x72, 0x01];

    println!("=== OUR BITSTREAM (trace state reading) ===");
    println!("Bytes: {:02x?}", our_bitstream);
    println!("Expected initial states: LL=28, OF=23, ML=42\n");

    // Manual trace of bit reading
    let last_byte = our_bitstream[9];
    let sentinel_pos = 7 - last_byte.leading_zeros() as i8;
    println!("Last byte: 0x{:02x} = {:08b}", last_byte, last_byte);
    println!("Sentinel at bit position: {}", sentinel_pos);

    // Show bytes in binary
    println!("\nBytes (MSB first, position from end):");
    for i in (0..10).rev() {
        println!(
            "  Byte {}: 0x{:02x} = {:08b}",
            i, our_bitstream[i], our_bitstream[i]
        );
    }

    // Now decode using the actual BitReader
    println!("\n=== Using BitReader ===");
    decode_sequences(&our_bitstream, 3);
}

fn decode_sequences(data: &[u8], num_sequences: usize) {
    println!("Bitstream bytes: {:02x?}", data);
    println!("Binary (LSB first per byte):");
    for (i, b) in data.iter().enumerate() {
        print!("  {:02}: {:08b} ", i, b);
        // Show LSB to MSB
        for j in 0..8 {
            print!("{}", (b >> j) & 1);
        }
        println!();
    }

    // Build predefined tables
    let ll_table =
        FseTable::from_predefined(&PREDEFINED_LL_DISTRIBUTION, PREDEFINED_LL_ACCURACY_LOG).unwrap();
    let of_table =
        FseTable::from_predefined(&PREDEFINED_OF_DISTRIBUTION, PREDEFINED_OF_ACCURACY_LOG).unwrap();
    let ml_table =
        FseTable::from_predefined(&PREDEFINED_ML_DISTRIBUTION, PREDEFINED_ML_ACCURACY_LOG).unwrap();

    // Create bit reader
    let mut bits = BitReader::new(data);
    bits.init_from_end().unwrap();

    // Initialize decoders
    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    // Read initial states
    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    println!(
        "\nInitial states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );

    // Switch to LSB mode for extra bits
    bits.switch_to_lsb_mode().unwrap();

    // Decode each sequence
    for i in 0..num_sequences {
        let is_last = i == num_sequences - 1;

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!(
            "\nSequence {}: codes LL={}, OF={}, ML={}",
            i, ll_code, of_code, ml_code
        );

        // Read extra bits in LL, ML, OF order
        let ll_extra_bits_needed = if ll_code < LITERAL_LENGTH_BASELINE.len() as u8 {
            LITERAL_LENGTH_BASELINE[ll_code as usize].0
        } else {
            0
        };

        let ml_extra_bits_needed = if ml_code < MATCH_LENGTH_BASELINE.len() as u8 {
            MATCH_LENGTH_BASELINE[ml_code as usize].0
        } else {
            0
        };

        let of_extra_bits_needed = offset_code_extra_bits(of_code);

        println!(
            "  Extra bits needed: LL={}, ML={}, OF={}",
            ll_extra_bits_needed, ml_extra_bits_needed, of_extra_bits_needed
        );

        let ll_extra = if ll_extra_bits_needed > 0 {
            bits.read_bits(ll_extra_bits_needed as usize).unwrap_or(0)
        } else {
            0
        };

        let ml_extra = if ml_extra_bits_needed > 0 {
            bits.read_bits(ml_extra_bits_needed as usize).unwrap_or(0)
        } else {
            0
        };

        let of_extra = if of_extra_bits_needed > 0 {
            bits.read_bits(of_extra_bits_needed as usize).unwrap_or(0)
        } else {
            0
        };

        println!(
            "  Extra bits read: LL={}, ML={}, OF={}",
            ll_extra, ml_extra, of_extra
        );

        if !is_last {
            // Update states
            ll_decoder.update_state(&mut bits).unwrap();
            ml_decoder.update_state(&mut bits).unwrap();
            of_decoder.update_state(&mut bits).unwrap();

            println!(
                "  New states: LL={}, OF={}, ML={}",
                ll_decoder.state(),
                of_decoder.state(),
                ml_decoder.state()
            );
        }

        // Decode final values
        let literal_length = decode_literal_length(ll_code, ll_extra);
        let match_length = decode_match_length(ml_code, ml_extra);
        let offset_value = decode_offset(of_code, of_extra);

        println!(
            "  Decoded: LL={}, ML={}, offset_value={}",
            literal_length, match_length, offset_value
        );
    }
}

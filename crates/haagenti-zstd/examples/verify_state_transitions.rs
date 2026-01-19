//! Verify FSE state transitions match between encoder and decoder.
//! This tests that encode_symbol produces bits that, when decoded, give the correct next state.

use haagenti_zstd::fse::{
    FseTable, TansEncoder, FseDecoder, BitReader, FseBitWriter,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
};

fn main() {
    println!("=== Verifying FSE State Transitions ===\n");

    // Test with our failing 2-sequence case
    // Seq 0: ll_code=9, of_code=3, ml_code=5
    // Seq 1: ll_code=1, of_code=0, ml_code=1

    println!("=== LL Table Transitions ===");
    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    ).unwrap();

    // Find states that decode to symbol 9 and symbol 1
    println!("States for LL symbols:");
    for state in 0..64 {
        let entry = ll_table.decode(state);
        if entry.symbol == 9 || entry.symbol == 1 {
            println!("  State {:2}: symbol={:2}, num_bits={}, baseline={}",
                     state, entry.symbol, entry.num_bits, entry.baseline);
        }
    }

    println!("\n=== Testing Encoder-Decoder Round Trip ===");

    // Test: encode symbols [9, 1] (last first in tANS order), verify decode
    test_round_trip(&ll_table, &[1, 9], "LL (codes 1, 9)");

    println!("\n=== OF Table Transitions ===");
    let of_table = FseTable::from_predefined(
        &OFFSET_DEFAULT_DISTRIBUTION,
        OFFSET_ACCURACY_LOG,
    ).unwrap();

    println!("States for OF symbols 0, 3:");
    for state in 0..32 {
        let entry = of_table.decode(state);
        if entry.symbol == 0 || entry.symbol == 3 {
            println!("  State {:2}: symbol={:2}, num_bits={}, baseline={}",
                     state, entry.symbol, entry.num_bits, entry.baseline);
        }
    }

    test_round_trip(&of_table, &[0, 3], "OF (codes 0, 3)");

    println!("\n=== ML Table Transitions ===");
    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    ).unwrap();

    println!("States for ML symbols 1, 5:");
    for state in 0..64 {
        let entry = ml_table.decode(state);
        if entry.symbol == 1 || entry.symbol == 5 {
            println!("  State {:2}: symbol={:2}, num_bits={}, baseline={}",
                     state, entry.symbol, entry.num_bits, entry.baseline);
        }
    }

    test_round_trip(&ml_table, &[1, 5], "ML (codes 1, 5)");

    println!("\n=== Full Interleaved Test ===");
    test_full_interleaved();
}

fn test_round_trip(table: &FseTable, symbols: &[u8], name: &str) {
    println!("\nTesting {} with symbols {:?}:", name, symbols);

    let accuracy_log = if table.size() == 64 { 6 } else { 5 };
    let mut encoder = TansEncoder::from_decode_table(table);

    // Init with LAST symbol (index 0 in our reversed order)
    encoder.init_state(symbols[0]);
    let init_state = encoder.get_state();
    println!("  Encoder init with symbol {}: state={}", symbols[0], init_state);

    // Verify init state decodes to last symbol
    let init_entry = table.decode(init_state as usize);
    println!("  State {} decodes to symbol {} (expected {})",
             init_state, init_entry.symbol, symbols[0]);
    if init_entry.symbol != symbols[0] {
        println!("  ERROR: Init state doesn't decode to expected symbol!");
        return;
    }

    // Encode remaining symbols in reverse order
    let mut all_bits = Vec::new();
    for &sym in symbols.iter().skip(1).rev() {
        let (bits, nbits) = encoder.encode_symbol(sym);
        println!("  Encode symbol {}: bits={} ({} bits), new_state={}",
                 sym, bits, nbits, encoder.get_state());
        all_bits.push((bits, nbits));
    }

    let final_state = encoder.get_state();
    println!("  Final encoder state: {}", final_state);

    // Verify final state decodes to first symbol
    let final_entry = table.decode(final_state as usize);
    println!("  State {} decodes to symbol {} (expected {})",
             final_state, final_entry.symbol, symbols[symbols.len() - 1]);

    // Now decode
    println!("\n  Decoding:");

    // Build bitstream: bits (forward), then state (at MSB end)
    let mut writer = FseBitWriter::new();
    for (bits, nbits) in all_bits.iter() {
        writer.write_bits(*bits, *nbits);
    }
    writer.write_bits(final_state, accuracy_log);
    let bitstream = writer.finish();
    println!("  Bitstream: {:02x?}", bitstream);

    // Decode
    let mut decoder = FseDecoder::new(table);
    let mut bits = BitReader::new(&bitstream);
    bits.init_from_end().unwrap();

    decoder.init_state(&mut bits).unwrap();
    println!("  Decoder init state: {} (symbol {})",
             decoder.state(), decoder.peek_symbol());

    bits.switch_to_lsb_mode().unwrap();

    // Decode each symbol (except last which doesn't need state update)
    for i in 0..symbols.len() {
        let sym = decoder.peek_symbol();
        println!("  Seq {}: decoded symbol {} (expected {})",
                 i, sym, symbols[symbols.len() - 1 - i]);

        if sym != symbols[symbols.len() - 1 - i] {
            println!("  ERROR: Symbol mismatch!");
        }

        if i < symbols.len() - 1 {
            decoder.decode_symbol(&mut bits).unwrap();
            println!("    New state: {} (symbol {})", decoder.state(), decoder.peek_symbol());
        }
    }
}

fn test_full_interleaved() {
    use haagenti_zstd::fse::InterleavedTansEncoder;

    // Our failing case:
    // Seq 0: ll=9, of=3, ml=5
    // Seq 1: ll=1, of=0, ml=1

    let mut tans = InterleavedTansEncoder::new_predefined();

    // Init with LAST sequence (seq 1)
    tans.init_states(1, 0, 1);
    let (ll_s, of_s, ml_s) = tans.get_states();
    println!("Init with (1, 0, 1): states = ({}, {}, {})", ll_s, of_s, ml_s);

    // Encode seq 0 (going backward)
    let fse_bits = tans.encode_sequence(9, 3, 5);
    let (ll_s, of_s, ml_s) = tans.get_states();
    println!("After encode (9, 3, 5): states = ({}, {}, {})", ll_s, of_s, ml_s);
    println!("FSE bits: LL=({},{}), OF=({},{}), ML=({},{})",
             fse_bits[0].0, fse_bits[0].1,
             fse_bits[1].0, fse_bits[1].1,
             fse_bits[2].0, fse_bits[2].1);

    // Build bitstream
    let mut writer = FseBitWriter::new();

    // Seq 0 extras (LL=9 has 0 extra, ML=5 has 0 extra, OF=3 has 3 extra bits with value 4)
    // OF value = 12 = (1<<3) + 4, so extra = 4
    writer.write_bits(4, 3);  // OF extra

    // Seq 0 FSE update (LL, ML, OF order as per zstd spec)
    writer.write_bits(fse_bits[0].0, fse_bits[0].1);  // LL
    writer.write_bits(fse_bits[2].0, fse_bits[2].1);  // ML
    writer.write_bits(fse_bits[1].0, fse_bits[1].1);  // OF

    // Seq 1 extras (all 0)
    // (nothing)

    // Initial states (ML, OF, LL order, read as LL, OF, ML)
    writer.write_bits(ml_s, 6);
    writer.write_bits(of_s, 5);
    writer.write_bits(ll_s, 6);

    let bitstream = writer.finish();
    println!("\nBuilt bitstream: {:02x?}", bitstream);

    // Now decode
    println!("\nDecoding:");

    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    ).unwrap();
    let of_table = FseTable::from_predefined(
        &OFFSET_DEFAULT_DISTRIBUTION,
        OFFSET_ACCURACY_LOG,
    ).unwrap();
    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    ).unwrap();

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    let mut bits = BitReader::new(&bitstream);
    bits.init_from_end().unwrap();

    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    println!("Initial states: LL={}, OF={}, ML={}",
             ll_decoder.state(), of_decoder.state(), ml_decoder.state());
    println!("Initial symbols: LL={}, OF={}, ML={}",
             ll_decoder.peek_symbol(), of_decoder.peek_symbol(), ml_decoder.peek_symbol());

    bits.switch_to_lsb_mode().unwrap();

    // Seq 0
    println!("\nSequence 0:");
    let ll_sym = ll_decoder.peek_symbol();
    let of_sym = of_decoder.peek_symbol();
    let ml_sym = ml_decoder.peek_symbol();
    println!("  Symbols: LL={} (expect 9), OF={} (expect 3), ML={} (expect 5)",
             ll_sym, of_sym, ml_sym);

    // Read extras
    let of_extra = bits.read_bits(3).unwrap();  // 3 bits for OF code 3
    println!("  OF extra: {} (expect 4)", of_extra);

    // FSE updates
    ll_decoder.decode_symbol(&mut bits).unwrap();
    ml_decoder.decode_symbol(&mut bits).unwrap();
    of_decoder.decode_symbol(&mut bits).unwrap();

    println!("  New states: LL={}, OF={}, ML={}",
             ll_decoder.state(), of_decoder.state(), ml_decoder.state());
    println!("  New symbols: LL={}, OF={}, ML={}",
             ll_decoder.peek_symbol(), of_decoder.peek_symbol(), ml_decoder.peek_symbol());

    // Seq 1
    println!("\nSequence 1:");
    let ll_sym = ll_decoder.peek_symbol();
    let of_sym = of_decoder.peek_symbol();
    let ml_sym = ml_decoder.peek_symbol();
    println!("  Symbols: LL={} (expect 1), OF={} (expect 0), ML={} (expect 1)",
             ll_sym, of_sym, ml_sym);

    println!("\nBits remaining: {}", bits.bits_remaining());
}

//! Check the OF predefined table for symbol 0 handling

use haagenti_zstd::fse::{
    FseTable, TansEncoder,
    OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
};

fn main() {
    println!("=== Checking OF Predefined Table ===\n");

    // Build the OF table
    let of_table = FseTable::from_predefined(
        &OFFSET_DEFAULT_DISTRIBUTION,
        OFFSET_ACCURACY_LOG,
    ).unwrap();

    println!("OF table: accuracy_log={}, size={}\n", OFFSET_ACCURACY_LOG, 1 << OFFSET_ACCURACY_LOG);

    // Print the full decode table
    println!("OF Decode Table:");
    for state in 0..32 {
        let entry = of_table.decode(state);
        println!("  state {:2}: symbol={:2}, num_bits={}, baseline={:2}",
                 state, entry.symbol, entry.num_bits, entry.baseline);
    }

    // Print the distribution
    println!("\nOF Default Distribution:");
    for (sym, &prob) in OFFSET_DEFAULT_DISTRIBUTION.iter().enumerate() {
        if prob != 0 {
            println!("  symbol {:2}: prob={:2}", sym, prob);
        }
    }

    // Find states that decode to symbol 0
    println!("\nStates that decode to symbol 0:");
    let mut count = 0;
    for state in 0..32 {
        if of_table.decode(state).symbol == 0 {
            let entry = of_table.decode(state);
            println!("  state {}: symbol={}, num_bits={}, baseline={}",
                     state, entry.symbol, entry.num_bits, entry.baseline);
            count += 1;
        }
    }
    println!("  Total: {} state(s)", count);

    // Now check the encoder
    println!("\n=== Checking OF Encoder ===");
    let mut encoder = TansEncoder::from_decode_table(&of_table);

    // Initialize with symbol 0
    encoder.init_state(0);
    let init_state = encoder.get_state();
    println!("\nEncoder init_state(0):");
    println!("  Resulting state: {}", init_state);
    println!("  This state decodes to symbol: {}", of_table.decode(init_state as usize).symbol);

    // Initialize with symbol 5 (to check we have different starting state)
    encoder.init_state(5);
    let state_5 = encoder.get_state();
    println!("\nEncoder init_state(5):");
    println!("  Resulting state: {}", state_5);
    println!("  This state decodes to symbol: {}", of_table.decode(state_5 as usize).symbol);

    // Now encode symbol 0 from state for symbol 5
    encoder.init_state(5);
    let (bits, num_bits) = encoder.encode_symbol(0);
    let new_state = encoder.get_state();
    println!("\nEncoder.encode_symbol(0) from state_5:");
    println!("  Bits output: {} ({} bits)", bits, num_bits);
    println!("  New state: {}", new_state);
    println!("  New state decodes to symbol: {}", of_table.decode(new_state as usize).symbol);

    // Verify the decoder can reconstruct symbol 0
    println!("\n=== Verification ===");
    println!("If decoder starts at state {} and reads {} bits:", new_state, num_bits);
    let entry = of_table.decode(new_state as usize);
    println!("  Entry: symbol={}, num_bits={}, baseline={}", entry.symbol, entry.num_bits, entry.baseline);
    println!("  Decoder would read {} bits from stream", entry.num_bits);
    println!("  With value {}, new state = {} + {} = {}", bits, entry.baseline, bits, entry.baseline as u32 + bits);

    // Check: what state gives symbol 5?
    println!("\n=== Cross-check: which state gives symbol 5? ===");
    for state in 0..32 {
        if of_table.decode(state).symbol == 5 {
            let entry = of_table.decode(state);
            println!("  state {}: symbol={}, num_bits={}, baseline={}",
                     state, entry.symbol, entry.num_bits, entry.baseline);
        }
    }

    // The key question: when encoder outputs (bits, num_bits) and goes to new_state,
    // can decoder read num_bits from current state and arrive at the same new_state?
    println!("\n=== Key Verification ===");

    // Start fresh for clear test
    encoder.init_state(5);  // This sets up for symbol 5
    let starting_decode_state = encoder.get_state();
    println!("Encoder state for symbol 5: {}", starting_decode_state);
    println!("  Decodes to: {}", of_table.decode(starting_decode_state as usize).symbol);

    // Now encode symbol 0
    let (out_bits, out_num) = encoder.encode_symbol(0);
    let ending_decode_state = encoder.get_state();
    println!("\nAfter encoding symbol 0:");
    println!("  Output: {} ({} bits)", out_bits, out_num);
    println!("  New state: {}", ending_decode_state);
    println!("  New state decodes to: {}", of_table.decode(ending_decode_state as usize).symbol);

    // Now simulate decoder
    // Decoder is at `ending_decode_state`, wants to update
    let decoder_entry = of_table.decode(ending_decode_state as usize);
    println!("\nDecoder at state {}:", ending_decode_state);
    println!("  Symbol (what we just decoded): {}", decoder_entry.symbol);
    println!("  num_bits to read for update: {}", decoder_entry.num_bits);
    println!("  baseline for update: {}", decoder_entry.baseline);

    // Decoder reads `decoder_entry.num_bits` bits and gets `out_bits`
    let decoder_new_state = decoder_entry.baseline as u32 + out_bits;
    println!("  If decoder reads {} bits and gets value {}:", decoder_entry.num_bits, out_bits);
    println!("  New state = {} + {} = {}", decoder_entry.baseline, out_bits, decoder_new_state);
    println!("  This state decodes to symbol: {}", of_table.decode(decoder_new_state as usize).symbol);

    // This should give us starting_decode_state (which decodes to symbol 5)
    if decoder_new_state == starting_decode_state {
        println!("\n✓ VERIFIED: Decoder correctly transitions back to symbol 5 state");
    } else {
        println!("\n✗ MISMATCH: Expected state {}, got {}", starting_decode_state, decoder_new_state);
    }
}

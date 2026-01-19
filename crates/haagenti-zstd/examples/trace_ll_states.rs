//! Trace LL states to find why state 17 is wrong.

use haagenti_zstd::fse::{
    FseTable, TansEncoder,
    LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
};

fn main() {
    println!("=== Tracing LL States ===\n");

    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    ).unwrap();

    // Print states around the problematic state 17
    println!("LL decode table (states 15-20):");
    for state in 15..=20 {
        let entry = ll_table.decode(state);
        println!("  State {:2}: symbol={:2}, num_bits={}, baseline={}",
                 state, entry.symbol, entry.num_bits, entry.baseline);
    }

    // Check what states symbol 25 uses
    println!("\nSymbol 25 states:");
    for state in 0..64 {
        if ll_table.decode(state).symbol == 25 {
            let entry = ll_table.decode(state);
            println!("  State {:2}: num_bits={}, baseline={}",
                     state, entry.num_bits, entry.baseline);
        }
    }

    // Trace what happens with LL codes 9 and 1 (from the failing case)
    println!("\n=== Failing Case: LL codes [9, 1] ===");
    trace_encoding(9, 1, &ll_table);

    // Trace working case: LL codes 4 and 1
    println!("\n=== Working Case: LL codes [4, 1] ===");
    trace_encoding(4, 1, &ll_table);
}

fn trace_encoding(first_ll: u8, last_ll: u8, table: &FseTable) {
    let mut encoder = TansEncoder::from_decode_table(table);

    // Init with last symbol
    encoder.init_state(last_ll);
    let init_state = encoder.get_state();
    println!("  Init with symbol {}: state={}", last_ll, init_state);
    println!("    State {} decodes to symbol {}", init_state, table.decode(init_state as usize).symbol);

    // Encode first symbol
    let (bits, nbits) = encoder.encode_symbol(first_ll);
    let final_state = encoder.get_state();
    println!("  Encode symbol {}: bits={}, nbits={}", first_ll, bits, nbits);
    println!("    New state: {}", final_state);
    println!("    State {} decodes to symbol {}", final_state, table.decode(final_state as usize).symbol);

    // Check if decoder can correctly use these
    let entry = table.decode(final_state as usize);
    println!("  Decoder at state {}:", final_state);
    println!("    Symbol: {}", entry.symbol);
    println!("    num_bits: {}", entry.num_bits);
    println!("    baseline: {}", entry.baseline);

    // Simulate decoder state transition
    let next_state = entry.baseline as u32 + bits;
    println!("    After reading {} bits with value {}: new_state = {} + {} = {}",
             entry.num_bits, bits, entry.baseline, bits, next_state);
    if next_state < 64 {
        println!("    State {} decodes to symbol {} (expected {})",
                 next_state, table.decode(next_state as usize).symbol, last_ll);
    }
}

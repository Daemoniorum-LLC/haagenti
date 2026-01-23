//! Trace which ML states are used for working vs failing cases.

use haagenti_zstd::fse::{
    FseTable, TansEncoder, MATCH_LENGTH_ACCURACY_LOG, MATCH_LENGTH_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== Tracing ML States ===\n");

    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    )
    .unwrap();

    // Print which states decode to which symbols
    println!("ML decode table (our table):");
    for state in 0..64 {
        let entry = ml_table.decode(state);
        println!(
            "  State {:2}: symbol={:2}, num_bits={}, baseline={}",
            state, entry.symbol, entry.num_bits, entry.baseline
        );
    }

    // Reference ML decode table (from zstd's seqSymbol format)
    // Note: These are what reference zstd expects
    println!("\n=== Symbol Mapping ===");
    for sym in 0..10 {
        println!("Symbol {} states in our table:", sym);
        for state in 0..64 {
            if ml_table.decode(state).symbol == sym {
                print!(" {}", state);
            }
        }
        println!();
    }

    // Now trace what happens for working case (ML codes 1, 1)
    println!("\n=== Working Case: ML codes [1, 1] ===");
    trace_encoding(1, 1, &ml_table);

    // Trace failing case (ML codes 5, 1)
    println!("\n=== Failing Case: ML codes [5, 1] ===");
    trace_encoding(5, 1, &ml_table);

    // Check if the issue is with specific states
    println!("\n=== State Analysis ===");
    // States that might be problematic (from verify_tables)
    for state in [23, 24, 25, 26, 27] {
        let entry = ml_table.decode(state);
        println!(
            "State {}: symbol={}, num_bits={}, baseline={}",
            state, entry.symbol, entry.num_bits, entry.baseline
        );
    }
}

fn trace_encoding(first_ml: u8, last_ml: u8, table: &FseTable) {
    let mut encoder = TansEncoder::from_decode_table(table);

    // Init with last symbol
    encoder.init_state(last_ml);
    let init_state = encoder.get_state();
    println!("  Init with symbol {}: state={}", last_ml, init_state);
    println!(
        "    State {} decodes to symbol {}",
        init_state,
        table.decode(init_state as usize).symbol
    );

    // Encode first symbol
    let (bits, nbits) = encoder.encode_symbol(first_ml);
    let final_state = encoder.get_state();
    println!(
        "  Encode symbol {}: bits={}, nbits={}",
        first_ml, bits, nbits
    );
    println!("    New state: {}", final_state);
    println!(
        "    State {} decodes to symbol {}",
        final_state,
        table.decode(final_state as usize).symbol
    );

    // Check if decoder can correctly use these
    println!("  Decoder would:");
    println!("    Start at state {}", final_state);
    println!(
        "    Read {} bits",
        table.decode(final_state as usize).num_bits
    );
    println!(
        "    Baseline = {}",
        table.decode(final_state as usize).baseline
    );

    // Simulate decoder
    let entry = table.decode(final_state as usize);
    let next_state = entry.baseline as u32 + bits;
    println!(
        "    New state after reading bits {} + {} = {}",
        entry.baseline, bits, next_state
    );
    if next_state < 64 {
        println!(
            "    State {} decodes to symbol {}",
            next_state,
            table.decode(next_state as usize).symbol
        );
    }
}

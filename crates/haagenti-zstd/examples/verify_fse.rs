//! Verify FSE state-to-symbol mapping

use haagenti_zstd::fse::{FseDecoder, FseTable};

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

fn main() {
    // Build predefined tables
    let ll_table =
        FseTable::from_predefined(&PREDEFINED_LL_DISTRIBUTION, PREDEFINED_LL_ACCURACY_LOG).unwrap();
    let of_table =
        FseTable::from_predefined(&PREDEFINED_OF_DISTRIBUTION, PREDEFINED_OF_ACCURACY_LOG).unwrap();
    let ml_table =
        FseTable::from_predefined(&PREDEFINED_ML_DISTRIBUTION, PREDEFINED_ML_ACCURACY_LOG).unwrap();

    println!("=== State to Symbol Mapping ===\n");

    // Check specific states from our failing test
    println!("Our final states: LL=4, OF=23, ML=36");
    println!("Expected seq0: ll_code=4, of_code=1, ml_code=32\n");

    // Check what symbols these states map to
    let _ll_decoder = FseDecoder::new(&ll_table);
    let _of_decoder = FseDecoder::new(&of_table);
    let _ml_decoder = FseDecoder::new(&ml_table);

    // Check a range of states
    println!("LL table (state -> symbol):");
    for state in 0..10u32 {
        let entry = ll_table.decode(state as usize);
        println!(
            "  state {} -> symbol {}, num_bits={}",
            state, entry.symbol, entry.num_bits
        );
    }

    println!("\nOF table (state -> symbol):");
    for state in 0..10u32 {
        let entry = of_table.decode(state as usize);
        println!(
            "  state {} -> symbol {}, num_bits={}",
            state, entry.symbol, entry.num_bits
        );
    }
    // Also check state 23
    let entry = of_table.decode(23);
    println!(
        "  state 23 -> symbol {}, num_bits={}",
        entry.symbol, entry.num_bits
    );

    println!("\nML table (state -> symbol):");
    for state in 30..40u32 {
        let entry = ml_table.decode(state as usize);
        println!(
            "  state {} -> symbol {}, num_bits={}",
            state, entry.symbol, entry.num_bits
        );
    }

    // Verify our specific case
    println!("\n=== Verification for our test case ===");
    let ll_entry = ll_table.decode(4);
    let of_entry = of_table.decode(23);
    let ml_entry = ml_table.decode(36);

    println!("LL state 4 -> symbol {} (expected 4)", ll_entry.symbol);
    println!("OF state 23 -> symbol {} (expected 1)", of_entry.symbol);
    println!("ML state 36 -> symbol {} (expected 32)", ml_entry.symbol);

    if ll_entry.symbol == 4 && of_entry.symbol == 1 && ml_entry.symbol == 32 {
        println!("\n✓ States correctly map to expected symbols!");
    } else {
        println!("\n✗ State-to-symbol mismatch detected!");
    }
}

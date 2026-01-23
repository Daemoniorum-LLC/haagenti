//! Check what OF states our encoder produces.

use haagenti_zstd::fse::{FseTable, TansEncoder, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION};

fn main() {
    println!("=== OF State Check ===\n");

    let of_table =
        FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();

    let mut encoder = TansEncoder::from_decode_table(&of_table);

    // Check what states decode to which symbols
    println!("OF decode table (first 28 states):");
    for state in 0..28 {
        let entry = of_table.decode(state);
        println!("  State {:2} → symbol {:2}", state, entry.symbol);
    }

    // Check init_state for OF codes 0-5
    println!("\nOF init_state results:");
    for code in 0..6 {
        encoder.init_state(code);
        let state = encoder.get_state();
        let entry = of_table.decode(state as usize);
        println!(
            "  Code {} → state {} → decodes to symbol {}",
            code, state, entry.symbol
        );
    }

    // Reference uses state 14 for OF_code=2
    // What state do we produce for OF_code=2?
    encoder.init_state(2);
    let our_state = encoder.get_state();
    println!("\nOur OF state for code 2: {}", our_state);
    println!("Reference uses: 14");
    println!("Match: {}", our_state == 14);
}

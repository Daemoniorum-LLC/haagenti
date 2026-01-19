//! Trace ML table state to symbol mapping

use haagenti_zstd::fse::{
    FseTable,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
};

fn main() {
    println!("=== ML Predefined Table Analysis ===\n");
    
    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    ).unwrap();
    
    println!("Table size: {} (accuracy log: {})", 1 << MATCH_LENGTH_ACCURACY_LOG, MATCH_LENGTH_ACCURACY_LOG);
    println!("\nState -> Symbol mapping:");
    
    // Find states that decode to each symbol
    let mut symbol_states: Vec<Vec<usize>> = vec![vec![]; 53];
    
    for state in 0..(1 << MATCH_LENGTH_ACCURACY_LOG) {
        let entry = ml_table.decode(state);
        let sym = entry.symbol as usize;
        if sym < symbol_states.len() {
            symbol_states[sym].push(state);
        }
        if state < 48 {
            println!("  State {:2} → Symbol {:2}", state, entry.symbol);
        }
    }
    
    println!("\nSymbol -> States mapping (for ML codes 34-52):");
    for sym in 34..=52 {
        if sym < symbol_states.len() {
            println!("  ML code {:2} → states {:?}", sym, symbol_states[sym]);
        }
    }
    
    // For bitstream [0xed, 0xab, 0x8e, 0x08], what ML state should we read?
    println!("\n=== Bitstream Analysis ===");
    let bits = 0x088eabed_u32;
    println!("Bitstream as u32 LE: 0x{:08x}", bits);
    println!("Binary: {:032b}", bits);
    
    let sentinel = 31 - bits.leading_zeros() as u32;
    println!("Sentinel at bit: {}", sentinel);
    
    // States read from MSB (after sentinel)
    let ll_state = (bits >> 21) & 0x3F;
    let of_state = (bits >> 16) & 0x1F;
    let ml_state = (bits >> 10) & 0x3F;
    
    println!("\nExtracted states (assuming sentinel at 27, standard order LL/OF/ML):");
    println!("  LL state (bits 26-21): {}", ll_state);
    println!("  OF state (bits 20-16): {}", of_state);
    println!("  ML state (bits 15-10): {}", ml_state);
    
    // What symbols do these map to?
    let ll_sym = haagenti_zstd::fse::FseTable::from_predefined(
        &haagenti_zstd::fse::LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        haagenti_zstd::fse::LITERAL_LENGTH_ACCURACY_LOG,
    ).unwrap().decode(ll_state as usize).symbol;
    
    let of_sym = haagenti_zstd::fse::FseTable::from_predefined(
        &haagenti_zstd::fse::OFFSET_DEFAULT_DISTRIBUTION,
        haagenti_zstd::fse::OFFSET_ACCURACY_LOG,
    ).unwrap().decode(of_state as usize).symbol;
    
    let ml_sym = ml_table.decode(ml_state as usize).symbol;
    
    println!("\nSymbols from states:");
    println!("  LL symbol: {} (want 4 for literal_length=4)", ll_sym);
    println!("  OF symbol: {} (want 2 for offset=4)", of_sym);
    println!("  ML symbol: {} (want 46 for match_length=496)", ml_sym);
    
    // For ML code 46: match_length = 259 + (8-bit extra)
    // So we need: 259 + extra = 496 → extra = 237 = 0xED
    println!("\nFor ML code 46 (baseline 259, 8 extra bits):");
    println!("  Need match_length 496");
    println!("  Extra bits needed: 496 - 259 = 237 = 0x{:02x}", 496 - 259);
}

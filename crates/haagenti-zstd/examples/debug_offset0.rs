//! Debug offset code 0 encoding

use haagenti_zstd::fse::{
    cached_of_table, cloned_of_encoder, FseDecoder, FseBitWriter, BitReader,
    OFFSET_ACCURACY_LOG,
};

fn main() {
    let of_table = cached_of_table();
    
    // Print all states that decode to each symbol
    println!("=== Offset table states by symbol ===");
    for sym in 0..8 {
        print!("Symbol {:2}: states [", sym);
        for state in 0..32 {
            if of_table.decode(state).symbol == sym {
                print!("{} ", state);
            }
        }
        println!("]");
    }
    
    // Create encoder and test offset code 0
    let mut of_encoder = cloned_of_encoder();
    
    println!("\n=== Testing offset code 0 encoding ===");
    
    // Test init with code 0
    of_encoder.init_state(0);
    let state_for_0 = of_encoder.get_state();
    println!("init_state(0) -> state={}", state_for_0);
    println!("  Decodes to symbol: {}", of_table.decode(state_for_0 as usize).symbol);
    
    // Now encode symbol 1 (transition from state for 0)
    let (bits, num_bits) = of_encoder.encode_symbol(1);
    let state_after_1 = of_encoder.get_state();
    println!("\nencode_symbol(1) from state {}:", state_for_0);
    println!("  bits={}, num_bits={}", bits, num_bits);
    println!("  new state={}", state_after_1);
    println!("  New state decodes to symbol: {}", of_table.decode(state_after_1 as usize).symbol);
    
    // Test the specific failing case: seq[1] has OF=1 (code=0), seq[0] has OF=12 (code=3)
    println!("\n=== Testing failing case: codes [3, 0] ===");
    of_encoder.init_state(0);  // Init with seq[1] code (0)
    let init_state = of_encoder.get_state();
    println!("init_state(0) -> state={}", init_state);
    
    // Encode seq[0] code (3)
    let (bits, num_bits) = of_encoder.encode_symbol(3);
    let final_state = of_encoder.get_state();
    println!("encode_symbol(3) from state {}:", init_state);
    println!("  bits={}, num_bits={}", bits, num_bits);
    println!("  final state={}", final_state);
    println!("  Final state decodes to symbol: {}", of_table.decode(final_state as usize).symbol);
    
    // Build a minimal bitstream and decode
    println!("\n=== Build and decode bitstream ===");
    let mut writer = FseBitWriter::new();
    
    // For 2 sequences with codes [3, 0]:
    // 1. Init with code 0
    // 2. Encode code 3, producing FSE bits
    // 3. Write seq[0] extras (3 bits for code 3)
    // 4. Write seq[0] FSE bits
    // 5. Write seq[1] extras (0 bits for code 0)
    // 6. Write final state
    
    of_encoder.init_state(0);
    let (fse_bits, fse_num_bits) = of_encoder.encode_symbol(3);
    let final_state = of_encoder.get_state();
    
    // seq[0] has offset_value=12, code=3, extra_bits=4, extra=4 (12 = 8 + 4)
    let of_extra_0: u32 = 4;
    let of_extra_bits_0: u8 = 3;
    
    // seq[1] has offset_value=1, code=0, extra_bits=0
    let of_extra_1: u32 = 0;
    let of_extra_bits_1: u8 = 0;
    
    // Write in correct order (forward for LSB reading)
    // seq[0]: extras, then FSE bits
    if of_extra_bits_0 > 0 {
        writer.write_bits(of_extra_0, of_extra_bits_0);
    }
    writer.write_bits(fse_bits, fse_num_bits);
    
    // seq[1]: just extras (no FSE for last seq)
    if of_extra_bits_1 > 0 {
        writer.write_bits(of_extra_1, of_extra_bits_1);
    }
    
    // Write final state
    writer.write_bits(final_state, OFFSET_ACCURACY_LOG);
    
    let bitstream = writer.finish();
    println!("Bitstream: {:02x?}", bitstream);
    
    // Decode
    let mut reader = BitReader::new(&bitstream);
    reader.init_from_end().unwrap();
    
    let mut decoder = FseDecoder::new(of_table);
    decoder.init_state(&mut reader).unwrap();
    let initial_state = decoder.state();
    let symbol_0 = decoder.peek_symbol();
    println!("\nDecoded initial state: {}, symbol: {}", initial_state, symbol_0);
    
    reader.switch_to_lsb_mode().unwrap();
    
    // Read seq[0] extras
    let decoded_extra_0 = if of_extra_bits_0 > 0 {
        reader.read_bits(of_extra_bits_0 as usize).unwrap()
    } else { 0 };
    println!("Decoded seq[0] extra: {} (expected {})", decoded_extra_0, of_extra_0);
    
    // Read seq[0] FSE bits and update state  
    decoder.update_state(&mut reader).unwrap();
    let symbol_1 = decoder.peek_symbol();
    println!("After update, symbol: {} (expected 0)", symbol_1);
    
    // Verify symbols
    println!("\n=== Verification ===");
    println!("seq[0] offset code: expected 3, got {}", symbol_0);
    println!("seq[1] offset code: expected 0, got {}", symbol_1);
    if symbol_0 == 3 && symbol_1 == 0 {
        println!("SUCCESS!");
    } else {
        println!("MISMATCH!");
    }
}

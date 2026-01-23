//! Trace the decoding of reference 500-byte compressed output

fn main() {
    let input: Vec<u8> = b"ABCD".iter().cycle().take(500).copied().collect();

    // Reference compressed output
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("Reference compressed: {:02x?}", ref_compressed);

    // The FSE bitstream is at the end: [ed, ab, 8e, 08]
    let fse_bytes = &ref_compressed[16..];
    println!("FSE bitstream: {:02x?}", fse_bytes);

    // Parse as little-endian u32
    let bits_u32 = u32::from_le_bytes([fse_bytes[0], fse_bytes[1], fse_bytes[2], fse_bytes[3]]);
    println!("\nBits as u32: 0x{:08x} = {:032b}", bits_u32, bits_u32);

    // Find sentinel (MSB 1-bit)
    let sentinel_pos = 31 - bits_u32.leading_zeros();
    println!("Sentinel position: bit {}", sentinel_pos);

    // After sentinel, the initial states are read MSB-first
    // For predefined tables: LL=6 bits, OF=5 bits, ML=6 bits
    // Total = 17 bits for states

    // Bits 27..21 (6 bits) = LL initial state
    // Bits 20..16 (5 bits) = OF initial state
    // Bits 15..10 (6 bits) = ML initial state

    // Actually let's work backward from the sentinel
    // If sentinel is at bit 27, then:
    // - LL state: bits 26..21 (6 bits) - MSB first after sentinel
    // - OF state: bits 20..16 (5 bits)
    // - ML state: bits 15..10 (6 bits)

    println!("\n=== Initial State Reading ===");

    // The bitstream after sentinel (read MSB first):
    let after_sentinel = bits_u32 & ((1 << sentinel_pos) - 1);
    println!(
        "After sentinel: {:0width$b}",
        after_sentinel,
        width = sentinel_pos as usize
    );

    // LL state (6 bits) - highest bits after sentinel
    let ll_bits = 6u32;
    let ll_state = (after_sentinel >> (sentinel_pos - ll_bits)) & ((1 << ll_bits) - 1);
    println!(
        "LL initial state: {} (bits {}..{})",
        ll_state,
        sentinel_pos - 1,
        sentinel_pos - ll_bits
    );

    // OF state (5 bits) - next
    let of_bits = 5u32;
    let of_state = (after_sentinel >> (sentinel_pos - ll_bits - of_bits)) & ((1 << of_bits) - 1);
    println!(
        "OF initial state: {} (bits {}..{})",
        of_state,
        sentinel_pos - ll_bits - 1,
        sentinel_pos - ll_bits - of_bits
    );

    // ML state (6 bits) - next
    let ml_bits = 6u32;
    let ml_state =
        (after_sentinel >> (sentinel_pos - ll_bits - of_bits - ml_bits)) & ((1 << ml_bits) - 1);
    println!(
        "ML initial state: {} (bits {}..{})",
        ml_state,
        sentinel_pos - ll_bits - of_bits - 1,
        sentinel_pos - ll_bits - of_bits - ml_bits
    );

    // What ML code is at state 42?
    println!("\n=== ML State Analysis ===");

    // Now check what our ML table says for state 42
    let ml_table = haagenti_zstd::fse::FseTable::from_hardcoded_ml().unwrap();

    for state in [ml_state, 42, 43, 44] {
        let entry = ml_table.decode(state as usize);
        println!(
            "ML State {} -> symbol={}, seq_base={}, seq_extra_bits={}",
            state, entry.symbol, entry.seq_base, entry.seq_extra_bits
        );
    }

    // Expected: For 500 bytes with 4 literals + 496 match:
    // - ML code 44 (baseline 259, 8 extra bits)
    // - extra = 496 - 259 = 237
    println!("\n=== Expected Values ===");
    println!("For match_length=496: need code 44 (baseline 259 + 8 extra bits)");
    println!("Extra value: 496 - 259 = {}", 496 - 259);

    // What's at the LSB side (extra bits)?
    println!("\n=== Extra Bits Analysis ===");
    let remaining_bits = sentinel_pos - ll_bits - of_bits - ml_bits;
    println!(
        "Remaining bits after states: {} (bits 0..{})",
        remaining_bits, remaining_bits
    );

    let extra_region = after_sentinel & ((1 << remaining_bits) - 1);
    println!(
        "Extra region value: {} = {:0width$b}",
        extra_region,
        extra_region,
        width = remaining_bits as usize
    );

    // The extra bits are: LL extra, ML extra, OF extra (in that order from LSB)
    // For this sequence: LL=4 (code 1, 0 extra bits), OF=2 (repeat, 0 extra bits), ML=496 (8 extra bits)
    println!("\nIf ML has 8 extra bits: extra = {}", extra_region & 0xFF);
}

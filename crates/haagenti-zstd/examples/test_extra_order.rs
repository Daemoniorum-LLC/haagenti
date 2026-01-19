//! Test if reversing the extra bit read order fixes decoding.

fn main() {
    // Reference bitstream [fd, e4, 88] for "ABCD" x 25
    let bitstream: &[u8] = &[0xfd, 0xe4, 0x88];

    // Load into u64 (little-endian)
    let mut container: u64 = 0;
    for (i, &byte) in bitstream.iter().enumerate() {
        container |= (byte as u64) << (i * 8);
    }
    println!("Container: 0x{:x} = {:024b}", container, container);

    // Find sentinel
    let sentinel_pos = 63 - container.leading_zeros() as usize;
    println!("Sentinel at bit {}", sentinel_pos);

    let total_bits = sentinel_pos; // 23 bits
    println!("Total data bits: {}", total_bits);

    // Read initial states from MSB end
    // LL (6 bits): bits 22-17
    let ll_state = ((container >> 17) & 0x3F) as usize;
    // OF (5 bits): bits 16-12
    let of_state = ((container >> 12) & 0x1F) as usize;
    // ML (6 bits): bits 11-6
    let ml_state = ((container >> 6) & 0x3F) as usize;

    println!("Initial states: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    // Remaining 6 bits for extras: bits 0-5
    let remaining = (container & 0x3F) as u32;
    println!("Remaining 6 bits: {:06b} = {}", remaining, remaining);

    // OF code 2 needs 2 extra bits, ML code 41 needs 4 extra bits

    println!("\n--- Reading OF first, then ML (current order) ---");
    let of_extra_current = remaining & 0x3;  // bits 0-1
    let ml_extra_current = (remaining >> 2) & 0xF;  // bits 2-5
    let offset_value_current = (1u32 << 2) + of_extra_current;
    let ml_value_current = 83 + ml_extra_current;
    println!("OF extra: {}, ML extra: {}", of_extra_current, ml_extra_current);
    println!("offset_value = {}, actual_offset = {}", offset_value_current, offset_value_current - 3);
    println!("ML = {}", ml_value_current);
    println!("Total bytes: 4 + {} = {}", ml_value_current, 4 + ml_value_current);

    println!("\n--- Reading ML first, then OF (reversed order) ---");
    let ml_extra_reversed = remaining & 0xF;  // bits 0-3
    let of_extra_reversed = (remaining >> 4) & 0x3;  // bits 4-5
    let offset_value_reversed = (1u32 << 2) + of_extra_reversed;
    let ml_value_reversed = 83 + ml_extra_reversed;
    println!("ML extra: {}, OF extra: {}", ml_extra_reversed, of_extra_reversed);
    println!("offset_value = {}, actual_offset = {}", offset_value_reversed, offset_value_reversed - 3);
    println!("ML = {}", ml_value_reversed);
    println!("Total bytes: 4 + {} = {}", ml_value_reversed, 4 + ml_value_reversed);
}

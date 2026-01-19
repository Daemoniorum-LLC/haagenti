//! Trace what offset values we're actually encoding.

fn main() {
    let input = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD";
    println!("Input: \"ABCD\" x 25 ({} bytes)", input.len());

    // Compress and extract the sequences
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let compressed = compressor.compress(input).unwrap();

    println!("\nCompressed: {} bytes", compressed.len());
    println!("Hex: {:02x?}", &compressed[..]);

    // Parse to get the sequence section
    let fhd = compressed[4];
    let single_segment = (fhd & 0x20) != 0;
    let fcs_size = match fhd >> 6 {
        0 => if single_segment { 1 } else { 0 },
        1 => 2, 2 => 4, 3 => 8, _ => 0,
    };
    let mut pos = 5;
    if !single_segment { pos += 1; } // window desc
    pos += fcs_size; // skip FCS

    // Block header
    let bh = u32::from_le_bytes([compressed[pos], compressed[pos+1], compressed[pos+2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        println!("Not a compressed block");
        return;
    }

    let block_data = &compressed[pos..pos+block_size];
    println!("Block data: {:02x?}", block_data);

    // Parse literals
    let lit_type = block_data[0] & 0x03;
    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => ((block_data[0] >> 3) as usize, 1),
            2 => (((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF, 2),
            _ => (0, 1),
        }
    } else { (0, 1) };

    let seq_section = &block_data[lit_header_size + lit_size..];
    println!("Literals: {} bytes", lit_size);
    println!("Sequence section: {:02x?}", seq_section);

    // What sequence should we encode?
    println!("\nExpected for ABCD pattern:");
    println!("  Position 0-3: 4 literals (ABCD)");
    println!("  Position 4-99: copy from position 0 (offset=4, length=96)");
    println!("  offset_value = 4 + 3 = 7");
    println!("  7 = (1 << 2) + 3 -> OF_code=2, OF_extra=3");

    // Encode what we SHOULD encode
    println!("\n=== What encode_offset produces ===");
    let actual_offset = 4u32;
    let offset_value = actual_offset + 3; // For non-repeat offsets
    let of_code = 31 - offset_value.leading_zeros();
    let of_baseline = 1u32 << of_code;
    let of_extra = offset_value - of_baseline;
    println!("actual_offset={}, offset_value={}", actual_offset, offset_value);
    println!("OF_code={}, OF_extra={}", of_code, of_extra);
    println!("Verify: (1 << {}) + {} = {}", of_code, of_extra, (1u32 << of_code) + of_extra);
}

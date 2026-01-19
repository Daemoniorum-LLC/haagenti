//! Analyze literal header format according to RFC 8878

fn main() {
    println!("RFC 8878 Section 3.1.1.3.1 - Literals Section Header:");
    println!();
    println!("For Raw_Literals_Block (type=0):");
    println!("  Size_Format 0b00 or 0b01: 1-byte header, 5-bit size (0-31)");
    println!("  Size_Format 0b10: 2-byte header, 12-bit size (0-4095)");
    println!("  Size_Format 0b11: 3-byte header, 20-bit size");
    println!();

    // Analyze what decoder expects for 1-byte header
    println!("Decoder for 1-byte Raw header:");
    println!("  Block_Type = header & 0x03 (bits 1:0)");
    println!("  Size_Format = (header >> 2) & 0x03 (bits 3:2)");
    println!("  Size = header >> 3 (bits 7:3)");
    println!();

    // Current encoding formula analysis
    println!("Current formula: header = (size << 3) | 0");
    println!();
    println!("Size | Header | Bits 3:2 | Size_Format | Valid?");
    println!("-----|--------|----------|-------------|-------");
    for size in 0..=31 {
        let header = (size << 3) | 0;
        let size_format = (header >> 2) & 0x03;
        let valid = size_format == 0 || size_format == 1;
        println!(
            "{:4} | 0x{:02x}   | {:02b}       | {}           | {}",
            size,
            header,
            size_format,
            size_format,
            if valid { "Y" } else { "N" }
        );
    }

    println!();
    println!("The problem: bit 3 of header = bit 0 of (size << 3) = size & 1");
    println!("For odd sizes, bit 3 = 1, making Size_Format = 2 or 3!");
    println!();

    // Calculate correct 2-byte header format
    println!("2-byte header format (Size_Format = 2):");
    println!("  byte0 = (size & 0x0F) << 4 | (2 << 2) | 0 = (size & 0x0F) << 4 | 8");
    println!("  byte1 = (size >> 4) & 0xFF");
    println!();

    for size in [1, 5, 11, 20, 31, 100] {
        let byte0 = ((size & 0x0F) << 4) | 8;
        let byte1 = (size >> 4) & 0xFF;

        // Verify decode
        let decoded_type = byte0 & 0x03;
        let decoded_sf = (byte0 >> 2) & 0x03;
        let decoded_size = ((byte0 >> 4) as usize) | ((byte1 as usize) << 4);

        println!(
            "Size {}: byte0=0x{:02x}, byte1=0x{:02x} -> type={}, sf={}, size={}",
            size, byte0, byte1, decoded_type, decoded_sf, decoded_size
        );
    }
}

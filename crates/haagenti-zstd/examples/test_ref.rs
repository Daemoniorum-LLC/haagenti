//! Quick test to compare our output with reference

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    let input = b"ABCDABCDABCDABCD";
    println!(
        "Input: {:?} ({} bytes)",
        String::from_utf8_lossy(input),
        input.len()
    );

    // Our compression
    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(input).unwrap();

    // Reference compression (level 1)
    let reference = zstd::encode_all(std::io::Cursor::new(input), 1).unwrap();

    println!("\nOurs ({} bytes):", ours.len());
    for b in &ours {
        print!("{:02x} ", b);
    }
    println!();

    println!("\nReference ({} bytes):", reference.len());
    for b in &reference {
        print!("{:02x} ", b);
    }
    println!();

    // Verify both decode correctly with reference
    println!("\n=== Decode verification ===");

    match zstd::decode_all(std::io::Cursor::new(&ours)) {
        Ok(dec) => println!("Our output decoded by ref: OK ({} bytes)", dec.len()),
        Err(e) => println!("Our output decoded by ref: FAILED - {:?}", e),
    }

    match zstd::decode_all(std::io::Cursor::new(&reference)) {
        Ok(dec) => println!("Ref output decoded by ref: OK ({} bytes)", dec.len()),
        Err(e) => println!("Ref output decoded by ref: FAILED - {:?}", e),
    }

    // Parse frames
    println!("\n=== Frame header analysis ===");
    analyze("Ours", &ours);
    analyze("Reference", &reference);
}

fn analyze(label: &str, data: &[u8]) {
    println!("\n{}:", label);

    // Magic
    println!(
        "  Magic: {:02x} {:02x} {:02x} {:02x}",
        data[0], data[1], data[2], data[3]
    );

    // FHD
    let fhd = data[4];
    let single_segment = (fhd >> 5) & 1 != 0;
    let content_checksum = (fhd >> 2) & 1 != 0;
    let dict_id_flag = fhd & 3;
    let fcs_flag = (fhd >> 6) & 3;
    println!("  FHD: 0x{:02x}", fhd);
    println!("    Single_Segment: {}", single_segment);
    println!("    Content_Checksum: {}", content_checksum);
    println!("    Dictionary_ID_flag: {}", dict_id_flag);
    println!("    Frame_Content_Size_flag: {}", fcs_flag);

    let mut pos = 5;
    if !single_segment {
        println!("  Window_Descriptor: 0x{:02x}", data[pos]);
        pos += 1;
    }

    // FCS if present
    let fcs_size = match fcs_flag {
        0 => {
            if single_segment {
                1
            } else {
                0
            }
        }
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    if fcs_size > 0 {
        print!("  Frame_Content_Size: ");
        for i in 0..fcs_size {
            print!("{:02x} ", data[pos + i]);
        }
        println!();
        pos += fcs_size;
    }

    // Block header
    let bh = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
    let last_block = bh & 1;
    let block_type = (bh >> 1) & 3;
    let block_size = bh >> 3;
    println!(
        "  Block header: last={}, type={}, size={}",
        last_block, block_type, block_size
    );
    pos += 3;

    // Literals header
    let lit_byte = data[pos];
    let lit_type = lit_byte & 3;
    let size_format = (lit_byte >> 2) & 3;
    println!("  Literals: type={}, size_format={}", lit_type, size_format);
}

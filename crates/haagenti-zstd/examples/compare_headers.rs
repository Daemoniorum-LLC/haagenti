//! Compare our frame header with reference

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Create a repeating pattern
    let input = b"The quick brown fox jumps over the lazy dog. ";
    let mut large_input = Vec::new();
    for _ in 0..10 {
        large_input.extend_from_slice(input);
    }
    println!("Input: {} bytes", large_input.len());

    // Our compression
    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(&large_input).unwrap();

    // Reference compression (level 1)
    let reference = zstd::encode_all(std::io::Cursor::new(&large_input), 1).unwrap();

    println!("\n=== Our output ({} bytes) ===", ours.len());
    print!("First 20 bytes: ");
    for b in ours.iter().take(20) {
        print!("{:02x} ", b);
    }
    println!();

    println!("\n=== Reference output ({} bytes) ===", reference.len());
    print!("First 20 bytes: ");
    for b in reference.iter().take(20) {
        print!("{:02x} ", b);
    }
    println!();

    // Parse our frame header
    println!("\n=== Our frame header ===");
    parse_frame("Ours", &ours);

    println!("\n=== Reference frame header ===");
    parse_frame("Reference", &reference);
}

fn parse_frame(label: &str, data: &[u8]) {
    println!("{}:", label);

    // Magic
    let magic_ok = data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;
    println!("  Magic: {:02x} {:02x} {:02x} {:02x} ({})", data[0], data[1], data[2], data[3],
             if magic_ok { "OK" } else { "BAD" });

    // FHD
    let fhd = data[4];
    println!("  FHD: 0x{:02x} = {:08b}", fhd, fhd);

    let fcs_flag = (fhd >> 6) & 3;
    let single_segment = (fhd >> 5) & 1 != 0;
    let unused = (fhd >> 3) & 1;
    let content_checksum = (fhd >> 2) & 1 != 0;
    let dict_id_flag = fhd & 3;

    println!("    FCS_flag: {}", fcs_flag);
    println!("    Single_Segment: {}", single_segment);
    println!("    Unused_bit: {} {}", unused, if unused != 0 { "(SHOULD BE 0!)" } else { "" });
    println!("    Content_Checksum: {}", content_checksum);
    println!("    Dictionary_ID_flag: {}", dict_id_flag);

    let mut pos = 5;

    // Window descriptor (if !Single_Segment)
    if !single_segment {
        let wd = data[pos];
        let exp = wd >> 3;
        let mantissa = wd & 7;
        let window_base = 1u64 << (10 + exp);
        let window_size = window_base + (window_base >> 3) * mantissa as u64;
        println!("  Window_Descriptor: 0x{:02x} (exp={}, mantissa={}, window={})", wd, exp, mantissa, window_size);
        pos += 1;
    } else {
        println!("  Window_Descriptor: (none - single segment)");
    }

    // Dictionary ID (if dict_id_flag != 0)
    if dict_id_flag != 0 {
        let dict_bytes = match dict_id_flag {
            1 => 1,
            2 => 2,
            3 => 4,
            _ => 0,
        };
        println!("  Dictionary_ID: {} bytes at pos {}", dict_bytes, pos);
        pos += dict_bytes;
    }

    // Frame Content Size
    let fcs_size = match fcs_flag {
        0 => if single_segment { 1 } else { 0 },
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    if fcs_size > 0 {
        print!("  Frame_Content_Size ({} bytes): ", fcs_size);
        let fcs = match fcs_size {
            1 => data[pos] as u64,
            2 => u16::from_le_bytes([data[pos], data[pos+1]]) as u64 + 256,
            4 => u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as u64,
            8 => u64::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3],
                                     data[pos+4], data[pos+5], data[pos+6], data[pos+7]]),
            _ => 0,
        };
        println!("{} bytes", fcs);
        pos += fcs_size;
    }

    // Block header
    let bh = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], 0]);
    let last = bh & 1;
    let block_type = (bh >> 1) & 3;
    let block_size = bh >> 3;
    println!("  Block header at 0x{:02x}: {:02x} {:02x} {:02x}", pos, data[pos], data[pos+1], data[pos+2]);
    println!("    Last_Block: {}", last);
    println!("    Block_Type: {} ({})", block_type, ["Raw", "RLE", "Compressed", "Reserved"][block_type as usize]);
    println!("    Block_Size: {}", block_size);
    pos += 3;

    // First byte of block content (literals header)
    if block_type == 2 {
        let lit_byte = data[pos];
        let lit_type = lit_byte & 3;
        let size_format = (lit_byte >> 2) & 3;
        println!("  Literals header at 0x{:02x}: 0x{:02x}", pos, lit_byte);
        println!("    Type: {} ({})", lit_type, ["Raw", "RLE", "Compressed", "Treeless"][lit_type as usize]);
        println!("    Size_Format: {}", size_format);
    }
}

//! Trace the encoding step by step

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Simple repeating pattern
    let input = b"ABCDABCDABCDABCD";
    println!("Input: {:?} ({} bytes)", String::from_utf8_lossy(input), input.len());

    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(input).unwrap();

    println!("\nCompressed: {} bytes", compressed.len());
    println!("Full hex dump:");
    for (i, b) in compressed.iter().enumerate() {
        print!("{:02x} ", b);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();

    // Parse frame
    println!("\n=== Frame Parsing ===");
    println!("Magic: {:02x} {:02x} {:02x} {:02x}", compressed[0], compressed[1], compressed[2], compressed[3]);

    let fhd = compressed[4];
    println!("FHD: 0x{:02x}", fhd);
    let single_segment = (fhd >> 5) & 1 != 0;
    let fcs_flag = (fhd >> 6) & 3;
    println!("  single_segment: {}, fcs_flag: {}", single_segment, fcs_flag);

    let mut pos = 5;
    if !single_segment {
        println!("Window descriptor: 0x{:02x}", compressed[pos]);
        pos += 1;
    }

    let fcs_size = match fcs_flag {
        0 => if single_segment { 1 } else { 0 },
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    if fcs_size > 0 {
        println!("FCS: {} bytes", fcs_size);
    }
    pos += fcs_size;

    // Block header (3 bytes)
    let bh = u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
    let is_last = bh & 1;
    let block_type = (bh >> 1) & 3;
    let block_size = bh >> 3;
    pos += 3;
    println!("\nBlock header at pos {}: {:02x} {:02x} {:02x}", pos - 3, compressed[pos - 3], compressed[pos - 2], compressed[pos - 1]);
    println!("  is_last: {}, block_type: {}, block_size: {}", is_last, block_type, block_size);

    // Block content
    if block_type == 2 {
        println!("\nLiterals section at pos {}:", pos);
        let lit_byte0 = compressed[pos];
        let lit_type = lit_byte0 & 0x03;
        let size_format = (lit_byte0 >> 2) & 0x03;
        println!("  byte0: 0x{:02x}", lit_byte0);
        println!("  lit_type: {} (0=Raw)", lit_type);
        println!("  size_format: {}", size_format);

        match size_format {
            0 | 1 => {
                let size = (lit_byte0 >> 3) as usize;
                println!("  1-byte header, size: {}", size);
                println!("  Literal bytes start at: {}", pos + 1);
            }
            2 => {
                let lit_byte1 = compressed[pos + 1];
                let size = ((lit_byte0 >> 4) as usize) | ((lit_byte1 as usize) << 4);
                println!("  byte1: 0x{:02x}", lit_byte1);
                println!("  2-byte header, size: {}", size);
                println!("  Literal bytes start at: {}", pos + 2);
            }
            3 => {
                let lit_byte1 = compressed[pos + 1];
                let lit_byte2 = compressed[pos + 2];
                let size = ((lit_byte0 >> 4) as usize) | ((lit_byte1 as usize) << 4) | ((lit_byte2 as usize) << 12);
                println!("  byte1: 0x{:02x}, byte2: 0x{:02x}", lit_byte1, lit_byte2);
                println!("  3-byte header, size: {}", size);
            }
            _ => {}
        }
    }

    // Try reference decode
    println!("\n=== Reference decode ===");
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(dec) => {
            println!("SUCCESS! Decoded {} bytes", dec.len());
            if &dec == &input[..] {
                println!("Content matches input!");
            } else {
                println!("Content mismatch!");
                println!("Expected: {:?}", String::from_utf8_lossy(input));
                println!("Got: {:?}", String::from_utf8_lossy(&dec));
            }
        }
        Err(e) => println!("FAILED: {:?}", e),
    }
}

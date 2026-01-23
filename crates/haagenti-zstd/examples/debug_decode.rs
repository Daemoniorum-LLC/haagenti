//! Debug decoding to trace the issue

use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    let input = b"ABCDABCDABCDABCD";
    println!(
        "Input: {:?} ({} bytes)",
        String::from_utf8_lossy(input),
        input.len()
    );

    let compressor = ZstdCompressor::new();
    let compressed = compressor.compress(input).unwrap();

    println!("\nCompressed: {} bytes", compressed.len());
    print!("Hex: ");
    for b in &compressed {
        print!("{:02x} ", b);
    }
    println!();

    // Parse sequence section manually to understand the issue
    println!("\n=== Manual parsing ===");

    // Find sequence section
    let mut pos = 4; // skip magic
    let fhd = compressed[pos];
    pos += 1;
    let single_segment = (fhd >> 5) & 1 != 0;
    if !single_segment {
        pos += 1; // window descriptor
    }
    let fcs_flag = (fhd >> 6) & 3;
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
    pos += fcs_size;

    // Block header
    pos += 3; // skip block header

    // Literals
    let lit_byte0 = compressed[pos];
    let size_format = (lit_byte0 >> 2) & 0x03;
    let (header_size, lit_size) = match size_format {
        0 | 1 => (1, (lit_byte0 >> 3) as usize),
        2 => {
            let byte1 = compressed[pos + 1];
            (2, ((lit_byte0 >> 4) as usize) | ((byte1 as usize) << 4))
        }
        _ => (3, 0),
    };
    pos += header_size + lit_size;

    println!("Sequence section starts at pos {}", pos);
    let seq_section = &compressed[pos..];
    println!(
        "Sequence section ({} bytes): {:02x?}",
        seq_section.len(),
        seq_section
    );

    // Parse count
    let count = seq_section[0] as usize;
    println!("Sequence count: {}", count);

    // Parse mode
    let mode_byte = seq_section[1];
    println!("Mode byte: 0x{:02x} = 0b{:08b}", mode_byte, mode_byte);

    // Extract modes (RFC 8878)
    let ll_mode = mode_byte & 0x03;
    let of_mode = (mode_byte >> 2) & 0x03;
    let ml_mode = (mode_byte >> 4) & 0x03;

    let mode_name = |m| match m {
        0 => "Predefined",
        1 => "RLE",
        2 => "FSE",
        3 => "Repeat",
        _ => "Unknown",
    };

    println!("LL mode: {} ({})", ll_mode, mode_name(ll_mode));
    println!("OF mode: {} ({})", of_mode, mode_name(of_mode));
    println!("ML mode: {} ({})", ml_mode, mode_name(ml_mode));

    println!("All RLE? {}", ll_mode == 1 && of_mode == 1 && ml_mode == 1);

    // Try our decoder
    println!("\n=== Our decoder ===");
    let decompressor = ZstdDecompressor;
    match decompressor.decompress(&compressed) {
        Ok(decompressed) => {
            println!("SUCCESS: {} bytes", decompressed.len());
            if decompressed == input {
                println!("Content matches!");
            }
        }
        Err(e) => println!("FAILED: {:?}", e),
    }
}

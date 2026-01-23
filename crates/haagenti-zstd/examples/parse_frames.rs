//! Parse frame structures of working vs failing cases

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn parse_frame(name: &str, data: &[u8]) {
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();

    println!("\n{}", "=".repeat(70));
    println!("=== {} ===", name);
    println!(
        "Input: {:?} ({} bytes)",
        String::from_utf8_lossy(data),
        data.len()
    );

    let result = match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(d) if d == data => "OK",
        Ok(_) => "MISMATCH",
        Err(_) => "FAILED",
    };
    println!("Decode result: {}", result);
    println!("\nCompressed hex ({} bytes):", compressed.len());
    for (i, chunk) in compressed.chunks(16).enumerate() {
        print!("  {:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        // ASCII
        print!(" |");
        for b in chunk {
            let c = if *b >= 32 && *b < 127 {
                *b as char
            } else {
                '.'
            };
            print!("{}", c);
        }
        println!("|");
    }

    // Parse frame
    if compressed.len() < 9 {
        println!("Too short to parse");
        return;
    }

    let magic = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    println!("\nFrame structure:");
    println!(
        "  Magic: 0x{:08x} ({})",
        magic,
        if magic == 0xFD2FB528 {
            "valid"
        } else {
            "INVALID"
        }
    );

    let fhd = compressed[4];
    let fcs_flag = (fhd >> 6) & 3;
    let single_seg = (fhd >> 5) & 1;
    println!("  FHD: 0x{:02x}", fhd);
    println!("    FCS_flag={}, Single_Segment={}", fcs_flag, single_seg);

    let mut pos = 5;
    if single_seg == 0 {
        println!("  Window_Desc: 0x{:02x}", compressed[pos]);
        pos += 1;
    }

    // Frame content size
    let fcs_size = match (fcs_flag, single_seg) {
        (0, 1) => 1,
        (1, _) => 2,
        (2, _) => 4,
        (3, _) => 8,
        _ => 0,
    };
    if fcs_size > 0 {
        print!("  FCS ({} bytes): ", fcs_size);
        for i in 0..fcs_size {
            if pos + i < compressed.len() {
                print!("{:02x} ", compressed[pos + i]);
            }
        }
        println!();
        pos += fcs_size;
    }

    // Block
    if pos + 3 <= compressed.len() {
        let bh = u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
        let last = bh & 1;
        let btype = (bh >> 1) & 3;
        let bsize = bh >> 3;
        println!("  Block @ offset {}: header=0x{:06x}", pos, bh & 0xFFFFFF);
        println!(
            "    Last={}, Type={} ({}), Size={}",
            last,
            btype,
            match btype {
                0 => "Raw",
                1 => "RLE",
                2 => "Compressed",
                _ => "Reserved",
            },
            bsize
        );

        pos += 3;
        let block_end = pos + bsize as usize;

        if btype == 2 && block_end <= compressed.len() {
            // Parse literals
            let lh = compressed[pos];
            let ltype = lh & 3;
            let sfmt = (lh >> 2) & 3;
            println!("    Literals @ {}: header=0x{:02x}", pos, lh);
            println!(
                "      Type={} ({}), Size_Format={}",
                ltype,
                match ltype {
                    0 => "Raw",
                    1 => "RLE",
                    2 => "Compressed",
                    _ => "Treeless",
                },
                sfmt
            );

            // Calculate literal size based on format
            let (lit_hdr_size, lit_size) = if ltype <= 1 {
                match sfmt {
                    0 | 2 => (1, (lh >> 3) as usize),
                    1 => {
                        let b1 = compressed[pos + 1];
                        let size = ((lh as usize >> 4) & 0xF) | ((b1 as usize) << 4);
                        (2, size)
                    }
                    _ => {
                        let b1 = compressed[pos + 1];
                        let b2 = compressed[pos + 2];
                        let size = ((lh as usize >> 4) & 0xF)
                            | ((b1 as usize) << 4)
                            | ((b2 as usize) << 12);
                        (3, size)
                    }
                }
            } else {
                // Compressed or Treeless - more complex
                (1, 0) // placeholder
            };
            println!(
                "      Header size: {}, Literal size: {}",
                lit_hdr_size, lit_size
            );

            let seq_start = pos + lit_hdr_size + if ltype == 1 { 1 } else { lit_size };
            println!("    Sequences @ {}", seq_start);
            if seq_start < compressed.len() {
                let count = compressed[seq_start];
                println!("      Count: {}", count);
                if seq_start + 1 < compressed.len() {
                    let mode = compressed[seq_start + 1];
                    println!("      Mode: 0x{:02x}", mode);
                    println!(
                        "        LL_Mode={}, OF_Mode={}, ML_Mode={}",
                        mode & 3,
                        (mode >> 2) & 3,
                        (mode >> 4) & 3
                    );
                }
            }
        }
    }
}

fn main() {
    // Working case
    parse_frame("WORKING", b"abcdabcdXYZabcd");

    // Failing case
    parse_frame("FAILING", b"abcdefghXabcdefghYabcd");
}

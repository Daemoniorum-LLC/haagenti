//! Debug exactly 2 log lines

use haagenti_core::{CompressionLevel, Compressor};
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // 2 log lines - this is the minimal failing case
    let data = b"[2024-01-01 10:00:00] INFO Request #0\n[2024-01-02 10:01:00] INFO Request #1000\n";
    println!("Input: {} bytes", data.len());
    println!("Data: {:?}", String::from_utf8_lossy(data));

    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(data).unwrap();

    println!("\nCompressed: {} bytes", compressed.len());
    println!("Hex: {:02x?}", &compressed);

    // Try to decode
    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("\nDecode: OK");
            } else {
                println!("\nDecode: CONTENT MISMATCH");
                println!("Expected {} bytes, got {} bytes", data.len(), dec.len());
            }
        }
        Err(e) => {
            println!("\nDecode: FAILED - {:?}", e);

            // Parse the frame manually to understand the issue
            println!("\n=== Frame Analysis ===");
            if compressed.len() < 4 {
                println!("Too short for magic");
                return;
            }

            let magic =
                u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
            println!("Magic: 0x{:08x} (expected 0xFD2FB528)", magic);

            if compressed.len() < 5 {
                println!("Too short for FHD");
                return;
            }

            let fhd = compressed[4];
            let fcs_size = match (fhd >> 6) & 3 {
                0 if (fhd & 0x20) != 0 => 1,
                0 => 0,
                1 => 2,
                2 => 4,
                3 => 8,
                _ => 0,
            };
            let single_segment = (fhd & 0x20) != 0;
            let content_checksum = (fhd & 0x04) != 0;
            let window_present = !single_segment;

            println!("FHD: 0x{:02x}", fhd);
            println!("  Single segment: {}", single_segment);
            println!("  Content checksum: {}", content_checksum);
            println!("  Window present: {}", window_present);
            println!("  FCS size: {}", fcs_size);

            let mut pos = 5;
            if window_present {
                println!("Window descriptor: 0x{:02x}", compressed[pos]);
                pos += 1;
            }
            pos += fcs_size;

            // Block header
            if pos + 3 > compressed.len() {
                println!("Too short for block header");
                return;
            }

            let bh =
                u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
            let last_block = (bh & 1) != 0;
            let block_type = (bh >> 1) & 3;
            let block_size = (bh >> 3) as usize;

            println!("\nBlock header at offset {}: 0x{:06x}", pos, bh);
            println!("  Last block: {}", last_block);
            println!(
                "  Block type: {} ({})",
                block_type,
                match block_type {
                    0 => "Raw",
                    1 => "RLE",
                    2 => "Compressed",
                    _ => "Reserved",
                }
            );
            println!("  Block size: {}", block_size);

            pos += 3;
            let block_start = pos;

            // Literals section
            if pos >= compressed.len() {
                println!("No literals");
                return;
            }

            let lit_byte0 = compressed[pos];
            let lit_type = lit_byte0 & 3;
            println!("\nLiterals at offset {}:", pos);
            println!(
                "  Type: {} ({})",
                lit_type,
                match lit_type {
                    0 => "Raw",
                    1 => "RLE",
                    2 => "Compressed",
                    3 => "Treeless",
                    _ => "?",
                }
            );
        }
    }

    // Also test 1 line (which should work)
    println!("\n=== Test 1 line (should work) ===");
    let data1 = b"[2024-01-01 10:00:00] INFO Request #0\n";
    let compressed1 = compressor.compress(data1).unwrap();
    match zstd::decode_all(Cursor::new(&compressed1)) {
        Ok(dec) if dec.as_slice() == data1 => println!("1 line: OK"),
        _ => println!("1 line: FAILED"),
    }
}

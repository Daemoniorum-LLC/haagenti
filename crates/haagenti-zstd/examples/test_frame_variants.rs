//! Test different frame header configurations

use std::io::Cursor;

fn main() {
    let expected = b"abcdefghXabcdefghYabcd";
    let literals = b"abcdefghXY";
    let bitstream = [0x14, 0x01, 0x10, 0xe5, 0x08];

    println!("=== Testing Frame Header Variants ===\n");

    // Variant 1: No FCS, no single segment (current)
    test_frame("No FCS, Multi-segment", 0x00, None, literals, &bitstream, expected);

    // Variant 2: With FCS (1 byte), single segment
    test_frame("FCS=22, Single-segment", 0x20, Some(expected.len() as u64), literals, &bitstream, expected);

    // Variant 3: With FCS (1 byte), multi-segment
    test_frame("FCS=22, Multi-segment", 0x00 | 0x01, Some(expected.len() as u64), literals, &bitstream, expected);

    // Variant 4: Try with checksum flag (even though we don't have a checksum)
    // test_frame("With checksum flag", 0x04, None, literals, &bitstream, expected);

    // Let me also test if the issue is with block boundaries
    println!("\n=== Testing raw block with same content ===");
    test_raw_frame("Raw block", 0x20, Some(expected.len() as u64), expected);
}

fn test_frame(name: &str, fhd: u8, fcs: Option<u64>, literals: &[u8], bitstream: &[u8], expected: &[u8]) {
    println!("Testing: {}", name);

    let mut frame = Vec::new();

    // Magic
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // FHD
    frame.push(fhd);

    // Window descriptor (only if not single segment)
    let single_segment = (fhd & 0x20) != 0;
    if !single_segment {
        frame.push(0x48);
    }

    // FCS (if specified)
    if let Some(size) = fcs {
        let fcs_flag = fhd & 0x03;
        match fcs_flag {
            0 if single_segment => {
                // Single segment with FCS_Field_Size=0 means 1-byte FCS
                frame.push(size as u8);
            }
            1 => {
                // 2-byte FCS
                frame.extend_from_slice(&(size as u16).to_le_bytes());
            }
            2 => {
                // 4-byte FCS
                frame.extend_from_slice(&(size as u32).to_le_bytes());
            }
            3 => {
                // 8-byte FCS
                frame.extend_from_slice(&size.to_le_bytes());
            }
            _ => {}
        }
    }

    // Block content
    let mut block = Vec::new();

    // Literals header (Raw, 10 bytes)
    block.push(0x50);
    block.extend_from_slice(literals);

    // Sequence count
    block.push(0x02);

    // Mode byte
    block.push(0x00);

    // FSE bitstream
    block.extend_from_slice(bitstream);

    // Block header
    let block_size = block.len() as u32;
    let block_header = (block_size << 3) | 0x05; // type=2, last=1
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    frame.extend_from_slice(&block);

    println!("  Frame: {:02x?}", &frame[..20.min(frame.len())]);
    println!("  FHD byte: 0x{:02x}", fhd);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == expected => println!("  Result: OK!"),
        Ok(decoded) => {
            println!("  Result: MISMATCH");
            println!("    Got {} bytes: {:?}", decoded.len(), String::from_utf8_lossy(&decoded));
        }
        Err(e) => println!("  Result: FAILED - {:?}", e),
    }
    println!();
}

fn test_raw_frame(name: &str, fhd: u8, fcs: Option<u64>, data: &[u8]) {
    println!("Testing: {}", name);

    let mut frame = Vec::new();

    // Magic
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // FHD
    frame.push(fhd);

    // Window descriptor (only if not single segment)
    let single_segment = (fhd & 0x20) != 0;
    if !single_segment {
        frame.push(0x48);
    }

    // FCS (if specified)
    if let Some(size) = fcs {
        if single_segment {
            // Single segment with FCS_Field_Size=0 means 1-byte FCS
            frame.push(size as u8);
        }
    }

    // Block header: type=0 (raw), last=1, size=data.len()
    let block_header = ((data.len() as u32) << 3) | 0x01;
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    // Raw data
    frame.extend_from_slice(data);

    println!("  Frame: {:02x?}", &frame[..20.min(frame.len())]);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == data => println!("  Result: OK!"),
        Ok(decoded) => println!("  Result: MISMATCH - got {} bytes", decoded.len()),
        Err(e) => println!("  Result: FAILED - {:?}", e),
    }
}

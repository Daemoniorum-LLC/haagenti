//! Debug decode failures with more detail.

use std::io::{Cursor, Read};

fn main() {
    let input = b"abcdefghXabcdefghYabcd";
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    println!("Input: {:?}", std::str::from_utf8(input).unwrap());
    println!(
        "Compressed ({} bytes): {:02x?}",
        our_compressed.len(),
        our_compressed
    );

    // Try to decode step by step
    println!("\n=== Attempting decode with zstd-safe ===");

    // Use the lower-level zstd interface
    match decode_with_debug(&our_compressed, input.len()) {
        Ok(decoded) => {
            println!(
                "Decoded {} bytes: {:?}",
                decoded.len(),
                String::from_utf8_lossy(&decoded)
            );
            if decoded == input {
                println!("SUCCESS: Output matches input!");
            } else {
                println!("MISMATCH: Output differs from input");
            }
        }
        Err(e) => {
            println!("FAILED: {}", e);
        }
    }

    // Also test byte-by-byte modifications to find sensitivity
    println!("\n=== Testing byte modifications ===");
    for pos in 0..our_compressed.len().min(15) {
        let mut modified = our_compressed.clone();
        // Try flipping each bit in this byte
        let original = modified[pos];
        let mut works_with_any = false;
        for bit in 0..8 {
            modified[pos] = original ^ (1 << bit);
            if zstd::decode_all(&modified[..]).is_ok() {
                works_with_any = true;
                println!("  Byte {} bit {}: flipping makes it decodable!", pos, bit);
            }
        }
        if !works_with_any && pos > 5 {
            // Only report for data bytes, not frame header
            // println!("  Byte {} (0x{:02x}): no single bit flip makes it work", pos, original);
        }
    }

    // Try to understand what changes in the FSE bitstream could help
    println!("\n=== Analyzing FSE bitstream structure ===");
    analyze_fse_bitstream(&our_compressed);
}

fn decode_with_debug(data: &[u8], expected_size: usize) -> Result<Vec<u8>, String> {
    use zstd::stream::read::Decoder;

    let mut decoder =
        Decoder::new(Cursor::new(data)).map_err(|e| format!("Failed to create decoder: {}", e))?;

    let mut output = Vec::with_capacity(expected_size);
    let mut buf = [0u8; 1024];

    loop {
        match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                output.extend_from_slice(&buf[..n]);
            }
            Err(e) => {
                return Err(format!("Read error after {} bytes: {}", output.len(), e));
            }
        }
    }

    Ok(output)
}

fn analyze_fse_bitstream(frame: &[u8]) {
    // Skip to sequence section
    if frame.len() < 7 {
        return;
    }

    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment {
        pos += 1;
    }

    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 {
        return;
    }

    let block_data = &frame[pos..pos + block_size];

    // Parse literals
    let lit_type = block_data[0] & 0x03;
    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => ((block_data[0] >> 3) as usize, 1),
            2 => (
                ((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF,
                2,
            ),
            _ => (0, 1),
        }
    } else {
        return;
    };

    let seq_section = &block_data[lit_header_size + lit_size..];
    println!("Sequence section: {:02x?}", seq_section);

    if seq_section.len() < 3 {
        return;
    }

    let seq_count = seq_section[0] as usize;
    let mode = seq_section[1];
    let bitstream = &seq_section[2..];

    println!("Sequences: {}, Mode: 0x{:02x}", seq_count, mode);
    println!("Bitstream: {:02x?} ({} bytes)", bitstream, bitstream.len());

    // Analyze bit layout
    if !bitstream.is_empty() {
        let last_byte = bitstream[bitstream.len() - 1];
        let sentinel_pos = 7 - last_byte.leading_zeros() as usize;
        let total_bits = (bitstream.len() - 1) * 8 + sentinel_pos;

        println!(
            "Last byte: 0x{:02x}, sentinel at bit {}",
            last_byte, sentinel_pos
        );
        println!("Total data bits: {} (plus sentinel)", total_bits);

        // For predefined mode (0x00), calculate expected bit usage
        if mode == 0x00 {
            // Initial states: LL (6 bits) + OF (5 bits) + ML (6 bits) = 17 bits
            let state_bits = 6 + 5 + 6;
            let remaining_bits = total_bits - state_bits;

            println!("\nExpected layout for predefined mode:");
            println!("  Initial states: 17 bits");
            println!("  Remaining for extras/FSE: {} bits", remaining_bits);

            // For 2 sequences:
            // Seq 0: extras (0 LL + 0 ML + 3 OF = 3 bits) + FSE (5+5+5 = 15 bits) = 18 bits
            // Seq 1: extras (0 LL + 0 ML + 0 OF = 0 bits) = 0 bits
            // Total: 18 bits for seq data
            if seq_count == 2 {
                println!("  For 2 seqs, expected ~18 bits for extras+FSE");
                println!("  Total expected: ~35 bits, actual: {} bits", total_bits);
            }
        }
    }
}

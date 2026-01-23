//! Trace sequence encoding in detail

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Simple repeating pattern
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

    // Parse to find sequence section
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
    let bh = u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
    let block_size = (bh >> 3) as usize;
    pos += 3;
    let block_start = pos;

    // Literals header
    let lit_byte0 = compressed[pos];
    let size_format = (lit_byte0 >> 2) & 0x03;
    let (header_size, lit_size) = match size_format {
        0 | 1 => (1, (lit_byte0 >> 3) as usize),
        2 => {
            let byte1 = compressed[pos + 1];
            (2, ((lit_byte0 >> 4) as usize) | ((byte1 as usize) << 4))
        }
        _ => (3, 0), // 3-byte header
    };
    pos += header_size + lit_size;

    println!("\n=== Sequence Section ===");
    println!("Starts at pos {} (block offset {})", pos, pos - block_start);
    println!(
        "Block size: {}, remaining: {}",
        block_size,
        block_start + block_size - pos
    );

    let seq_section = &compressed[pos..block_start + block_size];
    println!("Sequence bytes ({}):", seq_section.len());
    for b in seq_section {
        print!("{:02x} ", b);
    }
    println!();

    // Parse sequence count
    let seq_count = if seq_section[0] == 0 {
        0
    } else if seq_section[0] < 128 {
        seq_section[0] as usize
    } else if seq_section[0] < 255 {
        
        ((seq_section[0] as usize - 128) << 8) + seq_section[1] as usize
    } else {
        let n = (seq_section[1] as usize) | ((seq_section[2] as usize) << 8);
        n + 0x7F00
    };
    println!("Sequence count: {}", seq_count);

    // Parse compression mode byte
    let mode_pos = if seq_section[0] < 128 {
        1
    } else if seq_section[0] < 255 {
        2
    } else {
        3
    };
    let mode_byte = seq_section[mode_pos];
    println!(
        "\nCompression mode byte: 0x{:02x} (binary: {:08b})",
        mode_byte, mode_byte
    );

    let ll_mode = mode_byte & 0x03;
    let of_mode = (mode_byte >> 2) & 0x03;
    let ml_mode = (mode_byte >> 4) & 0x03;

    let mode_name = |m| match m {
        0 => "Predefined",
        1 => "RLE",
        2 => "FSE_Compressed",
        3 => "Repeat",
        _ => "Unknown",
    };

    println!("  LL mode: {} ({})", ll_mode, mode_name(ll_mode));
    println!("  OF mode: {} ({})", of_mode, mode_name(of_mode));
    println!("  ML mode: {} ({})", ml_mode, mode_name(ml_mode));

    // For RLE mode, next byte is the symbol
    let mut data_pos = mode_pos + 1;
    if ll_mode == 1 {
        println!("\n  LL RLE symbol: {}", seq_section[data_pos]);
        data_pos += 1;
    }
    if of_mode == 1 {
        println!("  OF RLE symbol: {}", seq_section[data_pos]);
        data_pos += 1;
    }
    if ml_mode == 1 {
        println!("  ML RLE symbol: {}", seq_section[data_pos]);
        data_pos += 1;
    }

    println!("\nBitstream starts at offset {}", data_pos);
    println!("Bitstream ({} bytes):", seq_section.len() - data_pos);
    for b in &seq_section[data_pos..] {
        print!("{:02x} ", b);
    }
    println!();

    // What we expect for input "ABCDABCDABCDABCD":
    // Match at pos 4, offset 4, length 12
    // Sequence: ll=4, offset=2 (repeat), ml=12
    // Codes: ll_code=4, of_code=1, ml_code=9
    println!("\n=== Expected Encoding ===");
    println!("Input: ABCDABCDABCDABCD (16 bytes)");
    println!("Match: pos=4, offset=4, length=12");
    println!("Sequence: ll=4, offset_value=2 (repeat_offset_2=4), ml=12");
    println!("Codes: ll_code=4, of_code=1, ml_code=9");

    // Try reference
    println!("\n=== Reference decode ===");
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(dec) => println!("SUCCESS: {} bytes", dec.len()),
        Err(e) => println!("FAILED: {:?}", e),
    }
}

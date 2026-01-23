//! Compare our output with reference zstd byte-by-byte

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

    // Reference compression (level 1 for fast mode)
    let reference = zstd::encode_all(std::io::Cursor::new(input), 1).unwrap();

    println!("\nOurs ({} bytes):", ours.len());
    print_hex(&ours);

    println!("\nReference ({} bytes):", reference.len());
    print_hex(&reference);

    // Verify both decode correctly
    println!("\n=== Decode verification ===");

    match zstd::decode_all(std::io::Cursor::new(&ours)) {
        Ok(dec) => println!("Our output: DECODE OK ({} bytes)", dec.len()),
        Err(e) => println!("Our output: DECODE FAILED - {:?}", e),
    }

    match zstd::decode_all(std::io::Cursor::new(&reference)) {
        Ok(dec) => println!("Reference output: DECODE OK ({} bytes)", dec.len()),
        Err(e) => println!("Reference output: DECODE FAILED - {:?}", e),
    }

    // Parse and compare structures
    println!("\n=== Frame comparison ===");
    parse_frame("Ours", &ours);
    parse_frame("Reference", &reference);
}

fn print_hex(data: &[u8]) {
    for (i, chunk) in data.chunks(16).enumerate() {
        print!("{:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
}

fn parse_frame(label: &str, data: &[u8]) {
    println!("\n{}:", label);

    // Skip magic
    let mut pos = 4;

    // Frame header
    let fhd = data[pos];
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
    let bh = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    println!(
        "  Block type: {} (0=Raw, 2=Compressed), size: {}",
        block_type, block_size
    );

    if block_type == 0 {
        println!("  Raw block - data follows directly");
        return;
    }

    // Compressed block - parse literals
    let lit_byte0 = data[pos];
    let lit_type = lit_byte0 & 0x03;
    let size_format = (lit_byte0 >> 2) & 0x03;
    println!("  Literals: type={}, size_format={}", lit_type, size_format);

    let (header_size, lit_size) = match size_format {
        0 | 1 => (1, (lit_byte0 >> 3) as usize),
        2 => {
            let byte1 = data[pos + 1];
            (2, ((lit_byte0 >> 4) as usize) | ((byte1 as usize) << 4))
        }
        _ => (3, 0),
    };
    println!(
        "  Literals size: {} (header {} bytes)",
        lit_size, header_size
    );
    pos += header_size + lit_size;

    // Sequence section
    let seq_start = pos;
    let seq_count = data[pos] as usize;
    println!("  Sequences: count={}", seq_count);

    if seq_count > 0 {
        pos += 1;
        let mode = data[pos];
        println!("  Mode byte: 0x{:02x}", mode);
        println!("    LL mode: {}", mode & 0x03);
        println!("    OF mode: {}", (mode >> 2) & 0x03);
        println!("    ML mode: {}", (mode >> 4) & 0x03);
        pos += 1;

        // Print remaining bytes as sequence data
        let remaining =
            &data[pos..seq_start + (block_size - (seq_start - (seq_start - seq_count.max(1))))];
        print!("  Sequence data: ");
        for b in &data[pos..] {
            print!("{:02x} ", b);
        }
        println!();
    }
}

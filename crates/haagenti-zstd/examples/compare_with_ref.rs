//! Compare our encoding with reference zstd.

use haagenti_zstd::compress::SpeculativeCompressor;

fn main() {
    let input = b"abcdefghXabcdefghYabcd";

    println!(
        "Input: {:?} ({} bytes)",
        std::str::from_utf8(input).unwrap(),
        input.len()
    );

    // Compress with reference zstd
    let ref_compressed = zstd::encode_all(&input[..], 1).expect("reference zstd failed");
    println!("\nReference zstd output ({} bytes):", ref_compressed.len());
    print_hex(&ref_compressed);

    // Compress with our implementation
    let compressor = SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).expect("our compression failed");
    println!("\nOur output ({} bytes):", our_compressed.len());
    print_hex(&our_compressed);

    // Verify reference can decode our output
    println!("\n=== Decoding Tests ===");

    match zstd::decode_all(&our_compressed[..]) {
        Ok(decoded) => {
            if decoded == input {
                println!("Reference decoded our output: OK");
            } else {
                println!("Reference decoded our output: WRONG DATA");
                println!(
                    "  Got: {:?}",
                    std::str::from_utf8(&decoded).unwrap_or("(invalid utf8)")
                );
            }
        }
        Err(e) => {
            println!("Reference FAILED to decode our output: {}", e);
        }
    }

    // Parse both frames
    println!("\n=== Frame Analysis ===");
    analyze_frame("Reference", &ref_compressed);
    analyze_frame("Ours", &our_compressed);
}

fn print_hex(data: &[u8]) {
    for (i, chunk) in data.chunks(16).enumerate() {
        print!("  {:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
}

fn analyze_frame(name: &str, data: &[u8]) {
    println!("\n{} frame:", name);
    if data.len() < 4 {
        println!("  Too short");
        return;
    }

    // Check magic
    if &data[0..4] == &[0x28, 0xb5, 0x2f, 0xfd] {
        println!("  Magic: OK");
    } else {
        println!("  Magic: {:02x?} (expected 28 b5 2f fd)", &data[0..4]);
    }

    let fhd = data[4];
    println!("  FHD: 0x{:02x}", fhd);
    let single_segment = (fhd & 0x20) != 0;
    let content_checksum = (fhd & 0x04) != 0;
    let fcs_field_size = match (fhd >> 6) & 0x3 {
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
    println!("    Single_Segment: {}", single_segment);
    println!("    Content_Checksum: {}", content_checksum);
    println!("    FCS_Field_Size: {}", fcs_field_size);

    let mut pos = 5;
    if !single_segment {
        println!(
            "    Window_Descriptor: 0x{:02x}",
            data.get(pos).unwrap_or(&0)
        );
        pos += 1;
    }

    if fcs_field_size > 0 {
        let fcs_bytes = &data[pos..pos + fcs_field_size];
        println!("    Frame_Content_Size: {:?}", fcs_bytes);
        pos += fcs_field_size;
    }

    // Block header
    if pos + 3 > data.len() {
        println!("  No block header");
        return;
    }
    let bh = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
    let last_block = (bh & 1) != 0;
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;

    let block_type_name = match block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Reserved",
        _ => "Unknown",
    };

    println!(
        "  Block header at offset {}: last={}, type={} ({}), size={}",
        pos, last_block, block_type, block_type_name, block_size
    );
    pos += 3;

    if block_type == 2 && pos < data.len() {
        // Compressed block - show literals and sequences
        let block_data = &data[pos..];
        println!(
            "  Compressed block data ({} bytes): {:02x?}",
            block_data.len(),
            block_data
        );

        // Parse literals header
        if !block_data.is_empty() {
            let lit_header = block_data[0];
            let lit_type = lit_header & 0x03;
            let lit_type_name = match lit_type {
                0 => "Raw",
                1 => "RLE",
                2 => "Compressed",
                3 => "Treeless",
                _ => "Unknown",
            };
            println!("    Literals type: {} ({})", lit_type, lit_type_name);

            // For raw literals (type 0)
            if lit_type == 0 {
                let (lit_size, header_bytes) = if (lit_header >> 2) & 0x3 == 0 {
                    ((lit_header >> 3) as usize, 1)
                } else if (lit_header >> 2) & 0x3 == 1 {
                    let size = ((lit_header as usize >> 4)
                        | ((block_data.get(1).copied().unwrap_or(0) as usize) << 4))
                        & 0xFFF;
                    (size, 2)
                } else {
                    (0, 1) // Simplified
                };
                println!(
                    "    Literals size: {} (header {} bytes)",
                    lit_size, header_bytes
                );

                let lit_end = header_bytes + lit_size;
                if lit_end <= block_data.len() {
                    let literals = &block_data[header_bytes..lit_end];
                    println!(
                        "    Literals: {:?}",
                        std::str::from_utf8(literals).unwrap_or("(binary)")
                    );

                    // Sequences section
                    let seq_section = &block_data[lit_end..];
                    println!(
                        "    Sequences section ({} bytes): {:02x?}",
                        seq_section.len(),
                        seq_section
                    );

                    if !seq_section.is_empty() {
                        let seq_count = seq_section[0] as usize;
                        println!("    Sequence count: {}", seq_count);

                        if seq_section.len() > 1 {
                            let mode_byte = seq_section[1];
                            println!("    Mode byte: 0x{:02x}", mode_byte);
                            let ll_mode = mode_byte & 0x3;
                            let of_mode = (mode_byte >> 2) & 0x3;
                            let ml_mode = (mode_byte >> 4) & 0x3;
                            let mode_names = ["Predefined", "RLE", "FSE Compressed", "Repeat"];
                            println!(
                                "      LL mode: {} ({})",
                                ll_mode,
                                mode_names.get(ll_mode as usize).unwrap_or(&"?")
                            );
                            println!(
                                "      OF mode: {} ({})",
                                of_mode,
                                mode_names.get(of_mode as usize).unwrap_or(&"?")
                            );
                            println!(
                                "      ML mode: {} ({})",
                                ml_mode,
                                mode_names.get(ml_mode as usize).unwrap_or(&"?")
                            );

                            let bitstream_start = if mode_byte == 0 { 2 } else { 2 }; // Simplified
                            let bitstream = &seq_section[bitstream_start..];
                            println!(
                                "    FSE bitstream ({} bytes): {:02x?}",
                                bitstream.len(),
                                bitstream
                            );
                        }
                    }
                }
            }
        }
    }
}

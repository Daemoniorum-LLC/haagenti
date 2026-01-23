//! Analyze block structure of both our and reference compression

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn parse_frame(data: &[u8], label: &str) {
    println!("\n=== {} ({} bytes) ===", label, data.len());

    if data.len() < 4 || data[0..4] != [0x28, 0xB5, 0x2F, 0xFD] {
        println!("  Invalid magic number");
        return;
    }

    let mut pos = 4;

    // Frame header descriptor
    let fhd = data[pos];
    pos += 1;

    let fcs_flag = (fhd >> 6) & 0x3;
    let single_segment = (fhd >> 5) & 0x1 != 0;
    let content_checksum = (fhd >> 2) & 0x1 != 0;
    let dict_id_flag = fhd & 0x3;

    println!("  Frame Header:");
    println!("    FCS flag: {}", fcs_flag);
    println!("    Single segment: {}", single_segment);
    println!("    Content checksum: {}", content_checksum);

    // Skip window descriptor if not single segment
    if !single_segment {
        let _window_desc = data[pos];
        pos += 1;
    }

    // Skip dictionary ID
    let dict_id_size = match dict_id_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => 0,
    };
    pos += dict_id_size;

    // Frame content size
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
    if fcs_size > 0 {
        let mut fcs = 0u64;
        for i in 0..fcs_size {
            fcs |= (data[pos + i] as u64) << (i * 8);
        }
        if fcs_flag == 1 {
            fcs += 256;
        }
        println!("    Content size: {}", fcs);
    }
    pos += fcs_size;

    println!("  Frame header size: {} bytes", pos - 4);

    // Parse blocks
    let mut block_num = 0;
    while pos + 3 <= data.len() {
        let header = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
        let is_last = (header & 1) != 0;
        let block_type = (header >> 1) & 0x3;
        let block_size = (header >> 3) as usize;
        pos += 3;

        let type_name = match block_type {
            0 => "Raw",
            1 => "RLE",
            2 => "Compressed",
            3 => "Reserved",
            _ => "Unknown",
        };

        println!(
            "\n  Block {}: type={} ({}), size={} bytes, last={}",
            block_num, block_type, type_name, block_size, is_last
        );

        if block_type == 2 && pos + block_size <= data.len() {
            // Analyze compressed block structure
            let block_data = &data[pos..pos + block_size];
            analyze_compressed_block(block_data);
        }

        pos += block_size;
        block_num += 1;

        if is_last {
            break;
        }
    }

    // Check for checksum
    if content_checksum && pos + 4 <= data.len() {
        println!(
            "\n  Checksum: {:02x}{:02x}{:02x}{:02x}",
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3]
        );
    }
}

fn analyze_compressed_block(block: &[u8]) {
    if block.is_empty() {
        return;
    }

    // Parse literals section header
    let header_byte = block[0];
    let lit_block_type = header_byte & 0x3;
    let size_format = (header_byte >> 2) & 0x3;

    let type_name = match lit_block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Treeless",
        _ => "Unknown",
    };

    println!(
        "    Literals: type={} ({}), size_format={}",
        lit_block_type, type_name, size_format
    );

    // Parse literal sizes based on format
    let (regen_size, compressed_size, header_size) = match lit_block_type {
        0 | 1 => {
            // Raw or RLE
            let (size, hdr) = match size_format {
                0 | 1 => ((header_byte >> 3) as usize, 1),
                2 => {
                    let s = ((header_byte >> 4) as usize) | ((block[1] as usize) << 4);
                    (s, 2)
                }
                3 => {
                    let s = ((header_byte >> 4) as usize)
                        | ((block[1] as usize) << 4)
                        | ((block[2] as usize) << 12);
                    (s, 3)
                }
                _ => (0, 1),
            };
            (size, size, hdr)
        }
        2 | 3 => {
            // Compressed or Treeless
            match size_format {
                0 => {
                    // 4 streams, sizes in 10+10 bits
                    if block.len() >= 3 {
                        let combined = ((header_byte >> 4) as u32)
                            | ((block[1] as u32) << 4)
                            | ((block[2] as u32) << 12);
                        let regen = (combined & 0x3FF) as usize;
                        let comp = ((combined >> 10) & 0x3FF) as usize;
                        (regen, comp, 3)
                    } else {
                        (0, 0, 1)
                    }
                }
                1 => {
                    // 4 streams, sizes in 12+12 bits
                    if block.len() >= 4 {
                        let combined = ((header_byte >> 4) as u32)
                            | ((block[1] as u32) << 4)
                            | ((block[2] as u32) << 12)
                            | ((block[3] as u32) << 20);
                        let regen = (combined & 0xFFF) as usize;
                        let comp = ((combined >> 12) & 0xFFF) as usize;
                        (regen, comp, 4)
                    } else {
                        (0, 0, 1)
                    }
                }
                2 => {
                    // 4 streams, sizes in 14+14 bits
                    if block.len() >= 5 {
                        let combined = ((header_byte >> 4) as u32)
                            | ((block[1] as u32) << 4)
                            | ((block[2] as u32) << 12)
                            | ((block[3] as u32) << 20)
                            | ((block[4] as u32) << 28);
                        let regen = (combined & 0x3FFF) as usize;
                        let comp = ((combined >> 14) & 0x3FFF) as usize;
                        (regen, comp, 5)
                    } else {
                        (0, 0, 1)
                    }
                }
                3 => {
                    // 1 stream, sizes in 18+18 bits
                    if block.len() >= 5 {
                        let combined = ((header_byte >> 4) as u64)
                            | ((block[1] as u64) << 4)
                            | ((block[2] as u64) << 12)
                            | ((block[3] as u64) << 20)
                            | ((block[4] as u64) << 28);
                        let regen = (combined & 0x3FFFF) as usize;
                        let comp = ((combined >> 18) & 0x3FFFF) as usize;
                        (regen, comp, 5)
                    } else {
                        (0, 0, 1)
                    }
                }
                _ => (0, 0, 1),
            }
        }
        _ => (0, 0, 1),
    };

    println!("      Regen size: {} bytes", regen_size);
    println!("      Compressed size: {} bytes", compressed_size);
    println!("      Header size: {} bytes", header_size);

    // Estimate sequences section
    let lit_section_size = header_size
        + if lit_block_type == 1 {
            1
        } else {
            compressed_size
        };
    if lit_section_size < block.len() {
        let seq_section = &block[lit_section_size..];
        println!("    Sequences section: {} bytes", seq_section.len());

        if !seq_section.is_empty() {
            let num_seq = if seq_section[0] == 0 {
                0
            } else if seq_section[0] < 128 {
                seq_section[0] as usize
            } else if seq_section[0] < 255 && seq_section.len() >= 2 {
                ((seq_section[0] as usize - 128) << 8) + seq_section[1] as usize
            } else if seq_section.len() >= 3 {
                (seq_section[1] as usize) + ((seq_section[2] as usize) << 8) + 0x7F00
            } else {
                0
            };
            println!("      Number of sequences: {}", num_seq);
        }
    }
}

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    for &size in &[1024, 4096, 16384, 65536] {
        let data: Vec<u8> = sample.iter().cycle().take(size).copied().collect();

        println!("\n============================================================");
        println!("INPUT SIZE: {} bytes", size);
        println!("============================================================");

        // Reference compression
        let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();
        parse_frame(&ref_compressed, "REFERENCE");

        // Our compression
        let compressor = ZstdCompressor::new();
        let our_compressed = compressor.compress(&data).unwrap();
        parse_frame(&our_compressed, "OURS");

        let gap = (our_compressed.len() as f64 / ref_compressed.len() as f64 - 1.0) * 100.0;
        println!(
            "\n  Size comparison: ref={}, ours={}, gap={:+.1}%",
            ref_compressed.len(),
            our_compressed.len(),
            gap
        );
    }
}

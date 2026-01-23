//! Correctly parse our frame output

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    let data: Vec<u8> = sample.iter().cycle().take(1024).copied().collect();

    let compressor = ZstdCompressor::new();
    let frame = compressor.compress(&data).unwrap();

    println!("Frame: {} bytes", frame.len());
    println!("First 50 bytes: {:02x?}", &frame[..50.min(frame.len())]);

    // Parse magic
    let magic = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
    println!("\nMagic: 0x{:08X}", magic);

    let mut pos = 4;

    // Frame header descriptor
    let fhd = frame[pos];
    println!("FHD: 0x{:02x}", fhd);
    let fcs_flag = (fhd >> 6) & 3;
    let single_segment = (fhd >> 5) & 1;
    let content_checksum = (fhd >> 2) & 1;
    let dict_id_flag = fhd & 3;
    println!("  FCS_flag: {}", fcs_flag);
    println!("  Single_Segment: {}", single_segment);
    println!("  Content_Checksum: {}", content_checksum);
    println!("  Dict_ID_flag: {}", dict_id_flag);
    pos += 1;

    // Window descriptor (if not single segment)
    if single_segment == 0 {
        let wd = frame[pos];
        let exp = (wd >> 3) as u32;
        let mantissa = (wd & 7) as u32;
        let window_size = (1u64 << (10 + exp)) * ((8 + mantissa) as u64) / 8;
        println!(
            "Window_Descriptor: 0x{:02x} (exp={}, mantissa={}, size={})",
            wd, exp, mantissa, window_size
        );
        pos += 1;
    }

    // Dict ID (if present)
    let dict_id_size = match dict_id_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => 0,
    };
    pos += dict_id_size;

    // FCS (if present)
    let fcs_size = match fcs_flag {
        0 => {
            if single_segment == 1 {
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
            fcs |= (frame[pos + i] as u64) << (i * 8);
        }
        println!("FCS: {} bytes -> content_size={}", fcs_size, fcs);
    }
    pos += fcs_size;

    println!("\nFrame header ends at position {}", pos);

    // Block header
    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let is_last = bh & 1;
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    println!("Block header: 0x{:06x}", bh);
    println!("  is_last: {}", is_last);
    println!("  block_type: {} (0=Raw, 1=RLE, 2=Compressed)", block_type);
    println!("  block_size: {}", block_size);
    pos += 3;

    if block_type == 2 {
        // Compressed block
        let block_start = pos;
        let block = &frame[pos..pos + block_size];
        println!(
            "\nBlock data ({} bytes): {:02x?}",
            block.len(),
            &block[..20.min(block.len())]
        );

        // Literals header
        let lh0 = block[0];
        let lit_type = lh0 & 3;
        let size_format = (lh0 >> 2) & 3;
        println!("\nLiterals header byte 0: 0x{:02x}", lh0);
        println!(
            "  lit_type: {} (0=Raw, 1=RLE, 2=Compressed, 3=Treeless)",
            lit_type
        );
        println!("  size_format: {}", size_format);

        let (regen_size, comp_size, header_size) = match lit_type {
            0 | 1 => {
                // Raw or RLE
                match size_format {
                    0 => {
                        let size = (lh0 >> 3) as usize;
                        println!("  Size (5-bit): {}", size);
                        (size, if lit_type == 1 { 1 } else { size }, 1)
                    }
                    1 => {
                        let size = ((lh0 >> 4) as usize) | ((block[1] as usize) << 4);
                        println!("  Bytes [0,1]: [{:02x}, {:02x}]", lh0, block[1]);
                        println!(
                            "  Size (12-bit): {} = (0x{:02x} >> 4) | (0x{:02x} << 4) = {} | {}",
                            size,
                            lh0,
                            block[1],
                            (lh0 >> 4),
                            (block[1] as usize) << 4
                        );
                        (size, if lit_type == 1 { 1 } else { size }, 2)
                    }
                    2 | 3 => {
                        let size = ((lh0 >> 4) as usize)
                            | ((block[1] as usize) << 4)
                            | ((block[2] as usize) << 12);
                        println!("  Size (20-bit): {}", size);
                        (size, if lit_type == 1 { 1 } else { size }, 3)
                    }
                    _ => (0, 0, 1),
                }
            }
            _ => {
                println!("  Compressed literals - parsing not implemented");
                (0, 0, 1)
            }
        };

        println!("\nLiterals section:");
        println!("  Header: {} bytes", header_size);
        println!("  Regen size: {} bytes", regen_size);
        println!("  Compressed size: {} bytes", comp_size);

        let lit_section_size = header_size + comp_size;
        println!("  Total literals section: {} bytes", lit_section_size);

        let seq_section = &block[lit_section_size..];
        println!("\nSequences section: {} bytes", seq_section.len());
        println!(
            "  First bytes: {:02x?}",
            &seq_section[..10.min(seq_section.len())]
        );

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
            println!("  Number of sequences: {}", num_seq);
        }
    }

    // Verify with reference zstd
    println!("\n--- Reference zstd decode ---");
    match zstd::decode_all(std::io::Cursor::new(&frame)) {
        Ok(decompressed) => {
            if decompressed == data {
                println!("SUCCESS: Reference decoded correctly!");
            } else {
                println!(
                    "MISMATCH: Decoded {} bytes, expected {}",
                    decompressed.len(),
                    data.len()
                );
            }
        }
        Err(e) => {
            println!("FAILED: {}", e);
        }
    }
}

//! Deep trace of our compressed frame

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    let input = b"ABCDABCDABCDABCD";

    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(input).unwrap();

    println!("Our output ({} bytes): {:02x?}", ours.len(), &ours);

    // Detailed parse
    let mut pos = 0;

    // Magic
    println!("\n0x{:04x}: Magic = {:02x} {:02x} {:02x} {:02x}", pos, ours[0], ours[1], ours[2], ours[3]);
    pos += 4;

    // FHD
    let fhd = ours[pos];
    println!("0x{:04x}: FHD = 0x{:02x} (binary: {:08b})", pos, fhd, fhd);
    let single_segment = (fhd >> 5) & 1 != 0;
    let fcs_flag = (fhd >> 6) & 3;
    println!("        Single_Segment = {}, FCS_flag = {}", single_segment, fcs_flag);
    pos += 1;

    // Window Descriptor
    if !single_segment {
        let wd = ours[pos];
        println!("0x{:04x}: Window_Descriptor = 0x{:02x}", pos, wd);
        let exp = wd >> 3;
        let mantissa = wd & 7;
        let window_base = 1u64 << (10 + exp);
        let window_size = window_base + window_base / 8 * mantissa as u64;
        println!("        exp={}, mantissa={}, window_size={} bytes", exp, mantissa, window_size);
        pos += 1;
    }

    // Block header
    let bh = u32::from_le_bytes([ours[pos], ours[pos+1], ours[pos+2], 0]);
    println!("0x{:04x}: Block header = {:02x} {:02x} {:02x}", pos, ours[pos], ours[pos+1], ours[pos+2]);
    let last = bh & 1;
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    println!("        last={}, type={}, size={}", last, block_type, block_size);
    pos += 3;

    // Block content
    let block_end = pos + block_size;
    println!("0x{:04x}: Block content ({} bytes):", pos, block_size);

    // Literals header
    let lit_byte = ours[pos];
    let lit_type = lit_byte & 3;
    let size_format = (lit_byte >> 2) & 3;
    println!("  0x{:04x}: Literals header = 0x{:02x}", pos, lit_byte);
    println!("          type={}, size_format={}", lit_type, size_format);

    let (lit_size, header_len) = match size_format {
        0 | 2 => ((lit_byte >> 3) as usize, 1usize),
        1 => {
            let size = ((ours[pos] >> 4) as usize) | ((ours[pos+1] as usize) << 4);
            (size, 2)
        }
        3 => {
            let size = ((ours[pos] >> 4) as usize)
                | ((ours[pos+1] as usize) << 4)
                | ((ours[pos+2] as usize) << 12);
            (size, 3)
        }
        _ => unreachable!()
    };
    println!("          lit_size={}, header_len={}", lit_size, header_len);
    pos += header_len;

    // Literal bytes
    print!("  0x{:04x}: Literals: ", pos);
    for i in 0..lit_size {
        print!("{:02x} ", ours[pos + i]);
    }
    println!("(\"{}\")", String::from_utf8_lossy(&ours[pos..pos+lit_size]));
    pos += lit_size;

    // Sequence section
    println!("  0x{:04x}: Sequences section:", pos);

    let seq_count = ours[pos];
    println!("    Count byte: 0x{:02x} = {}", seq_count, seq_count);
    pos += 1;

    if seq_count > 0 {
        let mode_byte = ours[pos];
        println!("    Mode byte: 0x{:02x} = {:08b}", mode_byte, mode_byte);
        let ll_mode = mode_byte & 3;
        let of_mode = (mode_byte >> 2) & 3;
        let ml_mode = (mode_byte >> 4) & 3;
        println!("      LL_mode={}, OF_mode={}, ML_mode={}", ll_mode, of_mode, ml_mode);
        pos += 1;

        // RLE symbols if all RLE
        if ll_mode == 1 && of_mode == 1 && ml_mode == 1 {
            println!("    RLE symbols: LL={}, OF={}, ML={}", ours[pos], ours[pos+1], ours[pos+2]);
            pos += 3;
        }

        // Bitstream
        println!("    Bitstream ({} bytes): {:02x?}", block_end - pos, &ours[pos..block_end]);
    }

    // Try to decode the sequence
    println!("\n=== Sequence interpretation ===");
    println!("LL code 4 -> literal_length = 4");
    println!("OF code 1 -> offset_value = (1 << 1) + 0 = 2");
    println!("            Using repeat_offset_2 = 4, actual offset = 4");
    println!("ML code 9 -> match_length = 3 + 9 = 12");
    println!();
    println!("Decoding:");
    println!("1. Copy 4 literals: ABCD");
    println!("2. Copy 12 bytes from offset 4: ABCDABCDABCD");
    println!("Result: ABCDABCDABCDABCD (16 bytes)");

    // Try reference decode
    println!("\n=== Reference decode ===");
    match zstd::decode_all(std::io::Cursor::new(&ours)) {
        Ok(dec) => {
            println!("SUCCESS: {} bytes", dec.len());
            println!("Content: {:?}", String::from_utf8_lossy(&dec));
        }
        Err(e) => println!("FAILED: {:?}", e),
    }
}

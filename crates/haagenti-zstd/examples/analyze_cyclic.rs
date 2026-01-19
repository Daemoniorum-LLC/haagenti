//! Investigate cyclic pattern compression

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;
use std::io::Cursor;

fn main() {
    // The "random-ish" test case: 0,1,2,...,255,0,1,2,...,255,...
    let cyclic: Vec<u8> = (0..256).cycle().take(1000).map(|x| x as u8).collect();

    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(&cyclic).unwrap();
    let reference = zstd::encode_all(Cursor::new(&cyclic), 1).unwrap();

    println!("Input: {} bytes (cyclic 0-255 pattern)", cyclic.len());
    println!("Our output: {} bytes ({:.1}%)", ours.len(), ours.len() as f64 / cyclic.len() as f64 * 100.0);
    println!("Ref output: {} bytes ({:.1}%)", reference.len(), reference.len() as f64 / cyclic.len() as f64 * 100.0);

    // Check if both decompress correctly
    let our_dec = zstd::decode_all(Cursor::new(&ours)).unwrap();
    let ref_dec = zstd::decode_all(Cursor::new(&reference)).unwrap();
    println!("\nDecompression check: ours={}, ref={}", our_dec == cyclic, ref_dec == cyclic);

    // Count sequences - the pattern should have many matches
    println!("\n=== Analysis ===");

    // The pattern has period 256, so any position > 256 could match at offset 256
    // For 1000 bytes: first 256 are literals, then 744 bytes can match at offset 256
    let expected_matches = 1000 - 256;
    let expected_literals = 256;
    println!("Expected: ~{} literal bytes, ~{} bytes via matches at offset 256", expected_literals, expected_matches);

    // Try compressing just 512 bytes - enough for one full match
    let small_cyclic: Vec<u8> = (0..256).cycle().take(512).map(|x| x as u8).collect();
    let small_ours = compressor.compress(&small_cyclic).unwrap();
    let small_ref = zstd::encode_all(Cursor::new(&small_cyclic), 1).unwrap();
    println!("\n512-byte version:");
    println!("Our output: {} bytes ({:.1}%)", small_ours.len(), small_ours.len() as f64 / 512.0 * 100.0);
    println!("Ref output: {} bytes ({:.1}%)", small_ref.len(), small_ref.len() as f64 / 512.0 * 100.0);

    // Analyze: are we even finding matches?
    println!("\n=== Our compressed data structure ===");
    parse_our_frame(&ours);

    println!("\n=== Reference compressed data structure ===");
    parse_our_frame(&reference);
}

fn parse_our_frame(data: &[u8]) {
    // Magic number (4 bytes)
    let mut pos = 4;

    // Frame header descriptor
    let fhd = data[pos];
    pos += 1;

    let single_segment = (fhd & 0x20) != 0;
    let fcs_field_size = match (fhd >> 6) & 3 {
        0 if single_segment => 1,
        0 => 0,
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };

    // Window descriptor (if not single segment)
    if !single_segment {
        pos += 1;
    }

    // FCS
    pos += fcs_field_size;

    println!("Frame header ends at offset {}", pos);

    // Block header
    let bh = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], 0]);
    let last = (bh & 1) != 0;
    let block_type = (bh >> 1) & 3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    println!("Block: last={}, type={} ({}), size={}",
             last,
             block_type,
             match block_type { 0 => "Raw", 1 => "RLE", 2 => "Compressed", _ => "Reserved" },
             block_size);

    if block_type != 2 {
        println!("Not a compressed block - cannot analyze further");
        return;
    }

    // Literals section
    let lit_byte = data[pos];
    let lit_type = lit_byte & 3;
    let size_format = (lit_byte >> 2) & 3;

    let (lit_regen_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        // Raw or RLE
        match size_format {
            0 | 2 => ((lit_byte >> 3) as usize, 1),
            1 => {
                let size = ((data[pos] >> 4) as usize) | ((data[pos+1] as usize) << 4);
                (size, 2)
            }
            3 => {
                let size = ((data[pos] >> 4) as usize)
                    | ((data[pos+1] as usize) << 4)
                    | ((data[pos+2] as usize) << 12);
                (size, 3)
            }
            _ => panic!(),
        }
    } else {
        // Compressed or Treeless - more complex
        match size_format {
            0 | 1 => {
                // 2 streams, 10-bit sizes
                let combined = ((data[pos] >> 4) as usize)
                    | ((data[pos+1] as usize) << 4)
                    | ((data[pos+2] as usize) << 12);
                let regen = combined & 0x3FF;
                (regen, 3)
            }
            2 => {
                // 4 streams, 14-bit sizes
                let combined = ((data[pos] >> 4) as usize)
                    | ((data[pos+1] as usize) << 4)
                    | ((data[pos+2] as usize) << 12)
                    | ((data[pos+3] as usize) << 20);
                let regen = combined & 0x3FFF;
                (regen, 4)
            }
            3 => {
                // 4 streams, 18-bit sizes
                let combined = ((data[pos] >> 4) as usize)
                    | ((data[pos+1] as usize) << 4)
                    | ((data[pos+2] as usize) << 12)
                    | ((data[pos+3] as usize) << 20)
                    | ((data[pos+4] as usize) << 28);
                let regen = combined & 0x3FFFF;
                (regen, 5)
            }
            _ => panic!(),
        }
    };

    println!("Literals: type={} ({}), size_format={}, regen_size={}, header_size={}",
             lit_type,
             match lit_type { 0 => "Raw", 1 => "RLE", 2 => "Compressed", 3 => "Treeless", _ => "?" },
             size_format,
             lit_regen_size,
             lit_header_size);

    // Calculate approximate literal data size
    let lit_data_start = pos + lit_header_size;
    let lit_data_end = if lit_type == 0 {
        lit_data_start + lit_regen_size
    } else if lit_type == 1 {
        lit_data_start + 1  // RLE is just 1 byte
    } else {
        // Compressed - harder to determine without full parsing
        0
    };

    if lit_type == 0 {
        println!("Literal bytes: {} (Raw)", lit_regen_size);
    }

    // Try to find sequence section
    // For raw/RLE literals, we know exactly where sequences start
    if lit_type == 0 || lit_type == 1 {
        let seq_start = if lit_type == 0 {
            lit_data_start + lit_regen_size
        } else {
            lit_data_start + 1
        };

        let seq_count_byte = data[seq_start];
        let (num_sequences, seq_header_size) = if seq_count_byte < 128 {
            (seq_count_byte as usize, 1)
        } else if seq_count_byte < 255 {
            let count = ((seq_count_byte as usize - 128) << 8) | (data[seq_start + 1] as usize);
            (count, 2)
        } else {
            let count = (data[seq_start + 1] as usize) | ((data[seq_start + 2] as usize) << 8) + 0x7F00;
            (count, 3)
        };

        println!("Sequences: count={}", num_sequences);

        if num_sequences > 0 {
            let mode_byte = data[seq_start + seq_header_size];
            let ll_mode = mode_byte & 3;
            let of_mode = (mode_byte >> 2) & 3;
            let ml_mode = (mode_byte >> 4) & 3;
            println!("Modes: LL={} OF={} ML={}", ll_mode, of_mode, ml_mode);
        }
    }
}

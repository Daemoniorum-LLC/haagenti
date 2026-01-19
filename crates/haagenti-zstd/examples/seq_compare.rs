//! Compare sequence section byte-by-byte

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    let input = b"The quick brown fox jumps over the lazy dog. ";
    let mut large_input = Vec::new();
    for _ in 0..10 {
        large_input.extend_from_slice(input);
    }

    let compressor = ZstdCompressor::new();
    let ours = compressor.compress(&large_input).unwrap();
    let reference = zstd::encode_all(std::io::Cursor::new(&large_input), 1).unwrap();

    // Parse to find sequence sections
    println!("=== Our sequence section ===");
    let our_seq = parse_seq_section(&ours);

    println!("\n=== Reference sequence section ===");
    let ref_seq = parse_seq_section(&reference);

    println!("\n=== Byte comparison ===");
    let max_len = our_seq.len().max(ref_seq.len());
    for i in 0..max_len {
        let o = our_seq.get(i);
        let r = ref_seq.get(i);
        let diff = if o != r { " <-- DIFF" } else { "" };
        match (o, r) {
            (Some(ob), Some(rb)) => {
                println!("{:02x}: ours={:02x} ref={:02x}{}", i, ob, rb, diff);
            }
            (Some(ob), None) => println!("{:02x}: ours={:02x} ref=--{}", i, ob, diff),
            (None, Some(rb)) => println!("{:02x}: ours=-- ref={:02x}{}", i, rb, diff),
            (None, None) => break,
        }
    }
}

fn parse_seq_section(data: &[u8]) -> Vec<u8> {
    // Skip magic + FHD + window
    let mut pos = 4 + 1 + 1; // assuming standard frame header

    // Block header
    let bh = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], 0]);
    let block_size = (bh >> 3) as usize;
    pos += 3;
    let block_start = pos;

    // Literals
    let lit_byte = data[pos];
    let lit_type = lit_byte & 3;
    let size_format = (lit_byte >> 2) & 3;

    let (lit_size, header_len) = match size_format {
        0 | 2 => ((lit_byte >> 3) as usize, 1usize),
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
    };

    pos += header_len + lit_size;

    println!("Block: type={}, size={}", (bh >> 1) & 3, block_size);
    println!("Literals: type={}, size={}", lit_type, lit_size);
    println!("Sequence section starts at offset 0x{:02x}", pos);

    // Return sequence section
    let block_end = block_start + block_size;
    data[pos..block_end].to_vec()
}

//! Test raw literals header encoding

fn main() {
    for size in [1, 5, 11, 20, 100] {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();

        println!("\n=== Size {} ===", size);
        println!("Compressed: {} bytes", compressed.len());

        // Full hex dump
        for (i, chunk) in compressed.chunks(16).enumerate() {
            print!("{:04x}: ", i * 16);
            for b in chunk {
                print!("{:02x} ", b);
            }
            println!();
        }

        // Parse block header
        let mut pos = 4; // skip magic
        let fhd = compressed[pos];
        pos += 1;
        let single_segment = (fhd >> 5) & 1 != 0;
        if !single_segment {
            pos += 1;
        }
        let fcs_flag = (fhd >> 6) & 3;
        pos += match fcs_flag {
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

        // Block header
        let block_header =
            u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
        let is_last = block_header & 1;
        let block_type = (block_header >> 1) & 3;
        let block_size = block_header >> 3;
        pos += 3;

        println!(
            "Block: type={} ({}), size={}, last={}",
            block_type,
            match block_type {
                0 => "Raw",
                1 => "RLE",
                2 => "Compressed",
                _ => "?",
            },
            block_size,
            is_last
        );

        if block_type == 2 && pos < compressed.len() {
            let lit_header = compressed[pos];
            let lit_type = lit_header & 0x03;
            let size_format = (lit_header >> 2) & 0x03;
            println!(
                "Literals: header=0x{:02x}, type={} ({}), size_format={}",
                lit_header,
                lit_type,
                match lit_type {
                    0 => "Raw",
                    1 => "RLE",
                    2 => "Compressed",
                    3 => "Treeless",
                    _ => "?",
                },
                size_format
            );
        }
    }
}

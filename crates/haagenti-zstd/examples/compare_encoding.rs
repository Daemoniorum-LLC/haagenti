//! Compare our encoding vs reference zstd

fn main() {
    // Simple failing pattern
    let mut data = Vec::new();
    data.extend(vec![b'X'; 100]);
    data.extend(vec![b'Y'; 100]);

    println!("Input size: {} bytes", data.len());
    println!("Input pattern: X*100 Y*100");

    // Reference compression
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();
    println!("\nReference zstd:");
    println!("  Compressed size: {} bytes", ref_compressed.len());
    print!("  Hex: ");
    for b in ref_compressed.iter().take(80) {
        print!("{:02x} ", b);
    }
    println!();

    // Our compression
    use haagenti_core::CompressionLevel;
    use haagenti_zstd::compress::CompressContext;
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let our_compressed = ctx.compress(&data).unwrap();
    println!("\nOur compression:");
    println!("  Compressed size: {} bytes", our_compressed.len());
    print!("  Hex: ");
    for b in our_compressed.iter().take(80) {
        print!("{:02x} ", b);
    }
    println!();

    // Decode block headers
    println!("\n=== Reference Frame Analysis ===");
    analyze_frame(&ref_compressed);

    println!("\n=== Our Frame Analysis ===");
    analyze_frame(&our_compressed);
}

fn analyze_frame(data: &[u8]) {
    if data.len() < 6 {
        println!("Too short for frame");
        return;
    }

    // Magic
    println!(
        "Magic: {:02x} {:02x} {:02x} {:02x}",
        data[0], data[1], data[2], data[3]
    );

    // Frame header descriptor
    let fhd = data[4];
    let fcs_size = (fhd >> 6) & 3;
    let single_segment = (fhd >> 5) & 1;
    let checksum = (fhd >> 2) & 1;
    let dict_id_flag = fhd & 3;
    println!(
        "FHD: {:02x} (FCS_size={}, single_seg={}, checksum={}, dict_id_flag={})",
        fhd, fcs_size, single_segment, checksum, dict_id_flag
    );

    let mut pos = 5;

    // Window descriptor (if not single segment)
    if single_segment == 0 {
        let wd = data[pos];
        let exp = wd >> 3;
        let mant = wd & 7;
        let window_size = (1 << (10 + exp)) * (1 + mant as usize / 8);
        println!(
            "Window_Desc: {:02x} (exp={}, mant={}, size={})",
            wd, exp, mant, window_size
        );
        pos += 1;
    }

    // FCS
    let fcs_bytes = match fcs_size {
        0 => 0,
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 0,
    };
    if fcs_bytes > 0 {
        println!("FCS: {} bytes at pos {}", fcs_bytes, pos);
        pos += fcs_bytes;
    }

    // Block header
    if pos + 3 <= data.len() {
        let bh =
            (data[pos] as u32) | ((data[pos + 1] as u32) << 8) | ((data[pos + 2] as u32) << 16);
        let last = bh & 1;
        let block_type = (bh >> 1) & 3;
        let block_size = bh >> 3;
        let type_name = match block_type {
            0 => "Raw",
            1 => "RLE",
            2 => "Compressed",
            3 => "Reserved",
            _ => "?",
        };
        println!(
            "Block: last={}, type={} ({}), size={}",
            last, block_type, type_name, block_size
        );
        pos += 3;

        // Block content
        if block_type == 2 && pos < data.len() {
            analyze_compressed_block(&data[pos..pos + block_size as usize]);
        }
    }
}

fn analyze_compressed_block(block: &[u8]) {
    println!(
        "  Block content ({} bytes): {:02x?}",
        block.len(),
        &block[..block.len().min(40)]
    );

    if block.is_empty() {
        return;
    }

    // Literals header
    let lh = block[0];
    let lit_type = lh & 3;
    let lit_type_name = match lit_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Treeless",
        _ => "?",
    };

    let size_format = (lh >> 2) & 3;

    let (lit_size, header_bytes) = match (lit_type, size_format) {
        (0, 0) | (1, 0) => ((lh >> 3) as usize, 1), // 5-bit
        (0, 1) | (1, 1) => {
            let s = ((lh as usize) >> 4) | ((block[1] as usize) << 4);
            (s & 0xFFF, 2)
        }
        _ => (0, 1), // Compressed has different format
    };

    println!(
        "  Literals: type={} ({}), size_format={}, size={}",
        lit_type, lit_type_name, size_format, lit_size
    );

    if lit_type == 0 {
        // Raw literals
        let lit_start = header_bytes;
        let lit_end = lit_start + lit_size;
        if lit_end <= block.len() {
            println!("  Literal bytes: {:02x?}", &block[lit_start..lit_end]);
        }
    }
}

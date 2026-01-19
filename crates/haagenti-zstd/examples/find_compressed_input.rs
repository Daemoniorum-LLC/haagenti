//! Find an input where reference zstd produces a Compressed block.

fn main() {
    // Try various inputs to find one that reference compresses
    let test_cases = [
        // Repetitive patterns
        "aaaaaaaaaaaaaaaa",
        "abababababababab",
        "abcabcabcabcabcabc",
        // Longer patterns with good compression ratio
        "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.",
        "hello world hello world hello world hello world",
        // Binary-like patterns
        &"01234567".repeat(10),
        // Random-ish but compressible
        &"this is a test of the compression algorithm with some repeated words words words words",
    ];

    for input_str in &test_cases {
        let input = input_str.as_bytes();
        let ref_compressed = zstd::encode_all(input, 1).unwrap();
        let block_type = get_block_type(&ref_compressed);

        if block_type == "Compressed" {
            println!("FOUND: {:?}", &input_str[..input_str.len().min(50)]);
            println!("  Input: {} bytes", input.len());
            println!("  Reference: {} bytes, block_type={}", ref_compressed.len(), block_type);

            // Try our compression
            let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
            let our_compressed = compressor.compress(input).unwrap();
            let our_block_type = get_block_type(&our_compressed);
            println!("  Ours: {} bytes, block_type={}", our_compressed.len(), our_block_type);

            // Test decode
            match zstd::decode_all(&our_compressed[..]) {
                Ok(decoded) if decoded == input => println!("  Status: Reference decodes OK"),
                Ok(_) => println!("  Status: WRONG DATA"),
                Err(e) => println!("  Status: DECODE ERROR ({})", e),
            }

            // If both are compressed, compare FSE bitstreams
            if our_block_type == "Compressed" {
                compare_bitstreams(&ref_compressed, &our_compressed);
            }
            println!();
        }
    }

    // Try even longer input
    println!("\n=== Testing with larger input ===");
    let long_input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                      Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                      Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    let input = long_input.as_bytes();
    let ref_compressed = zstd::encode_all(input, 1).unwrap();
    let block_type = get_block_type(&ref_compressed);
    println!("Input: {} bytes, \"{}...\"", input.len(), &long_input[..40]);
    println!("Reference: {} bytes, block_type={}", ref_compressed.len(), block_type);

    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();
    let our_block_type = get_block_type(&our_compressed);
    println!("Ours: {} bytes, block_type={}", our_compressed.len(), our_block_type);

    match zstd::decode_all(&our_compressed[..]) {
        Ok(decoded) if decoded == input => println!("Status: Reference decodes OK"),
        Ok(_) => println!("Status: WRONG DATA"),
        Err(e) => println!("Status: DECODE ERROR ({})", e),
    }
}

fn get_block_type(frame: &[u8]) -> &'static str {
    if frame.len() < 7 { return "too_short"; }

    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment { pos += 1; }

    if pos + 3 > frame.len() { return "no_block_header"; }

    let bh = u32::from_le_bytes([frame[pos], frame[pos+1], frame[pos+2], 0]);
    let block_type = (bh >> 1) & 0x3;

    match block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Reserved",
        _ => "Unknown",
    }
}

fn compare_bitstreams(ref_frame: &[u8], our_frame: &[u8]) {
    let ref_seq_section = get_seq_section(ref_frame);
    let our_seq_section = get_seq_section(our_frame);

    if let (Some(r), Some(o)) = (ref_seq_section, our_seq_section) {
        println!("  Ref sequence section: {:02x?}", r);
        println!("  Our sequence section: {:02x?}", o);

        if r.len() > 2 && o.len() > 2 {
            let ref_count = r[0];
            let our_count = o[0];
            println!("    Ref seq count: {}, Our seq count: {}", ref_count, our_count);

            if ref_count == our_count {
                let ref_mode = r[1];
                let our_mode = o[1];
                println!("    Ref mode: 0x{:02x}, Our mode: 0x{:02x}", ref_mode, our_mode);
            }
        }
    }
}

fn get_seq_section(frame: &[u8]) -> Option<&[u8]> {
    if frame.len() < 7 { return None; }

    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment { pos += 1; }

    if pos + 3 > frame.len() { return None; }

    let bh = u32::from_le_bytes([frame[pos], frame[pos+1], frame[pos+2], 0]);
    let block_type = (bh >> 1) & 0x3;
    let block_size = (bh >> 3) as usize;
    pos += 3;

    if block_type != 2 { return None; }
    if pos + block_size > frame.len() { return None; }

    let block_data = &frame[pos..pos+block_size];

    // Parse literals section
    let lit_type = block_data[0] & 0x03;
    let (lit_size, lit_header_size) = if lit_type == 0 || lit_type == 1 {
        let size_format = (block_data[0] >> 2) & 0x3;
        match size_format {
            0 | 1 => ((block_data[0] >> 3) as usize, 1),
            2 => (((block_data[0] as usize >> 4) | ((block_data[1] as usize) << 4)) & 0xFFF, 2),
            _ => (0, 1),
        }
    } else {
        return None; // Compressed literals
    };

    let seq_start = lit_header_size + lit_size;
    if seq_start < block_data.len() {
        Some(&block_data[seq_start..])
    } else {
        None
    }
}

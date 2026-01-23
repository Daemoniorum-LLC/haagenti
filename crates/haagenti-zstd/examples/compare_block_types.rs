//! Compare what block types we produce vs reference zstd.

fn main() {
    let cases = [
        ("1-seq", b"abcdabcd".as_slice()),
        ("2-seq simple", b"abcdabcdXabcd".as_slice()),
        ("2-seq complex", b"abcdefghXabcdefghYabcd".as_slice()),
        ("3-seq", b"abcdabcdXabcdYabcd".as_slice()),
    ];

    for (name, input) in cases.iter() {
        println!("=== {} ===", name);
        println!(
            "Input: {:?} ({} bytes)",
            std::str::from_utf8(input).unwrap(),
            input.len()
        );

        // Reference compression
        let ref_compressed = zstd::encode_all(*input, 1).unwrap();
        let ref_block_type = get_block_type(&ref_compressed);
        println!(
            "Reference: {} bytes, block_type={}",
            ref_compressed.len(),
            ref_block_type
        );

        // Our compression
        let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
        let our_compressed = compressor.compress(input).unwrap();
        let our_block_type = get_block_type(&our_compressed);
        println!(
            "Ours:      {} bytes, block_type={}",
            our_compressed.len(),
            our_block_type
        );

        // Test decode
        match zstd::decode_all(&our_compressed[..]) {
            Ok(decoded) if decoded == *input => println!("Status: Reference decodes OK"),
            Ok(_) => println!("Status: WRONG DATA"),
            Err(e) => println!("Status: DECODE ERROR ({})", e),
        }
        println!();
    }

    // Now test with different compression levels to find one that produces compressed blocks
    println!("=== Reference at different levels ===");
    let input = b"abcdefghXabcdefghYabcd";
    for level in 1..=5 {
        let compressed = zstd::encode_all(&input[..], level).unwrap();
        let block_type = get_block_type(&compressed);
        println!(
            "Level {}: {} bytes, block_type={}",
            level,
            compressed.len(),
            block_type
        );
    }
}

fn get_block_type(frame: &[u8]) -> &'static str {
    if frame.len() < 7 {
        return "too_short";
    }

    let fhd = frame[4];
    let single_segment = (fhd & 0x20) != 0;
    let mut pos = 5;
    if !single_segment {
        pos += 1;
    }

    if pos + 3 > frame.len() {
        return "no_block_header";
    }

    let bh = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (bh >> 1) & 0x3;

    match block_type {
        0 => "Raw",
        1 => "RLE",
        2 => "Compressed",
        3 => "Reserved",
        _ => "Unknown",
    }
}

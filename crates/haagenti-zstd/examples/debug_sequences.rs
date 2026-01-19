//! Debug what sequences we're actually producing.

fn main() {
    let input = b"abcdefghXabcdefghYabcd";
    println!("Input: {:?} ({} bytes)", std::str::from_utf8(input).unwrap(), input.len());

    // Analyze expected matches
    println!("\n=== Expected Matches ===");
    println!("Position 0-8: \"abcdefghX\" (9 literals)");
    println!("Position 9-16: \"abcdefgh\" matches position 0 (offset=9, length=8)");
    println!("Position 17: \"Y\" (1 literal)");
    println!("Position 18-21: \"abcd\" matches:");
    println!("  - Position 0 (offset=18)");
    println!("  - Position 10 (offset=8)");

    // Get our compressed output
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let compressed = compressor.compress(input).unwrap();

    // Also get reference compression
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();

    println!("\n=== Compression Comparison ===");
    println!("Reference: {} bytes (hex: {:02x?})", ref_compressed.len(), &ref_compressed[..]);
    println!("Ours: {} bytes (hex: {:02x?})", compressed.len(), &compressed[..]);

    // Check what reference does
    println!("\n=== Reference Block Type ===");
    if ref_compressed.len() >= 7 {
        let fhd = ref_compressed[4];
        let single_segment = (fhd & 0x20) != 0;
        let mut pos = 5;
        if !single_segment { pos += 1; }
        let bh = u32::from_le_bytes([ref_compressed[pos], ref_compressed[pos+1], ref_compressed[pos+2], 0]);
        let block_type = (bh >> 1) & 0x3;
        let block_type_name = match block_type { 0 => "Raw", 1 => "RLE", 2 => "Compressed", _ => "Reserved" };
        println!("Block type: {} ({})", block_type, block_type_name);
    }

    // Try decoding with our decoder to see what it produces
    println!("\n=== Our Decoder Output ===");
    match haagenti_zstd::decompress::decompress_frame(&compressed) {
        Ok(decoded) => {
            println!("Decoded: {} bytes", decoded.len());
            if decoded == input {
                println!("Content matches input!");
            } else {
                println!("CONTENT MISMATCH!");
                println!("Expected: {:?}", std::str::from_utf8(input).unwrap());
                println!("Got: {:?}", String::from_utf8_lossy(&decoded));
            }
        }
        Err(e) => println!("Our decode FAILED: {:?}", e),
    }

    // Try reference decoder on our output
    println!("\n=== Reference Decoder on Our Output ===");
    match zstd::decode_all(&compressed[..]) {
        Ok(decoded) => {
            println!("Decoded: {} bytes", decoded.len());
            if decoded == input {
                println!("Content matches input!");
            } else {
                println!("CONTENT MISMATCH!");
            }
        }
        Err(e) => println!("Reference decode FAILED: {}", e),
    }
}

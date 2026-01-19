//! Test 500-byte pattern with both implementations

fn main() {
    let input: Vec<u8> = b"ABCD".iter().cycle().take(500).copied().collect();
    println!("Input: {} bytes", input.len());

    // Compress with reference
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("Reference compressed: {} bytes", ref_compressed.len());
    println!("Hex: {:02x?}", ref_compressed);

    // Extract FSE bitstream from reference
    let fse_bytes = &ref_compressed[16..];
    println!("\nFSE bitstream: {:02x?}", fse_bytes);
    
    // Decode with reference 
    let ref_decoded = zstd::decode_all(&ref_compressed[..]).unwrap();
    println!("\nReference decodes reference: {} bytes, match={}", ref_decoded.len(), ref_decoded == input);

    // Decode with our implementation
    match haagenti_zstd::decompress::decompress_frame(&ref_compressed) {
        Ok(our_decoded) => {
            println!("Our decoder decodes reference: {} bytes, match={}", our_decoded.len(), our_decoded == input);
            if our_decoded != input {
                println!("Mismatch! First 20 bytes:");
                println!("  Expected: {:?}", &input[..20]);
                println!("  Got:      {:?}", &our_decoded[..20.min(our_decoded.len())]);
            }
        }
        Err(e) => println!("Our decoder fails: {:?}", e),
    }

    // Compress with our implementation
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(&input).unwrap();
    println!("\nOur compressed: {} bytes", our_compressed.len());
    println!("Hex: {:02x?}", our_compressed);

    // Cross decode
    match zstd::decode_all(&our_compressed[..]) {
        Ok(d) => println!("Reference decodes ours: {} bytes, match={}", d.len(), d == input),
        Err(e) => println!("Reference fails on ours: {}", e),
    }
}

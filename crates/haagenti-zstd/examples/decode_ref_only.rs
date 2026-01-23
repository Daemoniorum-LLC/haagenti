//! Decode using only reference zstd to understand what it produces.

fn main() {
    let input = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD";
    println!("Input: {} bytes", input.len());
    println!("Input: {:?}", String::from_utf8_lossy(&input[..20]));

    // Compress with reference
    let compressed = zstd::encode_all(&input[..], 1).unwrap();
    println!("\nReference compressed: {} bytes", compressed.len());
    println!("Hex: {:02x?}", &compressed);

    // Decode with reference
    let decompressed = zstd::decode_all(&compressed[..]).unwrap();
    println!("\nReference decompressed: {} bytes", decompressed.len());
    println!("Match: {}", decompressed == input);

    // Now decode our output with reference
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();
    println!("\nOur compressed: {} bytes", our_compressed.len());
    println!("Hex: {:02x?}", &our_compressed);

    match zstd::decode_all(&our_compressed[..]) {
        Ok(d) => {
            println!("Reference decodes ours: {} bytes", d.len());
            if d == input {
                println!("MATCH!");
            } else {
                println!("MISMATCH!");
                println!("Expected: {:?}", String::from_utf8_lossy(&input[..20]));
                println!(
                    "Got:      {:?}",
                    String::from_utf8_lossy(&d[..20.min(d.len())])
                );
            }
        }
        Err(e) => {
            println!("Reference FAILS to decode ours: {}", e);
        }
    }

    // Decode our output with our decoder
    match haagenti_zstd::decompress::decompress_frame(&our_compressed) {
        Ok(d) => {
            println!("\nOur decoder decodes ours: {} bytes", d.len());
            if d == input {
                println!("MATCH!");
            } else {
                println!("MISMATCH!");
            }
        }
        Err(e) => {
            println!("Our decoder FAILS: {:?}", e);
        }
    }
}

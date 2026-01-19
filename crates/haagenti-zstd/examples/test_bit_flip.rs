//! Test what the bit flip produces.

fn main() {
    let input = b"abcdefghXabcdefghYabcd";
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    println!("Original compressed: {:02x?}", our_compressed);
    println!("Byte 6 original: 0x{:02x} = {:08b}", our_compressed[6], our_compressed[6]);

    // Flip bit 2
    let mut modified = our_compressed.clone();
    modified[6] ^= 0x04;
    println!("Byte 6 modified: 0x{:02x} = {:08b}", modified[6], modified[6]);

    // Decode modified
    match zstd::decode_all(&modified[..]) {
        Ok(decoded) => {
            println!("\nDecoded {} bytes: {:?}", decoded.len(), String::from_utf8_lossy(&decoded));
            println!("Expected: {:?}", std::str::from_utf8(input).unwrap());
            println!("Match: {}", decoded == input);
        }
        Err(e) => {
            println!("Failed: {}", e);
        }
    }

    // Analyze block type
    println!("\nBlock header analysis:");
    let bh_orig = u32::from_le_bytes([our_compressed[6], our_compressed[7], our_compressed[8], 0]);
    let bh_mod = u32::from_le_bytes([modified[6], modified[7], modified[8], 0]);

    println!("Original: is_last={}, type={}, size={}",
             bh_orig & 1, (bh_orig >> 1) & 3, bh_orig >> 3);
    println!("Modified: is_last={}, type={}, size={}",
             bh_mod & 1, (bh_mod >> 1) & 3, bh_mod >> 3);
}

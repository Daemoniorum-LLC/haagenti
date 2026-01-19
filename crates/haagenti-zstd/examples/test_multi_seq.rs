//! Test actual multi-sequence encoding with debug output

use std::io::Cursor;

fn main() {
    // Create input that produces multiple sequences
    let mut input = Vec::new();
    for _ in 0..50 { input.extend_from_slice(b"ABCDEFGH"); }
    input.extend_from_slice(b"XXX");
    for _ in 0..30 { input.extend_from_slice(b"IJKLMNOP"); }
    input.extend_from_slice(b"YYY");
    for _ in 0..20 { input.extend_from_slice(b"QRSTUVWX"); }
    
    println!("Input size: {} bytes", input.len());
    
    // Compress
    use haagenti_core::{CompressionLevel, Compressor};
    use haagenti_zstd::ZstdCompressor;
    
    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let compressed = compressor.compress(&input).unwrap();
    
    println!("Compressed: {} bytes", compressed.len());
    
    // Parse block
    let block_header = compressed[6] as u32 | ((compressed[7] as u32) << 8) | ((compressed[8] as u32) << 16);
    let block_type = (block_header >> 1) & 0x03;
    let block_size = block_header >> 3;
    println!("Block type: {} (0=Raw, 1=RLE, 2=Compressed)", block_type);
    println!("Block size: {}", block_size);
    
    // Show the sequence section bytes
    if block_type == 2 {
        let block_start = 9;
        let lit_header = compressed[block_start];
        let lit_size_type = (lit_header >> 2) & 0x03;
        
        let (lit_header_size, lit_size) = match lit_size_type {
            0 => (1, (lit_header >> 3) as usize),
            1 => (2, ((lit_header >> 4) as usize) | ((compressed[block_start + 1] as usize) << 4)),
            _ => (3, 0),
        };
        
        println!("Literals: header_size={}, content_size={}", lit_header_size, lit_size);
        
        let seq_start = block_start + lit_header_size + lit_size;
        let seq_count = compressed[seq_start];
        let mode_byte = if seq_count > 0 { compressed[seq_start + 1] } else { 0 };
        
        println!("Sequences start at offset {}", seq_start);
        println!("Sequence count: {}", seq_count);
        println!("Mode byte: 0x{:02x} (LL={}, OF={}, ML={})", 
                 mode_byte, mode_byte & 3, (mode_byte >> 2) & 3, (mode_byte >> 4) & 3);
        
        // Show bitstream bytes
        let bitstream_start = seq_start + 2;
        let bitstream = &compressed[bitstream_start..];
        println!("Bitstream ({} bytes): {:02x?}", bitstream.len(), bitstream);
    }
    
    // Try reference zstd
    match zstd::decode_all(Cursor::new(&compressed)) {
        Ok(decoded) if decoded == input => {
            println!("\nSUCCESS! Reference zstd decoded {} bytes", decoded.len());
        }
        Ok(decoded) => {
            println!("\nMISMATCH! {} vs {} bytes", input.len(), decoded.len());
        }
        Err(e) => println!("\nFAILED: {:?}", e),
    }
    
    // Our roundtrip
    use haagenti_core::Decompressor;
    let decompressor = haagenti_zstd::ZstdDecompressor;
    match decompressor.decompress(&compressed) {
        Ok(decoded) if decoded == input => println!("Our roundtrip: OK"),
        _ => println!("Our roundtrip: FAILED"),
    }
}

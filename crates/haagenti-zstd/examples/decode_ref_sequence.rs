//! Decode the reference bitstream to see what sequence it actually contains.

use haagenti_zstd::decompress::decompress_frame;

fn main() {
    // Reference compressed frame for "ABCD" x 25
    let ref_frame: &[u8] = &[
        0x28, 0xb5, 0x2f, 0xfd, // Magic
        0x00, // FHD
        0x48, // Window descriptor
        0x55, 0x00, 0x00, // Block header (Last=1, Type=2, Size=10)
        0x20, // Literals header (Raw, size=4)
        0x41, 0x42, 0x43, 0x44, // Literals "ABCD"
        0x01, 0x00, // Sequences (count=1, mode=0x00)
        0xfd, 0xe4, 0x88, // FSE bitstream
    ];

    println!("Reference frame: {:02x?}", ref_frame);

    // Decode with our decoder
    match decompress_frame(ref_frame) {
        Ok(decompressed) => {
            println!("Our decoder output: {} bytes", decompressed.len());
            println!("Content: {:?}", String::from_utf8_lossy(&decompressed));

            // Check if it matches expected
            let expected = b"ABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCDABCD";
            println!("Matches expected: {}", decompressed == expected);
        }
        Err(e) => {
            println!("Our decoder failed: {:?}", e);
        }
    }

    // Also decode with reference
    match zstd::decode_all(&ref_frame[..]) {
        Ok(decompressed) => {
            println!("\nReference decoder output: {} bytes", decompressed.len());
            println!("Content: {:?}", String::from_utf8_lossy(&decompressed));
        }
        Err(e) => {
            println!("Reference decoder failed: {}", e);
        }
    }
}

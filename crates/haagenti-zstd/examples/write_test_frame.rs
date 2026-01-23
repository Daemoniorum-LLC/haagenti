//! Write our compressed frame to a file for external testing.

fn main() {
    let input = b"abcdefghXabcdefghYabcd";
    let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
    let our_compressed = compressor.compress(input).unwrap();

    println!(
        "Input: {:?} ({} bytes)",
        std::str::from_utf8(input).unwrap(),
        input.len()
    );
    println!("Compressed: {} bytes", our_compressed.len());

    print!("Hex: ");
    for b in &our_compressed {
        print!("{:02x} ", b);
    }
    println!();

    std::fs::write("/tmp/our_frame.zst", &our_compressed).unwrap();
    println!("Written to /tmp/our_frame.zst");

    // Also create a reference frame
    let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
    std::fs::write("/tmp/ref_frame.zst", &ref_compressed).unwrap();
    println!(
        "Reference written to /tmp/ref_frame.zst ({} bytes)",
        ref_compressed.len()
    );
}

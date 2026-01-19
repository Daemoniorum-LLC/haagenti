//! Test cross-compatibility with various sizes

fn main() {
    let test_cases = [
        ("100 bytes", 100),
        ("500 bytes", 500),
        ("1000 bytes", 1000),
        ("4000 bytes", 4000),
        ("65535 bytes", 65535),
    ];

    let mut all_passed = true;

    for (name, size) in test_cases {
        let input: Vec<u8> = b"ABCD".iter().cycle().take(size).copied().collect();

        // Reference compression/decompression
        let ref_compressed = zstd::encode_all(&input[..], 1).unwrap();
        let ref_decoded = zstd::decode_all(&ref_compressed[..]).unwrap();
        let ref_ok = ref_decoded == input;

        // Our decompression of reference
        let our_decode_ref = haagenti_zstd::decompress::decompress_frame(&ref_compressed)
            .map(|d| d == input)
            .unwrap_or(false);

        // Our compression
        let compressor = haagenti_zstd::compress::SpeculativeCompressor::new();
        let our_compressed = compressor.compress(&input).unwrap();

        // Reference decompression of ours
        let ref_decode_ours = zstd::decode_all(&our_compressed[..])
            .map(|d| d == input)
            .unwrap_or(false);

        // Our decompression of ours
        let our_decode_ours = haagenti_zstd::decompress::decompress_frame(&our_compressed)
            .map(|d| d == input)
            .unwrap_or(false);

        let passed = ref_ok && our_decode_ref && ref_decode_ours && our_decode_ours;
        all_passed = all_passed && passed;

        println!("{:15} ref_compressed={:6} our_compressed={:6} | ref_ok={} our_decode_ref={} ref_decode_ours={} our_decode_ours={}",
                 name, ref_compressed.len(), our_compressed.len(),
                 ref_ok, our_decode_ref, ref_decode_ours, our_decode_ours);
    }

    println!("\nAll tests passed: {}", all_passed);
}

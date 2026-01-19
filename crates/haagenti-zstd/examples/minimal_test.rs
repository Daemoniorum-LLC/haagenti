//! Minimal test: manually construct a valid zstd frame and verify

use std::io::Cursor;

fn main() {
    // Test 1: Construct a frame with just literals (no sequences)
    test_raw_block();

    // Test 2: Minimal compressed block with 1 sequence
    test_one_sequence();

    // Test 3: Minimal compressed block with 2 sequences
    test_two_sequences();

    // Test 4: Use reference zstd to compress, then compare
    test_reference_compression();
}

fn test_raw_block() {
    println!("\n=== Test: Raw block (no sequences) ===");
    let input = b"hello world";

    // Construct minimal frame:
    // Magic (4) + FHD (1) + Window (1) + Block header (3) + data
    let mut frame = Vec::new();

    // Magic number
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // Frame Header Descriptor: no checksum, no dict, no FCS, window descriptor present
    frame.push(0x00);

    // Window Descriptor (log2(window) = 10 + 8 = 18, so window = 256KB)
    frame.push(0x48);

    // Block header: last=1, type=0 (raw), size=11
    let block_header = (11u32 << 3) | 0x01; // size=11, type=0, last=1
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    // Block content (raw data)
    frame.extend_from_slice(input);

    println!("Frame hex: {:02x?}", frame);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == input => println!("Result: OK"),
        Ok(decoded) => println!("Result: MISMATCH - got {:?}", String::from_utf8_lossy(&decoded)),
        Err(e) => println!("Result: FAILED - {:?}", e),
    }
}

fn test_one_sequence() {
    println!("\n=== Test: Compressed block with 1 sequence ===");

    // Input: "abcdabcd" - 4 literals "abcd" + match(offset=4, length=4)
    let expected = b"abcdabcd";

    let mut frame = Vec::new();

    // Magic
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // FHD
    frame.push(0x00);

    // Window
    frame.push(0x48);

    // Build block content first to get size
    let mut block = Vec::new();

    // Literals section: Raw, 4 bytes "abcd"
    // Header: size << 3 | 0 = 4 << 3 = 0x20
    block.push(0x20);
    block.extend_from_slice(b"abcd");

    // Sequences section
    // Count: 1
    block.push(0x01);

    // Mode: 0x00 = all predefined
    block.push(0x00);

    // Build FSE bitstream for 1 sequence:
    // Seq 0: LL=4 (code=4), OF=4+3=7 (wait, offset=4 means actual offset 4)
    // Actually for 1 sequence, we just need:
    // - Initial states for LL, OF, ML
    // - Extra bits for the sequence

    // Let me use our actual encoder to build this correctly
    use haagenti_zstd::block::Sequence;
    use haagenti_zstd::compress::EncodedSequence;
    use haagenti_zstd::fse::{FseBitWriter, InterleavedTansEncoder};

    // Offset 4 matches initial repeat_offset_2, so encoded as 2
    let seq = Sequence::new(4, 2, 4);  // LL=4, OF_value=2 (repeat), ML=4
    let enc = EncodedSequence::from_sequence(&seq);

    println!("  Encoded: ll_code={}, of_code={}, ml_code={}",
             enc.ll_code, enc.of_code, enc.ml_code);
    println!("  Extras: ll={}({}b), of={}({}b), ml={}({}b)",
             enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits);

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    // Init with this sequence
    tans.init_states(enc.ll_code, enc.of_code, enc.ml_code);
    let (ll_state, of_state, ml_state) = tans.get_states();

    println!("  States: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    // Build bitstream: just extra bits + states
    let mut bits = FseBitWriter::new();

    // Extra bits for the only sequence
    if enc.ll_bits > 0 { bits.write_bits(enc.ll_extra, enc.ll_bits); }
    if enc.ml_bits > 0 { bits.write_bits(enc.ml_extra, enc.ml_bits); }
    if enc.of_bits > 0 { bits.write_bits(enc.of_extra, enc.of_bits); }

    // States (ML, OF, LL order for correct MSB reading)
    bits.write_bits(ml_state, ml_log);
    bits.write_bits(of_state, of_log);
    bits.write_bits(ll_state, ll_log);

    let bitstream = bits.finish();
    println!("  Bitstream: {:02x?}", bitstream);

    block.extend_from_slice(&bitstream);

    // Block header
    let block_size = block.len() as u32;
    let block_header = (block_size << 3) | 0x05; // size, type=2 (compressed), last=1
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    // Block content
    frame.extend_from_slice(&block);

    println!("  Frame hex: {:02x?}", frame);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == expected => println!("  Result: OK"),
        Ok(decoded) => println!("  Result: MISMATCH - got {:?}", String::from_utf8_lossy(&decoded)),
        Err(e) => println!("  Result: FAILED - {:?}", e),
    }
}

fn test_two_sequences() {
    println!("\n=== Test: Compressed block with 2 sequences ===");

    // Same as the failing case
    let expected = b"abcdefghXabcdefghYabcd";

    let mut frame = Vec::new();

    // Magic
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // FHD
    frame.push(0x00);

    // Window
    frame.push(0x48);

    // Build block content
    let mut block = Vec::new();

    // Literals: 10 bytes "abcdefghXY"
    // Header: 10 << 3 = 0x50
    block.push(0x50);
    block.extend_from_slice(b"abcdefghXY");

    // Sequences section
    block.push(0x02);  // count = 2
    block.push(0x00);  // mode = predefined

    // Build FSE bitstream for 2 sequences
    use haagenti_zstd::block::Sequence;
    use haagenti_zstd::compress::EncodedSequence;
    use haagenti_zstd::fse::{FseBitWriter, InterleavedTansEncoder};

    // Seq 0: LL=9, OF_value=12 (offset 9 + 3), ML=8
    // Seq 1: LL=1, OF_value=1 (repeat offset), ML=4
    let sequences = vec![
        Sequence::new(9, 12, 8),
        Sequence::new(1, 1, 4),
    ];

    let encoded: Vec<_> = sequences.iter()
        .map(|s| EncodedSequence::from_sequence(s))
        .collect();

    for (i, enc) in encoded.iter().enumerate() {
        println!("  Seq {}: ll_code={}, of_code={}, ml_code={}",
                 i, enc.ll_code, enc.of_code, enc.ml_code);
        println!("       extras: ll={}({}b), of={}({}b), ml={}({}b)",
                 enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits);
    }

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    let last = &encoded[1];
    tans.init_states(last.ll_code, last.of_code, last.ml_code);

    // Encode seq 0 (only non-last sequence)
    let fse_bits = tans.encode_sequence(encoded[0].ll_code, encoded[0].of_code, encoded[0].ml_code);
    let (ll_state, of_state, ml_state) = tans.get_states();

    println!("  FSE bits for seq 0: LL({},{}) OF({},{}) ML({},{})",
             fse_bits[0].0, fse_bits[0].1,
             fse_bits[1].0, fse_bits[1].1,
             fse_bits[2].0, fse_bits[2].1);
    println!("  Final states: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    // Build bitstream
    let mut bits = FseBitWriter::new();

    // Seq 0: extras + FSE bits
    let seq0 = &encoded[0];
    if seq0.ll_bits > 0 { bits.write_bits(seq0.ll_extra, seq0.ll_bits); }
    if seq0.ml_bits > 0 { bits.write_bits(seq0.ml_extra, seq0.ml_bits); }
    if seq0.of_bits > 0 { bits.write_bits(seq0.of_extra, seq0.of_bits); }

    bits.write_bits(fse_bits[0].0, fse_bits[0].1);  // LL
    bits.write_bits(fse_bits[2].0, fse_bits[2].1);  // ML
    bits.write_bits(fse_bits[1].0, fse_bits[1].1);  // OF

    // Seq 1 (last): just extras
    if last.ll_bits > 0 { bits.write_bits(last.ll_extra, last.ll_bits); }
    if last.ml_bits > 0 { bits.write_bits(last.ml_extra, last.ml_bits); }
    if last.of_bits > 0 { bits.write_bits(last.of_extra, last.of_bits); }

    // States
    bits.write_bits(ml_state, ml_log);
    bits.write_bits(of_state, of_log);
    bits.write_bits(ll_state, ll_log);

    let bitstream = bits.finish();
    println!("  Bitstream: {:02x?}", bitstream);

    block.extend_from_slice(&bitstream);

    // Block header
    let block_size = block.len() as u32;
    let block_header = (block_size << 3) | 0x05;
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    // Block content
    frame.extend_from_slice(&block);

    println!("  Frame hex: {:02x?}", frame);
    println!("  Block size: {}", block_size);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == expected => println!("  Result: OK"),
        Ok(decoded) => println!("  Result: MISMATCH - got {:?}", String::from_utf8_lossy(&decoded)),
        Err(e) => println!("  Result: FAILED - {:?}", e),
    }
}

fn test_reference_compression() {
    println!("\n=== Test: Reference zstd compression ===");

    let input = b"abcdefghXabcdefghYabcd";

    // Compress with reference zstd
    let compressed = zstd::encode_all(Cursor::new(input.as_slice()), 1).unwrap();

    println!("  Reference compressed: {:02x?}", compressed);
    println!("  Reference size: {} bytes", compressed.len());

    // Verify it decodes
    let decoded = zstd::decode_all(Cursor::new(&compressed)).unwrap();
    assert_eq!(decoded, input);
    println!("  Reference decode: OK");

    // Now compare with our compression
    use haagenti_core::{CompressionLevel, Compressor};
    use haagenti_zstd::ZstdCompressor;

    let compressor = ZstdCompressor::with_level(CompressionLevel::Fast);
    let our_compressed = compressor.compress(input).unwrap();

    println!("\n  Our compressed: {:02x?}", our_compressed);
    println!("  Our size: {} bytes", our_compressed.len());

    // Show byte-by-byte comparison
    println!("\n  Comparison:");
    let max_len = compressed.len().max(our_compressed.len());
    for i in 0..max_len {
        let ref_byte = compressed.get(i).map(|b| format!("{:02x}", b)).unwrap_or_else(|| "  ".to_string());
        let our_byte = our_compressed.get(i).map(|b| format!("{:02x}", b)).unwrap_or_else(|| "  ".to_string());
        let marker = if compressed.get(i) == our_compressed.get(i) { " " } else { "*" };
        println!("    {:3}: ref={} our={} {}", i, ref_byte, our_byte, marker);
    }
}

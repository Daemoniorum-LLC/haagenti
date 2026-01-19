//! Verify the exact bit order expected by reference zstd decoder

use std::io::Cursor;

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::fse::{FseBitWriter, InterleavedTansEncoder};

fn main() {
    println!("=== Verifying Bit Order ===\n");

    // The failing case: 2 sequences with repeat offset
    // Seq 0: LL=9, OF_value=12, ML=8  (offset 9 encoded as 12)
    // Seq 1: LL=1, OF_value=1, ML=4   (repeat offset)

    let sequences = vec![
        Sequence::new(9, 12, 8),
        Sequence::new(1, 1, 4),
    ];

    let encoded: Vec<_> = sequences.iter()
        .map(|s| EncodedSequence::from_sequence(s))
        .collect();

    println!("Encoded sequences:");
    for (i, enc) in encoded.iter().enumerate() {
        println!("  Seq {}: ll_code={}, of_code={}, ml_code={}",
                 i, enc.ll_code, enc.of_code, enc.ml_code);
        println!("       extras: ll={}({}b), of={}({}b), ml={}({}b)",
                 enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits);
    }

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    println!("\nAccuracy logs: LL={}, OF={}, ML={}", ll_log, of_log, ml_log);

    // Try BOTH bit orders and see which one the reference decoder accepts

    // Order 1: LL, ML, OF (our current order)
    let bitstream_llmlof = build_bitstream_order(&encoded, &mut tans, "LL,ML,OF",
        |bits, seq| {
            if seq.ll_bits > 0 { bits.write_bits(seq.ll_extra, seq.ll_bits); }
            if seq.ml_bits > 0 { bits.write_bits(seq.ml_extra, seq.ml_bits); }
            if seq.of_bits > 0 { bits.write_bits(seq.of_extra, seq.of_bits); }
        }
    );

    // Order 2: OF, ML, LL (reference zstd order based on decoder source)
    let mut tans2 = InterleavedTansEncoder::new_predefined();
    let bitstream_ofmlll = build_bitstream_order(&encoded, &mut tans2, "OF,ML,LL",
        |bits, seq| {
            if seq.of_bits > 0 { bits.write_bits(seq.of_extra, seq.of_bits); }
            if seq.ml_bits > 0 { bits.write_bits(seq.ml_extra, seq.ml_bits); }
            if seq.ll_bits > 0 { bits.write_bits(seq.ll_extra, seq.ll_bits); }
        }
    );

    // Build full frames with each bitstream and test
    println!("\n=== Testing full decompression ===");

    let expected = b"abcdefghXabcdefghYabcd";
    let literals = b"abcdefghXY";

    test_frame("LL,ML,OF order", literals, &bitstream_llmlof, expected);
    test_frame("OF,ML,LL order", literals, &bitstream_ofmlll, expected);
}

fn build_bitstream_order<F>(
    encoded: &[EncodedSequence],
    tans: &mut InterleavedTansEncoder,
    order_name: &str,
    write_extras: F,
) -> Vec<u8>
where
    F: Fn(&mut FseBitWriter, &EncodedSequence),
{
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    let last_idx = encoded.len() - 1;
    let last_seq = &encoded[last_idx];

    tans.init_states(last_seq.ll_code, last_seq.of_code, last_seq.ml_code);

    let mut fse_bits_per_seq: Vec<[(u32, u8); 3]> = Vec::with_capacity(last_idx);
    for i in (0..last_idx).rev() {
        let seq = &encoded[i];
        let fse_bits = tans.encode_sequence(seq.ll_code, seq.of_code, seq.ml_code);
        fse_bits_per_seq.insert(0, fse_bits);
    }

    let (ll_state, of_state, ml_state) = tans.get_states();

    println!("\nBitstream with {} order:", order_name);
    println!("  Final states: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    let mut bits = FseBitWriter::new();

    for i in 0..last_idx {
        let seq = &encoded[i];

        // Write extra bits in specified order
        write_extras(&mut bits, seq);

        // Write FSE update bits: LL, ML, OF order (this is the state update order)
        let [ll_fse, of_fse, ml_fse] = fse_bits_per_seq[i];
        bits.write_bits(ll_fse.0, ll_fse.1);
        bits.write_bits(ml_fse.0, ml_fse.1);
        bits.write_bits(of_fse.0, of_fse.1);
    }

    // Last sequence: just extras (no FSE update)
    write_extras(&mut bits, last_seq);

    // Write states: ML, OF, LL order (so decoder reads LL, OF, ML)
    bits.write_bits(ml_state, ml_log);
    bits.write_bits(of_state, of_log);
    bits.write_bits(ll_state, ll_log);

    let result = bits.finish();
    println!("  Bitstream: {:02x?}", result);
    result
}

fn test_frame(name: &str, literals: &[u8], bitstream: &[u8], expected: &[u8]) {
    println!("\nTesting {}:", name);

    let mut frame = Vec::new();

    // Magic
    frame.extend_from_slice(&0xFD2FB528u32.to_le_bytes());

    // FHD
    frame.push(0x00);

    // Window
    frame.push(0x48);

    // Build block content
    let mut block = Vec::new();

    // Literals header: Raw, 10 bytes
    block.push(0x50);
    block.extend_from_slice(literals);

    // Sequence count
    block.push(0x02);

    // Mode byte (predefined)
    block.push(0x00);

    // FSE bitstream
    block.extend_from_slice(bitstream);

    // Block header
    let block_size = block.len() as u32;
    let block_header = (block_size << 3) | 0x05;
    frame.extend_from_slice(&block_header.to_le_bytes()[0..3]);

    frame.extend_from_slice(&block);

    println!("  Frame: {:02x?}", frame);

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) if decoded == expected => println!("  Result: OK!"),
        Ok(decoded) => {
            println!("  Result: MISMATCH");
            println!("    Expected: {:?}", String::from_utf8_lossy(expected));
            println!("    Got: {:?}", String::from_utf8_lossy(&decoded));
        }
        Err(e) => println!("  Result: FAILED - {:?}", e),
    }
}

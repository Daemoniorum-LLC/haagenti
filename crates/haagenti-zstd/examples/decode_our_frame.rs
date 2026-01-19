//! Try to decode our compressed frame with our own decoder

use haagenti_zstd::decompress::decompress_frame;
use std::io::Cursor;

fn main() {
    println!("=== Decoding Our Frame with Our Decoder ===\n");

    // Our compressed frame for "abcdefghXabcdefghYabcd"
    let our_frame = vec![
        0x28, 0xb5, 0x2f, 0xfd, // Magic
        0x00, // FHD
        0x48, // Window
        0x95, 0x00, 0x00, // Block header: size=18, type=2, last=1
        0x50, // Literal header: Raw, 10 bytes
        0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x58, 0x59, // Literals
        0x02, // Seq count
        0x00, // Mode (predefined)
        0x14, 0x01, 0x10, 0xe5, 0x08, // Bitstream
    ];

    let expected = b"abcdefghXabcdefghYabcd";

    println!("Frame hex: {:02x?}", our_frame);
    println!("Expected output: {:?}", String::from_utf8_lossy(expected));

    // Try our decoder
    match decompress_frame(&our_frame) {
        Ok(decoded) => {
            println!("\nOur decoder result: {:?}", String::from_utf8_lossy(&decoded));
            if decoded == expected {
                println!("OUR DECODER: OK");
            } else {
                println!("OUR DECODER: MISMATCH");
                println!("  Expected {} bytes, got {} bytes", expected.len(), decoded.len());
            }
        }
        Err(e) => {
            println!("\nOur decoder FAILED: {:?}", e);
        }
    }

    // Try reference decoder
    match zstd::decode_all(Cursor::new(&our_frame)) {
        Ok(decoded) => {
            println!("\nReference decoder result: {:?}", String::from_utf8_lossy(&decoded));
            if decoded == expected {
                println!("REFERENCE DECODER: OK");
            } else {
                println!("REFERENCE DECODER: MISMATCH");
            }
        }
        Err(e) => {
            println!("\nReference decoder FAILED: {:?}", e);
        }
    }

    // Let's also try with a simpler case - raw block
    println!("\n=== Testing Raw Block ===");
    let raw_frame = vec![
        0x28, 0xb5, 0x2f, 0xfd, // Magic
        0x00, // FHD
        0x48, // Window
        0xb1, 0x00, 0x00, // Block header: size=22, type=0 (raw), last=1
        // Raw data: "abcdefghXabcdefghYabcd"
        0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x58,
        0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x59,
        0x61, 0x62, 0x63, 0x64,
    ];

    match decompress_frame(&raw_frame) {
        Ok(decoded) => {
            println!("Our decoder (raw block): {:?}", String::from_utf8_lossy(&decoded));
            if decoded == expected {
                println!("RAW BLOCK: OK");
            } else {
                println!("RAW BLOCK: MISMATCH");
            }
        }
        Err(e) => println!("Raw block FAILED: {:?}", e),
    }

    // Let's check if 1-sequence compressed block works with reference
    println!("\n=== Testing 1-Sequence Case ===");

    // For "abcdabcd" (8 bytes), we have:
    // Literals: "abcd" (4 bytes)
    // Seq 0: LL=4, offset=4 (matches repeat_offset_2), ML=4
    // offset_value = 2 (repeat_offset_2)
    // ll_code=4, of_code=1, ml_code=1

    use haagenti_zstd::block::Sequence;
    use haagenti_zstd::compress::EncodedSequence;
    use haagenti_zstd::fse::{FseBitWriter, InterleavedTansEncoder};

    let seq = Sequence::new(4, 2, 4);
    let enc = EncodedSequence::from_sequence(&seq);
    println!("1 seq: ll_code={}, of_code={}, ml_code={}",
             enc.ll_code, enc.of_code, enc.ml_code);
    println!("       extras: ll={}({}b), of={}({}b), ml={}({}b)",
             enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits);

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    tans.init_states(enc.ll_code, enc.of_code, enc.ml_code);
    let (ll_state, of_state, ml_state) = tans.get_states();

    println!("States: LL={}, OF={}, ML={}", ll_state, of_state, ml_state);

    let mut bits = FseBitWriter::new();

    // For single sequence, just write extras then states
    if enc.ll_bits > 0 { bits.write_bits(enc.ll_extra, enc.ll_bits); }
    if enc.ml_bits > 0 { bits.write_bits(enc.ml_extra, enc.ml_bits); }
    if enc.of_bits > 0 { bits.write_bits(enc.of_extra, enc.of_bits); }

    bits.write_bits(ml_state, ml_log);
    bits.write_bits(of_state, of_log);
    bits.write_bits(ll_state, ll_log);

    let bitstream = bits.finish();
    println!("Bitstream: {:02x?}", bitstream);

    let frame_1seq = vec![
        0x28, 0xb5, 0x2f, 0xfd, // Magic
        0x00, // FHD
        0x48, // Window
        // Block header: type=2, last=1, size = 1 + 4 + 1 + 1 + bitstream.len()
        ((1 + 4 + 1 + 1 + bitstream.len() as u32) << 3 | 0x05) as u8,
        (((1 + 4 + 1 + 1 + bitstream.len() as u32) << 3 | 0x05) >> 8) as u8,
        (((1 + 4 + 1 + 1 + bitstream.len() as u32) << 3 | 0x05) >> 16) as u8,
        0x20, // Literal header: Raw, 4 bytes
        0x61, 0x62, 0x63, 0x64, // Literals "abcd"
        0x01, // Seq count
        0x00, // Mode (predefined)
    ];

    let mut frame = frame_1seq.clone();
    frame.extend_from_slice(&bitstream);

    println!("Frame (1 seq): {:02x?}", frame);

    let expected_1seq = b"abcdabcd";

    match zstd::decode_all(Cursor::new(&frame)) {
        Ok(decoded) => {
            println!("Reference decoder (1 seq): {:?}", String::from_utf8_lossy(&decoded));
            if decoded == expected_1seq {
                println!("1-SEQ: OK!");
            } else {
                println!("1-SEQ: MISMATCH");
            }
        }
        Err(e) => println!("1-SEQ FAILED: {:?}", e),
    }
}

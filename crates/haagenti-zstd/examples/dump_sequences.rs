//! Dump the actual sequences our encoder generates

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::{CompressContext, EncodedSequence};
use haagenti_zstd::block::Sequence;

fn main() {
    // 2-sequence input
    let mut input = Vec::new();
    for _ in 0..10 { input.extend_from_slice(b"ABCD"); }
    input.extend_from_slice(b"XXXX");
    for _ in 0..5 { input.extend_from_slice(b"EFGH"); }

    println!("Input: {} bytes", input.len());
    println!("Pattern: ABCD*10 + XXXX + EFGH*5");

    // Let's manually trace what sequences make sense
    println!("\n=== Expected Sequences ===");
    println!("First 4 bytes: ABCD (must be literal)");
    println!("Bytes 4-40: ABCD repeated - match offset=4, length=36");
    println!("Bytes 40-44: XXXX (literal)");
    println!("Bytes 44-48: EFGH (literal)");
    println!("Bytes 48-64: EFGH repeated - match offset=4, length=16");

    // What sequences should look like:
    // Seq 0: literal_length=4, offset_value=7 (actual offset 4 = offset_value - 3), match_length=36
    //   offset_value = actual_offset + 3 = 4 + 3 = 7
    //   offset_code = floor(log2(7)) = 2
    //   offset_extra = 7 - 4 = 3 (2 bits)
    // Seq 1: literal_length=8 (XXXX + EFGH), offset_value=7, match_length=16

    println!("\n=== Expected Encoding ===");
    println!("Seq 0: ll=4, offset_value=7 (code=2, extra=3), ml=36 (code=33)");
    println!("Seq 1: ll=8, offset_value=7 (code=2, extra=3), ml=16 (code=13)");

    // Test encoding with our functions
    let seq0 = Sequence::new(4, 7, 36);
    let seq1 = Sequence::new(8, 7, 16);

    let enc0 = EncodedSequence::from_sequence(&seq0);
    let enc1 = EncodedSequence::from_sequence(&seq1);

    println!("\n=== Our Encoding of Correct Sequences ===");
    println!("Seq 0: ll_code={}, of_code={} (extra={}, bits={}), ml_code={} (extra={}, bits={})",
             enc0.ll_code, enc0.of_code, enc0.of_extra, enc0.of_bits, enc0.ml_code, enc0.ml_extra, enc0.ml_bits);
    println!("Seq 1: ll_code={}, of_code={} (extra={}, bits={}), ml_code={} (extra={}, bits={})",
             enc1.ll_code, enc1.of_code, enc1.of_extra, enc1.of_bits, enc1.ml_code, enc1.ml_extra, enc1.ml_bits);

    // Now let's see what our compressor actually produces
    println!("\n=== Actual Compression ===");
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let compressed = ctx.compress(&input).unwrap();

    println!("Compressed: {} bytes", compressed.len());
    println!("Hex: {:02x?}", &compressed);

    // Verify it works
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(decoded) => {
            if decoded == input {
                println!("Reference decode: SUCCESS");
            } else {
                println!("Reference decode: MISMATCH (decoded {} bytes)", decoded.len());
            }
        }
        Err(e) => println!("Reference decode: FAILED - {}", e),
    }

    // Try encoding just those two correct sequences manually
    println!("\n=== Manual Encoding Test ===");
    use haagenti_zstd::compress::encode_sequences_fse;
    let correct_seqs = vec![seq0, seq1];
    let mut manual_output = Vec::new();
    encode_sequences_fse(&correct_seqs, &mut manual_output).unwrap();
    println!("Manual FSE output: {:02x?}", manual_output);

    // Now decode the manual output to verify
    println!("\n=== Verify Manual Encoding ===");
    decode_manual_sequences(&manual_output[2..]); // Skip count and mode bytes
}

fn decode_manual_sequences(fse_bits: &[u8]) {
    use haagenti_zstd::fse::{
        FseTable, FseDecoder, BitReader,
        LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG,
        MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
        OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG,
    };

    let ll_table = FseTable::from_predefined(&LITERAL_LENGTH_DEFAULT_DISTRIBUTION, LITERAL_LENGTH_ACCURACY_LOG).unwrap();
    let of_table = FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(&MATCH_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG).unwrap();

    let mut bits = BitReader::new(fse_bits);
    bits.init_from_end().unwrap();

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    ll_decoder.init_state(&mut bits).unwrap();
    of_decoder.init_state(&mut bits).unwrap();
    ml_decoder.init_state(&mut bits).unwrap();

    println!("Initial states: LL={}, OF={}, ML={}",
             ll_decoder.state(), of_decoder.state(), ml_decoder.state());
    println!("Initial symbols: LL={}, OF={}, ML={}",
             ll_decoder.peek_symbol(), of_decoder.peek_symbol(), ml_decoder.peek_symbol());

    bits.switch_to_lsb_mode().unwrap();

    for i in 0..2 {
        let is_last = i == 1;
        println!("\nSeq {}: codes LL={}, OF={}, ML={}",
                 i, ll_decoder.peek_symbol(), of_decoder.peek_symbol(), ml_decoder.peek_symbol());

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        // Extra bits
        let ll_extra = 0u32; // code 4 and 8 have no extra bits
        let ml_extra = if ml_code == 33 { bits.read_bits(1).unwrap_or(0) } else { 0 };
        let of_extra = if of_code > 0 { bits.read_bits(of_code as usize).unwrap_or(0) } else { 0 };

        let ll_val = ll_code as u32;
        let of_val = if of_code > 0 { (1u32 << of_code) + of_extra } else { of_extra };
        let ml_val = get_ml_baseline(ml_code) + ml_extra;

        println!("  Values: literal_length={}, offset_value={} (code={}, extra={}), match_length={}",
                 ll_val, of_val, of_code, of_extra, ml_val);

        if !is_last {
            ll_decoder.update_state(&mut bits).ok();
            ml_decoder.update_state(&mut bits).ok();
            of_decoder.update_state(&mut bits).ok();
        }
    }

    println!("\nBits remaining: {}", bits.bits_remaining());
}

fn get_ml_baseline(code: u8) -> u32 {
    if code <= 31 {
        (code as u32) + 3
    } else {
        match code {
            32 => 35, 33 => 37, 34 => 39, 35 => 43, 36 => 47, 37 => 51,
            38 => 59, 39 => 67, 40 => 83, 41 => 99, 42 => 131, 43 => 259,
            44 => 515, 45 => 1027, 46 => 2051, 47 => 4099, 48 => 8195,
            49 => 16387, 50 => 32771, 51 => 65539, 52 => 131075, _ => 0,
        }
    }
}

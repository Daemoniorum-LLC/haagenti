//! Trace both working and failing cases

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::fse::{
    cached_ll_table, cached_ml_table, cached_of_table, BitReader, FseBitWriter, FseDecoder,
    InterleavedTansEncoder,
};

fn trace_case(name: &str, sequences: &[Sequence]) {
    println!("\n{}", "=".repeat(60));
    println!("=== {} ===", name);

    let encoded: Vec<_> = sequences
        .iter()
        .map(|s| EncodedSequence::from_sequence(s))
        .collect();

    println!("\nSequences:");
    for (i, (seq, enc)) in sequences.iter().zip(&encoded).enumerate() {
        println!(
            "  Seq {}: LL={} (code={}), OF={} (code={}, {}b), ML={} (code={})",
            i,
            seq.literal_length,
            enc.ll_code,
            seq.offset,
            enc.of_code,
            enc.of_bits,
            seq.match_length,
            enc.ml_code
        );
    }

    // Build FSE bitstream
    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();

    let last_idx = encoded.len() - 1;
    let last_seq = &encoded[last_idx];

    tans.init_states(last_seq.ll_code, last_seq.of_code, last_seq.ml_code);

    let mut fse_bits_per_seq: Vec<[(u32, u8); 3]> = Vec::new();
    for i in (0..last_idx).rev() {
        let seq = &encoded[i];
        let fse_bits = tans.encode_sequence(seq.ll_code, seq.of_code, seq.ml_code);
        fse_bits_per_seq.insert(0, fse_bits);
    }

    let (ll_final, of_final, ml_final) = tans.get_states();

    let mut bits = FseBitWriter::new();

    // Write forward
    let mut total_bits = 0;
    for i in 0..last_idx {
        let seq = &encoded[i];
        let [ll_fse, of_fse, ml_fse] = fse_bits_per_seq[i];

        if seq.ll_bits > 0 {
            bits.write_bits(seq.ll_extra, seq.ll_bits);
            total_bits += seq.ll_bits as usize;
        }
        if seq.ml_bits > 0 {
            bits.write_bits(seq.ml_extra, seq.ml_bits);
            total_bits += seq.ml_bits as usize;
        }
        if seq.of_bits > 0 {
            bits.write_bits(seq.of_extra, seq.of_bits);
            total_bits += seq.of_bits as usize;
        }

        bits.write_bits(ll_fse.0, ll_fse.1);
        total_bits += ll_fse.1 as usize;
        bits.write_bits(ml_fse.0, ml_fse.1);
        total_bits += ml_fse.1 as usize;
        bits.write_bits(of_fse.0, of_fse.1);
        total_bits += of_fse.1 as usize;
    }

    // Last sequence
    if last_seq.ll_bits > 0 {
        bits.write_bits(last_seq.ll_extra, last_seq.ll_bits);
        total_bits += last_seq.ll_bits as usize;
    }
    if last_seq.ml_bits > 0 {
        bits.write_bits(last_seq.ml_extra, last_seq.ml_bits);
        total_bits += last_seq.ml_bits as usize;
    }
    if last_seq.of_bits > 0 {
        bits.write_bits(last_seq.of_extra, last_seq.of_bits);
        total_bits += last_seq.of_bits as usize;
    }

    bits.write_bits(ml_final, ml_log);
    total_bits += ml_log as usize;
    bits.write_bits(of_final, of_log);
    total_bits += of_log as usize;
    bits.write_bits(ll_final, ll_log);
    total_bits += ll_log as usize;

    let bitstream = bits.finish();
    println!(
        "\nEncoded: {} data bits + 1 sentinel = {} bits, {} bytes",
        total_bits,
        total_bits + 1,
        bitstream.len()
    );
    println!("Bitstream: {:02x?}", bitstream);
    println!(
        "States (become decoder initial): LL={}, OF={}, ML={}",
        ll_final, of_final, ml_final
    );

    // Decode
    let ll_table = cached_ll_table();
    let of_table = cached_of_table();
    let ml_table = cached_ml_table();

    let mut reader = BitReader::new(&bitstream);
    if let Err(e) = reader.init_from_end() {
        println!("ERROR: init_from_end failed: {:?}", e);
        return;
    }

    let mut ll_decoder = FseDecoder::new(ll_table);
    let mut of_decoder = FseDecoder::new(of_table);
    let mut ml_decoder = FseDecoder::new(ml_table);

    if let Err(e) = ll_decoder.init_state(&mut reader) {
        println!("ERROR: LL init_state failed: {:?}", e);
        return;
    }
    if let Err(e) = of_decoder.init_state(&mut reader) {
        println!("ERROR: OF init_state failed: {:?}", e);
        return;
    }
    if let Err(e) = ml_decoder.init_state(&mut reader) {
        println!("ERROR: ML init_state failed: {:?}", e);
        return;
    }

    println!(
        "\nDecoded initial states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );

    if let Err(e) = reader.switch_to_lsb_mode() {
        println!("ERROR: switch_to_lsb_mode failed: {:?}", e);
        return;
    }

    let mut all_ok = true;
    for i in 0..encoded.len() {
        let expected = &encoded[i];

        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        let ok = ll_code == expected.ll_code
            && of_code == expected.of_code
            && ml_code == expected.ml_code;

        println!(
            "Seq {}: codes ({},{},{}) {} ({},{},{})",
            i,
            ll_code,
            of_code,
            ml_code,
            if ok { "==" } else { "!=" },
            expected.ll_code,
            expected.of_code,
            expected.ml_code
        );

        if !ok {
            all_ok = false;
        }

        // Read extras
        if expected.ll_bits > 0 {
            let _ = reader.read_bits(expected.ll_bits as usize);
        }
        if expected.ml_bits > 0 {
            let _ = reader.read_bits(expected.ml_bits as usize);
        }
        if expected.of_bits > 0 {
            let _ = reader.read_bits(expected.of_bits as usize);
        }

        if i < encoded.len() - 1 {
            if let Err(e) = ll_decoder.update_state(&mut reader) {
                println!("  ERROR: LL update_state failed: {:?}", e);
                all_ok = false;
                break;
            }
            if let Err(e) = ml_decoder.update_state(&mut reader) {
                println!("  ERROR: ML update_state failed: {:?}", e);
                all_ok = false;
                break;
            }
            if let Err(e) = of_decoder.update_state(&mut reader) {
                println!("  ERROR: OF update_state failed: {:?}", e);
                all_ok = false;
                break;
            }
        }
    }

    println!("\nResult: {}", if all_ok { "OK" } else { "FAILED" });
}

fn main() {
    // Working case: 2 seq, no repeat offset
    trace_case(
        "WORKING: 2 seq, no repeat",
        &[
            Sequence::new(4, 2, 4),  // LL=4, OF=2 (code=1), ML=4
            Sequence::new(3, 10, 4), // LL=3, OF=10 (code=3), ML=4
        ],
    );

    // Failing case: 2 seq, with repeat offset
    trace_case(
        "FAILING: 2 seq, with repeat",
        &[
            Sequence::new(9, 12, 8), // LL=9, OF=12 (code=3), ML=8
            Sequence::new(1, 1, 4),  // LL=1, OF=1 (code=0), ML=4
        ],
    );

    // Another test: 2 seq with just code 0 in seq 1
    trace_case(
        "TEST: OF code 0",
        &[
            Sequence::new(4, 4, 4), // LL=4, OF=4 (code=2), ML=4
            Sequence::new(1, 1, 4), // LL=1, OF=1 (code=0), ML=4
        ],
    );
}

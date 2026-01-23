//! Debug encode and decode for 2 sequences

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::fse::{
    cached_ll_table, cached_ml_table, cached_of_table, BitReader, FseBitWriter, FseDecoder,
    InterleavedTansEncoder,
};

fn main() {
    // Two sequences from failing case:
    // Seq 0: LL=8 (code=8), OF=3 (code=1), ML=7 (code=4)
    // Seq 1: LL=1 (code=1), OF=1 (code=0), ML=7 (code=4)
    let sequences = vec![
        Sequence::new(8, 3, 7), // OF=3 means offset_value=3, code=1
        Sequence::new(1, 1, 7), // OF=1 means repeat offset, code=0
    ];

    let encoded: Vec<_> = sequences
        .iter()
        .map(|s| EncodedSequence::from_sequence(s))
        .collect();

    println!("=== Encoded sequences ===");
    for (i, enc) in encoded.iter().enumerate() {
        println!(
            "Seq {}: LL_code={}, OF_code={}, ML_code={}",
            i, enc.ll_code, enc.of_code, enc.ml_code
        );
        println!(
            "       LL_extra={} ({} bits), OF_extra={} ({} bits), ML_extra={} ({} bits)",
            enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits
        );
    }

    // Build FSE bitstream manually following the algorithm
    println!("\n=== Building FSE bitstream ===");

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();
    println!("Accuracy logs: LL={}, OF={}, ML={}", ll_log, of_log, ml_log);

    let last_idx = encoded.len() - 1;
    let last_seq = &encoded[last_idx];

    // Step 1: Init with last sequence
    tans.init_states(last_seq.ll_code, last_seq.of_code, last_seq.ml_code);
    let (ll_init, of_init, ml_init) = tans.get_states();
    println!(
        "\nInit with seq[{}] codes ({}, {}, {}): states = ({}, {}, {})",
        last_idx, last_seq.ll_code, last_seq.of_code, last_seq.ml_code, ll_init, of_init, ml_init
    );

    // Step 2: Encode previous sequences in REVERSE order
    let mut fse_bits_per_seq: Vec<[(u32, u8); 3]> = Vec::new();
    for i in (0..last_idx).rev() {
        let seq = &encoded[i];
        let fse_bits = tans.encode_sequence(seq.ll_code, seq.of_code, seq.ml_code);
        let (ll_s, of_s, ml_s) = tans.get_states();
        println!(
            "Encode seq[{}] codes ({}, {}, {}): ",
            i, seq.ll_code, seq.of_code, seq.ml_code
        );
        println!(
            "  FSE bits: LL=({},{}) OF=({},{}) ML=({},{})",
            fse_bits[0].0,
            fse_bits[0].1,
            fse_bits[1].0,
            fse_bits[1].1,
            fse_bits[2].0,
            fse_bits[2].1
        );
        println!("  New states: ({}, {}, {})", ll_s, of_s, ml_s);
        fse_bits_per_seq.insert(0, fse_bits); // Insert at front to maintain order
    }

    // Get final states
    let (ll_final, of_final, ml_final) = tans.get_states();
    println!(
        "\nFinal states (become decoder initial): LL={}, OF={}, ML={}",
        ll_final, of_final, ml_final
    );

    // Step 3: Write bits in forward order
    let mut bits = FseBitWriter::new();

    println!("\n=== Writing bitstream ===");
    for i in 0..last_idx {
        let seq = &encoded[i];
        let [ll_fse, of_fse, ml_fse] = fse_bits_per_seq[i];

        println!("Seq [{}]:", i);
        // Extra bits
        if seq.ll_bits > 0 {
            println!("  Write LL extra: {} ({} bits)", seq.ll_extra, seq.ll_bits);
            bits.write_bits(seq.ll_extra, seq.ll_bits);
        }
        if seq.ml_bits > 0 {
            println!("  Write ML extra: {} ({} bits)", seq.ml_extra, seq.ml_bits);
            bits.write_bits(seq.ml_extra, seq.ml_bits);
        }
        if seq.of_bits > 0 {
            println!("  Write OF extra: {} ({} bits)", seq.of_extra, seq.of_bits);
            bits.write_bits(seq.of_extra, seq.of_bits);
        }

        // FSE bits
        println!("  Write LL FSE: {} ({} bits)", ll_fse.0, ll_fse.1);
        bits.write_bits(ll_fse.0, ll_fse.1);
        println!("  Write ML FSE: {} ({} bits)", ml_fse.0, ml_fse.1);
        bits.write_bits(ml_fse.0, ml_fse.1);
        println!("  Write OF FSE: {} ({} bits)", of_fse.0, of_fse.1);
        bits.write_bits(of_fse.0, of_fse.1);
    }

    // Last sequence: just extras
    println!("Seq [{}] (last):", last_idx);
    if last_seq.ll_bits > 0 {
        println!(
            "  Write LL extra: {} ({} bits)",
            last_seq.ll_extra, last_seq.ll_bits
        );
        bits.write_bits(last_seq.ll_extra, last_seq.ll_bits);
    }
    if last_seq.ml_bits > 0 {
        println!(
            "  Write ML extra: {} ({} bits)",
            last_seq.ml_extra, last_seq.ml_bits
        );
        bits.write_bits(last_seq.ml_extra, last_seq.ml_bits);
    }
    if last_seq.of_bits > 0 {
        println!(
            "  Write OF extra: {} ({} bits)",
            last_seq.of_extra, last_seq.of_bits
        );
        bits.write_bits(last_seq.of_extra, last_seq.of_bits);
    }

    // Write states
    println!("\nWrite states:");
    println!("  ML state: {} ({} bits)", ml_final, ml_log);
    bits.write_bits(ml_final, ml_log);
    println!("  OF state: {} ({} bits)", of_final, of_log);
    bits.write_bits(of_final, of_log);
    println!("  LL state: {} ({} bits)", ll_final, ll_log);
    bits.write_bits(ll_final, ll_log);

    let bitstream = bits.finish();
    println!(
        "\nFinal bitstream: {:02x?} ({} bytes)",
        bitstream,
        bitstream.len()
    );

    // === Now decode ===
    println!("\n=== Decoding bitstream ===");

    let ll_table = cached_ll_table();
    let of_table = cached_of_table();
    let ml_table = cached_ml_table();

    let mut reader = BitReader::new(&bitstream);
    reader.init_from_end().unwrap();

    // Read initial states (MSB-first)
    let mut ll_decoder = FseDecoder::new(ll_table);
    let mut of_decoder = FseDecoder::new(of_table);
    let mut ml_decoder = FseDecoder::new(ml_table);

    ll_decoder.init_state(&mut reader).unwrap();
    of_decoder.init_state(&mut reader).unwrap();
    ml_decoder.init_state(&mut reader).unwrap();

    println!(
        "Initial states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );

    // Switch to LSB mode
    reader.switch_to_lsb_mode().unwrap();

    // Decode each sequence
    for i in 0..encoded.len() {
        let expected = &encoded[i];

        // Peek symbols
        let ll_code = ll_decoder.peek_symbol();
        let of_code = of_decoder.peek_symbol();
        let ml_code = ml_decoder.peek_symbol();

        println!(
            "\nSeq [{}]: codes = ({}, {}, {}), expected = ({}, {}, {})",
            i, ll_code, of_code, ml_code, expected.ll_code, expected.of_code, expected.ml_code
        );

        if ll_code != expected.ll_code || of_code != expected.of_code || ml_code != expected.ml_code
        {
            println!("  *** MISMATCH! ***");
        }

        // Read extras
        if expected.ll_bits > 0 {
            let extra = reader.read_bits(expected.ll_bits as usize).unwrap();
            println!("  LL extra: {} (expected {})", extra, expected.ll_extra);
        }
        if expected.ml_bits > 0 {
            let extra = reader.read_bits(expected.ml_bits as usize).unwrap();
            println!("  ML extra: {} (expected {})", extra, expected.ml_extra);
        }
        if expected.of_bits > 0 {
            let extra = reader.read_bits(expected.of_bits as usize).unwrap();
            println!("  OF extra: {} (expected {})", extra, expected.of_extra);
        }

        // Update states (except for last)
        if i < encoded.len() - 1 {
            ll_decoder.update_state(&mut reader).unwrap();
            ml_decoder.update_state(&mut reader).unwrap();
            of_decoder.update_state(&mut reader).unwrap();
            println!(
                "  Updated states: LL={}, OF={}, ML={}",
                ll_decoder.state(),
                of_decoder.state(),
                ml_decoder.state()
            );
        }
    }

    println!("\n=== Done ===");
}

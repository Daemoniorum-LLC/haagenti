//! Detailed trace of FSE bitstream encoding vs decoding

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::fse::{
    BitReader, FseBitWriter, FseDecoder, FseTable, InterleavedTansEncoder,
    LITERAL_LENGTH_ACCURACY_LOG, LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== FSE Bitstream Trace ===\n");

    // Build predefined tables
    let ll_table = FseTable::from_predefined(
        &LITERAL_LENGTH_DEFAULT_DISTRIBUTION,
        LITERAL_LENGTH_ACCURACY_LOG,
    )
    .unwrap();
    let of_table =
        FseTable::from_predefined(&OFFSET_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG).unwrap();
    let ml_table = FseTable::from_predefined(
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        MATCH_LENGTH_ACCURACY_LOG,
    )
    .unwrap();

    // ACTUAL sequences with repeat offsets:
    // Seq 0: ll=4, offset_value=2 (repeat_offset_2=4), ml=36 → of_code=1, ml_code=32
    // Seq 1: ll=8, offset_value=1 (repeat_offset_1=4), ml=16 → of_code=0, ml_code=13
    // Note: offset_value=2 → of_code=1 (since 2 = (1<<1) + 0)
    // Note: offset_value=1 → of_code=0 (since 1 = (1<<0) + 0)
    let sequences = vec![Sequence::new(4, 2, 36), Sequence::new(8, 1, 16)];

    let encoded: Vec<EncodedSequence> = sequences
        .iter()
        .map(EncodedSequence::from_sequence)
        .collect();

    println!("Input sequences:");
    for (i, enc) in encoded.iter().enumerate() {
        println!(
            "  Seq[{}]: LL_code={}, OF_code={}, ML_code={}",
            i, enc.ll_code, enc.of_code, enc.ml_code
        );
        println!(
            "          extras: LL={}({} bits), ML={}({} bits), OF={}({} bits)",
            enc.ll_extra, enc.ll_bits, enc.ml_extra, enc.ml_bits, enc.of_extra, enc.of_bits
        );
    }

    // ENCODE
    println!("\n=== ENCODING ===");
    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();
    println!("Accuracy logs: LL={}, OF={}, ML={}", ll_log, of_log, ml_log);

    // Init with LAST sequence
    let last = &encoded[1];
    tans.init_states(last.ll_code, last.of_code, last.ml_code);
    let (ll_init, of_init, ml_init) = tans.get_states();
    println!(
        "\nAfter init with seq[1] codes ({}, {}, {}):",
        last.ll_code, last.of_code, last.ml_code
    );
    println!("  States: LL={}, OF={}, ML={}", ll_init, of_init, ml_init);
    println!(
        "  Verify: LL state {} -> symbol {}",
        ll_init,
        ll_table.decode(ll_init as usize).symbol
    );
    println!(
        "  Verify: OF state {} -> symbol {}",
        of_init,
        of_table.decode(of_init as usize).symbol
    );
    println!(
        "  Verify: ML state {} -> symbol {}",
        ml_init,
        ml_table.decode(ml_init as usize).symbol
    );

    // Encode seq[0]
    let fse_bits = tans.encode_sequence(encoded[0].ll_code, encoded[0].of_code, encoded[0].ml_code);
    let (ll_state, of_state, ml_state) = tans.get_states();
    println!(
        "\nAfter encode seq[0] codes ({}, {}, {}):",
        encoded[0].ll_code, encoded[0].of_code, encoded[0].ml_code
    );
    println!(
        "  FSE bits: LL({}, {} bits), OF({}, {} bits), ML({}, {} bits)",
        fse_bits[0].0, fse_bits[0].1, fse_bits[1].0, fse_bits[1].1, fse_bits[2].0, fse_bits[2].1
    );
    println!(
        "  New states: LL={}, OF={}, ML={}",
        ll_state, of_state, ml_state
    );
    println!(
        "  Verify: LL state {} -> symbol {}",
        ll_state,
        ll_table.decode(ll_state as usize).symbol
    );
    println!(
        "  Verify: OF state {} -> symbol {}",
        of_state,
        of_table.decode(of_state as usize).symbol
    );
    println!(
        "  Verify: ML state {} -> symbol {}",
        ml_state,
        ml_table.decode(ml_state as usize).symbol
    );

    // Build bitstream
    let mut bits = FseBitWriter::new();

    println!("\n=== BITSTREAM CONSTRUCTION (LSB to MSB) ===");
    let mut bit_pos = 0;

    // Seq[0] extras
    let seq0 = &encoded[0];
    if seq0.ll_bits > 0 {
        println!(
            "  [{:3}] seq0.ll_extra = {} ({} bits)",
            bit_pos, seq0.ll_extra, seq0.ll_bits
        );
        bits.write_bits(seq0.ll_extra, seq0.ll_bits);
        bit_pos += seq0.ll_bits as usize;
    }
    if seq0.ml_bits > 0 {
        println!(
            "  [{:3}] seq0.ml_extra = {} ({} bits)",
            bit_pos, seq0.ml_extra, seq0.ml_bits
        );
        bits.write_bits(seq0.ml_extra, seq0.ml_bits);
        bit_pos += seq0.ml_bits as usize;
    }
    if seq0.of_bits > 0 {
        println!(
            "  [{:3}] seq0.of_extra = {} ({} bits)",
            bit_pos, seq0.of_extra, seq0.of_bits
        );
        bits.write_bits(seq0.of_extra, seq0.of_bits);
        bit_pos += seq0.of_bits as usize;
    }

    // Seq[0] FSE update bits
    println!(
        "  [{:3}] seq0.ll_fse = {} ({} bits)",
        bit_pos, fse_bits[0].0, fse_bits[0].1
    );
    bits.write_bits(fse_bits[0].0, fse_bits[0].1);
    bit_pos += fse_bits[0].1 as usize;

    println!(
        "  [{:3}] seq0.ml_fse = {} ({} bits)",
        bit_pos, fse_bits[2].0, fse_bits[2].1
    );
    bits.write_bits(fse_bits[2].0, fse_bits[2].1);
    bit_pos += fse_bits[2].1 as usize;

    println!(
        "  [{:3}] seq0.of_fse = {} ({} bits)",
        bit_pos, fse_bits[1].0, fse_bits[1].1
    );
    bits.write_bits(fse_bits[1].0, fse_bits[1].1);
    bit_pos += fse_bits[1].1 as usize;

    // Seq[1] extras (last sequence, no FSE update)
    if last.ll_bits > 0 {
        println!(
            "  [{:3}] seq1.ll_extra = {} ({} bits)",
            bit_pos, last.ll_extra, last.ll_bits
        );
        bits.write_bits(last.ll_extra, last.ll_bits);
        bit_pos += last.ll_bits as usize;
    }
    if last.ml_bits > 0 {
        println!(
            "  [{:3}] seq1.ml_extra = {} ({} bits)",
            bit_pos, last.ml_extra, last.ml_bits
        );
        bits.write_bits(last.ml_extra, last.ml_bits);
        bit_pos += last.ml_bits as usize;
    }
    if last.of_bits > 0 {
        println!(
            "  [{:3}] seq1.of_extra = {} ({} bits)",
            bit_pos, last.of_extra, last.of_bits
        );
        bits.write_bits(last.of_extra, last.of_bits);
        bit_pos += last.of_bits as usize;
    }

    // Initial states (written last, read first from MSB)
    println!(
        "  [{:3}] ml_state = {} ({} bits)",
        bit_pos, ml_state, ml_log
    );
    bits.write_bits(ml_state, ml_log);
    bit_pos += ml_log as usize;

    println!(
        "  [{:3}] of_state = {} ({} bits)",
        bit_pos, of_state, of_log
    );
    bits.write_bits(of_state, of_log);
    bit_pos += of_log as usize;

    println!(
        "  [{:3}] ll_state = {} ({} bits)",
        bit_pos, ll_state, ll_log
    );
    bits.write_bits(ll_state, ll_log);
    bit_pos += ll_log as usize;

    let bitstream = bits.finish();
    println!(
        "\nFinal bitstream ({} bytes): {:02x?}",
        bitstream.len(),
        bitstream
    );
    println!("Total data bits: {}", bit_pos);
    println!(
        "With sentinel: {} bits = {} bytes",
        bit_pos + 1,
        (bit_pos + 8) / 8
    );

    // Print binary representation
    print!("Binary (LSB first): ");
    for byte in &bitstream {
        for i in 0..8 {
            print!("{}", (byte >> i) & 1);
        }
        print!(" ");
    }
    println!();

    // DECODE with our decoder
    println!("\n=== DECODING (our decoder) ===");
    let mut bits_reader = BitReader::new(&bitstream);
    bits_reader.init_from_end().unwrap();
    println!("Bits available: {}", bits_reader.bits_remaining());

    let mut ll_decoder = FseDecoder::new(&ll_table);
    let mut of_decoder = FseDecoder::new(&of_table);
    let mut ml_decoder = FseDecoder::new(&ml_table);

    // Read initial states
    ll_decoder.init_state(&mut bits_reader).unwrap();
    of_decoder.init_state(&mut bits_reader).unwrap();
    ml_decoder.init_state(&mut bits_reader).unwrap();

    println!(
        "Initial states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );
    println!(
        "  LL symbol = {} (expected {})",
        ll_decoder.peek_symbol(),
        encoded[0].ll_code
    );
    println!(
        "  OF symbol = {} (expected {})",
        of_decoder.peek_symbol(),
        encoded[0].of_code
    );
    println!(
        "  ML symbol = {} (expected {})",
        ml_decoder.peek_symbol(),
        encoded[0].ml_code
    );
    println!(
        "Bits remaining after states: {}",
        bits_reader.bits_remaining()
    );

    // Switch to LSB mode for extras
    bits_reader.switch_to_lsb_mode().unwrap();
    println!(
        "Bits remaining in LSB mode: {}",
        bits_reader.bits_remaining()
    );

    // Decode seq[0]
    println!("\nDecoding seq[0]:");
    let ll0_code = ll_decoder.peek_symbol();
    let of0_code = of_decoder.peek_symbol();
    let ml0_code = ml_decoder.peek_symbol();
    println!(
        "  Codes from states: LL={}, OF={}, ML={}",
        ll0_code, of0_code, ml0_code
    );

    // Read extras
    if seq0.ll_bits > 0 {
        let val = bits_reader.read_bits(seq0.ll_bits as usize).unwrap();
        println!(
            "  Read LL extra: {} ({} bits), expected {}",
            val, seq0.ll_bits, seq0.ll_extra
        );
    }
    if seq0.ml_bits > 0 {
        let val = bits_reader.read_bits(seq0.ml_bits as usize).unwrap();
        println!(
            "  Read ML extra: {} ({} bits), expected {}",
            val, seq0.ml_bits, seq0.ml_extra
        );
    }
    if seq0.of_bits > 0 {
        let val = bits_reader.read_bits(seq0.of_bits as usize).unwrap();
        println!(
            "  Read OF extra: {} ({} bits), expected {}",
            val, seq0.of_bits, seq0.of_extra
        );
    }

    // Update states for next sequence
    println!("  Updating states...");
    let ll_entry = ll_table.decode(ll_decoder.state());
    let ml_entry = ml_table.decode(ml_decoder.state());
    let of_entry = of_table.decode(of_decoder.state());
    println!(
        "    LL: state={}, needs {} bits, baseline={}",
        ll_decoder.state(),
        ll_entry.num_bits,
        ll_entry.baseline
    );
    println!(
        "    ML: state={}, needs {} bits, baseline={}",
        ml_decoder.state(),
        ml_entry.num_bits,
        ml_entry.baseline
    );
    println!(
        "    OF: state={}, needs {} bits, baseline={}",
        of_decoder.state(),
        of_entry.num_bits,
        of_entry.baseline
    );

    ll_decoder.update_state(&mut bits_reader).unwrap();
    ml_decoder.update_state(&mut bits_reader).unwrap();
    of_decoder.update_state(&mut bits_reader).unwrap();

    println!(
        "  New states: LL={}, OF={}, ML={}",
        ll_decoder.state(),
        of_decoder.state(),
        ml_decoder.state()
    );
    println!(
        "  New symbols: LL={} (expected {}), OF={} (expected {}), ML={} (expected {})",
        ll_decoder.peek_symbol(),
        last.ll_code,
        of_decoder.peek_symbol(),
        last.of_code,
        ml_decoder.peek_symbol(),
        last.ml_code
    );
    println!("Bits remaining: {}", bits_reader.bits_remaining());

    // Decode seq[1]
    println!("\nDecoding seq[1]:");
    let ll1_code = ll_decoder.peek_symbol();
    let of1_code = of_decoder.peek_symbol();
    let ml1_code = ml_decoder.peek_symbol();
    println!(
        "  Codes from states: LL={}, OF={}, ML={}",
        ll1_code, of1_code, ml1_code
    );

    // Read extras for last seq
    if last.ll_bits > 0 {
        let val = bits_reader.read_bits(last.ll_bits as usize).unwrap();
        println!(
            "  Read LL extra: {} ({} bits), expected {}",
            val, last.ll_bits, last.ll_extra
        );
    }
    if last.ml_bits > 0 {
        let val = bits_reader.read_bits(last.ml_bits as usize).unwrap();
        println!(
            "  Read ML extra: {} ({} bits), expected {}",
            val, last.ml_bits, last.ml_extra
        );
    }
    if last.of_bits > 0 {
        let val = bits_reader.read_bits(last.of_bits as usize).unwrap();
        println!(
            "  Read OF extra: {} ({} bits), expected {}",
            val, last.of_bits, last.of_extra
        );
    }
    println!("Bits remaining: {}", bits_reader.bits_remaining());

    // Now check decode table transitions in detail
    println!("\n=== DECODE TABLE ANALYSIS ===");

    // What state should decode to seq[1].ll_code=1?
    println!("\nStates that decode to LL symbol 1:");
    for s in 0..ll_table.size() {
        if ll_table.decode(s).symbol == 1 {
            let entry = ll_table.decode(s);
            println!(
                "  State {}: num_bits={}, baseline={}",
                s, entry.num_bits, entry.baseline
            );
        }
    }

    // What does the LL transition look like?
    let ll_entry = ll_table.decode(ll_state as usize);
    println!(
        "\nLL transition from state {} (symbol {}):",
        ll_state, ll_entry.symbol
    );
    println!(
        "  num_bits = {}, baseline = {}",
        ll_entry.num_bits, ll_entry.baseline
    );
    println!(
        "  FSE bits written: {} ({} bits)",
        fse_bits[0].0, fse_bits[0].1
    );
    let ll_new = ll_entry.baseline as u32 + fse_bits[0].0;
    println!(
        "  Expected new state: {} + {} = {}",
        ll_entry.baseline, fse_bits[0].0, ll_new
    );
    println!(
        "  Symbol at state {}: {}",
        ll_new,
        ll_table.decode(ll_new as usize).symbol
    );

    // Also check if the number of bits matches
    println!("\n=== BIT COUNT VERIFICATION ===");
    println!(
        "LL: entry.num_bits={}, fse_bits={} -- MATCH: {}",
        ll_entry.num_bits,
        fse_bits[0].1,
        ll_entry.num_bits == fse_bits[0].1
    );

    let of_entry = of_table.decode(of_state as usize);
    println!(
        "OF: entry.num_bits={}, fse_bits={} -- MATCH: {}",
        of_entry.num_bits,
        fse_bits[1].1,
        of_entry.num_bits == fse_bits[1].1
    );

    let ml_entry = ml_table.decode(ml_state as usize);
    println!(
        "ML: entry.num_bits={}, fse_bits={} -- MATCH: {}",
        ml_entry.num_bits,
        fse_bits[2].1,
        ml_entry.num_bits == fse_bits[2].1
    );
}

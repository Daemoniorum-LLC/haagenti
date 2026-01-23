//! Deep trace of FSE encoding to find the bug
#![allow(unused_imports)]

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::EncodedSequence;
use haagenti_zstd::fse::{
    BitReader, FseBitWriter, FseDecoder, FseTable, InterleavedTansEncoder,
    LITERAL_LENGTH_ACCURACY_LOG, LITERAL_LENGTH_DEFAULT_DISTRIBUTION, MATCH_LENGTH_ACCURACY_LOG,
    MATCH_LENGTH_DEFAULT_DISTRIBUTION, OFFSET_ACCURACY_LOG, OFFSET_DEFAULT_DISTRIBUTION,
};

fn main() {
    println!("=== Deep Trace of FSE Encoding ===\n");

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

    // The failing case: 2 sequences
    let sequences = vec![
        Sequence::new(9, 12, 8), // LL=9, OF_value=12, ML=8
        Sequence::new(1, 1, 4),  // LL=1, OF_value=1, ML=4
    ];

    let encoded: Vec<_> = sequences
        .iter()
        .map(|s| EncodedSequence::from_sequence(s))
        .collect();

    println!("Sequences:");
    for (i, enc) in encoded.iter().enumerate() {
        println!(
            "  Seq {}: codes (LL={}, OF={}, ML={})",
            i, enc.ll_code, enc.of_code, enc.ml_code
        );
        println!(
            "          extras: LL={}({}b), OF={}({}b), ML={}({}b)",
            enc.ll_extra, enc.ll_bits, enc.of_extra, enc.of_bits, enc.ml_extra, enc.ml_bits
        );
    }

    // Now trace the encoding step by step
    println!("\n=== ENCODING TRACE ===");

    let mut tans = InterleavedTansEncoder::new_predefined();
    let (ll_log, of_log, ml_log) = tans.accuracy_logs();
    println!("Accuracy logs: LL={}, OF={}, ML={}", ll_log, of_log, ml_log);

    let last_seq = &encoded[1];

    // Step 1: Init with last sequence codes
    println!(
        "\nStep 1: Init with Seq 1 codes ({}, {}, {})",
        last_seq.ll_code, last_seq.of_code, last_seq.ml_code
    );
    tans.init_states(last_seq.ll_code, last_seq.of_code, last_seq.ml_code);
    let (ll_s, of_s, ml_s) = tans.get_states();
    println!("  States after init: LL={}, OF={}, ML={}", ll_s, of_s, ml_s);

    // Verify states decode to correct symbols
    println!(
        "  Verify: LL[{}]={}, OF[{}]={}, ML[{}]={}",
        ll_s,
        ll_table.decode(ll_s as usize).symbol,
        of_s,
        of_table.decode(of_s as usize).symbol,
        ml_s,
        ml_table.decode(ml_s as usize).symbol
    );

    // Step 2: Encode Seq 0 (the non-last sequence)
    let seq0 = &encoded[0];
    println!(
        "\nStep 2: Encode Seq 0 codes ({}, {}, {})",
        seq0.ll_code, seq0.of_code, seq0.ml_code
    );

    let fse_bits = tans.encode_sequence(seq0.ll_code, seq0.of_code, seq0.ml_code);
    let [ll_fse, of_fse, ml_fse] = fse_bits;
    println!(
        "  FSE bits: LL=({}, {} bits), OF=({}, {} bits), ML=({}, {} bits)",
        ll_fse.0, ll_fse.1, of_fse.0, of_fse.1, ml_fse.0, ml_fse.1
    );

    let (ll_final, of_final, ml_final) = tans.get_states();
    println!(
        "  Final states: LL={}, OF={}, ML={}",
        ll_final, of_final, ml_final
    );

    // Verify final states decode to Seq 0's codes
    println!(
        "  Verify: LL[{}]={}, OF[{}]={}, ML[{}]={}",
        ll_final,
        ll_table.decode(ll_final as usize).symbol,
        of_final,
        of_table.decode(of_final as usize).symbol,
        ml_final,
        ml_table.decode(ml_final as usize).symbol
    );

    // Step 3: Build the bitstream
    println!("\n=== BITSTREAM CONSTRUCTION ===");

    let mut bits = FseBitWriter::new();

    // Write Seq 0 extras (our order: LL, ML, OF)
    println!("Writing Seq 0 extras (LL, ML, OF order):");
    if seq0.ll_bits > 0 {
        println!("  LL extra: {} ({} bits)", seq0.ll_extra, seq0.ll_bits);
        bits.write_bits(seq0.ll_extra, seq0.ll_bits);
    }
    if seq0.ml_bits > 0 {
        println!("  ML extra: {} ({} bits)", seq0.ml_extra, seq0.ml_bits);
        bits.write_bits(seq0.ml_extra, seq0.ml_bits);
    }
    if seq0.of_bits > 0 {
        println!("  OF extra: {} ({} bits)", seq0.of_extra, seq0.of_bits);
        bits.write_bits(seq0.of_extra, seq0.of_bits);
    }

    // Write Seq 0 FSE update bits (our order: LL, ML, OF)
    println!("\nWriting Seq 0 FSE bits (LL, ML, OF order):");
    println!("  LL FSE: {} ({} bits)", ll_fse.0, ll_fse.1);
    bits.write_bits(ll_fse.0, ll_fse.1);
    println!("  ML FSE: {} ({} bits)", ml_fse.0, ml_fse.1);
    bits.write_bits(ml_fse.0, ml_fse.1);
    println!("  OF FSE: {} ({} bits)", of_fse.0, of_fse.1);
    bits.write_bits(of_fse.0, of_fse.1);

    // Write Seq 1 extras (last sequence, no FSE update)
    println!("\nWriting Seq 1 extras (last sequence):");
    if last_seq.ll_bits > 0 {
        println!(
            "  LL extra: {} ({} bits)",
            last_seq.ll_extra, last_seq.ll_bits
        );
        bits.write_bits(last_seq.ll_extra, last_seq.ll_bits);
    }
    if last_seq.ml_bits > 0 {
        println!(
            "  ML extra: {} ({} bits)",
            last_seq.ml_extra, last_seq.ml_bits
        );
        bits.write_bits(last_seq.ml_extra, last_seq.ml_bits);
    }
    if last_seq.of_bits > 0 {
        println!(
            "  OF extra: {} ({} bits)",
            last_seq.of_extra, last_seq.of_bits
        );
        bits.write_bits(last_seq.of_extra, last_seq.of_bits);
    }

    // Write states
    println!("\nWriting states (ML, OF, LL order for MSB reading):");
    println!("  ML state: {} ({} bits)", ml_final, ml_log);
    bits.write_bits(ml_final, ml_log);
    println!("  OF state: {} ({} bits)", of_final, of_log);
    bits.write_bits(of_final, of_log);
    println!("  LL state: {} ({} bits)", ll_final, ll_log);
    bits.write_bits(ll_final, ll_log);

    let bitstream = bits.finish();
    println!("\nFinal bitstream: {:02x?}", bitstream);

    // Now decode and trace
    println!("\n=== DECODING TRACE ===");

    let mut reader = BitReader::new(&bitstream);
    reader.init_from_end().unwrap();
    println!("Bits after init_from_end: {}", reader.bits_remaining());

    // Read initial states
    let ll_state_read = reader.read_bits(ll_log as usize).unwrap();
    let of_state_read = reader.read_bits(of_log as usize).unwrap();
    let ml_state_read = reader.read_bits(ml_log as usize).unwrap();
    println!(
        "\nRead initial states: LL={}, OF={}, ML={}",
        ll_state_read, of_state_read, ml_state_read
    );

    // Verify these match what we wrote
    println!(
        "Expected: LL={}, OF={}, ML={}",
        ll_final, of_final, ml_final
    );
    println!(
        "Match: {}",
        ll_state_read == ll_final && of_state_read == of_final && ml_state_read == ml_final
    );

    // Get symbols from states
    let ll_sym = ll_table.decode(ll_state_read as usize).symbol;
    let of_sym = of_table.decode(of_state_read as usize).symbol;
    let ml_sym = ml_table.decode(ml_state_read as usize).symbol;
    println!(
        "\nSeq 0 symbols from states: LL={}, OF={}, ML={}",
        ll_sym, of_sym, ml_sym
    );
    println!(
        "Expected: LL={}, OF={}, ML={}",
        seq0.ll_code, seq0.of_code, seq0.ml_code
    );

    // Switch to LSB mode for extras and FSE bits
    reader.switch_to_lsb_mode().unwrap();
    println!(
        "\nSwitched to LSB mode. Bits remaining: {}",
        reader.bits_remaining()
    );

    // Read Seq 0 extras
    println!("\nReading Seq 0 extras (same order we wrote: LL, ML, OF):");
    if seq0.ll_bits > 0 {
        let ll_ex = reader.read_bits(seq0.ll_bits as usize).unwrap();
        println!("  LL extra: {} (expected {})", ll_ex, seq0.ll_extra);
    }
    if seq0.ml_bits > 0 {
        let ml_ex = reader.read_bits(seq0.ml_bits as usize).unwrap();
        println!("  ML extra: {} (expected {})", ml_ex, seq0.ml_extra);
    }
    if seq0.of_bits > 0 {
        let of_ex = reader.read_bits(seq0.of_bits as usize).unwrap();
        println!("  OF extra: {} (expected {})", of_ex, seq0.of_extra);
    }

    // Read Seq 0 FSE bits
    println!("\nReading Seq 0 FSE bits (same order we wrote: LL, ML, OF):");
    if ll_fse.1 > 0 {
        let ll_bits_read = reader.read_bits(ll_fse.1 as usize).unwrap();
        println!("  LL FSE: {} (expected {})", ll_bits_read, ll_fse.0);
    }
    if ml_fse.1 > 0 {
        let ml_bits_read = reader.read_bits(ml_fse.1 as usize).unwrap();
        println!("  ML FSE: {} (expected {})", ml_bits_read, ml_fse.0);
    }
    if of_fse.1 > 0 {
        let of_bits_read = reader.read_bits(of_fse.1 as usize).unwrap();
        println!("  OF FSE: {} (expected {})", of_bits_read, of_fse.0);
    }

    // Now simulate what the reference decoder does differently
    println!("\n=== REFERENCE DECODER ORDER ===");
    println!("Reference reads extras in order: OF, ML, LL (not LL, ML, OF)");
    println!("Reference reads FSE updates in order: LL, ML, OF (same as us)");
    println!("\nSo for Seq 0:");
    println!("  Reference would read OF extra first (3 bits) = expects offset extra");
    println!("  We wrote LL extra first (0 bits), then ML (0 bits), then OF (3 bits)");
    println!("  Since LL and ML have 0 bits, OF extra IS first in the stream!");
    println!("  So the extras should be correct.\n");

    // The difference might be in how FSE state updates work
    println!("Checking FSE state update computation...");

    // Manually track decoder states (no FseDecoder.set_state available)
    let mut ll_state_cur = ll_final as usize;
    let mut of_state_cur = of_final as usize;
    let mut ml_state_cur = ml_final as usize;

    println!(
        "\nDecoder states set to: LL={}, OF={}, ML={}",
        ll_state_cur, of_state_cur, ml_state_cur
    );

    // Read a fresh bitstream for FSE update simulation
    let mut reader2 = BitReader::new(&bitstream);
    reader2.init_from_end().unwrap();
    reader2.read_bits(ll_log as usize).unwrap(); // skip LL state
    reader2.read_bits(of_log as usize).unwrap(); // skip OF state
    reader2.read_bits(ml_log as usize).unwrap(); // skip ML state
    reader2.switch_to_lsb_mode().unwrap();

    // Skip extras for Seq 0
    if seq0.ll_bits > 0 {
        reader2.read_bits(seq0.ll_bits as usize).unwrap();
    }
    if seq0.ml_bits > 0 {
        reader2.read_bits(seq0.ml_bits as usize).unwrap();
    }
    if seq0.of_bits > 0 {
        reader2.read_bits(seq0.of_bits as usize).unwrap();
    }

    println!(
        "\nAfter skipping Seq 0 extras, bits remaining: {}",
        reader2.bits_remaining()
    );

    // Now update LL state
    println!("\nUpdating LL decoder:");
    let ll_entry = ll_table.decode(ll_state_cur);
    println!(
        "  Current state: {}, num_bits to read: {}",
        ll_state_cur, ll_entry.num_bits
    );
    let ll_update_bits = reader2.read_bits(ll_entry.num_bits as usize).unwrap();
    println!(
        "  Read {} bits: value={}",
        ll_entry.num_bits, ll_update_bits
    );
    let new_ll_state = ll_entry.baseline as usize + ll_update_bits as usize;
    ll_state_cur = new_ll_state;
    println!(
        "  New state: {} (baseline {} + bits {})",
        new_ll_state, ll_entry.baseline, ll_update_bits
    );
    println!("  New symbol: {}", ll_table.decode(new_ll_state).symbol);

    // Update ML state
    println!("\nUpdating ML decoder:");
    let ml_entry = ml_table.decode(ml_state_cur);
    println!(
        "  Current state: {}, num_bits to read: {}",
        ml_state_cur, ml_entry.num_bits
    );
    let ml_update_bits = reader2.read_bits(ml_entry.num_bits as usize).unwrap();
    println!(
        "  Read {} bits: value={}",
        ml_entry.num_bits, ml_update_bits
    );
    let new_ml_state = ml_entry.baseline as usize + ml_update_bits as usize;
    ml_state_cur = new_ml_state;
    println!(
        "  New state: {} (baseline {} + bits {})",
        new_ml_state, ml_entry.baseline, ml_update_bits
    );
    println!("  New symbol: {}", ml_table.decode(new_ml_state).symbol);

    // Update OF state
    println!("\nUpdating OF decoder:");
    let of_entry = of_table.decode(of_state_cur);
    println!(
        "  Current state: {}, num_bits to read: {}",
        of_state_cur, of_entry.num_bits
    );
    let of_update_bits = reader2.read_bits(of_entry.num_bits as usize).unwrap();
    println!(
        "  Read {} bits: value={}",
        of_entry.num_bits, of_update_bits
    );
    let new_of_state = of_entry.baseline as usize + of_update_bits as usize;
    of_state_cur = new_of_state;
    println!(
        "  New state: {} (baseline {} + bits {})",
        new_of_state, of_entry.baseline, of_update_bits
    );
    println!("  New symbol: {}", of_table.decode(new_of_state).symbol);

    println!("\n=== ANALYSIS ===");
    println!("After Seq 0 FSE updates:");
    println!(
        "  LL symbol: {} (expected {})",
        ll_table.decode(new_ll_state).symbol,
        last_seq.ll_code
    );
    println!(
        "  OF symbol: {} (expected {})",
        of_table.decode(new_of_state).symbol,
        last_seq.of_code
    );
    println!(
        "  ML symbol: {} (expected {})",
        ml_table.decode(new_ml_state).symbol,
        last_seq.ml_code
    );

    let ll_ok = ll_table.decode(new_ll_state).symbol == last_seq.ll_code;
    let of_ok = of_table.decode(new_of_state).symbol == last_seq.of_code;
    let ml_ok = ml_table.decode(new_ml_state).symbol == last_seq.ml_code;

    if ll_ok && of_ok && ml_ok {
        println!("\nAll symbols match! FSE encoding appears correct.");
        println!("The bug might be elsewhere in the decoder's expectations.");
    } else {
        println!("\nSYMBOL MISMATCH DETECTED - This is likely the bug!");
    }
}

//! Trace 2-sequence cases

use haagenti_zstd::compress::block::matches_to_sequences;
use haagenti_zstd::compress::{EncodedSequence, MatchFinder};

fn trace(name: &str, data: &[u8]) {
    println!("\n=== {} ({} bytes) ===", name, data.len());
    println!("Data: {:?}", String::from_utf8_lossy(data));

    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);
    println!("\nMatches:");
    for m in &matches {
        println!(
            "  pos={:2}, offset={:2}, len={:2}",
            m.position, m.offset, m.length
        );
    }

    let (_literals, sequences) = matches_to_sequences(data, &matches);
    println!(
        "\nSequences ({}): (offset is encoded as offset_value)",
        sequences.len()
    );
    for (i, seq) in sequences.iter().enumerate() {
        let enc = EncodedSequence::from_sequence(seq);
        println!("  Seq {}: LL={:2} (code={:2}), OF={:2} (code={:2}, extra={}, bits={}), ML={:2} (code={:2})",
                 i, seq.literal_length, enc.ll_code,
                 seq.offset, enc.of_code, enc.of_extra, enc.of_bits,
                 seq.match_length, enc.ml_code);
    }
}

fn main() {
    // Working 2-seq case
    trace("WORKS: abcdabcdXYZabcd", b"abcdabcdXYZabcd");

    // Failing 2-seq case
    trace("FAILS: abcdefghXabcdefghYabcd", b"abcdefghXabcdefghYabcd");

    // Another failing case
    trace("FAILS: PATTERN0PATTERN1PATTERN", b"PATTERN0PATTERN1PATTERN");
}

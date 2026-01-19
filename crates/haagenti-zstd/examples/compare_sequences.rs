//! Compare sequences between passing and failing patterns

use haagenti_zstd::compress::{LazyMatchFinder, block};

fn analyze(name: &str, data: &[u8]) {
    let mut mf = LazyMatchFinder::new(8);
    let matches = mf.find_matches_auto(data);
    let (literals, sequences) = block::matches_to_sequences(data, &matches);

    println!("\n=== {} ===", name);
    println!("Input: {} bytes, Matches: {}, Literals: {}, Sequences: {}",
             data.len(), matches.len(), literals.len(), sequences.len());

    // Show first few sequences
    for (i, seq) in sequences.iter().take(5).enumerate() {
        println!("  Seq {}: ll={}, offset={}, ml={}", i,
                 seq.literal_length, seq.offset, seq.match_length);
    }
    if sequences.len() > 5 {
        println!("  ... ({} more)", sequences.len() - 5);
    }

    // Check for patterns
    let all_offset_1 = sequences.iter().all(|s| s.offset == 1);
    let all_same_ll = sequences.windows(2).all(|w| w[0].literal_length == w[1].literal_length);

    println!("  All offset=1: {}, All same LL: {}", all_offset_1, all_same_ll);
}

fn main() {
    // Failing pattern: XYZW
    let mut xyzw = Vec::new();
    xyzw.extend(vec![b'X'; 100]);
    xyzw.extend(vec![b'Y'; 100]);
    xyzw.extend(vec![b'Z'; 100]);
    xyzw.extend(vec![b'W'; 100]);
    analyze("XYZW*400 (FAILS)", &xyzw);

    // Passing pattern: text
    let text = b"The quick brown fox jumps over the lazy dog. ";
    let text_data: Vec<u8> = text.iter().cycle().take(400).copied().collect();
    analyze("text*400 (PASSES)", &text_data);

    // Simpler failing case
    let mut xy = Vec::new();
    xy.extend(vec![b'X'; 100]);
    xy.extend(vec![b'Y'; 100]);
    analyze("XY*200 (FAILS)", &xy);

    // What about non-RLE offset patterns?
    let pattern = b"abcd";
    let repeated: Vec<u8> = pattern.iter().cycle().take(400).copied().collect();
    analyze("abcd*100 (should use offset 4)", &repeated);
}

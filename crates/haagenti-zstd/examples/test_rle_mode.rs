//! Test if RLE mode works for the failing patterns

use haagenti_zstd::block::Sequence;
use haagenti_zstd::compress::{analyze_for_rle, encode_sequences_rle};

fn main() {
    // The failing sequences
    let sequences = vec![Sequence::new(1, 1, 99), Sequence::new(1, 1, 99)];

    let suitability = analyze_for_rle(&sequences);

    println!("RLE Analysis:");
    println!(
        "  LL uniform: {} (code {})",
        suitability.ll_uniform, suitability.ll_code
    );
    println!(
        "  OF uniform: {} (code {})",
        suitability.of_uniform, suitability.of_code
    );
    println!(
        "  ML uniform: {} (code {})",
        suitability.ml_uniform, suitability.ml_code
    );
    println!("  All uniform: {}", suitability.all_uniform());

    if suitability.all_uniform() {
        println!("\nThis pattern IS suitable for RLE mode!");

        // Try encoding with RLE
        let mut output = Vec::new();
        encode_sequences_rle(&sequences, &suitability, &mut output).unwrap();
        println!("RLE encoded: {:02x?}", output);

        // Build a complete block and test
        let mut block = Vec::new();

        // Raw literals header (2 bytes: X and Y)
        block.push(0x10); // Raw, size_format=0, size=2
        block.push(b'X');
        block.push(b'Y');

        // Append RLE-encoded sequences
        block.extend_from_slice(&output);

        println!("\nComplete block ({} bytes): {:02x?}", block.len(), block);
    } else {
        println!("\nNot suitable for RLE mode");
    }
}

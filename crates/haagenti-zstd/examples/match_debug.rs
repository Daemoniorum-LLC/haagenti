//! Debug match finding for repeating text pattern

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    println!("Pattern length: {} bytes", sample.len());

    // Test with different sizes
    for size in [1024, 4096, 16384, 65536] {
        let data: Vec<u8> = sample.iter().cycle().take(size).copied().collect();
        println!("Test data: {} bytes\n", data.len());

        // Create context and find matches
        let mut ctx = CompressContext::new(CompressionLevel::Default);

        // Use internal match finder directly
        let matches = {
            use haagenti_zstd::compress::LazyMatchFinder;
            let mut mf = LazyMatchFinder::new(24);
            mf.find_matches_auto(&data)
        };

        println!("Found {} matches:\n", matches.len());

        // Group matches by offset
        let mut offset_counts: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new();
        for m in &matches {
            let entry = offset_counts.entry(m.offset).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += m.length;
        }

        println!("Offset distribution:");
        let mut offsets: Vec<_> = offset_counts.into_iter().collect();
        offsets.sort_by_key(|&(_, (count, _))| std::cmp::Reverse(count));
        for (offset, (count, total_len)) in offsets.iter().take(20) {
            println!(
                "  offset={}: {} matches, total {} bytes",
                offset, count, total_len
            );
        }

        println!("\nFirst 20 matches:");
        for (i, m) in matches.iter().take(20).enumerate() {
            let src_start = m.position - m.offset;
            println!(
                "  {}: pos={}, offset={}, len={} (src={})",
                i, m.position, m.offset, m.length, src_start
            );
        }

        // Calculate total coverage
        let total_matched: usize = matches.iter().map(|m| m.length).sum();
        let total_literals = data.len() - total_matched;
        println!("\nCoverage:");
        println!("  Matched bytes: {}", total_matched);
        println!("  Literal bytes: {}", total_literals);
        println!(
            "  Match coverage: {:.1}%",
            total_matched as f64 / data.len() as f64 * 100.0
        );

        // What would be ideal?
        // Pattern repeats every 161 bytes, so after position 161, every byte could be matched
        let ideal_matched = data.len() - sample.len();
        println!(
            "\nIdeal (assuming first {} bytes are literals):",
            sample.len()
        );
        println!("  Should match: {} bytes", ideal_matched);
        println!(
            "  Should have ~{} long matches at offset {}",
            (data.len() - sample.len()) / sample.len() + 1,
            sample.len()
        );
        println!("\n============================================================\n");
    }
}

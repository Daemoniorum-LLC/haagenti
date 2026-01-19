//! Check entropy estimates for different patterns

use haagenti_zstd::compress::CompressibilityFingerprint;

fn main() {
    // Test different patterns
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("cyclic 0-255", (0..256).cycle().take(1000).map(|x| x as u8).collect()),
        ("all zeros", vec![0u8; 1000]),
        ("random-looking", (0..1000).map(|i| (i * 7 % 256) as u8).collect()),
        ("text pattern", b"The quick brown fox jumps over the lazy dog. ".repeat(20)),
        ("binary mix", (0..1000).map(|i| ((i * i) % 256) as u8).collect()),
    ];

    println!("{:20} {:>8} {:>12} {:>15}", "Pattern", "Size", "Entropy", "Pattern Type");
    println!("{:-<60}", "");

    for (name, data) in &patterns {
        let fp = CompressibilityFingerprint::analyze(data);
        println!("{:20} {:8} {:12.2} {:15?}", name, data.len(), fp.entropy, fp.pattern);
    }

    println!("\n=== For 1000-byte cyclic pattern ===");
    let cyclic: Vec<u8> = (0..256).cycle().take(1000).map(|x| x as u8).collect();

    // Check if we classify it as random
    let fp = CompressibilityFingerprint::analyze(&cyclic);
    println!("Entropy: {:.2}", fp.entropy);
    println!("Pattern: {:?}", fp.pattern);
    println!("Strategy: {:?}", fp.strategy);

    // The problem: byte-level entropy is high (~8.0) but structural entropy is low
    // Shannon entropy only sees byte frequencies, not sequence patterns
    println!("\nProblem: Cyclic 0-255 has uniform byte distribution (entropy ~8.0)");
    println!("but is highly compressible via 256-byte offset matches!");
}

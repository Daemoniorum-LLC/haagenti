//! Analyze Huffman encoding difference between our encoder and reference

use haagenti_core::Compressor;
use haagenti_zstd::huffman::HuffmanEncoder;
use haagenti_zstd::ZstdCompressor;

fn entropy(data: &[u8]) -> f64 {
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let len = data.len() as f64;
    let mut h = 0.0;
    for &f in &freq {
        if f > 0 {
            let p = f as f64 / len;
            h -= p * p.log2();
        }
    }
    h
}

fn optimal_bits(data: &[u8]) -> usize {
    let h = entropy(data);
    ((h * data.len() as f64) as usize + 7) / 8
}

fn main() {
    // Test data: English text (should compress well with Huffman)
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    // Different sizes
    for size in [1024, 4096, 16384, 65536] {
        let data: Vec<u8> = sample.iter().cycle().take(size).copied().collect();

        // Count symbol frequencies
        let mut freq = [0u32; 256];
        let mut unique = 0;
        for &b in &data {
            if freq[b as usize] == 0 {
                unique += 1;
            }
            freq[b as usize] += 1;
        }

        // Our compression
        let our_compressor = ZstdCompressor::new();
        let our_compressed = our_compressor.compress(&data).unwrap();

        // Reference compression
        let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();

        let gap = (our_compressed.len() as f64 / ref_compressed.len() as f64 - 1.0) * 100.0;

        println!("=== Size: {} bytes ===", size);
        println!("  Unique symbols: {}", unique);
        println!("  Entropy: {:.3} bits/byte", entropy(&data));
        println!("  Optimal: ~{} bytes (entropy-based)", optimal_bits(&data));
        println!("  Reference: {} bytes", ref_compressed.len());
        println!("  Ours: {} bytes", our_compressed.len());
        println!("  Gap: {:+.1}%", gap);

        // Build our Huffman encoder and analyze
        if let Some(encoder) = HuffmanEncoder::build(&data) {
            let encoded_stream = encoder.encode(&data);
            let weight_table = encoder.serialize_weights();

            println!("\n  Our Huffman breakdown:");
            println!("    Weight table: {} bytes", weight_table.len());
            println!("    Encoded stream: {} bytes", encoded_stream.len());
            println!(
                "    Total literals: {} bytes",
                weight_table.len() + encoded_stream.len()
            );
            println!("    Max bits: {}", encoder.max_bits());

            // Show most frequent symbols
            let mut sorted_freq: Vec<_> =
                freq.iter().enumerate().filter(|&(_, f)| *f > 0).collect();
            sorted_freq.sort_by_key(|&(_, f)| std::cmp::Reverse(*f));

            println!("\n  Top 10 symbols (with optimal code lengths):");
            for &(byte, f) in sorted_freq.iter().take(10) {
                let count = *f;
                let char_repr = if byte >= 32 && byte < 127 {
                    format!("'{}'", byte as u8 as char)
                } else {
                    format!("0x{:02x}", byte)
                };
                let freq_pct = count as f64 / data.len() as f64 * 100.0;
                let optimal_bits = -((count as f64 / data.len() as f64).log2());
                println!(
                    "      {}: freq={} ({:.1}%), optimal={:.1} bits",
                    char_repr, count, freq_pct, optimal_bits
                );
            }
        }
        println!();
    }

    // Now test what happens with just literals (no matches)
    println!("\n=== Literals-only test ===");
    let data: Vec<u8> = sample.iter().cycle().take(4096).copied().collect();

    // Extract first 4 stream sizes from reference
    println!("  Analyzing reference block structure...");
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();

    // Skip frame header and analyze blocks
    let mut pos = 4; // Skip magic
    if ref_compressed.len() > pos {
        let fhd = ref_compressed[pos];
        let fcs_size = match (fhd >> 6) & 0x3 {
            0 => {
                if fhd & 0x20 != 0 {
                    1
                } else {
                    0
                }
            }
            1 => 2,
            2 => 4,
            3 => 8,
            _ => 0,
        };
        pos += 1 + fcs_size + if fhd & 0x08 != 0 { 4 } else { 0 };

        if pos + 3 < ref_compressed.len() {
            let block_header = u32::from_le_bytes([
                ref_compressed[pos],
                ref_compressed[pos + 1],
                ref_compressed[pos + 2],
                0,
            ]);
            let block_type = (block_header >> 1) & 0x3;
            let block_size = (block_header >> 3) as usize;

            println!("  Block type: {} (0=Raw, 1=RLE, 2=Compressed)", block_type);
            println!("  Block size: {} bytes", block_size);
            println!("  Reference total: {} bytes", ref_compressed.len());
        }
    }
}

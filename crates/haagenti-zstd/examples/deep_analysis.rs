//! Deep analysis of compression inefficiency

use haagenti_core::Compressor;
use haagenti_zstd::huffman::HuffmanEncoder;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Generate realistic text
    let words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what",
    ];

    let mut seed = 12345u64;
    let mut rng = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as usize
    };

    let mut text = String::new();
    for _ in 0..200 {
        let sentence_len = 5 + (rng() % 10);
        for j in 0..sentence_len {
            let word = words[rng() % words.len()];
            if j == 0 {
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    text.push(first.to_ascii_uppercase());
                    text.extend(chars);
                }
            } else {
                text.push_str(word);
            }
            if j < sentence_len - 1 {
                text.push(' ');
            }
        }
        text.push_str(". ");
    }

    let data = text.as_bytes();
    println!("Input: {} bytes", data.len());

    // Reference compression
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(data), 1).unwrap();

    // Our compression
    let compressor = ZstdCompressor::new();
    let our_compressed = compressor.compress(data).unwrap();

    println!("\n=== Frame Comparison ===");
    println!("Reference: {} bytes", ref_compressed.len());
    println!("Ours: {} bytes", our_compressed.len());
    println!(
        "Gap: {:+.1}%",
        (our_compressed.len() as f64 / ref_compressed.len() as f64 - 1.0) * 100.0
    );

    // Parse and compare block contents
    println!("\n=== Block Content Analysis ===");

    let ref_block = parse_block(&ref_compressed);
    let our_block = parse_block(&our_compressed);

    if let (Some(r), Some(o)) = (ref_block, our_block) {
        println!("\nLiterals section:");
        println!(
            "  Reference: regen={}, compressed={} bytes",
            r.lit_regen_size, r.lit_compressed_size
        );
        println!(
            "  Ours:      regen={}, compressed={} bytes",
            o.lit_regen_size, o.lit_compressed_size
        );
        println!(
            "  Difference: regen={:+}, compressed={:+}",
            o.lit_regen_size as i64 - r.lit_regen_size as i64,
            o.lit_compressed_size as i64 - r.lit_compressed_size as i64
        );

        println!("\nSequences section:");
        println!(
            "  Reference: {} sequences, {} bytes",
            r.num_sequences, r.seq_section_size
        );
        println!(
            "  Ours:      {} sequences, {} bytes",
            o.num_sequences, o.seq_section_size
        );
        println!(
            "  Difference: seq_count={:+}, bytes={:+}",
            o.num_sequences as i64 - r.num_sequences as i64,
            o.seq_section_size as i64 - r.seq_section_size as i64
        );

        // Calculate compression efficiency
        if r.lit_regen_size > 0 {
            println!("\nLiteral compression efficiency:");
            println!(
                "  Reference: {:.1}% of original",
                r.lit_compressed_size as f64 / r.lit_regen_size as f64 * 100.0
            );
            if o.lit_regen_size > 0 {
                println!(
                    "  Ours:      {:.1}% of original",
                    o.lit_compressed_size as f64 / o.lit_regen_size as f64 * 100.0
                );
            }
        }
    }

    // Test Huffman encoding directly on the input
    println!("\n=== Direct Huffman Analysis ===");
    if let Some(encoder) = HuffmanEncoder::build(data) {
        let encoded = encoder.encode(data);
        let weights = encoder.serialize_weights();
        println!("Our Huffman encoder on full input:");
        println!("  Weight table: {} bytes", weights.len());
        println!("  Encoded stream: {} bytes", encoded.len());
        println!("  Total: {} bytes", weights.len() + encoded.len());
        println!(
            "  Efficiency: {:.1}% of input",
            (weights.len() + encoded.len()) as f64 / data.len() as f64 * 100.0
        );
    }

    // Calculate theoretical entropy
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let entropy: f64 = freq
        .iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / data.len() as f64;
            -p * p.log2()
        })
        .sum();

    println!("\nTheoretical limits:");
    println!("  Entropy: {:.3} bits/byte", entropy);
    println!(
        "  Minimum size: {} bytes (entropy-based)",
        (entropy * data.len() as f64 / 8.0) as usize
    );
}

struct BlockInfo {
    lit_regen_size: usize,
    lit_compressed_size: usize,
    num_sequences: usize,
    seq_section_size: usize,
}

fn parse_block(frame: &[u8]) -> Option<BlockInfo> {
    if frame.len() < 4 || &frame[0..4] != &[0x28, 0xB5, 0x2F, 0xFD] {
        return None;
    }

    let mut pos = 4;
    let fhd = frame[pos];
    pos += 1;

    let single_segment = (fhd >> 5) & 0x1 != 0;
    if !single_segment {
        pos += 1;
    }

    let fcs_flag = (fhd >> 6) & 0x3;
    let fcs_size = match fcs_flag {
        0 => {
            if single_segment {
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
    pos += fcs_size;

    if pos + 3 > frame.len() {
        return None;
    }

    let header = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
    let block_type = (header >> 1) & 0x3;
    let block_size = (header >> 3) as usize;
    pos += 3;

    if block_type != 2 || pos + block_size > frame.len() {
        return None;
    }

    let block = &frame[pos..pos + block_size];
    let header_byte = block[0];
    let lit_block_type = header_byte & 0x3;
    let size_format = (header_byte >> 2) & 0x3;

    let (lit_regen_size, lit_compressed_size, header_size) = match lit_block_type {
        0 | 1 => {
            let (size, hdr) = match size_format {
                0 | 1 => ((header_byte >> 3) as usize, 1),
                2 => {
                    let s = ((header_byte >> 4) as usize)
                        | ((block.get(1).copied().unwrap_or(0) as usize) << 4);
                    (s, 2)
                }
                3 => {
                    let s = ((header_byte >> 4) as usize)
                        | ((block.get(1).copied().unwrap_or(0) as usize) << 4)
                        | ((block.get(2).copied().unwrap_or(0) as usize) << 12);
                    (s, 3)
                }
                _ => (0, 1),
            };
            (size, if lit_block_type == 1 { 1 } else { size }, hdr)
        }
        2 | 3 => match size_format {
            0 => {
                if block.len() >= 3 {
                    let combined = ((header_byte >> 4) as u32)
                        | ((block[1] as u32) << 4)
                        | ((block[2] as u32) << 12);
                    let regen = (combined & 0x3FF) as usize;
                    let comp = ((combined >> 10) & 0x3FF) as usize;
                    (regen, comp, 3)
                } else {
                    (0, 0, 1)
                }
            }
            1 => {
                if block.len() >= 4 {
                    let combined = ((header_byte >> 4) as u32)
                        | ((block[1] as u32) << 4)
                        | ((block[2] as u32) << 12)
                        | ((block[3] as u32) << 20);
                    let regen = (combined & 0xFFF) as usize;
                    let comp = ((combined >> 12) & 0xFFF) as usize;
                    (regen, comp, 4)
                } else {
                    (0, 0, 1)
                }
            }
            2 => {
                if block.len() >= 5 {
                    let combined = ((header_byte >> 4) as u32)
                        | ((block[1] as u32) << 4)
                        | ((block[2] as u32) << 12)
                        | ((block[3] as u32) << 20)
                        | ((block[4] as u32) << 28);
                    let regen = (combined & 0x3FFF) as usize;
                    let comp = ((combined >> 14) & 0x3FFF) as usize;
                    (regen, comp, 5)
                } else {
                    (0, 0, 1)
                }
            }
            3 => {
                if block.len() >= 5 {
                    let combined = ((header_byte >> 4) as u64)
                        | ((block[1] as u64) << 4)
                        | ((block[2] as u64) << 12)
                        | ((block[3] as u64) << 20)
                        | ((block[4] as u64) << 28);
                    let regen = (combined & 0x3FFFF) as usize;
                    let comp = ((combined >> 18) & 0x3FFFF) as usize;
                    (regen, comp, 5)
                } else {
                    (0, 0, 1)
                }
            }
            _ => (0, 0, 1),
        },
        _ => (0, 0, 1),
    };

    let lit_section_end = header_size + lit_compressed_size;
    let seq_section = if lit_section_end < block.len() {
        &block[lit_section_end..]
    } else {
        &[]
    };

    let num_sequences = if seq_section.is_empty() {
        0
    } else if seq_section[0] == 0 {
        0
    } else if seq_section[0] < 128 {
        seq_section[0] as usize
    } else if seq_section[0] < 255 && seq_section.len() >= 2 {
        ((seq_section[0] as usize - 128) << 8) + seq_section[1] as usize
    } else if seq_section.len() >= 3 {
        (seq_section[1] as usize) + ((seq_section[2] as usize) << 8) + 0x7F00
    } else {
        0
    };

    Some(BlockInfo {
        lit_regen_size,
        lit_compressed_size,
        num_sequences,
        seq_section_size: seq_section.len(),
    })
}

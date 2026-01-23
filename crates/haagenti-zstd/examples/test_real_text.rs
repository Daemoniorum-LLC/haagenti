//! Test compression with realistic text (not artificially repeated)

use haagenti_core::Compressor;
use haagenti_zstd::ZstdCompressor;

fn main() {
    // Generate pseudo-random but realistic text content
    // This simulates actual English text with natural word patterns
    let words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
        "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
        "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
        "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
        "these", "give", "day", "most", "us",
    ];

    // Simple PRNG for reproducible results
    let mut seed = 12345u64;
    let mut rng = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as usize
    };

    // Generate text with sentence structure
    let mut text = String::new();
    for _ in 0..500 {
        // Generate a sentence
        let sentence_len = 5 + (rng() % 15);
        let mut sentence = String::new();

        for j in 0..sentence_len {
            let word = words[rng() % words.len()];
            if j == 0 {
                // Capitalize first word
                let mut chars = word.chars();
                if let Some(first) = chars.next() {
                    sentence.push(first.to_ascii_uppercase());
                    sentence.extend(chars);
                }
            } else {
                sentence.push_str(word);
            }
            if j < sentence_len - 1 {
                sentence.push(' ');
            }
        }
        sentence.push_str(". ");
        text.push_str(&sentence);
    }

    let data = text.as_bytes();
    println!("Generated text: {} bytes", data.len());
    println!("Sample: {}...", &text[..200.min(text.len())]);

    // Compress with reference
    let ref_compressed = zstd::encode_all(std::io::Cursor::new(data), 1).unwrap();

    // Compress with ours
    let compressor = ZstdCompressor::new();
    let our_compressed = compressor.compress(data).unwrap();

    let gap = (our_compressed.len() as f64 / ref_compressed.len() as f64 - 1.0) * 100.0;

    println!("\nCompression Results:");
    println!(
        "  Reference: {} bytes ({:.2}x ratio)",
        ref_compressed.len(),
        data.len() as f64 / ref_compressed.len() as f64
    );
    println!(
        "  Ours: {} bytes ({:.2}x ratio)",
        our_compressed.len(),
        data.len() as f64 / our_compressed.len() as f64
    );
    println!("  Gap: {:+.1}%", gap);

    // Analyze block structure
    println!("\n--- Reference Block Structure ---");
    analyze_frame(&ref_compressed);

    println!("\n--- Our Block Structure ---");
    analyze_frame(&our_compressed);
}

fn analyze_frame(data: &[u8]) {
    if data.len() < 4 || data[0..4] != [0x28, 0xB5, 0x2F, 0xFD] {
        return;
    }

    let mut pos = 4;
    let fhd = data[pos];
    pos += 1;

    let single_segment = (fhd >> 5) & 0x1 != 0;
    if !single_segment {
        pos += 1; // window descriptor
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

    let mut block_num = 0;
    while pos + 3 <= data.len() {
        let header = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
        let is_last = (header & 1) != 0;
        let block_type = (header >> 1) & 0x3;
        let block_size = (header >> 3) as usize;
        pos += 3;

        let type_name = match block_type {
            0 => "Raw",
            1 => "RLE",
            2 => "Compressed",
            _ => "Unknown",
        };

        if block_type == 2 && pos + block_size <= data.len() {
            let block = &data[pos..pos + block_size];
            let header_byte = block[0];
            let lit_type = header_byte & 0x3;
            let lit_type_name = match lit_type {
                0 => "Raw",
                1 => "RLE",
                2 => "Compressed",
                3 => "Treeless",
                _ => "Unknown",
            };
            println!(
                "  Block {}: {} ({} bytes), literals={}",
                block_num, type_name, block_size, lit_type_name
            );
        } else {
            println!(
                "  Block {}: {} ({} bytes)",
                block_num, type_name, block_size
            );
        }

        pos += block_size;
        block_num += 1;

        if is_last {
            break;
        }
    }
}

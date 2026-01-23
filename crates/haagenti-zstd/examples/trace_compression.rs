//! Trace full compression pipeline

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::{block, CompressContext};

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";

    let data: Vec<u8> = sample.iter().cycle().take(1024).copied().collect();
    println!(
        "Input: {} bytes (pattern {} bytes)",
        data.len(),
        sample.len()
    );

    let mut ctx = CompressContext::new(CompressionLevel::Default);

    // Get the matches from context's internal find_matches
    // We can't access internal state directly, so let's trace through block encoding

    // Compress and analyze result
    let compressed = ctx.compress(&data).unwrap();
    println!("Compressed: {} bytes", compressed.len());

    // Parse the compressed output to see sequence count
    let mut pos = 4; // skip magic
    let fhd = compressed[pos];
    pos += 1;
    let single_segment = (fhd >> 5) & 1 != 0;
    if !single_segment {
        pos += 1;
    }
    let fcs_flag = (fhd >> 6) & 3;
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

    // Block header
    let header = u32::from_le_bytes([compressed[pos], compressed[pos + 1], compressed[pos + 2], 0]);
    let block_type = (header >> 1) & 3;
    let block_size = (header >> 3) as usize;
    pos += 3;

    println!("\nBlock: type={}, size={} bytes", block_type, block_size);

    if block_type == 2 {
        let block = &compressed[pos..pos + block_size];

        // Parse literals header
        let header_byte = block[0];
        let lit_type = header_byte & 3;
        let size_format = (header_byte >> 2) & 3;

        let (lit_regen, lit_comp, lit_header_size) = match lit_type {
            0 => {
                let size = match size_format {
                    0 | 1 => ((header_byte >> 3) as usize, 1),
                    2 => (
                        ((header_byte >> 4) as usize) | ((block[1] as usize) << 4),
                        2,
                    ),
                    3 => (
                        ((header_byte >> 4) as usize)
                            | ((block[1] as usize) << 4)
                            | ((block[2] as usize) << 12),
                        3,
                    ),
                    _ => (0, 1),
                };
                (size.0, size.0, size.1)
            }
            _ => (0, 0, 1),
        };

        println!(
            "Literals: type={}, regen={}, comp={}, header={}",
            lit_type, lit_regen, lit_comp, lit_header_size
        );

        let seq_start = lit_header_size + lit_comp;
        let seq_section = &block[seq_start..];

        println!("Sequences section: {} bytes", seq_section.len());

        if !seq_section.is_empty() {
            let num_seq = if seq_section[0] == 0 {
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
            println!("Number of sequences: {}", num_seq);

            // First few bytes of sequences section
            println!(
                "Seq section bytes: {:02x?}",
                &seq_section[..20.min(seq_section.len())]
            );
        }
    }

    // Now test with direct match finding
    println!("\n--- Direct match finding (same params as context) ---");
    use haagenti_zstd::compress::LazyMatchFinder;
    let mut mf = LazyMatchFinder::new(24);
    let matches = mf.find_matches_auto(&data);
    println!("Matches found: {}", matches.len());

    // Convert to sequences
    let (literals, sequences) = block::matches_to_sequences(&data, &matches);
    println!("After matches_to_sequences:");
    println!("  Literals: {} bytes", literals.len());
    println!("  Sequences: {}", sequences.len());

    // Show first few sequences
    for (i, seq) in sequences.iter().take(10).enumerate() {
        println!(
            "  Seq {}: ll={}, offset={}, ml={}",
            i, seq.literal_length, seq.offset, seq.match_length
        );
    }

    // Check if uniform
    use haagenti_zstd::compress::analyze_for_rle;
    let suitability = analyze_for_rle(&sequences);
    println!("\nRLE Suitability:");
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
}

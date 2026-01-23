//! Debug sequence encoding issue

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::{block, CompressContext};

fn main() {
    // Simple failing pattern: X*100 Y*100
    let mut data = Vec::new();
    data.extend(vec![b'X'; 100]);
    data.extend(vec![b'Y'; 100]);

    println!("Input size: {} bytes", data.len());

    // Create context and find matches
    let mut ctx = CompressContext::new(CompressionLevel::Fast);

    // Use our match finder
    let matches = {
        let mut mf = haagenti_zstd::compress::LazyMatchFinder::new(8);
        mf.find_matches_auto(&data)
    };

    println!("\nMatches found: {}", matches.len());
    for (i, m) in matches.iter().enumerate() {
        println!(
            "  Match {}: pos={}, offset={}, len={}",
            i, m.position, m.offset, m.length
        );
        // Verify the match
        let src_start = m.position - m.offset;
        let src_data = &data[src_start..src_start + m.length.min(10)];
        let dst_data = &data[m.position..m.position + m.length.min(10)];
        println!("    src[{}..]: {:?}", src_start, src_data);
        println!("    dst[{}..]: {:?}", m.position, dst_data);
    }

    // Convert to sequences
    let (literals, sequences) = block::matches_to_sequences(&data, &matches);
    println!("\nLiterals: {} bytes", literals.len());
    println!("Sequences: {}", sequences.len());
    for (i, seq) in sequences.iter().enumerate() {
        println!(
            "  Seq {}: ll={}, offset={}, ml={}",
            i, seq.literal_length, seq.offset, seq.match_length
        );
    }

    // Compress
    let compressed = ctx.compress(&data).unwrap();
    println!("\nCompressed: {} bytes", compressed.len());
    println!("Hex: {:02x?}", &compressed[..compressed.len().min(50)]);

    // Try reference decompression
    match zstd::decode_all(std::io::Cursor::new(&compressed)) {
        Ok(dec) => {
            if dec == data {
                println!("\nReference decompression: OK");
            } else {
                println!("\nReference decompression: MISMATCH");
                println!("  Expected {} bytes, got {} bytes", data.len(), dec.len());
            }
        }
        Err(e) => {
            println!("\nReference decompression: FAILED - {}", e);
        }
    }
}

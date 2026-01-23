//! Trace match finding for failing case

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn main() {
    // The failing 2-line case
    let data = b"[2024-01-01 10:00:00] INFO Request #0\n[2024-01-02 10:01:00] INFO Request #1000\n";
    println!("Input: {} bytes", data.len());
    println!("Data: {:?}", String::from_utf8_lossy(data));

    // Create compress context and find matches
    let _ctx = CompressContext::new(CompressionLevel::Fast);

    // We need to access match_finder - let's create our own
    use haagenti_zstd::compress::MatchFinder;

    let mut mf = MatchFinder::new(8);
    let matches = mf.find_matches(data);

    println!("\n=== Matches Found ({}) ===", matches.len());
    for (i, m) in matches.iter().enumerate() {
        println!(
            "Match {}: pos={}, offset={}, length={}",
            i, m.position, m.offset, m.length
        );

        // Show what this match refers to
        let match_start = m.position - m.offset;
        let match_content = &data[match_start..match_start + m.length.min(30)];
        let current_content = &data[m.position..m.position + m.length.min(30)];

        println!(
            "  Source (offset back): {:?}",
            String::from_utf8_lossy(match_content)
        );
        println!(
            "  Current position:     {:?}",
            String::from_utf8_lossy(current_content)
        );

        // Verify the match is valid
        if match_content != current_content {
            println!("  !!! INVALID MATCH !!!");
        }
    }

    // Now trace what literals and sequences should be generated
    println!("\n=== Expected Literals and Sequences ===");
    use haagenti_zstd::compress::block::matches_to_sequences;

    let (literals, sequences) = matches_to_sequences(data, &matches);

    println!("Literals: {} bytes", literals.len());
    println!("  Content: {:?}", String::from_utf8_lossy(&literals));

    println!("\nSequences: {}", sequences.len());
    for (i, seq) in sequences.iter().enumerate() {
        println!(
            "  Seq {}: LL={}, offset={}, ML={}",
            i, seq.literal_length, seq.offset, seq.match_length
        );
    }

    // Verify that literals + matches reconstruct the original
    println!("\n=== Reconstruction Check ===");
    let mut reconstructed = Vec::new();
    let mut lit_pos = 0;
    let mut data_pos = 0;

    for seq in &sequences {
        // Add literals
        let ll = seq.literal_length as usize;
        if lit_pos + ll > literals.len() {
            println!(
                "ERROR: Not enough literals at lit_pos={}, need {} more",
                lit_pos, ll
            );
            break;
        }
        reconstructed.extend_from_slice(&literals[lit_pos..lit_pos + ll]);
        lit_pos += ll;
        data_pos += ll;

        // Add match (offset is already encoded, need to decode)
        let ml = seq.match_length as usize;
        let offset = if seq.offset <= 3 {
            // Repeat offset - for now just note it
            println!("  Seq uses repeat offset {}", seq.offset);
            seq.offset as usize // This is wrong, but let's see
        } else {
            (seq.offset - 3) as usize
        };

        if data_pos < offset {
            println!(
                "ERROR: Match offset {} exceeds current position {}",
                offset, data_pos
            );
            break;
        }

        let match_start = reconstructed.len() - offset;
        for j in 0..ml {
            let byte = reconstructed[match_start + j];
            reconstructed.push(byte);
        }
        data_pos += ml;
    }

    // Add trailing literals
    if lit_pos < literals.len() {
        reconstructed.extend_from_slice(&literals[lit_pos..]);
    }

    println!("\nReconstructed: {} bytes", reconstructed.len());
    if reconstructed == data {
        println!("MATCH: Reconstruction is correct!");
    } else {
        println!("MISMATCH: Reconstruction differs!");
        println!("  Expected: {:?}", String::from_utf8_lossy(data));
        println!("  Got:      {:?}", String::from_utf8_lossy(&reconstructed));
    }
}

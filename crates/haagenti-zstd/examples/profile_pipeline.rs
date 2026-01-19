//! Detailed pipeline profiling for 64KB throughput investigation

use std::time::Instant;
use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::{
    CompressContext, CompressibilityFingerprint, LazyMatchFinder,
    block, Match,
};

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";
    let data: Vec<u8> = sample.iter().cycle().take(65536).copied().collect();

    let iterations = 1000;

    println!("=== Detailed Pipeline Profiling (64KB text) ===\n");

    // 1. Profile fingerprint analysis
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(CompressibilityFingerprint::analyze(&data));
    }
    let fingerprint_time = start.elapsed();
    println!("1. Fingerprint analysis: {:.3}ms avg",
             fingerprint_time.as_secs_f64() * 1000.0 / iterations as f64);

    // 2. Profile match finder (with pre-warmed state)
    let mut mf = LazyMatchFinder::new(8);
    // Warmup
    for _ in 0..10 {
        let _ = mf.find_matches_auto(&data);
    }

    let start = Instant::now();
    let mut matches_storage = Vec::new();
    for _ in 0..iterations {
        matches_storage = std::hint::black_box(mf.find_matches_auto(&data));
    }
    let match_find_time = start.elapsed();
    println!("2. Match finding: {:.3}ms avg ({} matches found)",
             match_find_time.as_secs_f64() * 1000.0 / iterations as f64,
             matches_storage.len());

    // 3. Profile matches_to_sequences
    let start = Instant::now();
    let mut literals_storage = Vec::new();
    let mut seqs_storage = Vec::new();
    for _ in 0..iterations {
        let (literals, seqs) = std::hint::black_box(
            block::matches_to_sequences(&data, &matches_storage)
        );
        literals_storage = literals;
        seqs_storage = seqs;
    }
    let seq_convert_time = start.elapsed();
    println!("3. Convert to sequences: {:.3}ms avg ({} literals, {} sequences)",
             seq_convert_time.as_secs_f64() * 1000.0 / iterations as f64,
             literals_storage.len(), seqs_storage.len());

    // 4. Profile literals encoding
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = Vec::with_capacity(data.len());
        std::hint::black_box(block::encode_literals(&literals_storage, &mut output)).unwrap();
    }
    let literals_time = start.elapsed();
    println!("4. Encode literals: {:.3}ms avg",
             literals_time.as_secs_f64() * 1000.0 / iterations as f64);

    // 5. Profile sequence encoding
    let suitability = haagenti_zstd::compress::analyze_for_rle(&seqs_storage);
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = Vec::with_capacity(data.len());
        std::hint::black_box(
            haagenti_zstd::compress::encode_sequences_fse_with_encoded(&suitability.encoded, &mut output)
        ).unwrap();
    }
    let seq_encode_time = start.elapsed();
    println!("5. Encode sequences: {:.3}ms avg",
             seq_encode_time.as_secs_f64() * 1000.0 / iterations as f64);

    // 6. Profile full compression pipeline (for comparison)
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    // Warmup
    for _ in 0..100 {
        let _ = ctx.compress(&data);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let full_time = start.elapsed();
    println!("\n6. Full compression: {:.3}ms avg ({:.1} MB/s)",
             full_time.as_secs_f64() * 1000.0 / iterations as f64,
             (data.len() * iterations) as f64 / full_time.as_secs_f64() / 1_000_000.0);

    // 7. Reference for comparison
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(zstd::encode_all(std::io::Cursor::new(&data), 1));
    }
    let ref_time = start.elapsed();
    println!("7. Reference zstd: {:.3}ms avg ({:.1} MB/s)",
             ref_time.as_secs_f64() * 1000.0 / iterations as f64,
             (data.len() * iterations) as f64 / ref_time.as_secs_f64() / 1_000_000.0);

    // Sum of components
    let component_sum = fingerprint_time + match_find_time + seq_convert_time + literals_time + seq_encode_time;
    println!("\n=== Summary ===");
    println!("Component sum: {:.3}ms", component_sum.as_secs_f64() * 1000.0 / iterations as f64);
    println!("Full pipeline: {:.3}ms", full_time.as_secs_f64() * 1000.0 / iterations as f64);
    println!("Reference:     {:.3}ms", ref_time.as_secs_f64() * 1000.0 / iterations as f64);
    println!("Overhead: {:.3}ms (explains {:.1}% of gap)",
             (full_time.as_secs_f64() - ref_time.as_secs_f64()) * 1000.0 / iterations as f64,
             (full_time.as_secs_f64() - ref_time.as_secs_f64()) / ref_time.as_secs_f64() * 100.0);

    // Breakdown percentages
    let total_components = component_sum.as_secs_f64();
    println!("\n=== Component Breakdown ===");
    println!("Fingerprint:  {:5.1}%", fingerprint_time.as_secs_f64() / total_components * 100.0);
    println!("Match find:   {:5.1}%", match_find_time.as_secs_f64() / total_components * 100.0);
    println!("Seq convert:  {:5.1}%", seq_convert_time.as_secs_f64() / total_components * 100.0);
    println!("Literals:     {:5.1}%", literals_time.as_secs_f64() / total_components * 100.0);
    println!("Seq encode:   {:5.1}%", seq_encode_time.as_secs_f64() / total_components * 100.0);
}

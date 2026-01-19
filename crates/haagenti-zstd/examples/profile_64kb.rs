//! Track down the 0.15ms overhead

use std::time::Instant;
use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";
    let data: Vec<u8> = sample.iter().cycle().take(65536).copied().collect();
    
    // Use a pre-warmed context
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    // Warmup
    for _ in 0..100 {
        let _ = ctx.compress(&data);
    }
    
    let iterations = 1000;
    
    // Time with pre-warmed context
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let our_time = start.elapsed();
    
    // Time reference
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(zstd::encode_all(std::io::Cursor::new(&data), 1));
    }
    let ref_time = start.elapsed();
    
    println!("Pre-warmed context:");
    println!("  Haagenti: {:.3}ms avg ({:.1} MB/s)", 
             our_time.as_secs_f64() * 1000.0 / iterations as f64,
             (data.len() * iterations) as f64 / our_time.as_secs_f64() / 1_000_000.0);
    println!("  Reference: {:.3}ms avg ({:.1} MB/s)", 
             ref_time.as_secs_f64() * 1000.0 / iterations as f64,
             (data.len() * iterations) as f64 / ref_time.as_secs_f64() / 1_000_000.0);
    println!("  Ratio: {:.1}%", our_time.as_secs_f64() / ref_time.as_secs_f64() * 100.0);
    
    // Now test with fresh context each time (to see context creation overhead)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let fresh_time = start.elapsed();
    
    println!("\nFresh context each time:");
    println!("  Haagenti: {:.3}ms avg ({:.1} MB/s)",
             fresh_time.as_secs_f64() * 1000.0 / iterations as f64,
             (data.len() * iterations) as f64 / fresh_time.as_secs_f64() / 1_000_000.0);
    println!("  Context overhead: {:.3}ms",
             (fresh_time.as_secs_f64() - our_time.as_secs_f64()) * 1000.0 / iterations as f64);
}

//! Investigate the hidden overhead in full compression

use haagenti_core::CompressionLevel;
use haagenti_zstd::compress::CompressContext;
use std::time::Instant;

fn main() {
    let sample = b"The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump! \
                   The five boxing wizards jump quickly. ";
    let data: Vec<u8> = sample.iter().cycle().take(65536).copied().collect();

    let iterations = 1000;

    println!("=== Hidden Overhead Investigation ===\n");

    // Test 1: Context creation overhead
    let start = Instant::now();
    for _ in 0..iterations {
        let ctx = std::hint::black_box(CompressContext::new(CompressionLevel::Fast));
        drop(ctx);
    }
    let ctx_create_time = start.elapsed();
    println!(
        "1. Context creation: {:.3}ms avg",
        ctx_create_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Test 2: Fresh context each time (no reuse)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut ctx = CompressContext::new(CompressionLevel::Fast);
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let fresh_ctx_time = start.elapsed();
    println!(
        "2. Fresh context each time: {:.3}ms avg",
        fresh_ctx_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Test 3: Reused context (first run - cold)
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let reused_ctx_time = start.elapsed();
    println!(
        "3. Reused context (first 1000): {:.3}ms avg",
        reused_ctx_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Test 4: Reused context after warmup
    for _ in 0..1000 {
        let _ = ctx.compress(&data);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(ctx.compress(&data));
    }
    let warmed_ctx_time = start.elapsed();
    println!(
        "4. Reused context (after 1000 warmup): {:.3}ms avg ({:.1} MB/s)",
        warmed_ctx_time.as_secs_f64() * 1000.0 / iterations as f64,
        (data.len() * iterations) as f64 / warmed_ctx_time.as_secs_f64() / 1_000_000.0
    );

    // Test 5: Reference
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = std::hint::black_box(zstd::encode_all(std::io::Cursor::new(&data), 1));
    }
    let ref_time = start.elapsed();
    println!(
        "5. Reference zstd: {:.3}ms avg ({:.1} MB/s)",
        ref_time.as_secs_f64() * 1000.0 / iterations as f64,
        (data.len() * iterations) as f64 / ref_time.as_secs_f64() / 1_000_000.0
    );

    // Test 6: Check output size
    let mut ctx = CompressContext::new(CompressionLevel::Fast);
    let our_result = ctx.compress(&data).unwrap();
    let ref_result = zstd::encode_all(std::io::Cursor::new(&data), 1).unwrap();
    println!("\n=== Compression Results ===");
    println!("Input size: {} bytes", data.len());
    println!(
        "Our output: {} bytes (ratio: {:.2}x)",
        our_result.len(),
        data.len() as f64 / our_result.len() as f64
    );
    println!(
        "Ref output: {} bytes (ratio: {:.2}x)",
        ref_result.len(),
        data.len() as f64 / ref_result.len() as f64
    );

    // Test 7: Profile Vec allocation in compress
    let start = Instant::now();
    for _ in 0..iterations {
        let v = Vec::<u8>::with_capacity(data.len() + 32);
        std::hint::black_box(v);
    }
    let vec_alloc_time = start.elapsed();
    println!(
        "\n6. Vec allocation ({} bytes): {:.3}ms avg",
        data.len() + 32,
        vec_alloc_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Test 8: Just hash table operations (simulating reset overhead)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut table = vec![0u32; 65536]; // 256KB hash table
        std::hint::black_box(&mut table);
    }
    let hash_table_alloc = start.elapsed();
    println!(
        "7. Hash table alloc (256KB): {:.3}ms avg",
        hash_table_alloc.as_secs_f64() * 1000.0 / iterations as f64
    );

    // Test 9: Chain table resize
    let start = Instant::now();
    for _ in 0..iterations {
        let mut chain = Vec::<u32>::new();
        chain.resize(65536, 0);
        std::hint::black_box(&mut chain);
    }
    let chain_resize = start.elapsed();
    println!(
        "8. Chain table resize (256KB): {:.3}ms avg",
        chain_resize.as_secs_f64() * 1000.0 / iterations as f64
    );
}

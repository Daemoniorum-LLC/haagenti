use std::time::Instant;
use haagenti_core::{Compressor, Decompressor};
use haagenti_zstd::{ZstdCompressor, ZstdDecompressor};

fn main() {
    let sizes = [4096usize, 16384, 65536];
    
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║          Haagenti-Zstd vs Reference Zstd Comparison              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    println!("=== COMPRESSION ===\n");
    println!("Size     │ Haagenti     │ Reference    │ Speed %  │ Ratio H  │ Ratio R");
    println!("─────────┼──────────────┼──────────────┼──────────┼──────────┼─────────");
    
    for size in sizes {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            data.extend_from_slice(pattern);
        }
        data.truncate(size);
        
        let compressor = ZstdCompressor::new();
        let start = Instant::now();
        let iterations = 100;
        let mut haagenti_compressed = Vec::new();
        for _ in 0..iterations {
            haagenti_compressed = compressor.compress(&data).unwrap();
        }
        let haagenti_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let haagenti_mbs = (size as f64 / 1024.0 / 1024.0) / (haagenti_us / 1_000_000.0);
        let haagenti_ratio = data.len() as f64 / haagenti_compressed.len() as f64;
        
        let start = Instant::now();
        let mut ref_compressed = Vec::new();
        for _ in 0..iterations {
            ref_compressed = zstd::encode_all(data.as_slice(), 3).unwrap();
        }
        let ref_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let ref_mbs = (size as f64 / 1024.0 / 1024.0) / (ref_us / 1_000_000.0);
        let ref_ratio = data.len() as f64 / ref_compressed.len() as f64;
        
        let speed_pct = haagenti_mbs / ref_mbs * 100.0;
        
        println!("{:>7} │ {:>8.1} MB/s │ {:>8.1} MB/s │ {:>6.1}%  │ {:>7.2}x │ {:>6.2}x", 
            format!("{} KB", size / 1024),
            haagenti_mbs,
            ref_mbs,
            speed_pct,
            haagenti_ratio,
            ref_ratio
        );
    }
    
    println!("\n=== DECOMPRESSION ===\n");
    println!("Size     │ Haagenti     │ Reference    │ Speed %");
    println!("─────────┼──────────────┼──────────────┼─────────");
    
    for size in sizes {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            data.extend_from_slice(pattern);
        }
        data.truncate(size);
        
        let compressor = ZstdCompressor::new();
        let haagenti_compressed = compressor.compress(&data).unwrap();
        let ref_compressed = zstd::encode_all(data.as_slice(), 3).unwrap();
        
        let decompressor = ZstdDecompressor::new();
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = decompressor.decompress(&haagenti_compressed).unwrap();
        }
        let haagenti_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let haagenti_mbs = (size as f64 / 1024.0 / 1024.0) / (haagenti_us / 1_000_000.0);
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = zstd::decode_all(ref_compressed.as_slice()).unwrap();
        }
        let ref_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let ref_mbs = (size as f64 / 1024.0 / 1024.0) / (ref_us / 1_000_000.0);
        
        let speed_pct = haagenti_mbs / ref_mbs * 100.0;
        
        println!("{:>7} │ {:>8.1} MB/s │ {:>8.1} MB/s │ {:>6.1}%", 
            format!("{} KB", size / 1024),
            haagenti_mbs,
            ref_mbs,
            speed_pct
        );
    }
    
    println!("\n=== SUMMARY ===");
    println!("• Haagenti is a pure Rust implementation (no C dependencies)");
    println!("• Reference zstd uses optimized C library with years of tuning");
    println!("• Current focus: correctness first, performance optimization ongoing");
    println!();
}

//! Adaptive Model Compression - Per-tensor optimal retention
//!
//! Compresses each tensor with individually optimized retention based on
//! spectral analysis, potentially achieving better compression at same quality.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features turbo --example compress_adaptive -- \
//!     --model /path/to/model \
//!     --output /path/to/output \
//!     --target-quality 0.95
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use haagenti::adaptive::AdaptiveSpectralEncoder;
use haagenti::pipeline::{discover_shards, ShardReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut output_dir = PathBuf::from("./compressed-adaptive");
    let mut target_quality = 0.95f32;
    let mut num_workers = num_cpus();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_path = args.get(i).cloned().unwrap_or_default();
            }
            "--output" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(args.get(i).cloned().unwrap_or_default());
            }
            "--target-quality" | "-q" => {
                i += 1;
                target_quality = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.95);
            }
            "--workers" | "-w" => {
                i += 1;
                num_workers = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(num_cpus());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Error: --model is required");
        print_help();
        std::process::exit(1);
    }

    println!("\nAdaptive Model Compression");
    println!("==========================");
    println!("Model:           {}", model_path);
    println!("Output:          {}", output_dir.display());
    println!("Target Quality:  {:.0}%", target_quality * 100.0);
    println!("Workers:         {}", num_workers);
    println!();

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Discover shards
    let model_dir = std::path::Path::new(&model_path);
    let shards = discover_shards(model_dir)?;

    if shards.is_empty() {
        eprintln!("No safetensors files found in {}", model_path);
        std::process::exit(1);
    }

    println!("Found {} shard(s)", shards.len());

    // Collect all tensor info: (shard_idx, name, shape)
    let mut all_tensors: Vec<(usize, String, Vec<usize>)> = Vec::new();

    for (shard_idx, shard_path) in shards.iter().enumerate() {
        let reader = ShardReader::open(shard_path)?;
        for entry in reader.tensors() {
            // Skip 1D tensors (biases, layernorms)
            if entry.shape.len() >= 2 {
                all_tensors.push((shard_idx, entry.name.clone(), entry.shape.clone()));
            }
        }
    }

    println!("Found {} 2D tensors to compress\n", all_tensors.len());

    // Set up thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .ok();

    let start_time = Instant::now();
    let completed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let total_input = AtomicU64::new(0);
    let total_output = AtomicU64::new(0);
    let total_tensors = all_tensors.len();

    // Track retention distribution
    let retention_buckets: Vec<AtomicUsize> = (0..10).map(|_| AtomicUsize::new(0)).collect();

    println!("Compressing {} tensors with adaptive retention...\n", total_tensors);

    // Process tensors in parallel
    all_tensors.par_iter().for_each(|(shard_idx, name, shape)| {
        // Read tensor data
        let reader = match ShardReader::open(&shards[*shard_idx]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error opening shard for {}: {}", name, e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        let data_f32: Vec<f32> = match reader.tensor_f32(name) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error reading {}: {}", name, e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        let original_size = data_f32.len() * 4;
        total_input.fetch_add(original_size as u64, Ordering::Relaxed);

        // Determine 2D dimensions
        let (width, height) = if shape.len() == 2 {
            (shape[1], shape[0])
        } else {
            let total: usize = shape.iter().product();
            let w = shape.last().copied().unwrap_or(1);
            (w, total / w)
        };

        // Encode using adaptive spectral encoder
        // Use 1 fragment with essential_ratio=1.0 (all retained coeffs in fragment 0)
        let encoder = AdaptiveSpectralEncoder::new(target_quality, 1);
        let (meta, fragments) = match encoder.encode_2d(&data_f32, width, height) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error encoding {}: {}", name, e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        // Track retention distribution
        let bucket = ((meta.retention_ratio * 10.0) as usize).min(9);
        retention_buckets[bucket].fetch_add(1, Ordering::Relaxed);

        // Serialize fragments
        let mut compressed_data = Vec::new();
        compressed_data.extend_from_slice(&(fragments.len() as u16).to_le_bytes());

        for frag in &fragments {
            compressed_data.extend_from_slice(&frag.index.to_le_bytes());
            compressed_data.extend_from_slice(&frag.flags.to_le_bytes());
            compressed_data.extend_from_slice(&frag.checksum.to_le_bytes());
            compressed_data.extend_from_slice(&(frag.data.len() as u32).to_le_bytes());
            compressed_data.extend_from_slice(&frag.data);
        }

        // Apply zstd compression
        let final_data = match zstd::encode_all(&compressed_data[..], 1) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error compressing {}: {}", name, e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        let output_size = final_data.len();
        total_output.fetch_add(output_size as u64, Ordering::Relaxed);

        // Write HCT file
        let safe_name = name.replace('.', "_");
        let output_path = output_dir.join(format!("{}.hct", safe_name));

        if let Err(e) = fs::File::create(&output_path).and_then(|mut f| f.write_all(&final_data)) {
            eprintln!("Error writing {}: {}", name, e);
            failed.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        let ratio = original_size as f64 / output_size as f64;
        println!("[{}/{}] {} - {:.1}x (retention: {:.0}%)",
                 done, total_tensors, name, ratio, meta.retention_ratio * 100.0);
    });

    let elapsed = start_time.elapsed().as_secs_f64();
    let input_gb = total_input.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
    let output_gb = total_output.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
    let compression_ratio = if output_gb > 0.0 { input_gb / output_gb } else { 0.0 };
    let completed_count = completed.load(Ordering::Relaxed);
    let failed_count = failed.load(Ordering::Relaxed);

    println!("\nCompression Complete!");
    println!("=====================");
    println!("Output:              {}", output_dir.display());
    println!("Tensors:             {}/{} compressed ({} failed)", completed_count, total_tensors, failed_count);
    println!("Input Size:          {:.2} GB", input_gb);
    println!("Output Size:         {:.2} GB", output_gb);
    println!("Compression:         {:.1}x", compression_ratio);
    println!("Time:                {:.1} seconds", elapsed);
    println!("Throughput:          {:.1} MB/s", (input_gb * 1000.0) / elapsed);

    // Print retention distribution
    println!("\nRetention Distribution:");
    for (i, count) in retention_buckets.iter().enumerate() {
        let c = count.load(Ordering::Relaxed);
        if c > 0 {
            let low = i as f32 * 10.0;
            let high = (i + 1) as f32 * 10.0;
            println!("  {:.0}-{:.0}%: {} tensors", low, high, c);
        }
    }

    println!("\nDone!");

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

fn print_help() {
    println!("Adaptive Model Compression");
    println!();
    println!("Compresses each tensor with individually optimized retention");
    println!("based on spectral analysis of the DCT coefficients.");
    println!();
    println!("USAGE:");
    println!("    compress_adaptive [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -m, --model <PATH>           Model path (required)");
    println!("    -o, --output <PATH>          Output directory [default: ./compressed-adaptive]");
    println!("    -q, --target-quality <RATIO> Target quality 0.0-1.0 [default: 0.95]");
    println!("    -w, --workers <N>            Number of parallel workers [default: auto]");
    println!("    -h, --help                   Print help");
    println!();
    println!("EXAMPLES:");
    println!("    # Compress with 95% target quality");
    println!("    compress_adaptive -m /path/to/model -o ./output -q 0.95");
    println!();
    println!("    # Compress with 90% target quality (more aggressive)");
    println!("    compress_adaptive -m /path/to/model -o ./output -q 0.90");
}

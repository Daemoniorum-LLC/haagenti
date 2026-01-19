//! Selective Model Compression - MLP-only strategy
//!
//! Compresses only MLP layers (gate_proj, up_proj, down_proj), keeping attention
//! layers uncompressed for maximum quality with good compression.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features turbo --example compress_selective -- \
//!     --model /path/to/model \
//!     --output /path/to/output \
//!     --mlp-retention 0.45
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use haagenti::compressive::CompressiveSpectralEncoder;
use haagenti::pipeline::{discover_shards, ShardReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut output_dir = PathBuf::from("./compressed-selective");
    let mut mlp_retention = 0.45f32;
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
            "--mlp-retention" | "-r" => {
                i += 1;
                mlp_retention = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.45);
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

    println!("\nSelective Model Compression (MLP-Only)");
    println!("=======================================");
    println!("Model:               {}", model_path);
    println!("Output:              {}", output_dir.display());
    println!("MLP Retention:       {:.0}%", mlp_retention * 100.0);
    println!("Attention:           100% (uncompressed)");
    println!("Workers:             {}", num_workers);
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

    // Categorize tensors
    let mlp_tensors: Vec<_> = all_tensors.iter()
        .filter(|(_, name, _)| is_mlp_tensor(name))
        .cloned()
        .collect();
    let attention_tensors: Vec<_> = all_tensors.iter()
        .filter(|(_, name, _)| is_attention_tensor(name))
        .collect();
    let other_tensors: Vec<_> = all_tensors.iter()
        .filter(|(_, name, _)| !is_mlp_tensor(name) && !is_attention_tensor(name))
        .collect();

    println!("\nTensor breakdown:");
    println!("  MLP tensors:       {} (will compress at {:.0}%)", mlp_tensors.len(), mlp_retention * 100.0);
    println!("  Attention tensors: {} (keeping uncompressed)", attention_tensors.len());
    println!("  Other tensors:     {} (skipped - loaded from safetensors)", other_tensors.len());
    println!();

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
    let total_mlp = mlp_tensors.len();

    println!("Compressing {} MLP tensors...\n", total_mlp);

    // Process MLP tensors in parallel
    mlp_tensors.par_iter().for_each(|(shard_idx, name, shape)| {
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

        // Encode using compressive spectral encoder
        // CRITICAL: With num_fragments=1, we must set essential_ratio=1.0 to store ALL
        // retained coefficients in fragment 0. Otherwise only 20% of retained are stored.
        let encoder = CompressiveSpectralEncoder::new(1, mlp_retention)
            .with_essential_ratio(1.0);
        let fragments = match encoder.encode_2d(&data_f32, width, height) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error encoding {}: {}", name, e);
                failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

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
        println!("[{}/{}] {} - {:.1}x", done, total_mlp, name, ratio);
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
    println!("MLP Tensors:         {}/{} compressed ({} failed)", completed_count, total_mlp, failed_count);
    println!("Attention Tensors:   {} (use from safetensors)", attention_tensors.len());
    println!("MLP Input Size:      {:.2} GB", input_gb);
    println!("MLP Output Size:     {:.2} GB", output_gb);
    println!("MLP Compression:     {:.1}x", compression_ratio);
    println!("Time:                {:.1} seconds", elapsed);
    println!("Throughput:          {:.1} MB/s", (input_gb * 1000.0) / elapsed);

    // Estimate overall compression (MLP is ~67% of model for Llama)
    let mlp_fraction = 0.67;
    let overall_compression = 1.0 / (mlp_fraction / compression_ratio + (1.0 - mlp_fraction));
    println!("\nEstimated Overall:   {:.2}x (MLP={:.0}% of params)", overall_compression, mlp_fraction * 100.0);

    println!("\nDone!");

    Ok(())
}

/// Check if tensor name indicates MLP layer
fn is_mlp_tensor(name: &str) -> bool {
    name.contains("mlp.gate_proj") ||
    name.contains("mlp.up_proj") ||
    name.contains("mlp.down_proj") ||
    name.contains("mlp_gate_proj") ||
    name.contains("mlp_up_proj") ||
    name.contains("mlp_down_proj") ||
    // Handle various naming conventions
    (name.contains("mlp") && (name.contains("gate") || name.contains("up") || name.contains("down")))
}

/// Check if tensor name indicates attention layer
fn is_attention_tensor(name: &str) -> bool {
    name.contains("self_attn.q_proj") ||
    name.contains("self_attn.k_proj") ||
    name.contains("self_attn.v_proj") ||
    name.contains("self_attn.o_proj") ||
    name.contains("self_attn_q_proj") ||
    name.contains("self_attn_k_proj") ||
    name.contains("self_attn_v_proj") ||
    name.contains("self_attn_o_proj") ||
    (name.contains("attn") && (name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") || name.contains("o_proj")))
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

fn print_help() {
    println!("Selective Model Compression (MLP-Only)");
    println!();
    println!("Compresses only MLP layers while keeping attention uncompressed.");
    println!("This exploits MLP robustness for better compression with quality preservation.");
    println!();
    println!("USAGE:");
    println!("    compress_selective [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -m, --model <PATH>           Model path (required)");
    println!("    -o, --output <PATH>          Output directory [default: ./compressed-selective]");
    println!("    -r, --mlp-retention <RATIO>  MLP retention ratio 0.0-1.0 [default: 0.45]");
    println!("    -w, --workers <N>            Number of parallel workers [default: auto]");
    println!("    -h, --help                   Print help");
    println!();
    println!("STRATEGY:");
    println!("    MLP layers (gate/up/down):   Compressed at specified retention");
    println!("    Attention layers (Q/K/V/O):  100% (loaded from safetensors)");
    println!("    Embeddings/LayerNorms:       100% (loaded from safetensors)");
}

//! Turbo Model Compression Pipeline
//!
//! High-performance parallel compression using rayon, with optional GPU acceleration.
//!
//! ## Usage
//!
//! ```bash
//! # 8 workers (default)
//! cargo run --release --features "parallel,zstd" --example compress_turbo -- \
//!     --model Qwen/Qwen2.5-Coder-7B-Instruct \
//!     --output /tmp/compressed-turbo \
//!     --retention 0.20
//!
//! # Maximum parallelism (16 workers)
//! cargo run --release --features "turbo" --example compress_turbo -- \
//!     --model Qwen/Qwen2.5-Coder-7B-Instruct \
//!     --output /tmp/compressed-turbo \
//!     --retention 0.20 \
//!     --workers 16
//!
//! # With GPU acceleration (requires cuda feature and NVIDIA GPU)
//! cargo run --release --features "cuda" --example compress_turbo -- \
//!     --model Qwen/Qwen2.5-Coder-7B-Instruct \
//!     --output /tmp/compressed-turbo \
//!     --retention 0.20 \
//!     --gpu
//! ```

use std::path::PathBuf;

use haagenti::pipeline::turbo::{TurboConfig, TurboPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut config = TurboConfig::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                config.model = args.get(i).cloned().unwrap_or_default();
            }
            "--output" | "-o" => {
                i += 1;
                config.output_dir = PathBuf::from(args.get(i).cloned().unwrap_or_default());
            }
            "--retention" | "-r" => {
                i += 1;
                config.retention = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.20);
            }
            "--workers" | "-w" => {
                i += 1;
                config.num_workers = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8);
            }
            "--gpu" | "-g" => {
                config.use_gpu = true;
            }
            "--gpu-device" => {
                i += 1;
                config.gpu_device_id = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
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

    if config.model.is_empty() {
        eprintln!("Error: --model is required");
        print_help();
        std::process::exit(1);
    }

    println!("\nTurbo Model Compression Pipeline");
    println!("=================================");
    println!("Model:               {}", config.model);
    println!("Output:              {}", config.output_dir.display());
    println!("Retention:           {:.0}%", config.retention * 100.0);
    println!("Workers:             {}", config.num_workers);
    println!(
        "GPU:                 {}",
        if config.use_gpu {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();

    // Create and run pipeline
    let mut pipeline = TurboPipeline::new(config)?;
    let report = pipeline.run()?;

    // Print report
    println!("\nCompression Complete!");
    println!("=====================");
    println!("Output:              {}", report.output_path.display());
    println!(
        "Tensors:             {}/{} completed, {} failed",
        report.tensors_completed,
        report.tensors_completed + report.tensors_failed + report.tensors_skipped,
        report.tensors_failed
    );
    println!(
        "Input Size:          {:.2} GB",
        report.total_input_bytes as f64 / 1_000_000_000.0
    );
    println!(
        "Output Size:         {:.2} GB",
        report.total_output_bytes as f64 / 1_000_000_000.0
    );
    println!("Compression Ratio:   {:.1}x", report.compression_ratio);
    println!("Time:                {:.1} seconds", report.elapsed_seconds);
    println!("Throughput:          {:.1} MB/s", report.throughput_mbps);
    println!("Workers Used:        {}", report.num_workers);
    println!(
        "GPU Used:            {}",
        if report.gpu_used { "yes" } else { "no" }
    );

    println!("\nDone!");

    Ok(())
}

fn print_help() {
    println!("Turbo Model Compression Pipeline");
    println!();
    println!("USAGE:");
    println!("    compress_turbo [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -m, --model <PATH>       Model path or HuggingFace ID (required)");
    println!("    -o, --output <PATH>      Output directory [default: ./compressed]");
    println!("    -r, --retention <RATIO>  Retention ratio 0.0-1.0 [default: 0.20]");
    println!("    -w, --workers <N>        Number of parallel workers [default: auto]");
    println!("    -g, --gpu                Enable GPU acceleration (requires cuda feature)");
    println!("        --gpu-device <ID>    GPU device ID [default: 0]");
    println!("    -h, --help               Print help");
}

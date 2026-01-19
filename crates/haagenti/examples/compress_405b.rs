//! Production CLI for 405B model compression with checkpointing.
//!
//! This tool compresses large language models using HCT (Holographic Compression Transform)
//! with resumable checkpointing for handling long-running jobs.
//!
//! ## Usage
//!
//! ```bash
//! # Start fresh compression
//! cargo run --release --example compress_405b -- \
//!     --model meta-llama/Llama-3.1-405B \
//!     --output ./llama-405b-compressed \
//!     --retention 0.70
//!
//! # Resume after interruption
//! cargo run --release --example compress_405b -- \
//!     --model meta-llama/Llama-3.1-405B \
//!     --output ./llama-405b-compressed \
//!     --resume
//!
//! # With quality validation
//! cargo run --release --example compress_405b -- \
//!     --model meta-llama/Llama-3.1-405B \
//!     --output ./llama-405b-compressed \
//!     --retention 0.70 \
//!     --quality-sample 0.10
//! ```

use std::path::PathBuf;
use std::time::Instant;

use haagenti::pipeline::{CompressionPipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut model = String::new();
    let mut output_dir = PathBuf::from("./compressed");
    let mut retention = 0.70f32;
    let mut quality_sample = 0.05f32;
    let mut resume = false;
    let mut checkpoint_interval = 10usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model = args[i].clone();
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_dir = PathBuf::from(&args[i]);
                }
            }
            "--retention" | "-r" => {
                i += 1;
                if i < args.len() {
                    retention = args[i].parse().unwrap_or(0.70);
                }
            }
            "--quality-sample" | "-q" => {
                i += 1;
                if i < args.len() {
                    quality_sample = args[i].parse().unwrap_or(0.05);
                }
            }
            "--checkpoint-interval" | "-c" => {
                i += 1;
                if i < args.len() {
                    checkpoint_interval = args[i].parse().unwrap_or(10);
                }
            }
            "--resume" => {
                resume = true;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return Ok(());
            }
        }
        i += 1;
    }

    if model.is_empty() && !resume {
        eprintln!("Error: --model is required unless --resume is specified");
        print_help();
        return Ok(());
    }

    // Check for resume
    let checkpoint_path = output_dir.join("checkpoint.json");
    if resume && checkpoint_path.exists() {
        println!("Resuming from checkpoint: {}", checkpoint_path.display());
        // Load model ID from checkpoint
        let checkpoint = haagenti::pipeline::CompressionCheckpoint::load(&checkpoint_path)?;
        if model.is_empty() {
            model = checkpoint.model_id.clone();
        }
        println!("Model: {}", model);
    }

    if model.is_empty() {
        eprintln!("Error: --model is required");
        return Ok(());
    }

    // Print configuration
    println!("\n405B Model Compression Pipeline");
    println!("================================");
    println!("Model:               {}", model);
    println!("Output:              {}", output_dir.display());
    println!("Retention:           {:.0}%", retention * 100.0);
    println!("Quality Sample Rate: {:.0}%", quality_sample * 100.0);
    println!("Checkpoint Interval: {} tensors", checkpoint_interval);
    println!("Resume:              {}", resume);
    println!();

    // Create pipeline configuration
    let config = PipelineConfig {
        model,
        output_dir,
        retention,
        quality_sample_rate: quality_sample,
        checkpoint_interval,
        ..Default::default()
    };

    // Create or resume pipeline
    let start = Instant::now();
    let mut pipeline = CompressionPipeline::new_or_resume(config)?;

    println!(
        "Pipeline initialized: {} tensors to process",
        pipeline.checkpoint().stats.total_tensors
    );

    let already_done = pipeline.checkpoint().stats.completed
        + pipeline.checkpoint().stats.failed
        + pipeline.checkpoint().stats.skipped;

    if already_done > 0 {
        println!(
            "Resuming from {}/{} tensors ({:.1}% complete)",
            already_done,
            pipeline.checkpoint().stats.total_tensors,
            pipeline.checkpoint().progress() * 100.0
        );
    }

    println!("\nStarting compression...\n");

    // Run the pipeline
    let report = pipeline.run()?;

    let elapsed = start.elapsed();

    // Print report
    println!("\n\nCompression Complete!");
    println!("=====================");
    println!("Output:              {}", report.output_path.display());
    println!(
        "Tensors:             {}/{} completed, {} failed, {} skipped",
        report.tensors_completed,
        report.tensors_processed,
        report.tensors_failed,
        report.tensors_skipped
    );
    println!(
        "Input Size:          {:.2} GB",
        report.total_input_bytes as f64 / 1e9
    );
    println!(
        "Output Size:         {:.2} GB",
        report.total_output_bytes as f64 / 1e9
    );
    println!("Compression Ratio:   {:.1}x", report.compression_ratio);
    println!("Time:                {:.1} seconds", elapsed.as_secs_f64());
    println!(
        "Throughput:          {:.1} MB/s",
        report.total_input_bytes as f64 / elapsed.as_secs_f64() / 1e6
    );

    if report.quality.sample_count > 0 {
        println!("\nQuality Summary ({} samples):", report.quality.sample_count);
        println!(
            "  Avg Cosine:        {:.4}",
            report.quality.avg_cosine_similarity
        );
        println!(
            "  Min Cosine:        {:.4}",
            report.quality.min_cosine_similarity
        );
        println!(
            "  Acceptable:        {:.0}%",
            report.quality.acceptable_fraction * 100.0
        );
        println!("  Overall Grade:     {}", report.quality.grade());
    }

    println!("\nDone!");

    Ok(())
}

fn print_help() {
    println!(
        r#"
405B Model Compression Pipeline

USAGE:
    compress_405b [OPTIONS]

OPTIONS:
    -m, --model <MODEL>              Model path or HuggingFace ID (required)
    -o, --output <DIR>               Output directory [default: ./compressed]
    -r, --retention <RATIO>          Retention ratio 0.0-1.0 [default: 0.70]
    -q, --quality-sample <RATIO>     Quality validation sample rate [default: 0.05]
    -c, --checkpoint-interval <N>    Tensors between checkpoints [default: 10]
        --resume                     Resume from existing checkpoint
    -h, --help                       Print help

EXAMPLES:
    # Compress Llama 405B at 70% retention
    compress_405b --model meta-llama/Llama-3.1-405B --output ./llama-compressed

    # Resume interrupted compression
    compress_405b --output ./llama-compressed --resume

    # High quality validation (10% of tensors)
    compress_405b --model meta-llama/Llama-3.1-405B --quality-sample 0.10
"#
    );
}

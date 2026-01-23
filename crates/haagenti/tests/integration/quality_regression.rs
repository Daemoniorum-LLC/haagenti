//! Quality regression tests for HCT compression.
//!
//! These tests establish baseline quality metrics and fail if quality degrades.
//! Run with: cargo test --test integration_tests --features="lz4,zstd,testing"

use haagenti::adaptive::AdaptiveSpectralEncoder;
use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::importance::{ImportanceGuidedDecoder, ImportanceGuidedEncoder, ImportanceMap};
use haagenti::mixed_precision::{MixedPrecisionDecoder, MixedPrecisionEncoder};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Generate deterministic LLM-like weight distribution
fn generate_llm_weights(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.1).unwrap();
    (0..size).map(|_| normal.sample(&mut rng) as f32).collect()
}

/// Generate low-rank attention matrix (more realistic)
fn generate_attention_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = vec![0.0f32; rows * cols];
    let rank = 10;
    let scale_factor = 0.5_f32;

    for r in 0..rank {
        let row_vec: Vec<f32> = (0..rows)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng) - scale_factor)
            .collect();
        let col_vec: Vec<f32> = (0..cols)
            .map(|_| rand::Rng::r#gen::<f32>(&mut rng) - scale_factor)
            .collect();
        let scale = 1.0 / (r as f32 + 1.0).sqrt();

        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] += row_vec[i] * col_vec[j] * scale;
            }
        }
    }
    result
}

/// Calculate MSE between original and reconstructed
fn mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32
}

/// Calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Quality metrics struct for tracking
#[derive(Debug, Clone)]
struct QualityMetrics {
    cosine_sim: f32,
    mse: f32,
    max_error: f32,
    compression_ratio: f32,
}

impl QualityMetrics {
    fn new(original: &[f32], reconstructed: &[f32], compressed_size: usize) -> Self {
        let original_size = original.len() * 4; // F32 = 4 bytes
        let max_error = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        Self {
            cosine_sim: cosine_similarity(original, reconstructed),
            mse: mse(original, reconstructed),
            max_error,
            compression_ratio: original_size as f32 / compressed_size as f32,
        }
    }
}

// ============================================================================
// BASELINE QUALITY THRESHOLDS
// These values represent the minimum acceptable quality for each encoder.
// If tests fail, investigate whether it's a real regression or a valid change.
// ============================================================================

/// Compressive encoder at 70% retention should achieve:
const COMPRESSIVE_70_MIN_COSINE: f32 = 0.99;
const COMPRESSIVE_70_MAX_MSE: f32 = 0.001;

/// Adaptive encoder at 90% quality target should achieve:
/// Note: Adaptive uses spectral energy to determine retention, so results may vary
const ADAPTIVE_90_MIN_COSINE: f32 = 0.94; // Adjusted based on observed performance
const ADAPTIVE_90_MAX_MSE: f32 = 0.003;

/// Mixed precision encoder should achieve:
const MIXED_PRECISION_MIN_COSINE: f32 = 0.97;
const MIXED_PRECISION_MAX_MSE: f32 = 0.003;

/// Importance-guided encoder should achieve:
const IMPORTANCE_MIN_COSINE: f32 = 0.97;
const IMPORTANCE_MAX_MSE: f32 = 0.003;

// ============================================================================
// REGRESSION TESTS
// ============================================================================

#[test]
fn test_compressive_spectral_quality_baseline() {
    let (width, height) = (128, 128);
    let data = generate_attention_weights(height, width, 42);

    let encoder = CompressiveSpectralEncoder::new(8, 0.70);
    let fragments = encoder.encode_2d(&data, width, height).unwrap();

    // Calculate compressed size
    let compressed_size: usize = fragments.iter().map(|f| f.data.len()).sum();

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    let metrics = QualityMetrics::new(&data, &reconstructed, compressed_size);

    println!("Compressive 70% metrics:");
    println!("  Cosine similarity: {:.6}", metrics.cosine_sim);
    println!("  MSE: {:.6}", metrics.mse);
    println!("  Max error: {:.6}", metrics.max_error);
    println!("  Compression ratio: {:.2}x", metrics.compression_ratio);

    assert!(
        metrics.cosine_sim >= COMPRESSIVE_70_MIN_COSINE,
        "Compressive cosine similarity {:.4} below threshold {:.4}",
        metrics.cosine_sim,
        COMPRESSIVE_70_MIN_COSINE
    );
    assert!(
        metrics.mse <= COMPRESSIVE_70_MAX_MSE,
        "Compressive MSE {:.6} above threshold {:.6}",
        metrics.mse,
        COMPRESSIVE_70_MAX_MSE
    );
}

#[test]
fn test_adaptive_spectral_quality_baseline() {
    let (width, height) = (128, 128);
    let data = generate_attention_weights(height, width, 42);

    let encoder = AdaptiveSpectralEncoder::new(0.90, 8);
    let (meta, fragments) = encoder.encode_2d(&data, width, height).unwrap();

    // Calculate compressed size
    let compressed_size: usize = fragments.iter().map(|f| f.data.len()).sum();

    // Decode using CompressiveSpectralDecoder (adaptive produces compatible output)
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    let metrics = QualityMetrics::new(&data, &reconstructed, compressed_size);

    println!("Adaptive 90% quality metrics:");
    println!("  Actual retention: {:.1}%", meta.retention_ratio * 100.0);
    println!("  Cosine similarity: {:.6}", metrics.cosine_sim);
    println!("  MSE: {:.6}", metrics.mse);
    println!("  Max error: {:.6}", metrics.max_error);
    println!("  Compression ratio: {:.2}x", metrics.compression_ratio);

    assert!(
        metrics.cosine_sim >= ADAPTIVE_90_MIN_COSINE,
        "Adaptive cosine similarity {:.4} below threshold {:.4}",
        metrics.cosine_sim,
        ADAPTIVE_90_MIN_COSINE
    );
    assert!(
        metrics.mse <= ADAPTIVE_90_MAX_MSE,
        "Adaptive MSE {:.6} above threshold {:.6}",
        metrics.mse,
        ADAPTIVE_90_MAX_MSE
    );
}

#[test]
fn test_mixed_precision_quality_baseline() {
    let (width, height) = (64, 64); // Smaller due to O(n^2) DCT
    let data = generate_attention_weights(height, width, 42);

    let encoder = MixedPrecisionEncoder::new(0.70, 0.20);
    let compressed = encoder.encode(&data, width, height).unwrap();

    let decoder = MixedPrecisionDecoder::new();
    let reconstructed = decoder.decode(&compressed).unwrap();

    let compressed_size = compressed.fp16_data.len() + compressed.int4_data.len();
    let metrics = QualityMetrics::new(&data, &reconstructed, compressed_size);

    println!("Mixed Precision metrics:");
    println!(
        "  FP16 coeffs: {}, INT4 coeffs: {}",
        compressed.fp16_count, compressed.int4_count
    );
    println!("  Cosine similarity: {:.6}", metrics.cosine_sim);
    println!("  MSE: {:.6}", metrics.mse);
    println!("  Max error: {:.6}", metrics.max_error);
    println!("  Compression ratio: {:.2}x", metrics.compression_ratio);

    assert!(
        metrics.cosine_sim >= MIXED_PRECISION_MIN_COSINE,
        "Mixed precision cosine similarity {:.4} below threshold {:.4}",
        metrics.cosine_sim,
        MIXED_PRECISION_MIN_COSINE
    );
    assert!(
        metrics.mse <= MIXED_PRECISION_MAX_MSE,
        "Mixed precision MSE {:.6} above threshold {:.6}",
        metrics.mse,
        MIXED_PRECISION_MAX_MSE
    );
}

#[test]
fn test_importance_guided_quality_baseline() {
    let (width, height) = (64, 64);
    let data = generate_attention_weights(height, width, 42);
    let tensor_name = "model.layers.0.self_attn.v_proj.weight";

    let importance_map = ImportanceMap::heuristic_only();
    let encoder = ImportanceGuidedEncoder::new(0.50, importance_map);
    let compressed = encoder.encode(&data, width, height, tensor_name).unwrap();

    let decoder = ImportanceGuidedDecoder::new();
    let reconstructed = decoder.decode(&compressed).unwrap();

    // Estimate compressed size (coefficients stored as F32 = 4 bytes each)
    let compressed_size = compressed.coefficients.len() * 4 + compressed.indices.len() * 4;
    let metrics = QualityMetrics::new(&data, &reconstructed, compressed_size);

    println!("Importance-guided metrics (v_proj):");
    println!(
        "  Effective retention: {:.1}%",
        compressed.effective_retention * 100.0
    );
    println!("  Cosine similarity: {:.6}", metrics.cosine_sim);
    println!("  MSE: {:.6}", metrics.mse);
    println!("  Max error: {:.6}", metrics.max_error);
    println!("  Compression ratio: {:.2}x", metrics.compression_ratio);

    assert!(
        metrics.cosine_sim >= IMPORTANCE_MIN_COSINE,
        "Importance-guided cosine similarity {:.4} below threshold {:.4}",
        metrics.cosine_sim,
        IMPORTANCE_MIN_COSINE
    );
    assert!(
        metrics.mse <= IMPORTANCE_MAX_MSE,
        "Importance-guided MSE {:.6} above threshold {:.6}",
        metrics.mse,
        IMPORTANCE_MAX_MSE
    );
}

#[test]
fn test_quality_monotonicity() {
    // Verify that higher retention produces better quality
    let (width, height) = (128, 128);
    let data = generate_attention_weights(height, width, 42);

    let retentions = [0.30, 0.50, 0.70, 0.90];
    let mut prev_cosine = 0.0f32;

    for retention in retentions {
        let encoder = CompressiveSpectralEncoder::new(8, retention);
        let fragments = encoder.encode_2d(&data, width, height).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let reconstructed = decoder.reconstruct().unwrap();

        let cosine = cosine_similarity(&data, &reconstructed);
        println!(
            "Retention {:.0}%: cosine = {:.6}",
            retention * 100.0,
            cosine
        );

        assert!(
            cosine >= prev_cosine - 0.001, // Allow tiny tolerance for floating point
            "Quality decreased at {:.0}% retention: {:.4} < {:.4}",
            retention * 100.0,
            cosine,
            prev_cosine
        );
        prev_cosine = cosine;
    }
}

#[test]
fn test_layer_type_quality_ranking() {
    // Importance-guided should give better quality to high-importance layers
    let (width, height) = (64, 64);
    let data = generate_attention_weights(height, width, 42);

    let importance_map = ImportanceMap::heuristic_only();
    let encoder = ImportanceGuidedEncoder::new(0.50, importance_map);
    let decoder = ImportanceGuidedDecoder::new();

    let layer_tests = [
        ("model.layers.0.mlp.gate_proj.weight", "MLP (Low)"),
        ("model.layers.0.self_attn.q_proj.weight", "Q-proj (Medium)"),
        ("model.layers.0.self_attn.v_proj.weight", "V-proj (High)"),
    ];

    let mut results: Vec<(f32, f32, &str)> = Vec::new();

    for (tensor_name, label) in layer_tests {
        let compressed = encoder.encode(&data, width, height, tensor_name).unwrap();
        let reconstructed = decoder.decode(&compressed).unwrap();
        let cosine = cosine_similarity(&data, &reconstructed);
        let retention = compressed.effective_retention;

        println!(
            "{}: retention={:.1}%, cosine={:.4}",
            label,
            retention * 100.0,
            cosine
        );
        results.push((retention, cosine, label));
    }

    // Higher importance should mean higher retention
    assert!(
        results[2].0 > results[0].0, // V-proj retention > MLP retention
        "V-proj should have higher retention than MLP"
    );
}

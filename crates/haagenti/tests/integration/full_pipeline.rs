//! Full pipeline integration tests.
//!
//! Tests the complete compression/decompression roundtrip:
//! 1. Generate synthetic tensor data
//! 2. Encode with CompressiveSpectralEncoder
//! 3. Optionally apply INT4 quantization
//! 4. Decode back to original format
//! 5. Verify quality metrics

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::testing::{compute_quality, dequantize_int4, quantize_int4, QualityReport};

/// Default number of fragments for testing.
const DEFAULT_FRAGMENTS: u16 = 4;

/// Generate synthetic neural network weight data.
///
/// Uses a combination of sine waves and noise to simulate
/// the frequency distribution of real model weights.
fn generate_synthetic_weights(width: usize, height: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut weights = Vec::with_capacity(width * height);
    let mut hasher = DefaultHasher::new();

    for i in 0..(width * height) {
        // Create pseudo-random but deterministic values
        (seed, i as u64).hash(&mut hasher);
        let hash = hasher.finish();
        hasher = DefaultHasher::new();

        // Mix of structured patterns and noise
        let x = (i % width) as f32 / width as f32;
        let y = (i / width) as f32 / height as f32;

        let structured = (x * std::f32::consts::TAU).sin() * (y * std::f32::consts::TAU).cos() * 0.3;
        let noise = ((hash as f32 / u64::MAX as f32) - 0.5) * 0.4;

        weights.push(structured + noise);
    }

    weights
}

#[test]
fn test_spectral_roundtrip_64x64() {
    let width = 64;
    let height = 64;
    let retention = 0.70;

    let original = generate_synthetic_weights(width, height, 12345);

    // Encode
    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&original, width, height).unwrap();

    assert!(
        !fragments.is_empty(),
        "Should produce at least one fragment"
    );

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    // Verify quality
    let report = compute_quality(&original, &reconstructed);

    assert!(
        report.cosine_similarity > 0.99,
        "Cosine similarity too low: {}",
        report.cosine_similarity
    );
    assert!(report.psnr > 25.0, "PSNR too low: {} dB", report.psnr);

    println!("64x64 spectral roundtrip: {}", report);
}

#[test]
fn test_spectral_roundtrip_128x256() {
    let width = 128;
    let height = 256;
    let retention = 0.70;

    let original = generate_synthetic_weights(width, height, 67890);

    // Encode
    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&original, width, height).unwrap();

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    // Verify quality
    let report = compute_quality(&original, &reconstructed);

    assert!(
        report.cosine_similarity > 0.99,
        "Cosine similarity too low: {}",
        report.cosine_similarity
    );

    println!("128x256 spectral roundtrip: {}", report);
}

#[test]
fn test_spectral_plus_int4_roundtrip() {
    let width = 64;
    let height = 64;
    let retention = 0.70;

    let original = generate_synthetic_weights(width, height, 11111);

    // Stage 1: Spectral encode
    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&original, width, height).unwrap();

    // Stage 2: INT4 quantize the coefficient data
    // Note: In production, we'd quantize just the coefficients, not the headers
    // For this test, we demonstrate the INT4 roundtrip separately

    // First, decode spectral without INT4
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let spectral_reconstructed = decoder.reconstruct().unwrap();

    // Now apply INT4 quantization to the reconstructed values
    let int4_data = quantize_int4(&spectral_reconstructed);
    let int4_reconstructed = dequantize_int4(&int4_data, spectral_reconstructed.len());

    // Verify combined quality (spectral + INT4)
    let combined_report = compute_quality(&original, &int4_reconstructed);

    // Combined pipeline has more loss, but should still be usable
    assert!(
        combined_report.cosine_similarity > 0.95,
        "Combined cosine similarity too low: {}",
        combined_report.cosine_similarity
    );

    println!("Spectral + INT4 combined: {}", combined_report);

    // Also verify INT4 alone
    let int4_only = quantize_int4(&original);
    let int4_only_reconstructed = dequantize_int4(&int4_only, original.len());
    let int4_only_report = compute_quality(&original, &int4_only_reconstructed);

    println!("INT4 only: {}", int4_only_report);
}

#[test]
fn test_retention_levels() {
    let width = 64;
    let height = 64;
    let original = generate_synthetic_weights(width, height, 22222);

    let retention_levels = [0.10, 0.30, 0.50, 0.70, 0.90];
    let mut reports: Vec<(f32, QualityReport)> = Vec::new();

    for &retention in &retention_levels {
        let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
        let fragments = encoder.encode_2d(&original, width, height).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let reconstructed = decoder.reconstruct().unwrap();

        let report = compute_quality(&original, &reconstructed);
        reports.push((retention, report));
    }

    // Verify monotonicity: higher retention should give better quality
    for i in 1..reports.len() {
        assert!(
            reports[i].1.cosine_similarity >= reports[i - 1].1.cosine_similarity * 0.95,
            "Quality should generally improve with retention: {}% gave {} vs {}% gave {}",
            reports[i - 1].0 * 100.0,
            reports[i - 1].1.cosine_similarity,
            reports[i].0 * 100.0,
            reports[i].1.cosine_similarity
        );
    }

    println!("\nRetention level quality comparison:");
    for (retention, report) in &reports {
        println!("  {:>3}% retention: {}", (retention * 100.0) as u32, report);
    }
}

#[test]
fn test_progressive_quality() {
    let width = 64;
    let height = 64;
    let retention = 0.70;

    let original = generate_synthetic_weights(width, height, 33333);

    // Encode
    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&original, width, height).unwrap();

    // Decode progressively, measuring quality at each step
    let mut decoder = CompressiveSpectralDecoder::new();
    let mut quality_curve: Vec<(usize, f32)> = Vec::new();

    decoder.add_essentials(&fragments[0]).unwrap();
    let essential_reconstructed = decoder.reconstruct().unwrap();
    let essential_report = compute_quality(&original, &essential_reconstructed);
    quality_curve.push((0, essential_report.cosine_similarity));

    for (i, frag) in fragments[1..].iter().enumerate() {
        decoder.add_detail(frag).unwrap();
        let reconstructed = decoder.reconstruct().unwrap();
        let report = compute_quality(&original, &reconstructed);
        quality_curve.push((i + 1, report.cosine_similarity));
    }

    // Quality should improve (or stay same) with each fragment
    for i in 1..quality_curve.len() {
        assert!(
            quality_curve[i].1 >= quality_curve[i - 1].1 - 0.001,
            "Quality decreased from fragment {} to {}: {} -> {}",
            quality_curve[i - 1].0,
            quality_curve[i].0,
            quality_curve[i - 1].1,
            quality_curve[i].1
        );
    }

    println!("\nProgressive quality curve:");
    for (frag_idx, cos_sim) in &quality_curve {
        println!(
            "  Fragment {}: cosine similarity = {:.6}",
            frag_idx, cos_sim
        );
    }
}

#[test]
fn test_non_square_tensors() {
    // Test various non-square dimensions
    let test_cases = [
        (32, 128), // Tall
        (128, 32), // Wide
        (64, 100), // Arbitrary
        (100, 64), // Arbitrary reversed
        (256, 64), // Very wide
    ];

    for (width, height) in test_cases {
        let original = generate_synthetic_weights(width, height, (width * height) as u64);

        let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, 0.70);
        let fragments = encoder.encode_2d(&original, width, height).unwrap();

        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let reconstructed = decoder.reconstruct().unwrap();

        let report = compute_quality(&original, &reconstructed);

        assert_eq!(
            reconstructed.len(),
            original.len(),
            "Output size mismatch for {}x{}",
            width,
            height
        );
        assert!(
            report.cosine_similarity > 0.98,
            "Poor quality for {}x{}: {}",
            width,
            height,
            report.cosine_similarity
        );

        println!("{}x{}: {}", width, height, report);
    }
}

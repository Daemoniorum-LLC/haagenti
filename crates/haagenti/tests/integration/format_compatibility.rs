//! HCT format compatibility tests.
//!
//! Tests that validate the compression pipeline produces valid, usable output:
//! - Fragment structure is valid
//! - Encode/decode roundtrip works
//! - Compression ratios are reasonable

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::testing::compute_quality;

/// Default number of fragments for testing.
const DEFAULT_FRAGMENTS: u16 = 4;

/// Validate that fragments are produced and have valid structure.
#[test]
fn test_fragment_structure() {
    let width = 64;
    let height = 64;
    let retention = 0.70;

    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&data, width, height).unwrap();

    // Should produce at least one fragment (essentials)
    assert!(!fragments.is_empty(), "Should produce at least one fragment");

    // First fragment should be index 0
    assert_eq!(fragments[0].index, 0, "First fragment should have index 0");

    // All fragments should have non-empty data
    for (i, frag) in fragments.iter().enumerate() {
        assert!(
            !frag.data.is_empty(),
            "Fragment {} should have non-empty data",
            i
        );
    }

    // Fragment indices should be sequential
    for (i, frag) in fragments.iter().enumerate() {
        assert_eq!(
            frag.index as usize, i,
            "Fragment {} has wrong index {}",
            i, frag.index
        );
    }

    println!(
        "Produced {} fragments with total {} bytes",
        fragments.len(),
        fragments.iter().map(|f| f.data.len()).sum::<usize>()
    );
}

#[test]
fn test_fragment_count_matches_retention() {
    let width = 64;
    let height = 64;
    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let low_retention = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, 0.30);
    let high_retention = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, 0.90);

    let low_fragments = low_retention.encode_2d(&data, width, height).unwrap();
    let high_fragments = high_retention.encode_2d(&data, width, height).unwrap();

    // Fragment data sizes should differ based on retention
    let low_total_size: usize = low_fragments.iter().map(|f| f.data.len()).sum();
    let high_total_size: usize = high_fragments.iter().map(|f| f.data.len()).sum();

    assert!(
        high_total_size >= low_total_size,
        "Higher retention should produce more data: {} vs {}",
        high_total_size,
        low_total_size
    );

    println!("30% retention: {} bytes in {} fragments", low_total_size, low_fragments.len());
    println!("90% retention: {} bytes in {} fragments", high_total_size, high_fragments.len());
}

#[test]
fn test_compression_ratio() {
    let width = 64;
    let height = 64;
    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let retention = 0.10; // 10% retention should give ~5x compression
    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, retention);
    let fragments = encoder.encode_2d(&data, width, height).unwrap();

    let input_bytes = data.len() * 4;
    let output_bytes: usize = fragments.iter().map(|f| f.data.len()).sum();
    let ratio = input_bytes as f32 / output_bytes as f32;

    assert!(
        ratio > 1.5,
        "Expected compression ratio > 1.5, got {}",
        ratio
    );

    println!(
        "10% retention: {} -> {} bytes ({:.2}x compression)",
        input_bytes, output_bytes, ratio
    );
}

#[test]
fn test_encode_decode_consistency() {
    let width = 64;
    let height = 64;
    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, 0.70);
    let fragments = encoder.encode_2d(&data, width, height).unwrap();

    // Decode
    let mut decoder = CompressiveSpectralDecoder::new();
    decoder.add_essentials(&fragments[0]).unwrap();
    for frag in &fragments[1..] {
        decoder.add_detail(frag).unwrap();
    }
    let reconstructed = decoder.reconstruct().unwrap();

    // Should have same number of elements
    assert_eq!(
        reconstructed.len(),
        data.len(),
        "Reconstructed size mismatch"
    );

    // Quality should be reasonable
    let report = compute_quality(&data, &reconstructed);
    assert!(
        report.cosine_similarity > 0.98,
        "Quality too low: {}",
        report.cosine_similarity
    );

    println!("Encode-decode roundtrip: {}", report);
}

#[test]
fn test_multiple_fragment_counts() {
    let width = 64;
    let height = 64;
    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    for num_frags in [1u16, 2, 4, 8, 16] {
        let encoder = CompressiveSpectralEncoder::new(num_frags, 0.50);
        let fragments = encoder.encode_2d(&data, width, height).unwrap();

        // Decode and verify
        let mut decoder = CompressiveSpectralDecoder::new();
        decoder.add_essentials(&fragments[0]).unwrap();
        for frag in &fragments[1..] {
            decoder.add_detail(frag).unwrap();
        }
        let reconstructed = decoder.reconstruct().unwrap();

        let report = compute_quality(&data, &reconstructed);
        assert!(
            report.cosine_similarity > 0.95,
            "Quality too low for {} fragments: {}",
            num_frags,
            report.cosine_similarity
        );

        println!(
            "{} fragments: {} bytes, cosine={:.6}",
            fragments.len(),
            fragments.iter().map(|f| f.data.len()).sum::<usize>(),
            report.cosine_similarity
        );
    }
}

#[test]
fn test_deterministic_encoding() {
    let width = 64;
    let height = 64;
    let data: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let encoder = CompressiveSpectralEncoder::new(DEFAULT_FRAGMENTS, 0.70);

    // Encode twice
    let fragments1 = encoder.encode_2d(&data, width, height).unwrap();
    let fragments2 = encoder.encode_2d(&data, width, height).unwrap();

    // Should produce identical output
    assert_eq!(
        fragments1.len(),
        fragments2.len(),
        "Fragment count mismatch"
    );

    for (i, (f1, f2)) in fragments1.iter().zip(fragments2.iter()).enumerate() {
        assert_eq!(
            f1.data, f2.data,
            "Fragment {} data mismatch",
            i
        );
    }

    println!("Encoding is deterministic: {} identical fragments", fragments1.len());
}

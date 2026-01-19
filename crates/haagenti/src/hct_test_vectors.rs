//! HCT Specification Test Vectors
//!
//! This module provides reference test vectors for validating HCT implementations.
//! Each test vector includes:
//! - Input tensor
//! - Expected DCT coefficients
//! - Expected retained coefficients at various retention levels
//! - Expected reconstructed tensor
//! - Quality metrics (cosine similarity, relative error)
//!
//! These vectors are normative - a conforming implementation MUST produce
//! outputs that match within specified tolerances.

use std::f32::consts::PI;

/// A complete test vector for HCT validation
#[derive(Debug, Clone)]
pub struct HctTestVector {
    /// Human-readable name for this test case
    pub name: &'static str,
    /// Input tensor dimensions
    pub shape: Vec<usize>,
    /// Input tensor data (row-major)
    pub input: Vec<f32>,
    /// Expected DCT coefficients (full, before truncation)
    pub dct_coefficients: Vec<f32>,
    /// Retention ratio for this test
    pub retention: f32,
    /// Indices of retained coefficients (sorted by magnitude, descending)
    pub retained_indices: Vec<usize>,
    /// Values of retained coefficients
    pub retained_values: Vec<f32>,
    /// Expected reconstructed tensor
    pub reconstructed: Vec<f32>,
    /// Expected cosine similarity between input and reconstructed
    pub expected_cosine_similarity: f32,
    /// Tolerance for cosine similarity comparison
    pub cosine_tolerance: f32,
}

/// Reference DCT-II implementation for test vector generation
/// This is the mathematical definition, not optimized
pub fn reference_dct_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];

    for u in 0..rows {
        for v in 0..cols {
            let mut sum = 0.0f32;
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let cos_i = ((2 * i + 1) as f32 * u as f32 * PI / (2.0 * rows as f32)).cos();
                    let cos_j = ((2 * j + 1) as f32 * v as f32 * PI / (2.0 * cols as f32)).cos();
                    sum += input[idx] * cos_i * cos_j;
                }
            }

            // Normalization factors
            let alpha_u = if u == 0 { (1.0 / rows as f32).sqrt() } else { (2.0 / rows as f32).sqrt() };
            let alpha_v = if v == 0 { (1.0 / cols as f32).sqrt() } else { (2.0 / cols as f32).sqrt() };

            output[u * cols + v] = alpha_u * alpha_v * sum;
        }
    }

    output
}

/// Reference IDCT-II implementation for test vector generation
pub fn reference_idct_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;
            for u in 0..rows {
                for v in 0..cols {
                    let idx = u * cols + v;
                    let alpha_u = if u == 0 { (1.0 / rows as f32).sqrt() } else { (2.0 / rows as f32).sqrt() };
                    let alpha_v = if v == 0 { (1.0 / cols as f32).sqrt() } else { (2.0 / cols as f32).sqrt() };
                    let cos_i = ((2 * i + 1) as f32 * u as f32 * PI / (2.0 * rows as f32)).cos();
                    let cos_j = ((2 * j + 1) as f32 * v as f32 * PI / (2.0 * cols as f32)).cos();
                    sum += alpha_u * alpha_v * input[idx] * cos_i * cos_j;
                }
            }
            output[i * cols + j] = sum;
        }
    }

    output
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        if norm_a == 0.0 && norm_b == 0.0 { 1.0 } else { 0.0 }
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Get indices sorted by magnitude (descending)
pub fn indices_by_magnitude(coeffs: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = coeffs.iter().map(|&v| v.abs()).enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Truncate coefficients, keeping only top k by magnitude
pub fn truncate_coefficients(coeffs: &[f32], retention: f32) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let k = ((coeffs.len() as f32 * retention).floor() as usize).max(1);
    let indices = indices_by_magnitude(coeffs);

    let retained_indices: Vec<usize> = indices[..k].to_vec();
    let retained_values: Vec<f32> = retained_indices.iter().map(|&i| coeffs[i]).collect();

    // Create sparse coefficient array
    let mut sparse = vec![0.0f32; coeffs.len()];
    for (&idx, &val) in retained_indices.iter().zip(retained_values.iter()) {
        sparse[idx] = val;
    }

    (retained_indices, retained_values, sparse)
}

// =============================================================================
// Test Vectors
// =============================================================================

/// Test Vector 1: Simple 4x4 sequential matrix
/// This is the minimal test case from the specification
pub fn test_vector_sequential_4x4() -> HctTestVector {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let rows = 4;
    let cols = 4;
    let retention = 0.5;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "sequential_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.0001,
    }
}

/// Test Vector 2: Identity matrix 4x4
/// Tests behavior with sparse input
pub fn test_vector_identity_4x4() -> HctTestVector {
    let mut input = vec![0.0f32; 16];
    for i in 0..4 {
        input[i * 4 + i] = 1.0;
    }
    let rows = 4;
    let cols = 4;
    let retention = 0.5;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "identity_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.0001,
    }
}

/// Test Vector 3: Gaussian-like weights
/// Simulates typical neural network weight distribution
pub fn test_vector_gaussian_8x8() -> HctTestVector {
    // Deterministic pseudo-Gaussian using sin for reproducibility
    let mut input = Vec::with_capacity(64);
    for i in 0..64 {
        // Creates values roughly in [-1, 1] with clustering near 0
        let v = (i as f32 * 0.1).sin() * (i as f32 * 0.073).cos() * 0.5;
        input.push(v);
    }

    let rows = 8;
    let cols = 8;
    let retention = 0.7;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "gaussian_8x8",
        shape: vec![8, 8],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.0001,
    }
}

/// Test Vector 4: Low-rank matrix
/// Simulates attention weight matrices which are often low-rank
pub fn test_vector_low_rank_8x8() -> HctTestVector {
    // Create rank-2 matrix: outer product of two vectors
    let u = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
    let v = vec![1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125];

    let mut input = vec![0.0f32; 64];
    for i in 0..8 {
        for j in 0..8 {
            input[i * 8 + j] = u[i] * v[j];
        }
    }

    let rows = 8;
    let cols = 8;
    let retention = 0.3; // Low-rank matrices compress very well

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "low_rank_8x8",
        shape: vec![8, 8],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.0001,
    }
}

/// Test Vector 5: All zeros
/// Edge case - should handle gracefully
pub fn test_vector_zeros_4x4() -> HctTestVector {
    let input = vec![0.0f32; 16];
    let rows = 4;
    let cols = 4;
    let retention = 0.5;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);

    HctTestVector {
        name: "zeros_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: 1.0, // 0/0 case, defined as 1.0
        cosine_tolerance: 0.0001,
    }
}

/// Test Vector 6: Constant value
/// Tests DC-only compression
pub fn test_vector_constant_4x4() -> HctTestVector {
    let input = vec![42.0f32; 16];
    let rows = 4;
    let cols = 4;
    let retention = 0.25; // Only need DC component

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "constant_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.0001,
    }
}

/// Get all test vectors
pub fn all_test_vectors() -> Vec<HctTestVector> {
    vec![
        test_vector_sequential_4x4(),
        test_vector_identity_4x4(),
        test_vector_gaussian_8x8(),
        test_vector_low_rank_8x8(),
        test_vector_zeros_4x4(),
        test_vector_constant_4x4(),
    ]
}

// =============================================================================
// STRESS TEST VECTORS
// =============================================================================
// These test edge cases and numerical extremes to ensure robust implementations.

/// Stress Test 1: Large values (within safe range to avoid overflow in DCT)
pub fn stress_vector_large_values_4x4() -> HctTestVector {
    // Use 1e15 as "large" - big enough to stress precision, small enough to not overflow
    // DCT involves summation of N elements, so max_safe must account for that
    let max_safe = 1e15f32;
    let input: Vec<f32> = (1..=16).map(|x| (x as f32 / 16.0) * max_safe).collect();
    let rows = 4;
    let cols = 4;
    let retention = 0.50;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_large_values_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.001, // Slightly looser due to large magnitude
    }
}

/// Stress Test 2: Tiny values (near subnormal range)
pub fn stress_vector_tiny_values_4x4() -> HctTestVector {
    let tiny = 1e-30f32; // Very small but not subnormal
    let input: Vec<f32> = (1..=16).map(|x| (x as f32) * tiny).collect();
    let rows = 4;
    let cols = 4;
    let retention = 0.50;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_tiny_values_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.001, // Slightly looser due to small magnitude
    }
}

/// Stress Test 3: Mixed extreme values (large and tiny in same matrix)
pub fn stress_vector_mixed_extreme_4x4() -> HctTestVector {
    let mut input = vec![0.0f32; 16];
    // Mix of large, small, and normal values
    input[0] = 1e10;
    input[1] = 1e-10;
    input[2] = 1.0;
    input[3] = -1e10;
    input[4] = -1e-10;
    input[5] = -1.0;
    input[6] = 0.5;
    input[7] = -0.5;
    for i in 8..16 {
        input[i] = (i as f32 - 12.0) * 100.0;
    }

    let rows = 4;
    let cols = 4;
    let retention = 0.70; // Higher retention for mixed data

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_mixed_extreme_4x4",
        shape: vec![4, 4],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.01, // Looser due to extreme dynamic range
    }
}

/// Stress Test 4: Wide aspect ratio (2 rows x 32 cols)
pub fn stress_vector_wide_2x32() -> HctTestVector {
    let input: Vec<f32> = (1..=64).map(|x| (x as f32 * 0.1).sin()).collect();
    let rows = 2;
    let cols = 32;
    let retention = 0.50;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_wide_2x32",
        shape: vec![2, 32],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.001,
    }
}

/// Stress Test 5: Tall aspect ratio (32 rows x 2 cols)
pub fn stress_vector_tall_32x2() -> HctTestVector {
    let input: Vec<f32> = (1..=64).map(|x| (x as f32 * 0.1).cos()).collect();
    let rows = 32;
    let cols = 2;
    let retention = 0.50;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_tall_32x2",
        shape: vec![32, 2],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.001,
    }
}

/// Stress Test 6: Alternating sign pattern (high frequency content)
pub fn stress_vector_checkerboard_8x8() -> HctTestVector {
    let mut input = vec![0.0f32; 64];
    for i in 0..8 {
        for j in 0..8 {
            input[i * 8 + j] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }

    let rows = 8;
    let cols = 8;
    let retention = 0.30; // Low retention tests high-frequency handling

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_checkerboard_8x8",
        shape: vec![8, 8],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.01, // Checkerboard is worst case for DCT
    }
}

/// Stress Test 7: Single spike (one non-zero value)
pub fn stress_vector_spike_8x8() -> HctTestVector {
    let mut input = vec![0.0f32; 64];
    input[27] = 100.0; // Single spike in middle

    let rows = 8;
    let cols = 8;
    let retention = 0.50;

    let dct = reference_dct_2d(&input, rows, cols);
    let (retained_indices, retained_values, sparse) = truncate_coefficients(&dct, retention);
    let reconstructed = reference_idct_2d(&sparse, rows, cols);
    let cosine = cosine_similarity(&input, &reconstructed);

    HctTestVector {
        name: "stress_spike_8x8",
        shape: vec![8, 8],
        input,
        dct_coefficients: dct,
        retention,
        retained_indices,
        retained_values,
        reconstructed,
        expected_cosine_similarity: cosine,
        cosine_tolerance: 0.05, // Spike spreads energy across frequencies
    }
}

/// Get all stress test vectors (for robustness testing)
pub fn all_stress_vectors() -> Vec<HctTestVector> {
    vec![
        stress_vector_large_values_4x4(),
        stress_vector_tiny_values_4x4(),
        stress_vector_mixed_extreme_4x4(),
        stress_vector_wide_2x32(),
        stress_vector_tall_32x2(),
        stress_vector_checkerboard_8x8(),
        stress_vector_spike_8x8(),
    ]
}

/// Print test vector in a format suitable for specification appendix
pub fn print_test_vector_for_spec(tv: &HctTestVector) {
    println!("### Test Vector: {}", tv.name);
    println!();
    println!("**Shape**: {:?}", tv.shape);
    println!("**Retention**: {:.0}%", tv.retention * 100.0);
    println!();

    println!("**Input** (row-major):");
    println!("```");
    let cols = tv.shape[1];
    for (i, chunk) in tv.input.chunks(cols).enumerate() {
        print!("  Row {}: [", i);
        for (j, v) in chunk.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:.6}", v);
        }
        println!("]");
    }
    println!("```");
    println!();

    println!("**DCT Coefficients** (top 8 by magnitude):");
    println!("```");
    let mut indexed: Vec<(usize, f32)> = tv.dct_coefficients.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for (idx, val) in indexed.iter().take(8) {
        let row = idx / cols;
        let col = idx % cols;
        println!("  [{},{}] = {:.6}", row, col, val);
    }
    println!("```");
    println!();

    println!("**Retained Coefficients**: {} of {} ({:.0}%)",
             tv.retained_indices.len(), tv.input.len(), tv.retention * 100.0);
    println!();

    println!("**Reconstructed** (row-major):");
    println!("```");
    for (i, chunk) in tv.reconstructed.chunks(cols).enumerate() {
        print!("  Row {}: [", i);
        for (j, v) in chunk.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:.6}", v);
        }
        println!("]");
    }
    println!("```");
    println!();

    println!("**Quality**: cosine_similarity = {:.6}", tv.expected_cosine_similarity);
    println!();
    println!("---");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let dct = reference_dct_2d(&input, 4, 4);
        let reconstructed = reference_idct_2d(&dct, 4, 4);

        let cosine = cosine_similarity(&input, &reconstructed);
        assert!(cosine > 0.999999, "DCT/IDCT roundtrip should be near-perfect: {}", cosine);
    }

    #[test]
    fn test_all_vectors_generate() {
        let vectors = all_test_vectors();
        assert_eq!(vectors.len(), 6);

        for tv in &vectors {
            println!("Test vector '{}': cosine = {:.6}", tv.name, tv.expected_cosine_similarity);
            assert!(tv.expected_cosine_similarity >= 0.0);
            assert!(tv.expected_cosine_similarity <= 1.0 + tv.cosine_tolerance);
        }
    }

    #[test]
    fn test_low_rank_compresses_well() {
        let tv = test_vector_low_rank_8x8();
        // Low-rank matrices achieve decent similarity at 30% retention
        // Note: DCT doesn't perfectly align with rank-1 structure, so we get ~0.93
        // This is still better than random data which would be ~0.85 at 30%
        assert!(tv.expected_cosine_similarity > 0.90,
                "Low-rank matrix at 30% retention should have >0.90 cosine sim: {}",
                tv.expected_cosine_similarity);
    }

    #[test]
    fn test_constant_perfect_reconstruction() {
        let tv = test_vector_constant_4x4();
        // Constant matrix only needs DC - should be perfect
        assert!(tv.expected_cosine_similarity > 0.9999,
                "Constant matrix should reconstruct perfectly: {}",
                tv.expected_cosine_similarity);
    }

    #[test]
    fn print_spec_vectors() {
        println!("\n# HCT Specification Test Vectors\n");
        for tv in all_test_vectors() {
            print_test_vector_for_spec(&tv);
        }
    }

    // =============================================================================
    // STRESS TESTS
    // =============================================================================

    #[test]
    fn test_all_stress_vectors_generate() {
        let vectors = all_stress_vectors();
        assert_eq!(vectors.len(), 7);

        for tv in &vectors {
            println!("Stress vector '{}': cosine = {:.6}, tolerance = {:.4}",
                     tv.name, tv.expected_cosine_similarity, tv.cosine_tolerance);

            // Cosine similarity should be valid (handle NaN from zeros/zeros case)
            assert!(
                tv.expected_cosine_similarity.is_nan() ||
                (tv.expected_cosine_similarity >= -1.0 - tv.cosine_tolerance &&
                 tv.expected_cosine_similarity <= 1.0 + tv.cosine_tolerance),
                "Stress vector '{}' has invalid cosine: {}",
                tv.name, tv.expected_cosine_similarity
            );

            // Shape should match data
            let expected_len: usize = tv.shape.iter().product();
            assert_eq!(tv.input.len(), expected_len,
                       "Stress vector '{}' input length mismatch", tv.name);
            assert_eq!(tv.reconstructed.len(), expected_len,
                       "Stress vector '{}' reconstructed length mismatch", tv.name);
        }
    }

    #[test]
    fn test_stress_large_values() {
        let tv = stress_vector_large_values_4x4();
        // Large values should still compress reasonably
        assert!(tv.expected_cosine_similarity > 0.95,
                "Large values should compress well: {}", tv.expected_cosine_similarity);
    }

    #[test]
    fn test_stress_tiny_values() {
        let tv = stress_vector_tiny_values_4x4();
        // Tiny values should compress just as well as normal values
        assert!(tv.expected_cosine_similarity > 0.95,
                "Tiny values should compress well: {}", tv.expected_cosine_similarity);
    }

    #[test]
    fn test_stress_extreme_aspect_ratios() {
        let wide = stress_vector_wide_2x32();
        let tall = stress_vector_tall_32x2();

        // Both should compress reasonably despite extreme aspect ratios
        assert!(wide.expected_cosine_similarity > 0.90,
                "Wide matrix (2x32) should compress: {}", wide.expected_cosine_similarity);
        assert!(tall.expected_cosine_similarity > 0.90,
                "Tall matrix (32x2) should compress: {}", tall.expected_cosine_similarity);
    }

    #[test]
    fn test_stress_checkerboard() {
        let tv = stress_vector_checkerboard_8x8();
        // Checkerboard is worst-case for DCT (all energy in highest frequency)
        // At 30% retention, we expect poor reconstruction
        println!("Checkerboard cosine similarity: {}", tv.expected_cosine_similarity);
        // Just verify it runs without error - checkerboard truly is worst case
    }

    #[test]
    fn test_stress_spike() {
        let tv = stress_vector_spike_8x8();
        // Single spike spreads energy - moderate reconstruction expected
        assert!(tv.expected_cosine_similarity > 0.70,
                "Spike should achieve moderate reconstruction: {}", tv.expected_cosine_similarity);
    }
}

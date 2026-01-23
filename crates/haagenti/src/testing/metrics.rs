//! Quality metrics for compression validation.
//!
//! Provides functions to measure reconstruction quality after compression:
//! - Mean Squared Error (MSE)
//! - Peak Signal-to-Noise Ratio (PSNR)
//! - Cosine Similarity
//! - Maximum Absolute Error

/// Aggregated quality metrics report.
#[derive(Debug, Clone, Default)]
pub struct QualityReport {
    /// Mean Squared Error between original and reconstructed.
    pub mse: f32,
    /// Peak Signal-to-Noise Ratio in decibels.
    pub psnr: f32,
    /// Cosine similarity (1.0 = identical, 0.0 = orthogonal).
    pub cosine_similarity: f32,
    /// Maximum absolute difference between any two corresponding elements.
    pub max_error: f32,
    /// Number of elements compared.
    pub num_elements: usize,
}

impl QualityReport {
    /// Returns true if the reconstruction is considered "good" quality.
    ///
    /// Based on empirical findings from HCT testing:
    /// - Cosine similarity >= 0.993 typically produces usable inference
    /// - PSNR >= 30 dB is generally acceptable
    pub fn is_good_quality(&self) -> bool {
        self.cosine_similarity >= 0.993 && self.psnr >= 30.0
    }

    /// Returns a human-readable quality grade.
    pub fn grade(&self) -> &'static str {
        if self.cosine_similarity >= 0.998 {
            "Excellent"
        } else if self.cosine_similarity >= 0.993 {
            "Good"
        } else if self.cosine_similarity >= 0.98 {
            "Acceptable"
        } else if self.cosine_similarity >= 0.95 {
            "Degraded"
        } else {
            "Poor"
        }
    }
}

impl std::fmt::Display for QualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MSE: {:.6}, PSNR: {:.2} dB, Cosine: {:.6}, MaxErr: {:.6} ({})",
            self.mse,
            self.psnr,
            self.cosine_similarity,
            self.max_error,
            self.grade()
        )
    }
}

/// Compute all quality metrics between original and reconstructed data.
///
/// # Arguments
///
/// * `original` - The original data before compression
/// * `reconstructed` - The data after compression/decompression roundtrip
///
/// # Returns
///
/// A `QualityReport` containing all computed metrics.
///
/// # Panics
///
/// Panics if `original` and `reconstructed` have different lengths.
pub fn compute_quality(original: &[f32], reconstructed: &[f32]) -> QualityReport {
    assert_eq!(
        original.len(),
        reconstructed.len(),
        "Arrays must have the same length"
    );

    if original.is_empty() {
        return QualityReport::default();
    }

    let mse_val = mse(original, reconstructed);
    let psnr_val = psnr_from_mse(mse_val, original);
    let cosine_val = cosine_similarity(original, reconstructed);
    let max_err = max_error(original, reconstructed);

    QualityReport {
        mse: mse_val,
        psnr: psnr_val,
        cosine_similarity: cosine_val,
        max_error: max_err,
        num_elements: original.len(),
    }
}

/// Compute Mean Squared Error between two arrays.
///
/// `MSE = (1/n) * Σ(original[i] - reconstructed[i])²`
pub fn mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() || original.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_sq / original.len() as f32
}

/// Compute Peak Signal-to-Noise Ratio from MSE.
///
/// PSNR = 10 * log10(MAX² / MSE)
///
/// Where MAX is the maximum value in the original signal.
/// Returns infinity if MSE is 0 (perfect reconstruction).
pub fn psnr_from_mse(mse: f32, original: &[f32]) -> f32 {
    if mse <= 0.0 {
        return f32::INFINITY;
    }

    let max_val = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_val <= 0.0 {
        return 0.0;
    }

    10.0 * (max_val.powi(2) / mse).log10()
}

/// Compute cosine similarity between two vectors.
///
/// cosine = (A · B) / (||A|| * ||B||)
///
/// Returns 1.0 for identical vectors, 0.0 for orthogonal vectors,
/// -1.0 for opposite vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm_a <= 1e-10 || norm_b <= 1e-10 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute maximum absolute error between two arrays.
pub fn max_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() {
        return 0.0;
    }

    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(mse(&a, &a), 0.0);
    }

    #[test]
    fn test_mse_known_value() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 5.0]; // One difference of 1.0
        assert!((mse(&a, &b) - 0.25).abs() < 1e-6); // 1²/4 = 0.25
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_psnr_perfect() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(psnr_from_mse(0.0, &a).is_infinite());
    }

    #[test]
    fn test_max_error() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.5, 3.0, 4.0];
        assert!((max_error(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_quality_report_grade() {
        let report = QualityReport {
            cosine_similarity: 0.999,
            ..Default::default()
        };
        assert_eq!(report.grade(), "Excellent");

        let report = QualityReport {
            cosine_similarity: 0.995,
            ..Default::default()
        };
        assert_eq!(report.grade(), "Good");

        let report = QualityReport {
            cosine_similarity: 0.985,
            ..Default::default()
        };
        assert_eq!(report.grade(), "Acceptable");

        let report = QualityReport {
            cosine_similarity: 0.96,
            ..Default::default()
        };
        assert_eq!(report.grade(), "Degraded");

        let report = QualityReport {
            cosine_similarity: 0.90,
            ..Default::default()
        };
        assert_eq!(report.grade(), "Poor");
    }

    #[test]
    fn test_compute_quality_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reconstructed = vec![1.01, 1.99, 3.02, 3.98, 5.01];

        let report = compute_quality(&original, &reconstructed);
        assert!(report.mse < 0.01);
        assert!(report.cosine_similarity > 0.999);
        assert!(report.max_error < 0.03);
    }
}

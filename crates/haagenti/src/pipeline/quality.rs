//! Quality validation utilities for compression pipeline.
//!
//! Provides metrics and sampling for validating compression quality.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Quality metrics for a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Tensor name.
    pub name: String,
    /// Cosine similarity (1.0 = identical).
    pub cosine_similarity: f32,
    /// Mean squared error.
    pub mse: f32,
    /// Maximum absolute error.
    pub max_error: f32,
    /// Peak signal-to-noise ratio (dB).
    pub psnr: f32,
    /// Number of elements.
    pub num_elements: usize,
}

impl QualityReport {
    /// Computes quality metrics between original and reconstructed tensors.
    pub fn compute(name: impl Into<String>, original: &[f32], reconstructed: &[f32]) -> Self {
        let name = name.into();
        let n = original.len();

        if n == 0 || n != reconstructed.len() {
            return Self {
                name,
                cosine_similarity: 0.0,
                mse: f32::INFINITY,
                max_error: f32::INFINITY,
                psnr: 0.0,
                num_elements: n,
            };
        }

        // Compute metrics
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        let mut mse_sum = 0.0f64;
        let mut max_err = 0.0f32;

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            let a = *a as f64;
            let b = *b as f64;

            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;

            let diff = (a - b).abs();
            mse_sum += diff * diff;
            max_err = max_err.max(diff as f32);
        }

        let cosine = if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
        } else {
            0.0
        };

        let mse = (mse_sum / n as f64) as f32;

        // PSNR relative to data range
        let data_max = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let psnr = if mse > 0.0 && data_max > 0.0 {
            20.0 * (data_max / mse.sqrt()).log10()
        } else if mse == 0.0 {
            f32::INFINITY
        } else {
            0.0
        };

        Self {
            name,
            cosine_similarity: cosine,
            mse,
            max_error: max_err,
            psnr,
            num_elements: n,
        }
    }

    /// Returns a quality grade based on cosine similarity.
    #[must_use]
    pub fn grade(&self) -> &'static str {
        if self.cosine_similarity >= 0.999 {
            "Excellent"
        } else if self.cosine_similarity >= 0.99 {
            "Good"
        } else if self.cosine_similarity >= 0.95 {
            "Acceptable"
        } else if self.cosine_similarity >= 0.90 {
            "Degraded"
        } else {
            "Poor"
        }
    }

    /// Returns true if quality is acceptable for inference.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.cosine_similarity >= 0.95
    }
}

/// Sampler for validating compression quality on a subset of tensors.
pub struct QualitySampler {
    /// Fraction of tensors to sample (0.0-1.0).
    sample_rate: f32,
    /// Collected reports.
    reports: Vec<QualityReport>,
    /// Random seed for reproducibility.
    seed: u64,
    /// Counter for deterministic sampling.
    counter: usize,
}

impl QualitySampler {
    /// Creates a new quality sampler.
    ///
    /// # Arguments
    /// * `sample_rate` - Fraction of tensors to sample (0.0-1.0)
    /// * `seed` - Random seed for reproducibility
    pub fn new(sample_rate: f32, seed: u64) -> Self {
        Self {
            sample_rate: sample_rate.clamp(0.0, 1.0),
            reports: Vec::new(),
            seed,
            counter: 0,
        }
    }

    /// Returns true if this tensor should be sampled.
    ///
    /// Uses a simple deterministic sampling based on tensor name hash.
    pub fn should_sample(&mut self, name: &str) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        if self.sample_rate <= 0.0 {
            return false;
        }

        // Simple hash-based sampling for reproducibility
        let hash = xxhash_rust::xxh3::xxh3_64(name.as_bytes());
        let threshold = (self.sample_rate * u64::MAX as f32) as u64;
        hash < threshold
    }

    /// Adds a quality report.
    pub fn add_report(&mut self, report: QualityReport) {
        self.reports.push(report);
    }

    /// Validates a tensor and adds the report if sampled.
    ///
    /// Returns the report if validation was performed, None if skipped.
    pub fn validate(
        &mut self,
        name: &str,
        original: &[f32],
        reconstructed: &[f32],
    ) -> Option<QualityReport> {
        if !self.should_sample(name) {
            return None;
        }

        let report = QualityReport::compute(name, original, reconstructed);
        self.reports.push(report.clone());
        Some(report)
    }

    /// Returns all collected reports.
    #[must_use]
    pub fn reports(&self) -> &[QualityReport] {
        &self.reports
    }

    /// Returns the number of samples collected.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.reports.len()
    }

    /// Computes aggregate statistics across all samples.
    #[must_use]
    pub fn summary(&self) -> QualitySummary {
        if self.reports.is_empty() {
            return QualitySummary::default();
        }

        let n = self.reports.len() as f32;

        let avg_cosine = self.reports.iter().map(|r| r.cosine_similarity).sum::<f32>() / n;
        let avg_mse = self.reports.iter().map(|r| r.mse).sum::<f32>() / n;
        let avg_psnr = self.reports.iter().map(|r| r.psnr).sum::<f32>() / n;

        let min_cosine = self
            .reports
            .iter()
            .map(|r| r.cosine_similarity)
            .fold(f32::INFINITY, f32::min);

        let max_mse = self
            .reports
            .iter()
            .map(|r| r.mse)
            .fold(0.0f32, f32::max);

        let acceptable_count = self.reports.iter().filter(|r| r.is_acceptable()).count();

        QualitySummary {
            sample_count: self.reports.len(),
            avg_cosine_similarity: avg_cosine,
            min_cosine_similarity: min_cosine,
            avg_mse,
            max_mse,
            avg_psnr,
            acceptable_fraction: acceptable_count as f32 / n,
        }
    }

    /// Returns reports grouped by quality grade.
    #[must_use]
    pub fn by_grade(&self) -> HashMap<&'static str, Vec<&QualityReport>> {
        let mut grouped: HashMap<&'static str, Vec<&QualityReport>> = HashMap::new();

        for report in &self.reports {
            grouped
                .entry(report.grade())
                .or_default()
                .push(report);
        }

        grouped
    }

    /// Returns the worst quality tensors.
    pub fn worst(&self, n: usize) -> Vec<&QualityReport> {
        let mut sorted: Vec<_> = self.reports.iter().collect();
        sorted.sort_by(|a, b| {
            a.cosine_similarity
                .partial_cmp(&b.cosine_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }
}

/// Aggregate quality statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualitySummary {
    /// Number of tensors sampled.
    pub sample_count: usize,
    /// Average cosine similarity.
    pub avg_cosine_similarity: f32,
    /// Minimum cosine similarity (worst tensor).
    pub min_cosine_similarity: f32,
    /// Average MSE.
    pub avg_mse: f32,
    /// Maximum MSE (worst tensor).
    pub max_mse: f32,
    /// Average PSNR.
    pub avg_psnr: f32,
    /// Fraction of tensors with acceptable quality.
    pub acceptable_fraction: f32,
}

impl QualitySummary {
    /// Returns overall quality grade.
    #[must_use]
    pub fn grade(&self) -> &'static str {
        if self.avg_cosine_similarity >= 0.999 && self.min_cosine_similarity >= 0.99 {
            "Excellent"
        } else if self.avg_cosine_similarity >= 0.99 && self.min_cosine_similarity >= 0.95 {
            "Good"
        } else if self.avg_cosine_similarity >= 0.95 && self.acceptable_fraction >= 0.95 {
            "Acceptable"
        } else if self.avg_cosine_similarity >= 0.90 {
            "Degraded"
        } else {
            "Poor"
        }
    }
}

/// Computes cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        0.0
    }
}

/// Computes mean squared error between two vectors.
#[must_use]
pub fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (*x as f64) - (*y as f64);
            diff * diff
        })
        .sum();

    (sum / a.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_report_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let report = QualityReport::compute("test", &data, &data);

        assert!((report.cosine_similarity - 1.0).abs() < 0.0001);
        assert!(report.mse < 0.0001);
        assert_eq!(report.grade(), "Excellent");
    }

    #[test]
    fn test_quality_report_different() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let noisy: Vec<f32> = original.iter().map(|x| x + 0.1).collect();
        let report = QualityReport::compute("test", &original, &noisy);

        assert!(report.cosine_similarity > 0.99);
        assert!(report.mse > 0.0);
        assert!(report.max_error < 0.2);
    }

    #[test]
    fn test_quality_sampler() {
        let mut sampler = QualitySampler::new(0.5, 42);

        // Should consistently sample or not based on name hash
        let first_result = sampler.should_sample("tensor.0");
        let second_result = sampler.should_sample("tensor.0");
        assert_eq!(first_result, second_result); // Deterministic

        // Different names may have different results
        // but the same name always gives the same result
    }

    #[test]
    fn test_quality_sampler_always() {
        let mut sampler = QualitySampler::new(1.0, 42);
        assert!(sampler.should_sample("any_tensor"));
    }

    #[test]
    fn test_quality_sampler_never() {
        let mut sampler = QualitySampler::new(0.0, 42);
        assert!(!sampler.should_sample("any_tensor"));
    }

    #[test]
    fn test_quality_summary() {
        let mut sampler = QualitySampler::new(1.0, 42);

        let data = vec![1.0, 2.0, 3.0];
        sampler.validate("t0", &data, &data);
        sampler.validate("t1", &data, &data);

        let summary = sampler.summary();
        assert_eq!(summary.sample_count, 2);
        assert!((summary.avg_cosine_similarity - 1.0).abs() < 0.0001);
        assert_eq!(summary.grade(), "Excellent");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        assert!(cosine_similarity(&a, &b).abs() < 0.0001); // Orthogonal
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.0001); // Identical
    }

    #[test]
    fn test_mean_squared_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = vec![2.0, 3.0, 4.0];

        assert!(mean_squared_error(&a, &b) < 0.0001); // Identical
        assert!((mean_squared_error(&a, &c) - 1.0).abs() < 0.0001); // Each off by 1
    }
}

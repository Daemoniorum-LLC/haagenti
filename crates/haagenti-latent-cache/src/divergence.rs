//! Divergence point prediction for cached latent reuse

use serde::{Deserialize, Serialize};

/// Predicted divergence point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergencePoint {
    /// Step at which to diverge from cached latent
    pub step: u32,
    /// Confidence in this prediction
    pub confidence: f32,
    /// Estimated quality at this step
    pub estimated_quality: f32,
    /// Steps saved by using cache
    pub steps_saved: u32,
    /// Percentage of work saved
    pub work_saved_percent: f32,
}

/// Predictor for determining optimal divergence points in latent caching.
///
/// When a cached latent is available with high similarity to the current prompt,
/// this predictor calculates the optimal step at which to diverge from the cache
/// and begin fresh computation, balancing quality and efficiency.
///
/// # Algorithm
///
/// Uses a non-linear mapping from similarity to divergence step:
/// - Higher similarity → later divergence (more steps saved)
/// - Similarity 0.85 → ~step 5 (for 20-step generation)
/// - Similarity 0.99 → ~step 18 (for 20-step generation)
///
/// # Example
///
/// ```ignore
/// let predictor = DivergencePredictor::new(20, 0.85);
/// if let Some(point) = predictor.predict(0.92) {
///     println!("Diverge at step {}, saving {}%", point.step, point.work_saved_percent);
/// }
/// ```
pub struct DivergencePredictor {
    /// Total steps in generation
    total_steps: u32,
    /// Minimum similarity for any cache use
    min_similarity: f32,
    /// Similarity-to-step mapping coefficients
    coefficients: DivergenceCoefficients,
}

/// Coefficients for divergence prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceCoefficients {
    /// Base step offset
    pub base_offset: f32,
    /// Similarity multiplier
    pub similarity_mult: f32,
    /// Quality penalty factor
    pub quality_penalty: f32,
}

impl Default for DivergenceCoefficients {
    fn default() -> Self {
        Self {
            base_offset: 0.0,
            similarity_mult: 0.8,
            quality_penalty: 0.1,
        }
    }
}

impl DivergencePredictor {
    /// Create a new predictor
    pub fn new(total_steps: u32, min_similarity: f32) -> Self {
        Self {
            total_steps,
            min_similarity,
            coefficients: DivergenceCoefficients::default(),
        }
    }

    /// Predict divergence point based on similarity
    pub fn predict(&self, similarity: f32) -> Option<DivergencePoint> {
        if similarity < self.min_similarity {
            return None;
        }

        // Higher similarity = later divergence point (more steps saved)
        // Mapping: similarity 0.85 -> step 5, similarity 0.99 -> step 18 (for 20 steps)

        let normalized_sim = (similarity - self.min_similarity) / (1.0 - self.min_similarity);

        // Non-linear mapping: more aggressive for high similarity
        let progress = normalized_sim.powf(0.7);

        let diverge_step = (self.coefficients.base_offset
            + progress * self.total_steps as f32 * self.coefficients.similarity_mult)
            .round() as u32;

        let diverge_step = diverge_step.min(self.total_steps - 2).max(1);
        let steps_saved = diverge_step;
        let work_saved_percent = steps_saved as f32 / self.total_steps as f32 * 100.0;

        // Estimate quality impact
        // Later divergence = higher quality, but also higher risk if similarity is wrong
        let quality_risk = (1.0 - similarity) * self.coefficients.quality_penalty;
        let estimated_quality =
            1.0 - quality_risk * (diverge_step as f32 / self.total_steps as f32);

        // Confidence is based on similarity and how conservative the prediction is
        let confidence = similarity * (1.0 - diverge_step as f32 / self.total_steps as f32 * 0.3);

        Some(DivergencePoint {
            step: diverge_step,
            confidence,
            estimated_quality,
            steps_saved,
            work_saved_percent,
        })
    }

    /// Get recommended checkpoints for caching
    pub fn recommended_checkpoints(&self) -> Vec<u32> {
        // Cache at strategic points that cover different similarity levels
        vec![
            self.total_steps / 4,      // Early (for high similarity matches)
            self.total_steps / 2,      // Mid (for medium similarity)
            self.total_steps * 3 / 4,  // Late (for lower similarity)
            self.total_steps - 2,      // Near-final (for very high similarity)
        ]
    }

    /// Update coefficients based on observed quality
    pub fn update_coefficients(&mut self, similarity: f32, used_step: u32, actual_quality: f32) {
        // Simple online learning
        let predicted = self.predict(similarity);
        if let Some(pred) = predicted {
            let _step_error = (pred.step as f32 - used_step as f32).abs();
            let quality_error = (pred.estimated_quality - actual_quality).abs();

            // Adjust quality penalty based on observed error
            if quality_error > 0.05 {
                self.coefficients.quality_penalty *= 1.1;
            } else if quality_error < 0.02 {
                self.coefficients.quality_penalty *= 0.95;
            }

            self.coefficients.quality_penalty = self.coefficients.quality_penalty.clamp(0.05, 0.3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_prediction() {
        let predictor = DivergencePredictor::new(20, 0.85);

        // Very high similarity should give late divergence
        let high_sim = predictor.predict(0.98).unwrap();
        assert!(high_sim.step >= 12);
        assert!(high_sim.work_saved_percent > 50.0);

        // Medium similarity should give earlier divergence
        let med_sim = predictor.predict(0.90).unwrap();
        assert!(med_sim.step < high_sim.step);

        // Below threshold should return None
        let low_sim = predictor.predict(0.80);
        assert!(low_sim.is_none());
    }

    #[test]
    fn test_checkpoints() {
        let predictor = DivergencePredictor::new(20, 0.85);
        let checkpoints = predictor.recommended_checkpoints();

        assert_eq!(checkpoints.len(), 4);
        assert!(checkpoints.contains(&5)); // 20/4
        assert!(checkpoints.contains(&10)); // 20/2
        assert!(checkpoints.contains(&15)); // 20*3/4
        assert!(checkpoints.contains(&18)); // 20-2
    }
}

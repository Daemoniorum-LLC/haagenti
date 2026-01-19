//! Mask prediction from prompt embeddings

use crate::{
    AttentionMask, CategoryMapping, HeadCategory, MaskBuilder, PromptCategory, Result,
    SparseError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for mask prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Target sparsity (fraction of heads to skip)
    pub target_sparsity: f32,
    /// Minimum heads to keep per layer
    pub min_active_heads: usize,
    /// Quality threshold for adaptive sparsity
    pub quality_threshold: f32,
    /// Step-dependent sparsity (more sparse early)
    pub step_adaptive: bool,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_layers: 70,
            target_sparsity: 0.5,
            min_active_heads: 4,
            quality_threshold: 0.98,
            step_adaptive: true,
        }
    }
}

/// Prediction result with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Predicted mask
    pub mask: AttentionMask,
    /// Confidence in prediction (0.0 - 1.0)
    pub confidence: f32,
    /// Detected prompt categories
    pub categories: Vec<PromptCategory>,
    /// Estimated quality impact (0.0 - 1.0, higher is better)
    pub estimated_quality: f32,
    /// Estimated compute savings (0.0 - 1.0)
    pub compute_savings: f32,
}

/// Predicts attention masks from prompt information
#[derive(Debug, Clone)]
pub struct MaskPredictor {
    config: PredictorConfig,
    category_mapping: CategoryMapping,
    /// Learned category-to-weight mappings
    category_profiles: HashMap<PromptCategory, HashMap<HeadCategory, f32>>,
    /// Step-dependent sparsity multipliers
    step_multipliers: Vec<f32>,
}

impl MaskPredictor {
    /// Create a new predictor with default settings
    pub fn new(config: PredictorConfig) -> Self {
        let category_mapping = CategoryMapping::sdxl_default();

        // Build category profiles from prompt category weights
        let mut category_profiles = HashMap::new();
        for category in &[
            PromptCategory::Portrait,
            PromptCategory::Landscape,
            PromptCategory::Abstract,
            PromptCategory::Photorealistic,
            PromptCategory::Anime,
            PromptCategory::Architecture,
            PromptCategory::Object,
            PromptCategory::Fantasy,
            PromptCategory::Mixed,
        ] {
            category_profiles.insert(*category, category.category_weights());
        }

        // Step multipliers: more sparsity early (high noise), less late (details)
        let step_multipliers = Self::compute_step_multipliers(50);

        Self {
            config,
            category_mapping,
            category_profiles,
            step_multipliers,
        }
    }

    /// Compute step-dependent sparsity multipliers
    fn compute_step_multipliers(total_steps: usize) -> Vec<f32> {
        (0..total_steps)
            .map(|step| {
                let t = step as f32 / total_steps as f32;
                // Early steps: higher sparsity (1.5x), late steps: lower (0.5x)
                1.5 - t
            })
            .collect()
    }

    /// Predict mask from prompt text
    pub fn predict(&self, prompt: &str, step: Option<u32>, total_steps: Option<u32>) -> Prediction {
        // Detect prompt categories
        let categories = PromptCategory::detect(prompt);

        // Compute head importance weights
        let weights = self.compute_weights(&categories);

        // Apply step-dependent adjustment
        let effective_sparsity = if self.config.step_adaptive {
            let step_idx = step.unwrap_or(0) as usize;
            let total = total_steps.unwrap_or(50) as usize;
            let multiplier = if step_idx < self.step_multipliers.len() {
                self.step_multipliers[step_idx]
            } else if total > 0 {
                let t = step_idx as f32 / total as f32;
                1.5 - t
            } else {
                1.0
            };
            (self.config.target_sparsity * multiplier).clamp(0.2, 0.8)
        } else {
            self.config.target_sparsity
        };

        // Build mask using category-based pruning
        let mask = MaskBuilder::new(self.config.num_heads, self.config.num_layers)
            .sparsity(effective_sparsity)
            .category_weights(weights.clone())
            .category_mapping(self.category_mapping.clone())
            .min_active(self.config.min_active_heads)
            .build();

        // Estimate quality impact
        let estimated_quality = self.estimate_quality(&mask, &weights);

        // Compute confidence based on category detection strength
        let confidence = self.compute_confidence(&categories, prompt);

        Prediction {
            mask: mask.clone(),
            confidence,
            categories: categories.to_vec(),
            estimated_quality,
            compute_savings: mask.overall_sparsity,
        }
    }

    /// Predict mask from embedding vector (for faster prediction)
    pub fn predict_from_embedding(
        &self,
        embedding: &[f32],
        step: Option<u32>,
        total_steps: Option<u32>,
    ) -> Prediction {
        // Use embedding to infer category weights directly
        // This is a simplified version - a real implementation would use a learned model
        let weights = self.embedding_to_weights(embedding);

        let effective_sparsity = if self.config.step_adaptive {
            let step_idx = step.unwrap_or(0) as usize;
            let total = total_steps.unwrap_or(50) as usize;
            let t = step_idx as f32 / total as f32;
            (self.config.target_sparsity * (1.5 - t)).clamp(0.2, 0.8)
        } else {
            self.config.target_sparsity
        };

        let mask = MaskBuilder::new(self.config.num_heads, self.config.num_layers)
            .sparsity(effective_sparsity)
            .category_weights(weights.clone())
            .category_mapping(self.category_mapping.clone())
            .min_active(self.config.min_active_heads)
            .build();

        let estimated_quality = self.estimate_quality(&mask, &weights);

        Prediction {
            mask: mask.clone(),
            confidence: 0.7, // Lower confidence for embedding-based
            categories: vec![PromptCategory::Mixed],
            estimated_quality,
            compute_savings: mask.overall_sparsity,
        }
    }

    /// Compute head category weights from detected prompt categories
    fn compute_weights(&self, categories: &[PromptCategory]) -> HashMap<HeadCategory, f32> {
        let mut combined = HashMap::new();

        for (i, category) in categories.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f32; // Decrease weight for less relevant categories

            if let Some(profile) = self.category_profiles.get(category) {
                for (&head_cat, &value) in profile {
                    *combined.entry(head_cat).or_insert(0.0) += value * weight;
                }
            }
        }

        // Normalize
        let max = combined.values().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for value in combined.values_mut() {
                *value /= max;
            }
        }

        // Ensure mandatory categories have high weight
        for category in HeadCategory::all() {
            if category.is_mandatory() {
                combined.insert(*category, 1.0);
            }
        }

        combined
    }

    /// Convert embedding to category weights
    fn embedding_to_weights(&self, embedding: &[f32]) -> HashMap<HeadCategory, f32> {
        // Simplified: use embedding dimensions to weight categories
        // Real implementation would use a learned projection
        let mut weights = HashMap::new();

        let dim = embedding.len();
        if dim > 0 {
            // Use different embedding regions for different categories
            let face_signal: f32 = embedding.iter().take(dim / 8).sum::<f32>().abs();
            let body_signal: f32 = embedding.iter().skip(dim / 8).take(dim / 8).sum::<f32>().abs();
            let bg_signal: f32 = embedding.iter().skip(dim / 4).take(dim / 4).sum::<f32>().abs();
            let style_signal: f32 = embedding.iter().skip(dim / 2).sum::<f32>().abs();

            let max = face_signal.max(body_signal).max(bg_signal).max(style_signal);
            if max > 0.0 {
                weights.insert(HeadCategory::Face, face_signal / max);
                weights.insert(HeadCategory::Body, body_signal / max);
                weights.insert(HeadCategory::Background, bg_signal / max);
                weights.insert(HeadCategory::Style, style_signal / max);
            }
        }

        // Add mandatory categories
        weights.insert(HeadCategory::General, 1.0);
        weights.insert(HeadCategory::Composition, 0.9);

        weights
    }

    /// Estimate quality impact of mask
    fn estimate_quality(&self, mask: &AttentionMask, weights: &HashMap<HeadCategory, f32>) -> f32 {
        // Quality is inversely related to how many important heads are masked
        let mut quality = 1.0f32;

        for layer in 0..mask.num_layers {
            let layer_importance: f32 = (0..mask.num_heads)
                .filter(|&head| !mask.is_active(layer, head))
                .map(|head| {
                    let category = self
                        .category_mapping
                        .get_category(layer, head)
                        .unwrap_or(HeadCategory::General);
                    weights.get(&category).copied().unwrap_or(0.5)
                })
                .sum();

            // Each layer contributes to quality loss
            quality *= 1.0 - (layer_importance * 0.001);
        }

        quality.clamp(0.9, 1.0)
    }

    /// Compute confidence based on category detection
    fn compute_confidence(&self, categories: &[PromptCategory], prompt: &str) -> f32 {
        if categories.is_empty() || categories[0] == PromptCategory::Mixed {
            return 0.5;
        }

        // Count keyword matches
        let primary = &categories[0];
        let matches = primary
            .keywords()
            .iter()
            .filter(|kw| prompt.to_lowercase().contains(*kw))
            .count();

        // More matches = higher confidence
        (0.6 + matches as f32 * 0.1).clamp(0.5, 0.95)
    }

    /// Update step multipliers for different total step counts
    pub fn set_total_steps(&mut self, total_steps: usize) {
        self.step_multipliers = Self::compute_step_multipliers(total_steps);
    }

    /// Get current configuration
    pub fn config(&self) -> &PredictorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_portrait() {
        let predictor = MaskPredictor::new(PredictorConfig::default());
        let prediction = predictor.predict("A portrait of a beautiful woman", None, None);

        assert!(prediction.categories.contains(&PromptCategory::Portrait));
        assert!(prediction.confidence > 0.6);
        assert!(prediction.compute_savings > 0.3);
    }

    #[test]
    fn test_predict_landscape() {
        let predictor = MaskPredictor::new(PredictorConfig::default());
        let prediction = predictor.predict("Mountain landscape at sunset", None, None);

        assert!(prediction.categories.contains(&PromptCategory::Landscape));
    }

    #[test]
    fn test_step_adaptive() {
        let config = PredictorConfig {
            step_adaptive: true,
            target_sparsity: 0.5,
            ..Default::default()
        };
        let predictor = MaskPredictor::new(config);

        // Early step should be more sparse
        let early = predictor.predict("test prompt", Some(0), Some(20));
        // Late step should be less sparse
        let late = predictor.predict("test prompt", Some(19), Some(20));

        assert!(early.mask.overall_sparsity > late.mask.overall_sparsity);
    }

    #[test]
    fn test_predict_from_embedding() {
        let predictor = MaskPredictor::new(PredictorConfig::default());
        let embedding: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0).sin()).collect();

        let prediction = predictor.predict_from_embedding(&embedding, Some(5), Some(20));
        assert!(prediction.compute_savings > 0.0);
    }
}

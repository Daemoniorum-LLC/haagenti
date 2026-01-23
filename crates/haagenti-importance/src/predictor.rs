//! Quality sensitivity prediction for layers

use haagenti_fragments::FragmentType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quality sensitivity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualitySensitivity {
    /// Can use very low quality (50% or less)
    VeryLow,
    /// Can use low quality (70%)
    Low,
    /// Needs medium quality (85%)
    Medium,
    /// Needs high quality (95%)
    High,
    /// Needs full quality (100%)
    Full,
}

impl QualitySensitivity {
    /// Get minimum quality level (0.0 - 1.0)
    pub fn min_quality(&self) -> f32 {
        match self {
            QualitySensitivity::VeryLow => 0.5,
            QualitySensitivity::Low => 0.7,
            QualitySensitivity::Medium => 0.85,
            QualitySensitivity::High => 0.95,
            QualitySensitivity::Full => 1.0,
        }
    }

    /// Get importance multiplier
    pub fn importance_multiplier(&self) -> f32 {
        match self {
            QualitySensitivity::VeryLow => 0.3,
            QualitySensitivity::Low => 0.5,
            QualitySensitivity::Medium => 0.7,
            QualitySensitivity::High => 0.9,
            QualitySensitivity::Full => 1.0,
        }
    }
}

/// Profile for a specific layer type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    /// Layer name pattern
    pub pattern: String,
    /// Base quality sensitivity
    pub sensitivity: QualitySensitivity,
    /// Sensitivity varies by step (early steps need less quality)
    pub step_adaptive: bool,
    /// Quality needed at step 0 (if step_adaptive)
    pub early_step_quality: f32,
    /// Quality needed at final step (if step_adaptive)
    pub final_step_quality: f32,
    /// Learned importance adjustment
    pub learned_adjustment: f32,
}

impl LayerProfile {
    /// Get quality needed at a given step
    pub fn quality_at_step(&self, step: u32, total_steps: u32) -> f32 {
        if !self.step_adaptive || total_steps == 0 {
            return self.sensitivity.min_quality();
        }

        let progress = step as f32 / total_steps as f32;
        let base = self.early_step_quality
            + (self.final_step_quality - self.early_step_quality) * progress;

        base.max(self.sensitivity.min_quality())
    }

    /// Get importance at a given step
    pub fn importance_at_step(&self, step: u32, total_steps: u32) -> f32 {
        let quality = self.quality_at_step(step, total_steps);
        quality * self.sensitivity.importance_multiplier() * (1.0 + self.learned_adjustment)
    }
}

/// Quality predictor for model layers
pub struct QualityPredictor {
    /// Default profiles by fragment type
    default_profiles: HashMap<FragmentType, LayerProfile>,
    /// Specific layer overrides
    layer_overrides: HashMap<String, LayerProfile>,
    /// Model-specific profiles
    model_profiles: HashMap<String, HashMap<String, LayerProfile>>,
}

impl QualityPredictor {
    /// Create a new quality predictor with default profiles
    pub fn new() -> Self {
        let mut default_profiles = HashMap::new();

        // Attention Q/K can tolerate lower quality at high noise steps
        default_profiles.insert(
            FragmentType::AttentionQuery,
            LayerProfile {
                pattern: "*.to_q*".into(),
                sensitivity: QualitySensitivity::Medium,
                step_adaptive: true,
                early_step_quality: 0.5,
                final_step_quality: 0.9,
                learned_adjustment: 0.0,
            },
        );

        default_profiles.insert(
            FragmentType::AttentionKey,
            LayerProfile {
                pattern: "*.to_k*".into(),
                sensitivity: QualitySensitivity::Medium,
                step_adaptive: true,
                early_step_quality: 0.5,
                final_step_quality: 0.9,
                learned_adjustment: 0.0,
            },
        );

        // V projections are more sensitive
        default_profiles.insert(
            FragmentType::AttentionValue,
            LayerProfile {
                pattern: "*.to_v*".into(),
                sensitivity: QualitySensitivity::High,
                step_adaptive: true,
                early_step_quality: 0.7,
                final_step_quality: 0.95,
                learned_adjustment: 0.0,
            },
        );

        // Output projections need high quality
        default_profiles.insert(
            FragmentType::AttentionOutput,
            LayerProfile {
                pattern: "*.to_out*".into(),
                sensitivity: QualitySensitivity::High,
                step_adaptive: false,
                early_step_quality: 0.9,
                final_step_quality: 0.95,
                learned_adjustment: 0.0,
            },
        );

        // FFN can use lower quality
        default_profiles.insert(
            FragmentType::FeedForward,
            LayerProfile {
                pattern: "*.mlp*".into(),
                sensitivity: QualitySensitivity::Low,
                step_adaptive: true,
                early_step_quality: 0.5,
                final_step_quality: 0.85,
                learned_adjustment: 0.0,
            },
        );

        // Normalization layers are small but important
        default_profiles.insert(
            FragmentType::Normalization,
            LayerProfile {
                pattern: "*.norm*".into(),
                sensitivity: QualitySensitivity::Full,
                step_adaptive: false,
                early_step_quality: 1.0,
                final_step_quality: 1.0,
                learned_adjustment: 0.0,
            },
        );

        // Embeddings need full quality
        default_profiles.insert(
            FragmentType::Embedding,
            LayerProfile {
                pattern: "*.embed*".into(),
                sensitivity: QualitySensitivity::Full,
                step_adaptive: false,
                early_step_quality: 1.0,
                final_step_quality: 1.0,
                learned_adjustment: 0.0,
            },
        );

        // Convolutions vary
        default_profiles.insert(
            FragmentType::Convolution,
            LayerProfile {
                pattern: "*.conv*".into(),
                sensitivity: QualitySensitivity::Medium,
                step_adaptive: true,
                early_step_quality: 0.6,
                final_step_quality: 0.9,
                learned_adjustment: 0.0,
            },
        );

        // Generic default
        default_profiles.insert(
            FragmentType::Generic,
            LayerProfile {
                pattern: "*".into(),
                sensitivity: QualitySensitivity::Medium,
                step_adaptive: false,
                early_step_quality: 0.8,
                final_step_quality: 0.8,
                learned_adjustment: 0.0,
            },
        );

        Self {
            default_profiles,
            layer_overrides: HashMap::new(),
            model_profiles: HashMap::new(),
        }
    }

    /// Get profile for a layer
    pub fn get_profile(&self, layer_name: &str, fragment_type: FragmentType) -> LayerProfile {
        // Check specific override first
        if let Some(profile) = self.layer_overrides.get(layer_name) {
            return profile.clone();
        }

        // Fall back to default by type
        self.default_profiles
            .get(&fragment_type)
            .cloned()
            .unwrap_or_else(|| self.default_profiles[&FragmentType::Generic].clone())
    }

    /// Get profile for a specific model
    pub fn get_model_profile(
        &self,
        model_id: &str,
        layer_name: &str,
        fragment_type: FragmentType,
    ) -> LayerProfile {
        // Check model-specific profile
        if let Some(model_profiles) = self.model_profiles.get(model_id) {
            if let Some(profile) = model_profiles.get(layer_name) {
                return profile.clone();
            }
        }

        // Fall back to general profile
        self.get_profile(layer_name, fragment_type)
    }

    /// Predict quality needed at step
    pub fn predict_quality(
        &self,
        layer_name: &str,
        fragment_type: FragmentType,
        step: u32,
        total_steps: u32,
    ) -> f32 {
        let profile = self.get_profile(layer_name, fragment_type);
        profile.quality_at_step(step, total_steps)
    }

    /// Predict importance at step
    pub fn predict_importance(
        &self,
        layer_name: &str,
        fragment_type: FragmentType,
        step: u32,
        total_steps: u32,
    ) -> f32 {
        let profile = self.get_profile(layer_name, fragment_type);
        profile.importance_at_step(step, total_steps)
    }

    /// Add a layer override
    pub fn add_override(&mut self, layer_name: impl Into<String>, profile: LayerProfile) {
        self.layer_overrides.insert(layer_name.into(), profile);
    }

    /// Add model-specific profile
    pub fn add_model_profile(
        &mut self,
        model_id: impl Into<String>,
        layer_name: impl Into<String>,
        profile: LayerProfile,
    ) {
        self.model_profiles
            .entry(model_id.into())
            .or_default()
            .insert(layer_name.into(), profile);
    }

    /// Update learned adjustment from history
    pub fn update_from_history(&mut self, layer_name: &str, adjustment: f32) {
        if let Some(profile) = self.layer_overrides.get_mut(layer_name) {
            // Exponential moving average
            let alpha = 0.1;
            profile.learned_adjustment =
                profile.learned_adjustment * (1.0 - alpha) + adjustment * alpha;
        }
    }

    /// Create SDXL-optimized predictor
    pub fn sdxl_optimized() -> Self {
        let mut predictor = Self::new();

        // SDXL-specific overrides
        predictor.add_override(
            "unet.down_blocks.0",
            LayerProfile {
                pattern: "unet.down_blocks.0.*".into(),
                sensitivity: QualitySensitivity::VeryLow,
                step_adaptive: true,
                early_step_quality: 0.4,
                final_step_quality: 0.7,
                learned_adjustment: 0.0,
            },
        );

        predictor.add_override(
            "vae.decoder",
            LayerProfile {
                pattern: "vae.decoder.*".into(),
                sensitivity: QualitySensitivity::Full,
                step_adaptive: false,
                early_step_quality: 1.0,
                final_step_quality: 1.0,
                learned_adjustment: 0.0,
            },
        );

        predictor
    }

    /// Create Flux-optimized predictor
    pub fn flux_optimized() -> Self {
        let mut predictor = Self::new();

        // Flux uses different architecture, adjust accordingly
        predictor.add_override(
            "transformer.single_transformer_blocks",
            LayerProfile {
                pattern: "*.single_transformer_blocks.*".into(),
                sensitivity: QualitySensitivity::Low,
                step_adaptive: true,
                early_step_quality: 0.5,
                final_step_quality: 0.85,
                learned_adjustment: 0.0,
            },
        );

        predictor
    }
}

impl Default for QualityPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_adaptive_quality() {
        let predictor = QualityPredictor::new();

        // Early step should need less quality
        let early = predictor.predict_quality("attn.to_q", FragmentType::AttentionQuery, 0, 20);
        let late = predictor.predict_quality("attn.to_q", FragmentType::AttentionQuery, 19, 20);

        assert!(early < late);
    }

    #[test]
    fn test_layer_sensitivity() {
        let predictor = QualityPredictor::new();

        // Normalization should need full quality
        let norm_quality =
            predictor.predict_quality("layer_norm", FragmentType::Normalization, 10, 20);

        // FFN should tolerate lower quality
        let ffn_quality = predictor.predict_quality("mlp", FragmentType::FeedForward, 10, 20);

        assert!(norm_quality > ffn_quality);
    }
}

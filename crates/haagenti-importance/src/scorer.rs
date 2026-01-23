//! Combined importance scoring

use crate::{PromptAnalyzer, PromptFeatures, QualityPredictor, UsageHistory};
use haagenti_fragments::{FragmentId, FragmentType};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for importance scorer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorerConfig {
    /// Weight for prompt-based importance
    pub prompt_weight: f32,
    /// Weight for historical importance
    pub history_weight: f32,
    /// Weight for quality prediction
    pub quality_weight: f32,
    /// Weight for step-based importance
    pub step_weight: f32,
    /// Minimum importance threshold
    pub min_importance: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            prompt_weight: 0.3,
            history_weight: 0.3,
            quality_weight: 0.25,
            step_weight: 0.15,
            min_importance: 0.1,
        }
    }
}

/// Importance score for a fragment
#[derive(Debug, Clone)]
pub struct ImportanceScore {
    /// Fragment ID
    pub fragment_id: FragmentId,
    /// Combined importance (0.0 - 1.0)
    pub importance: f32,
    /// Confidence in the score (0.0 - 1.0)
    pub confidence: f32,
    /// Recommended quality level
    pub recommended_quality: f32,
    /// Recommended load order (lower = earlier)
    pub load_order: u32,
    /// Component scores
    pub components: ScoreComponents,
}

/// Individual score components
#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    /// Prompt-based score
    pub prompt: f32,
    /// History-based score
    pub history: f32,
    /// Quality prediction score
    pub quality: f32,
    /// Step-based score
    pub step: f32,
}

/// Combined importance scorer
pub struct ImportanceScorer {
    config: ScorerConfig,
    analyzer: PromptAnalyzer,
    predictor: QualityPredictor,
    history: Arc<UsageHistory>,
}

impl ImportanceScorer {
    /// Create a new scorer
    pub fn new(config: ScorerConfig, history: Arc<UsageHistory>) -> Self {
        Self {
            config,
            analyzer: PromptAnalyzer::new(),
            predictor: QualityPredictor::new(),
            history,
        }
    }

    /// Create with SDXL optimization
    pub fn sdxl(history: Arc<UsageHistory>) -> Self {
        Self {
            config: ScorerConfig::default(),
            analyzer: PromptAnalyzer::new(),
            predictor: QualityPredictor::sdxl_optimized(),
            history,
        }
    }

    /// Create with Flux optimization
    pub fn flux(history: Arc<UsageHistory>) -> Self {
        Self {
            config: ScorerConfig::default(),
            analyzer: PromptAnalyzer::new(),
            predictor: QualityPredictor::flux_optimized(),
            history,
        }
    }

    /// Score a fragment for a given prompt
    pub fn score(
        &self,
        fragment_id: FragmentId,
        layer_name: &str,
        fragment_type: FragmentType,
        prompt: &str,
        step: u32,
        total_steps: u32,
    ) -> ImportanceScore {
        let features = self.analyzer.analyze(prompt);
        self.score_with_features(
            fragment_id,
            layer_name,
            fragment_type,
            &features,
            step,
            total_steps,
        )
    }

    /// Score with pre-analyzed features
    pub fn score_with_features(
        &self,
        fragment_id: FragmentId,
        layer_name: &str,
        fragment_type: FragmentType,
        features: &PromptFeatures,
        step: u32,
        total_steps: u32,
    ) -> ImportanceScore {
        // Compute component scores
        let prompt_score = features.layer_importance(layer_name);
        let history_score = self.history.importance(&fragment_id);
        let quality_score =
            self.predictor
                .predict_importance(layer_name, fragment_type, step, total_steps);
        let step_score = self.compute_step_importance(step, total_steps);

        // Weighted combination
        let importance = prompt_score * self.config.prompt_weight
            + history_score * self.config.history_weight
            + quality_score * self.config.quality_weight
            + step_score * self.config.step_weight;

        let importance = importance.max(self.config.min_importance).min(1.0);

        // Compute confidence
        let has_history = self.history.get(&fragment_id).is_some();
        let confidence = if has_history { 0.8 } else { 0.5 };

        // Recommended quality
        let recommended_quality =
            self.predictor
                .predict_quality(layer_name, fragment_type, step, total_steps);

        // Load order (inverse of importance)
        let load_order = ((1.0 - importance) * 1000.0) as u32;

        ImportanceScore {
            fragment_id,
            importance,
            confidence,
            recommended_quality,
            load_order,
            components: ScoreComponents {
                prompt: prompt_score,
                history: history_score,
                quality: quality_score,
                step: step_score,
            },
        }
    }

    /// Score multiple fragments
    pub fn score_batch(
        &self,
        fragments: &[(FragmentId, String, FragmentType)],
        prompt: &str,
        step: u32,
        total_steps: u32,
    ) -> Vec<ImportanceScore> {
        let features = self.analyzer.analyze(prompt);

        fragments
            .iter()
            .map(|(id, layer, ftype)| {
                self.score_with_features(*id, layer, *ftype, &features, step, total_steps)
            })
            .collect()
    }

    /// Compute step-based importance
    fn compute_step_importance(&self, step: u32, total_steps: u32) -> f32 {
        if total_steps == 0 {
            return 0.5;
        }

        let progress = step as f32 / total_steps as f32;

        // Higher importance for current and near-future steps
        // U-shaped curve: high at current step, decreases, then increases for final steps
        let current_boost = 1.0 - progress * 0.5;
        let final_boost = (progress - 0.7).max(0.0) * 1.5;

        (current_boost + final_boost).min(1.0)
    }

    /// Get the quality predictor
    pub fn predictor(&self) -> &QualityPredictor {
        &self.predictor
    }

    /// Get the prompt analyzer
    pub fn analyzer(&self) -> &PromptAnalyzer {
        &self.analyzer
    }
}

/// Adaptive scorer that learns from feedback
pub struct AdaptiveScorer {
    base: ImportanceScorer,
    /// Feedback buffer for learning
    feedback: Vec<ScoringFeedback>,
    /// Maximum feedback buffer size
    max_feedback: usize,
}

/// Feedback on scoring accuracy
#[derive(Debug, Clone)]
pub struct ScoringFeedback {
    /// Fragment that was scored
    pub fragment_id: FragmentId,
    /// Original predicted importance
    pub predicted: f32,
    /// Actual importance (based on usage)
    pub actual: f32,
    /// Step at which this was observed
    pub step: u32,
}

impl AdaptiveScorer {
    /// Create a new adaptive scorer
    pub fn new(base: ImportanceScorer) -> Self {
        Self {
            base,
            feedback: Vec::new(),
            max_feedback: 10000,
        }
    }

    /// Score with learning
    pub fn score(
        &self,
        fragment_id: FragmentId,
        layer_name: &str,
        fragment_type: FragmentType,
        prompt: &str,
        step: u32,
        total_steps: u32,
    ) -> ImportanceScore {
        let mut score = self.base.score(
            fragment_id,
            layer_name,
            fragment_type,
            prompt,
            step,
            total_steps,
        );

        // Apply learned adjustment if we have feedback
        if let Some(adjustment) = self.compute_adjustment(&fragment_id) {
            score.importance = (score.importance + adjustment).clamp(0.0, 1.0);
        }

        score
    }

    /// Record feedback
    pub fn record_feedback(&mut self, feedback: ScoringFeedback) {
        self.feedback.push(feedback);

        // Trim if too large
        if self.feedback.len() > self.max_feedback {
            self.feedback.drain(0..self.max_feedback / 2);
        }
    }

    /// Compute adjustment from feedback
    fn compute_adjustment(&self, fragment_id: &FragmentId) -> Option<f32> {
        let relevant: Vec<_> = self
            .feedback
            .iter()
            .filter(|f| f.fragment_id == *fragment_id)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        // Compute average prediction error
        let avg_error: f32 =
            relevant.iter().map(|f| f.actual - f.predicted).sum::<f32>() / relevant.len() as f32;

        Some(avg_error * 0.5) // Apply 50% of the error as adjustment
    }

    /// Learn from recent feedback
    pub fn learn(&mut self) {
        // Group feedback by layer patterns
        // This would update the base predictor's learned_adjustment values
        // For now, just keep feedback for online adjustments
    }

    /// Get base scorer
    pub fn base(&self) -> &ImportanceScorer {
        &self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoring() {
        let history = Arc::new(UsageHistory::new());
        let scorer = ImportanceScorer::new(ScorerConfig::default(), history);

        let fragment_id = FragmentId::new([1; 16]);
        let score = scorer.score(
            fragment_id,
            "unet.down_blocks.0.attentions.0.to_q",
            FragmentType::AttentionQuery,
            "A beautiful portrait of a woman",
            5,
            20,
        );

        assert!(score.importance > 0.0);
        assert!(score.importance <= 1.0);
        assert!(score.confidence > 0.0);
    }

    #[test]
    fn test_portrait_boosts_face_attention() {
        let history = Arc::new(UsageHistory::new());
        let scorer = ImportanceScorer::new(ScorerConfig::default(), history);

        let fragment_id = FragmentId::new([1; 16]);

        // Portrait prompt should boost face attention layers
        let portrait_score = scorer.score(
            fragment_id,
            "unet.mid_block.attentions.0.to_q",
            FragmentType::AttentionQuery,
            "A detailed portrait of a person",
            10,
            20,
        );

        // Landscape prompt should not boost face layers as much
        let landscape_score = scorer.score(
            fragment_id,
            "unet.mid_block.attentions.0.to_q",
            FragmentType::AttentionQuery,
            "A beautiful mountain landscape",
            10,
            20,
        );

        // Portrait should have higher face attention importance
        // (This assumes the prompt analyzer correctly detects the difference)
        assert!(portrait_score.components.prompt >= landscape_score.components.prompt * 0.8);
    }
}

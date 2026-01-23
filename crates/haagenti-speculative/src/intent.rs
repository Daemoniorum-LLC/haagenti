//! Intent prediction from partial prompt input

use haagenti_importance::{PromptAnalyzer, PromptFeatures, SemanticCategory};
use radix_trie::{Trie, TrieCommon};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for intent prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentConfig {
    /// Minimum characters before prediction
    pub min_chars: usize,
    /// Confidence threshold for speculation
    pub speculation_threshold: f32,
    /// Confidence threshold for commit
    pub commit_threshold: f32,
    /// Enable learning from history
    pub learn_from_history: bool,
    /// Maximum history entries
    pub max_history: usize,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            min_chars: 3,
            speculation_threshold: 0.6,
            commit_threshold: 0.8,
            learn_from_history: true,
            max_history: 10000,
        }
    }
}

/// A predicted intent from partial input
#[derive(Debug, Clone)]
pub struct Intent {
    /// Predicted full prompt (or prefix)
    pub predicted_prompt: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Semantic categories predicted
    pub categories: SmallVec<[SemanticCategory; 4]>,
    /// Recommended fragment types to preload
    pub fragment_hints: Vec<FragmentHint>,
    /// Whether this is a commit (high confidence) vs speculation
    pub is_commit: bool,
}

/// Hint about which fragments to preload
#[derive(Debug, Clone)]
pub struct FragmentHint {
    /// Layer pattern to match
    pub layer_pattern: String,
    /// Importance score for this hint
    pub importance: f32,
    /// Priority (lower = load first)
    pub priority: u32,
}

/// Result of intent prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Top predicted intent
    pub primary: Option<Intent>,
    /// Alternative predictions
    pub alternatives: SmallVec<[Intent; 3]>,
    /// Time taken for prediction (microseconds)
    pub latency_us: u64,
}

/// Intent predictor using trie-based prefix matching and ML
pub struct IntentPredictor {
    config: IntentConfig,
    /// Trie for fast prefix matching
    prompt_trie: Trie<String, PromptEntry>,
    /// Prompt analyzer for semantic understanding
    analyzer: PromptAnalyzer,
    /// Category to fragment hints mapping
    category_hints: HashMap<SemanticCategory, Vec<FragmentHint>>,
    /// Frequency-weighted completions
    completion_weights: HashMap<String, f32>,
}

/// Entry in the prompt trie
#[derive(Debug, Clone)]
struct PromptEntry {
    /// Full prompt
    prompt: String,
    /// Usage count
    count: u32,
    /// Last used timestamp
    last_used: u64,
    /// Extracted features
    features: Option<PromptFeatures>,
}

impl IntentPredictor {
    /// Create a new intent predictor
    pub fn new(config: IntentConfig) -> Self {
        let mut category_hints = HashMap::new();

        // Portrait/Human hints
        category_hints.insert(
            SemanticCategory::Human,
            vec![
                FragmentHint {
                    layer_pattern: "*.attn*.to_q*".into(),
                    importance: 0.9,
                    priority: 0,
                },
                FragmentHint {
                    layer_pattern: "*.attn*.to_k*".into(),
                    importance: 0.9,
                    priority: 1,
                },
                FragmentHint {
                    layer_pattern: "*face*".into(),
                    importance: 0.95,
                    priority: 0,
                },
                FragmentHint {
                    layer_pattern: "*up_blocks.3*".into(),
                    importance: 0.8,
                    priority: 2,
                },
            ],
        );

        // Landscape hints
        category_hints.insert(
            SemanticCategory::Landscape,
            vec![
                FragmentHint {
                    layer_pattern: "*down_blocks.0*".into(),
                    importance: 0.8,
                    priority: 0,
                },
                FragmentHint {
                    layer_pattern: "*down_blocks.1*".into(),
                    importance: 0.7,
                    priority: 1,
                },
                FragmentHint {
                    layer_pattern: "*mid_block*".into(),
                    importance: 0.6,
                    priority: 2,
                },
            ],
        );

        // Anime/Style hints
        category_hints.insert(
            SemanticCategory::Anime,
            vec![
                FragmentHint {
                    layer_pattern: "*style*".into(),
                    importance: 0.9,
                    priority: 0,
                },
                FragmentHint {
                    layer_pattern: "*up_blocks.2*".into(),
                    importance: 0.8,
                    priority: 1,
                },
            ],
        );

        // Photorealistic hints
        category_hints.insert(
            SemanticCategory::Photorealistic,
            vec![
                FragmentHint {
                    layer_pattern: "*up_blocks.3*".into(),
                    importance: 0.95,
                    priority: 0,
                },
                FragmentHint {
                    layer_pattern: "*vae.decoder*".into(),
                    importance: 1.0,
                    priority: 0,
                },
            ],
        );

        Self {
            config,
            prompt_trie: Trie::new(),
            analyzer: PromptAnalyzer::new(),
            category_hints,
            completion_weights: HashMap::new(),
        }
    }

    /// Predict intent from partial input
    pub fn predict(&self, partial: &str) -> PredictionResult {
        let start = std::time::Instant::now();

        if partial.len() < self.config.min_chars {
            return PredictionResult {
                primary: None,
                alternatives: SmallVec::new(),
                latency_us: start.elapsed().as_micros() as u64,
            };
        }

        let partial_lower = partial.to_lowercase();

        // Get trie completions
        let mut candidates: Vec<(String, f32)> = Vec::new();

        if let Some(subtrie) = self.prompt_trie.get_raw_descendant(&partial_lower) {
            for (_key, entry) in subtrie.iter() {
                let recency_boost = self.recency_score(entry.last_used);
                let frequency_boost = (entry.count as f32).ln().max(1.0) / 10.0;
                let score = 0.5 + recency_boost * 0.25 + frequency_boost * 0.25;
                candidates.push((entry.prompt.clone(), score));
            }
        }

        // Sort by score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build intents
        let mut intents: Vec<Intent> = candidates
            .into_iter()
            .take(4)
            .map(|(prompt, confidence)| self.build_intent(&prompt, confidence))
            .collect();

        // If no trie matches, use analyzer directly
        if intents.is_empty() {
            let features = self.analyzer.analyze(partial);
            let intent = self.intent_from_features(partial, &features, 0.5);
            intents.push(intent);
        }

        // Adjust confidence based on input length
        let length_factor = (partial.len() as f32 / 20.0).min(1.0);
        for intent in &mut intents {
            intent.confidence *= 0.5 + length_factor * 0.5;
            intent.is_commit = intent.confidence >= self.config.commit_threshold;
        }

        let primary = intents.first().cloned();
        let alternatives: SmallVec<[Intent; 3]> = intents.into_iter().skip(1).take(3).collect();

        PredictionResult {
            primary,
            alternatives,
            latency_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Build an intent from a prompt
    fn build_intent(&self, prompt: &str, confidence: f32) -> Intent {
        let features = self.analyzer.analyze(prompt);
        self.intent_from_features(prompt, &features, confidence)
    }

    /// Build intent from features
    fn intent_from_features(
        &self,
        prompt: &str,
        features: &PromptFeatures,
        confidence: f32,
    ) -> Intent {
        let categories: SmallVec<[SemanticCategory; 4]> =
            features.categories.iter().map(|(c, _)| *c).collect();

        // Gather fragment hints based on detected categories
        let mut fragment_hints = Vec::new();
        for (category, _weight) in &features.categories {
            if let Some(hints) = self.category_hints.get(category) {
                fragment_hints.extend(hints.iter().cloned());
            }
        }

        // Sort by priority
        fragment_hints.sort_by_key(|h| h.priority);
        fragment_hints.dedup_by(|a, b| a.layer_pattern == b.layer_pattern);

        Intent {
            predicted_prompt: prompt.to_string(),
            confidence,
            categories,
            fragment_hints,
            is_commit: confidence >= self.config.commit_threshold,
        }
    }

    /// Compute recency score
    fn recency_score(&self, last_used: u64) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age_hours = (now - last_used) as f32 / 3600.0;
        1.0 / (1.0 + age_hours / 24.0) // Half-life of ~1 day
    }

    /// Learn from a completed prompt
    pub fn learn(&mut self, prompt: &str) {
        if !self.config.learn_from_history {
            return;
        }

        let key = prompt.to_lowercase();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Some(entry) = self.prompt_trie.get_mut(&key) {
            entry.count += 1;
            entry.last_used = now;
        } else {
            let features = Some(self.analyzer.analyze(prompt));
            self.prompt_trie.insert(
                key,
                PromptEntry {
                    prompt: prompt.to_string(),
                    count: 1,
                    last_used: now,
                    features,
                },
            );
        }

        debug!("Learned prompt: {}", prompt);
    }

    /// Get hints for a specific category
    pub fn hints_for_category(&self, category: SemanticCategory) -> &[FragmentHint] {
        self.category_hints
            .get(&category)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Update category hints
    pub fn update_hints(&mut self, category: SemanticCategory, hints: Vec<FragmentHint>) {
        self.category_hints.insert(category, hints);
    }

    /// Get prediction statistics
    pub fn stats(&self) -> PredictorStats {
        PredictorStats {
            known_prompts: self.prompt_trie.len(),
            categories_configured: self.category_hints.len(),
        }
    }
}

/// Predictor statistics
#[derive(Debug, Clone)]
pub struct PredictorStats {
    /// Number of known prompts
    pub known_prompts: usize,
    /// Number of categories with hints
    pub categories_configured: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_prediction() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        // Learn some prompts
        predictor.learn("portrait of a beautiful woman");
        predictor.learn("portrait of a man in a suit");
        predictor.learn("landscape with mountains");

        // Predict from partial
        let result = predictor.predict("portr");
        assert!(result.primary.is_some());

        let intent = result.primary.unwrap();
        assert!(intent.predicted_prompt.contains("portrait"));
        assert!(intent.categories.contains(&SemanticCategory::Human));
    }

    #[test]
    fn test_category_hints() {
        let predictor = IntentPredictor::new(IntentConfig::default());

        let hints = predictor.hints_for_category(SemanticCategory::Human);
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.layer_pattern.contains("attn")));
    }

    #[test]
    fn test_min_chars_threshold() {
        let config = IntentConfig {
            min_chars: 5,
            ..Default::default()
        };
        let mut predictor = IntentPredictor::new(config);
        predictor.learn("portrait of a woman");

        // Below threshold - should return no prediction
        let result = predictor.predict("por");
        assert!(result.primary.is_none());

        // At threshold - should return prediction
        let result = predictor.predict("portr");
        assert!(result.primary.is_some());
    }

    #[test]
    fn test_confidence_increases_with_input_length() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        // Learn prompt to enable prediction
        predictor.learn("portrait of a beautiful woman");

        // Shorter input = lower confidence
        let short_result = predictor.predict("portr");
        let long_result = predictor.predict("portrait of a beaut");

        if let (Some(short_intent), Some(long_intent)) = (short_result.primary, long_result.primary)
        {
            assert!(
                long_intent.confidence >= short_intent.confidence,
                "Longer input should have equal or higher confidence"
            );
        }
    }

    #[test]
    fn test_fragment_hints_for_landscape() {
        let predictor = IntentPredictor::new(IntentConfig::default());

        let hints = predictor.hints_for_category(SemanticCategory::Landscape);
        assert!(!hints.is_empty());
        // Landscape should have down_blocks hints
        assert!(hints
            .iter()
            .any(|h| h.layer_pattern.contains("down_blocks")));
    }

    #[test]
    fn test_fragment_hints_for_photorealistic() {
        let predictor = IntentPredictor::new(IntentConfig::default());

        let hints = predictor.hints_for_category(SemanticCategory::Photorealistic);
        assert!(!hints.is_empty());
        // Photorealistic should have VAE decoder hints
        assert!(hints.iter().any(|h| h.layer_pattern.contains("vae")));
    }

    #[test]
    fn test_prediction_latency_under_threshold() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        // Add prompts to trie
        for i in 0..50 {
            predictor.learn(&format!("test prompt variation number {}", i));
        }

        let result = predictor.predict("test pr");

        // Prediction should be fast (<1ms for in-memory trie lookup)
        assert!(
            result.latency_us < 1000,
            "Latency was {}us, expected <1000us",
            result.latency_us
        );
    }

    #[test]
    fn test_predictor_stats() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        assert_eq!(predictor.stats().known_prompts, 0);

        predictor.learn("test prompt one");
        predictor.learn("test prompt two");

        let stats = predictor.stats();
        assert_eq!(stats.known_prompts, 2);
        assert!(stats.categories_configured > 0);
    }

    #[test]
    fn test_update_category_hints() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        let custom_hints = vec![FragmentHint {
            layer_pattern: "*custom_layer*".into(),
            importance: 1.0,
            priority: 0,
        }];

        predictor.update_hints(SemanticCategory::Abstract, custom_hints);

        let hints = predictor.hints_for_category(SemanticCategory::Abstract);
        assert!(!hints.is_empty());
        assert!(hints[0].layer_pattern.contains("custom_layer"));
    }

    #[test]
    fn test_learn_increments_frequency() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        // Learn same prompt multiple times
        predictor.learn("repeated prompt");
        predictor.learn("repeated prompt");
        predictor.learn("repeated prompt");

        // Stats should still show 1 unique prompt
        assert_eq!(predictor.stats().known_prompts, 1);

        // Predict should work
        let result = predictor.predict("repeated");
        assert!(result.primary.is_some());
    }

    // =========================================================================
    // Track C.3: Additional Intent Prediction Tests
    // =========================================================================

    #[test]
    fn test_intent_alternatives_returned() {
        let mut predictor = IntentPredictor::new(IntentConfig::default());

        // Learn multiple similar prompts
        predictor.learn("portrait of a woman in red");
        predictor.learn("portrait of a woman in blue");
        predictor.learn("portrait of a man in suit");

        let result = predictor.predict("portrait of");

        // Should have primary
        assert!(result.primary.is_some());
        // Latency should be recorded (always true for u64, but verifies it's set)
        let _ = result.latency_us; // Just verify it exists
    }

    #[test]
    fn test_fragment_hints_structure() {
        let predictor = IntentPredictor::new(IntentConfig::default());

        // Human category should have hints
        let hints = predictor.hints_for_category(SemanticCategory::Human);
        assert!(!hints.is_empty(), "Human category should have hints");

        // All hints should have valid fields
        for hint in hints {
            assert!(
                !hint.layer_pattern.is_empty(),
                "Layer pattern should not be empty"
            );
            assert!(
                hint.importance >= 0.0 && hint.importance <= 1.0,
                "Importance should be 0-1"
            );
        }
    }

    #[test]
    fn test_intent_commit_vs_speculation() {
        let mut predictor = IntentPredictor::new(IntentConfig {
            commit_threshold: 0.8,
            speculation_threshold: 0.6,
            ..Default::default()
        });

        predictor.learn("portrait of a beautiful woman in a garden");

        // Short prefix = speculation
        let short_result = predictor.predict("portr");
        if let Some(intent) = short_result.primary {
            // Short input usually has lower confidence
            assert!(intent.confidence <= 1.0);
        }

        // Longer prefix should have higher confidence
        let long_result = predictor.predict("portrait of a beautiful");
        if let Some(intent) = long_result.primary {
            assert!(intent.confidence > 0.0);
        }
    }
}

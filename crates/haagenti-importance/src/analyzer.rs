//! Prompt analysis for fragment importance prediction

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Semantic category detected in prompt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticCategory {
    /// Human subjects (face, portrait, person)
    Human,
    /// Animals
    Animal,
    /// Landscapes and nature
    Landscape,
    /// Architecture and buildings
    Architecture,
    /// Abstract/artistic styles
    Abstract,
    /// Text rendering
    Text,
    /// Vehicles
    Vehicle,
    /// Food and objects
    Object,
    /// Fantasy/sci-fi elements
    Fantasy,
    /// Photorealistic style
    Photorealistic,
    /// Anime/cartoon style
    Anime,
    /// Unknown/general
    General,
}

impl SemanticCategory {
    /// Get attention pattern hints for this category
    pub fn attention_hints(&self) -> AttentionHints {
        match self {
            SemanticCategory::Human => AttentionHints {
                face_attention: 0.9,
                body_attention: 0.7,
                background_attention: 0.3,
                detail_level: 0.8,
            },
            SemanticCategory::Landscape => AttentionHints {
                face_attention: 0.1,
                body_attention: 0.2,
                background_attention: 0.9,
                detail_level: 0.6,
            },
            SemanticCategory::Abstract => AttentionHints {
                face_attention: 0.2,
                body_attention: 0.3,
                background_attention: 0.5,
                detail_level: 0.4,
            },
            SemanticCategory::Photorealistic => AttentionHints {
                face_attention: 0.5,
                body_attention: 0.5,
                background_attention: 0.5,
                detail_level: 0.95,
            },
            SemanticCategory::Anime => AttentionHints {
                face_attention: 0.8,
                body_attention: 0.6,
                background_attention: 0.4,
                detail_level: 0.5,
            },
            _ => AttentionHints::default(),
        }
    }
}

/// Attention pattern hints from semantic analysis
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AttentionHints {
    /// Expected face/portrait attention (0-1)
    pub face_attention: f32,
    /// Expected body/figure attention (0-1)
    pub body_attention: f32,
    /// Expected background attention (0-1)
    pub background_attention: f32,
    /// Required detail level (0-1)
    pub detail_level: f32,
}

/// Extracted features from a prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFeatures {
    /// Detected semantic categories (ordered by confidence)
    pub categories: SmallVec<[(SemanticCategory, f32); 4]>,
    /// Estimated complexity (0-1)
    pub complexity: f32,
    /// Style keywords detected
    pub style_keywords: SmallVec<[String; 8]>,
    /// Subject keywords detected
    pub subject_keywords: SmallVec<[String; 8]>,
    /// Negative prompt strength (if provided)
    pub negative_strength: f32,
    /// Expected generation steps (affects loading schedule)
    pub expected_steps: Option<u32>,
    /// Combined attention hints
    pub attention: AttentionHints,
}

impl PromptFeatures {
    /// Get primary category
    pub fn primary_category(&self) -> SemanticCategory {
        self.categories
            .first()
            .map(|(c, _)| *c)
            .unwrap_or(SemanticCategory::General)
    }

    /// Check if high detail is required
    pub fn requires_high_detail(&self) -> bool {
        self.attention.detail_level > 0.7
    }

    /// Get layer importance multipliers based on features
    pub fn layer_importance(&self, layer_name: &str) -> f32 {
        let lower = layer_name.to_lowercase();

        // VAE decoder always important
        if lower.contains("vae") && lower.contains("decoder") {
            return 1.0;
        }

        // Face-related attention layers
        if lower.contains("attn") && (lower.contains("face") || lower.contains("portrait")) {
            return self.attention.face_attention;
        }

        // Later UNet blocks (fine details)
        if lower.contains("up_blocks") {
            let block_num = extract_block_number(&lower);
            // Higher block numbers = finer details
            let base = 0.5 + (block_num as f32 * 0.1).min(0.5);
            return base * self.attention.detail_level;
        }

        // Down blocks (coarse features)
        if lower.contains("down_blocks") {
            let block_num = extract_block_number(&lower);
            // Lower block numbers = more important for structure
            return 0.8 - (block_num as f32 * 0.1).min(0.3);
        }

        // Mid block (global structure)
        if lower.contains("mid_block") {
            return 0.7;
        }

        // Default importance
        0.5
    }
}

fn extract_block_number(s: &str) -> u32 {
    s.chars()
        .filter(|c| c.is_ascii_digit())
        .take(1)
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

/// Prompt analyzer for extracting semantic features
pub struct PromptAnalyzer {
    /// Keyword to category mapping
    category_keywords: HashMap<String, (SemanticCategory, f32)>,
    /// Style keywords
    style_keywords: Vec<String>,
}

impl PromptAnalyzer {
    /// Create a new prompt analyzer
    pub fn new() -> Self {
        let mut category_keywords = HashMap::new();

        // Human/Portrait keywords
        for kw in &[
            "portrait", "face", "person", "woman", "man", "girl", "boy", "human", "people",
        ] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Human, 0.9));
        }

        // Landscape keywords
        for kw in &[
            "landscape",
            "mountain",
            "forest",
            "ocean",
            "sky",
            "sunset",
            "nature",
            "scenery",
        ] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Landscape, 0.8));
        }

        // Architecture keywords
        for kw in &[
            "building",
            "architecture",
            "city",
            "street",
            "house",
            "castle",
            "interior",
        ] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Architecture, 0.8));
        }

        // Animal keywords
        for kw in &["cat", "dog", "animal", "bird", "horse", "wildlife", "pet"] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Animal, 0.8));
        }

        // Style keywords
        for kw in &["photorealistic", "realistic", "photography", "photo"] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Photorealistic, 0.9));
        }
        for kw in &["anime", "manga", "cartoon", "illustration"] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Anime, 0.9));
        }
        for kw in &["abstract", "artistic", "surreal", "conceptual"] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Abstract, 0.7));
        }
        for kw in &["fantasy", "sci-fi", "futuristic", "magical", "dragon"] {
            category_keywords.insert(kw.to_string(), (SemanticCategory::Fantasy, 0.8));
        }

        let style_keywords = vec![
            "detailed",
            "intricate",
            "8k",
            "4k",
            "hdr",
            "cinematic",
            "dramatic",
            "soft",
            "sharp",
            "vibrant",
            "muted",
            "dark",
            "bright",
            "professional",
            "amateur",
            "raw",
            "processed",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            category_keywords,
            style_keywords,
        }
    }

    /// Analyze a prompt
    pub fn analyze(&self, prompt: &str) -> PromptFeatures {
        let prompt_lower = prompt.to_lowercase();
        let words: Vec<&str> = prompt_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        // Detect categories
        let mut category_scores: HashMap<SemanticCategory, f32> = HashMap::new();
        let mut detected_styles = SmallVec::new();
        let mut detected_subjects = SmallVec::new();

        for word in &words {
            if let Some((category, confidence)) = self.category_keywords.get(*word) {
                *category_scores.entry(*category).or_default() += confidence;
                detected_subjects.push(word.to_string());
            }

            if self.style_keywords.iter().any(|s| s == *word) {
                detected_styles.push(word.to_string());
            }
        }

        // Sort categories by score
        let mut categories: SmallVec<[(SemanticCategory, f32); 4]> =
            category_scores.into_iter().collect();
        categories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        categories.truncate(4);

        // Normalize scores
        let max_score = categories.first().map(|(_, s)| *s).unwrap_or(1.0);
        for (_, score) in &mut categories {
            *score /= max_score;
        }

        // Estimate complexity
        let complexity = (words.len() as f32 / 50.0).min(1.0);

        // Compute attention hints
        let attention = if let Some((primary, _)) = categories.first() {
            primary.attention_hints()
        } else {
            AttentionHints::default()
        };

        // Adjust for style keywords
        let detail_boost = if detected_styles
            .iter()
            .any(|s: &String| s.contains("8k") || s.contains("detailed") || s.contains("intricate"))
        {
            0.2
        } else {
            0.0
        };

        let mut attention = attention;
        attention.detail_level = (attention.detail_level + detail_boost).min(1.0);

        PromptFeatures {
            categories,
            complexity,
            style_keywords: detected_styles,
            subject_keywords: detected_subjects,
            negative_strength: 0.0,
            expected_steps: None,
            attention,
        }
    }

    /// Analyze prompt with negative prompt
    pub fn analyze_with_negative(&self, prompt: &str, negative: &str) -> PromptFeatures {
        let mut features = self.analyze(prompt);

        // Negative prompt affects detail requirements
        let negative_lower = negative.to_lowercase();
        let negative_words: Vec<&str> = negative_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        features.negative_strength = (negative_words.len() as f32 / 20.0).min(1.0);

        // Strong negative prompts often require more precision
        if features.negative_strength > 0.5 {
            features.attention.detail_level = (features.attention.detail_level + 0.1).min(1.0);
        }

        features
    }
}

impl Default for PromptAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portrait_detection() {
        let analyzer = PromptAnalyzer::new();
        let features = analyzer.analyze("A beautiful portrait of a woman with flowing hair");

        assert_eq!(features.primary_category(), SemanticCategory::Human);
        assert!(features.attention.face_attention > 0.7);
    }

    #[test]
    fn test_landscape_detection() {
        let analyzer = PromptAnalyzer::new();
        let features = analyzer.analyze("Majestic mountain landscape at sunset");

        assert_eq!(features.primary_category(), SemanticCategory::Landscape);
        assert!(features.attention.background_attention > 0.7);
    }

    #[test]
    fn test_high_detail_detection() {
        let analyzer = PromptAnalyzer::new();
        let features = analyzer.analyze("8k detailed photorealistic portrait");

        assert!(features.requires_high_detail());
        assert!(features.attention.detail_level > 0.8);
    }
}

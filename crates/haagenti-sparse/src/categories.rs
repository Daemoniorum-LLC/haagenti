//! Attention head and prompt category definitions

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Categories of attention heads based on their learned function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeadCategory {
    /// Face/portrait focused heads (eyes, mouth, structure)
    Face,
    /// Body and pose attention
    Body,
    /// Background and scene composition
    Background,
    /// Style and texture patterns
    Style,
    /// Composition and layout
    Composition,
    /// Fine detail refinement
    Detail,
    /// Lighting and shadows
    Lighting,
    /// Color harmony
    Color,
    /// Object boundaries
    Edge,
    /// General purpose (always active)
    General,
}

impl HeadCategory {
    /// Default importance for this category (0.0 - 1.0)
    pub fn default_importance(&self) -> f32 {
        match self {
            HeadCategory::General => 1.0,
            HeadCategory::Composition => 0.9,
            HeadCategory::Style => 0.8,
            HeadCategory::Detail => 0.7,
            HeadCategory::Face => 0.6,
            HeadCategory::Body => 0.6,
            HeadCategory::Background => 0.5,
            HeadCategory::Lighting => 0.5,
            HeadCategory::Color => 0.4,
            HeadCategory::Edge => 0.4,
        }
    }

    /// Whether this category should always be active
    pub fn is_mandatory(&self) -> bool {
        matches!(self, HeadCategory::General | HeadCategory::Composition)
    }

    /// All categories
    pub fn all() -> &'static [HeadCategory] {
        &[
            HeadCategory::Face,
            HeadCategory::Body,
            HeadCategory::Background,
            HeadCategory::Style,
            HeadCategory::Composition,
            HeadCategory::Detail,
            HeadCategory::Lighting,
            HeadCategory::Color,
            HeadCategory::Edge,
            HeadCategory::General,
        ]
    }
}

/// Categories of prompts that influence head activation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PromptCategory {
    /// Portrait or character-focused
    Portrait,
    /// Landscape or scene
    Landscape,
    /// Abstract or artistic
    Abstract,
    /// Photorealistic
    Photorealistic,
    /// Anime/cartoon style
    Anime,
    /// Architecture or interiors
    Architecture,
    /// Objects or products
    Object,
    /// Fantasy or sci-fi
    Fantasy,
    /// Unknown/mixed
    Mixed,
}

impl PromptCategory {
    /// Keywords that indicate this category
    pub fn keywords(&self) -> &'static [&'static str] {
        match self {
            PromptCategory::Portrait => &[
                "portrait", "face", "person", "woman", "man", "girl", "boy",
                "character", "headshot", "bust", "selfie",
            ],
            PromptCategory::Landscape => &[
                "landscape", "mountain", "forest", "ocean", "sky", "sunset",
                "sunrise", "nature", "scenery", "vista", "horizon",
            ],
            PromptCategory::Abstract => &[
                "abstract", "geometric", "pattern", "fractal", "surreal",
                "dreamlike", "psychedelic", "minimalist",
            ],
            PromptCategory::Photorealistic => &[
                "photo", "photograph", "realistic", "hyperrealistic",
                "photorealistic", "raw", "unedited", "natural",
            ],
            PromptCategory::Anime => &[
                "anime", "manga", "cartoon", "illustrated", "cel-shaded",
                "2d", "chibi", "kawaii",
            ],
            PromptCategory::Architecture => &[
                "building", "architecture", "interior", "room", "house",
                "skyscraper", "cathedral", "bridge", "structure",
            ],
            PromptCategory::Object => &[
                "product", "object", "item", "thing", "device", "tool",
                "furniture", "vehicle", "food",
            ],
            PromptCategory::Fantasy => &[
                "fantasy", "magical", "dragon", "wizard", "elf", "fairy",
                "mythical", "enchanted", "sci-fi", "futuristic", "cyberpunk",
            ],
            PromptCategory::Mixed => &[],
        }
    }

    /// Head category importance modifiers for this prompt category
    pub fn category_weights(&self) -> HashMap<HeadCategory, f32> {
        let mut weights = HashMap::new();

        match self {
            PromptCategory::Portrait => {
                weights.insert(HeadCategory::Face, 1.0);
                weights.insert(HeadCategory::Body, 0.8);
                weights.insert(HeadCategory::Background, 0.3);
                weights.insert(HeadCategory::Style, 0.7);
                weights.insert(HeadCategory::Detail, 0.9);
                weights.insert(HeadCategory::Lighting, 0.7);
            }
            PromptCategory::Landscape => {
                weights.insert(HeadCategory::Face, 0.1);
                weights.insert(HeadCategory::Body, 0.2);
                weights.insert(HeadCategory::Background, 1.0);
                weights.insert(HeadCategory::Composition, 1.0);
                weights.insert(HeadCategory::Lighting, 0.9);
                weights.insert(HeadCategory::Color, 0.8);
            }
            PromptCategory::Abstract => {
                weights.insert(HeadCategory::Face, 0.0);
                weights.insert(HeadCategory::Body, 0.0);
                weights.insert(HeadCategory::Style, 1.0);
                weights.insert(HeadCategory::Composition, 1.0);
                weights.insert(HeadCategory::Color, 0.9);
                weights.insert(HeadCategory::Edge, 0.7);
            }
            PromptCategory::Photorealistic => {
                weights.insert(HeadCategory::Detail, 1.0);
                weights.insert(HeadCategory::Lighting, 1.0);
                weights.insert(HeadCategory::Color, 0.9);
                weights.insert(HeadCategory::Edge, 0.8);
            }
            PromptCategory::Anime => {
                weights.insert(HeadCategory::Face, 0.9);
                weights.insert(HeadCategory::Style, 1.0);
                weights.insert(HeadCategory::Edge, 0.9);
                weights.insert(HeadCategory::Color, 0.8);
                weights.insert(HeadCategory::Detail, 0.5);
            }
            PromptCategory::Architecture => {
                weights.insert(HeadCategory::Face, 0.0);
                weights.insert(HeadCategory::Composition, 1.0);
                weights.insert(HeadCategory::Edge, 1.0);
                weights.insert(HeadCategory::Detail, 0.9);
                weights.insert(HeadCategory::Lighting, 0.8);
            }
            PromptCategory::Object => {
                weights.insert(HeadCategory::Face, 0.0);
                weights.insert(HeadCategory::Detail, 1.0);
                weights.insert(HeadCategory::Edge, 0.9);
                weights.insert(HeadCategory::Lighting, 0.8);
                weights.insert(HeadCategory::Color, 0.7);
            }
            PromptCategory::Fantasy => {
                weights.insert(HeadCategory::Style, 1.0);
                weights.insert(HeadCategory::Composition, 0.9);
                weights.insert(HeadCategory::Lighting, 0.9);
                weights.insert(HeadCategory::Color, 0.8);
            }
            PromptCategory::Mixed => {
                // Use default weights
            }
        }

        weights
    }

    /// Detect category from prompt text
    pub fn detect(prompt: &str) -> SmallVec<[PromptCategory; 3]> {
        let prompt_lower = prompt.to_lowercase();
        let mut scores: Vec<(PromptCategory, usize)> = Vec::new();

        for category in &[
            PromptCategory::Portrait,
            PromptCategory::Landscape,
            PromptCategory::Abstract,
            PromptCategory::Photorealistic,
            PromptCategory::Anime,
            PromptCategory::Architecture,
            PromptCategory::Object,
            PromptCategory::Fantasy,
        ] {
            let count = category
                .keywords()
                .iter()
                .filter(|kw| prompt_lower.contains(*kw))
                .count();

            if count > 0 {
                scores.push((*category, count));
            }
        }

        scores.sort_by(|a, b| b.1.cmp(&a.1));

        let mut result: SmallVec<[PromptCategory; 3]> = scores
            .into_iter()
            .take(3)
            .map(|(cat, _)| cat)
            .collect();

        if result.is_empty() {
            result.push(PromptCategory::Mixed);
        }

        result
    }
}

/// Mapping from head indices to categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMapping {
    /// Number of heads per layer
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Category for each head in each layer
    pub head_categories: Vec<Vec<HeadCategory>>,
    /// Model this mapping was trained for
    pub model_id: String,
}

impl CategoryMapping {
    /// Create default mapping for SDXL-like architecture
    pub fn sdxl_default() -> Self {
        let num_heads = 32;
        let num_layers = 70;

        // Default category assignment based on typical SDXL behavior
        let head_categories: Vec<Vec<HeadCategory>> = (0..num_layers)
            .map(|layer| {
                (0..num_heads)
                    .map(|head| Self::default_head_category(layer, head, num_layers, num_heads))
                    .collect()
            })
            .collect();

        Self {
            num_heads,
            num_layers,
            head_categories,
            model_id: "sdxl-base".into(),
        }
    }

    /// Default category based on layer and head position
    fn default_head_category(
        layer: usize,
        head: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> HeadCategory {
        let layer_fraction = layer as f32 / num_layers as f32;
        let head_fraction = head as f32 / num_heads as f32;

        // Early layers: composition and structure
        if layer_fraction < 0.2 {
            return match head % 4 {
                0 => HeadCategory::Composition,
                1 => HeadCategory::General,
                2 => HeadCategory::Edge,
                _ => HeadCategory::Style,
            };
        }

        // Middle layers: content-specific
        if layer_fraction < 0.7 {
            if head_fraction < 0.3 {
                return HeadCategory::Face;
            } else if head_fraction < 0.5 {
                return HeadCategory::Body;
            } else if head_fraction < 0.7 {
                return HeadCategory::Background;
            } else {
                return HeadCategory::Lighting;
            }
        }

        // Late layers: detail and refinement
        match head % 5 {
            0 => HeadCategory::Detail,
            1 => HeadCategory::Color,
            2 => HeadCategory::Edge,
            3 => HeadCategory::Style,
            _ => HeadCategory::General,
        }
    }

    /// Get category for a specific head
    pub fn get_category(&self, layer: usize, head: usize) -> Option<HeadCategory> {
        self.head_categories
            .get(layer)
            .and_then(|heads| heads.get(head))
            .copied()
    }

    /// Get all heads in a category for a layer
    pub fn heads_in_category(&self, layer: usize, category: HeadCategory) -> Vec<usize> {
        self.head_categories
            .get(layer)
            .map(|heads| {
                heads
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c == category)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_category_detection() {
        let portrait = PromptCategory::detect("A beautiful portrait of a young woman");
        assert_eq!(portrait[0], PromptCategory::Portrait);

        let landscape = PromptCategory::detect("Mountain landscape at sunset");
        assert_eq!(landscape[0], PromptCategory::Landscape);

        let anime = PromptCategory::detect("Anime girl with blue hair");
        assert!(anime.contains(&PromptCategory::Anime));

        // "xyz abc" has no keyword matches, so should return Mixed
        let mixed = PromptCategory::detect("xyz abc");
        assert_eq!(mixed[0], PromptCategory::Mixed);
    }

    #[test]
    fn test_category_weights() {
        let portrait_weights = PromptCategory::Portrait.category_weights();
        assert_eq!(portrait_weights.get(&HeadCategory::Face), Some(&1.0));
        assert!(portrait_weights.get(&HeadCategory::Background).unwrap() < &0.5);

        let landscape_weights = PromptCategory::Landscape.category_weights();
        assert_eq!(landscape_weights.get(&HeadCategory::Background), Some(&1.0));
        assert!(landscape_weights.get(&HeadCategory::Face).unwrap() < &0.2);
    }

    #[test]
    fn test_sdxl_mapping() {
        let mapping = CategoryMapping::sdxl_default();
        assert_eq!(mapping.num_heads, 32);
        assert_eq!(mapping.num_layers, 70);

        // Should have a category for every head
        for layer in &mapping.head_categories {
            assert_eq!(layer.len(), 32);
        }
    }
}

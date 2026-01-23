//! Attention masks for sparse computation

use crate::{CategoryMapping, HeadCategory, Result, SparseError, MIN_ACTIVE_HEADS};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A mask indicating which attention heads to compute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMask {
    /// Number of heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Mask values: true = compute, false = skip
    /// Indexed as [layer][head]
    pub mask: Vec<Vec<bool>>,
    /// Per-layer sparsity (fraction of heads skipped)
    pub layer_sparsity: Vec<f32>,
    /// Overall sparsity
    pub overall_sparsity: f32,
}

impl AttentionMask {
    /// Create a mask with all heads active
    pub fn all_active(num_heads: usize, num_layers: usize) -> Self {
        let mask = vec![vec![true; num_heads]; num_layers];
        Self {
            num_heads,
            num_layers,
            mask,
            layer_sparsity: vec![0.0; num_layers],
            overall_sparsity: 0.0,
        }
    }

    /// Create a mask with uniform random sparsity
    pub fn random(num_heads: usize, num_layers: usize, sparsity: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mask: Vec<Vec<bool>> = (0..num_layers)
            .map(|_| {
                (0..num_heads)
                    .map(|_| rng.gen::<f32>() > sparsity)
                    .collect()
            })
            .collect();

        Self::from_mask(mask)
    }

    /// Create from raw mask data
    pub fn from_mask(mask: Vec<Vec<bool>>) -> Self {
        let num_layers = mask.len();
        let num_heads = mask.first().map(|l| l.len()).unwrap_or(0);

        let layer_sparsity: Vec<f32> = mask
            .iter()
            .map(|layer| {
                let inactive = layer.iter().filter(|&&active| !active).count();
                inactive as f32 / layer.len() as f32
            })
            .collect();

        let total_heads = num_heads * num_layers;
        let total_inactive: usize = mask
            .iter()
            .flat_map(|layer| layer.iter())
            .filter(|&&active| !active)
            .count();
        let overall_sparsity = total_inactive as f32 / total_heads as f32;

        Self {
            num_heads,
            num_layers,
            mask,
            layer_sparsity,
            overall_sparsity,
        }
    }

    /// Check if a head is active
    pub fn is_active(&self, layer: usize, head: usize) -> bool {
        self.mask
            .get(layer)
            .and_then(|l| l.get(head))
            .copied()
            .unwrap_or(true)
    }

    /// Set head activity
    pub fn set_active(&mut self, layer: usize, head: usize, active: bool) {
        if let Some(l) = self.mask.get_mut(layer) {
            if let Some(h) = l.get_mut(head) {
                *h = active;
            }
        }
        self.update_sparsity();
    }

    /// Get active head indices for a layer
    pub fn active_heads(&self, layer: usize) -> Vec<usize> {
        self.mask
            .get(layer)
            .map(|l| {
                l.iter()
                    .enumerate()
                    .filter(|(_, &active)| active)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get inactive head indices for a layer
    pub fn inactive_heads(&self, layer: usize) -> Vec<usize> {
        self.mask
            .get(layer)
            .map(|l| {
                l.iter()
                    .enumerate()
                    .filter(|(_, &active)| !active)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Number of active heads in a layer
    pub fn active_count(&self, layer: usize) -> usize {
        self.mask
            .get(layer)
            .map(|l| l.iter().filter(|&&a| a).count())
            .unwrap_or(0)
    }

    /// Update sparsity metrics
    fn update_sparsity(&mut self) {
        self.layer_sparsity = self
            .mask
            .iter()
            .map(|layer| {
                let inactive = layer.iter().filter(|&&active| !active).count();
                inactive as f32 / layer.len() as f32
            })
            .collect();

        let total_heads = self.num_heads * self.num_layers;
        let total_inactive: usize = self
            .mask
            .iter()
            .flat_map(|layer| layer.iter())
            .filter(|&&active| !active)
            .count();
        self.overall_sparsity = total_inactive as f32 / total_heads as f32;
    }

    /// Merge with another mask (AND operation - both must be active)
    pub fn merge_and(&self, other: &AttentionMask) -> Result<Self> {
        if self.num_heads != other.num_heads || self.num_layers != other.num_layers {
            return Err(SparseError::InvalidDimensions {
                expected_heads: self.num_heads,
                expected_layers: self.num_layers,
                actual_heads: other.num_heads,
                actual_layers: other.num_layers,
            });
        }

        let mask: Vec<Vec<bool>> = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(l1, l2)| l1.iter().zip(l2.iter()).map(|(&a, &b)| a && b).collect())
            .collect();

        Ok(Self::from_mask(mask))
    }

    /// Merge with another mask (OR operation - either can be active)
    pub fn merge_or(&self, other: &AttentionMask) -> Result<Self> {
        if self.num_heads != other.num_heads || self.num_layers != other.num_layers {
            return Err(SparseError::InvalidDimensions {
                expected_heads: self.num_heads,
                expected_layers: self.num_layers,
                actual_heads: other.num_heads,
                actual_layers: other.num_layers,
            });
        }

        let mask: Vec<Vec<bool>> = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(l1, l2)| l1.iter().zip(l2.iter()).map(|(&a, &b)| a || b).collect())
            .collect();

        Ok(Self::from_mask(mask))
    }

    /// Ensure minimum active heads per layer
    pub fn ensure_minimum(&mut self, min_active: usize) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        for layer_mask in &mut self.mask {
            let active_count = layer_mask.iter().filter(|&&a| a).count();
            if active_count < min_active {
                // Randomly activate more heads
                let mut inactive: Vec<usize> = layer_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &a)| !a)
                    .map(|(i, _)| i)
                    .collect();
                inactive.shuffle(&mut rng);

                for &idx in inactive.iter().take(min_active - active_count) {
                    layer_mask[idx] = true;
                }
            }
        }

        self.update_sparsity();
    }

    /// Convert to compact byte representation
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: num_heads (u16), num_layers (u16)
        bytes.extend_from_slice(&(self.num_heads as u16).to_le_bytes());
        bytes.extend_from_slice(&(self.num_layers as u16).to_le_bytes());

        // Bit-packed mask
        for layer in &self.mask {
            for chunk in layer.chunks(8) {
                let mut byte = 0u8;
                for (i, &active) in chunk.iter().enumerate() {
                    if active {
                        byte |= 1 << i;
                    }
                }
                bytes.push(byte);
            }
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(SparseError::InvalidDimensions {
                expected_heads: 0,
                expected_layers: 0,
                actual_heads: 0,
                actual_layers: 0,
            });
        }

        let num_heads = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let num_layers = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;

        let bytes_per_layer = num_heads.div_ceil(8);
        let mut mask = Vec::with_capacity(num_layers);

        let mut offset = 4;
        for _ in 0..num_layers {
            let mut layer = Vec::with_capacity(num_heads);
            for byte_idx in 0..bytes_per_layer {
                if offset + byte_idx >= bytes.len() {
                    break;
                }
                let byte = bytes[offset + byte_idx];
                for bit in 0..8 {
                    if layer.len() < num_heads {
                        layer.push((byte >> bit) & 1 == 1);
                    }
                }
            }
            mask.push(layer);
            offset += bytes_per_layer;
        }

        Ok(Self::from_mask(mask))
    }
}

impl Default for AttentionMask {
    /// Creates an empty attention mask with no heads or layers.
    ///
    /// Use `AttentionMask::all_active(num_heads, num_layers)` to create
    /// a mask with specific dimensions where all heads are active.
    fn default() -> Self {
        Self {
            num_heads: 0,
            num_layers: 0,
            mask: Vec::new(),
            layer_sparsity: Vec::new(),
            overall_sparsity: 0.0,
        }
    }
}

/// Pattern for mask generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskPattern {
    /// Skip every Nth head
    Strided { stride: usize },
    /// Skip first N heads per layer
    SkipFirst { count: usize },
    /// Skip last N heads per layer
    SkipLast { count: usize },
    /// Skip heads below importance threshold
    Threshold,
    /// Use category-based pruning
    CategoryBased,
}

/// Builder for attention masks
#[derive(Debug, Clone)]
pub struct MaskBuilder {
    num_heads: usize,
    num_layers: usize,
    pattern: MaskPattern,
    target_sparsity: f32,
    min_active: usize,
    category_weights: Option<HashMap<HeadCategory, f32>>,
    category_mapping: Option<CategoryMapping>,
}

impl MaskBuilder {
    /// Create a new builder
    pub fn new(num_heads: usize, num_layers: usize) -> Self {
        Self {
            num_heads,
            num_layers,
            pattern: MaskPattern::CategoryBased,
            target_sparsity: 0.5,
            min_active: MIN_ACTIVE_HEADS,
            category_weights: None,
            category_mapping: None,
        }
    }

    /// Set target sparsity
    pub fn sparsity(mut self, sparsity: f32) -> Self {
        self.target_sparsity = sparsity.clamp(0.0, 0.9);
        self
    }

    /// Set mask pattern
    pub fn pattern(mut self, pattern: MaskPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set minimum active heads
    pub fn min_active(mut self, min: usize) -> Self {
        self.min_active = min;
        self
    }

    /// Set category weights for category-based masking
    pub fn category_weights(mut self, weights: HashMap<HeadCategory, f32>) -> Self {
        self.category_weights = Some(weights);
        self
    }

    /// Set category mapping
    pub fn category_mapping(mut self, mapping: CategoryMapping) -> Self {
        self.category_mapping = Some(mapping);
        self
    }

    /// Build the mask
    pub fn build(self) -> AttentionMask {
        let mut mask = match self.pattern {
            MaskPattern::Strided { stride } => self.build_strided(stride),
            MaskPattern::SkipFirst { count } => self.build_skip_first(count),
            MaskPattern::SkipLast { count } => self.build_skip_last(count),
            MaskPattern::Threshold => self.build_threshold(),
            MaskPattern::CategoryBased => self.build_category_based(),
        };

        mask.ensure_minimum(self.min_active);
        mask
    }

    fn build_strided(&self, stride: usize) -> AttentionMask {
        let mask: Vec<Vec<bool>> = (0..self.num_layers)
            .map(|_| (0..self.num_heads).map(|head| head % stride != 0).collect())
            .collect();
        AttentionMask::from_mask(mask)
    }

    fn build_skip_first(&self, count: usize) -> AttentionMask {
        let mask: Vec<Vec<bool>> = (0..self.num_layers)
            .map(|_| (0..self.num_heads).map(|head| head >= count).collect())
            .collect();
        AttentionMask::from_mask(mask)
    }

    fn build_skip_last(&self, count: usize) -> AttentionMask {
        let mask: Vec<Vec<bool>> = (0..self.num_layers)
            .map(|_| {
                (0..self.num_heads)
                    .map(|head| head < self.num_heads - count)
                    .collect()
            })
            .collect();
        AttentionMask::from_mask(mask)
    }

    fn build_threshold(&self) -> AttentionMask {
        // Use uniform random threshold matching target sparsity
        AttentionMask::random(self.num_heads, self.num_layers, self.target_sparsity)
    }

    fn build_category_based(&self) -> AttentionMask {
        let mapping = self
            .category_mapping
            .clone()
            .unwrap_or_else(CategoryMapping::sdxl_default);
        let weights = self.category_weights.clone().unwrap_or_default();

        let heads_to_skip = (self.num_heads as f32 * self.target_sparsity) as usize;

        let mask: Vec<Vec<bool>> = (0..self.num_layers)
            .map(|layer| {
                // Score each head by its category weight
                let mut head_scores: Vec<(usize, f32)> = (0..self.num_heads)
                    .map(|head| {
                        let category = mapping
                            .get_category(layer, head)
                            .unwrap_or(HeadCategory::General);
                        let weight = weights
                            .get(&category)
                            .copied()
                            .unwrap_or_else(|| category.default_importance());
                        (head, weight)
                    })
                    .collect();

                // Sort by weight (lowest first - these will be skipped)
                head_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Create mask
                let skip_heads: std::collections::HashSet<usize> = head_scores
                    .iter()
                    .take(heads_to_skip)
                    .map(|(head, _)| *head)
                    .collect();

                (0..self.num_heads)
                    .map(|head| !skip_heads.contains(&head))
                    .collect()
            })
            .collect();

        AttentionMask::from_mask(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let mask = AttentionMask::default();

        assert_eq!(mask.num_heads, 0);
        assert_eq!(mask.num_layers, 0);
        assert!(mask.mask.is_empty());
        assert!(mask.layer_sparsity.is_empty());
        assert_eq!(mask.overall_sparsity, 0.0);
    }

    #[test]
    fn test_all_active() {
        let mask = AttentionMask::all_active(32, 10);
        assert_eq!(mask.overall_sparsity, 0.0);
        assert_eq!(mask.active_count(0), 32);
    }

    #[test]
    fn test_strided_mask() {
        let mask = MaskBuilder::new(32, 10)
            .pattern(MaskPattern::Strided { stride: 2 })
            .min_active(0)
            .build();

        // Every other head should be skipped
        for layer in 0..10 {
            assert!(!mask.is_active(layer, 0)); // Skipped
            assert!(mask.is_active(layer, 1)); // Active
            assert!(!mask.is_active(layer, 2)); // Skipped
        }
    }

    #[test]
    fn test_minimum_active() {
        let mut mask = AttentionMask::from_mask(vec![vec![false; 32]; 10]);
        mask.ensure_minimum(8);

        for layer in 0..10 {
            assert!(mask.active_count(layer) >= 8);
        }
    }

    #[test]
    fn test_byte_serialization() {
        let original = AttentionMask::random(32, 10, 0.5);
        let bytes = original.to_bytes();
        let restored = AttentionMask::from_bytes(&bytes).unwrap();

        assert_eq!(original.num_heads, restored.num_heads);
        assert_eq!(original.num_layers, restored.num_layers);
        for layer in 0..10 {
            for head in 0..32 {
                assert_eq!(
                    original.is_active(layer, head),
                    restored.is_active(layer, head)
                );
            }
        }
    }

    #[test]
    fn test_merge() {
        let mask1 = MaskBuilder::new(32, 10)
            .pattern(MaskPattern::SkipFirst { count: 8 })
            .min_active(0)
            .build();
        let mask2 = MaskBuilder::new(32, 10)
            .pattern(MaskPattern::SkipLast { count: 8 })
            .min_active(0)
            .build();

        // AND: only middle 16 heads active
        let merged_and = mask1.merge_and(&mask2).unwrap();
        assert_eq!(merged_and.active_count(0), 16);

        // OR: all but 0 heads active (first 8 or last 8 cover everything... wait no)
        // Actually: SkipFirst skips 0-7, SkipLast skips 24-31
        // OR means: active if either mask says so
        // mask1 active: 8-31, mask2 active: 0-23
        // union: 0-31 = all 32
        let merged_or = mask1.merge_or(&mask2).unwrap();
        assert_eq!(merged_or.active_count(0), 32);
    }
}

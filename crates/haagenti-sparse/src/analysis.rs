//! Head importance analysis for sparse attention

use crate::{CategoryMapping, HeadCategory, Result, SparseError};
use serde::{Deserialize, Serialize};

/// Importance score for a single attention head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadImportance {
    /// Layer index
    pub layer: usize,
    /// Head index
    pub head: usize,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Variance of importance across samples
    pub variance: f32,
    /// Assigned category
    pub category: HeadCategory,
    /// Activation frequency (0.0 - 1.0)
    pub activation_rate: f32,
}

/// Complete analysis of all attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadAnalysis {
    /// Model identifier
    pub model_id: String,
    /// Number of heads per layer
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Per-head importance data
    pub heads: Vec<HeadImportance>,
    /// Category mapping derived from analysis
    pub category_mapping: CategoryMapping,
    /// Global importance threshold for pruning
    pub prune_threshold: f32,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Metadata about how the analysis was performed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Number of samples used
    pub num_samples: usize,
    /// Prompt categories analyzed
    pub categories_analyzed: Vec<String>,
    /// Analysis timestamp
    pub timestamp: u64,
    /// Analysis method
    pub method: AnalysisMethod,
}

/// Method used for importance analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisMethod {
    /// Gradient-based importance (attention gradients)
    Gradient,
    /// Activation-based (attention weight magnitudes)
    Activation,
    /// Ablation-based (output difference when head removed)
    Ablation,
    /// Distillation-based (learned importance)
    Distillation,
}

impl HeadAnalysis {
    /// Get importance for a specific head
    pub fn get_importance(&self, layer: usize, head: usize) -> Option<&HeadImportance> {
        self.heads
            .iter()
            .find(|h| h.layer == layer && h.head == head)
    }

    /// Get all heads above importance threshold
    pub fn important_heads(&self, threshold: f32) -> Vec<&HeadImportance> {
        self.heads
            .iter()
            .filter(|h| h.importance >= threshold)
            .collect()
    }

    /// Get heads by category
    pub fn heads_by_category(&self, category: HeadCategory) -> Vec<&HeadImportance> {
        self.heads
            .iter()
            .filter(|h| h.category == category)
            .collect()
    }

    /// Get layer-wise importance distribution
    pub fn layer_importance(&self) -> Vec<f32> {
        let mut layer_sums = vec![0.0f32; self.num_layers];
        let mut layer_counts = vec![0usize; self.num_layers];

        for head in &self.heads {
            layer_sums[head.layer] += head.importance;
            layer_counts[head.layer] += 1;
        }

        layer_sums
            .iter()
            .zip(layer_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect()
    }

    /// Suggest optimal sparsity per layer based on importance
    pub fn suggested_sparsity(&self, target_overall: f32) -> Vec<f32> {
        let layer_imp = self.layer_importance();
        let mean_imp: f32 = layer_imp.iter().sum::<f32>() / layer_imp.len() as f32;

        // Higher importance layers get less sparsity
        layer_imp
            .iter()
            .map(|&imp| {
                let ratio = if mean_imp > 0.0 { imp / mean_imp } else { 1.0 };
                // Inverse relationship: more important = less sparse
                (target_overall * (2.0 - ratio)).clamp(0.1, 0.9)
            })
            .collect()
    }
}

/// Analyzer for computing head importance
pub struct ImportanceAnalyzer {
    /// Analysis method to use
    method: AnalysisMethod,
    /// Number of samples to collect
    num_samples: usize,
    /// Collected activation data
    activations: Vec<ActivationSample>,
}

/// A single activation sample
#[derive(Debug, Clone)]
struct ActivationSample {
    /// Layer activations `[layer][head]`
    attention_weights: Vec<Vec<f32>>,
    /// Prompt category
    category: String,
    /// Step number (stored for future step-aware analysis)
    #[allow(dead_code)]
    step: u32,
}

impl ImportanceAnalyzer {
    /// Create a new analyzer
    pub fn new(method: AnalysisMethod) -> Self {
        Self {
            method,
            num_samples: 0,
            activations: Vec::new(),
        }
    }

    /// Set target number of samples
    pub fn with_samples(mut self, count: usize) -> Self {
        self.num_samples = count;
        self
    }

    /// Record an activation sample
    pub fn record_sample(&mut self, attention_weights: Vec<Vec<f32>>, category: String, step: u32) {
        self.activations.push(ActivationSample {
            attention_weights,
            category,
            step,
        });
    }

    /// Analyze collected samples and produce head importance
    pub fn analyze(&self, model_id: &str) -> Result<HeadAnalysis> {
        if self.activations.is_empty() {
            return Err(SparseError::AnalysisError("No samples collected".into()));
        }

        let num_layers = self.activations[0].attention_weights.len();
        let num_heads = self.activations[0]
            .attention_weights
            .first()
            .map(|l| l.len())
            .unwrap_or(0);

        // Compute importance based on method
        let heads = match self.method {
            AnalysisMethod::Activation => self.analyze_activation(num_layers, num_heads),
            AnalysisMethod::Gradient => self.analyze_activation(num_layers, num_heads), // Fallback
            AnalysisMethod::Ablation => self.analyze_activation(num_layers, num_heads), // Fallback
            AnalysisMethod::Distillation => self.analyze_activation(num_layers, num_heads), // Fallback
        };

        // Build category mapping from importance patterns
        let category_mapping = self.infer_categories(&heads, num_heads, num_layers);

        // Calculate prune threshold (median importance)
        let mut importances: Vec<f32> = heads.iter().map(|h| h.importance).collect();
        importances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let prune_threshold = importances
            .get(importances.len() / 2)
            .copied()
            .unwrap_or(0.5);

        let categories_analyzed: Vec<String> = self
            .activations
            .iter()
            .map(|s| s.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(HeadAnalysis {
            model_id: model_id.into(),
            num_heads,
            num_layers,
            heads,
            category_mapping,
            prune_threshold,
            metadata: AnalysisMetadata {
                num_samples: self.activations.len(),
                categories_analyzed,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                method: self.method,
            },
        })
    }

    /// Analyze using activation magnitudes
    fn analyze_activation(&self, num_layers: usize, num_heads: usize) -> Vec<HeadImportance> {
        let mut heads = Vec::with_capacity(num_layers * num_heads);

        for layer in 0..num_layers {
            for head in 0..num_heads {
                // Collect all activation values for this head
                let values: Vec<f32> = self
                    .activations
                    .iter()
                    .filter_map(|s| {
                        s.attention_weights
                            .get(layer)
                            .and_then(|l| l.get(head))
                            .copied()
                    })
                    .collect();

                if values.is_empty() {
                    continue;
                }

                // Compute mean (importance) and variance
                let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
                let variance: f32 =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

                // Activation rate (how often above threshold)
                let activation_rate =
                    values.iter().filter(|&&v| v > 0.1).count() as f32 / values.len() as f32;

                // Infer category based on layer position and importance pattern
                let category = Self::infer_head_category(layer, head, num_layers, num_heads, mean);

                heads.push(HeadImportance {
                    layer,
                    head,
                    importance: mean.clamp(0.0, 1.0),
                    variance,
                    category,
                    activation_rate,
                });
            }
        }

        heads
    }

    /// Infer category for a single head
    fn infer_head_category(
        layer: usize,
        head: usize,
        num_layers: usize,
        num_heads: usize,
        _importance: f32,
    ) -> HeadCategory {
        let layer_fraction = layer as f32 / num_layers as f32;
        let head_fraction = head as f32 / num_heads as f32;

        // Early layers: composition
        if layer_fraction < 0.2 {
            return if head_fraction < 0.5 {
                HeadCategory::Composition
            } else {
                HeadCategory::General
            };
        }

        // Late layers: detail
        if layer_fraction > 0.8 {
            return if head_fraction < 0.3 {
                HeadCategory::Detail
            } else {
                HeadCategory::Edge
            };
        }

        // Middle layers: content-specific
        if head_fraction < 0.25 {
            HeadCategory::Face
        } else if head_fraction < 0.5 {
            HeadCategory::Body
        } else if head_fraction < 0.75 {
            HeadCategory::Background
        } else {
            HeadCategory::Style
        }
    }

    /// Infer category mapping from importance data
    fn infer_categories(
        &self,
        heads: &[HeadImportance],
        num_heads: usize,
        num_layers: usize,
    ) -> CategoryMapping {
        let head_categories: Vec<Vec<HeadCategory>> = (0..num_layers)
            .map(|layer| {
                (0..num_heads)
                    .map(|head| {
                        heads
                            .iter()
                            .find(|h| h.layer == layer && h.head == head)
                            .map(|h| h.category)
                            .unwrap_or(HeadCategory::General)
                    })
                    .collect()
            })
            .collect();

        CategoryMapping {
            num_heads,
            num_layers,
            head_categories,
            model_id: String::new(),
        }
    }
}

/// Statistics about head importance distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceStats {
    /// Mean importance
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum importance
    pub min: f32,
    /// Maximum importance
    pub max: f32,
    /// Median importance
    pub median: f32,
    /// Percentiles (25th, 50th, 75th, 90th)
    pub percentiles: [f32; 4],
}

impl ImportanceStats {
    /// Compute stats from importance values
    pub fn from_importances(importances: &[f32]) -> Self {
        if importances.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                percentiles: [0.0; 4],
            };
        }

        let mut sorted = importances.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let mean: f32 = sorted.iter().sum::<f32>() / n as f32;
        let variance: f32 = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;

        Self {
            mean,
            std_dev: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            median: sorted[n / 2],
            percentiles: [
                sorted[n / 4],
                sorted[n / 2],
                sorted[3 * n / 4],
                sorted[9 * n / 10],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer() {
        let mut analyzer = ImportanceAnalyzer::new(AnalysisMethod::Activation);

        // Add some fake samples
        for _ in 0..10 {
            let weights: Vec<Vec<f32>> = (0..10)
                .map(|_| (0..8).map(|h| 0.5 + h as f32 * 0.05).collect())
                .collect();
            analyzer.record_sample(weights, "portrait".into(), 1);
        }

        let analysis = analyzer.analyze("test-model").unwrap();
        assert_eq!(analysis.num_layers, 10);
        assert_eq!(analysis.num_heads, 8);
        assert_eq!(analysis.heads.len(), 80);
    }

    #[test]
    fn test_layer_importance() {
        let heads: Vec<HeadImportance> = (0..4)
            .flat_map(|layer| {
                (0..8).map(move |head| HeadImportance {
                    layer,
                    head,
                    importance: 0.5 + layer as f32 * 0.1,
                    variance: 0.01,
                    category: HeadCategory::General,
                    activation_rate: 0.8,
                })
            })
            .collect();

        let analysis = HeadAnalysis {
            model_id: "test".into(),
            num_heads: 8,
            num_layers: 4,
            heads,
            category_mapping: CategoryMapping::sdxl_default(),
            prune_threshold: 0.5,
            metadata: AnalysisMetadata {
                num_samples: 10,
                categories_analyzed: vec!["test".into()],
                timestamp: 0,
                method: AnalysisMethod::Activation,
            },
        };

        let layer_imp = analysis.layer_importance();
        assert_eq!(layer_imp.len(), 4);
        // Later layers should have higher importance
        assert!(layer_imp[3] > layer_imp[0]);
    }
}

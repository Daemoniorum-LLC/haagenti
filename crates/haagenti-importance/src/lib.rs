//! ML-Guided Fragment Importance Scoring
//!
//! This module implements intelligent fragment prioritization using:
//!
//! - **Prompt Analysis**: Extract semantic features to predict attention patterns
//! - **Historical Learning**: Track actual fragment usage across generations
//! - **Quality Sensitivity**: Learn which layers tolerate compression
//! - **Adaptive Scoring**: Adjust importance based on generation step
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                     Importance Scorer                               │
//! ├────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
//! │  │     Prompt      │    │   Historical    │    │    Quality      │ │
//! │  │    Analyzer     │ -> │    Tracker      │ -> │   Predictor     │ │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
//! │           │                     │                      │           │
//! │           ↓                     ↓                      ↓           │
//! │  ┌────────────────────────────────────────────────────────────────┐│
//! │  │                    Importance Scores                           ││
//! │  │    fragment_id → (importance: f32, confidence: f32)            ││
//! │  └────────────────────────────────────────────────────────────────┘│
//! │                              ↓                                     │
//! │  ┌────────────────────────────────────────────────────────────────┐│
//! │  │                  Priority Queue                                ││
//! │  │    (integrates with haagenti-network)                          ││
//! │  └────────────────────────────────────────────────────────────────┘│
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Insights
//!
//! 1. **Early steps use coarse features**: High-noise denoising steps
//!    don't need precise weights, so load low-quality fragments first
//!
//! 2. **Attention patterns are predictable**: Given a prompt, certain
//!    attention heads are more active (e.g., "portrait" → face attention)
//!
//! 3. **Layer importance varies**: Some layers (VAE decoder) always need
//!    high quality, others (early UNet blocks) tolerate approximation
//!
//! 4. **Usage patterns repeat**: Similar prompts use similar fragment sets

mod analyzer;
mod error;
mod history;
mod predictor;
mod scorer;

pub use analyzer::{PromptAnalyzer, PromptFeatures, SemanticCategory};
pub use error::{ImportanceError, Result};
pub use history::{FragmentUsage, UsageHistory, UsageStats};
pub use predictor::{LayerProfile, QualityPredictor, QualitySensitivity};
pub use scorer::{AdaptiveScorer, ImportanceScore, ImportanceScorer, ScorerConfig};

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        ImportanceScore, ImportanceScorer, PromptAnalyzer, PromptFeatures, QualityPredictor,
        Result, UsageHistory,
    };
}

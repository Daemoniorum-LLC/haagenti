//! Sparse Attention Masks
//!
//! This module implements prompt-aware attention head masking to skip
//! computation for heads that don't contribute to the output.
//!
//! # Key Insight
//!
//! Not all attention heads are equally important for every prompt. Portrait
//! prompts activate face-focused heads while landscape prompts activate
//! background/composition heads. By predicting which heads matter, we can
//! skip 50-70% of attention computation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Sparse Attention                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Standard Attention (32 heads × 64 layers = 2048 computations)  │
//! │  ════════════════════════════════════════════════════════════   │
//! │  [████████████████████████████████] 100% compute                │
//! │                                                                  │
//! │  Sparse Attention (prompt-aware masking)                        │
//! │  ════════════════════════════════════════════════════════════   │
//! │  "Portrait of a woman"                                          │
//! │  [████████░░░░░░░░████░░░░░░░░░░░░] 35% compute                 │
//! │   ↑ face  ↑ skip  ↑ style                                       │
//! │                                                                  │
//! │  "Mountain landscape at sunset"                                 │
//! │  [░░░░░░░░████████████████████░░░░] 45% compute                 │
//! │   ↑ skip  ↑ background/lighting  ↑ skip                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod analysis;
mod categories;
mod error;
mod kernel;
mod mask;
mod predictor;

pub use analysis::{HeadAnalysis, HeadImportance, ImportanceAnalyzer, ImportanceStats};
pub use categories::{CategoryMapping, HeadCategory, PromptCategory};
pub use error::{Result, SparseError};
pub use kernel::{KernelConfig, KernelStats, SparseKernel};
pub use mask::{AttentionMask, MaskBuilder, MaskPattern};
pub use predictor::{MaskPredictor, Prediction, PredictorConfig};

/// Default sparsity target (fraction of heads to skip)
pub const DEFAULT_SPARSITY: f32 = 0.5;

/// Minimum heads to keep active per layer
pub const MIN_ACTIVE_HEADS: usize = 4;

/// Maximum quality degradation allowed
pub const MAX_QUALITY_LOSS: f32 = 0.02;

/// Prelude for common imports
pub mod prelude {
    pub use super::{AttentionMask, HeadCategory, MaskPredictor, PromptCategory, Result};
}

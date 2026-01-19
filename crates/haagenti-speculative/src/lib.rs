//! Speculative Fragment Loading
//!
//! This module implements intelligent prefetching based on:
//! - Keystroke intent prediction (trie-based + ML)
//! - Session history analysis
//! - Confidence-based loading thresholds
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 Speculative Loader                           │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Keystrokes ──> Intent Predictor ──> Fragment Pre-Warmer    │
//! │       │              │                      │                │
//! │       ↓              ↓                      ↓                │
//! │  "portr..."    "portrait"           Load face attention     │
//! │  "lands..."    "landscape"          Cancel, load background │
//! │                                                              │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │              Speculation Buffer                       │   │
//! │  │   Hot:  [face_attn] (90% confidence)                 │   │
//! │  │   Warm: [body_attn] (60% confidence)                 │   │
//! │  │   Cold: [cancelled fragments]                        │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod buffer;
mod error;
mod intent;
mod loader;
mod session;

pub use buffer::{SpeculationBuffer, BufferConfig, BufferEntry, BufferStats};
pub use error::{SpeculativeError, Result};
pub use intent::{IntentPredictor, Intent, IntentConfig, PredictionResult};
pub use loader::{SpeculativeLoader, LoaderConfig};
pub use session::{SessionHistory, SessionPattern, UserPreferences};

/// Default confidence threshold for starting speculation
pub const DEFAULT_SPECULATION_THRESHOLD: f32 = 0.6;

/// Default confidence threshold for committing to load
pub const DEFAULT_COMMIT_THRESHOLD: f32 = 0.8;

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        IntentPredictor, Intent, SpeculativeLoader, SpeculationBuffer,
        SessionHistory, Result,
    };
}

//! Adaptive Precision Scheduling
//!
//! This module implements dynamic precision adjustment during inference:
//! - INT4 for high-noise early steps (structure formation)
//! - INT8 for mid-generation (feature refinement)
//! - FP16 for final steps (detail preservation)
//!
//! # Key Insight
//!
//! Early denoising steps operate on highly noisy latents where
//! precision errors are masked by the noise itself. Later steps
//! require higher precision as the signal-to-noise ratio improves.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 Precision Schedule                           │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Noise:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     │
//! │          High ─────────────────────────────────────> Low    │
//! │                                                              │
//! │  Step:   1    5    10   15   20   25   30   35   40   50    │
//! │          │    │     │    │    │    │    │    │    │    │    │
//! │  Prec:  INT4 INT4  INT8 INT8 INT8 FP16 FP16 FP16 FP16 FP16  │
//! │          │    │     │    │    │    │    │    │    │    │    │
//! │  VRAM:  25%  25%   50%  50%  50%  100% 100% 100% 100% 100%  │
//! │          │    │     │    │    │    │    │    │    │    │    │
//! │  Speed:  4x   4x    2x   2x   2x   1x   1x   1x   1x   1x   │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod error;
mod precision;
mod profile;
mod schedule;
mod transition;

pub use error::{AdaptiveError, Result};
pub use precision::{Precision, PrecisionCapabilities};
pub use profile::{PrecisionProfile, ProfilePreset};
pub use schedule::{PrecisionSchedule, ScheduleConfig, StepPrecision};
pub use transition::{PrecisionTransition, TransitionStrategy};

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        Precision, PrecisionProfile, PrecisionSchedule, ProfilePreset, Result,
    };
}

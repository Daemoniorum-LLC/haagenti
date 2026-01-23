// Test modules have minor lints that don't affect production code
#![cfg_attr(test, allow(clippy::useless_vec))]

//! Model merging techniques
//!
//! This crate provides advanced model merging capabilities:
//! - TIES (Trim, Elect, Merge) for task vector merging
//! - DARE (Drop And REscale) for efficient parameter pruning
//! - Evolutionary merging with genetic algorithms
//! - SLERP (Spherical Linear Interpolation) for smooth blending
//! - Linear and weighted averaging methods

mod dare;
mod error;
mod evolutionary;
mod linear;
mod slerp;
mod ties;
mod weights;

pub use dare::{DareConfig, DareMerger};
pub use error::{MergeError, Result};
pub use evolutionary::{EvolutionaryConfig, EvolutionaryMerger, Genome};
pub use linear::{LinearConfig, LinearMerger};
pub use slerp::{SlerpConfig, SlerpMerger};
pub use ties::{TiesConfig, TiesMerger};
pub use weights::{ModelWeights, WeightDelta, WeightTensor};

/// Merge strategy
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Simple linear interpolation
    Linear(LinearConfig),
    /// SLERP interpolation
    Slerp(SlerpConfig),
    /// TIES merging
    Ties(TiesConfig),
    /// DARE merging
    Dare(DareConfig),
    /// Evolutionary merging
    Evolutionary(EvolutionaryConfig),
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::Linear(LinearConfig::default())
    }
}

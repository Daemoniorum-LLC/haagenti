// Test modules have minor lints that don't affect production code
#![cfg_attr(test, allow(clippy::manual_range_contains))]

//! Self-optimization and auto-tuning
//!
//! This crate provides automatic optimization capabilities:
//! - Bayesian optimization for hyperparameter tuning
//! - Genetic algorithms for architecture search
//! - Runtime profiling and bottleneck detection
//! - Hardware-aware optimization
//! - Automatic batch size tuning
//! - Memory usage optimization

mod bayesian;
mod error;
mod genetic;
mod hardware;
mod profiler;
mod tuner;

pub use bayesian::{AcquisitionFunction, BayesianConfig, BayesianOptimizer};
pub use error::{OptError, Result};
pub use genetic::{GeneticConfig, GeneticSearch, SearchSpace};
pub use hardware::{DeviceCapability, HardwareOptimizer, HardwareProfile, ProfileDatabase};
pub use profiler::{Bottleneck, ProfileResult, Profiler, ScopedTimer};
pub use tuner::{presets, AutoTuner, TunerConfig, TuningResult};

/// Optimization strategy
#[derive(Debug, Clone)]
pub enum OptStrategy {
    /// Bayesian optimization
    Bayesian(BayesianConfig),
    /// Genetic algorithm
    Genetic(GeneticConfig),
    /// Grid search
    GridSearch,
    /// Random search
    RandomSearch { n_trials: usize },
    /// Hardware-aware
    HardwareAware,
}

impl Default for OptStrategy {
    fn default() -> Self {
        Self::Bayesian(BayesianConfig::default())
    }
}

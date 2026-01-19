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

pub use bayesian::{BayesianOptimizer, BayesianConfig, AcquisitionFunction};
pub use error::{OptError, Result};
pub use genetic::{GeneticSearch, GeneticConfig, SearchSpace};
pub use hardware::{HardwareProfile, HardwareOptimizer, DeviceCapability};
pub use profiler::{Profiler, ProfileResult, Bottleneck};
pub use tuner::{AutoTuner, TunerConfig, TuningResult};

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

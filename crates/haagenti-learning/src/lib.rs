//! Continuous learning and online adaptation
//!
//! This crate provides continuous learning capabilities:
//! - LoRA adapter training and merging
//! - Experience replay buffer for online learning
//! - Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
//! - Progressive layer unfreezing
//! - Learning rate scheduling
//! - Gradient accumulation and checkpointing

mod adapter;
mod buffer;
mod consolidation;
mod error;
mod scheduler;
mod trainer;

pub use adapter::{AdapterRegistry, LoraAdapter, LoraConfig};
pub use buffer::{BufferConfig, Experience, ReplayBuffer, ReservoirBuffer};
pub use consolidation::{EwcConfig, EwcRegularizer, FisherInfo, SynapticIntelligence};
pub use error::{LearningError, Result};
pub use scheduler::{LearningRateScheduler, ParamGroupScheduler, SchedulerConfig, WarmupScheduler};
pub use trainer::{OnlineTrainer, TrainerConfig, TrainingStats};

/// Learning strategies
#[derive(Debug, Clone)]
pub enum LearningStrategy {
    /// Full fine-tuning
    FullFineTune,
    /// LoRA adaptation
    Lora(LoraConfig),
    /// Progressive unfreezing
    Progressive { layers_per_epoch: usize },
    /// Elastic Weight Consolidation
    Ewc(EwcConfig),
}

impl Default for LearningStrategy {
    fn default() -> Self {
        Self::Lora(LoraConfig::default())
    }
}

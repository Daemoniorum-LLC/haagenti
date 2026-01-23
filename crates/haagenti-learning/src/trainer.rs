//! Online trainer for continuous learning

use crate::{
    adapter::AdapterRegistry,
    buffer::{BufferConfig, Experience, ReplayBuffer},
    consolidation::EwcRegularizer,
    scheduler::{LearningRateScheduler, SchedulerConfig},
    LearningError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Trainer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Learning rate scheduler config
    pub scheduler: SchedulerConfig,
    /// Replay buffer config
    pub buffer: BufferConfig,
    /// Gradient accumulation steps
    pub gradient_accumulation: usize,
    /// Gradient clipping (max norm)
    pub gradient_clip: Option<f32>,
    /// Save checkpoint every N steps
    pub checkpoint_interval: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            buffer: BufferConfig::default(),
            gradient_accumulation: 1,
            gradient_clip: Some(1.0),
            checkpoint_interval: 1000,
            log_interval: 100,
            mixed_precision: false,
            early_stopping_patience: Some(10),
        }
    }
}

/// Training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Total steps
    pub total_steps: usize,
    /// Total samples processed
    pub total_samples: usize,
    /// Current epoch
    pub epoch: usize,
    /// Running loss
    pub running_loss: f32,
    /// Best loss seen
    pub best_loss: f32,
    /// Steps since improvement
    pub steps_without_improvement: usize,
    /// Training time (seconds)
    pub training_time_secs: f64,
    /// Samples per second
    pub samples_per_sec: f32,
    /// Current learning rate
    pub current_lr: f32,
}

impl TrainingStats {
    /// Update with new loss
    pub fn update(&mut self, loss: f32, lr: f32, samples: usize) {
        self.total_steps += 1;
        self.total_samples += samples;
        self.current_lr = lr;

        // Exponential moving average of loss
        if self.running_loss == 0.0 {
            self.running_loss = loss;
        } else {
            self.running_loss = 0.99 * self.running_loss + 0.01 * loss;
        }

        if loss < self.best_loss {
            self.best_loss = loss;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }
    }
}

/// Online trainer for continuous learning
pub struct OnlineTrainer {
    /// Configuration
    config: TrainerConfig,
    /// Learning rate scheduler
    scheduler: LearningRateScheduler,
    /// Experience replay buffer
    buffer: ReplayBuffer,
    /// Adapter registry
    adapters: AdapterRegistry,
    /// EWC regularizer (optional)
    ewc: Option<EwcRegularizer>,
    /// Training statistics
    stats: TrainingStats,
    /// Accumulated gradients
    accumulated_gradients: HashMap<String, Vec<f32>>,
    /// Accumulation step counter
    accumulation_step: usize,
    /// Training start time
    start_time: Option<Instant>,
}

impl OnlineTrainer {
    /// Create new online trainer
    pub fn new(config: TrainerConfig) -> Self {
        let scheduler = LearningRateScheduler::new(config.scheduler.clone());
        let buffer = ReplayBuffer::new(config.buffer.clone());

        Self {
            config,
            scheduler,
            buffer,
            adapters: AdapterRegistry::new(),
            ewc: None,
            stats: TrainingStats {
                best_loss: f32::MAX,
                ..Default::default()
            },
            accumulated_gradients: HashMap::new(),
            accumulation_step: 0,
            start_time: None,
        }
    }

    /// Set EWC regularizer
    pub fn with_ewc(&mut self, ewc: EwcRegularizer) {
        self.ewc = Some(ewc);
    }

    /// Get adapter registry
    pub fn adapters(&self) -> &AdapterRegistry {
        &self.adapters
    }

    /// Get mutable adapter registry
    pub fn adapters_mut(&mut self) -> &mut AdapterRegistry {
        &mut self.adapters
    }

    /// Add experience to buffer
    pub fn add_experience(&mut self, experience: Experience) {
        self.buffer.add(experience);
    }

    /// Train one step
    pub fn step<F>(&mut self, forward_backward: F) -> Result<f32>
    where
        F: Fn(&[&Experience], f32) -> Result<(f32, HashMap<String, Vec<f32>>)>,
    {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Sample from buffer and extract data before releasing borrow
        let (experiences, sample_indices): (Vec<Experience>, Vec<usize>) = {
            let samples = self.buffer.sample();
            if samples.is_empty() {
                return Err(LearningError::TrainingError("Buffer empty".into()));
            }
            // Clone experiences and collect indices to release the borrow
            let experiences: Vec<Experience> =
                samples.iter().map(|(_, e, _)| (*e).clone()).collect();
            let indices: Vec<usize> = samples.iter().map(|(i, _, _)| *i).collect();
            (experiences, indices)
        };

        let experience_refs: Vec<&Experience> = experiences.iter().collect();
        let lr = self.scheduler.get_lr();
        let num_experiences = experiences.len();

        // Forward and backward pass
        let (loss, gradients) = forward_backward(&experience_refs, lr)?;

        // Accumulate gradients
        self.accumulate_gradients(gradients);
        self.accumulation_step += 1;

        // Apply gradients after accumulation
        if self.accumulation_step >= self.config.gradient_accumulation {
            self.apply_gradients()?;
            self.accumulation_step = 0;
            self.scheduler.step();
        }

        // Update priorities for prioritized replay
        if self.config.buffer.prioritized {
            let losses = vec![loss; sample_indices.len()];
            self.buffer.update_priorities(&sample_indices, &losses);
        }

        // Update stats
        self.stats.update(loss, lr, num_experiences);

        if let Some(start) = self.start_time {
            self.stats.training_time_secs = start.elapsed().as_secs_f64();
            self.stats.samples_per_sec =
                self.stats.total_samples as f32 / self.stats.training_time_secs as f32;
        }

        Ok(loss)
    }

    /// Accumulate gradients
    fn accumulate_gradients(&mut self, gradients: HashMap<String, Vec<f32>>) {
        for (name, grad) in gradients {
            let acc = self
                .accumulated_gradients
                .entry(name)
                .or_insert_with(|| vec![0.0; grad.len()]);

            for (a, g) in acc.iter_mut().zip(&grad) {
                *a += g / self.config.gradient_accumulation as f32;
            }
        }
    }

    /// Apply accumulated gradients
    fn apply_gradients(&mut self) -> Result<()> {
        // Clip gradients if configured
        if let Some(max_norm) = self.config.gradient_clip {
            self.clip_gradients(max_norm);
        }

        // Add EWC gradient contribution
        if let Some(ref _ewc) = self.ewc {
            // Would need current params to compute EWC gradient
            // For now, just a placeholder
        }

        // Apply gradients to adapters
        let lr = self.scheduler.get_lr();
        for (name, grad) in &self.accumulated_gradients {
            if let Some(adapter) = self.adapters.get_mut(name) {
                // Update adapter weights
                let mut a = adapter.get_a().to_vec();
                let mut b = adapter.get_b().to_vec();

                // Simple SGD update (in practice would be Adam/AdamW)
                for (w, g) in a.iter_mut().chain(b.iter_mut()).zip(grad) {
                    *w -= lr * g;
                }

                adapter.set_a(a)?;
                adapter.set_b(b)?;
            }
        }

        self.accumulated_gradients.clear();
        Ok(())
    }

    /// Clip gradients by global norm
    fn clip_gradients(&mut self, max_norm: f32) {
        // Calculate global norm
        let mut total_norm_sq = 0.0f32;
        for grad in self.accumulated_gradients.values() {
            total_norm_sq += grad.iter().map(|g| g * g).sum::<f32>();
        }
        let total_norm = total_norm_sq.sqrt();

        // Clip if necessary
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for grad in self.accumulated_gradients.values_mut() {
                for g in grad.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    /// Check early stopping
    pub fn should_stop(&self) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            self.stats.steps_without_improvement >= patience
        } else {
            false
        }
    }

    /// Get training statistics
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }

    /// Save checkpoint
    pub fn checkpoint(&self) -> TrainerCheckpoint {
        TrainerCheckpoint {
            stats: self.stats.clone(),
            scheduler_step: self.scheduler.current_step(),
            // Would include adapter weights, buffer state, etc.
        }
    }

    /// Reset trainer
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.buffer.clear();
        self.stats = TrainingStats {
            best_loss: f32::MAX,
            ..Default::default()
        };
        self.accumulated_gradients.clear();
        self.accumulation_step = 0;
        self.start_time = None;
    }
}

impl std::fmt::Debug for OnlineTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineTrainer")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Trainer checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerCheckpoint {
    /// Training stats
    pub stats: TrainingStats,
    /// Scheduler step
    pub scheduler_step: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::default();
        let trainer = OnlineTrainer::new(config);

        assert_eq!(trainer.stats().total_steps, 0);
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats {
            best_loss: f32::MAX,
            ..Default::default()
        };

        stats.update(1.0, 0.001, 32);
        assert_eq!(stats.total_steps, 1);
        assert_eq!(stats.total_samples, 32);
        assert_eq!(stats.best_loss, 1.0);

        stats.update(0.5, 0.001, 32);
        assert_eq!(stats.best_loss, 0.5);
        assert_eq!(stats.steps_without_improvement, 0);

        stats.update(0.6, 0.001, 32);
        assert_eq!(stats.steps_without_improvement, 1);
    }

    #[test]
    fn test_early_stopping() {
        let config = TrainerConfig {
            early_stopping_patience: Some(3),
            ..Default::default()
        };
        let mut trainer = OnlineTrainer::new(config);

        trainer.stats.steps_without_improvement = 2;
        assert!(!trainer.should_stop());

        trainer.stats.steps_without_improvement = 3;
        assert!(trainer.should_stop());
    }

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();

        assert_eq!(config.gradient_accumulation, 1);
        assert_eq!(config.gradient_clip, Some(1.0));
        assert_eq!(config.checkpoint_interval, 1000);
        assert_eq!(config.log_interval, 100);
        assert!(!config.mixed_precision);
        assert_eq!(config.early_stopping_patience, Some(10));
    }

    #[test]
    fn test_trainer_checkpoint() {
        let config = TrainerConfig::default();
        let mut trainer = OnlineTrainer::new(config);

        // Simulate some training
        trainer.stats.total_steps = 100;
        trainer.stats.total_samples = 3200;
        trainer.stats.running_loss = 0.5;

        let checkpoint = trainer.checkpoint();

        assert_eq!(checkpoint.stats.total_steps, 100);
        assert_eq!(checkpoint.stats.total_samples, 3200);
        assert_eq!(checkpoint.stats.running_loss, 0.5);
    }

    #[test]
    fn test_trainer_reset() {
        let config = TrainerConfig::default();
        let mut trainer = OnlineTrainer::new(config);

        // Simulate some training
        trainer.stats.total_steps = 100;
        trainer.stats.total_samples = 3200;
        trainer.stats.running_loss = 0.5;
        trainer.stats.steps_without_improvement = 5;

        // Reset
        trainer.reset();

        assert_eq!(trainer.stats().total_steps, 0);
        assert_eq!(trainer.stats().total_samples, 0);
        assert_eq!(trainer.stats().running_loss, 0.0);
        assert_eq!(trainer.stats().steps_without_improvement, 0);
        assert_eq!(trainer.stats().best_loss, f32::MAX);
    }

    #[test]
    fn test_training_stats_running_loss_ema() {
        let mut stats = TrainingStats {
            best_loss: f32::MAX,
            ..Default::default()
        };

        // First update - running loss should be set directly
        stats.update(1.0, 0.001, 32);
        assert_eq!(stats.running_loss, 1.0);

        // Second update - should use EMA
        stats.update(0.5, 0.001, 32);
        // EMA: 0.99 * 1.0 + 0.01 * 0.5 = 0.995
        assert!((stats.running_loss - 0.995).abs() < 0.001);
    }

    #[test]
    fn test_trainer_get_lr() {
        let config = TrainerConfig::default();
        let trainer = OnlineTrainer::new(config);

        // Initial learning rate should be from scheduler default
        let lr = trainer.get_lr();
        assert!(lr > 0.0);
    }

    #[test]
    fn test_early_stopping_disabled() {
        let config = TrainerConfig {
            early_stopping_patience: None,
            ..Default::default()
        };
        let mut trainer = OnlineTrainer::new(config);

        // Even with high steps without improvement, shouldn't stop
        trainer.stats.steps_without_improvement = 1000;
        assert!(!trainer.should_stop());
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();

        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.epoch, 0);
        assert_eq!(stats.running_loss, 0.0);
        assert_eq!(stats.best_loss, 0.0);
        assert_eq!(stats.steps_without_improvement, 0);
    }

    #[test]
    fn test_trainer_adapters_access() {
        let config = TrainerConfig::default();
        let mut trainer = OnlineTrainer::new(config);

        // Should be able to access adapters
        let adapters = trainer.adapters();
        assert!(adapters.list().is_empty());

        // Should be able to get mutable access
        let _adapters_mut = trainer.adapters_mut();
    }
}

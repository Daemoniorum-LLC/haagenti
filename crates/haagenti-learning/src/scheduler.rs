//! Learning rate scheduling

use serde::{Deserialize, Serialize};

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Final learning rate
    pub final_lr: f32,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Total training steps
    pub total_steps: usize,
    /// Scheduler type
    pub scheduler_type: SchedulerType,
}

/// Scheduler type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,
    /// Linear decay
    Linear,
    /// Cosine annealing
    Cosine,
    /// Exponential decay
    Exponential,
    /// Step decay
    Step,
    /// Warmup + cosine
    WarmupCosine,
    /// One cycle
    OneCycle,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            initial_lr: 1e-4,
            final_lr: 1e-6,
            warmup_steps: 100,
            total_steps: 10000,
            scheduler_type: SchedulerType::WarmupCosine,
        }
    }
}

/// Learning rate scheduler
#[derive(Debug)]
pub struct LearningRateScheduler {
    /// Configuration
    config: SchedulerConfig,
    /// Current step
    current_step: usize,
    /// Current learning rate
    current_lr: f32,
}

impl LearningRateScheduler {
    /// Create new scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let initial_lr = if config.warmup_steps > 0 {
            config.final_lr // Start low for warmup
        } else {
            config.initial_lr
        };

        Self {
            config,
            current_step: 0,
            current_lr: initial_lr,
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.current_lr = self.compute_lr(self.current_step);
        self.current_lr
    }

    /// Compute learning rate for a given step
    fn compute_lr(&self, step: usize) -> f32 {
        let config = &self.config;

        // Handle warmup
        if step < config.warmup_steps {
            let warmup_progress = step as f32 / config.warmup_steps as f32;
            return config.final_lr + (config.initial_lr - config.final_lr) * warmup_progress;
        }

        let post_warmup_step = step - config.warmup_steps;
        let post_warmup_total = config.total_steps.saturating_sub(config.warmup_steps);

        if post_warmup_total == 0 {
            return config.initial_lr;
        }

        let progress = (post_warmup_step as f32 / post_warmup_total as f32).min(1.0);

        match config.scheduler_type {
            SchedulerType::Constant => config.initial_lr,
            SchedulerType::Linear => {
                config.initial_lr + (config.final_lr - config.initial_lr) * progress
            }
            SchedulerType::Cosine | SchedulerType::WarmupCosine => {
                let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                config.final_lr + (config.initial_lr - config.final_lr) * cosine_decay
            }
            SchedulerType::Exponential => {
                let decay_rate = (config.final_lr / config.initial_lr).ln();
                config.initial_lr * (decay_rate * progress).exp()
            }
            SchedulerType::Step => {
                // Decay by 0.1 every 30% of training
                let num_decays = (progress / 0.3) as u32;
                config.initial_lr * 0.1f32.powi(num_decays as i32)
            }
            SchedulerType::OneCycle => {
                // 1cycle: warmup to peak, then decay
                if progress < 0.4 {
                    // Increase to peak
                    let phase_progress = progress / 0.4;
                    config.initial_lr + (config.initial_lr * 10.0 - config.initial_lr) * phase_progress
                } else {
                    // Decrease from peak
                    let phase_progress = (progress - 0.4) / 0.6;
                    let peak = config.initial_lr * 10.0;
                    peak + (config.final_lr - peak) * phase_progress
                }
            }
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = if self.config.warmup_steps > 0 {
            self.config.final_lr
        } else {
            self.config.initial_lr
        };
    }

    /// Check if training is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.config.total_steps
    }
}

/// Warmup scheduler helper
#[derive(Debug)]
pub struct WarmupScheduler {
    /// Base scheduler
    base_lr: f32,
    /// Warmup steps
    warmup_steps: usize,
    /// Current step
    current_step: usize,
}

impl WarmupScheduler {
    /// Create new warmup scheduler
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
        }
    }

    /// Get learning rate
    pub fn get_lr(&self) -> f32 {
        if self.current_step >= self.warmup_steps {
            self.base_lr
        } else {
            self.base_lr * (self.current_step as f32 / self.warmup_steps as f32)
        }
    }

    /// Step the scheduler
    pub fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.get_lr()
    }

    /// Is warmup complete
    pub fn is_warmed_up(&self) -> bool {
        self.current_step >= self.warmup_steps
    }
}

/// Group-specific learning rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamGroupScheduler {
    /// Group name to multiplier
    pub groups: std::collections::HashMap<String, f32>,
    /// Base scheduler config
    pub base_config: SchedulerConfig,
}

impl ParamGroupScheduler {
    /// Create new group scheduler
    pub fn new(base_config: SchedulerConfig) -> Self {
        Self {
            groups: std::collections::HashMap::new(),
            base_config,
        }
    }

    /// Add parameter group with multiplier
    pub fn add_group(&mut self, name: impl Into<String>, multiplier: f32) {
        self.groups.insert(name.into(), multiplier);
    }

    /// Get learning rate for group
    pub fn get_lr(&self, group: &str, base_lr: f32) -> f32 {
        let multiplier = self.groups.get(group).copied().unwrap_or(1.0);
        base_lr * multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let config = SchedulerConfig {
            initial_lr: 0.001,
            warmup_steps: 0,
            scheduler_type: SchedulerType::Constant,
            ..Default::default()
        };

        let mut scheduler = LearningRateScheduler::new(config);

        for _ in 0..100 {
            let lr = scheduler.step();
            assert_eq!(lr, 0.001);
        }
    }

    #[test]
    fn test_warmup() {
        let config = SchedulerConfig {
            initial_lr: 0.001,
            final_lr: 0.0,
            warmup_steps: 10,
            total_steps: 100,
            scheduler_type: SchedulerType::Constant,
        };

        let mut scheduler = LearningRateScheduler::new(config);

        // During warmup, LR should increase
        let lr_0 = scheduler.get_lr();
        scheduler.step();
        scheduler.step();
        scheduler.step();
        let lr_3 = scheduler.get_lr();

        assert!(lr_3 > lr_0);
    }

    #[test]
    fn test_linear_decay() {
        let config = SchedulerConfig {
            initial_lr: 1.0,
            final_lr: 0.0,
            warmup_steps: 0,
            total_steps: 100,
            scheduler_type: SchedulerType::Linear,
        };

        let mut scheduler = LearningRateScheduler::new(config);

        for _ in 0..50 {
            scheduler.step();
        }

        // Should be around 0.5 at midpoint
        assert!((scheduler.get_lr() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_cosine_annealing() {
        let config = SchedulerConfig {
            initial_lr: 1.0,
            final_lr: 0.0,
            warmup_steps: 0,
            total_steps: 100,
            scheduler_type: SchedulerType::Cosine,
        };

        let mut scheduler = LearningRateScheduler::new(config);

        let lr_start = scheduler.get_lr();
        for _ in 0..100 {
            scheduler.step();
        }
        let lr_end = scheduler.get_lr();

        assert!(lr_start > lr_end);
        assert!(lr_end < 0.1);
    }

    #[test]
    fn test_warmup_scheduler() {
        let mut scheduler = WarmupScheduler::new(0.001, 10);

        assert!(!scheduler.is_warmed_up());

        for _ in 0..10 {
            scheduler.step();
        }

        assert!(scheduler.is_warmed_up());
        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_param_groups() {
        let config = SchedulerConfig::default();
        let mut groups = ParamGroupScheduler::new(config);

        groups.add_group("encoder", 0.1);
        groups.add_group("decoder", 1.0);

        let base_lr = 0.001;
        assert_eq!(groups.get_lr("encoder", base_lr), 0.0001);
        assert_eq!(groups.get_lr("decoder", base_lr), 0.001);
    }
}

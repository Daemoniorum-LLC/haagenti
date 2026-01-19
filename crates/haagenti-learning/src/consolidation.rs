//! Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention

use crate::{LearningError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// EWC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwcConfig {
    /// Lambda (importance weight)
    pub lambda: f32,
    /// Number of samples for Fisher estimation
    pub fisher_samples: usize,
    /// Damping factor for numerical stability
    pub damping: f32,
    /// Online EWC (running average of Fisher)
    pub online: bool,
    /// Decay factor for online EWC
    pub gamma: f32,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 100.0,
            fisher_samples: 200,
            damping: 1e-3,
            online: false,
            gamma: 0.9,
        }
    }
}

/// Fisher information for a parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherInfo {
    /// Parameter name
    pub name: String,
    /// Fisher diagonal (importance weights)
    pub fisher: Vec<f32>,
    /// Optimal parameter values (from previous task)
    pub optimal_params: Vec<f32>,
    /// Shape
    pub shape: Vec<usize>,
}

impl FisherInfo {
    /// Create new Fisher info
    pub fn new(name: impl Into<String>, params: Vec<f32>, shape: Vec<usize>) -> Self {
        let n = params.len();
        Self {
            name: name.into(),
            fisher: vec![0.0; n],
            optimal_params: params,
            shape,
        }
    }

    /// Update Fisher with gradient sample
    pub fn update_fisher(&mut self, gradient: &[f32]) {
        if gradient.len() != self.fisher.len() {
            return;
        }

        for (f, g) in self.fisher.iter_mut().zip(gradient) {
            *f += g * g;
        }
    }

    /// Normalize Fisher by sample count
    pub fn normalize(&mut self, sample_count: usize) {
        if sample_count > 0 {
            let scale = 1.0 / sample_count as f32;
            for f in &mut self.fisher {
                *f *= scale;
            }
        }
    }

    /// Compute EWC penalty
    pub fn penalty(&self, current_params: &[f32], lambda: f32) -> f32 {
        if current_params.len() != self.optimal_params.len() {
            return 0.0;
        }

        let mut penalty = 0.0;
        for ((f, opt), curr) in self.fisher.iter().zip(&self.optimal_params).zip(current_params) {
            let diff = curr - opt;
            penalty += f * diff * diff;
        }

        0.5 * lambda * penalty
    }

    /// Compute EWC gradient contribution
    pub fn gradient(&self, current_params: &[f32], lambda: f32) -> Vec<f32> {
        current_params
            .iter()
            .zip(&self.optimal_params)
            .zip(&self.fisher)
            .map(|((curr, opt), f)| lambda * f * (curr - opt))
            .collect()
    }
}

/// EWC regularizer
#[derive(Debug)]
pub struct EwcRegularizer {
    /// Configuration
    config: EwcConfig,
    /// Fisher information per parameter
    fisher_info: HashMap<String, FisherInfo>,
    /// Task count
    task_count: usize,
}

impl EwcRegularizer {
    /// Create new EWC regularizer
    pub fn new(config: EwcConfig) -> Self {
        Self {
            config,
            fisher_info: HashMap::new(),
            task_count: 0,
        }
    }

    /// Register parameters for a new task
    pub fn register_task(&mut self, params: HashMap<String, (Vec<f32>, Vec<usize>)>) {
        for (name, (values, shape)) in params {
            let fisher = FisherInfo::new(&name, values, shape);
            self.fisher_info.insert(name, fisher);
        }
        self.task_count += 1;
    }

    /// Update Fisher information with a gradient sample
    pub fn update_fisher(&mut self, gradients: &HashMap<String, Vec<f32>>) {
        for (name, gradient) in gradients {
            if let Some(fisher) = self.fisher_info.get_mut(name) {
                fisher.update_fisher(gradient);
            }
        }
    }

    /// Finalize Fisher estimation after sampling
    pub fn finalize_fisher(&mut self) {
        for fisher in self.fisher_info.values_mut() {
            fisher.normalize(self.config.fisher_samples);

            // Add damping
            for f in &mut fisher.fisher {
                *f += self.config.damping;
            }
        }
    }

    /// Compute total EWC penalty
    pub fn compute_penalty(&self, current_params: &HashMap<String, Vec<f32>>) -> f32 {
        let mut total_penalty = 0.0;

        for (name, params) in current_params {
            if let Some(fisher) = self.fisher_info.get(name) {
                total_penalty += fisher.penalty(params, self.config.lambda);
            }
        }

        total_penalty
    }

    /// Compute EWC gradient
    pub fn compute_gradient(&self, current_params: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
        let mut gradients = HashMap::new();

        for (name, params) in current_params {
            if let Some(fisher) = self.fisher_info.get(name) {
                gradients.insert(name.clone(), fisher.gradient(params, self.config.lambda));
            }
        }

        gradients
    }

    /// Online EWC update (merge new Fisher with existing)
    pub fn online_update(&mut self, new_fisher: HashMap<String, FisherInfo>) {
        let gamma = self.config.gamma;

        for (name, new) in new_fisher {
            if let Some(existing) = self.fisher_info.get_mut(&name) {
                // Weighted combination of old and new Fisher
                for (old_f, new_f) in existing.fisher.iter_mut().zip(&new.fisher) {
                    *old_f = gamma * *old_f + (1.0 - gamma) * new_f;
                }
                // Update optimal params to current
                existing.optimal_params = new.optimal_params;
            } else {
                self.fisher_info.insert(name, new);
            }
        }
    }

    /// Get Fisher info for a parameter
    pub fn get_fisher(&self, name: &str) -> Option<&FisherInfo> {
        self.fisher_info.get(name)
    }

    /// Get all parameter names
    pub fn param_names(&self) -> Vec<&str> {
        self.fisher_info.keys().map(|s| s.as_str()).collect()
    }

    /// Task count
    pub fn task_count(&self) -> usize {
        self.task_count
    }

    /// Total parameters tracked
    pub fn total_params(&self) -> usize {
        self.fisher_info.values().map(|f| f.fisher.len()).sum()
    }
}

/// Synaptic Intelligence (SI) - alternative to EWC
#[derive(Debug)]
pub struct SynapticIntelligence {
    /// Importance weights
    omega: HashMap<String, Vec<f32>>,
    /// Running importance
    running_omega: HashMap<String, Vec<f32>>,
    /// Previous parameters
    prev_params: HashMap<String, Vec<f32>>,
    /// Damping factor
    damping: f32,
    /// Regularization strength
    c: f32,
}

impl SynapticIntelligence {
    /// Create new SI regularizer
    pub fn new(c: f32, damping: f32) -> Self {
        Self {
            omega: HashMap::new(),
            running_omega: HashMap::new(),
            prev_params: HashMap::new(),
            damping,
            c,
        }
    }

    /// Initialize for parameters
    pub fn init(&mut self, params: &HashMap<String, Vec<f32>>) {
        for (name, values) in params {
            let n = values.len();
            self.omega.insert(name.clone(), vec![0.0; n]);
            self.running_omega.insert(name.clone(), vec![0.0; n]);
            self.prev_params.insert(name.clone(), values.clone());
        }
    }

    /// Update during training
    pub fn update_importance(
        &mut self,
        params: &HashMap<String, Vec<f32>>,
        gradients: &HashMap<String, Vec<f32>>,
    ) {
        for (name, grad) in gradients {
            if let (Some(running), Some(prev)) = (
                self.running_omega.get_mut(name),
                self.prev_params.get(name),
            ) {
                if let Some(curr) = params.get(name) {
                    for (((r, g), p), c) in running
                        .iter_mut()
                        .zip(grad)
                        .zip(prev)
                        .zip(curr)
                    {
                        *r += -g * (c - p);
                    }
                }
            }
        }
    }

    /// Consolidate importance after task
    pub fn consolidate(&mut self, params: &HashMap<String, Vec<f32>>) {
        for (name, omega) in &mut self.omega {
            if let (Some(running), Some(prev), Some(curr)) = (
                self.running_omega.get(name),
                self.prev_params.get(name),
                params.get(name),
            ) {
                for (((o, r), p), c) in omega
                    .iter_mut()
                    .zip(running)
                    .zip(prev)
                    .zip(curr)
                {
                    let delta = (c - p).abs() + self.damping;
                    *o += r / delta;
                }
            }

            // Reset running omega
            if let Some(running) = self.running_omega.get_mut(name) {
                running.fill(0.0);
            }
        }

        // Update prev_params
        self.prev_params = params.clone();
    }

    /// Compute SI penalty
    pub fn penalty(&self, current_params: &HashMap<String, Vec<f32>>) -> f32 {
        let mut total = 0.0;

        for (name, omega) in &self.omega {
            if let (Some(prev), Some(curr)) = (
                self.prev_params.get(name),
                current_params.get(name),
            ) {
                for ((o, p), c) in omega.iter().zip(prev).zip(curr) {
                    let diff = c - p;
                    total += o * diff * diff;
                }
            }
        }

        self.c * total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_config() {
        let config = EwcConfig::default();
        assert_eq!(config.lambda, 100.0);
    }

    #[test]
    fn test_fisher_info() {
        let mut fisher = FisherInfo::new("layer", vec![1.0, 2.0, 3.0], vec![3]);

        fisher.update_fisher(&[1.0, 2.0, 3.0]);
        assert_eq!(fisher.fisher, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_fisher_penalty() {
        let fisher = FisherInfo {
            name: "layer".into(),
            fisher: vec![1.0, 1.0],
            optimal_params: vec![0.0, 0.0],
            shape: vec![2],
        };

        let current = vec![1.0, 1.0];
        let penalty = fisher.penalty(&current, 1.0);

        // penalty = 0.5 * 1.0 * (1.0*1.0 + 1.0*1.0) = 1.0
        assert_eq!(penalty, 1.0);
    }

    #[test]
    fn test_ewc_regularizer() {
        let config = EwcConfig::default();
        let mut ewc = EwcRegularizer::new(config);

        let mut params = HashMap::new();
        params.insert("layer1".into(), (vec![0.0, 0.0], vec![2]));
        ewc.register_task(params);

        assert_eq!(ewc.task_count(), 1);
        assert!(ewc.get_fisher("layer1").is_some());
    }

    #[test]
    fn test_synaptic_intelligence() {
        let mut si = SynapticIntelligence::new(0.1, 1e-3);

        let mut params = HashMap::new();
        params.insert("layer".into(), vec![0.0, 0.0]);

        si.init(&params);

        let mut gradients = HashMap::new();
        gradients.insert("layer".into(), vec![1.0, 2.0]);

        params.insert("layer".into(), vec![0.1, 0.2]);
        si.update_importance(&params, &gradients);
        si.consolidate(&params);

        let penalty = si.penalty(&params);
        assert!(penalty >= 0.0);
    }
}

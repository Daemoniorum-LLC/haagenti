//! Bayesian optimization for hyperparameter tuning

use crate::{OptError, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bayesian optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Initial random samples
    pub initial_samples: usize,
    /// Acquisition function
    pub acquisition: AcquisitionFunction,
    /// Exploration parameter (kappa for UCB, xi for EI)
    pub exploration: f32,
    /// Random seed
    pub seed: Option<u64>,
    /// Early stopping threshold
    pub early_stopping_threshold: Option<f32>,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            initial_samples: 10,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            exploration: 0.01,
            seed: None,
            early_stopping_threshold: None,
        }
    }
}

/// Acquisition function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound,
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Thompson Sampling
    ThompsonSampling,
}

/// Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
}

/// Parameter type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Continuous in range [low, high]
    Continuous { low: f32, high: f32 },
    /// Integer in range [low, high]
    Integer { low: i32, high: i32 },
    /// Categorical choices
    Categorical { choices: Vec<String> },
    /// Log-scale continuous
    LogUniform { low: f32, high: f32 },
}

impl Parameter {
    /// Create continuous parameter
    pub fn continuous(name: impl Into<String>, low: f32, high: f32) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Continuous { low, high },
        }
    }

    /// Create integer parameter
    pub fn integer(name: impl Into<String>, low: i32, high: i32) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Integer { low, high },
        }
    }

    /// Create categorical parameter
    pub fn categorical(name: impl Into<String>, choices: Vec<String>) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::Categorical { choices },
        }
    }

    /// Create log-uniform parameter
    pub fn log_uniform(name: impl Into<String>, low: f32, high: f32) -> Self {
        Self {
            name: name.into(),
            param_type: ParameterType::LogUniform { low, high },
        }
    }

    /// Sample random value
    pub fn sample(&self, rng: &mut StdRng) -> ParameterValue {
        match &self.param_type {
            ParameterType::Continuous { low, high } => {
                ParameterValue::Float(rng.gen::<f32>() * (high - low) + low)
            }
            ParameterType::Integer { low, high } => {
                ParameterValue::Int(rng.gen_range(*low..=*high))
            }
            ParameterType::Categorical { choices } => {
                let idx = rng.gen_range(0..choices.len());
                ParameterValue::String(choices[idx].clone())
            }
            ParameterType::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                let log_val = rng.gen::<f32>() * (log_high - log_low) + log_low;
                ParameterValue::Float(log_val.exp())
            }
        }
    }
}

/// Parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    Int(i32),
    String(String),
}

impl ParameterValue {
    /// Get as float
    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParameterValue::Float(f) => Some(*f),
            ParameterValue::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Get as int
    pub fn as_int(&self) -> Option<i32> {
        match self {
            ParameterValue::Int(i) => Some(*i),
            ParameterValue::Float(f) => Some(*f as i32),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParameterValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Observation from evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Parameter values
    pub params: HashMap<String, ParameterValue>,
    /// Objective value (higher is better)
    pub objective: f32,
    /// Evaluation time (ms)
    pub eval_time_ms: u64,
}

/// Bayesian optimizer
#[derive(Debug)]
pub struct BayesianOptimizer {
    /// Configuration
    config: BayesianConfig,
    /// Parameter definitions
    parameters: Vec<Parameter>,
    /// Observations
    observations: Vec<Observation>,
    /// Random number generator
    rng: StdRng,
    /// Best observation
    best: Option<Observation>,
    /// Current iteration
    iteration: usize,
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer
    pub fn new(config: BayesianConfig, parameters: Vec<Parameter>) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            config,
            parameters,
            observations: Vec::new(),
            rng,
            best: None,
            iteration: 0,
        }
    }

    /// Suggest next parameters to evaluate
    pub fn suggest(&mut self) -> HashMap<String, ParameterValue> {
        self.iteration += 1;

        // Initial random exploration
        if self.observations.len() < self.config.initial_samples {
            return self.random_sample();
        }

        // Use acquisition function to suggest
        self.acquisition_sample()
    }

    /// Random sample
    fn random_sample(&mut self) -> HashMap<String, ParameterValue> {
        self.parameters
            .iter()
            .map(|p| (p.name.clone(), p.sample(&mut self.rng)))
            .collect()
    }

    /// Sample using acquisition function
    fn acquisition_sample(&mut self) -> HashMap<String, ParameterValue> {
        // Simplified: generate candidates and pick best by acquisition
        let n_candidates = 100;
        let mut best_acquisition = f32::NEG_INFINITY;
        let mut best_params = self.random_sample();

        for _ in 0..n_candidates {
            let params = self.random_sample();
            let acq_value = self.compute_acquisition(&params);

            if acq_value > best_acquisition {
                best_acquisition = acq_value;
                best_params = params;
            }
        }

        best_params
    }

    /// Compute acquisition function value
    fn compute_acquisition(&self, params: &HashMap<String, ParameterValue>) -> f32 {
        // Simplified surrogate model: kernel-weighted average
        let (mean, std) = self.predict(params);

        let best_obj = self.best.as_ref().map(|b| b.objective).unwrap_or(0.0);

        match self.config.acquisition {
            AcquisitionFunction::ExpectedImprovement => {
                let z = if std > 1e-8 {
                    (mean - best_obj - self.config.exploration) / std
                } else {
                    0.0
                };
                // Simplified EI approximation
                std * (z * normal_cdf(z) + normal_pdf(z))
            }
            AcquisitionFunction::UpperConfidenceBound => {
                mean + self.config.exploration * std
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                let z = if std > 1e-8 {
                    (mean - best_obj - self.config.exploration) / std
                } else {
                    0.0
                };
                normal_cdf(z)
            }
            AcquisitionFunction::ThompsonSampling => {
                // Sample from posterior
                mean + self.rng.clone().gen::<f32>() * std
            }
        }
    }

    /// Predict mean and std for parameters (simplified GP)
    fn predict(&self, params: &HashMap<String, ParameterValue>) -> (f32, f32) {
        if self.observations.is_empty() {
            return (0.0, 1.0);
        }

        // Kernel-weighted prediction
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weighted_sq_sum = 0.0;

        for obs in &self.observations {
            let dist = self.param_distance(params, &obs.params);
            let weight = (-dist * 10.0).exp(); // RBF-like kernel

            weighted_sum += weight * obs.objective;
            weighted_sq_sum += weight * obs.objective * obs.objective;
            weight_sum += weight;
        }

        if weight_sum < 1e-8 {
            return (0.0, 1.0);
        }

        let mean = weighted_sum / weight_sum;
        let variance = (weighted_sq_sum / weight_sum - mean * mean).max(0.01);
        let std = variance.sqrt();

        (mean, std)
    }

    /// Compute distance between parameter sets
    fn param_distance(
        &self,
        a: &HashMap<String, ParameterValue>,
        b: &HashMap<String, ParameterValue>,
    ) -> f32 {
        let mut dist = 0.0;

        for param in &self.parameters {
            if let (Some(va), Some(vb)) = (a.get(&param.name), b.get(&param.name)) {
                let d = match (&param.param_type, va, vb) {
                    (ParameterType::Continuous { low, high }, ParameterValue::Float(fa), ParameterValue::Float(fb)) => {
                        ((fa - fb) / (high - low)).powi(2)
                    }
                    (ParameterType::Integer { low, high }, ParameterValue::Int(ia), ParameterValue::Int(ib)) => {
                        ((ia - ib) as f32 / (high - low) as f32).powi(2)
                    }
                    (ParameterType::Categorical { .. }, ParameterValue::String(sa), ParameterValue::String(sb)) => {
                        if sa == sb { 0.0 } else { 1.0 }
                    }
                    _ => 0.0,
                };
                dist += d;
            }
        }

        dist.sqrt()
    }

    /// Register observation
    pub fn observe(&mut self, observation: Observation) {
        if self.best.is_none() || observation.objective > self.best.as_ref().unwrap().objective {
            self.best = Some(observation.clone());
        }
        self.observations.push(observation);
    }

    /// Get best observation
    pub fn best(&self) -> Option<&Observation> {
        self.best.as_ref()
    }

    /// Current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Check if should stop
    pub fn should_stop(&self) -> bool {
        if self.iteration >= self.config.max_iterations {
            return true;
        }

        if let Some(threshold) = self.config.early_stopping_threshold {
            if let Some(ref best) = self.best {
                if best.objective >= threshold {
                    return true;
                }
            }
        }

        false
    }

    /// Get all observations
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + (x / std::f32::consts::SQRT_2).tanh())
}

/// Standard normal PDF
fn normal_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sampling() {
        let mut rng = StdRng::seed_from_u64(42);

        let param = Parameter::continuous("lr", 0.0, 1.0);
        let value = param.sample(&mut rng);

        if let ParameterValue::Float(f) = value {
            assert!(f >= 0.0 && f <= 1.0);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_bayesian_suggest() {
        let config = BayesianConfig {
            initial_samples: 2,
            seed: Some(42),
            ..Default::default()
        };

        let params = vec![
            Parameter::continuous("lr", 0.0001, 0.1),
            Parameter::integer("batch_size", 8, 64),
        ];

        let mut optimizer = BayesianOptimizer::new(config, params);

        // Should get random samples initially
        let suggestion = optimizer.suggest();
        assert!(suggestion.contains_key("lr"));
        assert!(suggestion.contains_key("batch_size"));
    }

    #[test]
    fn test_observation() {
        let config = BayesianConfig::default();
        let params = vec![Parameter::continuous("x", -1.0, 1.0)];

        let mut optimizer = BayesianOptimizer::new(config, params);

        let mut p = HashMap::new();
        p.insert("x".into(), ParameterValue::Float(0.5));

        optimizer.observe(Observation {
            params: p,
            objective: 0.9,
            eval_time_ms: 100,
        });

        assert!(optimizer.best().is_some());
        assert_eq!(optimizer.best().unwrap().objective, 0.9);
    }

    #[test]
    fn test_acquisition_functions() {
        let configs = vec![
            AcquisitionFunction::ExpectedImprovement,
            AcquisitionFunction::UpperConfidenceBound,
            AcquisitionFunction::ProbabilityOfImprovement,
        ];

        for acq in configs {
            let config = BayesianConfig {
                acquisition: acq,
                seed: Some(42),
                ..Default::default()
            };
            let params = vec![Parameter::continuous("x", 0.0, 1.0)];
            let mut optimizer = BayesianOptimizer::new(config, params);

            // Add some observations
            for i in 0..5 {
                let mut p = HashMap::new();
                p.insert("x".into(), ParameterValue::Float(i as f32 * 0.2));
                optimizer.observe(Observation {
                    params: p,
                    objective: (i as f32 * 0.1).sin(),
                    eval_time_ms: 10,
                });
            }

            // Should be able to suggest
            let _ = optimizer.suggest();
        }
    }
}

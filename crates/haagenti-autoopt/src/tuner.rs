//! Auto-tuner for inference optimization

use crate::{
    bayesian::{BayesianConfig, BayesianOptimizer, Observation, Parameter, ParameterValue},
    hardware::HardwareProfile,
    profiler::{Profiler, ProfileResult},
    OptError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Auto-tuner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Maximum tuning time
    pub max_time: Duration,
    /// Maximum trials
    pub max_trials: usize,
    /// Target metric (latency_ms, throughput_tps, memory_mb)
    pub target_metric: String,
    /// Minimize target (true) or maximize (false)
    pub minimize: bool,
    /// Constraints
    pub constraints: HashMap<String, f32>,
    /// Enable profiling during tuning
    pub profile: bool,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            max_time: Duration::from_secs(300),
            max_trials: 50,
            target_metric: "latency_ms".into(),
            minimize: true,
            constraints: HashMap::new(),
            profile: true,
        }
    }
}

/// Tuning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    /// Best parameters found
    pub best_params: HashMap<String, ParameterValue>,
    /// Best metric value
    pub best_value: f32,
    /// Number of trials
    pub trials: usize,
    /// Tuning duration
    pub duration_secs: f64,
    /// Improvement from baseline
    pub improvement_percent: f32,
    /// All trial results
    pub history: Vec<TrialResult>,
}

/// Single trial result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Trial number
    pub trial: usize,
    /// Parameters
    pub params: HashMap<String, ParameterValue>,
    /// Metric value
    pub value: f32,
    /// Evaluation time
    pub eval_time_ms: u64,
    /// Valid (meets constraints)
    pub valid: bool,
}

/// Auto-tuner
pub struct AutoTuner {
    /// Configuration
    config: TunerConfig,
    /// Parameters to tune
    parameters: Vec<Parameter>,
    /// Bayesian optimizer
    optimizer: BayesianOptimizer,
    /// Profiler
    profiler: Option<Profiler>,
    /// Trial history
    history: Vec<TrialResult>,
    /// Start time
    start_time: Option<Instant>,
    /// Baseline value
    baseline: Option<f32>,
}

impl AutoTuner {
    /// Create new auto-tuner
    pub fn new(config: TunerConfig, parameters: Vec<Parameter>) -> Self {
        let bayesian_config = BayesianConfig {
            max_iterations: config.max_trials,
            ..Default::default()
        };

        let optimizer = BayesianOptimizer::new(bayesian_config, parameters.clone());
        let profiler = if config.profile {
            Some(Profiler::new())
        } else {
            None
        };

        Self {
            config,
            parameters,
            optimizer,
            profiler,
            history: Vec::new(),
            start_time: None,
            baseline: None,
        }
    }

    /// Set baseline value
    pub fn set_baseline(&mut self, value: f32) {
        self.baseline = Some(value);
    }

    /// Run a single trial
    pub fn trial<F>(&mut self, evaluate: F) -> Result<TrialResult>
    where
        F: Fn(&HashMap<String, ParameterValue>) -> f32,
    {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Check time limit
        if let Some(start) = self.start_time {
            if start.elapsed() > self.config.max_time {
                return Err(OptError::Timeout {
                    timeout_secs: self.config.max_time.as_secs(),
                });
            }
        }

        // Get suggestion
        let params = self.optimizer.suggest();
        let trial_num = self.history.len() + 1;

        // Evaluate
        let eval_start = Instant::now();
        let value = evaluate(&params);
        let eval_time = eval_start.elapsed();

        // Check constraints
        let valid = self.check_constraints(&params, value);

        // Record observation (negate if minimizing)
        let objective = if self.config.minimize { -value } else { value };
        let observation = Observation {
            params: params.clone(),
            objective: if valid { objective } else { f32::NEG_INFINITY },
            eval_time_ms: eval_time.as_millis() as u64,
        };
        self.optimizer.observe(observation);

        let result = TrialResult {
            trial: trial_num,
            params,
            value,
            eval_time_ms: eval_time.as_millis() as u64,
            valid,
        };

        self.history.push(result.clone());

        Ok(result)
    }

    /// Check constraints
    fn check_constraints(&self, _params: &HashMap<String, ParameterValue>, value: f32) -> bool {
        for (constraint, limit) in &self.config.constraints {
            // Simple constraint checking
            if constraint == "max_latency_ms" && self.config.target_metric == "latency_ms" {
                if value > *limit {
                    return false;
                }
            }
            if constraint == "max_memory_mb" {
                // Would need memory measurement
            }
        }
        true
    }

    /// Run full tuning
    pub fn tune<F>(&mut self, evaluate: F) -> Result<TuningResult>
    where
        F: Fn(&HashMap<String, ParameterValue>) -> f32,
    {
        let start = Instant::now();

        for _ in 0..self.config.max_trials {
            match self.trial(&evaluate) {
                Ok(_) => {}
                Err(OptError::Timeout { .. }) => break,
                Err(e) => return Err(e),
            }

            // Check time limit
            if start.elapsed() > self.config.max_time {
                break;
            }
        }

        self.result()
    }

    /// Get current best result
    pub fn result(&self) -> Result<TuningResult> {
        // Find best valid trial
        let best = self
            .history
            .iter()
            .filter(|t| t.valid)
            .min_by(|a, b| {
                if self.config.minimize {
                    a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
            .ok_or_else(|| OptError::NoImprovement { trials: self.history.len() })?;

        let improvement = if let Some(baseline) = self.baseline {
            if self.config.minimize {
                (baseline - best.value) / baseline * 100.0
            } else {
                (best.value - baseline) / baseline * 100.0
            }
        } else {
            0.0
        };

        Ok(TuningResult {
            best_params: best.params.clone(),
            best_value: best.value,
            trials: self.history.len(),
            duration_secs: self.start_time
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0),
            improvement_percent: improvement,
            history: self.history.clone(),
        })
    }

    /// Get profiler
    pub fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }

    /// Get trial history
    pub fn history(&self) -> &[TrialResult] {
        &self.history
    }
}

impl std::fmt::Debug for AutoTuner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoTuner")
            .field("config", &self.config)
            .field("parameters", &self.parameters)
            .field("trials", &self.history.len())
            .finish()
    }
}

/// Quick-tune helper for common scenarios
pub mod presets {
    use super::*;

    /// Create tuner for batch size optimization
    pub fn batch_size_tuner(min: i32, max: i32) -> AutoTuner {
        let config = TunerConfig {
            target_metric: "throughput_tps".into(),
            minimize: false,
            max_trials: 20,
            ..Default::default()
        };

        let params = vec![Parameter::integer("batch_size", min, max)];

        AutoTuner::new(config, params)
    }

    /// Create tuner for precision optimization
    pub fn precision_tuner() -> AutoTuner {
        let config = TunerConfig {
            target_metric: "latency_ms".into(),
            minimize: true,
            max_trials: 10,
            ..Default::default()
        };

        let params = vec![Parameter::categorical(
            "precision",
            vec!["fp32".into(), "fp16".into(), "int8".into(), "int4".into()],
        )];

        AutoTuner::new(config, params)
    }

    /// Create tuner for KV cache configuration
    pub fn kv_cache_tuner() -> AutoTuner {
        let config = TunerConfig::default();

        let params = vec![
            Parameter::integer("num_blocks", 64, 512),
            Parameter::integer("block_size", 8, 64),
            Parameter::categorical(
                "cache_dtype",
                vec!["fp16".into(), "fp8".into(), "int8".into()],
            ),
        ];

        AutoTuner::new(config, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_creation() {
        let config = TunerConfig::default();
        let params = vec![Parameter::continuous("lr", 0.0001, 0.1)];

        let tuner = AutoTuner::new(config, params);
        assert!(tuner.history().is_empty());
    }

    #[test]
    fn test_single_trial() {
        let config = TunerConfig {
            max_trials: 5,
            ..Default::default()
        };
        let params = vec![Parameter::continuous("x", -1.0, 1.0)];

        let mut tuner = AutoTuner::new(config, params);

        let evaluate = |params: &HashMap<String, ParameterValue>| {
            let x = params.get("x").and_then(|v| v.as_float()).unwrap_or(0.0);
            x * x // Minimize x^2
        };

        let result = tuner.trial(evaluate).unwrap();
        assert_eq!(result.trial, 1);
    }

    #[test]
    fn test_full_tuning() {
        let config = TunerConfig {
            max_trials: 10,
            max_time: Duration::from_secs(10),
            minimize: true,
            ..Default::default()
        };
        let params = vec![Parameter::continuous("x", -5.0, 5.0)];

        let mut tuner = AutoTuner::new(config, params);
        tuner.set_baseline(25.0); // x=5 gives 25

        let evaluate = |params: &HashMap<String, ParameterValue>| {
            let x = params.get("x").and_then(|v| v.as_float()).unwrap_or(0.0);
            x * x
        };

        let result = tuner.tune(evaluate).unwrap();

        assert!(result.trials > 0);
        assert!(result.best_value < 25.0); // Should find better than baseline
    }

    #[test]
    fn test_presets() {
        let _ = presets::batch_size_tuner(1, 64);
        let _ = presets::precision_tuner();
        let _ = presets::kv_cache_tuner();
    }
}

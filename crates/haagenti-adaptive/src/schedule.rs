//! Precision schedule for a specific generation

use crate::{
    AdaptiveError, Precision, PrecisionCapabilities, PrecisionProfile, ProfilePreset, Result,
    TransitionStrategy,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Configuration for schedule generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    /// Total number of denoising steps
    pub total_steps: u32,
    /// Profile preset to use
    pub preset: ProfilePreset,
    /// Custom profile (overrides preset if set)
    pub custom_profile: Option<PrecisionProfile>,
    /// Hardware capabilities
    pub capabilities: PrecisionCapabilities,
    /// Minimum quality threshold (0.0 - 1.0)
    pub min_quality: f32,
    /// Transition strategy between precisions
    pub transition_strategy: TransitionStrategy,
    /// Whether to force capabilities check
    pub strict_capabilities: bool,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            total_steps: 30,
            preset: ProfilePreset::Balanced,
            custom_profile: None,
            capabilities: PrecisionCapabilities::default(),
            min_quality: 0.90,
            transition_strategy: TransitionStrategy::Immediate,
            strict_capabilities: true,
        }
    }
}

impl ScheduleConfig {
    /// Create config for a specific preset
    pub fn with_preset(preset: ProfilePreset) -> Self {
        Self {
            preset,
            ..Default::default()
        }
    }

    /// Set total steps
    pub fn steps(mut self, steps: u32) -> Self {
        self.total_steps = steps;
        self
    }

    /// Set hardware capabilities
    pub fn capabilities(mut self, caps: PrecisionCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Set minimum quality
    pub fn min_quality(mut self, quality: f32) -> Self {
        self.min_quality = quality;
        self
    }
}

/// Precision assignment for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepPrecision {
    /// Step number
    pub step: u32,
    /// Assigned precision
    pub precision: Precision,
    /// Whether this is a transition step
    pub is_transition: bool,
    /// Blending factor (for gradual transitions)
    pub blend_factor: Option<f32>,
    /// Expected VRAM usage ratio
    pub vram_ratio: f32,
    /// Expected quality impact
    pub quality_factor: f32,
}

/// Complete precision schedule for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSchedule {
    /// Configuration used to generate this schedule
    pub config: ScheduleConfig,
    /// Profile used
    pub profile_name: String,
    /// Per-step precision assignments
    pub steps: Vec<StepPrecision>,
    /// Precision transition points
    pub transitions: Vec<(u32, Precision, Precision)>,
    /// Estimated average VRAM ratio
    pub avg_vram_ratio: f32,
    /// Estimated quality factor
    pub avg_quality_factor: f32,
    /// Estimated speedup factor
    pub estimated_speedup: f32,
}

impl PrecisionSchedule {
    /// Generate a schedule from configuration
    pub fn generate(config: ScheduleConfig) -> Result<Self> {
        if config.total_steps == 0 {
            return Err(AdaptiveError::ScheduleError(
                "Total steps must be > 0".into(),
            ));
        }

        let profile = config
            .custom_profile
            .clone()
            .unwrap_or_else(|| config.preset.build());

        // Validate profile
        profile.validate()?;

        // Generate step assignments
        let mut steps = Vec::with_capacity(config.total_steps as usize);
        let mut transitions = Vec::new();
        let mut prev_precision: Option<Precision> = None;

        for step in 0..config.total_steps {
            let fraction = step as f32 / config.total_steps as f32;
            let mut precision = profile.precision_at(fraction);

            // Adjust for hardware capabilities
            if config.strict_capabilities && !config.capabilities.supports(precision) {
                precision = config.capabilities.best_supported(precision);
            }

            let is_transition = prev_precision.map_or(false, |p| p != precision);

            if is_transition {
                if let Some(prev) = prev_precision {
                    transitions.push((step, prev, precision));
                }
            }

            let blend_factor = if is_transition {
                match config.transition_strategy {
                    TransitionStrategy::Immediate => None,
                    TransitionStrategy::Gradual { steps: blend_steps } => {
                        Some(1.0 / blend_steps as f32)
                    }
                    TransitionStrategy::StepAware => Some(0.5),
                }
            } else {
                None
            };

            steps.push(StepPrecision {
                step,
                precision,
                is_transition,
                blend_factor,
                vram_ratio: precision.vram_ratio(),
                quality_factor: precision.quality_factor(),
            });

            prev_precision = Some(precision);
        }

        // Calculate averages
        let avg_vram_ratio =
            steps.iter().map(|s| s.vram_ratio).sum::<f32>() / steps.len() as f32;
        let avg_quality_factor =
            steps.iter().map(|s| s.quality_factor).sum::<f32>() / steps.len() as f32;
        let estimated_speedup = steps
            .iter()
            .map(|s| s.precision.speedup_factor())
            .sum::<f32>()
            / steps.len() as f32;

        // Check quality constraint
        if avg_quality_factor < config.min_quality {
            return Err(AdaptiveError::QualityConstraint {
                actual: avg_quality_factor,
                threshold: config.min_quality,
            });
        }

        Ok(Self {
            config,
            profile_name: profile.name,
            steps,
            transitions,
            avg_vram_ratio,
            avg_quality_factor,
            estimated_speedup,
        })
    }

    /// Get precision for a specific step
    pub fn precision_at(&self, step: u32) -> Result<Precision> {
        self.steps
            .get(step as usize)
            .map(|s| s.precision)
            .ok_or(AdaptiveError::InvalidStep {
                step,
                total_steps: self.config.total_steps,
            })
    }

    /// Get step precision info
    pub fn step_info(&self, step: u32) -> Result<&StepPrecision> {
        self.steps
            .get(step as usize)
            .ok_or(AdaptiveError::InvalidStep {
                step,
                total_steps: self.config.total_steps,
            })
    }

    /// Get next transition after a given step
    pub fn next_transition(&self, after_step: u32) -> Option<(u32, Precision, Precision)> {
        self.transitions
            .iter()
            .find(|(step, _, _)| *step > after_step)
            .copied()
    }

    /// Get all steps using a specific precision
    pub fn steps_at_precision(&self, precision: Precision) -> SmallVec<[u32; 32]> {
        self.steps
            .iter()
            .filter(|s| s.precision == precision)
            .map(|s| s.step)
            .collect()
    }

    /// Total time at each precision
    pub fn precision_distribution(&self) -> Vec<(Precision, usize, f32)> {
        let mut counts: std::collections::HashMap<Precision, usize> = std::collections::HashMap::new();

        for step in &self.steps {
            *counts.entry(step.precision).or_insert(0) += 1;
        }

        let total = self.steps.len() as f32;
        let mut result: Vec<_> = counts
            .into_iter()
            .map(|(p, count)| (p, count, count as f32 / total))
            .collect();

        result.sort_by_key(|(p, _, _)| *p);
        result
    }

    /// Format as a visual timeline
    pub fn format_timeline(&self) -> String {
        let mut result = String::new();

        result.push_str("Step: ");
        for step in &self.steps {
            if step.step % 5 == 0 {
                result.push_str(&format!("{:2} ", step.step));
            }
        }
        result.push('\n');

        result.push_str("Prec: ");
        for step in &self.steps {
            let symbol = match step.precision {
                Precision::INT4 => '4',
                Precision::INT8 => '8',
                Precision::BF16 => 'B',
                Precision::FP16 => 'H',
                Precision::FP32 => 'F',
            };
            result.push(symbol);
        }
        result.push('\n');

        result
    }
}

/// Quick schedule generation for common cases
pub fn quick_schedule(preset: ProfilePreset, steps: u32) -> Result<PrecisionSchedule> {
    PrecisionSchedule::generate(ScheduleConfig::with_preset(preset).steps(steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schedule() {
        let schedule = quick_schedule(ProfilePreset::Balanced, 20).unwrap();

        assert_eq!(schedule.steps.len(), 20);
        assert!(!schedule.transitions.is_empty());

        // Early steps should be lower precision
        assert!(schedule.steps[0].precision <= Precision::INT8);
        // Late steps should be higher precision
        assert!(schedule.steps[19].precision >= Precision::FP16);
    }

    #[test]
    fn test_precision_distribution() {
        let schedule = quick_schedule(ProfilePreset::Performance, 30).unwrap();
        let dist = schedule.precision_distribution();

        // Should have multiple precisions
        assert!(dist.len() >= 2);

        // Percentages should sum to ~1.0
        let total: f32 = dist.iter().map(|(_, _, pct)| pct).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_step_lookup() {
        let schedule = quick_schedule(ProfilePreset::Balanced, 20).unwrap();

        let info = schedule.step_info(10).unwrap();
        assert_eq!(info.step, 10);

        // Invalid step should error
        assert!(schedule.step_info(100).is_err());
    }

    #[test]
    fn test_quality_constraint() {
        let config = ScheduleConfig::with_preset(ProfilePreset::Performance)
            .steps(20)
            .min_quality(0.999); // Impossible with Performance preset

        let result = PrecisionSchedule::generate(config);
        assert!(matches!(result, Err(AdaptiveError::QualityConstraint { .. })));
    }

    #[test]
    fn test_capabilities_adjustment() {
        // Legacy GPU doesn't support INT4
        let config = ScheduleConfig::with_preset(ProfilePreset::Performance)
            .steps(20)
            .capabilities(PrecisionCapabilities::legacy_gpu(4096));

        let schedule = PrecisionSchedule::generate(config).unwrap();

        // Should have adjusted to supported precisions
        for step in &schedule.steps {
            assert!(step.precision >= Precision::FP16);
        }
    }
}

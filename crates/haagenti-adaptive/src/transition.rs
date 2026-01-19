//! Precision transition strategies

use crate::{AdaptiveError, Precision, Result};
use serde::{Deserialize, Serialize};

/// Strategy for transitioning between precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionStrategy {
    /// Immediate switch (most efficient)
    Immediate,
    /// Gradual blend over multiple steps
    Gradual {
        /// Number of steps to blend over
        steps: u32,
    },
    /// Adaptive based on noise level
    StepAware,
}

impl Default for TransitionStrategy {
    fn default() -> Self {
        TransitionStrategy::Immediate
    }
}

/// A precision transition with blending information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionTransition {
    /// Source precision
    pub from: Precision,
    /// Target precision
    pub to: Precision,
    /// Step at which transition starts
    pub start_step: u32,
    /// Step at which transition completes
    pub end_step: u32,
    /// Strategy used
    pub strategy: TransitionStrategy,
    /// Precomputed blend factors per step
    pub blend_factors: Vec<f32>,
}

impl PrecisionTransition {
    /// Create a new immediate transition
    pub fn immediate(from: Precision, to: Precision, step: u32) -> Self {
        Self {
            from,
            to,
            start_step: step,
            end_step: step,
            strategy: TransitionStrategy::Immediate,
            blend_factors: vec![1.0],
        }
    }

    /// Create a gradual transition
    pub fn gradual(from: Precision, to: Precision, start_step: u32, duration: u32) -> Self {
        let blend_factors: Vec<f32> = (0..=duration)
            .map(|i| i as f32 / duration as f32)
            .collect();

        Self {
            from,
            to,
            start_step,
            end_step: start_step + duration,
            strategy: TransitionStrategy::Gradual { steps: duration },
            blend_factors,
        }
    }

    /// Check if a step is within this transition
    pub fn contains_step(&self, step: u32) -> bool {
        step >= self.start_step && step <= self.end_step
    }

    /// Get blend factor for a specific step (0.0 = from, 1.0 = to)
    pub fn blend_at(&self, step: u32) -> Option<f32> {
        if !self.contains_step(step) {
            return None;
        }

        match self.strategy {
            TransitionStrategy::Immediate => Some(1.0),
            TransitionStrategy::Gradual { steps } => {
                let progress = (step - self.start_step) as f32 / steps as f32;
                Some(progress.clamp(0.0, 1.0))
            }
            TransitionStrategy::StepAware => {
                // Use smooth step for noise-aware transitions
                let t = (step - self.start_step) as f32 / (self.end_step - self.start_step) as f32;
                Some(smooth_step(t))
            }
        }
    }

    /// Get the effective precision at a step
    pub fn effective_precision(&self, step: u32) -> Precision {
        match self.blend_at(step) {
            Some(blend) if blend < 0.5 => self.from,
            Some(_) => self.to,
            None => self.to,
        }
    }

    /// Validate the transition
    pub fn validate(&self) -> Result<()> {
        if self.end_step < self.start_step {
            return Err(AdaptiveError::InvalidTransition {
                from: self.from,
                to: self.to,
                reason: "End step before start step".into(),
            });
        }

        // Check for valid precision ordering in most cases
        // (going from lower to higher is normal, but reverse is allowed for special cases)

        Ok(())
    }

    /// Compute VRAM requirement during transition
    pub fn peak_vram_ratio(&self) -> f32 {
        match self.strategy {
            TransitionStrategy::Immediate => self.to.vram_ratio(),
            TransitionStrategy::Gradual { .. } | TransitionStrategy::StepAware => {
                // During gradual transition, may need both precisions loaded
                self.from.vram_ratio().max(self.to.vram_ratio()) * 1.2
            }
        }
    }
}

/// Smooth step function for gradual transitions
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step function (Ken Perlin's version)
#[allow(dead_code)]
fn smoother_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Transition planner for optimizing precision changes
#[derive(Debug, Clone)]
pub struct TransitionPlanner {
    /// Available VRAM
    vram_mb: u64,
    /// Preferred strategy
    preferred_strategy: TransitionStrategy,
    /// Minimum steps between transitions
    min_gap: u32,
}

impl TransitionPlanner {
    /// Create a new planner
    pub fn new(vram_mb: u64) -> Self {
        Self {
            vram_mb,
            preferred_strategy: TransitionStrategy::Immediate,
            min_gap: 3,
        }
    }

    /// Set preferred strategy
    pub fn with_strategy(mut self, strategy: TransitionStrategy) -> Self {
        self.preferred_strategy = strategy;
        self
    }

    /// Plan transitions between precision zones
    pub fn plan_transitions(
        &self,
        zones: &[(u32, u32, Precision)], // (start, end, precision)
    ) -> Vec<PrecisionTransition> {
        if zones.len() <= 1 {
            return Vec::new();
        }

        let mut transitions = Vec::new();

        for window in zones.windows(2) {
            let (_, end1, prec1) = window[0];
            let (start2, _, prec2) = window[1];

            if prec1 == prec2 {
                continue;
            }

            let transition = match self.preferred_strategy {
                TransitionStrategy::Immediate => PrecisionTransition::immediate(prec1, prec2, start2),
                TransitionStrategy::Gradual { steps } => {
                    // Ensure transition doesn't exceed zone boundaries
                    let safe_steps = steps.min(end1.saturating_sub(1));
                    PrecisionTransition::gradual(prec1, prec2, end1.saturating_sub(safe_steps), safe_steps)
                }
                TransitionStrategy::StepAware => {
                    // Use 2 steps for step-aware transitions
                    PrecisionTransition::gradual(prec1, prec2, end1.saturating_sub(1), 2)
                }
            };

            transitions.push(transition);
        }

        transitions
    }

    /// Optimize transitions to minimize VRAM spikes
    pub fn optimize_for_vram(&self, transitions: &mut [PrecisionTransition]) {
        for transition in transitions {
            // If gradual transition would cause VRAM spike, make it immediate
            if transition.peak_vram_ratio() > 0.9 {
                *transition = PrecisionTransition::immediate(
                    transition.from,
                    transition.to,
                    transition.start_step,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immediate_transition() {
        let trans = PrecisionTransition::immediate(Precision::INT4, Precision::FP16, 10);

        assert_eq!(trans.start_step, 10);
        assert_eq!(trans.end_step, 10);
        assert!(trans.contains_step(10));
        assert!(!trans.contains_step(9));
        assert_eq!(trans.blend_at(10), Some(1.0));
    }

    #[test]
    fn test_gradual_transition() {
        let trans = PrecisionTransition::gradual(Precision::INT8, Precision::FP16, 10, 4);

        assert_eq!(trans.start_step, 10);
        assert_eq!(trans.end_step, 14);

        assert_eq!(trans.blend_at(10), Some(0.0));
        assert_eq!(trans.blend_at(12), Some(0.5));
        assert_eq!(trans.blend_at(14), Some(1.0));
    }

    #[test]
    fn test_effective_precision() {
        let trans = PrecisionTransition::gradual(Precision::INT4, Precision::FP16, 10, 4);

        assert_eq!(trans.effective_precision(10), Precision::INT4);
        assert_eq!(trans.effective_precision(11), Precision::INT4);
        assert_eq!(trans.effective_precision(12), Precision::FP16);
        assert_eq!(trans.effective_precision(14), Precision::FP16);
    }

    #[test]
    fn test_smooth_step() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_transition_planner() {
        let planner = TransitionPlanner::new(8192);

        let zones = vec![
            (0, 10, Precision::INT4),
            (10, 20, Precision::INT8),
            (20, 30, Precision::FP16),
        ];

        let transitions = planner.plan_transitions(&zones);
        assert_eq!(transitions.len(), 2);

        assert_eq!(transitions[0].from, Precision::INT4);
        assert_eq!(transitions[0].to, Precision::INT8);

        assert_eq!(transitions[1].from, Precision::INT8);
        assert_eq!(transitions[1].to, Precision::FP16);
    }
}

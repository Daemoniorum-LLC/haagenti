//! Precision profiles for different use cases

use crate::{AdaptiveError, Precision, Result};
use serde::{Deserialize, Serialize};

/// A precision profile defining step-to-precision mapping rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionProfile {
    /// Profile name
    pub name: String,
    /// Description
    pub description: String,
    /// Precision zones (sorted by start_step)
    pub zones: Vec<PrecisionZone>,
    /// Quality target (0.0 - 1.0)
    pub quality_target: f32,
    /// VRAM target percentage (0.0 - 1.0)
    pub vram_target: f32,
}

/// A zone where a specific precision is used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionZone {
    /// Start step (inclusive)
    pub start_step: f32,
    /// End step (exclusive), as fraction of total steps
    pub end_step: f32,
    /// Precision to use in this zone
    pub precision: Precision,
    /// Reason for this choice
    pub rationale: String,
}

impl PrecisionProfile {
    /// Create a new profile
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            zones: Vec::new(),
            quality_target: 0.95,
            vram_target: 0.8,
        }
    }

    /// Add a precision zone
    pub fn add_zone(
        mut self,
        start: f32,
        end: f32,
        precision: Precision,
        rationale: impl Into<String>,
    ) -> Self {
        self.zones.push(PrecisionZone {
            start_step: start,
            end_step: end,
            precision,
            rationale: rationale.into(),
        });
        self.zones.sort_by(|a, b| {
            a.start_step
                .partial_cmp(&b.start_step)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Get precision for a given step fraction (0.0 - 1.0)
    pub fn precision_at(&self, step_fraction: f32) -> Precision {
        for zone in &self.zones {
            if step_fraction >= zone.start_step && step_fraction < zone.end_step {
                return zone.precision;
            }
        }
        // Default to FP16 if no zone matches
        Precision::FP16
    }

    /// Validate the profile
    pub fn validate(&self) -> Result<()> {
        if self.zones.is_empty() {
            return Err(AdaptiveError::ProfileError(
                "Profile must have at least one zone".into(),
            ));
        }

        // Check for gaps
        let mut expected_start = 0.0f32;
        for zone in &self.zones {
            if (zone.start_step - expected_start).abs() > 0.001 {
                return Err(AdaptiveError::ProfileError(format!(
                    "Gap in zones at step fraction {}",
                    expected_start
                )));
            }
            expected_start = zone.end_step;
        }

        if (expected_start - 1.0).abs() > 0.001 {
            return Err(AdaptiveError::ProfileError(
                "Zones must cover the full range [0.0, 1.0)".into(),
            ));
        }

        Ok(())
    }

    /// Estimate average VRAM usage ratio
    pub fn estimated_vram_ratio(&self) -> f32 {
        let mut total_weight = 0.0f32;
        let mut weighted_ratio = 0.0f32;

        for zone in &self.zones {
            let weight = zone.end_step - zone.start_step;
            total_weight += weight;
            weighted_ratio += weight * zone.precision.vram_ratio();
        }

        if total_weight > 0.0 {
            weighted_ratio / total_weight
        } else {
            1.0
        }
    }

    /// Estimate average quality factor
    pub fn estimated_quality(&self) -> f32 {
        let mut total_weight = 0.0f32;
        let mut weighted_quality = 0.0f32;

        for zone in &self.zones {
            let weight = zone.end_step - zone.start_step;
            // Later steps matter more for quality
            let quality_weight = weight * (0.5 + zone.start_step * 0.5);
            total_weight += quality_weight;
            weighted_quality += quality_weight * zone.precision.quality_factor();
        }

        if total_weight > 0.0 {
            weighted_quality / total_weight
        } else {
            1.0
        }
    }

    /// Count precision transitions
    pub fn transition_count(&self) -> usize {
        if self.zones.len() <= 1 {
            return 0;
        }

        self.zones
            .windows(2)
            .filter(|w| w[0].precision != w[1].precision)
            .count()
    }
}

/// Preset profiles for common use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ProfilePreset {
    /// Maximum performance, lower quality
    Performance,
    /// Balanced performance and quality
    #[default]
    Balanced,
    /// Maximum quality, slower
    Quality,
    /// Aggressive VRAM savings
    LowVram,
    /// Conservative, mostly FP16
    Conservative,
    /// Custom noise-aware schedule
    NoiseAdaptive,
}

impl ProfilePreset {
    /// Build the profile for this preset
    pub fn build(self) -> PrecisionProfile {
        match self {
            ProfilePreset::Performance => {
                PrecisionProfile::new("Performance", "Maximum speed with aggressive INT4 usage")
                    .add_zone(
                        0.0,
                        0.4,
                        Precision::INT4,
                        "High noise masks quantization errors",
                    )
                    .add_zone(0.4, 0.7, Precision::INT8, "Medium noise tolerates INT8")
                    .add_zone(0.7, 1.0, Precision::FP16, "Low noise requires precision")
            }

            ProfilePreset::Balanced => {
                PrecisionProfile::new("Balanced", "Good balance of speed and quality")
                    .add_zone(0.0, 0.25, Precision::INT4, "Early steps: high noise")
                    .add_zone(
                        0.25,
                        0.5,
                        Precision::INT8,
                        "Mid-early steps: moderate noise",
                    )
                    .add_zone(
                        0.5,
                        1.0,
                        Precision::FP16,
                        "Later steps: detail preservation",
                    )
            }

            ProfilePreset::Quality => {
                PrecisionProfile::new("Quality", "Maximum quality with minimal quantization")
                    .add_zone(0.0, 0.15, Precision::INT8, "Only very early INT8")
                    .add_zone(0.15, 1.0, Precision::FP16, "FP16 for most steps")
            }

            ProfilePreset::LowVram => PrecisionProfile::new("LowVRAM", "Aggressive memory savings")
                .add_zone(0.0, 0.5, Precision::INT4, "Extended INT4 zone")
                .add_zone(0.5, 0.85, Precision::INT8, "INT8 for refinement")
                .add_zone(0.85, 1.0, Precision::FP16, "FP16 only for final details"),

            ProfilePreset::Conservative => {
                PrecisionProfile::new("Conservative", "Minimal quantization for maximum quality")
                    .add_zone(0.0, 0.1, Precision::INT8, "Brief INT8 at start")
                    .add_zone(0.1, 1.0, Precision::FP16, "FP16 throughout")
            }

            ProfilePreset::NoiseAdaptive => PrecisionProfile::new(
                "NoiseAdaptive",
                "Precision matched to noise level at each step",
            )
            .add_zone(0.0, 0.2, Precision::INT4, "Noise sigma > 5.0")
            .add_zone(0.2, 0.35, Precision::INT8, "Noise sigma 2.0-5.0")
            .add_zone(0.35, 0.6, Precision::INT8, "Noise sigma 0.5-2.0")
            .add_zone(0.6, 0.8, Precision::FP16, "Noise sigma 0.1-0.5")
            .add_zone(0.8, 1.0, Precision::FP16, "Noise sigma < 0.1"),
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            ProfilePreset::Performance => "Maximum speed (4x faster, ~8% quality loss)",
            ProfilePreset::Balanced => "Balanced (2.5x faster, ~3% quality loss)",
            ProfilePreset::Quality => "Maximum quality (1.5x faster, ~1% quality loss)",
            ProfilePreset::LowVram => "Low VRAM (30% less memory, ~5% quality loss)",
            ProfilePreset::Conservative => "Conservative (1.2x faster, minimal quality loss)",
            ProfilePreset::NoiseAdaptive => "Noise-aware scheduling (optimal quality/speed)",
        }
    }

    /// List all presets
    pub fn all() -> &'static [ProfilePreset] {
        &[
            ProfilePreset::Performance,
            ProfilePreset::Balanced,
            ProfilePreset::Quality,
            ProfilePreset::LowVram,
            ProfilePreset::Conservative,
            ProfilePreset::NoiseAdaptive,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_profile() {
        let profile = ProfilePreset::Balanced.build();
        assert_eq!(profile.zones.len(), 3);

        // Early step should be INT4
        assert_eq!(profile.precision_at(0.1), Precision::INT4);
        // Mid step should be INT8
        assert_eq!(profile.precision_at(0.3), Precision::INT8);
        // Late step should be FP16
        assert_eq!(profile.precision_at(0.8), Precision::FP16);
    }

    #[test]
    fn test_profile_validation() {
        let valid = ProfilePreset::Balanced.build();
        assert!(valid.validate().is_ok());

        // Invalid: gap in zones
        let invalid = PrecisionProfile::new("Invalid", "Has gaps")
            .add_zone(0.0, 0.3, Precision::INT4, "")
            .add_zone(0.5, 1.0, Precision::FP16, ""); // Gap at 0.3-0.5

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_estimated_metrics() {
        let performance = ProfilePreset::Performance.build();
        let quality = ProfilePreset::Quality.build();

        // Performance should use less VRAM
        assert!(performance.estimated_vram_ratio() < quality.estimated_vram_ratio());

        // Quality should have better quality factor
        assert!(quality.estimated_quality() > performance.estimated_quality());
    }

    #[test]
    fn test_transition_count() {
        let balanced = ProfilePreset::Balanced.build();
        assert_eq!(balanced.transition_count(), 2); // INT4->INT8, INT8->FP16

        let conservative = ProfilePreset::Conservative.build();
        assert_eq!(conservative.transition_count(), 1); // INT8->FP16
    }
}

//! Error types for adaptive precision

use thiserror::Error;

/// Errors that can occur during adaptive precision operations
#[derive(Debug, Error)]
pub enum AdaptiveError {
    /// Invalid step number
    #[error("Invalid step {step}: must be in range [0, {total_steps})")]
    InvalidStep { step: u32, total_steps: u32 },

    /// Invalid precision transition
    #[error("Invalid transition from {from:?} to {to:?}: {reason}")]
    InvalidTransition {
        from: crate::Precision,
        to: crate::Precision,
        reason: String,
    },

    /// Precision not supported by hardware
    #[error("Precision {0:?} not supported by hardware")]
    UnsupportedPrecision(crate::Precision),

    /// Profile configuration error
    #[error("Profile error: {0}")]
    ProfileError(String),

    /// Schedule generation failed
    #[error("Failed to generate schedule: {0}")]
    ScheduleError(String),

    /// VRAM constraint violation
    #[error("VRAM constraint violated: required {required_mb}MB, available {available_mb}MB")]
    VramConstraint { required_mb: u64, available_mb: u64 },

    /// Quality constraint violation
    #[error("Quality below threshold: {actual:.3} < {threshold:.3}")]
    QualityConstraint { actual: f32, threshold: f32 },
}

/// Result type for adaptive precision operations
pub type Result<T> = std::result::Result<T, AdaptiveError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AdaptiveError::InvalidStep {
            step: 50,
            total_steps: 30,
        };
        assert!(err.to_string().contains("50"));
        assert!(err.to_string().contains("30"));
    }
}

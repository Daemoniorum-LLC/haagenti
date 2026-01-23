//! Preview scheduling strategies

use crate::PreviewQuality;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Schedule mode for preview generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleMode {
    /// Preview every step (smooth but slow)
    EveryStep,
    /// Preview every N steps
    Interval { steps: u32 },
    /// Preview at specific steps
    Fixed,
    /// Adaptive based on change detection
    Adaptive,
    /// Thumbnail every step, full at end
    ThumbnailOnly,
    /// No previews until final
    FinalOnly,
}

impl Default for ScheduleMode {
    fn default() -> Self {
        ScheduleMode::Interval { steps: 5 }
    }
}

/// Event indicating a preview should be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewEvent {
    /// Step to generate preview for
    pub step: u32,
    /// Quality level for this preview
    pub quality: PreviewQuality,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Whether this is the final preview
    pub is_final: bool,
}

/// Scheduler for determining when to generate previews
#[derive(Debug, Clone)]
pub struct PreviewScheduler {
    /// Schedule mode
    mode: ScheduleMode,
    /// Total steps in generation
    total_steps: u32,
    /// Fixed preview steps (for Fixed mode)
    fixed_steps: SmallVec<[u32; 8]>,
    /// Quality schedule by step fraction
    quality_schedule: Vec<(f32, PreviewQuality)>,
    /// Last step that generated a preview
    last_preview_step: Option<u32>,
    /// Change threshold for adaptive mode
    change_threshold: f32,
}

impl PreviewScheduler {
    /// Create a new scheduler
    pub fn new(mode: ScheduleMode, total_steps: u32) -> Self {
        let fixed_steps = Self::default_fixed_steps(total_steps);
        let quality_schedule = Self::default_quality_schedule();

        Self {
            mode,
            total_steps,
            fixed_steps,
            quality_schedule,
            last_preview_step: None,
            change_threshold: 0.1,
        }
    }

    /// Default fixed preview steps
    fn default_fixed_steps(total_steps: u32) -> SmallVec<[u32; 8]> {
        let mut steps = SmallVec::new();
        // First, middle, and last
        steps.push(0);
        if total_steps > 4 {
            steps.push(total_steps / 4);
        }
        if total_steps > 2 {
            steps.push(total_steps / 2);
        }
        if total_steps > 4 {
            steps.push(3 * total_steps / 4);
        }
        steps.push(total_steps);
        steps
    }

    /// Default quality schedule
    fn default_quality_schedule() -> Vec<(f32, PreviewQuality)> {
        vec![
            (0.0, PreviewQuality::Thumbnail),
            (0.3, PreviewQuality::Low),
            (0.6, PreviewQuality::Medium),
            (0.9, PreviewQuality::Full),
        ]
    }

    /// Check if a preview should be generated at this step
    pub fn should_preview(&self, step: u32) -> bool {
        if step > self.total_steps {
            return false;
        }

        match self.mode {
            ScheduleMode::EveryStep => true,
            ScheduleMode::Interval { steps } => {
                step.is_multiple_of(steps) || step == self.total_steps
            }
            ScheduleMode::Fixed => self.fixed_steps.contains(&step),
            ScheduleMode::Adaptive => self.adaptive_check(step),
            ScheduleMode::ThumbnailOnly => true,
            ScheduleMode::FinalOnly => step == self.total_steps,
        }
    }

    /// Adaptive check based on step importance
    fn adaptive_check(&self, step: u32) -> bool {
        let progress = step as f32 / self.total_steps as f32;

        // More previews early (structure forming) and late (details)
        let interval = if progress < 0.2 {
            2 // Early: every 2 steps
        } else if progress < 0.7 {
            5 // Middle: every 5 steps
        } else {
            3 // Late: every 3 steps (details matter)
        };

        step.is_multiple_of(interval) || step == self.total_steps
    }

    /// Get the quality for a step
    pub fn quality_for_step(&self, step: u32) -> PreviewQuality {
        if self.mode == ScheduleMode::ThumbnailOnly && step < self.total_steps {
            return PreviewQuality::Thumbnail;
        }

        let progress = step as f32 / self.total_steps as f32;

        // Find appropriate quality for this progress
        for i in (0..self.quality_schedule.len()).rev() {
            if progress >= self.quality_schedule[i].0 {
                return self.quality_schedule[i].1;
            }
        }

        PreviewQuality::Thumbnail
    }

    /// Generate preview event for a step
    pub fn event_for_step(&self, step: u32) -> Option<PreviewEvent> {
        if !self.should_preview(step) {
            return None;
        }

        let is_final = step >= self.total_steps;
        let quality = if is_final {
            PreviewQuality::Full
        } else {
            self.quality_for_step(step)
        };

        // Priority: higher for final and late steps
        let priority = if is_final {
            255
        } else {
            (step as f32 / self.total_steps as f32 * 200.0) as u8
        };

        Some(PreviewEvent {
            step,
            quality,
            priority,
            is_final,
        })
    }

    /// Get all scheduled preview events
    pub fn all_events(&self) -> Vec<PreviewEvent> {
        (0..=self.total_steps)
            .filter_map(|step| self.event_for_step(step))
            .collect()
    }

    /// Estimated overhead percentage
    pub fn estimated_overhead(&self) -> f32 {
        let events = self.all_events();
        if events.is_empty() || self.total_steps == 0 {
            return 0.0;
        }

        let decode_time: u32 = events.iter().map(|e| e.quality.decode_time_ms()).sum();
        let step_time = 100; // Assume 100ms per step

        let total_gen_time = self.total_steps * step_time;
        decode_time as f32 / total_gen_time as f32 * 100.0
    }

    /// Update schedule after a preview was generated
    pub fn mark_preview(&mut self, step: u32) {
        self.last_preview_step = Some(step);
    }

    /// Set custom fixed steps
    pub fn set_fixed_steps(&mut self, steps: impl IntoIterator<Item = u32>) {
        self.fixed_steps = steps.into_iter().collect();
    }

    /// Set change threshold for adaptive mode
    pub fn set_change_threshold(&mut self, threshold: f32) {
        self.change_threshold = threshold;
    }

    /// Get schedule mode
    pub fn mode(&self) -> ScheduleMode {
        self.mode
    }

    /// Get total steps
    pub fn total_steps(&self) -> u32 {
        self.total_steps
    }
}

/// Calculate schedule statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleStats {
    /// Total preview events
    pub total_previews: usize,
    /// Estimated overhead percentage
    pub overhead_percent: f32,
    /// Steps between previews (average)
    pub avg_interval: f32,
    /// Quality distribution
    pub quality_distribution: Vec<(PreviewQuality, usize)>,
}

impl ScheduleStats {
    /// Compute from scheduler
    pub fn from_scheduler(scheduler: &PreviewScheduler) -> Self {
        let events = scheduler.all_events();
        let total_previews = events.len();

        let mut quality_counts = std::collections::HashMap::new();
        for event in &events {
            *quality_counts.entry(event.quality).or_insert(0) += 1;
        }

        let quality_distribution: Vec<_> = quality_counts.into_iter().collect();

        let avg_interval = if total_previews > 1 {
            scheduler.total_steps() as f32 / (total_previews - 1) as f32
        } else {
            scheduler.total_steps() as f32
        };

        Self {
            total_previews,
            overhead_percent: scheduler.estimated_overhead(),
            avg_interval,
            quality_distribution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_schedule() {
        let scheduler = PreviewScheduler::new(ScheduleMode::Interval { steps: 5 }, 20);

        assert!(scheduler.should_preview(0));
        assert!(!scheduler.should_preview(1));
        assert!(scheduler.should_preview(5));
        assert!(scheduler.should_preview(10));
        assert!(scheduler.should_preview(20)); // Final
    }

    #[test]
    fn test_every_step() {
        let scheduler = PreviewScheduler::new(ScheduleMode::EveryStep, 10);

        for step in 0..=10 {
            assert!(scheduler.should_preview(step));
        }
    }

    #[test]
    fn test_final_only() {
        let scheduler = PreviewScheduler::new(ScheduleMode::FinalOnly, 20);

        assert!(!scheduler.should_preview(0));
        assert!(!scheduler.should_preview(10));
        assert!(scheduler.should_preview(20));
    }

    #[test]
    fn test_quality_schedule() {
        let scheduler = PreviewScheduler::new(ScheduleMode::Interval { steps: 5 }, 20);

        // Early steps: thumbnail/low
        assert!(matches!(
            scheduler.quality_for_step(0),
            PreviewQuality::Thumbnail
        ));

        // Late steps: higher quality
        let late_quality = scheduler.quality_for_step(18);
        assert!(matches!(
            late_quality,
            PreviewQuality::Medium | PreviewQuality::Full
        ));
    }

    #[test]
    fn test_estimated_overhead() {
        let every_step = PreviewScheduler::new(ScheduleMode::EveryStep, 20);
        let interval = PreviewScheduler::new(ScheduleMode::Interval { steps: 5 }, 20);
        let final_only = PreviewScheduler::new(ScheduleMode::FinalOnly, 20);

        assert!(every_step.estimated_overhead() > interval.estimated_overhead());
        assert!(interval.estimated_overhead() > final_only.estimated_overhead());
    }

    #[test]
    fn test_schedule_stats() {
        let scheduler = PreviewScheduler::new(ScheduleMode::Interval { steps: 5 }, 20);
        let stats = ScheduleStats::from_scheduler(&scheduler);

        assert!(stats.total_previews > 0);
        assert!(stats.avg_interval > 0.0);
    }
}

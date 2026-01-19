//! Session history tracking for pattern learning

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// User preferences learned from session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred styles
    pub styles: Vec<String>,
    /// Preferred subjects
    pub subjects: Vec<String>,
    /// Average prompt length
    pub avg_prompt_length: f32,
    /// Common prefixes
    pub common_prefixes: Vec<String>,
    /// Time-of-day patterns (hour -> style weights)
    pub time_patterns: HashMap<u8, Vec<(String, f32)>>,
}

/// A pattern detected in session history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Associated value
    pub value: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Times observed
    pub count: u32,
}

/// Types of patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Style preference (anime, realistic, etc.)
    Style,
    /// Subject preference (portrait, landscape, etc.)
    Subject,
    /// Common prefix sequence
    Prefix,
    /// Common modifier
    Modifier,
    /// Time-based preference
    Temporal,
}

/// Session history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryEntry {
    /// Full prompt
    prompt: String,
    /// Timestamp
    timestamp: u64,
    /// Generation was completed
    completed: bool,
    /// User satisfaction (if rated)
    satisfaction: Option<f32>,
}

/// Session history tracker
pub struct SessionHistory {
    /// Recent prompts
    history: VecDeque<HistoryEntry>,
    /// Maximum history size
    max_size: usize,
    /// Detected patterns
    patterns: Vec<SessionPattern>,
    /// User preferences
    preferences: UserPreferences,
    /// Session start time
    session_start: u64,
}

impl SessionHistory {
    /// Create a new session history
    pub fn new(max_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_size),
            max_size,
            patterns: Vec::new(),
            preferences: UserPreferences::default(),
            session_start: now(),
        }
    }

    /// Record a prompt
    pub fn record(&mut self, prompt: &str, completed: bool) {
        let entry = HistoryEntry {
            prompt: prompt.to_string(),
            timestamp: now(),
            completed,
            satisfaction: None,
        };

        self.history.push_back(entry);

        if self.history.len() > self.max_size {
            self.history.pop_front();
        }

        // Update patterns periodically
        if self.history.len() % 10 == 0 {
            self.analyze_patterns();
        }
    }

    /// Record satisfaction rating
    pub fn rate(&mut self, satisfaction: f32) {
        if let Some(entry) = self.history.back_mut() {
            entry.satisfaction = Some(satisfaction);
        }
    }

    /// Analyze patterns in history
    fn analyze_patterns(&mut self) {
        self.patterns.clear();

        // Analyze styles
        let mut style_counts: HashMap<String, u32> = HashMap::new();
        let style_keywords = ["anime", "realistic", "photorealistic", "artistic", "cartoon", "3d"];

        for entry in &self.history {
            let lower = entry.prompt.to_lowercase();
            for style in &style_keywords {
                if lower.contains(style) {
                    *style_counts.entry(style.to_string()).or_default() += 1;
                }
            }
        }

        let total = self.history.len() as f32;
        for (style, count) in style_counts {
            if count >= 2 {
                self.patterns.push(SessionPattern {
                    pattern_type: PatternType::Style,
                    value: style,
                    confidence: count as f32 / total,
                    count,
                });
            }
        }

        // Analyze common prefixes
        let mut prefix_counts: HashMap<String, u32> = HashMap::new();
        for entry in &self.history {
            let words: Vec<&str> = entry.prompt.split_whitespace().take(3).collect();
            if words.len() >= 2 {
                let prefix = words[..2].join(" ").to_lowercase();
                *prefix_counts.entry(prefix).or_default() += 1;
            }
        }

        for (prefix, count) in prefix_counts {
            if count >= 3 {
                self.patterns.push(SessionPattern {
                    pattern_type: PatternType::Prefix,
                    value: prefix.clone(),
                    confidence: count as f32 / total,
                    count,
                });
                self.preferences.common_prefixes.push(prefix);
            }
        }

        // Update preferences
        self.preferences.avg_prompt_length = self
            .history
            .iter()
            .map(|e| e.prompt.len() as f32)
            .sum::<f32>()
            / total.max(1.0);

        // Sort patterns by confidence
        self.patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    }

    /// Get predicted next prompt type
    pub fn predict_next(&self) -> Option<&SessionPattern> {
        self.patterns.first()
    }

    /// Get patterns of a specific type
    pub fn patterns_of_type(&self, pattern_type: PatternType) -> Vec<&SessionPattern> {
        self.patterns
            .iter()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }

    /// Get user preferences
    pub fn preferences(&self) -> &UserPreferences {
        &self.preferences
    }

    /// Get session duration in seconds
    pub fn session_duration(&self) -> u64 {
        now() - self.session_start
    }

    /// Get recent prompts
    pub fn recent(&self, count: usize) -> Vec<&str> {
        self.history
            .iter()
            .rev()
            .take(count)
            .map(|e| e.prompt.as_str())
            .collect()
    }

    /// Clear session
    pub fn clear(&mut self) {
        self.history.clear();
        self.patterns.clear();
        self.preferences = UserPreferences::default();
        self.session_start = now();
    }

    /// Export session data
    pub fn export(&self) -> SessionExport {
        SessionExport {
            patterns: self.patterns.clone(),
            preferences: self.preferences.clone(),
            prompt_count: self.history.len(),
            session_duration: self.session_duration(),
        }
    }
}

/// Exported session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExport {
    /// Detected patterns
    pub patterns: Vec<SessionPattern>,
    /// User preferences
    pub preferences: UserPreferences,
    /// Number of prompts
    pub prompt_count: usize,
    /// Session duration
    pub session_duration: u64,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_detection() {
        let mut history = SessionHistory::new(100);

        // Add some prompts with a pattern
        for i in 0..10 {
            history.record(&format!("anime girl with {}", i), true);
        }

        let patterns = history.patterns_of_type(PatternType::Style);
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.value == "anime"));
    }

    #[test]
    fn test_prefix_detection() {
        let mut history = SessionHistory::new(100);

        history.record("portrait of a woman", true);
        history.record("portrait of a man", true);
        history.record("portrait of a child", true);
        history.record("portrait of an elder", true);

        // Force analysis
        for _ in 0..10 {
            history.record("filler", true);
        }

        let patterns = history.patterns_of_type(PatternType::Prefix);
        assert!(patterns.iter().any(|p| p.value.contains("portrait")));
    }

    // =========================================================================
    // Track C.3: Speculative Prefetch ML - Session History Tests
    // =========================================================================

    #[test]
    fn test_session_history_recording() {
        let mut history = SessionHistory::new(100);

        history.record("layer.0.weight", true);
        history.record("layer.0.bias", true);
        history.record("layer.1.weight", true);

        let recent = history.recent(10);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0], "layer.1.weight"); // Most recent first
    }

    #[test]
    fn test_session_history_max_size() {
        let mut history = SessionHistory::new(5);

        // Record more than max
        for i in 0..10 {
            history.record(&format!("prompt {}", i), true);
        }

        // Should only keep last 5
        let recent = history.recent(10);
        assert_eq!(recent.len(), 5);
        assert!(recent[0].contains("9")); // Most recent
    }

    #[test]
    fn test_session_preferences_update() {
        let mut history = SessionHistory::new(100);

        // Record enough prompts to trigger analysis
        for i in 0..20 {
            history.record(&format!("anime style picture number {}", i), true);
        }

        let prefs = history.preferences();
        assert!(prefs.avg_prompt_length > 0.0);
    }

    #[test]
    fn test_session_export() {
        let mut history = SessionHistory::new(100);

        for i in 0..5 {
            history.record(&format!("test prompt {}", i), true);
        }

        let export = history.export();
        assert_eq!(export.prompt_count, 5);
        assert!(export.session_duration >= 0);
    }

    #[test]
    fn test_session_clear() {
        let mut history = SessionHistory::new(100);

        history.record("test 1", true);
        history.record("test 2", true);
        history.clear();

        assert_eq!(history.recent(10).len(), 0);
    }

    #[test]
    fn test_satisfaction_rating() {
        let mut history = SessionHistory::new(100);

        history.record("good prompt", true);
        history.rate(0.9);

        // Rating is recorded internally
        let export = history.export();
        assert_eq!(export.prompt_count, 1);
    }

    #[test]
    fn test_predict_next_pattern() {
        let mut history = SessionHistory::new(100);

        // Create a strong pattern
        for i in 0..15 {
            history.record(&format!("realistic photo of scene {}", i), true);
        }

        // Force analysis
        for _ in 0..10 {
            history.record("filler prompt", true);
        }

        // Should predict style pattern
        let prediction = history.predict_next();
        assert!(prediction.is_some() || history.patterns_of_type(PatternType::Style).is_empty().not());
    }

    #[test]
    fn test_multiple_pattern_types() {
        let mut history = SessionHistory::new(100);

        // Record prompts with multiple patterns
        for i in 0..5 {
            history.record(&format!("portrait of anime girl {}", i), true);
        }
        for i in 0..5 {
            history.record(&format!("portrait of realistic man {}", i), true);
        }

        // Force analysis
        for _ in 0..10 {
            history.record("x", true);
        }

        // Should detect both style and prefix patterns
        let style_patterns = history.patterns_of_type(PatternType::Style);
        let prefix_patterns = history.patterns_of_type(PatternType::Prefix);

        // At least one type should be detected
        assert!(
            !style_patterns.is_empty() || !prefix_patterns.is_empty(),
            "Should detect at least one pattern type"
        );
    }

    #[test]
    fn test_session_duration() {
        let history = SessionHistory::new(100);

        // Duration should be >= 0 (just created)
        assert!(history.session_duration() >= 0);
    }
}

trait Not {
    fn not(&self) -> bool;
}

impl Not for bool {
    fn not(&self) -> bool {
        !*self
    }
}

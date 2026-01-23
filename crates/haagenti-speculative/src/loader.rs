//! Speculative loader coordinating intent prediction and fragment loading

use crate::{
    BufferConfig, BufferEntry, Intent, IntentConfig, IntentPredictor, Result, SessionHistory,
    SpeculationBuffer,
};
use haagenti_fragments::FragmentId;
use haagenti_importance::SemanticCategory;
use haagenti_network::{LoadRequest, NetworkLoader, Priority};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for speculative loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Intent prediction config
    pub intent: IntentConfig,
    /// Buffer config
    pub buffer: BufferConfig,
    /// Maximum concurrent speculative loads
    pub max_concurrent: usize,
    /// Debounce time for keystrokes (ms)
    pub debounce_ms: u64,
    /// Enable session learning
    pub enable_learning: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            intent: IntentConfig::default(),
            buffer: BufferConfig::default(),
            max_concurrent: 4,
            debounce_ms: 100,
            enable_learning: true,
        }
    }
}

/// Event sent to the loader
#[derive(Debug, Clone)]
pub enum LoaderEvent {
    /// Keystroke update
    Keystroke { partial: String },
    /// User committed (pressed enter)
    Commit { prompt: String },
    /// Cancel current speculation
    Cancel,
    /// Clear all
    Clear,
}

/// Speculative loader coordinating all components
pub struct SpeculativeLoader {
    config: LoaderConfig,
    predictor: Arc<RwLock<IntentPredictor>>,
    buffer: Arc<SpeculationBuffer>,
    session: Arc<RwLock<SessionHistory>>,
    network: Arc<NetworkLoader>,
    /// Current intent being speculated
    current_intent: Arc<RwLock<Option<Intent>>>,
    /// Fragment resolver (maps layer patterns to fragment IDs)
    resolver: Arc<dyn FragmentResolver + Send + Sync>,
}

/// Trait for resolving layer patterns to fragment IDs
#[async_trait::async_trait]
pub trait FragmentResolver: Send + Sync {
    /// Resolve a layer pattern to fragment IDs
    async fn resolve(&self, pattern: &str, model_id: &str) -> Vec<FragmentId>;

    /// Get all fragments for a category
    async fn fragments_for_category(
        &self,
        category: SemanticCategory,
        model_id: &str,
    ) -> Vec<FragmentId>;
}

impl SpeculativeLoader {
    /// Create a new speculative loader
    pub fn new(
        config: LoaderConfig,
        network: Arc<NetworkLoader>,
        resolver: Arc<dyn FragmentResolver + Send + Sync>,
    ) -> Self {
        Self {
            predictor: Arc::new(RwLock::new(IntentPredictor::new(config.intent.clone()))),
            buffer: Arc::new(SpeculationBuffer::new(config.buffer.clone())),
            session: Arc::new(RwLock::new(SessionHistory::new(1000))),
            network,
            current_intent: Arc::new(RwLock::new(None)),
            resolver,
            config,
        }
    }

    /// Process a keystroke update
    pub async fn on_keystroke(&self, partial: &str) -> Result<Option<Intent>> {
        let prediction = {
            let predictor = self.predictor.read().await;
            predictor.predict(partial)
        };

        if let Some(intent) = &prediction.primary {
            // Check if intent changed
            let intent_changed = {
                let current = self.current_intent.read().await;
                current
                    .as_ref()
                    .map(|c| c.predicted_prompt != intent.predicted_prompt)
                    .unwrap_or(true)
            };

            if intent_changed {
                // Cancel old speculation
                if let Some(old) = self.current_intent.read().await.as_ref() {
                    self.buffer.cancel_intent(&old.predicted_prompt);
                }

                // Start new speculation if confident enough
                if intent.confidence >= self.config.intent.speculation_threshold {
                    self.start_speculation(intent).await?;
                }

                // Update current intent
                *self.current_intent.write().await = Some(intent.clone());
            } else {
                // Update confidence
                self.buffer
                    .update_confidence(&intent.predicted_prompt, intent.confidence);
            }

            return Ok(Some(intent.clone()));
        }

        Ok(None)
    }

    /// Start speculation for an intent
    async fn start_speculation(&self, intent: &Intent) -> Result<()> {
        let mut fragment_ids = Vec::new();

        // Resolve fragment hints to actual IDs
        for hint in &intent.fragment_hints {
            let ids = self.resolver.resolve(&hint.layer_pattern, "default").await;
            fragment_ids.extend(ids);
        }

        // Also get fragments for detected categories
        for category in &intent.categories {
            let ids = self
                .resolver
                .fragments_for_category(*category, "default")
                .await;
            fragment_ids.extend(ids);
        }

        // Deduplicate
        fragment_ids.sort_by_key(|id| id.0);
        fragment_ids.dedup();

        if fragment_ids.is_empty() {
            debug!(
                "No fragments to speculate for '{}'",
                intent.predicted_prompt
            );
            return Ok(());
        }

        // Add to buffer
        self.buffer.speculate(intent, fragment_ids.clone()).await?;

        // Start loading
        for (idx, fragment_id) in fragment_ids.iter().enumerate() {
            let request = LoadRequest::new(*fragment_id, format!("fragments/{}", fragment_id))
                .with_priority(Priority::Low)
                .with_importance(0.5 - idx as f32 * 0.05);

            let buffer = self.buffer.clone();
            let network = self.network.clone();
            let fragment_id = *fragment_id;

            tokio::spawn(async move {
                match network.load(request).await {
                    haagenti_network::LoadResult::Success { data, .. } => {
                        buffer.mark_ready(&fragment_id, data.len());
                    }
                    haagenti_network::LoadResult::Failed { .. } => {
                        buffer.cancel(&fragment_id);
                    }
                }
            });
        }

        info!(
            "Started speculation for '{}' ({} fragments)",
            intent.predicted_prompt,
            fragment_ids.len()
        );

        Ok(())
    }

    /// User committed to a prompt
    pub async fn on_commit(&self, prompt: &str) -> Vec<BufferEntry> {
        // Record in session
        if self.config.enable_learning {
            self.session.write().await.record(prompt, true);
            self.predictor.write().await.learn(prompt);
        }

        // Check what we speculated
        let entries = self.buffer.entries_for_intent(prompt);

        // Log hit rate
        let ready_count = entries
            .iter()
            .filter(|e| e.state == crate::buffer::EntryState::Ready)
            .count();

        if !entries.is_empty() {
            info!(
                "Commit '{}': {}/{} fragments ready (hit rate: {:.1}%)",
                prompt,
                ready_count,
                entries.len(),
                ready_count as f32 / entries.len() as f32 * 100.0
            );
        }

        // Clear current intent
        *self.current_intent.write().await = None;

        entries
    }

    /// Cancel current speculation
    pub async fn cancel(&self) {
        if let Some(intent) = self.current_intent.write().await.take() {
            self.buffer.cancel_intent(&intent.predicted_prompt);
        }
    }

    /// Check if a fragment is ready
    pub fn is_ready(&self, fragment_id: &FragmentId) -> bool {
        self.buffer.is_ready(fragment_id)
    }

    /// Get a ready fragment
    pub async fn get(&self, fragment_id: &FragmentId) -> Option<BufferEntry> {
        self.buffer.get(fragment_id).await
    }

    /// Get current prediction
    pub async fn current_intent(&self) -> Option<Intent> {
        self.current_intent.read().await.clone()
    }

    /// Get buffer statistics
    pub async fn stats(&self) -> LoaderStats {
        let buffer_stats = self.buffer.stats().await;
        let predictor_stats = self.predictor.read().await.stats();
        let session_export = self.session.read().await.export();

        LoaderStats {
            buffer: buffer_stats,
            known_prompts: predictor_stats.known_prompts,
            session_prompts: session_export.prompt_count,
            session_patterns: session_export.patterns.len(),
        }
    }

    /// Clear all state
    pub async fn clear(&self) {
        self.buffer.clear();
        *self.current_intent.write().await = None;
        self.session.write().await.clear();
    }
}

/// Combined loader statistics
#[derive(Debug, Clone)]
pub struct LoaderStats {
    /// Buffer statistics
    pub buffer: crate::buffer::BufferStats,
    /// Known prompts in predictor
    pub known_prompts: usize,
    /// Prompts in current session
    pub session_prompts: usize,
    /// Patterns detected in session
    pub session_patterns: usize,
}

/// Simple in-memory fragment resolver for testing
pub struct SimpleResolver {
    fragments: std::collections::HashMap<String, Vec<FragmentId>>,
    category_fragments: std::collections::HashMap<SemanticCategory, Vec<FragmentId>>,
}

impl SimpleResolver {
    /// Create a new simple resolver
    pub fn new() -> Self {
        Self {
            fragments: std::collections::HashMap::new(),
            category_fragments: std::collections::HashMap::new(),
        }
    }

    /// Register fragments for a pattern
    pub fn register(&mut self, pattern: &str, ids: Vec<FragmentId>) {
        self.fragments.insert(pattern.to_string(), ids);
    }

    /// Register fragments for a category
    pub fn register_category(&mut self, category: SemanticCategory, ids: Vec<FragmentId>) {
        self.category_fragments.insert(category, ids);
    }
}

#[async_trait::async_trait]
impl FragmentResolver for SimpleResolver {
    async fn resolve(&self, pattern: &str, _model_id: &str) -> Vec<FragmentId> {
        self.fragments.get(pattern).cloned().unwrap_or_default()
    }

    async fn fragments_for_category(
        &self,
        category: SemanticCategory,
        _model_id: &str,
    ) -> Vec<FragmentId> {
        self.category_fragments
            .get(&category)
            .cloned()
            .unwrap_or_default()
    }
}

impl Default for SimpleResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();

        assert_eq!(config.max_concurrent, 4);
        assert_eq!(config.debounce_ms, 100);
        assert!(config.enable_learning);
    }

    #[test]
    fn test_loader_config_custom() {
        let config = LoaderConfig {
            intent: IntentConfig::default(),
            buffer: BufferConfig::default(),
            max_concurrent: 8,
            debounce_ms: 200,
            enable_learning: false,
        };

        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.debounce_ms, 200);
        assert!(!config.enable_learning);
    }

    #[test]
    fn test_loader_event_keystroke() {
        let event = LoaderEvent::Keystroke {
            partial: "test".to_string(),
        };

        match event {
            LoaderEvent::Keystroke { partial } => {
                assert_eq!(partial, "test");
            }
            _ => panic!("Expected Keystroke event"),
        }
    }

    #[test]
    fn test_loader_event_commit() {
        let event = LoaderEvent::Commit {
            prompt: "full prompt".to_string(),
        };

        match event {
            LoaderEvent::Commit { prompt } => {
                assert_eq!(prompt, "full prompt");
            }
            _ => panic!("Expected Commit event"),
        }
    }

    #[test]
    fn test_loader_event_cancel() {
        let event = LoaderEvent::Cancel;

        assert!(matches!(event, LoaderEvent::Cancel));
    }

    #[test]
    fn test_loader_event_clear() {
        let event = LoaderEvent::Clear;

        assert!(matches!(event, LoaderEvent::Clear));
    }

    #[test]
    fn test_simple_resolver_new() {
        let resolver = SimpleResolver::new();

        assert!(resolver.fragments.is_empty());
        assert!(resolver.category_fragments.is_empty());
    }

    #[test]
    fn test_simple_resolver_default() {
        let resolver = SimpleResolver::default();

        assert!(resolver.fragments.is_empty());
        assert!(resolver.category_fragments.is_empty());
    }

    fn make_fragment_id(val: u8) -> FragmentId {
        let mut bytes = [0u8; 16];
        bytes[0] = val;
        FragmentId::new(bytes)
    }

    #[test]
    fn test_simple_resolver_register_pattern() {
        let mut resolver = SimpleResolver::new();
        let ids = vec![make_fragment_id(1), make_fragment_id(2)];

        resolver.register("*.attn*", ids.clone());

        assert_eq!(resolver.fragments.get("*.attn*"), Some(&ids));
    }

    #[test]
    fn test_simple_resolver_register_category() {
        let mut resolver = SimpleResolver::new();
        let ids = vec![make_fragment_id(100), make_fragment_id(200)];

        resolver.register_category(SemanticCategory::Human, ids.clone());

        assert_eq!(
            resolver.category_fragments.get(&SemanticCategory::Human),
            Some(&ids)
        );
    }

    #[tokio::test]
    async fn test_simple_resolver_resolve_existing() {
        let mut resolver = SimpleResolver::new();
        let ids = vec![make_fragment_id(10), make_fragment_id(20)];

        resolver.register("pattern_a", ids.clone());

        let result = resolver.resolve("pattern_a", "model").await;
        assert_eq!(result, ids);
    }

    #[tokio::test]
    async fn test_simple_resolver_resolve_missing() {
        let resolver = SimpleResolver::new();

        let result = resolver.resolve("nonexistent", "model").await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_simple_resolver_fragments_for_category() {
        let mut resolver = SimpleResolver::new();
        let ids = vec![make_fragment_id(50), make_fragment_id(60)];

        resolver.register_category(SemanticCategory::Landscape, ids.clone());

        let result = resolver
            .fragments_for_category(SemanticCategory::Landscape, "model")
            .await;
        assert_eq!(result, ids);
    }

    #[tokio::test]
    async fn test_simple_resolver_category_missing() {
        let resolver = SimpleResolver::new();

        let result = resolver
            .fragments_for_category(SemanticCategory::Abstract, "model")
            .await;
        assert!(result.is_empty());
    }

    #[test]
    fn test_loader_stats_structure() {
        let stats = LoaderStats {
            buffer: crate::buffer::BufferStats {
                total_entries: 10,
                ready_entries: 5,
                loading_entries: 3,
                cancelled_entries: 2,
                total_size: 1024,
                hits: 8,
                misses: 2,
                wasted_bytes: 256,
            },
            known_prompts: 100,
            session_prompts: 25,
            session_patterns: 5,
        };

        assert_eq!(stats.buffer.total_entries, 10);
        assert_eq!(stats.known_prompts, 100);
        assert_eq!(stats.session_prompts, 25);
        assert_eq!(stats.session_patterns, 5);
    }

    #[test]
    fn test_loader_config_intent_inheritance() {
        let config = LoaderConfig::default();

        // Default intent config should have min_chars = 3
        assert_eq!(config.intent.min_chars, 3);
        // Default commit threshold
        assert!(config.intent.commit_threshold > 0.0);
    }

    #[test]
    fn test_loader_config_buffer_inheritance() {
        let config = LoaderConfig::default();

        // Default buffer config should have reasonable max entries
        assert!(config.buffer.max_entries > 0);
    }
}

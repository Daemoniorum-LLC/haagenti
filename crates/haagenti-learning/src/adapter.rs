//! LoRA (Low-Rank Adaptation) adapters

use crate::{LearningError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of low-rank matrices
    pub rank: usize,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Dropout rate
    pub dropout: f32,
    /// Target modules (layer name patterns)
    pub target_modules: Vec<String>,
    /// Fan-in initialization
    pub fan_in_init: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
            ],
            fan_in_init: true,
        }
    }
}

impl LoraConfig {
    /// Get scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Check if layer should have adapter
    pub fn matches(&self, layer_name: &str) -> bool {
        self.target_modules.iter().any(|m| layer_name.contains(m))
    }
}

/// LoRA adapter for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    /// Adapter name
    pub name: String,
    /// Layer this adapter applies to
    pub layer: String,
    /// Low-rank matrix A (in_dim x rank)
    pub lora_a: Vec<f32>,
    /// Low-rank matrix B (rank x out_dim)
    pub lora_b: Vec<f32>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
    /// Rank
    pub rank: usize,
    /// Scaling factor
    pub scaling: f32,
    /// Whether adapter is enabled
    pub enabled: bool,
}

impl LoraAdapter {
    /// Create new LoRA adapter
    pub fn new(
        name: impl Into<String>,
        layer: impl Into<String>,
        in_dim: usize,
        out_dim: usize,
        config: &LoraConfig,
    ) -> Self {
        let rank = config.rank;

        // Initialize A with random normal, B with zeros (as per LoRA paper)
        let lora_a = Self::init_a(in_dim, rank, config.fan_in_init);
        let lora_b = vec![0.0; rank * out_dim];

        Self {
            name: name.into(),
            layer: layer.into(),
            lora_a,
            lora_b,
            in_dim,
            out_dim,
            rank,
            scaling: config.scaling(),
            enabled: true,
        }
    }

    /// Initialize A matrix
    fn init_a(in_dim: usize, rank: usize, fan_in: bool) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();

        let std = if fan_in {
            1.0 / (in_dim as f32).sqrt()
        } else {
            1.0 / (rank as f32).sqrt()
        };

        (0..in_dim * rank)
            .map(|_| rng.gen::<f32>() * std * 2.0 - std)
            .collect()
    }

    /// Forward pass: compute BA @ x
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(LearningError::ShapeMismatch {
                expected: vec![self.in_dim],
                got: vec![x.len()],
            });
        }

        if !self.enabled {
            return Ok(vec![0.0; self.out_dim]);
        }

        // Compute A @ x (in_dim x rank) @ (in_dim,) = (rank,)
        let mut ax = vec![0.0; self.rank];
        for r in 0..self.rank {
            for i in 0..self.in_dim {
                ax[r] += self.lora_a[i * self.rank + r] * x[i];
            }
        }

        // Compute B @ (A @ x) (rank x out_dim) @ (rank,) = (out_dim,)
        let mut result = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            for r in 0..self.rank {
                result[o] += self.lora_b[r * self.out_dim + o] * ax[r];
            }
            result[o] *= self.scaling;
        }

        Ok(result)
    }

    /// Merge adapter into base weights
    pub fn merge(&self, base_weights: &mut [f32]) -> Result<()> {
        if base_weights.len() != self.in_dim * self.out_dim {
            return Err(LearningError::ShapeMismatch {
                expected: vec![self.in_dim * self.out_dim],
                got: vec![base_weights.len()],
            });
        }

        // Compute BA and add to base weights
        for i in 0..self.in_dim {
            for o in 0..self.out_dim {
                let mut delta = 0.0;
                for r in 0..self.rank {
                    delta += self.lora_a[i * self.rank + r] * self.lora_b[r * self.out_dim + o];
                }
                base_weights[i * self.out_dim + o] += delta * self.scaling;
            }
        }

        Ok(())
    }

    /// Number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    /// Get A matrix
    pub fn get_a(&self) -> &[f32] {
        &self.lora_a
    }

    /// Get B matrix
    pub fn get_b(&self) -> &[f32] {
        &self.lora_b
    }

    /// Update A matrix
    pub fn set_a(&mut self, a: Vec<f32>) -> Result<()> {
        if a.len() != self.in_dim * self.rank {
            return Err(LearningError::ShapeMismatch {
                expected: vec![self.in_dim * self.rank],
                got: vec![a.len()],
            });
        }
        self.lora_a = a;
        Ok(())
    }

    /// Update B matrix
    pub fn set_b(&mut self, b: Vec<f32>) -> Result<()> {
        if b.len() != self.rank * self.out_dim {
            return Err(LearningError::ShapeMismatch {
                expected: vec![self.rank * self.out_dim],
                got: vec![b.len()],
            });
        }
        self.lora_b = b;
        Ok(())
    }
}

/// Registry of LoRA adapters
#[derive(Debug, Default)]
pub struct AdapterRegistry {
    /// Adapters by name
    adapters: HashMap<String, LoraAdapter>,
    /// Active adapter name
    active: Option<String>,
}

impl AdapterRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an adapter
    pub fn register(&mut self, adapter: LoraAdapter) {
        let name = adapter.name.clone();
        self.adapters.insert(name.clone(), adapter);

        if self.active.is_none() {
            self.active = Some(name);
        }
    }

    /// Get adapter by name
    pub fn get(&self, name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(name)
    }

    /// Get mutable adapter by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut LoraAdapter> {
        self.adapters.get_mut(name)
    }

    /// Get active adapter
    pub fn active(&self) -> Option<&LoraAdapter> {
        self.active.as_ref().and_then(|name| self.adapters.get(name))
    }

    /// Set active adapter
    pub fn set_active(&mut self, name: &str) -> Result<()> {
        if !self.adapters.contains_key(name) {
            return Err(LearningError::AdapterError(format!(
                "Adapter '{}' not found",
                name
            )));
        }
        self.active = Some(name.to_string());
        Ok(())
    }

    /// List adapter names
    pub fn list(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Total trainable parameters across all adapters
    pub fn total_params(&self) -> usize {
        self.adapters.values().map(|a| a.num_params()).sum()
    }

    /// Remove adapter
    pub fn remove(&mut self, name: &str) -> Option<LoraAdapter> {
        if self.active.as_deref() == Some(name) {
            self.active = None;
        }
        self.adapters.remove(name)
    }

    /// Create adapters for a model based on config
    pub fn create_for_model(
        &mut self,
        adapter_name: &str,
        layers: &[(String, usize, usize)], // (layer_name, in_dim, out_dim)
        config: &LoraConfig,
    ) {
        for (layer_name, in_dim, out_dim) in layers {
            if config.matches(layer_name) {
                let name = format!("{}_{}", adapter_name, layer_name);
                let adapter = LoraAdapter::new(&name, layer_name, *in_dim, *out_dim, config);
                self.register(adapter);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.scaling(), 2.0); // 16 / 8
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoraConfig::default();
        let adapter = LoraAdapter::new("test", "q_proj", 512, 512, &config);

        assert_eq!(adapter.in_dim, 512);
        assert_eq!(adapter.out_dim, 512);
        assert_eq!(adapter.rank, 8);
        assert_eq!(adapter.num_params(), 512 * 8 + 8 * 512);
    }

    #[test]
    fn test_lora_forward() {
        let config = LoraConfig { rank: 2, alpha: 2.0, ..Default::default() };
        let mut adapter = LoraAdapter::new("test", "layer", 4, 3, &config);

        // Set known values
        adapter.lora_a = vec![1.0; 4 * 2];
        adapter.lora_b = vec![1.0; 2 * 3];

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let result = adapter.forward(&x).unwrap();

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_adapter_registry() {
        let mut registry = AdapterRegistry::new();
        let config = LoraConfig::default();

        let adapter1 = LoraAdapter::new("adapter1", "q_proj", 512, 512, &config);
        let adapter2 = LoraAdapter::new("adapter2", "k_proj", 512, 512, &config);

        registry.register(adapter1);
        registry.register(adapter2);

        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("adapter1").is_some());
    }

    #[test]
    fn test_target_module_matching() {
        let config = LoraConfig {
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            ..Default::default()
        };

        assert!(config.matches("model.layers.0.self_attn.q_proj"));
        assert!(config.matches("model.layers.0.self_attn.v_proj"));
        assert!(!config.matches("model.layers.0.self_attn.k_proj"));
    }
}

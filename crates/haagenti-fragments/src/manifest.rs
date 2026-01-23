//! Model manifests for fragment references
//!
//! Each model has a manifest that maps layer names to fragment references.
//! This enables loading models from the shared fragment library.

use crate::{FragmentId, FragmentType, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Reference to a fragment for a specific tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRef {
    /// Fragment ID in the library
    pub fragment_id: FragmentId,
    /// Offset within the fragment (for sub-tensors)
    pub offset: u64,
    /// Length in bytes
    pub length: u64,
    /// Original tensor shape
    pub shape: SmallVec<[u64; 4]>,
    /// Data type
    pub dtype: String,
    /// Fragment type classification
    pub fragment_type: FragmentType,
    /// Quality level (0-255, for progressive loading)
    pub quality_level: u8,
    /// Fragment index for progressive loading order
    pub fragment_index: u32,
}

/// Mapping of layer names to tensor references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMapping {
    /// Layer name in the original model
    pub name: String,
    /// References to fragments (may be multiple for large tensors)
    pub refs: SmallVec<[TensorRef; 1]>,
    /// Total size of the layer
    pub total_size: u64,
    /// Load priority (lower = load first)
    pub priority: u32,
}

/// Model manifest containing all layer-to-fragment mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model identifier
    pub model_id: String,
    /// Model name (human readable)
    pub model_name: String,
    /// Model version/revision
    pub revision: String,
    /// Source format (safetensors, pytorch, etc.)
    pub source_format: String,
    /// Layer mappings (ordered by load priority)
    pub layers: IndexMap<String, LayerMapping>,
    /// Total model size (uncompressed)
    pub total_size: u64,
    /// Total compressed size in library
    pub compressed_size: u64,
    /// Number of unique fragments
    pub unique_fragments: usize,
    /// Number of shared fragments (referenced by other models)
    pub shared_fragments: usize,
    /// Storage savings from deduplication (bytes)
    pub dedup_savings: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Library version this manifest was created for
    pub library_version: u32,
}

impl ModelManifest {
    /// Create a new empty manifest
    pub fn new(
        model_id: impl Into<String>,
        model_name: impl Into<String>,
        revision: impl Into<String>,
        source_format: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_name: model_name.into(),
            revision: revision.into(),
            source_format: source_format.into(),
            layers: IndexMap::new(),
            total_size: 0,
            compressed_size: 0,
            unique_fragments: 0,
            shared_fragments: 0,
            dedup_savings: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            library_version: crate::LIBRARY_FORMAT_VERSION,
        }
    }

    /// Add a layer mapping
    pub fn add_layer(&mut self, mapping: LayerMapping) {
        self.total_size += mapping.total_size;
        self.layers.insert(mapping.name.clone(), mapping);
    }

    /// Get layers sorted by priority
    pub fn layers_by_priority(&self) -> Vec<&LayerMapping> {
        let mut layers: Vec<_> = self.layers.values().collect();
        layers.sort_by_key(|l| l.priority);
        layers
    }

    /// Get layers for progressive loading at a given quality level
    pub fn layers_for_quality(&self, min_quality: u8) -> Vec<&LayerMapping> {
        self.layers
            .values()
            .filter(|l| l.refs.iter().any(|r| r.quality_level >= min_quality))
            .collect()
    }

    /// Get all unique fragment IDs
    pub fn fragment_ids(&self) -> Vec<FragmentId> {
        let mut ids = Vec::new();
        for layer in self.layers.values() {
            for r in &layer.refs {
                if !ids.contains(&r.fragment_id) {
                    ids.push(r.fragment_id);
                }
            }
        }
        ids
    }

    /// Calculate deduplication ratio
    pub fn dedup_ratio(&self) -> f32 {
        if self.total_size == 0 {
            return 1.0;
        }
        1.0 - (self.dedup_savings as f32 / self.total_size as f32)
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.total_size == 0 {
            return 1.0;
        }
        self.compressed_size as f32 / self.total_size as f32
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_builder() {
        let mut builder = ManifestBuilder::new("sdxl-base", "SDXL Base 1.0", "main", "safetensors");

        let frag_id = FragmentId::new([0; 16]);

        builder
            .add_layer(
                "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
                frag_id,
                smallvec::smallvec![320, 320],
                "fp16",
                204800,
                false,
            )
            .set_compressed_size(100000)
            .set_dedup_savings(50000);

        let manifest = builder.build();

        assert_eq!(manifest.layers.len(), 1);
        assert_eq!(manifest.unique_fragments, 1);
        assert_eq!(manifest.shared_fragments, 0);
    }

    #[test]
    fn test_serialize_deserialize() {
        let manifest = ManifestBuilder::new("test", "Test Model", "v1", "safetensors").build();

        let bytes = manifest.to_bytes().unwrap();
        let loaded = ModelManifest::from_bytes(&bytes).unwrap();

        assert_eq!(manifest.model_id, loaded.model_id);
    }
}

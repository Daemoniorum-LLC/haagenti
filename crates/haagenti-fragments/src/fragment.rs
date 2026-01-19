//! Fragment data structures

use arcanum_primitives::prelude::Blake3;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Unique identifier for a fragment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FragmentId(pub [u8; 16]);

impl FragmentId {
    /// Create a new fragment ID from bytes
    pub fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Create from a content hash
    pub fn from_content(data: &[u8]) -> Self {
        let hash = Blake3::hash(data);
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash[..16]);
        Self(id)
    }

    /// Get as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex::decode(s).ok()?;
        if bytes.len() != 16 {
            return None;
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes);
        Some(Self(id))
    }
}

impl std::fmt::Display for FragmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.to_hex()[..8])
    }
}

/// Type of fragment based on neural network layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FragmentType {
    /// Query projection weights (attention)
    AttentionQuery,
    /// Key projection weights (attention)
    AttentionKey,
    /// Value projection weights (attention)
    AttentionValue,
    /// Output projection weights (attention)
    AttentionOutput,
    /// Feed-forward network weights
    FeedForward,
    /// Normalization layer weights
    Normalization,
    /// Embedding weights
    Embedding,
    /// Convolutional weights
    Convolution,
    /// Generic/unknown type
    Generic,
}

impl FragmentType {
    /// Infer fragment type from layer name
    pub fn from_layer_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower.contains("q_proj") || lower.contains("to_q") || lower.contains("query") {
            FragmentType::AttentionQuery
        } else if lower.contains("k_proj") || lower.contains("to_k") || lower.contains("key") {
            FragmentType::AttentionKey
        } else if lower.contains("v_proj") || lower.contains("to_v") || lower.contains("value") {
            FragmentType::AttentionValue
        } else if lower.contains("out_proj") || lower.contains("to_out") || lower.contains("o_proj")
        {
            FragmentType::AttentionOutput
        } else if lower.contains("mlp") || lower.contains("ff") || lower.contains("feed_forward") {
            FragmentType::FeedForward
        } else if lower.contains("norm") || lower.contains("ln") || lower.contains("layer_norm") {
            FragmentType::Normalization
        } else if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") {
            FragmentType::Embedding
        } else if lower.contains("conv") {
            FragmentType::Convolution
        } else {
            FragmentType::Generic
        }
    }

    /// Get sharing potential (0.0 = unique, 1.0 = highly shareable)
    pub fn sharing_potential(&self) -> f32 {
        match self {
            // Q/K projections have similar patterns across models
            FragmentType::AttentionQuery => 0.7,
            FragmentType::AttentionKey => 0.7,
            // V/O projections are more model-specific
            FragmentType::AttentionValue => 0.5,
            FragmentType::AttentionOutput => 0.5,
            // FFN weights are fairly unique
            FragmentType::FeedForward => 0.3,
            // Normalization weights are small but similar
            FragmentType::Normalization => 0.8,
            // Embeddings are highly model-specific
            FragmentType::Embedding => 0.1,
            // Convolutions can share across similar architectures
            FragmentType::Convolution => 0.4,
            FragmentType::Generic => 0.2,
        }
    }
}

/// Metadata about a fragment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentMetadata {
    /// Fragment type
    pub fragment_type: FragmentType,
    /// Original tensor shape
    pub shape: SmallVec<[u64; 4]>,
    /// Data type (fp16, bf16, int8, int4)
    pub dtype: String,
    /// Compression algorithm used
    pub compression: String,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Number of models referencing this fragment
    pub ref_count: u32,
    /// Model names that reference this fragment
    pub models: SmallVec<[String; 4]>,
    /// Quality level (for progressive loading)
    pub quality_level: u8,
}

/// A compressed tensor fragment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    /// Unique identifier
    pub id: FragmentId,
    /// Fragment metadata
    pub metadata: FragmentMetadata,
    /// Compressed data
    pub data: Vec<u8>,
    /// Locality-sensitive hash for similarity search
    pub signature: [u8; 32],
}

impl Fragment {
    /// Create a new fragment
    pub fn new(
        data: Vec<u8>,
        fragment_type: FragmentType,
        shape: impl Into<SmallVec<[u64; 4]>>,
        dtype: impl Into<String>,
        compression: impl Into<String>,
        compression_ratio: f32,
    ) -> Self {
        let id = FragmentId::from_content(&data);
        let signature = Self::compute_signature(&data);

        Self {
            id,
            metadata: FragmentMetadata {
                fragment_type,
                shape: shape.into(),
                dtype: dtype.into(),
                compression: compression.into(),
                compression_ratio,
                ref_count: 1,
                models: SmallVec::new(),
                quality_level: 255, // Full quality
            },
            data,
            signature,
        }
    }

    /// Compute locality-sensitive hash for similarity detection
    fn compute_signature(data: &[u8]) -> [u8; 32] {
        // Use BLAKE3 as base, but we'll enhance this with LSH
        Blake3::hash(data)
    }

    /// Get uncompressed size estimate
    pub fn uncompressed_size(&self) -> usize {
        (self.data.len() as f32 / self.metadata.compression_ratio) as usize
    }

    /// Add a model reference
    pub fn add_model_ref(&mut self, model_name: &str) {
        self.metadata.ref_count += 1;
        if !self.metadata.models.contains(&model_name.to_string()) {
            self.metadata.models.push(model_name.to_string());
        }
    }

    /// Remove a model reference
    pub fn remove_model_ref(&mut self, model_name: &str) -> bool {
        if self.metadata.ref_count > 0 {
            self.metadata.ref_count -= 1;
        }
        self.metadata.models.retain(|m| m != model_name);
        self.metadata.ref_count == 0
    }
}

// Hex encoding helper
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            result.push(HEX_CHARS[(b >> 4) as usize] as char);
            result.push(HEX_CHARS[(b & 0xf) as usize] as char);
        }
        result
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        if s.len() % 2 != 0 {
            return Err(());
        }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
            .collect()
    }
}

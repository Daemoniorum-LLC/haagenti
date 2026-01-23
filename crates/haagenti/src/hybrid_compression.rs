//! Hybrid Compression Pipeline
//!
//! Automatically selects the best compression method (SVD or DCT) based on
//! tensor type and characteristics.
//!
//! ## Algorithm Selection
//!
//! | Tensor Type | Default Method | Reason |
//! |-------------|----------------|--------|
//! | Attention Q/K | DCT | Better quality at same compression |
//! | Attention V/O | DCT | Better quality at same compression |
//! | MLP/FFN | DCT | Much better quality (0.89 vs 0.72) |
//! | Embeddings | DCT | High-frequency patterns |
//! | LayerNorm | None | Keep at full precision |
//!
//! ## Usage
//!
//! ```ignore
//! use haagenti::hybrid_compression::{HybridEncoder, HybridDecoder, CompressionMethod};
//!
//! let encoder = HybridEncoder::new(0.30);  // 30% retention
//!
//! // Auto-select method based on tensor name
//! let compressed = encoder.compress_auto(&data, rows, cols, "model.layers.0.self_attn.q_proj")?;
//!
//! // Or specify method explicitly
//! let compressed = encoder.compress(&data, rows, cols, CompressionMethod::Dct)?;
//!
//! let decoder = HybridDecoder::new();
//! let reconstructed = decoder.decompress(&compressed)?;
//! ```

use crate::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use crate::holotensor::HoloFragment;
use crate::svd_compression::{SvdCompressedWeight, SvdDecoder, SvdEncoder};
use haagenti_core::{Error, Result};

/// Compression method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMethod {
    /// SVD-based low-rank approximation
    Svd,
    /// DCT-based spectral compression
    #[default]
    Dct,
    /// No compression (keep original)
    None,
}

/// Classification of tensor types for compression selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// Attention Q projection
    AttentionQuery,
    /// Attention K projection
    AttentionKey,
    /// Attention V projection
    AttentionValue,
    /// Attention O projection
    AttentionOutput,
    /// Combined QKV projection
    AttentionQkv,
    /// MLP/FFN layers
    Mlp,
    /// Embedding layers
    Embedding,
    /// Layer normalization
    LayerNorm,
    /// Output/LM head
    OutputHead,
    /// Bias vectors
    Bias,
    /// Unknown type
    Unknown,
}

impl TensorType {
    /// Classify a tensor by its name.
    pub fn from_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        // Bias vectors
        if name_lower.contains(".bias") || name_lower.ends_with("_bias") {
            return Self::Bias;
        }

        // Layer normalization
        if name_lower.contains("layernorm")
            || name_lower.contains("layer_norm")
            || name_lower.contains("ln_")
            || name_lower.contains("_ln")
            || name_lower.contains("norm.weight")
        {
            return Self::LayerNorm;
        }

        // Embeddings
        if name_lower.contains("embed_tokens")
            || name_lower.contains("wte")
            || name_lower.contains("word_embed")
            || name_lower.contains("token_embed")
            || name_lower.contains("position_embed")
        {
            return Self::Embedding;
        }

        // Output head
        if name_lower.contains("lm_head") || name_lower.contains("output.weight") {
            return Self::OutputHead;
        }

        // Combined QKV
        if name_lower.contains("qkv") || name_lower.contains("c_attn") {
            return Self::AttentionQkv;
        }

        // Individual attention projections
        if name_lower.contains("q_proj")
            || name_lower.contains(".wq.")
            || name_lower.contains("query")
        {
            return Self::AttentionQuery;
        }
        if name_lower.contains("k_proj")
            || name_lower.contains(".wk.")
            || name_lower.contains("key")
        {
            return Self::AttentionKey;
        }
        if name_lower.contains("v_proj")
            || name_lower.contains(".wv.")
            || name_lower.contains("value")
        {
            return Self::AttentionValue;
        }
        if name_lower.contains("o_proj")
            || name_lower.contains(".wo.")
            || name_lower.contains("dense")
        {
            return Self::AttentionOutput;
        }

        // MLP/FFN
        if name_lower.contains("mlp.")
            || name_lower.contains("feed_forward")
            || name_lower.contains("ffn")
            || name_lower.contains(".fc1")
            || name_lower.contains(".fc2")
            || name_lower.contains("up_proj")
            || name_lower.contains("down_proj")
            || name_lower.contains("gate_proj")
            || name_lower.contains("w1.")
            || name_lower.contains("w2.")
            || name_lower.contains("w3.")
        {
            return Self::Mlp;
        }

        Self::Unknown
    }

    /// Get the recommended compression method for this tensor type.
    pub fn recommended_method(&self) -> CompressionMethod {
        match self {
            // Critical layers: no compression
            Self::LayerNorm | Self::Bias => CompressionMethod::None,

            // All weight matrices: DCT (better quality than SVD at same compression)
            Self::AttentionQuery
            | Self::AttentionKey
            | Self::AttentionValue
            | Self::AttentionOutput
            | Self::AttentionQkv
            | Self::Mlp
            | Self::Embedding
            | Self::OutputHead
            | Self::Unknown => CompressionMethod::Dct,
        }
    }

    /// Get recommended retention/rank ratio for this tensor type.
    pub fn recommended_retention(&self, base_retention: f32) -> f32 {
        match self {
            // Critical: full retention
            Self::LayerNorm | Self::Bias => 1.0,

            // Embedding: slightly higher retention
            Self::Embedding => (base_retention * 1.2).min(1.0),

            // Output head: slightly higher retention
            Self::OutputHead => (base_retention * 1.1).min(1.0),

            // Attention: base retention
            Self::AttentionQuery
            | Self::AttentionKey
            | Self::AttentionValue
            | Self::AttentionOutput
            | Self::AttentionQkv => base_retention,

            // MLP: can be more aggressive
            Self::Mlp => (base_retention * 0.9).max(0.1),

            // Unknown: use base
            Self::Unknown => base_retention,
        }
    }
}

/// Compressed weight representation that can be either SVD or DCT.
#[derive(Debug, Clone)]
pub enum HybridCompressedWeight {
    /// SVD-compressed weight
    Svd(SvdCompressedWeight),
    /// DCT-compressed weight (fragments)
    Dct {
        width: usize,
        height: usize,
        fragments: Vec<HoloFragment>,
    },
    /// Uncompressed (original f32 data)
    None {
        width: usize,
        height: usize,
        data: Vec<f32>,
    },
}

impl HybridCompressedWeight {
    /// Get the compression method used.
    pub fn method(&self) -> CompressionMethod {
        match self {
            Self::Svd(_) => CompressionMethod::Svd,
            Self::Dct { .. } => CompressionMethod::Dct,
            Self::None { .. } => CompressionMethod::None,
        }
    }

    /// Get compressed storage size in bytes.
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::Svd(svd) => svd.storage_bytes(),
            Self::Dct { fragments, .. } => fragments.iter().map(|f| f.data.len()).sum(),
            Self::None { data, .. } => data.len() * 4,
        }
    }

    /// Get original size in bytes.
    pub fn original_bytes(&self) -> usize {
        match self {
            Self::Svd(svd) => svd.original_bytes(),
            Self::Dct { width, height, .. } => width * height * 4,
            Self::None { data, .. } => data.len() * 4,
        }
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let orig = self.original_bytes();
        let compressed = self.storage_bytes();
        if compressed == 0 {
            0.0
        } else {
            orig as f32 / compressed as f32
        }
    }
}

/// Hybrid encoder that can use SVD or DCT.
#[derive(Debug, Clone)]
pub struct HybridEncoder {
    /// Base retention/rank ratio
    base_retention: f32,
    /// Number of fragments for DCT
    num_fragments: u16,
    /// SVD encoder
    svd_encoder: SvdEncoder,
    /// Use automatic method selection based on tensor name
    auto_select: bool,
}

impl Default for HybridEncoder {
    fn default() -> Self {
        Self::new(0.30)
    }
}

impl HybridEncoder {
    /// Create a new hybrid encoder with base retention.
    pub fn new(base_retention: f32) -> Self {
        Self {
            base_retention: base_retention.clamp(0.01, 1.0),
            num_fragments: 8,
            svd_encoder: SvdEncoder::new(base_retention),
            auto_select: true,
        }
    }

    /// Set number of DCT fragments.
    pub fn with_num_fragments(mut self, n: u16) -> Self {
        self.num_fragments = n.max(1);
        self
    }

    /// Enable/disable automatic method selection.
    pub fn with_auto_select(mut self, auto: bool) -> Self {
        self.auto_select = auto;
        self
    }

    /// Compress with automatic method selection based on tensor name.
    pub fn compress_auto(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        tensor_name: &str,
    ) -> Result<HybridCompressedWeight> {
        let tensor_type = TensorType::from_name(tensor_name);
        let method = tensor_type.recommended_method();
        let retention = tensor_type.recommended_retention(self.base_retention);

        self.compress_with_method(data, rows, cols, method, retention)
    }

    /// Compress with explicit method selection.
    pub fn compress(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        method: CompressionMethod,
    ) -> Result<HybridCompressedWeight> {
        self.compress_with_method(data, rows, cols, method, self.base_retention)
    }

    /// Compress with specific method and retention.
    pub fn compress_with_method(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        method: CompressionMethod,
        retention: f32,
    ) -> Result<HybridCompressedWeight> {
        if data.len() != rows * cols {
            return Err(Error::corrupted("data size mismatch"));
        }

        match method {
            CompressionMethod::None => Ok(HybridCompressedWeight::None {
                width: cols,
                height: rows,
                data: data.to_vec(),
            }),

            CompressionMethod::Svd => {
                let encoder = SvdEncoder::new(retention);
                let compressed = encoder.compress(data, rows, cols)?;
                Ok(HybridCompressedWeight::Svd(compressed))
            }

            CompressionMethod::Dct => {
                let encoder = CompressiveSpectralEncoder::new(self.num_fragments, retention);
                let fragments = encoder.encode_2d(data, cols, rows)?;
                Ok(HybridCompressedWeight::Dct {
                    width: cols,
                    height: rows,
                    fragments,
                })
            }
        }
    }

    /// Get the tensor type classification for a name.
    pub fn classify_tensor(&self, name: &str) -> TensorType {
        TensorType::from_name(name)
    }
}

/// Hybrid decoder for compressed weights.
#[derive(Debug, Clone, Default)]
pub struct HybridDecoder {
    svd_decoder: SvdDecoder,
}

impl HybridDecoder {
    /// Create a new hybrid decoder.
    pub fn new() -> Self {
        Self {
            svd_decoder: SvdDecoder::new(),
        }
    }

    /// Decompress a hybrid-compressed weight.
    pub fn decompress(&self, compressed: &HybridCompressedWeight) -> Result<Vec<f32>> {
        match compressed {
            HybridCompressedWeight::None { data, .. } => Ok(data.clone()),

            HybridCompressedWeight::Svd(svd) => self.svd_decoder.decompress(svd),

            HybridCompressedWeight::Dct { fragments, .. } => {
                let mut decoder = CompressiveSpectralDecoder::new();
                decoder.add_essentials(&fragments[0])?;
                for frag in &fragments[1..] {
                    decoder.add_detail(frag)?;
                }
                decoder.reconstruct()
            }
        }
    }
}

/// Compression statistics for a batch of tensors.
#[derive(Debug, Clone, Default)]
pub struct HybridCompressionStats {
    /// Number of tensors processed
    pub tensors_processed: usize,
    /// Tensors by method
    pub svd_count: usize,
    pub dct_count: usize,
    pub none_count: usize,
    /// Total bytes
    pub total_original_bytes: usize,
    pub total_compressed_bytes: usize,
    /// Tensors by type
    pub type_counts: std::collections::HashMap<String, usize>,
}

impl HybridCompressionStats {
    /// Get overall compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        if self.total_compressed_bytes == 0 {
            0.0
        } else {
            self.total_original_bytes as f32 / self.total_compressed_bytes as f32
        }
    }

    /// Record a compressed tensor.
    pub fn record(&mut self, compressed: &HybridCompressedWeight, tensor_type: TensorType) {
        self.tensors_processed += 1;
        self.total_original_bytes += compressed.original_bytes();
        self.total_compressed_bytes += compressed.storage_bytes();

        match compressed.method() {
            CompressionMethod::Svd => self.svd_count += 1,
            CompressionMethod::Dct => self.dct_count += 1,
            CompressionMethod::None => self.none_count += 1,
        }

        let type_name = format!("{:?}", tensor_type);
        *self.type_counts.entry(type_name).or_insert(0) += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_classification() {
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.q_proj.weight"),
            TensorType::AttentionQuery
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.k_proj.weight"),
            TensorType::AttentionKey
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.v_proj.weight"),
            TensorType::AttentionValue
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.o_proj.weight"),
            TensorType::AttentionOutput
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.mlp.gate_proj.weight"),
            TensorType::Mlp
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.mlp.up_proj.weight"),
            TensorType::Mlp
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.mlp.down_proj.weight"),
            TensorType::Mlp
        );
        assert_eq!(
            TensorType::from_name("model.embed_tokens.weight"),
            TensorType::Embedding
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.input_layernorm.weight"),
            TensorType::LayerNorm
        );
        assert_eq!(
            TensorType::from_name("lm_head.weight"),
            TensorType::OutputHead
        );
        assert_eq!(
            TensorType::from_name("model.layers.0.self_attn.q_proj.bias"),
            TensorType::Bias
        );
    }

    #[test]
    fn test_recommended_method() {
        assert_eq!(
            TensorType::AttentionQuery.recommended_method(),
            CompressionMethod::Dct
        );
        assert_eq!(TensorType::Mlp.recommended_method(), CompressionMethod::Dct);
        assert_eq!(
            TensorType::LayerNorm.recommended_method(),
            CompressionMethod::None
        );
        assert_eq!(
            TensorType::Bias.recommended_method(),
            CompressionMethod::None
        );
    }

    #[test]
    fn test_hybrid_encoder_auto() {
        let encoder = HybridEncoder::new(0.30);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        // Attention Q should use DCT
        let compressed = encoder
            .compress_auto(&data, 8, 8, "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(compressed.method(), CompressionMethod::Dct);

        // LayerNorm should use None
        let compressed = encoder
            .compress_auto(&data, 8, 8, "model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(compressed.method(), CompressionMethod::None);
    }

    #[test]
    fn test_hybrid_roundtrip_dct() {
        let encoder = HybridEncoder::new(0.50);
        let decoder = HybridDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let compressed = encoder
            .compress(&data, 8, 8, CompressionMethod::Dct)
            .unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        assert_eq!(reconstructed.len(), data.len());
    }

    #[test]
    fn test_hybrid_roundtrip_svd() {
        let encoder = HybridEncoder::new(0.50);
        let decoder = HybridDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let compressed = encoder
            .compress(&data, 8, 8, CompressionMethod::Svd)
            .unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        assert_eq!(reconstructed.len(), data.len());
    }

    #[test]
    fn test_hybrid_roundtrip_none() {
        let encoder = HybridEncoder::new(0.50);
        let decoder = HybridDecoder::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let compressed = encoder
            .compress(&data, 8, 8, CompressionMethod::None)
            .unwrap();
        let reconstructed = decoder.decompress(&compressed).unwrap();

        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_compression_stats() {
        let encoder = HybridEncoder::new(0.30);
        let mut stats = HybridCompressionStats::default();

        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();

        let compressed1 = encoder
            .compress_auto(&data, 8, 8, "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        stats.record(&compressed1, TensorType::AttentionQuery);

        let compressed2 = encoder
            .compress_auto(&data, 8, 8, "model.layers.0.input_layernorm.weight")
            .unwrap();
        stats.record(&compressed2, TensorType::LayerNorm);

        assert_eq!(stats.tensors_processed, 2);
        assert_eq!(stats.dct_count, 1);
        assert_eq!(stats.none_count, 1);
    }

    #[test]
    fn test_recommended_retention() {
        let base = 0.30;

        // LayerNorm should be 100%
        assert_eq!(TensorType::LayerNorm.recommended_retention(base), 1.0);

        // MLP should be more aggressive
        assert!(TensorType::Mlp.recommended_retention(base) < base);

        // Embeddings should have higher retention
        assert!(TensorType::Embedding.recommended_retention(base) > base);
    }
}

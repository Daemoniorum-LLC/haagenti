//! Neural Compression
//!
//! 10x compression using learned codebooks for model weights.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Neural Compression                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Traditional: weights.safetensors (16GB)                        │
//! │               ↓ LZ4                                              │
//! │               weights.hct (8GB) — 2:1 compression               │
//! │                                                                  │
//! │  Neural:      weights.safetensors (16GB)                        │
//! │               ↓ Learned Encoder                                  │
//! │               weights.nct (1.6GB) — 10:1 compression            │
//! │               ↓ Learned Decoder (GPU)                            │
//! │               weights (reconstructed)                            │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │              Compression Pipeline                         │   │
//! │  │                                                           │   │
//! │  │  Encoder (offline, CPU):                                 │   │
//! │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
//! │  │  │ Weight  │ -> │ Vector  │ -> │ Codebook│ -> indices   │   │
//! │  │  │ Tensor  │    │ Quantize│    │ Lookup  │              │   │
//! │  │  └─────────┘    └─────────┘    └─────────┘              │   │
//! │  │                                                           │   │
//! │  │  Decoder (runtime, GPU):                                 │   │
//! │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
//! │  │  │ Indices │ -> │ Codebook│ -> │ Residual│ -> weights   │   │
//! │  │  │         │    │ Lookup  │    │ Refine  │              │   │
//! │  │  └─────────┘    └─────────┘    └─────────┘              │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Layer-Specific Codebooks
//!
//! Different layer types use different codebook sizes:
//! - Attention Q/K: 4096 centroids, 12-bit indices
//! - Attention V/O: 8192 centroids, 13-bit indices
//! - FFN: 2048 centroids, 11-bit indices
//! - Normalization: 256 centroids, 8-bit (lossless)

mod codebook;
mod decoder;
mod encoder;
mod error;
mod format;
mod residual;
mod training;

pub use codebook::{Codebook, CodebookConfig, CodebookStats, LayerCodebook, LayerType};
pub use decoder::{DecoderConfig, GpuDecoder, NeuralDecoder};
pub use encoder::{EncodedTensor, EncoderConfig, NeuralEncoder};
pub use error::{NeuralError, Result};
pub use format::{NctFile, NctHeader, NctMetadata};
pub use residual::{RefinerConfig, ResidualRefiner};
pub use training::{CodebookTrainer, TrainingConfig};

/// File extension for neural compressed tensors
pub const NCT_EXTENSION: &str = "nct";

/// Magic bytes for NCT format
pub const NCT_MAGIC: [u8; 4] = [0x4E, 0x43, 0x54, 0x00]; // "NCT\0"

/// Default compression ratio target
pub const DEFAULT_COMPRESSION_RATIO: f32 = 10.0;

/// Prelude for common imports
pub mod prelude {
    pub use super::{Codebook, NctFile, NeuralDecoder, NeuralEncoder, Result};
}

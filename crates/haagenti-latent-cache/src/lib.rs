//! Latent Caching for Similar Prompts
//!
//! This module implements intelligent latent caching that enables:
//! - Reusing intermediate latents for similar prompts
//! - Semantic similarity search via CLIP embeddings
//! - Divergence point prediction for optimal cache usage
//!
//! # Key Insight
//!
//! Similar prompts produce similar latents in early denoising steps.
//! "A cat sitting" and "A cat sleeping" diverge only in later steps.
//! By caching latents at key steps, we can skip redundant computation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Latent Cache                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Prompt ──> CLIP Embed ──> HNSW Search ──> Cached Latent    │
//! │                                 │                            │
//! │                                 ↓                            │
//! │                    Divergence Point Predictor                │
//! │                                 │                            │
//! │                                 ↓                            │
//! │              "Start at step 15 (similarity 0.94)"           │
//! │                                                              │
//! │  Storage:                                                    │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │  prompt_hash -> {                                     │   │
//! │  │    embedding: [f32; 768],                            │   │
//! │  │    latents: {                                         │   │
//! │  │      step_5: tensor,                                  │   │
//! │  │      step_10: tensor,                                 │   │
//! │  │      step_15: tensor,                                 │   │
//! │  │    }                                                  │   │
//! │  │  }                                                    │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod cache;
mod divergence;
mod embedding;
mod error;
mod search;
mod storage;

pub use cache::{LatentCache, CacheConfig, CacheEntry, CacheStats};
pub use divergence::{DivergencePredictor, DivergencePoint};
pub use embedding::{EmbeddingProvider, ClipEmbedding};
pub use error::{CacheError, Result};
pub use search::{SimilaritySearch, SearchResult, HnswConfig};
pub use storage::{LatentStorage, StorageConfig, StoredLatent};

/// Default similarity threshold for cache hits
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Default number of latent checkpoints per generation
pub const DEFAULT_CHECKPOINT_COUNT: usize = 4;

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        LatentCache, CacheConfig, DivergencePredictor, EmbeddingProvider,
        SimilaritySearch, SearchResult, Result,
    };
}

//! Cross-Model Fragment Sharing Library
//!
//! This module implements a shared fragment library that enables:
//! - Deduplication of similar tensor fragments across models
//! - Single-load-multi-use for shared attention patterns
//! - Significant storage reduction (30-50% for model families)
//! - Faster model switching by reusing cached fragments
//!
//! # Key Concepts
//!
//! - **Fragment**: A compressed tensor chunk (typically 64KB-1MB)
//! - **Signature**: Locality-sensitive hash for similarity detection
//! - **Library**: Collection of unique fragments with reference tracking
//! - **Manifest**: Per-model mapping from layer names to fragment refs
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Fragment Library                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  FragmentId → Fragment (deduplicated storage)               │
//! │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐               │
//! │  │ frag_0 │ │ frag_1 │ │ frag_2 │ │ frag_3 │  ...          │
//! │  └────────┘ └────────┘ └────────┘ └────────┘               │
//! └─────────────────────────────────────────────────────────────┘
//!                    ↑           ↑           ↑
//!            ┌──────┴───┐   ┌───┴───┐   ┌───┴───┐
//!            │  SDXL    │   │ SD3.5 │   │ Flux  │
//!            │ manifest │   │manifest│   │manifest│
//!            └──────────┘   └───────┘   └───────┘
//! ```


mod error;
mod fragment;
mod library;
mod manifest;
mod signature;
mod similarity;

pub use error::{FragmentError, Result};
pub use fragment::{Fragment, FragmentId, FragmentMetadata, FragmentType};
pub use library::{FragmentLibrary, LibraryConfig, LibraryStats};
pub use manifest::{LayerMapping, ModelManifest, TensorRef};
pub use signature::{FragmentSignature, SignatureConfig};
pub use similarity::{SimilarityIndex, SimilarityMatch, SimilarityThreshold};

/// Version of the fragment library format
pub const LIBRARY_FORMAT_VERSION: u32 = 1;

/// Default similarity threshold for fragment deduplication
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.995;

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        Fragment, FragmentId, FragmentLibrary, FragmentSignature, LibraryConfig, ModelManifest,
        Result, SimilarityIndex, TensorRef,
    };
}

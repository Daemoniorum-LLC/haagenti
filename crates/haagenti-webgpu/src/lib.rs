//! WebGPU Deployment
//!
//! Browser-based inference using WebGPU compute shaders.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Browser Runtime                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │                    Web Worker                               │ │
//! │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
//! │  │  │   Fragment  │  │   WebGPU    │  │    Inference        │ │ │
//! │  │  │   Streamer  │->│   Decoder   │->│    Engine           │ │ │
//! │  │  │   (fetch)   │  │   (WGSL)    │  │    (WGSL)           │ │ │
//! │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! │                              ↓                                   │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │                    Main Thread                              │ │
//! │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
//! │  │  │   Progress  │  │   Canvas    │  │    User             │ │ │
//! │  │  │   UI        │  │   Render    │  │    Controls         │ │ │
//! │  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! │                                                                  │
//! │  ┌────────────────────────────────────────────────────────────┐ │
//! │  │                 Service Worker (Cache)                      │ │
//! │  │  • Fragment cache (IndexedDB, 500MB quota)                 │ │
//! │  │  • Offline-first model loading                              │ │
//! │  │  • Background prefetch                                      │ │
//! │  └────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod buffer;
mod cache;
mod context;
mod error;
mod pipeline;
mod shader;

pub use buffer::{BufferPool, BufferUsage, GpuBuffer};
pub use cache::{CacheConfig, CacheEntry, FragmentCache};
pub use context::{ContextConfig, DeviceCapabilities, WebGpuContext};
pub use error::{Result, WebGpuError};
pub use pipeline::{ComputePipeline, PipelineConfig};
pub use shader::{ShaderModule, WgslSource};

/// Maximum GPU memory target for browser (4GB typical limit)
pub const MAX_GPU_MEMORY_MB: u64 = 4096;

/// IndexedDB storage quota target
pub const STORAGE_QUOTA_MB: u64 = 500;

/// Prelude for common imports
pub mod prelude {
    pub use super::{ComputePipeline, FragmentCache, GpuBuffer, Result, WebGpuContext};
}

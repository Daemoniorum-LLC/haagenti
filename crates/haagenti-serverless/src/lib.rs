//! Serverless deployment with cold start optimization
//!
//! This crate provides optimized serverless deployment with:
//! - Pre-warmed fragment pools for sub-100ms cold starts
//! - GPU memory snapshot/restore for fast instance recovery
//! - Efficient state serialization for function hibernation
//! - Multi-provider support (AWS Lambda, Cloudflare Workers)

mod cold_start;
mod error;
mod fragment_pool;
mod provider;
mod snapshot;
mod state;

pub use cold_start::{
    ColdStartMetrics, ColdStartOptimizer, WarmupConfig, WarmupScheduler, WarmupStats,
};
pub use error::{Result, ServerlessError};
pub use fragment_pool::{FragmentPool, FragmentPrewarmer, PoolConfig, PooledFragment};
pub use provider::{Provider, ProviderCapabilities, ProviderConfig, ProviderType, RequestContext};
pub use snapshot::{GpuSnapshot, SnapshotConfig, SnapshotManager};
pub use state::{FunctionState, StateDiff, StateManager, StateSerializer};

/// Serverless deployment environment
pub mod env {
    /// Check if running in AWS Lambda
    pub fn is_lambda() -> bool {
        std::env::var("AWS_LAMBDA_FUNCTION_NAME").is_ok()
    }

    /// Check if running in Cloudflare Workers
    pub fn is_cloudflare() -> bool {
        // Cloudflare Workers runtime detection
        std::env::var("CF_WORKER").is_ok()
    }

    /// Check if running in any serverless environment
    pub fn is_serverless() -> bool {
        is_lambda() || is_cloudflare()
    }

    /// Get function name
    pub fn function_name() -> Option<String> {
        std::env::var("AWS_LAMBDA_FUNCTION_NAME")
            .ok()
            .or_else(|| std::env::var("CF_WORKER_NAME").ok())
    }

    /// Get memory limit in MB
    pub fn memory_limit_mb() -> Option<u64> {
        std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
    }
}

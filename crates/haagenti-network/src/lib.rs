//! CDN-Based Network Streaming for HoloTensor Fragments
//!
//! This module implements intelligent network streaming for loading
//! model fragments from CDN/edge locations with:
//!
//! - **Range Requests**: Partial downloads for progressive loading
//! - **Parallel Fetching**: Concurrent chunk downloads
//! - **Smart Caching**: ETag/Last-Modified validation
//! - **Retry with Backoff**: Resilient to transient failures
//! - **Priority Queue**: Critical fragments first
//! - **Bandwidth Adaptation**: Adjusts to network conditions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      NetworkLoader                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │   Priority   │ -> │  Scheduler   │ -> │  Fetcher     │      │
//! │  │   Queue      │    │  (rate limit)│    │  (parallel)  │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │         ↑                    ↑                   ↓              │
//! │         │                    │            ┌──────────────┐      │
//! │   Importance           Bandwidth          │    Cache     │      │
//! │    Weights              Monitor           │  (disk+mem)  │      │
//! │                                           └──────────────┘      │
//! │                                                  ↓              │
//! │                                           ┌──────────────┐      │
//! │                                           │   Fragment   │      │
//! │                                           │   Library    │      │
//! │                                           └──────────────┘      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod cache;
mod client;
mod config;
mod error;
mod loader;
mod priority;
mod scheduler;

pub use cache::{FragmentCache, CacheConfig, CacheEntry, CacheStats};
pub use client::{HttpClient, ClientConfig, RangeRequest};
pub use config::{NetworkConfig, CdnEndpoint, RetryConfig};
pub use error::{NetworkError, Result};
pub use loader::{NetworkLoader, LoadRequest, LoadResult, StreamingLoader};
pub use priority::{Priority, PriorityQueue, PrioritizedFragment};
pub use scheduler::{Scheduler, SchedulerConfig, BandwidthMonitor};

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        NetworkLoader, NetworkConfig, CdnEndpoint, LoadRequest, LoadResult,
        Priority, FragmentCache, Result,
    };
}

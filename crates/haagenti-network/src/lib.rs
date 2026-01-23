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

pub use cache::{CacheConfig, CacheEntry, CacheStats, FragmentCache};
pub use client::{ClientConfig, HttpClient, RangeRequest};
pub use config::{CdnEndpoint, NetworkConfig, RetryConfig};
pub use error::{NetworkError, Result};
pub use loader::{LoadRequest, LoadResult, NetworkLoader, StreamingLoader};
pub use priority::{PrioritizedFragment, Priority, PriorityQueue};
pub use scheduler::{BandwidthMonitor, Scheduler, SchedulerConfig};

/// Prelude for common imports
pub mod prelude {
    pub use super::{
        CdnEndpoint, FragmentCache, LoadRequest, LoadResult, NetworkConfig, NetworkLoader,
        Priority, Result,
    };
}

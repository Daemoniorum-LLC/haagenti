//! Network configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// CDN endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnEndpoint {
    /// Base URL for the endpoint
    pub url: String,
    /// Priority (lower = preferred)
    pub priority: u32,
    /// Geographic region (for latency-based selection)
    pub region: Option<String>,
    /// Whether this endpoint supports range requests
    pub supports_range: bool,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Endpoint-specific headers
    pub headers: Vec<(String, String)>,
}

impl CdnEndpoint {
    /// Create a new CDN endpoint
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            priority: 0,
            region: None,
            supports_range: true,
            max_connections: 8,
            headers: Vec::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Add header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Add jitter to backoff
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Network loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// CDN endpoints (ordered by preference)
    pub endpoints: Vec<CdnEndpoint>,
    /// Request timeout
    pub timeout: Duration,
    /// Connect timeout
    pub connect_timeout: Duration,
    /// Maximum concurrent downloads
    pub max_concurrent: usize,
    /// Chunk size for range requests
    pub chunk_size: usize,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Enable compression
    pub compression: bool,
    /// User agent string
    pub user_agent: String,
    /// Cache directory
    pub cache_dir: Option<std::path::PathBuf>,
    /// Maximum cache size (bytes)
    pub max_cache_size: u64,
    /// Enable bandwidth monitoring
    pub monitor_bandwidth: bool,
    /// Minimum acceptable bandwidth (bytes/sec)
    pub min_bandwidth: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            endpoints: Vec::new(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            max_concurrent: 4,
            chunk_size: 1024 * 1024, // 1MB chunks
            retry: RetryConfig::default(),
            compression: true,
            user_agent: format!("haagenti-network/{}", env!("CARGO_PKG_VERSION")),
            cache_dir: None,
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10GB
            monitor_bandwidth: true,
            min_bandwidth: 1024 * 1024, // 1MB/s minimum
        }
    }
}

impl NetworkConfig {
    /// Add a CDN endpoint
    pub fn with_endpoint(mut self, endpoint: CdnEndpoint) -> Self {
        self.endpoints.push(endpoint);
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set maximum concurrent downloads
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Configure for Hugging Face Hub
    pub fn huggingface_hub() -> Self {
        Self::default()
            .with_endpoint(
                CdnEndpoint::new("https://huggingface.co")
                    .with_priority(0)
                    .with_region("global")
            )
            .with_endpoint(
                CdnEndpoint::new("https://cdn-lfs.huggingface.co")
                    .with_priority(1)
                    .with_region("global")
            )
    }

    /// Configure for Civitai
    pub fn civitai() -> Self {
        Self::default()
            .with_endpoint(
                CdnEndpoint::new("https://civitai.com/api/download")
                    .with_priority(0)
            )
    }

    /// Configure for custom CDN
    pub fn custom_cdn(base_url: impl Into<String>) -> Self {
        Self::default()
            .with_endpoint(CdnEndpoint::new(base_url))
    }
}

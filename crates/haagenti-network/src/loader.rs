//! Network loader for fragment streaming

use crate::{
    CacheConfig, CacheEntry, ClientConfig, FragmentCache, HttpClient, NetworkConfig, NetworkError,
    PrioritizedFragment, Priority, RangeRequest, Result, Scheduler, SchedulerConfig,
};
use bytes::Bytes;
use haagenti_fragments::FragmentId;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::warn;

/// Request to load a fragment
#[derive(Debug, Clone)]
pub struct LoadRequest {
    /// Fragment ID
    pub fragment_id: FragmentId,
    /// CDN path
    pub path: String,
    /// Priority
    pub priority: Priority,
    /// Expected size (for progress tracking)
    pub expected_size: Option<u64>,
    /// Importance score (from ML model)
    pub importance: f32,
}

impl LoadRequest {
    /// Create a new load request
    pub fn new(fragment_id: FragmentId, path: impl Into<String>) -> Self {
        Self {
            fragment_id,
            path: path.into(),
            priority: Priority::Normal,
            expected_size: None,
            importance: 0.5,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set expected size
    pub fn with_expected_size(mut self, size: u64) -> Self {
        self.expected_size = Some(size);
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
}

/// Result of loading a fragment
#[derive(Debug)]
pub enum LoadResult {
    /// Successfully loaded
    Success {
        fragment_id: FragmentId,
        data: Bytes,
        duration: Duration,
        from_cache: bool,
    },
    /// Failed to load
    Failed {
        fragment_id: FragmentId,
        error: NetworkError,
    },
}

impl LoadResult {
    /// Check if successful
    pub fn is_success(&self) -> bool {
        matches!(self, LoadResult::Success { .. })
    }

    /// Get fragment ID
    pub fn fragment_id(&self) -> FragmentId {
        match self {
            LoadResult::Success { fragment_id, .. } => *fragment_id,
            LoadResult::Failed { fragment_id, .. } => *fragment_id,
        }
    }
}

/// Network loader for fragment streaming
pub struct NetworkLoader {
    clients: Vec<HttpClient>,
    cache: Option<FragmentCache>,
    scheduler: Scheduler,
}

impl NetworkLoader {
    /// Create a new network loader
    pub async fn new(config: NetworkConfig) -> Result<Self> {
        // Create HTTP clients for each endpoint
        let client_config = ClientConfig::from(&config);
        let mut clients = Vec::new();

        for endpoint in &config.endpoints {
            let client = HttpClient::new(endpoint.clone(), client_config.clone())?;
            clients.push(client);
        }

        if clients.is_empty() {
            return Err(NetworkError::Configuration(
                "No CDN endpoints configured".into(),
            ));
        }

        // Create cache if configured
        let cache = if let Some(ref path) = config.cache_dir {
            let cache_config = CacheConfig {
                path: path.clone(),
                max_size: config.max_cache_size,
                ..Default::default()
            };
            Some(FragmentCache::open(cache_config).await?)
        } else {
            None
        };

        let scheduler = Scheduler::new(SchedulerConfig::from(&config));

        Ok(Self {
            clients,
            cache,
            scheduler,
        })
    }

    /// Load a single fragment
    pub async fn load(&self, request: LoadRequest) -> LoadResult {
        let start = Instant::now();

        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(data) = cache.get(&request.fragment_id).await {
                return LoadResult::Success {
                    fragment_id: request.fragment_id,
                    data,
                    duration: start.elapsed(),
                    from_cache: true,
                };
            }
        }

        // Try each endpoint
        for client in &self.clients {
            match client.fetch(&request.path).await {
                Ok(data) => {
                    let duration = start.elapsed();

                    // Cache the result
                    if let Some(ref cache) = self.cache {
                        let entry = CacheEntry::new(request.fragment_id, data.len() as u64);
                        if let Err(e) = cache.put(request.fragment_id, data.clone(), entry).await {
                            warn!("Failed to cache fragment: {:?}", e);
                        }
                    }

                    // Record bandwidth
                    self.scheduler
                        .record_success(data.len() as u64, duration)
                        .await;

                    return LoadResult::Success {
                        fragment_id: request.fragment_id,
                        data,
                        duration,
                        from_cache: false,
                    };
                }
                Err(e) => {
                    warn!("Endpoint failed: {:?}", e);
                    continue;
                }
            }
        }

        self.scheduler.record_failure();
        LoadResult::Failed {
            fragment_id: request.fragment_id,
            error: NetworkError::RetriesExhausted("All endpoints failed".into()),
        }
    }

    /// Load a range of bytes (for progressive loading)
    pub async fn load_range(&self, request: LoadRequest, start: u64, end: u64) -> LoadResult {
        let range = RangeRequest::new(start, end);
        let start_time = Instant::now();

        for client in &self.clients {
            match client.fetch_range(&request.path, range.clone()).await {
                Ok(data) => {
                    let duration = start_time.elapsed();
                    self.scheduler
                        .record_success(data.len() as u64, duration)
                        .await;

                    return LoadResult::Success {
                        fragment_id: request.fragment_id,
                        data,
                        duration,
                        from_cache: false,
                    };
                }
                Err(e) => {
                    warn!("Range request failed: {:?}", e);
                    continue;
                }
            }
        }

        self.scheduler.record_failure();
        LoadResult::Failed {
            fragment_id: request.fragment_id,
            error: NetworkError::RetriesExhausted("All endpoints failed".into()),
        }
    }

    /// Enqueue requests for background loading
    pub fn enqueue(&self, request: LoadRequest) {
        let prioritized = PrioritizedFragment::new(request.fragment_id, request.priority)
            .with_importance(request.importance)
            .with_size(request.expected_size.unwrap_or(0) as usize);

        self.scheduler.enqueue(prioritized);
    }

    /// Enqueue multiple requests
    pub fn enqueue_many(&self, requests: impl IntoIterator<Item = LoadRequest>) {
        for request in requests {
            self.enqueue(request);
        }
    }

    /// Get scheduler for advanced control
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Get cache for direct access
    pub fn cache(&self) -> Option<&FragmentCache> {
        self.cache.as_ref()
    }

    /// Sync cache to disk
    pub async fn sync(&self) -> Result<()> {
        if let Some(ref cache) = self.cache {
            cache.sync().await?;
        }
        Ok(())
    }
}

/// Streaming loader for continuous fragment loading
pub struct StreamingLoader {
    loader: Arc<NetworkLoader>,
    path_prefix: String,
    rx: mpsc::Receiver<LoadResult>,
    tx: mpsc::Sender<LoadResult>,
}

impl StreamingLoader {
    /// Create a new streaming loader
    pub fn new(loader: Arc<NetworkLoader>, path_prefix: impl Into<String>, buffer: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer);
        Self {
            loader,
            path_prefix: path_prefix.into(),
            rx,
            tx,
        }
    }

    /// Start loading fragments
    pub async fn start(&mut self, requests: Vec<LoadRequest>) {
        for request in requests {
            let loader = self.loader.clone();
            let tx = self.tx.clone();
            let path = format!("{}/{}", self.path_prefix, request.path);
            let request = LoadRequest { path, ..request };

            tokio::spawn(async move {
                let result = loader.load(request).await;
                let _ = tx.send(result).await;
            });
        }
    }

    /// Receive next result
    pub async fn next(&mut self) -> Option<LoadResult> {
        self.rx.recv().await
    }

    /// Receive with timeout
    pub async fn next_timeout(&mut self, timeout: Duration) -> Option<LoadResult> {
        tokio::time::timeout(timeout, self.rx.recv())
            .await
            .ok()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // Integration tests would use wiremock here
}

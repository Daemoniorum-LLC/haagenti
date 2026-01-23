//! HTTP client for CDN communication

use crate::{CdnEndpoint, NetworkConfig, NetworkError, Result, RetryConfig};
use bytes::Bytes;
use reqwest::{header, Client, Response, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{debug, warn};

/// Configuration for the HTTP client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Request timeout
    pub timeout: Duration,
    /// Connect timeout
    pub connect_timeout: Duration,
    /// Enable compression
    pub compression: bool,
    /// User agent
    pub user_agent: String,
    /// Retry configuration
    pub retry: RetryConfig,
}

impl From<&NetworkConfig> for ClientConfig {
    fn from(config: &NetworkConfig) -> Self {
        Self {
            timeout: config.timeout,
            connect_timeout: config.connect_timeout,
            compression: config.compression,
            user_agent: config.user_agent.clone(),
            retry: config.retry.clone(),
        }
    }
}

/// Range request for partial downloads
#[derive(Debug, Clone)]
pub struct RangeRequest {
    /// Start byte
    pub start: u64,
    /// End byte (inclusive)
    pub end: u64,
}

impl RangeRequest {
    /// Create a new range request
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Get the range header value
    pub fn header_value(&self) -> String {
        format!("bytes={}-{}", self.start, self.end)
    }

    /// Get the expected content length
    pub fn content_length(&self) -> u64 {
        self.end - self.start + 1
    }
}

/// HTTP client for CDN requests
pub struct HttpClient {
    client: Client,
    config: ClientConfig,
    endpoint: CdnEndpoint,
    semaphore: Arc<Semaphore>,
}

impl HttpClient {
    /// Create a new HTTP client
    pub fn new(endpoint: CdnEndpoint, config: ClientConfig) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(&config.user_agent)
            .pool_max_idle_per_host(endpoint.max_connections);

        if config.compression {
            builder = builder.gzip(true).brotli(true);
        }

        let client = builder
            .build()
            .map_err(|e| NetworkError::Configuration(e.to_string()))?;
        let semaphore = Arc::new(Semaphore::new(endpoint.max_connections));

        Ok(Self {
            client,
            config,
            endpoint,
            semaphore,
        })
    }

    /// Fetch a fragment by path
    pub async fn fetch(&self, path: &str) -> Result<Bytes> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| NetworkError::Cancelled)?;

        let url = format!(
            "{}/{}",
            self.endpoint.url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        debug!("Fetching: {}", url);

        self.fetch_with_retry(&url, None).await
    }

    /// Fetch a range of bytes
    pub async fn fetch_range(&self, path: &str, range: RangeRequest) -> Result<Bytes> {
        if !self.endpoint.supports_range {
            return Err(NetworkError::Configuration(
                "Endpoint does not support range requests".into(),
            ));
        }

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| NetworkError::Cancelled)?;

        let url = format!(
            "{}/{}",
            self.endpoint.url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        debug!("Fetching range {}-{}: {}", range.start, range.end, url);

        self.fetch_with_retry(&url, Some(range)).await
    }

    /// Fetch with retry logic
    async fn fetch_with_retry(&self, url: &str, range: Option<RangeRequest>) -> Result<Bytes> {
        let mut last_error = NetworkError::Connection("No attempts made".into());
        let mut backoff = self.config.retry.initial_backoff;

        for attempt in 0..=self.config.retry.max_retries {
            if attempt > 0 {
                debug!("Retry attempt {} after {:?}", attempt, backoff);
                tokio::time::sleep(backoff).await;

                // Exponential backoff with jitter
                backoff = Duration::from_secs_f64(
                    (backoff.as_secs_f64() * self.config.retry.multiplier)
                        .min(self.config.retry.max_backoff.as_secs_f64()),
                );

                if self.config.retry.jitter {
                    let jitter = rand::random::<f64>() * 0.3;
                    backoff = Duration::from_secs_f64(backoff.as_secs_f64() * (1.0 + jitter));
                }
            }

            match self.fetch_once(url, range.clone()).await {
                Ok(bytes) => return Ok(bytes),
                Err(e) => {
                    if !e.is_retryable() {
                        return Err(e);
                    }

                    // Check for rate limiting
                    if let Some(retry_after) = e.retry_after() {
                        backoff = retry_after;
                    }

                    warn!("Request failed (attempt {}): {:?}", attempt + 1, e);
                    last_error = e;
                }
            }
        }

        Err(NetworkError::RetriesExhausted(last_error.to_string()))
    }

    /// Single fetch attempt
    async fn fetch_once(&self, url: &str, range: Option<RangeRequest>) -> Result<Bytes> {
        let mut request = self.client.get(url);

        // Add custom headers
        for (key, value) in &self.endpoint.headers {
            request = request.header(key, value);
        }

        // Add range header if specified
        if let Some(ref range) = range {
            request = request.header(header::RANGE, range.header_value());
        }

        let response = request.send().await?;
        self.handle_response(response, range).await
    }

    /// Handle HTTP response
    async fn handle_response(
        &self,
        response: Response,
        range: Option<RangeRequest>,
    ) -> Result<Bytes> {
        let status = response.status();

        match status {
            StatusCode::OK | StatusCode::PARTIAL_CONTENT => {
                // Validate content length for range requests
                if let Some(ref range) = range {
                    if let Some(len) = response.content_length() {
                        if len != range.content_length() {
                            warn!(
                                "Content length mismatch: expected {}, got {}",
                                range.content_length(),
                                len
                            );
                        }
                    }
                }

                response.bytes().await.map_err(|e| e.into())
            }

            StatusCode::NOT_FOUND => Err(NetworkError::NotFound("Fragment not found".into())),

            StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = response
                    .headers()
                    .get(header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(60)
                    * 1000;

                Err(NetworkError::RateLimited {
                    retry_after_ms: retry_after,
                })
            }

            _ => Err(NetworkError::Http {
                status: status.as_u16(),
                message: response.text().await.unwrap_or_default(),
            }),
        }
    }

    /// Get HEAD information (for cache validation)
    pub async fn head(&self, path: &str) -> Result<HeadInfo> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| NetworkError::Cancelled)?;

        let url = format!(
            "{}/{}",
            self.endpoint.url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        let mut request = self.client.head(&url);
        for (key, value) in &self.endpoint.headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(NetworkError::Http {
                status: response.status().as_u16(),
                message: "HEAD request failed".into(),
            });
        }

        let headers = response.headers();

        Ok(HeadInfo {
            content_length: response.content_length(),
            etag: headers
                .get(header::ETAG)
                .and_then(|v| v.to_str().ok())
                .map(String::from),
            last_modified: headers
                .get(header::LAST_MODIFIED)
                .and_then(|v| v.to_str().ok())
                .map(String::from),
            accepts_ranges: headers
                .get(header::ACCEPT_RANGES)
                .and_then(|v| v.to_str().ok())
                .map(|v| v == "bytes")
                .unwrap_or(false),
        })
    }
}

/// HEAD response information
#[derive(Debug, Clone)]
pub struct HeadInfo {
    /// Content length
    pub content_length: Option<u64>,
    /// ETag for cache validation
    pub etag: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<String>,
    /// Whether server accepts range requests
    pub accepts_ranges: bool,
}

// Random number helper
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T: From<f64>>() -> T {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        T::from(nanos as f64 / u32::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_header() {
        let range = RangeRequest::new(0, 1023);
        assert_eq!(range.header_value(), "bytes=0-1023");
        assert_eq!(range.content_length(), 1024);
    }
}

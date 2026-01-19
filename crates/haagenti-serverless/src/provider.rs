//! Multi-provider support for serverless deployment

use crate::{Result, ServerlessError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderType {
    /// AWS Lambda
    AwsLambda,
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Google Cloud Functions
    GoogleCloudFunctions,
    /// Azure Functions
    AzureFunctions,
    /// Custom/Self-hosted
    Custom,
}

impl ProviderType {
    /// Get provider name
    pub fn name(&self) -> &'static str {
        match self {
            ProviderType::AwsLambda => "AWS Lambda",
            ProviderType::CloudflareWorkers => "Cloudflare Workers",
            ProviderType::GoogleCloudFunctions => "Google Cloud Functions",
            ProviderType::AzureFunctions => "Azure Functions",
            ProviderType::Custom => "Custom",
        }
    }

    /// Detect from environment
    pub fn detect() -> Option<Self> {
        if std::env::var("AWS_LAMBDA_FUNCTION_NAME").is_ok() {
            Some(ProviderType::AwsLambda)
        } else if std::env::var("CF_WORKER").is_ok() {
            Some(ProviderType::CloudflareWorkers)
        } else if std::env::var("FUNCTION_NAME").is_ok()
            && std::env::var("GOOGLE_CLOUD_PROJECT").is_ok()
        {
            Some(ProviderType::GoogleCloudFunctions)
        } else if std::env::var("FUNCTIONS_WORKER_RUNTIME").is_ok() {
            Some(ProviderType::AzureFunctions)
        } else {
            None
        }
    }
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    pub provider_type: ProviderType,
    /// Memory limit in MB
    pub memory_mb: u64,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// GPU available
    pub gpu_available: bool,
    /// Maximum payload size in bytes
    pub max_payload_size: u64,
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Custom settings
    pub custom: HashMap<String, String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::Custom,
            memory_mb: 1024,
            timeout_seconds: 30,
            gpu_available: false,
            max_payload_size: 6 * 1024 * 1024, // 6MB
            env_vars: HashMap::new(),
            custom: HashMap::new(),
        }
    }
}

impl ProviderConfig {
    /// Create config for AWS Lambda
    pub fn aws_lambda(memory_mb: u64) -> Self {
        Self {
            provider_type: ProviderType::AwsLambda,
            memory_mb,
            timeout_seconds: 900, // 15 minutes max
            gpu_available: false,
            max_payload_size: 6 * 1024 * 1024,
            ..Default::default()
        }
    }

    /// Create config for Cloudflare Workers
    pub fn cloudflare_workers() -> Self {
        Self {
            provider_type: ProviderType::CloudflareWorkers,
            memory_mb: 128,
            timeout_seconds: 30,
            gpu_available: false,
            max_payload_size: 100 * 1024 * 1024, // 100MB
            ..Default::default()
        }
    }
}

/// Provider abstraction
#[derive(Debug)]
pub struct Provider {
    /// Configuration
    config: ProviderConfig,
    /// Initialized
    initialized: bool,
}

impl Provider {
    /// Create new provider
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    /// Detect and create from environment
    pub fn from_env() -> Result<Self> {
        let provider_type = ProviderType::detect()
            .ok_or_else(|| ServerlessError::ProviderError("Unknown provider".into()))?;

        let config = match provider_type {
            ProviderType::AwsLambda => {
                let memory = std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1024);
                ProviderConfig::aws_lambda(memory)
            }
            ProviderType::CloudflareWorkers => ProviderConfig::cloudflare_workers(),
            _ => ProviderConfig::default(),
        };

        Ok(Self::new(config))
    }

    /// Initialize provider
    pub async fn initialize(&mut self) -> Result<()> {
        // Provider-specific initialization
        match self.config.provider_type {
            ProviderType::AwsLambda => {
                self.init_lambda().await?;
            }
            ProviderType::CloudflareWorkers => {
                self.init_cloudflare().await?;
            }
            _ => {}
        }

        self.initialized = true;
        Ok(())
    }

    async fn init_lambda(&self) -> Result<()> {
        // Lambda-specific initialization
        // - Set up X-Ray tracing
        // - Configure memory allocator
        // - Initialize extensions
        Ok(())
    }

    async fn init_cloudflare(&self) -> Result<()> {
        // Cloudflare-specific initialization
        // - Configure KV access
        // - Set up Workers AI bindings
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get remaining execution time (ms)
    pub fn remaining_time_ms(&self) -> Option<u64> {
        match self.config.provider_type {
            ProviderType::AwsLambda => {
                // Lambda provides this via context
                Some(self.config.timeout_seconds * 1000)
            }
            _ => Some(self.config.timeout_seconds * 1000),
        }
    }

    /// Check if request size is within limits
    pub fn validate_payload_size(&self, size: usize) -> Result<()> {
        if size as u64 > self.config.max_payload_size {
            return Err(ServerlessError::ProviderError(format!(
                "Payload size {} exceeds limit {}",
                size, self.config.max_payload_size
            )));
        }
        Ok(())
    }

    /// Get available memory
    pub fn available_memory_mb(&self) -> u64 {
        self.config.memory_mb
    }

    /// Check GPU availability
    pub fn has_gpu(&self) -> bool {
        self.config.gpu_available
    }
}

/// Provider capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Supports WebSocket
    pub websocket: bool,
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports GPU
    pub gpu: bool,
    /// Supports persistent storage
    pub storage: bool,
    /// Supports scheduled execution
    pub scheduled: bool,
    /// Maximum memory MB
    pub max_memory_mb: u64,
    /// Maximum timeout seconds
    pub max_timeout_seconds: u64,
}

impl ProviderCapabilities {
    /// Get capabilities for provider type
    pub fn for_provider(provider: ProviderType) -> Self {
        match provider {
            ProviderType::AwsLambda => Self {
                websocket: false,
                streaming: true,
                gpu: false,
                storage: false, // Need S3
                scheduled: true,
                max_memory_mb: 10240,
                max_timeout_seconds: 900,
            },
            ProviderType::CloudflareWorkers => Self {
                websocket: true,
                streaming: true,
                gpu: false,
                storage: true, // KV, R2, D1
                scheduled: true,
                max_memory_mb: 128,
                max_timeout_seconds: 30,
            },
            ProviderType::GoogleCloudFunctions => Self {
                websocket: false,
                streaming: false,
                gpu: false,
                storage: false,
                scheduled: true,
                max_memory_mb: 32768,
                max_timeout_seconds: 3600,
            },
            ProviderType::AzureFunctions => Self {
                websocket: false,
                streaming: true,
                gpu: false,
                storage: false,
                scheduled: true,
                max_memory_mb: 14336,
                max_timeout_seconds: 600,
            },
            ProviderType::Custom => Self {
                websocket: true,
                streaming: true,
                gpu: true,
                storage: true,
                scheduled: true,
                max_memory_mb: u64::MAX,
                max_timeout_seconds: u64::MAX,
            },
        }
    }
}

/// Request context from provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    /// Request ID
    pub request_id: String,
    /// Function name
    pub function_name: String,
    /// Invocation count
    pub invocation_count: u64,
    /// Memory limit MB
    pub memory_limit_mb: u64,
    /// Timeout remaining ms
    pub timeout_remaining_ms: u64,
    /// Is cold start
    pub is_cold_start: bool,
}

impl RequestContext {
    /// Create from AWS Lambda context
    pub fn from_lambda_env() -> Option<Self> {
        Some(Self {
            request_id: std::env::var("_X_AMZN_TRACE_ID").ok()?,
            function_name: std::env::var("AWS_LAMBDA_FUNCTION_NAME").ok()?,
            invocation_count: 0, // Not available in env
            memory_limit_mb: std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
                .ok()?
                .parse()
                .ok()?,
            timeout_remaining_ms: 0, // Set from context
            is_cold_start: false,    // Determined at runtime
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_name() {
        assert_eq!(ProviderType::AwsLambda.name(), "AWS Lambda");
        assert_eq!(ProviderType::CloudflareWorkers.name(), "Cloudflare Workers");
    }

    #[test]
    fn test_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.provider_type, ProviderType::Custom);
        assert_eq!(config.memory_mb, 1024);
    }

    #[test]
    fn test_aws_lambda_config() {
        let config = ProviderConfig::aws_lambda(2048);
        assert_eq!(config.provider_type, ProviderType::AwsLambda);
        assert_eq!(config.memory_mb, 2048);
        assert_eq!(config.timeout_seconds, 900);
    }

    #[test]
    fn test_cloudflare_config() {
        let config = ProviderConfig::cloudflare_workers();
        assert_eq!(config.provider_type, ProviderType::CloudflareWorkers);
        assert_eq!(config.memory_mb, 128);
    }

    #[test]
    fn test_provider_creation() {
        let config = ProviderConfig::default();
        let provider = Provider::new(config);

        assert!(!provider.is_initialized());
        assert_eq!(provider.available_memory_mb(), 1024);
    }

    #[test]
    fn test_payload_validation() {
        let config = ProviderConfig {
            max_payload_size: 1024,
            ..Default::default()
        };
        let provider = Provider::new(config);

        assert!(provider.validate_payload_size(512).is_ok());
        assert!(provider.validate_payload_size(2048).is_err());
    }

    #[test]
    fn test_capabilities() {
        let lambda_caps = ProviderCapabilities::for_provider(ProviderType::AwsLambda);
        assert!(!lambda_caps.websocket);
        assert!(lambda_caps.streaming);
        assert_eq!(lambda_caps.max_memory_mb, 10240);

        let cf_caps = ProviderCapabilities::for_provider(ProviderType::CloudflareWorkers);
        assert!(cf_caps.websocket);
        assert!(cf_caps.storage);
    }
}

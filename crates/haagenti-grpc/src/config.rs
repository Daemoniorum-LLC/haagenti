//! Server configuration.

use serde::Deserialize;
use std::path::Path;

use crate::tls::{TlsConfig, TlsResult};

/// Server configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Enable TLS
    #[serde(default)]
    pub tls_enabled: bool,

    /// Path to TLS certificate
    pub tls_cert: Option<String>,

    /// Path to TLS key
    pub tls_key: Option<String>,

    /// Path to client CA certificate (for mTLS)
    pub tls_client_ca: Option<String>,

    /// Require client certificates (mTLS)
    #[serde(default)]
    pub tls_require_client_cert: bool,

    /// Maximum message size (bytes)
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,

    /// Enable Prometheus metrics
    #[serde(default = "default_metrics_enabled")]
    pub metrics_enabled: bool,

    /// Metrics port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            tls_enabled: false,
            tls_cert: None,
            tls_key: None,
            tls_client_ca: None,
            tls_require_client_cert: false,
            max_message_size: default_max_message_size(),
            metrics_enabled: default_metrics_enabled(),
            metrics_port: default_metrics_port(),
            log_level: default_log_level(),
        }
    }
}

impl ServerConfig {
    /// Build a TLS configuration from this server config.
    ///
    /// Returns None if TLS is not enabled or required paths are missing.
    pub fn build_tls_config(&self) -> Option<TlsResult<TlsConfig>> {
        if !self.tls_enabled {
            return None;
        }

        let cert_path = self.tls_cert.as_ref()?;
        let key_path = self.tls_key.as_ref()?;

        let mut config = match TlsConfig::from_pem(Path::new(cert_path), Path::new(key_path)) {
            Ok(c) => c,
            Err(e) => return Some(Err(e)),
        };

        // Add client CA for mTLS if specified
        if let Some(ref client_ca_path) = self.tls_client_ca {
            config = match config.with_client_ca(Path::new(client_ca_path)) {
                Ok(c) => c,
                Err(e) => return Some(Err(e)),
            };
        }

        // Set client cert requirement
        config = config.require_client_cert(self.tls_require_client_cert);

        Some(Ok(config))
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    50051
}

fn default_max_message_size() -> usize {
    64 * 1024 * 1024 // 64MB
}

fn default_metrics_enabled() -> bool {
    true
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_log_level() -> String {
    "info".to_string()
}

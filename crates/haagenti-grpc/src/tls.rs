//! TLS Configuration for Haagenti gRPC Server
//!
//! Provides secure TLS and mutual TLS (mTLS) configuration
//! for production deployments.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use haagenti_grpc::tls::TlsConfig;
//! use std::path::Path;
//!
//! // Basic TLS (server authentication only)
//! let config = TlsConfig::from_pem(
//!     Path::new("server.crt"),
//!     Path::new("server.key"),
//! )?;
//!
//! // Mutual TLS (client + server authentication)
//! let config = TlsConfig::from_pem(
//!     Path::new("server.crt"),
//!     Path::new("server.key"),
//! )?
//! .with_client_ca(Path::new("ca.crt"))?;
//! ```

use std::fs;
use std::path::Path;
use std::sync::Arc;

use tonic::transport::{Certificate, Identity, ServerTlsConfig, ClientTlsConfig};

/// TLS configuration errors
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("Failed to read certificate file: {0}")]
    CertificateRead(#[from] std::io::Error),

    #[error("Invalid certificate format: {0}")]
    InvalidCertificate(String),

    #[error("TLS configuration error: {0}")]
    Configuration(String),
}

/// Result type for TLS operations
pub type TlsResult<T> = Result<T, TlsError>;

/// TLS configuration for secure connections
///
/// Supports both standard TLS (server-side authentication only)
/// and mutual TLS (mTLS) with client certificate verification.
#[derive(Clone)]
pub struct TlsConfig {
    /// Server certificate chain (PEM format)
    cert_chain: Vec<u8>,
    /// Server private key (PEM format)
    private_key: Vec<u8>,
    /// Client CA certificate for mTLS (PEM format)
    client_ca: Option<Vec<u8>>,
    /// Server CA certificate for client verification
    server_ca: Option<Vec<u8>>,
    /// Whether to require client certificates
    require_client_cert: bool,
}

impl std::fmt::Debug for TlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsConfig")
            .field("cert_chain_len", &self.cert_chain.len())
            .field("private_key_len", &self.private_key.len())
            .field("client_ca", &self.client_ca.is_some())
            .field("server_ca", &self.server_ca.is_some())
            .field("require_client_cert", &self.require_client_cert)
            .finish()
    }
}

impl TlsConfig {
    /// Create a TLS configuration from PEM-encoded certificate and key files.
    ///
    /// # Arguments
    /// * `cert_path` - Path to the server certificate file (PEM format)
    /// * `key_path` - Path to the server private key file (PEM format)
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = TlsConfig::from_pem(
    ///     Path::new("/etc/ssl/certs/server.crt"),
    ///     Path::new("/etc/ssl/private/server.key"),
    /// )?;
    /// ```
    pub fn from_pem(cert_path: &Path, key_path: &Path) -> TlsResult<Self> {
        let cert_chain = fs::read(cert_path)?;
        let private_key = fs::read(key_path)?;

        // Basic validation
        if !cert_chain.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "Certificate must be in PEM format".to_string(),
            ));
        }

        if !private_key.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "Private key must be in PEM format".to_string(),
            ));
        }

        Ok(Self {
            cert_chain,
            private_key,
            client_ca: None,
            server_ca: None,
            require_client_cert: false,
        })
    }

    /// Create a TLS configuration from in-memory PEM data.
    ///
    /// Useful for testing or when certificates are loaded from
    /// environment variables or secrets managers.
    pub fn from_pem_bytes(cert_chain: Vec<u8>, private_key: Vec<u8>) -> TlsResult<Self> {
        if !cert_chain.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "Certificate must be in PEM format".to_string(),
            ));
        }

        if !private_key.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "Private key must be in PEM format".to_string(),
            ));
        }

        Ok(Self {
            cert_chain,
            private_key,
            client_ca: None,
            server_ca: None,
            require_client_cert: false,
        })
    }

    /// Add a client CA certificate for mutual TLS (mTLS).
    ///
    /// When set, the server will verify that connecting clients
    /// present certificates signed by this CA.
    ///
    /// # Arguments
    /// * `ca_path` - Path to the CA certificate file (PEM format)
    pub fn with_client_ca(mut self, ca_path: &Path) -> TlsResult<Self> {
        let ca_cert = fs::read(ca_path)?;

        if !ca_cert.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "CA certificate must be in PEM format".to_string(),
            ));
        }

        self.client_ca = Some(ca_cert);
        self.require_client_cert = true;
        Ok(self)
    }

    /// Add a client CA certificate from in-memory PEM data.
    pub fn with_client_ca_bytes(mut self, ca_cert: Vec<u8>) -> TlsResult<Self> {
        if !ca_cert.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "CA certificate must be in PEM format".to_string(),
            ));
        }

        self.client_ca = Some(ca_cert);
        self.require_client_cert = true;
        Ok(self)
    }

    /// Add a server CA certificate for client-side verification.
    ///
    /// Used by clients to verify the server's certificate.
    pub fn with_ca_cert(mut self, ca_path: &Path) -> TlsResult<Self> {
        let ca_cert = fs::read(ca_path)?;

        if !ca_cert.starts_with(b"-----BEGIN") {
            return Err(TlsError::InvalidCertificate(
                "CA certificate must be in PEM format".to_string(),
            ));
        }

        self.server_ca = Some(ca_cert);
        Ok(self)
    }

    /// Set whether client certificates are required (for mTLS).
    ///
    /// When true, connections without valid client certificates
    /// will be rejected.
    pub fn require_client_cert(mut self, require: bool) -> Self {
        self.require_client_cert = require;
        self
    }

    /// Check if this configuration enables mutual TLS.
    pub fn is_mtls(&self) -> bool {
        self.client_ca.is_some() && self.require_client_cert
    }

    /// Get the server certificate chain.
    pub fn cert_chain(&self) -> &[u8] {
        &self.cert_chain
    }

    /// Get the private key (for internal use only).
    pub(crate) fn private_key(&self) -> &[u8] {
        &self.private_key
    }

    /// Build a tonic ServerTlsConfig from this configuration.
    pub fn to_server_tls_config(&self) -> TlsResult<ServerTlsConfig> {
        let identity = Identity::from_pem(&self.cert_chain, &self.private_key);

        let mut config = ServerTlsConfig::new().identity(identity);

        if let Some(ref client_ca) = self.client_ca {
            let ca = Certificate::from_pem(client_ca);
            config = config.client_ca_root(ca);
        }

        Ok(config)
    }

    /// Build a tonic ClientTlsConfig for connecting to TLS servers.
    pub fn to_client_tls_config(&self, domain: &str) -> TlsResult<ClientTlsConfig> {
        let mut config = ClientTlsConfig::new().domain_name(domain);

        // Add server CA if specified
        if let Some(ref server_ca) = self.server_ca {
            let ca = Certificate::from_pem(server_ca);
            config = config.ca_certificate(ca);
        }

        // Add client identity if we have certificates
        let identity = Identity::from_pem(&self.cert_chain, &self.private_key);
        config = config.identity(identity);

        Ok(config)
    }
}

/// Builder for TLS configuration
#[derive(Default)]
pub struct TlsConfigBuilder {
    cert_path: Option<std::path::PathBuf>,
    key_path: Option<std::path::PathBuf>,
    client_ca_path: Option<std::path::PathBuf>,
    server_ca_path: Option<std::path::PathBuf>,
    require_client_cert: bool,
}

impl TlsConfigBuilder {
    /// Create a new TLS configuration builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the server certificate path.
    pub fn cert(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cert_path = Some(path.into());
        self
    }

    /// Set the server private key path.
    pub fn key(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.key_path = Some(path.into());
        self
    }

    /// Set the client CA certificate path (enables mTLS).
    pub fn client_ca(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.client_ca_path = Some(path.into());
        self.require_client_cert = true;
        self
    }

    /// Set the server CA certificate path.
    pub fn server_ca(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.server_ca_path = Some(path.into());
        self
    }

    /// Set whether client certificates are required.
    pub fn require_client_cert(mut self, require: bool) -> Self {
        self.require_client_cert = require;
        self
    }

    /// Build the TLS configuration.
    pub fn build(self) -> TlsResult<TlsConfig> {
        let cert_path = self.cert_path.ok_or_else(|| {
            TlsError::Configuration("Certificate path is required".to_string())
        })?;

        let key_path = self.key_path.ok_or_else(|| {
            TlsError::Configuration("Key path is required".to_string())
        })?;

        let mut config = TlsConfig::from_pem(&cert_path, &key_path)?;

        if let Some(ca_path) = self.client_ca_path {
            config = config.with_client_ca(&ca_path)?;
        }

        if let Some(ca_path) = self.server_ca_path {
            config = config.with_ca_cert(&ca_path)?;
        }

        config = config.require_client_cert(self.require_client_cert);

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // Generate self-signed test certificates for testing
    fn generate_test_certs(dir: &TempDir) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        // Minimal self-signed cert for testing (not cryptographically valid)
        let cert_pem = b"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpMgAAUwMAoGCCqGSM49BAMCMBQxEjAQBgNVBAMMCWxvY2Fs
aG9zdDAeFw0yNTAxMDEwMDAwMDBaFw0yNjAxMDEwMDAwMDBaMBQxEjAQBgNVBAMM
CWxvY2FsaG9zdDBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABC7ksXU7FqKLRtwD
qAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABX
qWJmozEwLzAtBgNVHREEJjAkgglsb2NhbGhvc3SHBH8AAAGHEAAAAAAAAAAAAABp
MAoGCCqGSM49BAMCA0gAMEUCIQC7kpMgAAAAAAAAAAAAAAAAAAA=
-----END CERTIFICATE-----
";

        let key_pem = b"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=
-----END PRIVATE KEY-----
";

        let ca_pem = b"-----BEGIN CERTIFICATE-----
MIIBhTCCASugAwIBAgIJAKHBfpMgAAUxMAoGCCqGSM49BAMCMBIxEDAOBgNVBAMM
B1Rlc3QgQ0EwHhcNMjUwMTAxMDAwMDAwWhcNMjYwMTAxMDAwMDAwWjASMRAwDgYD
VQQDDAdUZXN0IENBMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAELuSxdTsWootG
3AOpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AFepYmajMTAvMC0GA1UdEQQmMCSCCWxvY2FsaG9zdIcEfwAAAYcQAAAAAAAAAAAA
AHAAAAAwCgYIKoZIzj0EAwIDSAAwRQIhALuSkyAAAAAAAAAAAAAAAAAAAAA=
-----END CERTIFICATE-----
";

        let cert_path = dir.path().join("server.crt");
        let key_path = dir.path().join("server.key");
        let ca_path = dir.path().join("ca.crt");

        fs::write(&cert_path, cert_pem).unwrap();
        fs::write(&key_path, key_pem).unwrap();
        fs::write(&ca_path, ca_pem).unwrap();

        (cert_path, key_path, ca_path)
    }

    #[test]
    fn test_tls_config_from_pem() {
        let dir = TempDir::new().unwrap();
        let (cert_path, key_path, _) = generate_test_certs(&dir);

        let config = TlsConfig::from_pem(&cert_path, &key_path).unwrap();

        assert!(!config.cert_chain.is_empty());
        assert!(!config.private_key.is_empty());
        assert!(config.client_ca.is_none());
        assert!(!config.is_mtls());
    }

    #[test]
    fn test_tls_config_with_client_ca() {
        let dir = TempDir::new().unwrap();
        let (cert_path, key_path, ca_path) = generate_test_certs(&dir);

        let config = TlsConfig::from_pem(&cert_path, &key_path)
            .unwrap()
            .with_client_ca(&ca_path)
            .unwrap();

        assert!(config.client_ca.is_some());
        assert!(config.is_mtls());
        assert!(config.require_client_cert);
    }

    #[test]
    fn test_tls_config_invalid_cert() {
        let dir = TempDir::new().unwrap();
        let cert_path = dir.path().join("invalid.crt");
        let key_path = dir.path().join("server.key");

        // Write invalid (non-PEM) data
        fs::write(&cert_path, b"not a pem file").unwrap();
        fs::write(&key_path, b"-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----").unwrap();

        let result = TlsConfig::from_pem(&cert_path, &key_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_builder() {
        let dir = TempDir::new().unwrap();
        let (cert_path, key_path, ca_path) = generate_test_certs(&dir);

        let config = TlsConfigBuilder::new()
            .cert(&cert_path)
            .key(&key_path)
            .client_ca(&ca_path)
            .require_client_cert(true)
            .build()
            .unwrap();

        assert!(config.is_mtls());
    }

    #[test]
    fn test_tls_builder_missing_cert() {
        let result = TlsConfigBuilder::new()
            .key("/path/to/key.pem")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_tls_config_from_bytes() {
        let cert = b"-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----".to_vec();
        let key = b"-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----".to_vec();

        let config = TlsConfig::from_pem_bytes(cert, key).unwrap();
        assert!(!config.is_mtls());
    }

    #[test]
    fn test_to_server_tls_config() {
        let dir = TempDir::new().unwrap();
        let (cert_path, key_path, _) = generate_test_certs(&dir);

        let config = TlsConfig::from_pem(&cert_path, &key_path).unwrap();
        let server_config = config.to_server_tls_config();

        // Should succeed in building config (actual TLS validation happens at runtime)
        assert!(server_config.is_ok());
    }
}

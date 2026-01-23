//! Haagenti gRPC Compression Service
//!
//! High-performance compression service exposing Haagenti's pure Rust
//! compression algorithms over gRPC.
//!
//! ## Features
//!
//! - Multiple algorithms: LZ4, Zstd, Brotli, Deflate/Gzip
//! - One-shot and streaming compression
//! - Dictionary compression for improved ratios
//! - SIMD acceleration when available
//! - Prometheus metrics
//!
//! ## Usage
//!
//! ```bash
//! # Start the server
//! haagenti-server --port 50051
//!
//! # With all algorithms
//! haagenti-server --port 50051 --enable-all-algorithms
//! ```

pub mod proto {
    tonic::include_proto!("haagenti.compression.v1");
}

pub mod config;
pub mod metrics;
pub mod service;
pub mod tls;

pub use config::ServerConfig;
pub use service::CompressionServiceImpl;
pub use tls::{TlsConfig, TlsConfigBuilder, TlsError, TlsResult};

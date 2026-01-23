//! Haagenti gRPC Compression Server
//!
//! High-performance compression service for the Daemoniorum ecosystem.
//!
//! ## Usage
//!
//! ```bash
//! # Start with default settings (port 50051)
//! haagenti-server
//!
//! # Custom port
//! haagenti-server --port 8080
//!
//! # With metrics enabled
//! haagenti-server --port 50051 --metrics-port 9090
//!
//! # Production mode with TLS
//! haagenti-server --port 50051 --tls-cert cert.pem --tls-key key.pem
//! ```

use std::net::SocketAddr;
use std::path::Path;

use clap::Parser;
use tonic::transport::Server;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use haagenti_grpc::proto::compression_service_server::CompressionServiceServer;
use haagenti_grpc::service::CompressionServiceImpl;
use haagenti_grpc::tls::TlsConfig;

#[derive(Parser, Debug)]
#[command(name = "haagenti-server")]
#[command(author = "Daemoniorum LLC")]
#[command(version)]
#[command(about = "Haagenti gRPC Compression Server", long_about = None)]
struct Args {
    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "50051")]
    port: u16,

    /// Enable TLS
    #[arg(long)]
    tls: bool,

    /// Path to TLS certificate
    #[arg(long)]
    tls_cert: Option<String>,

    /// Path to TLS key
    #[arg(long)]
    tls_key: Option<String>,

    /// Path to client CA certificate (for mTLS)
    #[arg(long)]
    tls_client_ca: Option<String>,

    /// Require client certificates (mTLS)
    #[arg(long)]
    tls_require_client_cert: bool,

    /// Maximum message size in MB
    #[arg(long, default_value = "64")]
    max_message_size_mb: usize,

    /// Enable Prometheus metrics
    #[arg(long, default_value = "true")]
    metrics: bool,

    /// Metrics port
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let level = match args.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(true)
        .with_thread_ids(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    // Initialize metrics if enabled
    if args.metrics {
        let metrics_addr: SocketAddr = format!("0.0.0.0:{}", args.metrics_port).parse()?;
        info!("Starting Prometheus metrics server on {}", metrics_addr);

        // Start metrics server in background
        tokio::spawn(async move {
            let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
            builder
                .with_http_listener(metrics_addr)
                .install()
                .expect("Failed to install Prometheus recorder");
        });
    }

    // Create service
    let service = CompressionServiceImpl::new();
    let max_message_size = args.max_message_size_mb * 1024 * 1024;

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║           HAAGENTI COMPRESSION SERVER                        ║");
    info!("║                                                               ║");
    info!("║  The 48th demon, who transmutes substances                    ║");
    info!("╚══════════════════════════════════════════════════════════════╝");
    info!("");
    info!("Starting Haagenti gRPC server...");
    info!("  Address:          {}", addr);
    info!("  Max message size: {} MB", args.max_message_size_mb);
    info!(
        "  TLS:              {}",
        if args.tls { "enabled" } else { "disabled" }
    );
    info!(
        "  Metrics:          {}",
        if args.metrics {
            format!("port {}", args.metrics_port)
        } else {
            "disabled".to_string()
        }
    );
    info!("");
    info!("Available algorithms:");
    info!("  • Zstd (levels 1-22)");
    #[cfg(feature = "lz4")]
    info!("  • LZ4 / LZ4-HC");
    #[cfg(feature = "brotli")]
    info!("  • Brotli");
    #[cfg(feature = "deflate")]
    info!("  • Deflate / Gzip / Zlib");
    info!("");
    info!("Ready to accept connections!");

    // Build and run server
    if args.tls {
        // Validate TLS arguments
        let cert_path = args
            .tls_cert
            .as_ref()
            .ok_or({ "TLS enabled but --tls-cert not provided" })?;
        let key_path = args
            .tls_key
            .as_ref()
            .ok_or({ "TLS enabled but --tls-key not provided" })?;

        // Build TLS configuration
        let mut tls_config = TlsConfig::from_pem(Path::new(cert_path), Path::new(key_path))?;

        // Add client CA for mTLS if specified
        if let Some(ref client_ca_path) = args.tls_client_ca {
            tls_config = tls_config.with_client_ca(Path::new(client_ca_path))?;
            info!(
                "  mTLS:             enabled (client CA: {})",
                client_ca_path
            );
        }

        if args.tls_require_client_cert {
            tls_config = tls_config.require_client_cert(true);
        }

        let server_tls_config = tls_config.to_server_tls_config()?;

        info!("Starting server with TLS enabled");

        let grpc_service = CompressionServiceServer::new(service)
            .max_decoding_message_size(max_message_size)
            .max_encoding_message_size(max_message_size);

        Server::builder()
            .tls_config(server_tls_config)?
            .add_service(grpc_service)
            .serve(addr)
            .await?;
    } else {
        let grpc_service = CompressionServiceServer::new(service)
            .max_decoding_message_size(max_message_size)
            .max_encoding_message_size(max_message_size);

        Server::builder()
            .add_service(grpc_service)
            .serve(addr)
            .await?;
    }

    Ok(())
}

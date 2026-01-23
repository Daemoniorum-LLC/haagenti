//! gRPC service implementation for compression operations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use haagenti_core::{CompressionLevel as HLevel, Compressor, Decompressor};
use haagenti_zstd::{ZstdCodec, ZstdDictCompressor, ZstdDictDecompressor, ZstdDictionary};

use crate::proto::compression_service_server::CompressionService;
use crate::proto::*;

/// Service statistics
pub struct ServiceStats {
    pub start_time: Instant,
    pub total_compressed: AtomicU64,
    pub total_decompressed: AtomicU64,
    pub total_output: AtomicU64,
    pub compress_ops: AtomicU64,
    pub decompress_ops: AtomicU64,
}

impl Default for ServiceStats {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            total_compressed: AtomicU64::new(0),
            total_decompressed: AtomicU64::new(0),
            total_output: AtomicU64::new(0),
            compress_ops: AtomicU64::new(0),
            decompress_ops: AtomicU64::new(0),
        }
    }
}

/// Compression service implementation
pub struct CompressionServiceImpl {
    stats: Arc<ServiceStats>,
}

impl CompressionServiceImpl {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(ServiceStats::default()),
        }
    }

    fn get_compression_level(level: CompressionLevel, custom: i32) -> HLevel {
        if custom > 0 {
            return HLevel::Custom(custom);
        }
        match level {
            CompressionLevel::LevelFast => HLevel::Fast,
            CompressionLevel::LevelDefault => HLevel::Default,
            CompressionLevel::LevelBest => HLevel::Best,
            CompressionLevel::LevelUltra => HLevel::Ultra,
            _ => HLevel::Default,
        }
    }

    fn calculate_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }
}

impl Default for CompressionServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl CompressionService for CompressionServiceImpl {
    async fn compress(
        &self,
        request: Request<CompressRequest>,
    ) -> Result<Response<CompressResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let level = Self::get_compression_level(
            CompressionLevel::try_from(req.level).unwrap_or(CompressionLevel::LevelDefault),
            req.custom_level,
        );

        // Currently only Zstd is fully implemented
        let codec = ZstdCodec::with_level(level);

        let compressed = codec
            .compress(&req.data)
            .map_err(|e| Status::internal(format!("Compression failed: {}", e)))?;

        let elapsed = start.elapsed();

        // Update stats
        self.stats
            .total_compressed
            .fetch_add(req.data.len() as u64, Ordering::Relaxed);
        self.stats
            .total_output
            .fetch_add(compressed.len() as u64, Ordering::Relaxed);
        self.stats.compress_ops.fetch_add(1, Ordering::Relaxed);

        let checksum = if req.calculate_checksum {
            Self::calculate_checksum(&compressed)
        } else {
            String::new()
        };

        let ratio = if !compressed.is_empty() {
            req.data.len() as f64 / compressed.len() as f64
        } else {
            0.0
        };

        Ok(Response::new(CompressResponse {
            compressed_data: compressed,
            original_size: req.data.len() as i64,
            compressed_size: 0, // Set below
            compression_ratio: ratio,
            checksum,
            compression_time_nanos: elapsed.as_nanos() as i64,
        }))
    }

    async fn decompress(
        &self,
        request: Request<DecompressRequest>,
    ) -> Result<Response<DecompressResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        // Verify checksum if requested
        if req.verify_checksum && !req.expected_checksum.is_empty() {
            let actual = Self::calculate_checksum(&req.compressed_data);
            if actual != req.expected_checksum {
                return Err(Status::invalid_argument("Checksum verification failed"));
            }
        }

        let codec = ZstdCodec::new();
        let decompressed = codec
            .decompress(&req.compressed_data)
            .map_err(|e| Status::internal(format!("Decompression failed: {}", e)))?;

        let elapsed = start.elapsed();

        // Update stats
        self.stats
            .total_decompressed
            .fetch_add(decompressed.len() as u64, Ordering::Relaxed);
        self.stats.decompress_ops.fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(DecompressResponse {
            data: decompressed,
            decompressed_size: 0, // Set by response
            decompression_time_nanos: elapsed.as_nanos() as i64,
            checksum_valid: true,
        }))
    }

    type CompressStreamStream =
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<CompressedChunk, Status>> + Send>>;

    async fn compress_stream(
        &self,
        request: Request<Streaming<CompressChunk>>,
    ) -> Result<Response<Self::CompressStreamStream>, Status> {
        let mut stream = request.into_inner();
        let stats = self.stats.clone();

        let output = async_stream::try_stream! {
            let mut buffer = Vec::new();
            let mut total_processed: i64 = 0;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk.data);
                total_processed += chunk.data.len() as i64;

                if chunk.is_last {
                    // Compress accumulated buffer
                    let codec = ZstdCodec::new();
                    let compressed = codec.compress(&buffer)
                        .map_err(|e| Status::internal(format!("Compression failed: {}", e)))?;

                    stats.total_compressed.fetch_add(buffer.len() as u64, Ordering::Relaxed);
                    stats.total_output.fetch_add(compressed.len() as u64, Ordering::Relaxed);
                    stats.compress_ops.fetch_add(1, Ordering::Relaxed);

                    yield CompressedChunk {
                        data: compressed,
                        is_last: true,
                        bytes_processed: total_processed,
                    };
                }
            }
        };

        Ok(Response::new(Box::pin(output)))
    }

    type DecompressStreamStream =
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<DecompressedChunk, Status>> + Send>>;

    async fn decompress_stream(
        &self,
        request: Request<Streaming<CompressedChunk>>,
    ) -> Result<Response<Self::DecompressStreamStream>, Status> {
        let mut stream = request.into_inner();
        let stats = self.stats.clone();

        let output = async_stream::try_stream! {
            let mut buffer = Vec::new();
            let mut total_processed: i64 = 0;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk.data);
                total_processed += chunk.data.len() as i64;

                if chunk.is_last {
                    let codec = ZstdCodec::new();
                    let decompressed = codec.decompress(&buffer)
                        .map_err(|e| Status::internal(format!("Decompression failed: {}", e)))?;

                    stats.total_decompressed.fetch_add(decompressed.len() as u64, Ordering::Relaxed);
                    stats.decompress_ops.fetch_add(1, Ordering::Relaxed);

                    yield DecompressedChunk {
                        data: decompressed,
                        is_last: true,
                        bytes_processed: total_processed,
                    };
                }
            }
        };

        Ok(Response::new(Box::pin(output)))
    }

    async fn train_dictionary(
        &self,
        request: Request<TrainDictionaryRequest>,
    ) -> Result<Response<TrainDictionaryResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let dict_size = if req.dictionary_size > 0 {
            req.dictionary_size as usize
        } else {
            65536 // 64KB default
        };

        // Collect samples
        let samples: Vec<&[u8]> = req.samples.iter().map(|s| s.as_slice()).collect();

        let dictionary = ZstdDictionary::train(&samples, dict_size)
            .map_err(|e| Status::internal(format!("Dictionary training failed: {}", e)))?;

        let elapsed = start.elapsed();

        Ok(Response::new(TrainDictionaryResponse {
            dictionary: dictionary.serialize(),
            dictionary_size: dictionary.size() as i32,
            sample_count: samples.len() as i32,
            training_time_nanos: elapsed.as_nanos() as i64,
        }))
    }

    async fn compress_with_dictionary(
        &self,
        request: Request<CompressWithDictRequest>,
    ) -> Result<Response<CompressResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let dictionary = ZstdDictionary::parse(&req.dictionary)
            .map_err(|e| Status::internal(format!("Failed to parse dictionary: {}", e)))?;

        let compressor = ZstdDictCompressor::new(dictionary);
        let compressed = compressor
            .compress(&req.data)
            .map_err(|e| Status::internal(format!("Compression failed: {}", e)))?;

        let elapsed = start.elapsed();

        self.stats
            .total_compressed
            .fetch_add(req.data.len() as u64, Ordering::Relaxed);
        self.stats
            .total_output
            .fetch_add(compressed.len() as u64, Ordering::Relaxed);
        self.stats.compress_ops.fetch_add(1, Ordering::Relaxed);

        let ratio = if !compressed.is_empty() {
            req.data.len() as f64 / compressed.len() as f64
        } else {
            0.0
        };

        Ok(Response::new(CompressResponse {
            compressed_data: compressed,
            original_size: req.data.len() as i64,
            compressed_size: 0,
            compression_ratio: ratio,
            checksum: String::new(),
            compression_time_nanos: elapsed.as_nanos() as i64,
        }))
    }

    async fn decompress_with_dictionary(
        &self,
        request: Request<DecompressWithDictRequest>,
    ) -> Result<Response<DecompressResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let dictionary = ZstdDictionary::parse(&req.dictionary)
            .map_err(|e| Status::internal(format!("Failed to parse dictionary: {}", e)))?;

        let decompressor = ZstdDictDecompressor::new(dictionary);
        let decompressed = decompressor
            .decompress(&req.compressed_data)
            .map_err(|e| Status::internal(format!("Decompression failed: {}", e)))?;

        let elapsed = start.elapsed();

        self.stats
            .total_decompressed
            .fetch_add(decompressed.len() as u64, Ordering::Relaxed);
        self.stats.decompress_ops.fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(DecompressResponse {
            data: decompressed,
            decompressed_size: 0,
            decompression_time_nanos: elapsed.as_nanos() as i64,
            checksum_valid: true,
        }))
    }

    async fn get_stats(
        &self,
        _request: Request<GetStatsRequest>,
    ) -> Result<Response<CompressionStats>, Status> {
        let total_compressed = self.stats.total_compressed.load(Ordering::Relaxed);
        let total_output = self.stats.total_output.load(Ordering::Relaxed);

        let avg_ratio = if total_output > 0 {
            total_compressed as f64 / total_output as f64
        } else {
            0.0
        };

        Ok(Response::new(CompressionStats {
            total_bytes_compressed: total_compressed as i64,
            total_bytes_decompressed: self.stats.total_decompressed.load(Ordering::Relaxed) as i64,
            total_compressed_output: total_output as i64,
            average_compression_ratio: avg_ratio,
            compression_operations: self.stats.compress_ops.load(Ordering::Relaxed) as i64,
            decompression_operations: self.stats.decompress_ops.load(Ordering::Relaxed) as i64,
            uptime_seconds: self.stats.start_time.elapsed().as_secs() as i64,
            operations_by_algorithm: std::collections::HashMap::new(),
        }))
    }

    async fn measure_compression(
        &self,
        request: Request<MeasureRequest>,
    ) -> Result<Response<MeasureResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let codec = ZstdCodec::new();
        let compressed = codec
            .compress(&req.data)
            .map_err(|e| Status::internal(format!("Compression failed: {}", e)))?;

        let elapsed = start.elapsed();

        let ratio = if !compressed.is_empty() {
            req.data.len() as f64 / compressed.len() as f64
        } else {
            0.0
        };

        let savings = if !req.data.is_empty() {
            100.0 * (1.0 - compressed.len() as f64 / req.data.len() as f64)
        } else {
            0.0
        };

        Ok(Response::new(MeasureResponse {
            original_size: req.data.len() as i64,
            compressed_size: compressed.len() as i64,
            compression_ratio: ratio,
            savings_percent: savings,
            compression_time_nanos: elapsed.as_nanos() as i64,
        }))
    }

    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let req = request.into_inner();

        let mut algorithms = Vec::new();
        if req.include_algorithms {
            algorithms.push(Algorithm::Zstd as i32);
            #[cfg(feature = "lz4")]
            algorithms.push(Algorithm::Lz4 as i32);
            #[cfg(feature = "brotli")]
            algorithms.push(Algorithm::Brotli as i32);
            #[cfg(feature = "deflate")]
            {
                algorithms.push(Algorithm::Gzip as i32);
                algorithms.push(Algorithm::Deflate as i32);
            }
        }

        // Detect SIMD level
        let (simd_available, simd_level) = detect_simd();

        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            available_algorithms: algorithms,
            simd_available,
            simd_level,
        }))
    }
}

fn detect_simd() -> (bool, String) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return (true, "avx512".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            return (true, "avx2".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            return (true, "sse4.2".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64
        return (true, "neon".to_string());
    }

    (false, "none".to_string())
}

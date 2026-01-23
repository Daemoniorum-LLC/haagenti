// Allow explicit indexing in numerical/DSP code where it's clearer
#![allow(clippy::needless_range_loop)]
// Allow from_str methods that don't implement FromStr trait
#![allow(clippy::should_implement_trait)]
// Allow complex types in turbo pipeline - matches batch processing patterns
#![allow(clippy::type_complexity)]
// Testing module uses manual div_ceil for clarity
#![allow(clippy::manual_div_ceil)]
// Testing uses cfg(feature = "half") that may not be in Cargo.toml
#![allow(unexpected_cfgs)]

//! # Haagenti
//!
//! Next-generation compression library for the Daemoniorum ecosystem.
//!
//! Haagenti provides a unified interface for multiple compression algorithms
//! with SIMD acceleration and streaming support.
//!
//! ## Quick Start
//!
//! ```ignore
//! use haagenti::{Codec, Lz4Codec, ZstdCodec};
//!
//! // LZ4 - fastest compression/decompression
//! let lz4 = Lz4Codec::new();
//! let compressed = lz4.compress(b"Hello, Haagenti!")?;
//! let original = lz4.decompress(&compressed)?;
//!
//! // Zstd - balanced speed and ratio
//! let zstd = ZstdCodec::new();
//! let compressed = zstd.compress(b"Hello, Haagenti!")?;
//! ```
//!
//! ## Available Algorithms
//!
//! | Algorithm | Feature | Speed | Ratio | Best For |
//! |-----------|---------|-------|-------|----------|
//! | LZ4 | `lz4` | ⚡⚡⚡ | ⭐⭐ | Real-time, databases |
//! | Zstd | `zstd` | ⚡⚡ | ⭐⭐⭐ | General purpose |
//! | Brotli | `brotli` | ⚡ | ⭐⭐⭐⭐ | Web, static content |
//! | Deflate | `deflate` | ⚡⚡ | ⭐⭐⭐ | Compatibility |
//!
//! ## Feature Flags
//!
//! - `lz4` - LZ4 compression (default)
//! - `zstd` - Zstandard compression (default)
//! - `brotli` - Brotli compression (default)
//! - `deflate` - Deflate/Gzip/Zlib compression (default)
//! - `simd` - SIMD-accelerated primitives (default)
//! - `stream` - Streaming I/O adapters (default)
//! - `full` - All features enabled

// Compressed tensor format for LLM weights
pub mod tensor;
pub use tensor::{
    // Utilities
    compress_file,
    BlockIndex,
    BlockIndexV2,
    ChecksumError,
    // Core types
    CompressionAlgorithm,
    CompressionStats as HctCompressionStats,
    DType,
    HctHeader,
    // Reader/Writer
    HctReader,
    HctReaderV2,
    HctWriter,
    HctWriterV2,
    QuantizationMetadata,
    // V2 types
    QuantizationScheme,
    DEFAULT_BLOCK_SIZE,
    FLAG_BLOCK_CHECKSUMS,
    FLAG_HEADER_CHECKSUM,
    FLAG_HOLOGRAPHIC,
    FLAG_QUANTIZATION,
    FLAG_TENSOR_NAME,
    // Format constants
    HCT_MAGIC,
    HCT_VERSION,
    HCT_VERSION_V2,
};

// Holographic compression for neural network weights
pub mod holotensor;
pub use holotensor::{
    // DCT primitives
    dct_1d,
    dct_2d,
    decode_from_file,
    decode_from_file_progressive,
    encode_to_file,
    idct_1d,
    idct_2d,
    open_holotensor,
    read_holotensor,
    // Convenience functions
    write_holotensor,
    FragmentIndexEntry,
    HoloFragment,
    HoloTensorDecoder,
    // Unified API
    HoloTensorEncoder,
    // Header
    HoloTensorHeader,
    HoloTensorReader,
    // File I/O
    HoloTensorWriter,
    // Core types
    HolographicEncoding,
    LrdfDecoder,
    // Low-Rank Distributed encoder/decoder
    LrdfEncoder,
    QualityCurve,
    RphDecoder,
    // Random Projection encoder/decoder
    RphEncoder,
    // Seeded RNG
    SeededRng,
    SpectralDecoder,
    // Spectral encoder/decoder
    SpectralEncoder,
    HOLO_FLAG_ESSENTIAL_FIRST,
    HOLO_FLAG_FRAGMENT_CHECKSUMS,
    HOLO_FLAG_HEADER_CHECKSUM,
    HOLO_FLAG_INTERLEAVED,
    HOLO_FLAG_QUALITY_CURVE,
    HOLO_FLAG_QUANTIZATION,
    // Format constants
    HOLO_MAGIC,
    HOLO_VERSION,
};

// Compressive spectral encoding for storage-optimized compression
pub mod compressive;
pub use compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};

// Spectral analysis for adaptive retention
pub mod spectral_analysis;
pub use spectral_analysis::{Compressibility, SpectralAnalyzer, SpectralStats};

// Adaptive spectral encoding with per-tensor retention
pub mod adaptive;
pub use adaptive::{
    AdaptiveBatchEncoder, AdaptiveEncodingMeta, AdaptiveSpectralDecoder, AdaptiveSpectralEncoder,
    BatchEncodingStats,
};

// SVD-based compression for attention weights
pub mod svd_compression;
pub use svd_compression::{SvdCompressedWeight, SvdDecoder, SvdEncoder};

// Hybrid compression pipeline (auto-selects SVD or DCT per tensor)
pub mod hybrid_compression;
pub use hybrid_compression::{
    CompressionMethod, HybridCompressedWeight, HybridCompressionStats, HybridDecoder,
    HybridEncoder, TensorType,
};

// Mixed precision compression (FP16 essentials + INT4 details)
pub mod mixed_precision;
pub use mixed_precision::{MixedPrecisionDecoder, MixedPrecisionEncoder, MixedPrecisionWeight};

// Importance-guided compression (training-informed coefficient selection)
pub mod importance;
pub use importance::{
    ImportanceCompressedWeight, ImportanceGuidedDecoder, ImportanceGuidedEncoder, ImportanceMap,
    Sensitivity, TensorImportance,
};

// Streaming decompression for progressive inference loading
pub mod streaming;
pub use streaming::{
    LoadPriority, LoadStatus, ProgressCallback, ProgressiveLoadConfig, StreamingModelLoader,
    StreamingTensorLoader,
};

// Production pipeline for large model compression with checkpointing
pub mod pipeline;
pub use pipeline::{
    CompressionCheckpoint, CompressionConfig, CompressionPipeline, CompressionReport,
    IncrementalHctWriter, PipelineConfig, QualityReport, QualitySampler, QualitySummary,
    ShardReader, ShardStatus, TensorEntry, TensorIndexEntry, TensorResult, TensorStatus,
};

// HCT Specification Test Vectors for formal verification
pub mod hct_test_vectors;
pub use hct_test_vectors::{
    all_stress_vectors, all_test_vectors, cosine_similarity, reference_dct_2d, reference_idct_2d,
    HctTestVector,
};

// Testing utilities (safetensors parsing, quality metrics, INT4 quantization)
// Available when `testing` feature is enabled or during tests
#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Re-export core traits and types
pub use haagenti_core::{
    Algorithm, Codec, CompressionLevel, CompressionStats, Compressor, Decompressor,
    DictionaryCompressor, DictionaryDecompressor, Error, Result,
};

// Re-export LZ4
#[cfg(feature = "lz4")]
pub use haagenti_lz4::{Lz4Codec, Lz4Compressor, Lz4Decompressor};

// Re-export Zstd
#[cfg(feature = "zstd")]
pub use haagenti_zstd::{
    ZstdCodec,
    ZstdCompressor,
    ZstdDecompressor,
    ZstdDictCompressor,
    ZstdDictDecompressor,
    // Dictionary compression support
    ZstdDictionary,
};

// Re-export Zstd compression analysis (entropy fingerprinting)
#[cfg(feature = "zstd")]
pub mod entropy {
    //! Fast entropy fingerprinting for compression decisions.
    //!
    //! Phase 3 optimization: Ultra-fast (~100 cycles) entropy estimation
    //! to skip compression for incompressible data.
    pub use haagenti_zstd::compress::{
        fast_entropy_estimate, fast_predict_block_type, fast_should_compress,
        CompressibilityFingerprint, CompressionStrategy, FastBlockType, PatternType,
    };
}

// Re-export Brotli
#[cfg(feature = "brotli")]
pub use haagenti_brotli::{BrotliCodec, BrotliCompressor, BrotliDecompressor};

// Re-export Deflate/Gzip/Zlib
#[cfg(feature = "deflate")]
pub use haagenti_deflate::{
    DeflateCodec, DeflateCompressor, DeflateDecompressor, GzipCodec, GzipCompressor,
    GzipDecompressor, ZlibCodec, ZlibCompressor, ZlibDecompressor,
};

// Re-export SIMD utilities
#[cfg(feature = "simd")]
pub mod simd {
    //! SIMD-accelerated primitives.
    pub use haagenti_simd::{
        copy_match, detect_simd, fill_repeat, find_match_length, find_match_length_safe, has_avx2,
        has_avx512, has_neon, simd_level, SimdLevel,
    };
}

// Re-export streaming utilities
#[cfg(feature = "stream")]
pub mod stream {
    //! Streaming compression adapters.
    pub use haagenti_stream::{
        clamp_buffer_size, CompressWriter, DecompressReader, ReadAdapter, StreamBuffer,
        WriteAdapter, DEFAULT_BUFFER_SIZE, MAX_BUFFER_SIZE, MIN_BUFFER_SIZE,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// INFERENCE STACK - GPU, Mobile, Distributed, and optimization support for Infernum
// ═══════════════════════════════════════════════════════════════════════════════

// Re-export CUDA GPU acceleration
#[cfg(feature = "cuda")]
pub mod cuda {
    //! CUDA GPU decompression and inference acceleration.
    //!
    //! Provides zero-copy decompression directly to GPU memory with
    //! native kernel support for SM 7.0+ devices.
    pub use haagenti_cuda::{
        device_info,
        is_available,
        AsyncDecompressor,
        BatchDctConfig,
        // Errors
        CudaError,
        DctMode,
        DecompressConfig,
        DecompressStats,
        // Pipelines
        DecompressionPipeline,
        DeviceInfo,
        // Memory management
        GpuBuffer,
        // Core context
        GpuContext,
        // DCT
        GpuDctContext,
        GpuDecompressor,
        // Decompression
        Lz4GpuDecompressor,
        MemoryPool,
        NeuralDecoder,
        NeuralGpuDecoder,
        // Neural GPU
        NeuralGpuPipeline,
        PinnedBuffer,
        PipelineConfig,
        Result,
        StreamingDecoder,
        ZstdGpuDecompressor,
    };
}

// Re-export WebGPU compute shaders (browser/cross-platform)
#[cfg(feature = "webgpu")]
pub mod webgpu {
    //! WebGPU compute shaders for browser-based and cross-platform inference.
    //!
    //! Includes ready-to-use WGSL shaders for transformer operations:
    //! matmul, gelu, softmax, layer_norm, and INT4 dequantization.
    pub use haagenti_webgpu::{
        // Prelude
        prelude,
        BufferPool,
        BufferUsage,
        CacheConfig,
        CacheEntry,
        // Pipelines
        ComputePipeline,
        ContextConfig,
        DeviceCapabilities,
        // Cache
        FragmentCache,
        // Buffers
        GpuBuffer,
        PipelineConfig,
        Result,
        // Shaders
        ShaderModule,
        // Context
        WebGpuContext,
        // Errors
        WebGpuError,
        WgslSource,
        // Constants
        MAX_GPU_MEMORY_MB,
        STORAGE_QUOTA_MB,
    };
}

// Re-export Mobile inference backends (CoreML, NNAPI)
#[cfg(feature = "mobile")]
pub mod mobile {
    //! Mobile inference for iOS (CoreML) and Android (NNAPI).
    //!
    //! Includes thermal management, battery-aware scheduling,
    //! and INT4 quantization for memory efficiency.
    pub use haagenti_mobile::{
        operations,
        // Platform detection
        platform,
        BatchContext,
        CompletionHandler,
        CoreMLConfig,
        CoreMLMetadata,
        CoreMLModel,
        // CoreML (iOS)
        CoreMLRuntime,
        ExecutionContext,
        // Quantization
        Int4Quantizer,
        // Errors
        MobileError,
        // Unified runtime
        MobileRuntime,
        NnapiConfig,
        NnapiModel,
        // NNAPI (Android)
        NnapiRuntime,
        OperationSupport,
        QuantizationConfig,
        QuantizationMetrics,
        QuantizedTensor,
        Result,
        RuntimeConfig,
        RuntimeStats,
        TensorDescription,
        ThermalEvent,
        ThermalHistory,
        // Thermal management
        ThermalManager,
        ThermalPolicy,
        ThermalState,
    };
}

// Re-export distributed inference topologies
#[cfg(feature = "distributed")]
pub mod distributed {
    //! Distributed inference across multiple nodes.
    //!
    //! Supports tensor, pipeline, and expert parallelism with
    //! fault-tolerant execution and ring all-reduce communication.
    pub use haagenti_distributed::{
        // Communication primitives
        comm,
        // Parallelism strategies
        parallelism,
        // Coordination
        Coordinator,
        CoordinatorConfig,
        // Errors
        DistributedError,
        JobStatus,
        Mesh,
        // Protocol
        Message,
        MessageType,
        ModelPartition,
        // Nodes
        Node,
        NodeConfig,
        NodeRole,
        NodeStatus,
        // Partitioning
        PartitionStrategy,
        Protocol,
        Result,
        Ring,
        TensorPartition,
        // Topology
        Topology,
        TopologyConfig,
    };
}

// Re-export serverless cold-start optimization
#[cfg(feature = "serverless")]
pub mod serverless {
    //! Serverless deployment with cold-start optimization.
    //!
    //! Sub-100ms cold starts via pre-warmed fragment pools,
    //! GPU memory snapshots, and efficient state serialization.
    pub use haagenti_serverless::{
        // Environment detection
        env,
        ColdStartMetrics,
        // Cold start
        ColdStartOptimizer,
        // Fragment pools
        FragmentPool,
        FragmentPrewarmer,
        // State management
        FunctionState,
        // Snapshots
        GpuSnapshot,
        PoolConfig,
        PooledFragment,
        // Providers
        Provider,
        ProviderCapabilities,
        ProviderConfig,
        ProviderType,
        RequestContext,
        Result,
        // Errors
        ServerlessError,
        SnapshotConfig,
        SnapshotManager,
        StateDiff,
        StateManager,
        StateSerializer,
        WarmupConfig,
        WarmupScheduler,
        WarmupStats,
    };
}

// Re-export progressive streaming decompression
#[cfg(feature = "inference-streaming")]
pub mod inference_streaming {
    //! Real-time streaming generation with progressive preview.
    //!
    //! Display progressively improving results during generation
    //! with adaptive quality and mid-generation control.
    pub use haagenti_streaming::{
        // Prelude
        prelude,
        // Adaptive
        AdaptiveStreamManager,
        CommandHandler,
        ControlCommand,
        ControlResponse,
        DataFormat,
        DecodedFrame,
        DecoderConfig,
        // Stream management
        GenerationStream,
        MessageType,
        NetworkConditions,
        PreviewBuffer,
        PreviewConfig,
        PreviewData,
        PreviewEvent,
        // Preview
        PreviewFrame,
        PreviewQuality,
        // Scheduling
        PreviewScheduler,
        QualityPolicy,
        RecommendedQuality,
        Result,
        ScheduleMode,
        ScheduleStats,
        StreamConfig,
        // Control
        StreamController,
        // Decoder
        StreamDecoder,
        // Errors
        StreamError,
        // Protocol
        StreamMessage,
        StreamProtocol,
        StreamState,
        // Constants
        DEFAULT_PREVIEW_INTERVAL,
        DEFAULT_THUMBNAIL_SIZE,
        MAX_PREVIEW_LATENCY_MS,
    };
}

// Re-export cross-model fragment sharing
#[cfg(feature = "fragments")]
pub mod fragments {
    //! Cross-model fragment sharing for deduplication.
    //!
    //! Identifies and shares identical or similar fragments across models
    //! for 30-50% storage deduplication.
    pub use haagenti_fragments::{
        // Prelude
        prelude,
        // Fragment types
        Fragment,
        // Errors
        FragmentError,
        FragmentId,
        // Library
        FragmentLibrary,
        FragmentMetadata,
        // Signature
        FragmentSignature,
        FragmentType,
        // Manifest
        LayerMapping,
        LibraryConfig,
        LibraryStats,
        ModelManifest,
        Result,
        SignatureConfig,
        // Similarity
        SimilarityIndex,
        SimilarityMatch,
        SimilarityThreshold,
        TensorRef,
    };
}

// Re-export ML-guided importance scoring
#[cfg(feature = "importance")]
pub mod importance_scoring {
    //! ML-guided fragment importance scoring.
    //!
    //! Uses training data or heuristics to score fragment importance
    //! for prioritized loading during inference.
    pub use haagenti_importance::{
        // Prelude
        prelude,
        AdaptiveScorer,
        // History
        FragmentUsage,
        // Errors
        ImportanceError,
        // Scorer
        ImportanceScore,
        ImportanceScorer,
        LayerProfile,
        // Analyzer
        PromptAnalyzer,
        PromptFeatures,
        // Predictor
        QualityPredictor,
        QualitySensitivity,
        Result,
        ScorerConfig,
        SemanticCategory,
        UsageHistory,
        UsageStats,
    };
}

// Re-export speculative fragment loading
#[cfg(feature = "speculative")]
pub mod speculative {
    //! Speculative fragment prefetching.
    //!
    //! Predicts which fragments will be needed next based on
    //! user input patterns and preloads them.
    pub use haagenti_speculative::{
        // Prelude
        prelude,
        BufferConfig,
        BufferEntry,
        BufferStats,
        Intent,
        IntentConfig,
        // Intent prediction
        IntentPredictor,
        LoaderConfig,
        PredictionResult,
        Result,
        // Session
        SessionHistory,
        SessionPattern,
        // Buffer
        SpeculationBuffer,
        // Errors
        SpeculativeError,
        // Loader
        SpeculativeLoader,
        UserPreferences,
        DEFAULT_COMMIT_THRESHOLD,
        // Constants
        DEFAULT_SPECULATION_THRESHOLD,
    };
}

// Re-export runtime auto-optimization
#[cfg(feature = "autoopt")]
pub mod autoopt {
    //! Self-optimization and auto-tuning.
    //!
    //! Bayesian optimization, genetic algorithms, runtime profiling,
    //! and hardware-aware optimization for peak performance.
    pub use haagenti_autoopt::{
        presets,
        AcquisitionFunction,
        // Auto-tuning
        AutoTuner,
        BayesianConfig,
        // Optimization
        BayesianOptimizer,
        Bottleneck,
        DeviceCapability,
        GeneticConfig,
        GeneticSearch,
        HardwareOptimizer,
        // Hardware
        HardwareProfile,
        // Errors
        OptError,
        OptStrategy,
        ProfileDatabase,
        ProfileResult,
        // Profiling
        Profiler,
        Result,
        ScopedTimer,
        SearchSpace,
        TunerConfig,
        TuningResult,
    };
}

// Re-export online learning and adaptation
#[cfg(feature = "learning")]
pub mod learning {
    //! Continuous learning and online adaptation.
    //!
    //! LoRA adapters, experience replay, elastic weight consolidation,
    //! and progressive layer unfreezing for local model adaptation.
    pub use haagenti_learning::{
        AdapterRegistry,
        BufferConfig,
        EwcConfig,
        // Consolidation
        EwcRegularizer,
        Experience,
        FisherInfo,
        // Errors
        LearningError,
        // Scheduling
        LearningRateScheduler,
        // Strategy
        LearningStrategy,
        // Adapters
        LoraAdapter,
        LoraConfig,
        // Training
        OnlineTrainer,
        ParamGroupScheduler,
        // Replay buffers
        ReplayBuffer,
        ReservoirBuffer,
        Result,
        SchedulerConfig,
        SynapticIntelligence,
        TrainerConfig,
        TrainingStats,
        WarmupScheduler,
    };
}

/// Compress data using the specified algorithm.
///
/// This is a convenience function for one-shot compression.
///
/// # Example
///
/// ```ignore
/// use haagenti::{compress, Algorithm};
///
/// let compressed = compress(b"Hello!", Algorithm::Lz4)?;
/// ```
pub fn compress(data: &[u8], algorithm: Algorithm) -> Result<Vec<u8>> {
    compress_with_level(data, algorithm, CompressionLevel::Default)
}

/// Compress data using the specified algorithm and level.
pub fn compress_with_level(
    data: &[u8],
    algorithm: Algorithm,
    level: CompressionLevel,
) -> Result<Vec<u8>> {
    match algorithm {
        #[cfg(feature = "lz4")]
        Algorithm::Lz4 => Lz4Codec::with_level(level).compress(data),

        #[cfg(feature = "zstd")]
        Algorithm::Zstd => ZstdCodec::with_level(level).compress(data),

        #[cfg(feature = "brotli")]
        Algorithm::Brotli => BrotliCodec::with_level(level).compress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Deflate => DeflateCodec::with_level(level).compress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Gzip => GzipCodec::with_level(level).compress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Zlib => ZlibCodec::with_level(level).compress(data),

        #[allow(unreachable_patterns)]
        _ => Err(Error::algorithm("unsupported", "algorithm not enabled")),
    }
}

/// Decompress data using the specified algorithm.
///
/// # Example
///
/// ```ignore
/// use haagenti::{decompress, Algorithm};
///
/// let original = decompress(&compressed, Algorithm::Lz4)?;
/// ```
pub fn decompress(data: &[u8], algorithm: Algorithm) -> Result<Vec<u8>> {
    match algorithm {
        #[cfg(feature = "lz4")]
        Algorithm::Lz4 => Lz4Codec::new().decompress(data),

        #[cfg(feature = "zstd")]
        Algorithm::Zstd => ZstdCodec::new().decompress(data),

        #[cfg(feature = "brotli")]
        Algorithm::Brotli => BrotliCodec::new().decompress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Deflate => DeflateCodec::new().decompress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Gzip => GzipCodec::new().decompress(data),

        #[cfg(feature = "deflate")]
        Algorithm::Zlib => ZlibCodec::new().decompress(data),

        #[allow(unreachable_patterns)]
        _ => Err(Error::algorithm("unsupported", "algorithm not enabled")),
    }
}

/// Auto-detect algorithm from compressed data header.
///
/// Returns `None` if the format cannot be detected.
pub fn detect_algorithm(data: &[u8]) -> Option<Algorithm> {
    if data.len() < 2 {
        return None;
    }

    // Gzip magic: 1f 8b
    if data[0] == 0x1f && data[1] == 0x8b {
        return Some(Algorithm::Gzip);
    }

    // Zlib: first byte indicates compression method
    // CMF = 0x78 (deflate with 32K window) is most common
    if data[0] == 0x78 && (data[1] == 0x01 || data[1] == 0x5e || data[1] == 0x9c || data[1] == 0xda)
    {
        return Some(Algorithm::Zlib);
    }

    // Zstd magic: 28 b5 2f fd
    if data.len() >= 4 && data[0] == 0x28 && data[1] == 0xb5 && data[2] == 0x2f && data[3] == 0xfd {
        return Some(Algorithm::Zstd);
    }

    // LZ4 frame magic: 04 22 4d 18
    if data.len() >= 4 && data[0] == 0x04 && data[1] == 0x22 && data[2] == 0x4d && data[3] == 0x18 {
        return Some(Algorithm::Lz4);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "lz4")]
    fn test_lz4_codec_roundtrip() {
        // LZ4 block format requires using compress_with_size/decompress_sized
        let codec = Lz4Codec::new();
        let data = b"Hello, Haagenti! This is a test of LZ4 compression.";
        let (compressed, size) = codec.compress_with_size(data).unwrap();
        let decompressed = codec.decompress_sized(&compressed, size).unwrap();
        assert_eq!(decompressed.as_slice(), data);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_zstd_roundtrip() {
        let data = b"Hello, Haagenti! This is a test of Zstd compression.";
        let compressed = compress(data, Algorithm::Zstd).unwrap();
        let decompressed = decompress(&compressed, Algorithm::Zstd).unwrap();
        assert_eq!(decompressed.as_slice(), data);
    }

    #[test]
    #[cfg(feature = "brotli")]
    fn test_brotli_roundtrip() {
        let data = b"Hello, Haagenti! This is a test of Brotli compression.";
        let compressed = compress(data, Algorithm::Brotli).unwrap();
        let decompressed = decompress(&compressed, Algorithm::Brotli).unwrap();
        assert_eq!(decompressed.as_slice(), data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn test_gzip_roundtrip() {
        let data = b"Hello, Haagenti! This is a test of Gzip compression.";
        let compressed = compress(data, Algorithm::Gzip).unwrap();
        let decompressed = decompress(&compressed, Algorithm::Gzip).unwrap();
        assert_eq!(decompressed.as_slice(), data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn test_detect_gzip() {
        let data = b"Hello, Haagenti!";
        let compressed = compress(data, Algorithm::Gzip).unwrap();
        assert_eq!(detect_algorithm(&compressed), Some(Algorithm::Gzip));
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_detect_zstd() {
        let data = b"Hello, Haagenti!";
        let compressed = compress(data, Algorithm::Zstd).unwrap();
        assert_eq!(detect_algorithm(&compressed), Some(Algorithm::Zstd));
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_compression_levels() {
        // Use Zstd for level testing since it has a proper frame format
        let data = b"Test data for compression level testing with some extra content.";
        for level in [
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let compressed = compress_with_level(data, Algorithm::Zstd, level).unwrap();
            let decompressed = decompress(&compressed, Algorithm::Zstd).unwrap();
            assert_eq!(decompressed.as_slice(), data);
        }
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn test_lz4_codec_levels() {
        let data = b"Test data for LZ4 compression level testing.";
        for level in [
            CompressionLevel::Fast,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ] {
            let codec = Lz4Codec::with_level(level);
            let (compressed, size) = codec.compress_with_size(data).unwrap();
            let decompressed = codec.decompress_sized(&compressed, size).unwrap();
            assert_eq!(decompressed.as_slice(), data);
        }
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_dictionary_compression_roundtrip() {
        // Create sample data with repeating patterns (dictionary-friendly)
        let samples: Vec<&[u8]> = vec![
            b"The quick brown fox jumps over the lazy dog.",
            b"The quick brown cat jumps over the lazy mouse.",
            b"The quick brown bird jumps over the lazy snake.",
            b"The quick brown fish jumps over the lazy frog.",
            b"The quick brown deer jumps over the lazy bear.",
        ];

        // Train dictionary from samples
        let dict = ZstdDictionary::train(&samples, 4096).unwrap();
        assert!(dict.id() != 0);
        assert!(dict.size() > 0);

        // Compress with dictionary
        let compressor = ZstdDictCompressor::new(dict.clone());
        let test_data = b"The quick brown wolf jumps over the lazy rabbit.";
        let compressed = compressor.compress(test_data).unwrap();

        // Decompress with dictionary
        let decompressor = ZstdDictDecompressor::new(dict);
        let decompressed = decompressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), test_data);
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn test_dictionary_serialization() {
        let content = b"Sample dictionary content for testing serialization.";
        let dict = ZstdDictionary::from_content(content.to_vec()).unwrap();

        // Serialize and parse
        let serialized = dict.serialize();
        let parsed = ZstdDictionary::parse(&serialized).unwrap();

        assert_eq!(dict.id(), parsed.id());
    }
}

# Haagenti Feature Flags & Configuration Reference

**Purpose:** Comprehensive documentation of all feature flags, configuration structs, public constants, and hardcoded values across the 28-crate Haagenti workspace.

**Last Updated:** 2026-01-22

**See Also:** [DESIGN_DIVERGENCES.md](DESIGN_DIVERGENCES.md) - Documents intentional divergences from reference compression implementations (Zstd, LZ4, Brotli, Deflate) and explains the performance rationale.

---

## Table of Contents

1. [Umbrella Crate (haagenti)](#umbrella-crate-haagenti)
2. [Core Stack](#core-stack)
3. [Compression Stack](#compression-stack)
4. [Tensor/Hologram Stack](#tensorhologram-stack)
5. [GPU Acceleration Stack](#gpu-acceleration-stack)
6. [Mobile Deployment Stack](#mobile-deployment-stack)
7. [Distributed Stack](#distributed-stack)
8. [ML/Adaptation Stack](#mladaptation-stack)
9. [Integration Stack](#integration-stack)
10. [Hardcoded Values Summary](#hardcoded-values-summary)
11. [Default Configuration Matrix](#default-configuration-matrix)

---

## Umbrella Crate (haagenti)

The main `haagenti` crate re-exports from subcrates and provides feature bundles.

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | Standard compression | `lz4`, `zstd`, `brotli`, `deflate`, `simd`, `stream` |
| `lz4` | LZ4 algorithm | `haagenti-lz4` |
| `zstd` | Zstandard algorithm | `haagenti-zstd` |
| `brotli` | Brotli algorithm | `haagenti-brotli` |
| `deflate` | Deflate/Gzip/Zlib | `haagenti-deflate` |
| `simd` | SIMD acceleration | `haagenti-simd` |
| `stream` | Streaming API | `haagenti-stream` |
| `parallel` | Parallel processing | `rayon` |
| `testing` | Test utilities | (none) |
| `full` | All algorithms | `lz4`, `zstd`, `brotli`, `deflate`, `simd`, `stream` |
| `turbo` | High-performance | `parallel`, `zstd` |

### Inference Bundle Features

| Bundle | Description | Includes |
|--------|-------------|----------|
| `inference` | Full local inference | `cuda`, `webgpu`, `inference-streaming`, `autoopt`, `fragments`, `importance`, `speculative` |
| `inference-mobile` | iOS/Android inference | `mobile`, `inference-streaming` |
| `inference-distributed` | Multi-node inference | `distributed`, `inference-streaming`, `serverless` |
| `sovereign` | Everything | `full`, `inference`, `inference-mobile`, `inference-distributed`, `learning` |

### Individual Inference Features

| Feature | Description | Dependency |
|---------|-------------|------------|
| `cuda` | NVIDIA GPU acceleration | `haagenti-cuda`, `turbo` |
| `webgpu` | WebGPU compute shaders | `haagenti-webgpu` |
| `mobile` | CoreML/NNAPI backends | `haagenti-mobile` |
| `distributed` | Distributed topologies | `haagenti-distributed` |
| `serverless` | Cold-start optimization | `haagenti-serverless` |
| `inference-streaming` | Progressive decompression | `haagenti-streaming` |
| `fragments` | Cross-model fragment sharing | `haagenti-fragments` |
| `importance` | ML-guided importance scoring | `haagenti-importance`, `fragments` |
| `autoopt` | Runtime auto-optimization | `haagenti-autoopt` |
| `learning` | LoRA, reservoir computing | `haagenti-learning` |
| `speculative` | Speculative execution | `haagenti-speculative`, `fragments` |

---

## Core Stack

### haagenti-core

**Location:** `crates/haagenti-core/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `dct` | DCT (Discrete Cosine Transform) | No |
| `simd` | SIMD-accelerated primitives | No |
| `std` | Standard library support | Yes |
| `alloc` | Heap allocation support | Yes |

#### Configuration Structs

```rust
struct DctConfig {
    block_size: usize,              // default: 8
    precision: Precision,           // default: F32
    normalize: bool,                // default: true
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DCT_BLOCK_SIZE` | 8 | Standard DCT block dimension |
| `MAX_BLOCK_SIZE` | 64 | Maximum supported block size |

---

## Compression Stack

### haagenti-lz4

**Location:** `crates/haagenti-lz4/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library | Yes |
| `alloc` | Heap allocation | Yes |
| `frame` | LZ4 frame format | Yes |
| `hc` | High-compression mode | No |

#### Configuration Structs

```rust
struct Lz4Config {
    acceleration: i32,              // default: 1 (range: 1-65537)
    dictionary: Option<Vec<u8>>,    // default: None
    content_checksum: bool,         // default: false
    block_checksum: bool,           // default: false
    block_size: BlockSize,          // default: Max64KB
    block_mode: BlockMode,          // default: Linked
}

struct Lz4HcConfig {
    compression_level: i32,         // default: 9 (range: 1-12)
    favor_decompression_speed: bool,// default: false
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `LZ4_MAX_INPUT_SIZE` | 2,113,929,216 | ~2GB maximum input |
| `LZ4_ACCELERATION_DEFAULT` | 1 | Default acceleration factor |
| `LZ4_ACCELERATION_MAX` | 65537 | Maximum acceleration |
| `LZ4HC_CLEVEL_MIN` | 1 | Minimum HC compression level |
| `LZ4HC_CLEVEL_DEFAULT` | 9 | Default HC compression level |
| `LZ4HC_CLEVEL_MAX` | 12 | Maximum HC compression level |

---

### haagenti-zstd

**Location:** `crates/haagenti-zstd/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Full functionality | `std`, `alloc`, `dictionary`, `streaming` |
| `std` | Standard library | Yes |
| `alloc` | Heap allocation | Yes |
| `dictionary` | Dictionary compression | Yes |
| `streaming` | Streaming compression | Yes |
| `parallel` | Multi-threaded | No |
| `simd` | SIMD acceleration | Yes (with `haagenti-simd`) |
| `experimental` | Experimental features | No |

#### Configuration Structs

```rust
struct ZstdCompressor {
    level: i32,                     // default: 3 (range: 1-22)
    dictionary: Option<ZstdDict>,   // default: None
    checksum: bool,                 // default: false
    workers: usize,                 // default: 0 (single-threaded)
    long_distance_matching: bool,   // default: false
    window_log: u32,                // default: 0 (auto)
}

struct ZstdDecompressor {
    dictionary: Option<ZstdDict>,   // default: None
    window_log_max: u32,            // default: 0 (auto)
}

struct DictTrainingConfig {
    max_dict_size: usize,           // default: 112,640 (110KB)
    compression_level: i32,         // default: 3
    notification_level: u32,        // default: 0
    dict_id: u32,                   // default: 0 (auto)
    steps: u32,                     // default: 0 (auto)
    threads: u32,                   // default: 0 (auto)
    d: u32,                         // default: 0 (auto)
    k: u32,                         // default: 0 (auto)
    shrink_dict: bool,              // default: false
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ZSTD_CLEVEL_DEFAULT` | 3 | Default compression level |
| `ZSTD_CLEVEL_MIN` | 1 | Minimum level |
| `ZSTD_CLEVEL_MAX` | 22 | Maximum level |
| `ZSTD_MAGIC_NUMBER` | 0xFD2FB528 | Frame magic number |
| `ZSTD_DICT_MAGIC` | 0xEC30A437 | Dictionary magic |
| `ZSTD_CONTENTSIZE_UNKNOWN` | 0 | Unknown content size sentinel |
| `ZSTD_WINDOWLOG_MIN` | 10 | Minimum window log |
| `ZSTD_WINDOWLOG_MAX` | 31 | Maximum window log |
| `ZSTD_HASHLOG_MIN` | 6 | Minimum hash log |
| `ZSTD_HASHLOG_MAX` | 30 | Maximum hash log |
| `FSE_MAX_SYMBOL_VALUE` | 255 | Maximum FSE symbol |
| `HUF_TABLELOG_DEFAULT` | 11 | Default Huffman table log |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `fse.rs:42` | `1 << 15` (32KB) | Max block size | Extract to `FSE_MAX_BLOCK_SIZE` |
| `huffman.rs:156` | `1 << 12` (4KB) | Max single Huffman block | Extract to constant |
| `match_finder.rs:89` | `3` | Minimum match length | Make configurable |
| `match_finder.rs:267` | `128` | Max hash chain length | Performance tunable |
| `speculative.rs:45` | `5` | Parallel strategy count | Extract to constant |

---

### haagenti-brotli

**Location:** `crates/haagenti-brotli/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library | Yes |
| `alloc` | Heap allocation | Yes |

#### Configuration Structs

```rust
struct BrotliConfig {
    quality: u32,                   // default: 6 (range: 0-11)
    lgwin: u32,                     // default: 22 (range: 10-24)
    mode: BrotliMode,               // default: Generic
}

enum BrotliMode {
    Generic,    // General-purpose compression
    Text,       // UTF-8 text optimization
    Font,       // WOFF2 font optimization
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BROTLI_QUALITY_MIN` | 0 | Minimum quality (fastest) |
| `BROTLI_QUALITY_DEFAULT` | 6 | Default quality |
| `BROTLI_QUALITY_MAX` | 11 | Maximum quality (smallest) |
| `BROTLI_LGWIN_MIN` | 10 | Minimum window size log |
| `BROTLI_LGWIN_DEFAULT` | 22 | Default window size log |
| `BROTLI_LGWIN_MAX` | 24 | Maximum window size log |

---

### haagenti-deflate

**Location:** `crates/haagenti-deflate/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library | Yes |
| `alloc` | Heap allocation | Yes |
| `gzip` | Gzip format support | Yes |
| `zlib` | Zlib format support | Yes |

#### Configuration Structs

```rust
struct DeflateConfig {
    level: u32,                     // default: 6 (range: 0-9)
    strategy: DeflateStrategy,      // default: Default
    window_bits: u32,               // default: 15 (range: 8-15)
}

enum DeflateStrategy {
    Default,        // Mixed literal/length strategy
    Filtered,       // Data with many small values
    HuffmanOnly,    // Force Huffman, no LZ77
    Rle,            // Run-length encoding
    Fixed,          // Fixed Huffman codes only
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFLATE_LEVEL_NONE` | 0 | No compression (store) |
| `DEFLATE_LEVEL_FAST` | 1 | Fastest compression |
| `DEFLATE_LEVEL_DEFAULT` | 6 | Default compression |
| `DEFLATE_LEVEL_BEST` | 9 | Best compression |
| `ZLIB_HEADER` | `0x789C` | Zlib default header |
| `GZIP_MAGIC` | `0x1F8B` | Gzip magic number |

---

### haagenti-simd

**Location:** `crates/haagenti-simd/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Auto-detect SIMD | Yes |
| `avx2` | AVX2 instructions | Auto-detected |
| `avx512` | AVX-512 instructions | Auto-detected |
| `neon` | ARM NEON | Auto-detected |
| `fallback` | Scalar fallback | Yes |

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SIMD_ALIGNMENT` | 64 | AVX-512 alignment |
| `AVX2_VECTOR_SIZE` | 32 | 256-bit vector |
| `AVX512_VECTOR_SIZE` | 64 | 512-bit vector |
| `NEON_VECTOR_SIZE` | 16 | 128-bit vector |

---

### haagenti-stream

**Location:** `crates/haagenti-stream/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library | Yes |
| `alloc` | Heap allocation | Yes |
| `async` | Async streaming | No |

#### Configuration Structs

```rust
struct StreamConfig {
    buffer_size: usize,             // default: 65536 (64KB)
    auto_flush: bool,               // default: false
    checksum: bool,                 // default: false
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_BUFFER_SIZE` | 65,536 | 64KB default buffer |
| `MIN_BUFFER_SIZE` | 4,096 | 4KB minimum |
| `MAX_BUFFER_SIZE` | 16,777,216 | 16MB maximum |

---

## Tensor/Hologram Stack

### haagenti-hct

**Location:** `crates/haagenti-hct/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Full support | `full` |
| `lz4` | LZ4 compression | No |
| `zstd` | Zstd compression | No |
| `full` | All compression | `lz4`, `zstd` |

#### Configuration Structs

```rust
struct HoloTensorConfig {
    encoding: HolographicEncoding,  // default: Spectral
    n_fragments: u16,               // default: 8
    essential_ratio: f32,           // default: 0.1
    seed: Option<u64>,              // default: None
    compression: CompressionAlgorithm, // default: Zstd
    quality_target: f32,            // default: 0.95
}

enum HolographicEncoding {
    Spectral,           // DCT-based frequency distribution
    RandomProjection,   // Random matrix projection
    LowRankDistributed, // SVD-based low-rank approximation (LRDF)
}

struct SpectralEncoderConfig {
    block_size: usize,              // default: 8
    quality: f32,                   // default: 0.95
    progressive: bool,              // default: true
}

struct CompressiveSpectralConfig {
    sensing_ratio: f32,             // default: 0.5 (50% measurements)
    iterations: usize,              // default: 100
    tolerance: f32,                 // default: 1e-6
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `HCT_MAGIC` | `0x48435400` | "HCT\0" file magic |
| `HCT_VERSION` | 1 | Current format version |
| `DEFAULT_FRAGMENT_COUNT` | 8 | Default number of fragments |
| `DEFAULT_ESSENTIAL_RATIO` | 0.1 | 10% essential data |
| `MAX_FRAGMENTS` | 256 | Maximum fragment count |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `holotensor.rs:156` | `0.95` | Default quality target | Already configurable |
| `holotensor.rs:312` | `0.01` - `0.5` | Essential ratio range | Extract to constants |
| `spectral.rs:89` | `1e-10` | Numerical stability epsilon | Extract to constant |

---

### haagenti-fragments

**Location:** `crates/haagenti-fragments/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Base functionality | Yes |
| `dedup` | Cross-model deduplication | No |
| `cache` | Fragment caching | No |

#### Configuration Structs

```rust
struct FragmentLibraryConfig {
    storage_path: PathBuf,          // required
    max_size_bytes: u64,            // default: 10GB
    dedup_enabled: bool,            // default: true
    hash_algorithm: HashAlgorithm,  // default: XXHash64
}

struct FragmentConfig {
    min_size: usize,                // default: 1024 (1KB)
    max_size: usize,                // default: 1,048,576 (1MB)
    alignment: usize,               // default: 64
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_MAX_LIBRARY_SIZE` | 10,737,418,240 | 10GB |
| `DEFAULT_MIN_FRAGMENT_SIZE` | 1,024 | 1KB |
| `DEFAULT_MAX_FRAGMENT_SIZE` | 1,048,576 | 1MB |
| `FRAGMENT_ALIGNMENT` | 64 | Cache-line alignment |

---

### haagenti-sparse

**Location:** `crates/haagenti-sparse/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Standard formats | `csr`, `coo` |
| `csr` | Compressed Sparse Row | Yes |
| `csc` | Compressed Sparse Column | No |
| `coo` | Coordinate format | Yes |
| `dense` | Dense conversion | Yes |

#### Configuration Structs

```rust
struct SparseConfig {
    format: SparseFormat,           // default: Csr
    threshold: f32,                 // default: 1e-6 (sparsity threshold)
    sort_indices: bool,             // default: true
}
```

---

## GPU Acceleration Stack

### haagenti-cuda

**Location:** `crates/haagenti-cuda/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Standard features | Yes |
| `f16` | Half-precision support | Yes |
| `bf16` | BFloat16 support | No |
| `tensor-cores` | Tensor core acceleration | No |
| `multi-gpu` | Multi-GPU support | No |

#### Configuration Structs

```rust
struct CudaConfig {
    device_id: i32,                 // default: 0
    stream_count: usize,            // default: 4
    memory_pool_size: usize,        // default: 0 (auto)
    async_copy: bool,               // default: true
    unified_memory: bool,           // default: false
}

struct KernelConfig {
    block_size: (u32, u32, u32),    // default: (256, 1, 1)
    shared_memory: usize,           // default: 0
    stream: Option<CudaStream>,     // default: None (default stream)
}

struct Lz4GpuConfig {
    max_block_size: usize,          // default: 65536
    stream_count: usize,            // default: 2
    async_transfer: bool,           // default: true
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CUDA_MAX_BLOCK_SIZE` | 1,024 | Max threads per block |
| `CUDA_MAX_GRID_DIM` | 2,147,483,647 | Max grid dimension |
| `CUDA_WARP_SIZE` | 32 | Threads per warp |
| `GPU_MEMORY_ALIGNMENT` | 256 | Memory alignment |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `device.rs:45` | `256` | Default block threads | Make configurable |
| `kernel.rs:89` | `16` | Tile size for matrix ops | Profile-dependent |
| `memory.rs:67` | `1GB` | Default memory pool | Should auto-detect |

---

### haagenti-webgpu

**Location:** `crates/haagenti-webgpu/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Standard features | Yes |
| `spirv` | SPIR-V shader support | No |
| `wgsl` | WGSL shader support | Yes |
| `compute` | Compute pipeline | Yes |
| `render` | Render pipeline | No |

#### Configuration Structs

```rust
struct WebGpuConfig {
    power_preference: PowerPreference,  // default: HighPerformance
    force_fallback_adapter: bool,       // default: false
    limits: Limits,                     // default: Limits::default()
}

struct ComputePipelineConfig {
    entry_point: String,            // default: "main"
    constants: HashMap<String, f64>,// default: empty
    workgroup_size: (u32, u32, u32),// default: (64, 1, 1)
}
```

#### Shader Pipelines

| Pipeline | Entry Point | Description |
|----------|-------------|-------------|
| `matmul` | `main` | Matrix multiplication |
| `gelu` | `main` | GELU activation |
| `softmax` | `main` | Softmax normalization |
| `layer_norm` | `main` | Layer normalization |
| `dequantize_int4` | `main` | INT4 dequantization |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `pipeline.rs:34` | `64` | Workgroup size | Make configurable |
| `shaders/matmul.wgsl` | `16` | Tile size | Profile-dependent |
| `shaders/softmax.wgsl` | `256` | Max sequence length | Increase or dynamic |

---

## Mobile Deployment Stack

### haagenti-mobile

**Location:** `crates/haagenti-mobile/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Auto-detect platform | Yes |
| `coreml` | Apple CoreML | iOS/macOS |
| `nnapi` | Android NNAPI | Android |
| `int4` | INT4 quantization | Yes |
| `int8` | INT8 quantization | Yes |
| `thermal` | Thermal management | Yes |
| `battery` | Battery-aware scheduling | Yes |

#### Configuration Structs

```rust
struct MobileConfig {
    compute_units: ComputeUnits,    // default: All
    allow_low_precision: bool,      // default: true
    thermal_state_check: bool,      // default: true
    battery_optimization: bool,     // default: true
    max_memory_mb: usize,           // default: 0 (auto)
}

struct CoreMLConfig {
    compute_units: MLComputeUnits,  // default: All (CPU+GPU+ANE)
    allow_low_precision: bool,      // default: true
    optimization_level: u32,        // default: 2
}

struct NnapiConfig {
    prefer_npu: bool,               // default: true
    allow_fp16: bool,               // default: true
    allow_dynamic: bool,            // default: false
    cache_dir: Option<PathBuf>,     // default: None
}

struct ThermalConfig {
    max_temperature_c: f32,         // default: 45.0
    throttle_threshold_c: f32,      // default: 40.0
    check_interval_ms: u64,         // default: 1000
}

struct BatteryConfig {
    low_battery_threshold: f32,     // default: 0.2 (20%)
    critical_threshold: f32,        // default: 0.1 (10%)
    reduce_on_low: bool,            // default: true
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `THERMAL_CHECK_INTERVAL_MS` | 1,000 | 1 second check interval |
| `DEFAULT_MAX_TEMP_C` | 45.0 | Maximum operating temp |
| `DEFAULT_THROTTLE_TEMP_C` | 40.0 | Throttling threshold |
| `LOW_BATTERY_THRESHOLD` | 0.2 | 20% battery |
| `CRITICAL_BATTERY_THRESHOLD` | 0.1 | 10% battery |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `coreml.rs:89` | `2` | Optimization level | Make configurable |
| `nnapi.rs:145` | `8` | Max concurrent ops | Should be device-dependent |
| `thermal.rs:67` | `45.0°C` | Max temp | Already configurable |

---

## Distributed Stack

### haagenti-distributed

**Location:** `crates/haagenti-distributed/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Ring topology | `ring` |
| `ring` | Ring topology | Yes |
| `mesh` | Mesh topology | No |
| `tree` | Tree topology | No |
| `hierarchical` | Hierarchical topology | No |
| `fully-connected` | Fully connected | No |
| `all-topologies` | All topologies | All above |

#### Configuration Structs

```rust
struct DistributedConfig {
    topology: Topology,             // default: Ring
    world_size: usize,              // required
    rank: usize,                    // required
    backend: Backend,               // default: Tcp
    timeout_ms: u64,                // default: 30000
}

struct RingConfig {
    chunk_size: usize,              // default: 1048576 (1MB)
    async_ops: bool,                // default: true
}

struct MeshConfig {
    dimensions: Vec<usize>,         // required (e.g., [4, 4] for 4x4 mesh)
    wrap_around: bool,              // default: false
}

struct TreeConfig {
    branching_factor: usize,        // default: 2
    root_rank: usize,               // default: 0
}

struct HierarchicalConfig {
    local_size: usize,              // required (nodes per local group)
    global_topology: Topology,      // default: Ring
    local_topology: Topology,       // default: Ring
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_CHUNK_SIZE` | 1,048,576 | 1MB chunk |
| `DEFAULT_TIMEOUT_MS` | 30,000 | 30 second timeout |
| `MAX_WORLD_SIZE` | 65,535 | Max nodes |
| `DEFAULT_BRANCHING_FACTOR` | 2 | Binary tree default |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `ring.rs:67` | `1MB` | Ring chunk size | Already configurable |
| `mesh.rs:89` | `100` | Max retries | Extract to constant |
| `coordinator.rs:156` | `5000ms` | Heartbeat interval | Make configurable |

---

### haagenti-serverless

**Location:** `crates/haagenti-serverless/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Standard features | Yes |
| `aws-lambda` | AWS Lambda support | No |
| `cloudflare` | Cloudflare Workers | No |
| `vercel` | Vercel Edge Functions | No |

#### Configuration Structs

```rust
struct ServerlessConfig {
    cold_start_optimization: bool,  // default: true
    max_memory_mb: usize,           // default: 512
    timeout_ms: u64,                // default: 10000
    warmup_enabled: bool,           // default: true
}

struct ColdStartConfig {
    preload_fragments: Vec<String>, // default: empty
    lazy_init: bool,                // default: true
    memory_map: bool,               // default: true
}

struct WarmupConfig {
    schedule: WarmupSchedule,       // default: Periodic(300s)
    concurrent: usize,              // default: 1
}

enum WarmupSchedule {
    Periodic(u64),      // Every N seconds
    Cron(String),       // Cron expression
    Traffic(f32),       // Based on traffic threshold
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_WARMUP_INTERVAL_S` | 300 | 5 minute warmup |
| `DEFAULT_MAX_MEMORY_MB` | 512 | 512MB default |
| `DEFAULT_TIMEOUT_MS` | 10,000 | 10 second timeout |
| `COLD_START_TARGET_MS` | 100 | Target cold start time |

---

### haagenti-streaming

**Location:** `crates/haagenti-streaming/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Standard streaming | Yes |
| `progressive` | Progressive loading | Yes |
| `preview` | Preview generation | No |
| `adaptive` | Adaptive quality | No |

#### Configuration Structs

```rust
struct StreamingConfig {
    buffer_size: usize,             // default: 262144 (256KB)
    prefetch_count: usize,          // default: 2
    progressive: bool,              // default: true
    min_quality: f32,               // default: 0.5
}

struct PreviewConfig {
    preview_fragments: usize,       // default: 1
    preview_quality: f32,           // default: 0.3
    timeout_ms: u64,                // default: 100
}

struct AdaptiveConfig {
    target_latency_ms: u64,         // default: 50
    quality_step: f32,              // default: 0.1
    min_quality: f32,               // default: 0.3
    max_quality: f32,               // default: 1.0
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_BUFFER_SIZE` | 262,144 | 256KB buffer |
| `DEFAULT_PREFETCH_COUNT` | 2 | Prefetch 2 fragments |
| `MIN_PREVIEW_QUALITY` | 0.3 | 30% quality minimum |

---

### haagenti-network

**Location:** `crates/haagenti-network/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | TCP transport | `tcp` |
| `tcp` | TCP transport | Yes |
| `rdma` | RDMA transport | No |
| `shared-memory` | Shared memory | No |
| `tls` | TLS encryption | No |

#### Configuration Structs

```rust
struct NetworkConfig {
    transport: Transport,           // default: Tcp
    bind_address: SocketAddr,       // default: 0.0.0.0:0
    max_connections: usize,         // default: 100
    buffer_size: usize,             // default: 65536
}

struct TcpConfig {
    nodelay: bool,                  // default: true
    keepalive: Option<Duration>,    // default: Some(60s)
    recv_buffer: usize,             // default: 262144
    send_buffer: usize,             // default: 262144
}

struct TlsConfig {
    cert_path: PathBuf,             // required
    key_path: PathBuf,              // required
    ca_path: Option<PathBuf>,       // default: None
    verify_peer: bool,              // default: true
}
```

---

## ML/Adaptation Stack

### haagenti-learning

**Location:** `crates/haagenti-learning/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `lora` | LoRA adapter support | No |
| `replay-buffer` | Experience replay | No |
| `ewc` | Elastic Weight Consolidation | No |
| `progressive` | Progressive layer unfreezing | No |

#### Configuration Structs

```rust
struct LoraConfig {
    rank: usize,                    // default: 8
    alpha: f32,                     // default: 16.0
    dropout: f32,                   // default: 0.0
    target_modules: Vec<String>,    // default: ["q_proj", "k_proj", "v_proj", "o_proj"]
    fan_in_init: bool,              // default: true
}

struct BufferConfig {
    max_size: usize,                // default: 10000
    batch_size: usize,              // default: 32
    prioritized: bool,              // default: false
    priority_alpha: f32,            // default: 0.6
    importance_beta: f32,           // default: 0.4
    seed: Option<u64>,              // default: None
}

struct SchedulerConfig {
    initial_lr: f32,                // default: 1e-4
    final_lr: f32,                  // default: 1e-6
    warmup_steps: usize,            // default: 100
    total_steps: usize,             // default: 10000
    scheduler_type: SchedulerType,  // default: WarmupCosine
}

struct TrainerConfig {
    scheduler: SchedulerConfig,
    buffer: BufferConfig,
    gradient_accumulation: usize,   // default: 1
    gradient_clip: Option<f32>,     // default: Some(1.0)
    checkpoint_interval: usize,     // default: 1000
    log_interval: usize,            // default: 100
    mixed_precision: bool,          // default: false
    early_stopping_patience: Option<usize>, // default: Some(10)
}

struct EwcConfig {
    lambda: f32,                    // default: 100.0
    fisher_samples: usize,          // default: 200
    damping: f32,                   // default: 1e-3
    online: bool,                   // default: false
    gamma: f32,                     // default: 0.9
}
```

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `trainer.rs:84` | `0.99` | EMA decay factor | Make configurable |
| `buffer.rs:72` | `1.0` | Initial priority | Make configurable |
| `scheduler.rs:119` | `0.3` | Step decay ratio | Make configurable |
| `scheduler.rs:127` | `10.0` | OneCycle peak multiplier | Make configurable |
| `scheduler.rs:124` | `0.4/0.6` | OneCycle phase split | Make configurable |

---

### haagenti-autoopt

**Location:** `crates/haagenti-autoopt/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `bayesian` | Bayesian optimization | No |
| `genetic` | Genetic algorithms | No |
| `profiling` | Runtime profiling | No |
| `hardware-aware` | Hardware-aware optimization | No |

#### Configuration Structs

```rust
struct BayesianConfig {
    max_iterations: usize,          // default: 50
    initial_samples: usize,         // default: 10
    acquisition: AcquisitionFunction, // default: ExpectedImprovement
    exploration: f32,               // default: 0.01
    seed: Option<u64>,              // default: None
    early_stopping_threshold: Option<f32>, // default: None
}

struct GeneticConfig {
    population_size: usize,         // default: 50
    generations: usize,             // default: 100
    mutation_rate: f32,             // default: 0.1
    crossover_rate: f32,            // default: 0.8
    elite_count: usize,             // default: 2
    tournament_size: usize,         // default: 3
    seed: Option<u64>,              // default: None
}

struct TunerConfig {
    max_time: Duration,             // default: 300 seconds
    max_trials: usize,              // default: 50
    target_metric: String,          // default: "latency_ms"
    minimize: bool,                 // default: true
    constraints: HashMap<String, f32>, // default: empty
    profile: bool,                  // default: true
}

struct HardwareProfile {
    name: String,
    memory_bytes: u64,              // e.g., 16GB for CPU
    memory_bandwidth_gbps: f32,     // e.g., 50 for CPU, 900 for GPU
    compute_tflops: f32,            // e.g., 0.5 for CPU, 80 for GPU
    stream_processors: Option<u32>, // e.g., 80 SMs for NVIDIA
    clock_mhz: u32,                 // e.g., 3000 for CPU
}
```

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `bayesian.rs:236` | `100` | Acquisition candidates | Make configurable |
| `bayesian.rs:300` | `10.0` | RBF kernel width | Make configurable |
| `bayesian.rs:312` | `0.01` | Min variance floor | Make configurable |
| `genetic.rs:219` | `0.1` | Mutation std ratio | Already in config |
| `hardware.rs:65` | `16GB` | Default CPU memory | Auto-detect |
| `hardware.rs:82` | `80` | Typical NVIDIA SMs | Auto-detect |

---

### haagenti-importance

**Location:** `crates/haagenti-importance/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `ml` | ML-based scoring (Candle) | No |

#### Configuration Structs

```rust
struct ScorerConfig {
    prompt_weight: f32,             // default: 0.3
    history_weight: f32,            // default: 0.3
    quality_weight: f32,            // default: 0.25
    step_weight: f32,               // default: 0.15
    min_importance: f32,            // default: 0.1
}
```

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `scorer.rs:144` | `0.8/0.5` | Confidence values | Make configurable |
| `predictor.rs` | `0.95` | High quality sensitivity | Make configurable |
| `analyzer.rs` | `0.95` | Detail level for faces | Make configurable |
| `analyzer.rs` | `0.5+0.1*i` | Block importance | Make configurable |
| `history.rs` | `0.1` | Alpha for updates | Make configurable |
| `history.rs` | `86400.0` | Recency decay (seconds) | Make configurable |

---

### haagenti-speculative

**Location:** `crates/haagenti-speculative/`

#### Configuration Structs

```rust
struct IntentConfig {
    min_chars: usize,               // default: 3
    speculation_threshold: f32,     // default: 0.6
    commit_threshold: f32,          // default: 0.8
    learn_from_history: bool,       // default: true
    max_history: usize,             // default: 10000
}

struct BufferConfig {
    max_size: usize,                // default: 536870912 (512MB)
    max_entries: usize,             // default: 1000
    eviction_threshold: f32,        // default: 0.8
    ttl_ms: u64,                    // default: 30000
}

struct LoaderConfig {
    intent: IntentConfig,
    buffer: BufferConfig,
    max_concurrent: usize,          // default: 4
    debounce_ms: u64,               // default: 100
    enable_learning: bool,          // default: true
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SPECULATION_THRESHOLD` | 0.6 | 60% confidence to speculate |
| `DEFAULT_COMMIT_THRESHOLD` | 0.8 | 80% confidence to commit |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `intent.rs` | `0.9-0.95` | Human/face importance | Make configurable |
| `intent.rs` | `0.6-0.8` | Landscape importance | Make configurable |
| `buffer.rs:83` | `0.3` | Confidence eviction | Make configurable |
| `intent.rs` | `0.5` | Base confidence | Make configurable |
| `intent.rs` | `0.25` | Recency/frequency boost | Make configurable |

---

### haagenti-merging

**Location:** `crates/haagenti-merging/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `ties` | TIES algorithm | No |
| `dare` | DARE algorithm | No |
| `evolutionary` | Genetic merging | No |
| `slerp` | Spherical interpolation | No |

#### Configuration Structs

```rust
struct TiesConfig {
    trim_ratio: f32,                // default: 0.2
    scaling_factor: f32,            // default: 1.0
    model_weights: Vec<f32>,        // default: empty
    layer_trim_ratios: HashMap<String, f32>, // default: empty
}

struct DareConfig {
    drop_rate: f32,                 // default: 0.9
    seed: Option<u64>,              // default: None
    rescale_method: RescaleMethod,  // default: Inverse
    layer_drop_rates: HashMap<String, f32>, // default: empty
    combine_method: CombineMethod,  // default: Sum
}

struct LinearConfig {
    weights: Vec<f32>,              // default: [0.5, 0.5]
    normalize: bool,                // default: true
    layer_weights: HashMap<String, Vec<f32>>, // default: empty
}

struct EvolutionaryConfig {
    population_size: usize,         // default: 50
    generations: usize,             // default: 100
    mutation_rate: f32,             // default: 0.1
    mutation_strength: f32,         // default: 0.1
    crossover_rate: f32,            // default: 0.8
    elite_count: usize,             // default: 2
    tournament_size: usize,         // default: 3
    seed: Option<u64>,              // default: None
    early_stopping_patience: usize, // default: 10
}
```

---

### haagenti-latent-cache

**Location:** `crates/haagenti-latent-cache/`

#### Configuration Structs

```rust
struct CacheConfig {
    storage: StorageConfig,
    hnsw: HnswConfig,
    min_similarity: f32,            // default: 0.85
    total_steps: u32,               // default: 20
    checkpoint_steps: Vec<u32>,     // default: [5, 10, 15, 18]
}

struct HnswConfig {
    ef_construction: usize,         // default: 100
    ef_search: usize,               // default: 50
    m: usize,                       // default: 16
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SIMILARITY_THRESHOLD` | 0.85 | 85% similarity required |
| `DEFAULT_CHECKPOINT_COUNT` | 4 | 4 checkpoints per generation |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `cache.rs:36` | `[5,10,15,18]` | Checkpoint schedule | Make configurable per model |
| `embedding.rs` | `768` | CLIP embedding dim | Model-dependent |

---

## Integration Stack

### haagenti-grpc

**Location:** `crates/haagenti-grpc/`

#### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Zstd + LZ4 | `zstd`, `lz4` |
| `zstd` | Zstd algorithm | Yes |
| `lz4` | LZ4 algorithm | Yes |
| `brotli` | Brotli algorithm | No |
| `deflate` | Deflate algorithm | No |
| `all-algorithms` | All compression | All above |

#### Configuration Structs

```rust
struct ServerConfig {
    host: String,                   // default: "0.0.0.0"
    port: u16,                      // default: 50051
    tls_enabled: bool,              // default: false
    tls_cert: Option<String>,       // default: None
    tls_key: Option<String>,        // default: None
    tls_client_ca: Option<String>,  // default: None
    tls_require_client_cert: bool,  // default: false
    max_message_size: usize,        // default: 67108864 (64MB)
    metrics_enabled: bool,          // default: true
    metrics_port: u16,              // default: 9090
    log_level: String,              // default: "info"
}
```

#### Public Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_HOST` | `"0.0.0.0"` | Bind all interfaces |
| `DEFAULT_PORT` | `50051` | Standard gRPC port |
| `DEFAULT_MAX_MESSAGE_SIZE` | `67,108,864` | 64MB |
| `DEFAULT_METRICS_PORT` | `9090` | Prometheus port |

#### Hardcoded Values

| Location | Value | Description | Recommendation |
|----------|-------|-------------|----------------|
| `service.rs:252` | `65536` | Dict size (64KB) | Make configurable |
| `config.rs:112` | `64*1024*1024` | Max message | Already configurable |

---

### haagenti-python

**Location:** `crates/haagenti-python/`

#### Configuration Structs

```rust
// Python class: HoloTensorEncoder
struct HoloTensorEncoder {
    encoding: HolographicEncoding,  // Spectral, RandomProjection, LowRankDistributed
    n_fragments: u16,               // default: 8
    seed: Option<u64>,              // default: None
    essential_ratio: Option<f32>,   // default: None (0.01-0.5 range)
    max_rank: Option<usize>,        // default: None
}

// Python class: HctWriter
struct HctWriter {
    algorithm: CompressionAlgorithm, // Zstd or Lz4
    dtype: DType,                   // F32, F16, BF16, I8, I4
    shape: Vec<u64>,
    block_size: Option<u32>,        // default: None
}

// Python class: ZstdDict
struct ZstdDict {
    id: u32,
    data: Vec<u8>,
}
```

#### Python API Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fragments` | 8 | Number of HoloTensor fragments |
| `compression` | `"zstd"` | Compression algorithm |
| `level` | `"default"` | Compression level |
| `max_dict_size` | 8192 | Dictionary max size |
| `min_samples` | 5 | Min samples for dict training |

---

## Hardcoded Values Summary

### Critical (Performance Impact)

| Crate | Location | Value | Impact |
|-------|----------|-------|--------|
| haagenti-zstd | match_finder.rs:267 | `128` | Hash chain length affects compression ratio/speed |
| haagenti-cuda | device.rs:45 | `256` | Block threads affects GPU utilization |
| haagenti-webgpu | shaders/softmax.wgsl | `256` | Max sequence length limits attention |
| haagenti-learning | trainer.rs:84 | `0.99` | EMA decay affects loss tracking |
| haagenti-autoopt | bayesian.rs:300 | `10.0` | Kernel width affects optimization |

### Medium (Quality Impact)

| Crate | Location | Value | Impact |
|-------|----------|-------|--------|
| haagenti-hct | holotensor.rs:312 | `0.01-0.5` | Essential ratio range |
| haagenti-importance | scorer.rs:144 | `0.8/0.5` | Confidence thresholds |
| haagenti-speculative | intent.rs | `0.6-0.95` | Category importance |
| haagenti-merging | dare.rs | `0.9` | Aggressive drop rate |
| haagenti-latent-cache | cache.rs:36 | `[5,10,15,18]` | Fixed checkpoint schedule |

### Low (Convenience)

| Crate | Location | Value | Impact |
|-------|----------|-------|--------|
| haagenti-python | lib.rs:515 | `8` | Default fragment count |
| haagenti-grpc | service.rs:252 | `65536` | Dictionary size |
| haagenti-distributed | mesh.rs:89 | `100` | Max retries |

---

## Default Configuration Matrix

### Compression Levels

| Algorithm | Min | Default | Max | Best For |
|-----------|-----|---------|-----|----------|
| LZ4 | 1 (acceleration) | 1 | 65537 | Speed |
| LZ4-HC | 1 | 9 | 12 | Ratio |
| Zstd | 1 | 3 | 22 | Balance |
| Brotli | 0 | 6 | 11 | Web |
| Deflate | 0 | 6 | 9 | Compat |

### Memory Defaults

| Component | Default | Min | Max |
|-----------|---------|-----|-----|
| Stream buffer | 64KB | 4KB | 16MB |
| Fragment size | 1KB-1MB | 1KB | 1MB |
| Fragment library | 10GB | N/A | N/A |
| gRPC message | 64MB | N/A | N/A |
| Serverless memory | 512MB | N/A | N/A |

### Quality Defaults

| Component | Default | Range |
|-----------|---------|-------|
| HoloTensor quality | 0.95 | 0.0-1.0 |
| Essential ratio | 0.1 | 0.01-0.5 |
| Preview quality | 0.3 | 0.0-1.0 |
| Similarity threshold | 0.85 | 0.0-1.0 |
| Speculation threshold | 0.6 | 0.0-1.0 |

### Timing Defaults

| Component | Default | Purpose |
|-----------|---------|---------|
| Distributed timeout | 30s | Node communication |
| Serverless timeout | 10s | Function execution |
| Warmup interval | 300s | Keep-alive |
| Thermal check | 1s | Temperature monitoring |
| TTL (speculative) | 30s | Buffer eviction |

---

## Benchmarking

### Running Benchmarks

```bash
# Full benchmark suite (all crates)
./scripts/benchmark-all.sh

# Quick mode (fewer iterations, faster results)
./scripts/benchmark-all.sh --quick

# Specific stack only
./scripts/benchmark-all.sh --stack compression   # haagenti, haagenti-zstd
./scripts/benchmark-all.sh --stack tensor        # haagenti-fragments, sparse, neural, adaptive
./scripts/benchmark-all.sh --stack gpu           # haagenti-cuda
./scripts/benchmark-all.sh --stack ml            # haagenti-learning, autoopt, importance, speculative, merging, latent-cache
./scripts/benchmark-all.sh --stack distributed   # haagenti-distributed, streaming, serverless, network

# Single crate
./scripts/benchmark-all.sh --crate haagenti-zstd

# Compare against baseline
HAAGENTI_BENCH_BASELINE=target/criterion-baseline ./scripts/benchmark-all.sh --compare
```

### Comprehensive Smoke Test

```bash
# Quick validation of all components
cargo run --release --example benchmark_comprehensive --features "lz4,zstd"
```

### Individual Crate Benchmarks

```bash
# Zstd (vs. reference C library)
cargo bench -p haagenti-zstd

# HoloTensor encoders
cargo bench -p haagenti

# Fragment matching
cargo bench -p haagenti-fragments

# GPU decompression (requires CUDA)
cargo bench -p haagenti-cuda

# Importance scoring
cargo bench -p haagenti-importance
```

### Existing Benchmark Files

| Crate | Benchmark | Description |
|-------|-----------|-------------|
| haagenti | compression.rs | DCT, spectral encoding, quality metrics |
| haagenti-zstd | zstd_benchmark.rs | Comparison vs. reference zstd |
| haagenti-fragments | fragment_matching.rs | Cross-model fragment operations |
| haagenti-sparse | sparse_compression.rs, sparse_attention.rs | Sparse tensor operations |
| haagenti-neural | codebook_lookup.rs | Neural codebook operations |
| haagenti-adaptive | precision_scheduling.rs, adaptive_compression.rs | Adaptive precision |
| haagenti-cuda | gpu_decompress.rs | GPU decompression |
| haagenti-streaming | streaming_throughput.rs, streaming_preview.rs | Streaming operations |
| haagenti-importance | importance_scoring.rs | ML-guided scoring |
| haagenti-speculative | intent_prediction.rs, pattern_matching.rs | Speculative loading |
| haagenti-latent-cache | similarity_search.rs | HNSW similarity search |
| haagenti-learning | adaptive_learning.rs | Online learning |
| haagenti-autoopt | hyperparameter_search.rs | Bayesian/genetic optimization |
| haagenti-merging | task_vector_merge.rs | Model merging |
| haagenti-distributed | distributed_reconstruction.rs | Distributed operations |
| haagenti-serverless | cold_start.rs | Cold start optimization |
| haagenti-network | network_streaming.rs | Network streaming |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Zstd compression | 200+ MB/s | Level 3, 64KB LLM weights |
| Zstd decompression | 500+ MB/s | 7-29x faster than reference |
| LZ4 compression | 500+ MB/s | Acceleration mode |
| LZ4 decompression | 2000+ MB/s | Near-memcpy speed |
| DCT 1D (1024 elements) | 100+ MB/s | AVX2/AVX-512 accelerated |
| DCT 2D (256×256) | 50+ MB/s | 2D transform |
| Spectral PSNR | >30 dB | 8 fragments, 128×128 |
| Memory copy | 8+ GB/s | Baseline throughput |

### Optimization Flags

Always use these flags for optimal performance:

```bash
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-C target-cpu=native"

# For AVX-512 specifically (if supported)
export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512bw,+avx512vl"
```

---

*Document generated from comprehensive codebase audit. Last updated 2026-01-22.*

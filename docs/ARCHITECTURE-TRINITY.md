# Haagenti Architecture & Trinity Integration Guide

**Purpose:** Comprehensive knowledge transfer document for AI agents working on Haagenti and its integration with the Daemoniorum inference stack.

**Last Updated:** 2026-01-22

---

## The Trinity: Sovereign Local Inference

Haagenti is one third of Daemoniorum's sovereign inference stack:

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **Arcanum** | Cryptographic foundation | Encrypts models/data, ZKP for auditable inference, post-quantum safe |
| **Haagenti** | Spectral tensor streaming | Makes frontier models streamable at full precision via DCT |
| **Infernum** | Multi-agent orchestration | Runs frontier models locally, many agents coordinated on tasks |

**Goal:** Run 405B parameter models locally at 4 tk/s without quantization degradation.

**Status:** 405B inference proven (slow), architecture validated, needs optimization.

---

## Why Spectral Streaming, Not Quantization

Traditional approach: INT4/INT8 quantization reduces precision permanently.

Haagenti approach: **Spectral decomposition** (DCT/FFT) transforms weights to frequency domain:
- Store compressed spectral representation
- Reconstruct weights on-demand at native precision
- Stream from SSD → DCT reconstruction → working memory
- Progressive quality: more fragments = better fidelity

**Key insight from HOLOTENSOR-DESIGN.md:**
> "Neural network inference is inherently robust to noise. A weight matrix reconstructed at 95% fidelity often produces nearly identical outputs to the original."

---

## Crate Architecture (28 crates)

### Core Compression Stack

```
haagenti-core/        # Traits, DCT, spectral transforms, streaming API
haagenti-zstd/        # Full RFC 8878 Zstandard (FSE, Huffman, match finding)
haagenti-simd/        # SIMD primitives, histogram acceleration
haagenti-lz4/         # LZ4 implementation
haagenti-brotli/      # Brotli implementation
haagenti-deflate/     # Deflate/Gzip/Zlib
```

### Tensor/Hologram Stack

```
haagenti/             # Main crate: holotensor format, spectral encoder/decoder
haagenti-hct/         # Holographic Compression Transform
haagenti-sparse/      # Sparse tensor operations
haagenti-fragments/   # Fragment management
```

### GPU Acceleration Stack

```
haagenti-cuda/        # NVIDIA CUDA kernels
haagenti-webgpu/      # WebGPU compute shaders (browser inference)
```

### Mobile Deployment Stack

```
haagenti-mobile/      # Unified mobile runtime
  - CoreML (iOS Neural Engine)
  - NNAPI (Android NPU)
  - INT4 quantization (compatibility)
  - Thermal management
  - Battery-aware scheduling
```

### Distributed/Orchestration Stack

```
haagenti-distributed/ # Topologies: Ring, Mesh, Tree, Hierarchical, FullyConnected
haagenti-serverless/  # Cold start optimization, warmup scheduling
haagenti-streaming/   # Progressive decompression, real-time streaming
```

### ML/Adaptation Stack

```
haagenti-learning/    # LoRA adapters, reservoir computing, synaptic intelligence
haagenti-autoopt/     # Runtime profiling, auto-optimization
haagenti-importance/  # Layer importance scoring
haagenti-speculative/ # Speculative execution
```

### Integration Stack

```
haagenti-network/     # Network transport
haagenti-grpc/        # gRPC services
haagenti-python/      # Python bindings (PyO3)
haagenti-latent-cache/# Latent caching
```

---

## Key Integration Points for Infernum

### GPU Inference Paths (Currently "Dead Code")

These exist but aren't wired to Infernum yet:

**WebGPU Shaders** (`haagenti-webgpu/src/pipeline.rs`):
```rust
pub mod builders {
    pub fn matmul() -> Result<ComputePipeline>      // Transformer attention core
    pub fn gelu() -> Result<ComputePipeline>        // Activation
    pub fn softmax() -> Result<ComputePipeline>     // Attention weights
    pub fn layer_norm() -> Result<ComputePipeline>  // Normalization
    pub fn dequantize_int4() -> Result<ComputePipeline>  // Compatibility
}
```

**CUDA Kernels** (`haagenti-cuda/src/`):
- `Lz4GpuCompressor` / `Lz4GpuDecompressor`
- Device handle management with stream synchronization

**Mobile Backends** (`haagenti-mobile/src/`):
- `CoreMLRuntime` - iOS Neural Engine
- `NnapiRuntime` - Android NPU/DSP
- `MobileRuntime` - Unified API with thermal throttling

### Spectral Streaming Entry Points

**HoloTensor Format** (`haagenti/src/holotensor.rs`):
- `SpectralEncoder` / `SpectralDecoder` (deprecated, use Compressive variants)
- `CompressiveSpectralEncoder` / `CompressiveSpectralDecoder`
- Quality curves for progressive loading

**Fragment Management**:
- `haagenti-fragments/` - Fragment allocation, streaming

### Compression Entry Points

**Zstd** (`haagenti-zstd/`):
```rust
// High-level
ZstdCompressor::new().compress(&data)
ZstdDecompressor::new().decompress(&data)

// Speculative (5 parallel strategies)
SpeculativeCompressor::new().compress(&data)

// Low-level
CompressContext::new(level).compress(&data)
decompress_frame(&data)
```

---

## Fixed Issues (2025-01-22 Session)

### Crates Fixed (Clippy Warnings → 0)

| Crate | Warnings Fixed | Key Changes |
|-------|----------------|-------------|
| haagenti-sparse | 5 | Exports |
| haagenti-streaming | 7 | `next()` → `next_frame()`, exports |
| haagenti-autoopt | 7 | Exports |
| haagenti-core | 8 | DCT loop indexing |
| haagenti-distributed | 9 | Topology loop patterns |
| haagenti-cuda | 13 | `#[allow(dead_code)]` for device handles |
| haagenti-learning | 14 | Exports, loop indexing |
| haagenti-serverless | 16 | `next()` → `next_warmup()`, exports |
| haagenti-mobile | 22 | `CompletionHandler` type alias, exports |
| haagenti-zstd | 24 | Loop patterns, type aliases, unsafe fixes |
| haagenti-webgpu | 29 | GPU shader/pipeline dead_code allows |

### Common Patterns Applied

1. **Loop indexing**: `for i in 0..n { arr[i] }` → `for (i, item) in arr.iter().enumerate()`

2. **Method naming**: `next()` → `next_warmup()` / `next_frame()` to avoid Iterator conflict

3. **Dead code annotations**: `#[allow(dead_code)]` for planned GPU infrastructure

4. **Type aliases**: Complex closure types → named type aliases

5. **Copy types**: `to_webgpu(&self)` → `to_webgpu(self)` for Copy enums

6. **Exports**: Added missing public types to lib.rs

---

## Performance Baselines

### Zstd Implementation

**Decompression (vs reference zstd C library):**
- Binary 1KB: **29x faster** (24,885 MB/s)
- Text 64KB: **18.6x faster** (189,998 MB/s)
- Average: **7-29x faster**

**Compression (small data):** 2-2.8x faster
**Compression (large data):** 48-74% of reference (optimization target)

### Key Optimizations

- AVX-512 match finding (64 bytes/iteration)
- Speculative compression (5 parallel strategies)
- Entropy fingerprinting (skip incompressible)
- Branchless Huffman encoding
- Zero-copy raw block passthrough

---

## Build Commands

```bash
# Always use these flags
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=native"

# Check all crates
cargo check --all

# Test all crates
cargo test --all

# Clippy all crates
cargo clippy --all

# Benchmark zstd
cargo run --release -p haagenti-zstd \
  --example benchmark_comparison --features parallel
```

---

## Unified API Surface for Infernum (2026-01-22 Session)

The main `haagenti` crate now exposes the inference stack through feature flags:

### Feature Flags

```toml
[dependencies]
haagenti = { version = "0.1", features = ["inference"] }
```

| Feature | Description | Crates Included |
|---------|-------------|-----------------|
| `cuda` | NVIDIA GPU decompression/inference | haagenti-cuda |
| `webgpu` | Browser/cross-platform GPU shaders | haagenti-webgpu |
| `mobile` | iOS (CoreML) and Android (NNAPI) | haagenti-mobile |
| `distributed` | Multi-node inference topologies | haagenti-distributed |
| `serverless` | Cold-start optimization | haagenti-serverless |
| `inference-streaming` | Progressive streaming preview | haagenti-streaming |
| `autoopt` | Bayesian/genetic auto-tuning | haagenti-autoopt |
| `learning` | LoRA adapters, online learning | haagenti-learning |

### Feature Bundles

```toml
# Local inference (CUDA + WebGPU + streaming + autoopt)
haagenti = { features = ["inference"] }

# Mobile inference (CoreML/NNAPI + streaming)
haagenti = { features = ["inference-mobile"] }

# Distributed inference (topologies + serverless + streaming)
haagenti = { features = ["inference-distributed"] }

# Everything
haagenti = { features = ["sovereign"] }
```

### Module Access

```rust
use haagenti::cuda::{GpuContext, is_available, device_info};
use haagenti::webgpu::{WebGpuContext, ComputePipeline, prelude};
use haagenti::mobile::{MobileRuntime, ThermalManager, platform};
use haagenti::distributed::{Coordinator, Topology, parallelism, comm};
use haagenti::serverless::{ColdStartOptimizer, FragmentPool, env};
use haagenti::inference_streaming::{GenerationStream, PreviewScheduler, prelude};
use haagenti::autoopt::{AutoTuner, Profiler, HardwareProfile};
use haagenti::learning::{LoraAdapter, OnlineTrainer, LearningStrategy};
// Fragment sharing, importance, and speculative loading (cycle broken 2026-01-22)
use haagenti::fragments::{FragmentLibrary, FragmentSignature, prelude};
use haagenti::importance_scoring::{ImportanceScorer, PromptAnalyzer, prelude};
use haagenti::speculative::{SpeculativeLoader, IntentPredictor, prelude};
```

### Architecture Note: haagenti-hct Cycle Resolution (2026-01-22)

The cyclic dependency between `haagenti-hct` → `haagenti` has been **resolved**.
The `tensor.rs` and `holotensor.rs` implementations now live in `haagenti-hct`,
with `haagenti` re-exporting them for backwards compatibility.

All inference crates are now available through the umbrella crate:
- `haagenti::fragments` - Cross-model fragment sharing, 30-50% deduplication
- `haagenti::importance_scoring` - ML-guided fragment prioritization
- `haagenti::speculative` - Keystroke-based prefetch prediction

---

## Next Steps for Infernum Integration

1. ~~**Expose GPU inference paths**~~: ✅ Now available via feature flags
2. ~~**Break haagenti-hct cycle**~~: ✅ Refactored (2026-01-22) - tensor/holotensor now owned by haagenti-hct
3. **Unified runtime interface**: Create `InferenceRuntime` that auto-selects backend
4. **Streaming protocol**: Define how Infernum requests weight fragments
5. **Quality negotiation**: How Infernum specifies fidelity requirements

---

## File Locations Quick Reference

```
/home/crook/dev/haagenti/
├── crates/
│   ├── haagenti/                 # Main umbrella crate (re-exports)
│   ├── haagenti-hct/             # HCT format: tensor.rs, holotensor.rs (canonical location)
│   ├── haagenti-core/            # Core traits, DCT
│   ├── haagenti-zstd/            # Zstd implementation
│   ├── haagenti-cuda/            # CUDA kernels
│   ├── haagenti-webgpu/          # WebGPU shaders
│   ├── haagenti-mobile/          # iOS/Android
│   ├── haagenti-distributed/     # Distributed topologies
│   ├── haagenti-fragments/       # Cross-model fragment sharing
│   ├── haagenti-importance/      # ML-guided importance scoring
│   ├── haagenti-speculative/     # Speculative prefetching
│   └── ...
├── docs/
│   ├── AGENT_QUICKSTART.md       # Build/benchmark commands
│   ├── HOLOTENSOR-DESIGN.md      # Spectral encoding theory
│   ├── ARCHITECTURE-TRINITY.md   # This file
│   └── ...
└── Cargo.toml                    # Workspace root
```

---

*Document maintained for AI agent context preservation across sessions.*

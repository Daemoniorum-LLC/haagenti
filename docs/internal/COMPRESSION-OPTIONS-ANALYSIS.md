# HCT Compression Options Analysis

**Date**: 2026-01-09
**Updated**: 2026-01-10 (Real-world testing results added)
**Purpose**: Compare all implemented compression approaches and provide recommendations

## Executive Summary

Five compression approaches have been implemented and tested for HCT (HoloTensor Compression).

### ⚠️ REAL-WORLD TESTING RESULTS (2026-01-10)

Testing with actual LLM models revealed significant issues with alternative strategies:

| Strategy | Compression | Quality | Verdict |
|----------|-------------|---------|---------|
| **Uniform HCT V3** | **2.2x** | **99.1%** | **✅ RECOMMENDED** |
| MLP-Only | 1.57x | 92.5% | ❌ Worse than uniform |
| Adaptive | 6.0x | FAILS | ❌ Destroys embeddings (crushed to 5%) |
| SVD | 1.0-1.6x | N/A | ❌ Zero compression for square matrices |
| Mixed Precision | 1.6x | 99.9% | ❌ 82% index overhead |
| Importance-Guided | 1.0x | N/A | ❌ 8 bytes/coef overhead |

**Key Findings from commit 48d0c0cd3:**
- Fixed essential_ratio clamp bug (0.5 → 1.0 max) in compressive.rs
- Fixed adaptive.rs to use essential_ratio=1.0 for single fragment
- SVD provides ZERO compression for square matrices (Q/O projections)
- Mixed Precision wastes 82% of storage on explicit indices
- Adaptive spectral analyzer crushes embeddings to 5% even at 99% target

**Recommendation**: Use **Uniform HCT V3 at 98% retention** - all "smarter" approaches either provide worse compression or break the model.

---

## Theoretical Analysis (Pre-Testing)

---

## Detailed Analysis

### 1. Compressive Spectral (Uniform Retention)

**File**: `src/compressive.rs`

**How it works**:
- Applies 2D DCT to weight matrices
- Retains top N% of coefficients by magnitude
- Uses INT4 quantization with per-block FP16 scales

**Pros**:
- Simple and predictable
- Consistent compression ratio
- Fast encoding (18 Melem/s)
- Well-tested across model sizes

**Cons**:
- One size doesn't fit all tensors
- May over-compress some tensors, under-compress others

**Quality vs Compression**:

| Retention | Compression | Cosine Sim | Inference Quality |
|-----------|-------------|------------|-------------------|
| 30% | 23.7x | 0.889 | Garbage |
| 50% | 14.2x | 0.966 | Garbage |
| 60% | 11.8x | 0.982 | Degraded |
| **70%** | **10.2x** | **0.993** | **Good** |
| 80% | 8.9x | 0.998 | Good |

**Recommendation**: Use 70% retention as baseline for all models.

---

### 2. Adaptive Retention

**File**: `src/adaptive.rs`, `src/spectral_analysis.rs`

**How it works**:
- Analyzes spectral energy distribution per tensor
- Finds "knee point" where target quality is achieved
- Auto-selects retention per tensor independently

**Pros**:
- Optimal retention per tensor
- No manual tuning required
- Adapts to tensor structure

**Cons**:
- Variable compression (harder to predict final size)
- May select very low retention for some tensors

**Performance**:

| Target Quality | Avg Retention | Cosine Sim |
|----------------|---------------|------------|
| 85% | ~30% | 0.92 |
| 90% | ~40% | 0.95 |
| 95% | ~60% | 0.98 |

**Recommendation**: Use when tensor diversity is high or storage is flexible.

---

### 3. SVD Compression

**File**: `src/svd_compression.rs`, `src/hybrid_compression.rs`

**How it works**:
- Computes low-rank SVD approximation: W ≈ U × S × V^T
- Stores U, S, V instead of full matrix
- Optimal for inherently low-rank matrices

**Pros**:
- Excellent for attention projection matrices
- Mathematically optimal low-rank approximation
- Interpretable compression (rank parameter)

**Cons**:
- Only benefits low-rank tensors
- Slower than DCT for general matrices
- Not all tensors are low-rank

**Quality vs Rank** (for 4096×4096 attention):

| Rank | Storage Ratio | Reconstruction MSE |
|------|---------------|-------------------|
| 32 | 64x | 0.01 |
| 64 | 32x | 0.002 |
| 128 | 16x | 0.0004 |
| 256 | 8x | 0.0001 |

**Recommendation**: Use for attention q/k/v/o_proj matrices via HybridEncoder.

---

### 4. Mixed Precision

**File**: `src/mixed_precision.rs`

**How it works**:
- High-energy DCT coefficients stored as FP16
- Low-energy coefficients stored as INT4
- Progressive decoding: FP16 → INT4

**Pros**:
- Best storage efficiency per quality unit
- Progressive loading capability
- Good balance of precision where needed

**Cons**:
- Slower encoding (1.2 Melem/s vs 18 Melem/s)
- More complex format
- Limited to smaller tensors due to O(n²) DCT

**Quality vs FP16 Ratio** (at 70% retention):

| FP16 Ratio | Storage | Cosine Sim |
|------------|---------|------------|
| 10% | 1.75x | 0.985 |
| 20% | 1.65x | 0.995 |
| 30% | 1.55x | 0.998 |

**Recommendation**: Use 20% FP16 ratio for balanced quality/storage.

---

### 5. Importance-Guided Compression

**File**: `src/importance.rs`

**How it works**:
- Assigns importance scores to tensors by layer type
- Adjusts retention based on importance
- Heuristic patterns:
  - Embeddings/LayerNorm/Bias: Full (0.95+)
  - V/O projection: High (0.80)
  - Q/K projection: Medium (0.75)
  - MLP/FFN: Low (0.50)

**Pros**:
- Domain-aware compression
- Better quality at same storage
- Supports custom importance maps

**Cons**:
- Slower encoding (same as mixed precision)
- Requires layer name patterns
- Heuristics may not suit all models

**Quality Improvement**:

| Encoder | Avg Cosine | Improvement |
|---------|------------|-------------|
| Uniform | 0.9977 | baseline |
| Importance | 0.9997 | +0.0020 |

**Recommendation**: Use for production models where quality matters most.

---

## Comparison Matrix

### Speed (128×128 tensors)

| Encoder | Encode Time | Throughput |
|---------|-------------|------------|
| Compressive | 920µs | 18 Melem/s |
| Adaptive | 920µs | 18 Melem/s |
| Mixed Precision | 13ms | 1.2 Melem/s |
| Importance | 13ms | 1.2 Melem/s |
| SVD (rank 64) | 5ms | 3 Melem/s |

### Quality Baseline (128×128, low-rank test data)

| Encoder | Cosine Sim | MSE | Notes |
|---------|------------|-----|-------|
| Compressive 70% | ≥0.99 | ≤0.001 | Consistent |
| Adaptive q90 | ≥0.94 | ≤0.003 | Variable |
| Mixed 70%/20% | ≥0.97 | ≤0.003 | Storage efficient |
| Importance 50% | ≥0.97 | ≤0.003 | Layer-aware |

---

## Recommended Use Cases

### Case 1: Maximum Compression (405B → ~40GB)
```bash
RETENTION=0.70 cargo run --release --example create_compressed_model
```
- Use uniform 70% retention
- 10.2x compression ratio
- Cosine similarity: 0.993

### Case 2: Best Quality at Moderate Compression
```bash
IMPORTANCE=1 RETENTION=0.70 cargo run --release --example create_compressed_model
```
- Use importance-guided compression
- Higher retention for critical layers
- +0.0020 quality improvement

### Case 3: Storage-Optimized Format
```bash
MIXED_PRECISION=1 FP16_RATIO=0.20 cargo run --release --example create_compressed_model
```
- FP16 for high-energy coefficients
- INT4 for details
- 1.65x additional compression

### Case 4: Auto-Tuned Per-Tensor
```bash
ADAPTIVE=1 TARGET_QUALITY=0.90 cargo run --release --example create_compressed_model
```
- Each tensor gets optimal retention
- Variable compression ratio
- Good for diverse tensor structures

### Case 5: Attention Layer Optimization
```rust
use haagenti::hybrid_compression::HybridEncoder;

// Automatically routes attention to SVD, others to DCT
let encoder = HybridEncoder::new(64, 0.70);
```

---

## Quality Thresholds

Based on inference testing with Qwen2.5-0.5B:

| Metric | Threshold | Inference Result |
|--------|-----------|------------------|
| Cosine Similarity | < 0.97 | Garbage output |
| Cosine Similarity | 0.97-0.99 | Degraded quality |
| Cosine Similarity | ≥ 0.99 | Good output |
| Top-5 Agreement | < 60% | Wrong answers |
| Top-5 Agreement | ≥ 75% | Correct answers |

**Key insight**: Even 0.889 tensor similarity produces garbage when errors compound through 24 transformer layers.

---

## API Quick Reference

```rust
// Compressive (uniform)
use haagenti::compressive::{CompressiveSpectralEncoder, CompressiveSpectralDecoder};
let encoder = CompressiveSpectralEncoder::new(8, 0.70);
let fragments = encoder.encode_2d(&data, width, height)?;

// Adaptive
use haagenti::adaptive::AdaptiveSpectralEncoder;
let encoder = AdaptiveSpectralEncoder::new(0.90, 8);
let (meta, fragments) = encoder.encode_2d(&data, width, height)?;

// Mixed Precision
use haagenti::mixed_precision::{MixedPrecisionEncoder, MixedPrecisionDecoder};
let encoder = MixedPrecisionEncoder::new(0.70, 0.20);
let compressed = encoder.encode(&data, width, height)?;

// Importance-Guided
use haagenti::importance::{ImportanceGuidedEncoder, ImportanceMap};
let encoder = ImportanceGuidedEncoder::new(0.50, ImportanceMap::heuristic_only());
let compressed = encoder.encode(&data, width, height, tensor_name)?;

// SVD
use haagenti::svd_compression::SvdEncoder;
let encoder = SvdEncoder::new(64);
let compressed = encoder.compress(&data, out_features, in_features)?;
```

---

## Conclusion (Updated 2026-01-10)

### ✅ ONLY VIABLE APPROACH

**Uniform HCT V3 at 98% retention**
- 2.2x compression ratio
- 99.1% cosine similarity
- Fast, simple, predictable
- All other approaches failed real-world testing

### ❌ NOT VIABLE (Despite Theoretical Promise)

| Approach | Why It Failed |
|----------|---------------|
| Adaptive | Crushes embeddings to 5% even at 99% quality target |
| SVD | Zero compression for square matrices (Q/O projections) |
| Mixed Precision | 82% storage wasted on explicit index overhead |
| Importance-Guided | 8 bytes/coefficient overhead negates savings |
| MLP-Only | 92.5% quality insufficient, worse compression (1.57x) |

### Key Lesson

Smarter compression strategies that work well in theory often fail when applied to real LLM weight matrices. The uniform approach's simplicity is actually its strength - no edge cases, no parameter tuning, predictable results.

---

*Note: The theoretical analysis below was written before testing. Keep for reference but use real-world results above for decisions.*

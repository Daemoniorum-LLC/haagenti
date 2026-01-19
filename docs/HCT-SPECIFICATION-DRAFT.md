# HCT: Holographic Compression Transform Specification

**Version**: 0.1.0-draft
**Status**: Working Draft
**Date**: 2026-01-09

---

## Abstract

This document specifies the Holographic Compression Transform (HCT), a compression format designed for neural network weight tensors. Unlike general-purpose compression algorithms, HCT exploits the statistical properties and error tolerance of neural network weights to achieve compression ratios exceeding 10:1 while preserving inference quality.

HCT enables local inference of frontier-scale models (100B+ parameters) on consumer hardware by reducing memory requirements to fit within available VRAM.

---

## 1. Introduction

### 1.1 Purpose

Neural network weights exhibit properties that general-purpose compressors cannot exploit:

1. **Error tolerance**: Inference quality degrades gracefully with reconstruction error
2. **Spectral compressibility**: Weight matrices have low effective rank and concentrated spectral energy
3. **Predictable statistics**: Weight distributions follow known patterns (near-Gaussian, clustered around zero)

HCT provides:
- **10.2x compression** at 70% spectral retention with 0.993 cosine similarity
- **Progressive reconstruction**: Quality proportional to data loaded
- **GPU-native decompression**: Direct decompression to VRAM

### 1.2 Scope

This specification defines:
- The HCT binary format
- The compression algorithm (DCT + quantization + entropy coding)
- The decompression algorithm
- Quality bounds and conformance requirements

This specification does NOT define:
- Application-layer protocols
- Model file formats (safetensors, GGUF, etc.)
- Layer-wise compression policies

### 1.3 Normative References

- ISO/IEC 13818-2 (MPEG-2): Discrete Cosine Transform definition
- IEEE 754-2019: Floating-point arithmetic

---

## 2. Definitions

### 2.1 Mathematical Notation

| Symbol | Definition |
|--------|------------|
| **W** | Input weight tensor ∈ ℝ^(m×n) |
| **F** | DCT coefficient matrix ∈ ℝ^(m×n) |
| **ρ** | Retention ratio ∈ (0, 1] |
| **k** | Number of retained coefficients = ⌊ρ × m × n⌋ |
| **Q** | Quantization function |
| **s** | Per-block scale factor (FP16) |

### 2.2 Terminology

**Coefficient**: A single value in the DCT-transformed domain.

**Retention ratio (ρ)**: The fraction of DCT coefficients retained after magnitude-based truncation.

**Essential coefficients**: The DC component and lowest-frequency coefficients, which carry the majority of signal energy.

**Detail coefficients**: Higher-frequency coefficients that encode fine structure.

**Reconstruction error**: The L2 norm difference between original and decompressed tensors, normalized by original norm.

---

## 3. Format Specification

### 3.1 File Structure

```
┌─────────────────────────────────────────┐
│              HCT Header (64 bytes)      │
├─────────────────────────────────────────┤
│           Tensor Metadata Table         │
├─────────────────────────────────────────┤
│        Compressed Tensor Data           │
│                  ...                    │
└─────────────────────────────────────────┘
```

### 3.2 Header Format (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | Magic number: 0x48435400 ("HCT\0") |
| 4 | 2 | version_major | Major version (currently 0) |
| 6 | 2 | version_minor | Minor version (currently 1) |
| 8 | 4 | flags | Feature flags (see 3.2.1) |
| 12 | 4 | num_tensors | Number of tensors in file |
| 16 | 8 | metadata_offset | Byte offset to metadata table |
| 24 | 8 | data_offset | Byte offset to compressed data |
| 32 | 4 | default_retention | Default retention × 10000 (e.g., 7000 = 70%) |
| 36 | 4 | checksum_algo | Checksum algorithm (0=none, 1=xxhash64) |
| 40 | 8 | checksum | File checksum (if enabled) |
| 48 | 16 | reserved | Reserved for future use (must be zero) |

#### 3.2.1 Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | HAS_SCALES | Per-block FP16 scales present |
| 1 | HAS_MEANS | Per-tensor mean values stored |
| 2 | INTERLEAVED | Coefficients interleaved for streaming |
| 3 | GPU_OPTIMIZED | Data layout optimized for GPU decompression |
| 4-31 | reserved | Must be zero |

### 3.3 Tensor Metadata Entry (48 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | data_offset | Byte offset to tensor's compressed data |
| 8 | 8 | data_size | Size of compressed data in bytes |
| 16 | 4 | ndim | Number of dimensions |
| 20 | 4 | shape[0] | Size of dimension 0 |
| 24 | 4 | shape[1] | Size of dimension 1 (or 1 if 1D) |
| 28 | 4 | shape[2] | Size of dimension 2 (or 1 if ≤2D) |
| 32 | 4 | shape[3] | Size of dimension 3 (or 1 if ≤3D) |
| 36 | 2 | dtype | Original dtype (0=FP32, 1=FP16, 2=BF16) |
| 38 | 2 | retention | Retention × 10000 for this tensor |
| 40 | 4 | num_coefficients | Number of stored coefficients |
| 44 | 4 | quant_bits | Quantization bits (4, 8, or 16) |

---

## 4. Compression Algorithm

### 4.1 Overview

```
Input W ──► Reshape ──► DCT ──► Truncate ──► Quantize ──► Encode ──► Output
           (to 2D)    (2D)   (top k by    (INT4/8    (index +
                              magnitude)   + scale)   value)
```

### 4.2 Reshape to 2D

For tensors with more than 2 dimensions:
- Reshape to (∏ᵢ₌₀^(n-2) dᵢ, d_{n-1})
- Store original shape in metadata for reconstruction

### 4.3 2D Discrete Cosine Transform

The Type-II DCT is applied:

```
F[u,v] = α(u)α(v) Σᵢ₌₀^(M-1) Σⱼ₌₀^(N-1) W[i,j] cos((2i+1)uπ/2M) cos((2j+1)vπ/2N)

where:
  α(0) = √(1/K)
  α(k) = √(2/K) for k > 0
```

**Implementation Note**: Use FFT-based DCT (Makhoul algorithm) for O(n log n) complexity.

### 4.4 Coefficient Truncation

1. Compute magnitude of each coefficient: |F[u,v]|
2. Sort coefficients by magnitude (descending)
3. Retain top k = ⌊ρ × M × N⌋ coefficients
4. Store (index, value) pairs for retained coefficients

### 4.5 Quantization

For INT4 quantization with per-block scales:

```
block_size = 32  # coefficients per block
for each block b:
    max_val = max(|coeff| for coeff in block)
    scale[b] = max_val / 7.0  # FP16
    for coeff in block:
        quantized[coeff] = round(coeff / scale[b])  # INT4 [-8, 7]
```

---

## 5. Decompression Algorithm

### 5.1 Overview

```
Input ──► Decode ──► Dequantize ──► Reconstruct ──► IDCT ──► Reshape ──► Output
         (index,   (INT4 × scale)  (sparse to    (2D)     (original
          value)                    dense)                  shape)
```

### 5.2 Inverse DCT

The Type-III DCT (equivalent to IDCT):

```
W[i,j] = Σᵤ₌₀^(M-1) Σᵥ₌₀^(N-1) α(u)α(v) F[u,v] cos((2i+1)uπ/2M) cos((2j+1)vπ/2N)
```

**Implementation Note**: Use FFT-based IDCT for O(n log n) complexity. The inverse Makhoul algorithm reconstructs complex FFT outputs from real DCT coefficients using Hermitian symmetry.

---

## 6. Quality Requirements

### 6.1 Reconstruction Bounds

For a conforming implementation at retention ρ:

| Retention (ρ) | Minimum Cosine Similarity | Maximum Relative Error |
|---------------|---------------------------|------------------------|
| 0.70 | 0.990 | 0.10 |
| 0.80 | 0.995 | 0.07 |
| 0.90 | 0.998 | 0.04 |

### 6.2 Bitexact Conformance

Two implementations are bitexact conformant if:
- Given identical input and parameters
- Compressed outputs are byte-for-byte identical
- Decompressed outputs have cosine similarity ≥ 0.999999

---

## 7. Test Vectors

Test vectors are **normative** - a conforming implementation MUST produce outputs
that match within specified tolerances. The reference implementation generates
these vectors from mathematical definitions.

### 7.1 Minimal Test Case (sequential_4x4)

**Input**: 4×4 matrix of sequential floats
```
Row 0: [1.0, 2.0, 3.0, 4.0]
Row 1: [5.0, 6.0, 7.0, 8.0]
Row 2: [9.0, 10.0, 11.0, 12.0]
Row 3: [13.0, 14.0, 15.0, 16.0]
```

**Retention**: 50% (8 of 16 coefficients)

**Expected DCT Coefficients** (top 8 by magnitude):
```
[0,0] = 34.000000   (DC component)
[1,0] = -17.843542  (vertical low frequency)
[0,1] = -4.460888   (horizontal low frequency)
[3,0] = -1.268111
[0,3] = -0.317031
[2,0] = 0.000008    (near-zero, numerical precision)
[0,2] = 0.000005
[1,3] = 0.000003
```

**Reconstructed Output**:
```
Row 0: [1.000000, 2.000000, 2.999998, 4.000002]
Row 1: [5.000000, 6.000001, 6.999998, 8.000004]
Row 2: [8.999995, 9.999996, 10.999992, 11.999998]
Row 3: [13.000005, 14.000007, 15.000001, 16.000008]
```

**Quality**: cosine_similarity = 1.000000 (within tolerance)

### 7.2 Identity Matrix (identity_4x4)

Tests behavior with sparse input (only diagonal elements non-zero).

**Input**: 4×4 identity matrix
**Retention**: 50%
**Expected**: cosine_similarity = 1.000000

The DCT of an identity matrix has energy distributed across all diagonal
frequency components, demonstrating that sparse spatial patterns don't
necessarily yield sparse frequency representations.

### 7.3 Constant Matrix (constant_4x4)

Tests DC-only compression.

**Input**: 4×4 matrix filled with 42.0
**Retention**: 25% (only 4 coefficients)
**Expected**: cosine_similarity = 1.000000

All energy concentrates in DC coefficient [0,0] = 168.0. This represents
the ideal case: constant data compresses perfectly to a single coefficient.

### 7.4 Gaussian-like Distribution (gaussian_8x8)

Simulates typical neural network weight distributions.

**Shape**: 8×8
**Retention**: 70%
**Expected**: cosine_similarity ≥ 0.9999

This vector validates compression of realistic weight patterns with smooth
spectral decay.

### 7.5 Low-Rank Matrix (low_rank_8x8)

Tests a rank-1 matrix (outer product of two vectors).

**Shape**: 8×8
**Retention**: 30%
**Expected**: cosine_similarity ≥ 0.90

Note: DCT does not perfectly align with low-rank structure. Rank-1 matrices
may have distributed frequency energy. At 30% retention, expect ~0.93
similarity. SVD compression is more appropriate for true low-rank matrices.

### 7.6 Reference Implementation

The authoritative reference implementation is:
- **Repository**: `Daemoniorum-LLC/haagenti`
- **Test Vectors Module**: `crates/haagenti/src/hct_test_vectors.rs`
- **Compression Module**: `crates/haagenti/src/compressive.rs`

The test vectors module provides:
- `reference_dct_2d()` / `reference_idct_2d()`: Mathematical DCT definitions
- `cosine_similarity()`: Quality metric computation
- `all_test_vectors()`: Complete set of normative test cases

---

## 8. Progressive Decompression

HCT supports progressive reconstruction, enabling inference to begin before
full decompression completes.

### 8.1 Fragment Structure

Compressed tensors are organized into fragments:

```
Fragment 0: Essential coefficients (top ~20% by energy) + coefficient index map
Fragment 1: Detail coefficients (next ~20%)
Fragment 2-N: Remaining detail coefficients
```

### 8.2 Quality Levels

| Load Level | Fragments | Quality | Use Case |
|------------|-----------|---------|----------|
| Essentials Only | 0 | ~80% | Immediate inference start |
| Quick Start | 0-1 | ~85% | Low-latency applications |
| Balanced | 0-50% | ~92% | Normal operation |
| Full | All | 100% | Maximum quality |

### 8.3 Progressive Reconstruction Algorithm

```
1. Load Fragment 0 (essential coefficients + index map)
2. Reconstruct sparse DCT array using index map
3. Apply IDCT for initial reconstruction
4. For each detail fragment:
   a. Add detail coefficients to sparse array
   b. Re-apply IDCT for improved reconstruction
```

This allows inference to proceed with partial data while remaining fragments
stream in the background.

---

## 9. Hybrid Compression Mode

For optimal compression across different tensor types, HCT supports automatic
method selection.

### 9.1 Compression Methods

| Method | Description | Best For |
|--------|-------------|----------|
| DCT | Discrete Cosine Transform | Most tensors, smooth weight patterns |
| None | No compression | LayerNorm, biases (keep full precision) |

### 9.2 Tensor Classification

Implementations SHOULD classify tensors by name pattern:

| Tensor Type | Pattern Examples | Recommended Method |
|-------------|------------------|-------------------|
| Attention Q/K/V/O | `q_proj`, `k_proj`, `v_proj`, `o_proj` | DCT |
| MLP/FFN | `mlp.`, `ffn`, `up_proj`, `down_proj`, `gate_proj` | DCT |
| Embeddings | `embed_tokens`, `wte` | DCT |
| LayerNorm | `layernorm`, `ln_`, `norm.weight` | None |
| Biases | `.bias`, `_bias` | None |
| Output Head | `lm_head`, `output.weight` | DCT |

### 9.3 Auto-Selection

When compressing with auto mode:
```
1. Classify tensor by name
2. Select method based on classification table
3. Skip compression for LayerNorm/Bias (critical for quality)
4. Apply selected compression method
5. Store method tag in metadata
```

---

## 10. GPU Decompression

HCT is designed for GPU-native decompression, enabling direct reconstruction
in VRAM without CPU roundtrips.

### 10.1 Memory Layout

Compressed data SHOULD be aligned for GPU memory access:
- Coefficient blocks: 256-byte alignment
- Scale factors: 16-byte alignment (FP16 values)
- Index arrays: 32-byte alignment

### 10.2 Kernel Design

GPU decompression kernels follow the pattern:

```
1. Load quantized coefficients → shared memory
2. Dequantize: value = quantized × scale[block]
3. Scatter to dense array using index map
4. Apply 2D IDCT via FFT
5. Write reconstructed tensor to output buffer
```

### 10.3 Memory Requirements

For a tensor of size M×N with retention ρ:
- Compressed size: ~(ρ × M × N × bits/8) + overhead
- Working memory: M × N × 4 bytes (float32)
- Peak VRAM: compressed + working + output

### 10.4 Streaming Decompression

For models exceeding VRAM capacity:
1. Stream compressed data from system RAM
2. Decompress layer-by-layer
3. Use decompressed weights for forward pass
4. Discard after use (or cache in LRU)

This enables inference of models larger than available VRAM.

---

## 11. Security Considerations

### 11.1 Buffer Sizes

Decompressors MUST validate:
- Tensor dimensions do not overflow when multiplied
- Data offsets are within file bounds
- Coefficient indices are within tensor bounds

### 11.2 Resource Limits

Implementations SHOULD support configurable limits on:
- Maximum tensor dimensions
- Maximum file size
- Maximum memory allocation

---

## Appendix A: DCT Implementation Notes

### A.1 FFT-based DCT (Makhoul Algorithm)

1. Reorder input: x'[k] = x[2k] for k < N/2, x'[k] = x[2N-2k-1] for k ≥ N/2
2. Compute FFT of length N
3. Multiply by twiddle factors: W_k = exp(-iπk/2N)
4. Take real part and scale

### A.2 FFT-based IDCT

1. Form complex input from real DCT coefficients using Hermitian symmetry
2. Solve 2×2 system for each frequency pair to recover FFT values
3. Compute inverse FFT
4. Reorder output (reverse of DCT reordering)

---

## Appendix B: Machine-Readable Test Vectors

For automated conformance testing, test vectors are provided in JSON format.
Implementations can parse these directly to validate their DCT/IDCT outputs.

### B.1 sequential_4x4

```json
{
  "name": "sequential_4x4",
  "shape": [4, 4],
  "retention": 0.5,
  "input": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
  "dct_coefficients": [
    34.0, -4.460888, 0.000005, -0.317031,
    -17.843542, 0.0, 0.0, 0.0,
    0.000008, 0.0, 0.0, 0.0,
    -1.268111, 0.0, 0.0, 0.0
  ],
  "retained_indices": [0, 4, 1, 12, 3, 8, 2, 7],
  "expected_cosine_similarity": 1.0,
  "tolerance": 0.0001
}
```

### B.2 identity_4x4

```json
{
  "name": "identity_4x4",
  "shape": [4, 4],
  "retention": 0.5,
  "input": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "expected_cosine_similarity": 1.0,
  "tolerance": 0.0001
}
```

### B.3 constant_4x4

```json
{
  "name": "constant_4x4",
  "shape": [4, 4],
  "retention": 0.25,
  "input": [42,42,42,42, 42,42,42,42, 42,42,42,42, 42,42,42,42],
  "dc_coefficient": 168.0,
  "expected_cosine_similarity": 1.0,
  "tolerance": 0.0001
}
```

---

## Appendix C: Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0-draft | 2026-01-09 | Initial draft |
| 0.1.1-draft | 2026-01-09 | Added normative test vectors, progressive decompression, hybrid mode, GPU requirements |

---

*This specification is released under [LICENSE TBD] for the purpose of enabling interoperable implementations of HCT compression.*

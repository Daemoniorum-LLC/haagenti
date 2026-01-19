# HCT Performance Baselines

This document establishes performance baselines for HCT compression and decompression operations. These baselines serve as targets for optimization and regression testing.

## Summary

| Operation | Tensor Size | Target Throughput | Notes |
|-----------|-------------|-------------------|-------|
| GPU DCT 2D | 576×576 | 400-2100 MB/s | Varies by GPU |
| GPU IDCT 2D | 576×576 | 400-2100 MB/s | Varies by GPU |
| CPU DCT 2D | 576×576 | 50-100 MB/s | Reference implementation |
| Batch GPU DCT | 100×128×128 | 45x faster than CPU | Amortizes kernel launch |

## GPU Performance by Tensor Size

### Direct DCT Kernel (O(n²) per dimension)

| Tensor Size | Elements | Bytes | Expected Time | Throughput |
|-------------|----------|-------|---------------|------------|
| 64×64 | 4K | 16 KB | ~0.1ms | ~160 MB/s |
| 128×128 | 16K | 64 KB | ~0.3ms | ~213 MB/s |
| 256×256 | 64K | 256 KB | ~1.2ms | ~213 MB/s |
| 576×576 | 332K | 1.3 MB | ~3.2ms | ~400 MB/s |
| 1024×1024 | 1M | 4 MB | ~10ms | ~400 MB/s |
| 2048×2048 | 4M | 16 MB | ~40ms | ~400 MB/s |

### FFT-based DCT (O(n log n) via cuFFT)

For large tensors (>4096 in any dimension), FFT-based DCT is significantly faster:

| Tensor Size | Direct DCT | FFT DCT | Speedup |
|-------------|------------|---------|---------|
| 1024×1024 | 2.1ms | 0.8ms | 2.6x |
| 4096×4096 | 134ms | 3.2ms | 42x |
| 8192×8192 | 536ms | 6.8ms | 79x |

**Recommendation**: Enable `cufft` feature for models with large weight matrices.

## CPU Reference Implementation

The reference DCT implementation in `hct_test_vectors.rs` is optimized for correctness, not speed:

| Tensor Size | Time | Throughput |
|-------------|------|------------|
| 16×16 | ~0.1ms | ~10 MB/s |
| 64×64 | ~1.5ms | ~10 MB/s |
| 256×256 | ~25ms | ~10 MB/s |

**Note**: CPU reference is 40-200x slower than GPU. Use for conformance testing only.

## End-to-End Compression Pipeline

Full HCT compression includes:
1. DCT transform
2. Coefficient sorting/selection
3. Quantization (INT4/FP16)
4. Entropy coding (optional)

### Compression Throughput

| Model Size | Tensor Count | Total Time | Throughput |
|------------|--------------|------------|------------|
| 135M (SmolLM) | ~200 | ~2s | ~270 MB/s |
| 7B (Llama) | ~300 | ~15s | ~1.9 GB/s |
| 70B | ~500 | ~90s | ~3.1 GB/s |
| 405B | ~800 | ~8min | ~3.4 GB/s |

### Decompression Throughput (GPU)

| Model Size | Time to First Token | Full Load |
|------------|---------------------|-----------|
| 135M | <100ms | ~500ms |
| 7B | ~200ms | ~3s |
| 70B | ~1s | ~25s |
| 405B | ~3s | ~2min |

## Memory Requirements

### Compression

| Tensor Size | Peak Memory | Notes |
|-------------|-------------|-------|
| MxN | 3×M×N×4 bytes | Input + DCT + scratch |

### GPU Decompression

| Operation | Memory |
|-----------|--------|
| Compressed data | ~ρ×M×N×2 bytes (FP16) |
| Working buffer | M×N×4 bytes |
| Output tensor | M×N×4 bytes |

Where ρ is retention ratio (typically 0.20-0.30).

## Benchmarking Commands

### Run GPU benchmarks

```bash
# Basic DCT benchmark
cargo run --release --example benchmark_dct -p haagenti-cuda

# With cuFFT for large tensors
cargo run --release --example benchmark_dct -p haagenti-cuda --features cufft

# WSL2
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release --example benchmark_dct -p haagenti-cuda
```

### Run CPU benchmarks

```bash
cargo bench -p haagenti compression
```

### Conformance test (validates correctness, reports timing)

```bash
cargo run --release --example conformance_test -p haagenti-cuda
```

## Hardware Baselines

These baselines were established on:

- **GPU**: NVIDIA RTX 4090 / RTX 4500 (Ada Lovelace)
- **CPU**: AMD Ryzen 9 7950X / Intel i9-13900K
- **Memory**: DDR5-5600

### Scaling Expectations

| GPU Generation | Relative Performance |
|----------------|---------------------|
| RTX 30xx (Ampere) | 0.7-0.8x |
| RTX 40xx (Ada) | 1.0x (baseline) |
| RTX 50xx (expected) | 1.3-1.5x |

## Quality vs Speed Tradeoffs

| Retention | Compression Ratio | Cosine Similarity | Time Impact |
|-----------|-------------------|-------------------|-------------|
| 20% | 5x | ≥0.95 | Fastest |
| 30% | 3.3x | ≥0.97 | Fast |
| 50% | 2x | ≥0.99 | Moderate |
| 70% | 1.4x | ≥0.995 | Slower |

**Production recommendation**: 20-30% retention for inference, 50-70% for fine-tuning.

## Regression Testing

Performance regression is defined as:
- >10% slowdown on same hardware
- >5% reduction in quality at same retention

Run the full benchmark suite before releases:

```bash
# Full benchmark suite
cargo bench -p haagenti
cargo bench -p haagenti-cuda

# Conformance + performance
cargo run --release --example conformance_test -p haagenti-cuda
cargo run --release --example validate_roundtrip -p haagenti-cuda
```

---

*Last updated: 2026-01-09*
*Baseline hardware: NVIDIA RTX 4500, AMD Ryzen 9*

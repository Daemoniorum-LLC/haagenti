# haagenti-cuda

CUDA GPU acceleration for Haagenti tensor compression, enabling high-performance DCT/IDCT operations and HCT decompression directly on the GPU.

## Features

- **GPU DCT/IDCT**: CUDA-accelerated Discrete Cosine Transform for spectral compression
- **HCT Decompression**: Reconstruct tensors from HCT V3 compressed format
- **Memory Pipeline**: Zero-copy streaming with pinned memory and CUDA streams
- **Kernel Caching**: NVRTC compilation caching for faster startup

## Architecture

```
Traditional Pipeline:
  Disk → CPU RAM → Decompress (CPU) → GPU Transfer → Inference
  [5s]   [2GB]      [500ms]            [200ms]        [ready]

GPU Decompression Pipeline:
  Disk → Pinned Memory → GPU Transfer → Decompress (GPU) → Inference
  [3s]   [staged]        [150ms]        [50ms]             [ready]
```

## Usage

### GPU DCT/IDCT

```rust
use haagenti_cuda::dct_gpu::GpuDctContext;

// Create GPU context
let mut ctx = GpuDctContext::new(0)?;

// Forward DCT (compress)
let dct_coeffs = ctx.dct_2d(&input_data, width, height)?;

// Inverse DCT (decompress)
let reconstructed = ctx.idct_2d(&dct_coeffs, width, height)?;
```

### HCT Decompression

```rust
use haagenti_cuda::decompress::{GpuDecompressor, decompress_cpu};

// GPU decompression
let mut decompressor = GpuDecompressor::new(0)?;
let tensor = decompressor.decompress(&compressed_data, &[576, 576])?;

// CPU fallback (no GPU required)
let tensor = decompress_cpu(&compressed_data, &[576, 576])?;
```

### Batch Decompression with Statistics

```rust
use haagenti_cuda::decompress::GpuDecompressor;

let mut decompressor = GpuDecompressor::new(0)?;

// Batch decompress multiple tensors with performance stats
let tensors: Vec<(&[u8], &[usize])> = vec![
    (&compressed1, &[576, 576]),
    (&compressed2, &[1024, 512]),
];

let (results, stats) = decompressor.decompress_batch_pipelined(&tensors)?;
println!("{}", stats.summary());
// "2 tensors, 1.2 MB -> 4.5 MB (3.8x), 12.3ms (365.8 MB/s)"
```

### Direct GPU Memory

```rust
use haagenti_cuda::decompress::GpuDecompressor;

let mut decompressor = GpuDecompressor::new(0)?;

// Decompress directly to GPU memory (no host copy)
let gpu_tensor = decompressor.decompress_to_gpu(&compressed, &shape)?;

// Use directly in inference without data transfer
```

## Performance

| Operation | Size | Throughput |
|-----------|------|------------|
| GPU DCT 2D | 576x576 | 400-2100 MB/s |
| GPU IDCT 2D | 576x576 | 400-2100 MB/s |
| CPU DCT fallback | 576x576 | 50-100 MB/s |

## Kernel Architecture

DCT kernels automatically select optimal implementation:

- **Shared Memory Kernel**: For tensors ≤ 32 in any dimension (fast)
- **Direct Kernel**: For medium tensors (no shared memory limit)
- **FFT-based Kernel**: For large tensors > 4096 (O(n log n) via cuFFT)

Both use NVRTC runtime compilation with caching for fast startup.

### FFT-based DCT for Large Tensors

For tensors with dimensions > 4096, enable the `cufft` feature for O(n log n) DCT:

```rust
use haagenti_cuda::dct_gpu::GpuDctContext;

let mut ctx = GpuDctContext::new(0)?;

// Automatic selection: uses FFT for dimensions > 4096
let coeffs = ctx.dct_2d(&large_data, 8192, 8192)?;

// Manual control
ctx.set_fft_threshold(2048); // Use FFT for dimensions > 2048

// Force direct method (bypass FFT)
let direct_coeffs = ctx.dct_2d_direct(&data, width, height)?;
```

| Tensor Size | Direct DCT | FFT DCT | Speedup |
|-------------|------------|---------|---------|
| 1024x1024   | 2.1ms      | 0.8ms   | 2.6x    |
| 4096x4096   | 134ms      | 3.2ms   | 42x     |
| 8192x8192   | 536ms      | 6.8ms   | 79x     |

## Features Flags

- `default`: LZ4, Zstd, CPU fallback
- `cuda`: Enable GPU tests (requires NVIDIA GPU)
- `cufft`: FFT-based DCT via cuFFT (requires libcufft)
- `cpu-fallback`: CPU DCT using haagenti-core
- `async`: Tokio async support
- `python`: PyO3 bindings

## Conformance Testing

GPU implementations must pass the HCT conformance test suite to be considered compliant with the HCT specification.

### Running Conformance Tests

```bash
# Standard run (requires NVIDIA GPU)
cargo run --release --example conformance_test -p haagenti-cuda

# With cuFFT support for large tensors
cargo run --release --example conformance_test -p haagenti-cuda --features cufft

# WSL2 with GPU passthrough
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --release --example conformance_test -p haagenti-cuda
```

### Expected Output

```
╔═══════════════════════════════════════════════════════════════╗
║           HCT GPU CONFORMANCE TEST SUITE                      ║
║   Reference: HCT-SPECIFICATION-DRAFT.md Section 7             ║
╚═══════════════════════════════════════════════════════════════╝

Running 6 conformance tests...

───────────────────────────────────────────────────────────────────────────
Test Vector                      Shape   GPU Cosine     Expected   Status
───────────────────────────────────────────────────────────────────────────
sequential_4x4                     4x4     1.000000     1.000000     PASS
identity_4x4                       4x4     1.000000     1.000000     PASS
gaussian_8x8                       8x8     0.999998     0.999998     PASS
low_rank_8x8                       8x8     0.934903     0.934903     PASS
zeros_4x4                          4x4     1.000000     1.000000     PASS
constant_4x4                       4x4     1.000000     1.000000     PASS
───────────────────────────────────────────────────────────────────────────

Result: ALL CONFORMANCE TESTS PASSED
```

### Test Vectors

The conformance tests use reference test vectors from the HCT specification:

| Vector | Shape | Retention | Purpose |
|--------|-------|-----------|---------|
| sequential_4x4 | 4x4 | 50% | Basic DCT validation |
| identity_4x4 | 4x4 | 50% | Sparse input handling |
| gaussian_8x8 | 8x8 | 70% | Realistic weight distribution |
| low_rank_8x8 | 8x8 | 30% | Low-retention quality |
| zeros_4x4 | 4x4 | 50% | Edge case: all zeros |
| constant_4x4 | 4x4 | 25% | DC-only compression |

See `docs/HCT-SPECIFICATION-DRAFT.md` Section 7 for complete test vector definitions.

### Exit Codes

- `0`: All conformance tests passed
- `1`: One or more tests failed
- `2`: GPU initialization failed

## Requirements

- CUDA 12.0+ with NVRTC
- NVIDIA GPU with compute capability 7.0+
- cudarc 0.12+

## HCT V3 Format

The HCT (Holographic Compressed Tensor) format stores spectral coefficients:

```
Header:
  2 bytes: num_fragments (u16 LE)

For each fragment:
  2 bytes: index (u16 LE)
  2 bytes: flags (u16 LE)
  8 bytes: checksum (u64 LE)
  4 bytes: data_len (u32 LE)
  data_len bytes: fragment data

Fragment data (V3 with bitmap):
  4 bytes: num_coefficients (u32 LE)
  bitmap: (num_elements + 7) / 8 bytes
  coefficients: num_coefficients * 2 bytes (f16 LE)
```

## License

MIT

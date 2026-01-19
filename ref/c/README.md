# HCT Reference Decoder - C Implementation

A minimal, standalone C implementation of the HCT (Holographic Compressed Tensor) decoder, suitable for embedding in inference runtimes that don't use Rust.

## Features

- **Single-header library**: Just include `hct_decoder.h`
- **No external dependencies**: Uses only C standard library (`stdlib.h`, `math.h`)
- **Spec-compliant**: Matches the Rust reference implementation exactly
- **Embeddable**: Designed for integration into inference engines

## Quick Start

```c
#define HCT_IMPLEMENTATION
#include "hct_decoder.h"

// Decode compressed tensor
float* tensor = hct_decode(compressed_data, data_len, rows, cols);

// Use tensor for inference...

free(tensor);
```

## Building the Test Program

```bash
# Using make
make

# Or directly
gcc -O2 -o test_decoder test_decoder.c -lm

# Run tests
./test_decoder
```

## API Reference

### Decode compressed tensor

```c
float* hct_decode(const uint8_t* data, size_t data_len, size_t rows, size_t cols);
```

Decodes an HCT V3 compressed tensor.

- `data`: Pointer to compressed data
- `data_len`: Length of compressed data in bytes
- `rows`: Number of rows in output tensor
- `cols`: Number of columns in output tensor
- Returns: Newly allocated float array, or NULL on error

### DCT/IDCT

```c
void hct_dct_2d(const float* input, float* output, size_t rows, size_t cols);
void hct_idct_2d(const float* input, float* output, size_t rows, size_t cols);
```

Direct 2D DCT-II and IDCT implementations. These are reference implementations optimized for correctness, not speed.

### Quality Metrics

```c
float hct_cosine_similarity(const float* a, const float* b, size_t len);
```

Compute cosine similarity between two arrays. Returns value in [-1, 1].

## HCT V3 Format

The decoder expects HCT V3 format:

```
Header:
  2 bytes: num_fragments (u16 LE)

For each fragment:
  2 bytes: index (u16 LE)
  2 bytes: flags (u16 LE)
  8 bytes: checksum (u64 LE)
  4 bytes: data_len (u32 LE)
  data_len bytes: fragment data

Fragment data:
  4 bytes: num_coefficients (u32 LE)
  bitmap: (num_elements + 7) / 8 bytes
  coefficients: num_coefficients * 2 bytes (f16 LE)
```

## Performance Notes

This reference implementation prioritizes correctness over speed:

- O(n²) DCT/IDCT per dimension (not FFT-based)
- Single-threaded
- No SIMD optimization

For production use, consider:
- GPU implementation (see `haagenti-cuda`)
- FFT-based DCT for large tensors
- SIMD-optimized kernels

Expected throughput: ~10 MB/s on modern CPU (good enough for model loading).

## License

MIT (same as Haagenti)

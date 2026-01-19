# haagenti-python

Python bindings for the Haagenti tensor compression library.

## Features

- **HCT Format**: Block-compressed tensor storage with LZ4/Zstd compression
- **50-70% compression** typical for neural network weights (fp16/bf16)
- **2-5x faster loading** compared to safetensors (planned with GPU decompression)
- **Progressive loading** via HoloTensor (coming in Phase 4)

## Installation

```bash
# Build from source (requires Rust 1.85+)
cd crates/haagenti-python
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/haagenti-*.whl
```

## Quick Start

```python
import haagenti
import numpy as np

# Compress a tensor
data = np.random.randn(1024, 1024).astype(np.float32)
stats = haagenti.compress_tensor("weights.hct", data, algorithm="lz4")
print(f"Compression ratio: {stats['ratio']:.2f}x")

# Load a tensor
loaded = haagenti.load_tensor("weights.hct")
assert np.allclose(data, loaded)

# Convert safetensors to HCT
haagenti.convert_safetensors("model.safetensors", "model.hct")
```

## Low-level API

```python
from haagenti import HctReader, HctWriter, CompressionAlgorithm, DType

# Read HCT files
reader = HctReader("weights.hct")
header = reader.header()
print(f"Shape: {header.shape}, DType: {header.dtype}")
print(f"Compression: {header.compression_ratio():.2f}x")

# Decompress all data
data = reader.decompress_all()

# Write HCT files
writer = HctWriter("output.hct", CompressionAlgorithm.Zstd, DType.F32, [1024, 1024])
writer.compress_data(data)
writer.finish()
```

## Compression Algorithms

| Algorithm | Speed | Ratio | Best For |
|-----------|-------|-------|----------|
| LZ4 | Fast | 1.5-2x | Real-time loading |
| Zstd | Medium | 2-3x | Storage efficiency |

## Data Types

- `F32` - 32-bit float (default)
- `F16` - 16-bit float
- `BF16` - BFloat16
- `I8` - 8-bit integer (quantized)
- `I4` - 4-bit integer (quantized)

## Roadmap

- [x] Phase 1: INT8/INT4 quantization support
- [x] Phase 2: Smart component offloading
- [x] Phase 3: Python bindings (this crate)
- [ ] Phase 4: HoloTensor progressive loading
- [ ] Phase 5: GPU decompression kernels

## License

MIT

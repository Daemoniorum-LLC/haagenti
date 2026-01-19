"""Haagenti - High-performance tensor compression for neural networks.

This package provides Python bindings for the Haagenti tensor compression
library, enabling efficient storage and loading of large neural network weights.

Features:
- HCT format: Block-compressed tensor storage with LZ4/Zstd
- HoloTensor: Progressive holographic encoding for instant loading
- 50-70% compression ratios typical for fp16/bf16 weights
- 2-5x faster loading vs safetensors (with GPU decompression)

Quick Start:
    >>> import haagenti
    >>> import numpy as np

    # Compress a tensor
    >>> data = np.random.randn(1024, 1024).astype(np.float32)
    >>> haagenti.compress_tensor("model.hct", data, algorithm="lz4")

    # Load a tensor
    >>> loaded = haagenti.load_tensor("model.hct")

    # Convert safetensors to HCT
    >>> haagenti.convert_safetensors("model.safetensors", "model.hct")
"""

from haagenti._haagenti_python import (
    # Enums
    CompressionAlgorithm,
    DType,
    QuantizationScheme,
    HolographicEncoding,
    # HCT classes
    HctHeader,
    HctReader,
    HctWriter,
    # HoloTensor classes
    HoloTensorEncoder,
    HoloTensorDecoder,
    HoloTensorHeaderPy as HoloTensorHeader,
    HoloFragmentPy as HoloFragment,
    # Functions
    convert_safetensors_to_hct,
    version,
)

import numpy as np
from pathlib import Path
from typing import Union, Optional

__version__ = version()
__all__ = [
    # Enums
    "CompressionAlgorithm",
    "DType",
    "QuantizationScheme",
    "HolographicEncoding",
    # HCT Classes
    "HctHeader",
    "HctReader",
    "HctWriter",
    # HoloTensor Classes
    "HoloTensorEncoder",
    "HoloTensorDecoder",
    "HoloTensorHeader",
    "HoloFragment",
    # High-level functions
    "compress_tensor",
    "load_tensor",
    "convert_safetensors",
    "encode_progressive",
    "ProgressiveLoader",
    "version",
    # GPU module
    "gpu",
    "is_gpu_available",
]

# GPU submodule (lazy import to avoid CUDA dependency if not needed)
def is_gpu_available() -> bool:
    """Check if GPU-accelerated decompression is available."""
    try:
        from haagenti.gpu import is_cuda_available
        return is_cuda_available()
    except ImportError:
        return False


def compress_tensor(
    path: Union[str, Path],
    data: np.ndarray,
    algorithm: str = "lz4",
    block_size: Optional[int] = None,
) -> dict:
    """Compress a numpy array to HCT format.

    Args:
        path: Output file path
        data: Numpy array to compress (will be converted to float32)
        algorithm: Compression algorithm ("lz4" or "zstd")
        block_size: Optional block size for compression (default: 16KB)

    Returns:
        dict with compression stats (original_size, compressed_size, ratio)
    """
    path = str(path)

    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Flatten for compression
    flat = data.ravel()
    shape = list(data.shape)

    # Select algorithm
    algo = CompressionAlgorithm.Lz4 if algorithm.lower() == "lz4" else CompressionAlgorithm.Zstd

    # Create writer and compress
    writer = HctWriter(path, algo, DType.F32, [int(s) for s in shape], block_size)
    writer.compress_data(flat)
    writer.finish()

    # Get file size for stats
    import os
    compressed_size = os.path.getsize(path)
    original_size = flat.nbytes

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": original_size / compressed_size if compressed_size > 0 else 0,
        "shape": shape,
    }


def load_tensor(path: Union[str, Path]) -> np.ndarray:
    """Load a tensor from HCT format.

    Args:
        path: Path to HCT file

    Returns:
        Numpy array with the decompressed tensor
    """
    path = str(path)
    reader = HctReader(path)
    header = reader.header()

    # Decompress to flat array
    data = reader.decompress_all()

    # Reshape to original dimensions
    shape = tuple(header.shape)
    return data.reshape(shape)


def convert_safetensors(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    algorithm: str = "lz4",
) -> dict:
    """Convert a safetensors file to HCT format.

    Args:
        input_path: Path to input safetensors file
        output_path: Path for output HCT file
        algorithm: Compression algorithm ("lz4" or "zstd")

    Returns:
        dict with conversion stats
    """
    algo = CompressionAlgorithm.Lz4 if algorithm.lower() == "lz4" else CompressionAlgorithm.Zstd
    original, compressed, ratio = convert_safetensors_to_hct(
        str(input_path),
        str(output_path),
        algo,
    )
    return {
        "original_size": original,
        "compressed_size": compressed,
        "ratio": ratio,
    }


def encode_progressive(
    data: np.ndarray,
    encoding: str = "spectral",
    n_fragments: int = 8,
    seed: Optional[int] = None,
) -> tuple:
    """Encode a tensor for progressive loading.

    Args:
        data: Numpy array to encode (will be flattened)
        encoding: Encoding scheme ("spectral", "random_projection", or "low_rank")
        n_fragments: Number of fragments to create (default 8)
        seed: Random seed for deterministic encoding

    Returns:
        Tuple of (header, list of fragments)

    Example:
        >>> data = np.random.randn(4096, 4096).astype(np.float32)
        >>> header, fragments = haagenti.encode_progressive(data, n_fragments=8)
        >>> # First fragment gives ~30% quality
        >>> loader = haagenti.ProgressiveLoader(header)
        >>> loader.add_fragment(fragments[0])
        >>> approx = loader.reconstruct()
    """
    # Map encoding name to enum
    encoding_map = {
        "spectral": HolographicEncoding.Spectral,
        "random_projection": HolographicEncoding.RandomProjection,
        "low_rank": HolographicEncoding.LowRankDistributed,
    }
    enc = encoding_map.get(encoding.lower())
    if enc is None:
        raise ValueError(f"Unknown encoding: {encoding}. Use one of: {list(encoding_map.keys())}")

    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Create encoder
    encoder = HoloTensorEncoder(enc, n_fragments, seed)

    # Encode based on dimensions
    if data.ndim == 1:
        return encoder.encode_1d(data)
    elif data.ndim == 2:
        rows, cols = data.shape
        return encoder.encode_2d(data.ravel(), rows, cols)
    else:
        # For higher dimensions, flatten to 2D (batch x features)
        flat = data.reshape(-1, data.shape[-1])
        rows, cols = flat.shape
        return encoder.encode_2d(flat.ravel(), rows, cols)


class ProgressiveLoader:
    """Progressive tensor loader with quality-aware reconstruction.

    Enables loading model weights incrementally, starting with a rough
    approximation and progressively improving quality. Ideal for:
    - Fast model startup (usable weights in <1s)
    - Memory-constrained environments
    - Quality-aware inference (use low quality for noisy steps)

    Example:
        >>> loader = haagenti.ProgressiveLoader(header)
        >>> loader.add_fragment(fragments[0])  # ~30% quality
        >>> print(f"Quality: {loader.quality:.1%}")
        >>> weights = loader.reconstruct()
        >>> loader.add_fragment(fragments[1])  # ~50% quality
        >>> weights = loader.reconstruct()  # Better approximation
    """

    def __init__(self, header: HoloTensorHeader):
        """Create a progressive loader from a header.

        Args:
            header: HoloTensor header from encoding
        """
        self._decoder = HoloTensorDecoder(header)
        self._header = header
        self._shape = tuple(header.shape)

    @property
    def quality(self) -> float:
        """Current reconstruction quality (0.0-1.0)."""
        return self._decoder.quality

    @property
    def fragments_loaded(self) -> int:
        """Number of fragments loaded."""
        return self._decoder.fragments_loaded

    @property
    def total_fragments(self) -> int:
        """Total number of fragments available."""
        return self._decoder.total_fragments

    @property
    def shape(self) -> tuple:
        """Original tensor shape."""
        return self._shape

    def add_fragment(self, fragment: HoloFragment) -> float:
        """Add a fragment and return new quality level.

        Args:
            fragment: Fragment to add

        Returns:
            New quality level (0.0-1.0)
        """
        return self._decoder.add_fragment(fragment)

    def add_fragments(self, fragments: list) -> float:
        """Add multiple fragments.

        Args:
            fragments: List of fragments to add

        Returns:
            Final quality level
        """
        quality = 0.0
        for f in fragments:
            quality = self._decoder.add_fragment(f)
        return quality

    def can_reconstruct(self) -> bool:
        """Check if minimum fragments for reconstruction are loaded."""
        return self._decoder.can_reconstruct()

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the tensor from loaded fragments.

        Returns:
            Numpy array with reconstructed weights, reshaped to original dimensions
        """
        flat = self._decoder.reconstruct()
        return flat.reshape(self._shape)

    def load_to_quality(self, fragments: list, target_quality: float) -> np.ndarray:
        """Load fragments until reaching target quality.

        Args:
            fragments: List of all fragments
            target_quality: Target quality level (0.0-1.0)

        Returns:
            Reconstructed tensor at target quality
        """
        for f in fragments:
            self.add_fragment(f)
            if self.quality >= target_quality:
                break
        return self.reconstruct()

    def __repr__(self) -> str:
        return (
            f"ProgressiveLoader(quality={self.quality:.1%}, "
            f"fragments={self.fragments_loaded}/{self.total_fragments}, "
            f"shape={self.shape})"
        )

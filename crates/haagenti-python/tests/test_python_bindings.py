"""
Track C.5: Python Bindings Completion Tests

These tests validate the Python bindings for Haagenti tensor compression library.
Run with: pytest tests/test_python_bindings.py -v

Prerequisites:
- Build and install the haagenti wheel: maturin develop --release
- Install pytest: pip install pytest numpy
"""

import pytest

# Import will fail until wheel is built
try:
    import haagenti
    HAAGENTI_AVAILABLE = True
except ImportError:
    HAAGENTI_AVAILABLE = False
    haagenti = None


pytestmark = pytest.mark.skipif(
    not HAAGENTI_AVAILABLE,
    reason="haagenti module not available - build with: maturin develop --release"
)


class TestCompressionBindings:
    """Tests for compression/decompression functions."""

    def test_zstd_compress_decompress(self):
        """Test Zstd roundtrip compression."""
        data = b"Hello, Haagenti Python!" * 1000
        compressed = haagenti.compress(data, algorithm="zstd")
        decompressed = haagenti.decompress(compressed, algorithm="zstd")
        assert decompressed == data

    def test_lz4_compress_decompress(self):
        """Test LZ4 roundtrip compression."""
        data = b"LZ4 test data" * 1000
        compressed = haagenti.compress(data, algorithm="lz4")
        decompressed = haagenti.decompress(compressed, algorithm="lz4")
        assert decompressed == data

    def test_compression_levels(self):
        """Test that 'best' level produces smaller output than 'fast'."""
        data = b"Test data for compression levels" * 1000
        fast = haagenti.compress(data, algorithm="zstd", level="fast")
        best = haagenti.compress(data, algorithm="zstd", level="best")
        assert len(best) <= len(fast)

    def test_compression_ratio(self):
        """Test compression achieves reasonable ratio on repetitive data."""
        data = b"Repetitive data for compression ratio test. " * 1000
        compressed = haagenti.compress(data, algorithm="zstd")
        ratio = len(data) / len(compressed)
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}x below expected"


class TestHoloTensorBindings:
    """Tests for HoloTensor progressive loading."""

    def test_holotensor_encoder_creation(self):
        """Test HoloTensor encoder can be created."""
        encoder = haagenti.HoloTensorEncoder(
            haagenti.HolographicEncoding.Spectral,
            n_fragments=8
        )
        assert encoder.n_fragments == 8
        assert encoder.encoding == haagenti.HolographicEncoding.Spectral

    def test_holotensor_encode_decode(self):
        """Test HoloTensor encode/decode roundtrip."""
        import numpy as np
        tensor = np.random.randn(512, 512).astype(np.float32).flatten()

        encoder = haagenti.HoloTensorEncoder(
            haagenti.HolographicEncoding.Spectral,
            n_fragments=16
        )
        header, fragments = encoder.encode_2d(tensor, 512, 512)

        decoder = haagenti.HoloTensorDecoder(header)
        for frag in fragments:
            decoder.add_fragment(frag)

        decoded = decoder.reconstruct()

        # Allow some quality loss
        correlation = np.corrcoef(tensor, decoded)[0, 1]
        assert correlation > 0.95, f"Correlation {correlation:.3f} below threshold"

    def test_holotensor_progressive_decode(self):
        """Test progressive decoding with partial fragments."""
        import numpy as np
        tensor = np.random.randn(256, 256).astype(np.float32).flatten()

        encoder = haagenti.HoloTensorEncoder(
            haagenti.HolographicEncoding.Spectral,
            n_fragments=16
        )
        header, fragments = encoder.encode_2d(tensor, 256, 256)

        # Partial decode with first 8 fragments
        decoder = haagenti.HoloTensorDecoder(header)
        for frag in fragments[:8]:
            decoder.add_fragment(frag)

        partial = decoder.reconstruct()
        assert partial.shape == tensor.shape
        assert decoder.quality() < 1.0  # Not full quality with partial fragments


class TestStreamingBindings:
    """Tests for streaming compression/decompression."""

    def test_streaming_encoder(self):
        """Test streaming encoder context manager."""
        data = b"Streaming test data" * 10000

        with haagenti.streaming.Encoder("zstd") as encoder:
            for chunk in [data[i:i+1000] for i in range(0, len(data), 1000)]:
                encoder.write(chunk)
            compressed = encoder.finish()

        assert len(compressed) < len(data)

    def test_streaming_decoder(self):
        """Test streaming decoder context manager."""
        data = b"Streaming decoder test data" * 10000
        compressed = haagenti.compress(data, algorithm="zstd")

        chunks = []
        with haagenti.streaming.Decoder("zstd") as decoder:
            for chunk in [compressed[i:i+100] for i in range(0, len(compressed), 100)]:
                decoded = decoder.write(chunk)
                if decoded:
                    chunks.append(decoded)
            chunks.append(decoder.finish())

        assert b"".join(chunks) == data


class TestDictionaryBindings:
    """Tests for Zstd dictionary compression."""

    def test_dictionary_training(self):
        """Test dictionary training from samples."""
        samples = [f"model.layers.{i}.weight".encode() for i in range(100)]
        dict_obj = haagenti.ZstdDict.train(samples, max_size=8192)

        assert dict_obj.id != 0
        assert len(dict_obj.as_bytes()) <= 8192

    def test_dictionary_too_few_samples(self):
        """Test that training fails with too few samples."""
        samples = [b"single sample", b"another sample"]
        with pytest.raises(ValueError):
            haagenti.ZstdDict.train(samples, max_size=8192)


class TestHctBindings:
    """Tests for HCT file format."""

    def test_hct_header_properties(self):
        """Test HctHeader property accessors."""
        # Create an HctWriter to get a header
        import tempfile
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".hct", delete=False) as f:
            path = f.name

        try:
            writer = haagenti.HctWriter(
                path,
                haagenti.CompressionAlgorithm.Lz4,
                haagenti.DType.F32,
                [1024]
            )
            data = np.random.randn(1024).astype(np.float32)
            writer.compress_data(data)
            writer.finish()

            reader = haagenti.HctReader(path)
            header = reader.header()

            assert header.algorithm == haagenti.CompressionAlgorithm.Lz4
            assert header.dtype == haagenti.DType.F32
            assert header.shape == [1024]
            assert header.numel() == 1024
        finally:
            import os
            os.unlink(path)


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError):
            haagenti.compress(b"test", algorithm="invalid")

    def test_corrupted_data(self):
        """Test that corrupted data raises DecompressionError."""
        with pytest.raises(haagenti.DecompressionError):
            haagenti.decompress(b"not valid compressed data", algorithm="zstd")

    def test_empty_input(self):
        """Test that empty input compresses/decompresses correctly."""
        compressed = haagenti.compress(b"", algorithm="zstd")
        decompressed = haagenti.decompress(compressed, algorithm="zstd")
        assert decompressed == b""

    def test_invalid_compression_level(self):
        """Test that invalid compression level raises ValueError."""
        with pytest.raises(ValueError):
            haagenti.compress(b"test", algorithm="zstd", level="ultra")


class TestEnumerations:
    """Tests for enumeration types."""

    def test_compression_algorithm_values(self):
        """Test CompressionAlgorithm enum values."""
        assert haagenti.CompressionAlgorithm.Lz4 is not None
        assert haagenti.CompressionAlgorithm.Zstd is not None

    def test_dtype_bits(self):
        """Test DType bit sizes."""
        assert haagenti.DType.F32.bits() == 32
        assert haagenti.DType.F16.bits() == 16
        assert haagenti.DType.I8.bits() == 8
        assert haagenti.DType.I4.bits() == 4

    def test_holographic_encoding_values(self):
        """Test HolographicEncoding enum values."""
        assert haagenti.HolographicEncoding.Spectral is not None
        assert haagenti.HolographicEncoding.RandomProjection is not None
        assert haagenti.HolographicEncoding.LowRankDistributed is not None


class TestVersionInfo:
    """Tests for version information."""

    def test_version_string(self):
        """Test that version returns a string."""
        version = haagenti.version()
        assert isinstance(version, str)
        assert len(version) > 0


# Test count summary for C.5:
# - TestCompressionBindings: 4 tests
# - TestHoloTensorBindings: 3 tests
# - TestStreamingBindings: 2 tests
# - TestDictionaryBindings: 2 tests
# - TestHctBindings: 1 test
# - TestErrorHandling: 4 tests
# - TestEnumerations: 3 tests
# - TestVersionInfo: 1 test
# Total: 20 tests (exceeds 15 required)

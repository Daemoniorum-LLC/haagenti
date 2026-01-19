# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-19

### Added

- Initial open source release
- LZ4 and LZ4-HC compression
- Zstandard compression with dictionary support
- Brotli compression
- Deflate/Gzip/Zlib compression
- SIMD acceleration (AVX2, AVX-512, NEON)
- Streaming compression API with backpressure
- no_std compatible core traits
- HoloTensor holographic encoding for progressive reconstruction
- Fragment similarity detection via LSH
- Pure Rust Zstd implementation (RFC 8878)

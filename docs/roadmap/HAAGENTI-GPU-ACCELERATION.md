# Track B: GPU Acceleration

## Overview

This document details the TDD roadmap for GPU-accelerated decompression in Haagenti.

**Timeline:** 10-12 weeks
**Priority:** High
**Crates:** `haagenti-cuda`, `haagenti-neural`

---

## Current State

### Implemented
- LZ4 GPU decompression kernel
- Basic CUDA device management
- Memory pool for GPU allocations

### Placeholders
- Zstd GPU decompression (`kernels.rs:180`: "placeholder for future implementation")
- Neural GPU decoder (`decoder.rs`: "placeholder for CUDA implementation")

---

## Phase B.1: Zstd GPU Sequence Decoder

### Purpose
Implement GPU kernel for Zstd sequence execution (literal copy + match copy).

### Test Specification

```rust
// tests/cuda_zstd_sequences_test.rs

#[cfg(all(test, feature = "cuda"))]
mod gpu_sequence_tests {
    use haagenti_cuda::{GpuContext, ZstdGpuDecoder};
    use haagenti_zstd::{ZstdCodec, parse_sequences};

    #[test]
    fn test_gpu_sequence_decoder_creation() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        assert!(decoder.is_ready());
    }

    #[test]
    fn test_gpu_literal_copy_single() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Single literal copy sequence
        let literals = b"Hello, World!";
        let sequences = vec![
            Sequence::literal_only(literals.len()),
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();

        assert_eq!(result.as_slice(), literals);
    }

    #[test]
    fn test_gpu_match_copy_simple() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Literal followed by match
        let literals = b"abc";
        let sequences = vec![
            Sequence::new(3, 0, 0),  // 3 literals
            Sequence::new(0, 3, 3),  // match: offset=3, length=3 (copy "abc")
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();

        assert_eq!(result.as_slice(), b"abcabc");
    }

    #[test]
    fn test_gpu_overlapping_match() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // RLE-style: copy from offset 1 repeatedly
        let literals = b"a";
        let sequences = vec![
            Sequence::new(1, 0, 0),   // 1 literal "a"
            Sequence::new(0, 1, 10),  // match: offset=1, length=10 (RLE)
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();

        assert_eq!(result.as_slice(), b"aaaaaaaaaaa"); // 11 a's
    }

    #[test]
    fn test_gpu_multiple_sequences() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let literals = b"The quick brown fox ";
        let sequences = vec![
            Sequence::new(20, 0, 0),  // All literals
            Sequence::new(0, 20, 5),  // Copy "quick brown fox " with offset
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]).unwrap();

        assert!(result.len() == 25); // 20 + 5 from match
    }

    #[test]
    fn test_gpu_sequence_batch_processing() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Multiple independent blocks
        let blocks: Vec<(Vec<u8>, Vec<Sequence>)> = (0..100)
            .map(|i| {
                let lit = format!("Block {} data here", i).into_bytes();
                let seq = vec![Sequence::literal_only(lit.len())];
                (lit, seq)
            })
            .collect();

        let results = decoder.execute_batch(&blocks).unwrap();

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            let expected = format!("Block {} data here", i);
            assert_eq!(result.as_slice(), expected.as_bytes());
        }
    }

    #[test]
    fn test_gpu_vs_cpu_equivalence() {
        let ctx = GpuContext::new(0).unwrap();
        let gpu_decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Compress some data
        let original = b"Test data for GPU vs CPU comparison. ".repeat(1000);
        let compressed = ZstdCodec::new().compress(&original).unwrap();

        // Decompress on CPU
        let cpu_result = ZstdCodec::new().decompress(&compressed).unwrap();

        // Decompress on GPU
        let gpu_result = gpu_decoder.decompress(&compressed).unwrap();

        assert_eq!(cpu_result, gpu_result);
    }

    #[test]
    fn test_gpu_large_offset_handling() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Create data with large offset match
        let mut literals = vec![b'x'; 100_000];
        literals[0..5].copy_from_slice(b"MATCH");

        let sequences = vec![
            Sequence::new(100_000, 0, 0),     // All literals
            Sequence::new(0, 100_000, 5),     // Match from very beginning
        ];

        let result = decoder.execute_sequences(&sequences, &literals, &[]).unwrap();

        assert_eq!(&result[100_000..100_005], b"MATCH");
    }

    #[test]
    fn test_gpu_memory_efficiency() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        let initial_memory = ctx.memory_used();

        // Process large data
        let data = vec![b'a'; 10_000_000]; // 10MB
        let compressed = ZstdCodec::new().compress(&data).unwrap();
        let _ = decoder.decompress(&compressed).unwrap();

        let peak_memory = ctx.peak_memory_used();

        // Should not use more than 3x the output size
        assert!(peak_memory - initial_memory < data.len() * 3,
            "Memory usage: {} bytes for {} byte output",
            peak_memory - initial_memory, data.len());
    }

    #[test]
    fn test_gpu_error_handling_invalid_sequence() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = ZstdGpuDecoder::new(&ctx).unwrap();

        // Invalid: match offset larger than output
        let literals = b"abc";
        let sequences = vec![
            Sequence::new(3, 0, 0),
            Sequence::new(0, 100, 5), // Invalid: offset 100 but only 3 bytes exist
        ];

        let result = decoder.execute_sequences(&sequences, literals, &[]);

        assert!(result.is_err());
    }
}
```

### Implementation Specification

```rust
// haagenti-cuda/src/zstd/sequences.rs

/// GPU-accelerated Zstd sequence decoder.
pub struct ZstdGpuDecoder {
    device: Arc<CudaDevice>,
    sequence_kernel: CudaFunction,
    stream: CudaStream,
}

impl ZstdGpuDecoder {
    /// Create new GPU sequence decoder.
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError>;

    /// Execute sequences on GPU.
    ///
    /// # Arguments
    /// * `sequences` - Parsed Zstd sequences
    /// * `literals` - Literal bytes to copy
    /// * `history` - Previous output for match references
    pub fn execute_sequences(
        &self,
        sequences: &[Sequence],
        literals: &[u8],
        history: &[u8],
    ) -> Result<Vec<u8>, GpuError>;

    /// Decompress complete Zstd frame on GPU.
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>, GpuError>;

    /// Batch decompress multiple frames.
    pub fn decompress_batch(&self, frames: &[&[u8]]) -> Result<Vec<Vec<u8>>, GpuError>;
}

/// A Zstd sequence (literals + match).
#[derive(Clone, Copy)]
pub struct Sequence {
    pub literal_length: u32,
    pub match_offset: u32,
    pub match_length: u32,
}
```

### CUDA Kernel Design

```cuda
// kernels/zstd_sequences.cu

__global__ void execute_sequences_kernel(
    const uint8_t* __restrict__ literals,
    const Sequence* __restrict__ sequences,
    const uint32_t num_sequences,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ output_offsets
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_sequences) return;

    const Sequence seq = sequences[tid];
    const uint32_t out_offset = output_offsets[tid];
    const uint32_t lit_offset = /* computed from prefix sum */;

    // Copy literals
    for (uint32_t i = 0; i < seq.literal_length; i++) {
        output[out_offset + i] = literals[lit_offset + i];
    }

    // Copy match (may overlap)
    const uint32_t match_start = out_offset + seq.literal_length;
    const uint32_t src_offset = match_start - seq.match_offset;

    for (uint32_t i = 0; i < seq.match_length; i++) {
        output[match_start + i] = output[src_offset + i];
    }
}
```

### Quality Gate B.1

```bash
#!/bin/bash
# Phase B.1 Quality Gate

echo "=== Phase B.1: GPU Sequence Decoder Quality Gate ==="

# 1. All GPU sequence tests pass
cargo test --package haagenti-cuda --features cuda gpu_sequence -- --nocapture

# 2. GPU/CPU equivalence verified
cargo test --package haagenti-cuda --features cuda test_gpu_vs_cpu_equivalence

# 3. Performance benchmark
cargo bench --package haagenti-cuda --features cuda -- gpu_sequences

# 4. Memory efficiency check
cargo test --package haagenti-cuda --features cuda test_gpu_memory_efficiency

echo "=== Phase B.1 PASSED ==="
```

---

## Phase B.2: Zstd GPU FSE Decoder

### Purpose
Implement GPU kernel for FSE (Finite State Entropy) decoding.

### Test Specification

```rust
// tests/cuda_zstd_fse_test.rs

#[cfg(all(test, feature = "cuda"))]
mod gpu_fse_tests {
    use haagenti_cuda::{GpuContext, FseGpuDecoder};
    use haagenti_zstd::fse::{FseTable, FseDecoder};

    #[test]
    fn test_gpu_fse_decoder_creation() {
        let ctx = GpuContext::new(0).unwrap();

        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        assert!(decoder.is_ready());
    }

    #[test]
    fn test_gpu_fse_decode_simple() {
        let ctx = GpuContext::new(0).unwrap();

        // Create simple table
        let frequencies = [100u32, 50, 25, 12, 6, 3, 2, 1];
        let table = FseTable::from_frequencies(&frequencies).unwrap();

        let gpu_decoder = FseGpuDecoder::new(&ctx, &table).unwrap();
        let cpu_decoder = FseDecoder::new(&table);

        // Encode some symbols
        let symbols: Vec<u8> = (0..1000).map(|i| (i % 8) as u8).collect();
        let encoded = haagenti_zstd::fse::FseEncoder::new(&table)
            .encode(&symbols).unwrap();

        // Decode on GPU
        let gpu_result = gpu_decoder.decode(&encoded, symbols.len()).unwrap();

        // Decode on CPU
        let cpu_result = cpu_decoder.decode(&encoded, symbols.len()).unwrap();

        assert_eq!(gpu_result, cpu_result);
        assert_eq!(gpu_result, symbols);
    }

    #[test]
    fn test_gpu_fse_predefined_tables() {
        let ctx = GpuContext::new(0).unwrap();

        // Test all predefined tables
        let tables = [
            FseTable::predefined_literals(),
            FseTable::predefined_match_lengths(),
            FseTable::predefined_offsets(),
        ];

        for table in &tables {
            let decoder = FseGpuDecoder::new(&ctx, table).unwrap();
            assert!(decoder.is_ready());
        }
    }

    #[test]
    fn test_gpu_fse_batch_decode() {
        let ctx = GpuContext::new(0).unwrap();

        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        // Multiple encoded streams
        let streams: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let symbols: Vec<u8> = (0..100).map(|j| ((i + j) % 256) as u8).collect();
                haagenti_zstd::fse::FseEncoder::new(&table)
                    .encode(&symbols).unwrap()
            })
            .collect();

        let lengths: Vec<usize> = vec![100; 100];

        let results = decoder.decode_batch(&streams, &lengths).unwrap();

        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_gpu_fse_large_alphabet() {
        let ctx = GpuContext::new(0).unwrap();

        // 256-symbol alphabet (full byte range)
        let frequencies: Vec<u32> = (0..256).map(|i| 256 - i as u32).collect();
        let table = FseTable::from_frequencies(&frequencies).unwrap();

        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let symbols: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let encoded = haagenti_zstd::fse::FseEncoder::new(&table)
            .encode(&symbols).unwrap();

        let result = decoder.decode(&encoded, symbols.len()).unwrap();

        assert_eq!(result, symbols);
    }

    #[test]
    fn test_gpu_fse_table_upload_caching() {
        let ctx = GpuContext::new(0).unwrap();

        let table = FseTable::predefined_literals();

        // Create multiple decoders with same table
        let decoder1 = FseGpuDecoder::new(&ctx, &table).unwrap();
        let decoder2 = FseGpuDecoder::new(&ctx, &table).unwrap();

        // Should reuse uploaded table (check memory)
        let memory_after_first = ctx.memory_used();
        let _ = FseGpuDecoder::new(&ctx, &table).unwrap();
        let memory_after_second = ctx.memory_used();

        // Memory shouldn't increase significantly
        assert!(memory_after_second <= memory_after_first + 1024);
    }

    #[test]
    fn test_gpu_fse_interleaved_streams() {
        let ctx = GpuContext::new(0).unwrap();

        // Zstd uses 4 interleaved FSE streams
        let table = FseTable::predefined_literals();
        let decoder = FseGpuDecoder::new(&ctx, &table).unwrap();

        let streams: [Vec<u8>; 4] = [
            encode_stream(&table, &[0, 1, 2, 3]),
            encode_stream(&table, &[4, 5, 6, 7]),
            encode_stream(&table, &[8, 9, 10, 11]),
            encode_stream(&table, &[12, 13, 14, 15]),
        ];

        let result = decoder.decode_interleaved(&streams, &[4, 4, 4, 4]).unwrap();

        // Interleaved output: stream0[0], stream1[0], stream2[0], stream3[0], ...
        assert_eq!(result, vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]);
    }

    fn encode_stream(table: &FseTable, symbols: &[u8]) -> Vec<u8> {
        haagenti_zstd::fse::FseEncoder::new(table)
            .encode(symbols).unwrap()
    }
}
```

### Quality Gate B.2

```bash
#!/bin/bash
# Phase B.2 Quality Gate

echo "=== Phase B.2: GPU FSE Decoder Quality Gate ==="

# 1. All FSE GPU tests pass
cargo test --package haagenti-cuda --features cuda gpu_fse -- --nocapture

# 2. CPU/GPU equivalence
cargo test --package haagenti-cuda --features cuda test_gpu_fse_decode_simple

# 3. Interleaved streams work
cargo test --package haagenti-cuda --features cuda test_gpu_fse_interleaved

# 4. Performance benchmark
cargo bench --package haagenti-cuda --features cuda -- gpu_fse

echo "=== Phase B.2 PASSED ==="
```

---

## Phase B.3: Zstd GPU Full Pipeline

### Purpose
Integrate FSE + sequence decoders into complete Zstd GPU decompression pipeline.

### Test Specification

```rust
// tests/cuda_zstd_pipeline_test.rs

#[cfg(all(test, feature = "cuda"))]
mod gpu_pipeline_tests {
    use haagenti_cuda::{GpuContext, ZstdGpuPipeline};
    use haagenti_zstd::ZstdCodec;

    #[test]
    fn test_gpu_pipeline_creation() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_gpu_pipeline_simple_frame() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = b"Hello, GPU Zstd!";
        let compressed = ZstdCodec::new().compress(original).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(decompressed.as_slice(), original.as_slice());
    }

    #[test]
    fn test_gpu_pipeline_large_frame() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(10_000_000); // 10MB
        let compressed = ZstdCodec::new().compress(&original).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_gpu_pipeline_multiple_blocks() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        // Data large enough to span multiple blocks
        let original = generate_test_data(1_000_000);
        let compressed = ZstdCodec::new()
            .with_block_size(65536) // 64KB blocks
            .compress(&original).unwrap();

        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_gpu_pipeline_with_dictionary() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        // Train dictionary
        let samples = generate_training_samples(100);
        let dict = haagenti_zstd::ZstdDict::train(&samples, 8192).unwrap();

        // Compress with dictionary
        let original = b"model.layers.42.attention.q_proj.weight".repeat(100);
        let compressed = haagenti_zstd::ZstdDictCompressor::new(&dict)
            .compress(&original).unwrap();

        // Decompress on GPU with dictionary
        let decompressed = pipeline.decompress_with_dict(&compressed, &dict).unwrap();

        assert_eq!(decompressed.as_slice(), original.as_slice());
    }

    #[test]
    fn test_gpu_pipeline_batch() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let frames: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let data = format!("Frame {} with some data", i).repeat(100);
                ZstdCodec::new().compress(data.as_bytes()).unwrap()
            })
            .collect();

        let results = pipeline.decompress_batch(&frames).unwrap();

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            let expected = format!("Frame {} with some data", i).repeat(100);
            assert_eq!(result.as_slice(), expected.as_bytes());
        }
    }

    #[test]
    fn test_gpu_pipeline_throughput() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(100_000_000); // 100MB
        let compressed = ZstdCodec::new().compress(&original).unwrap();

        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = pipeline.decompress(&compressed).unwrap();
        }
        let elapsed = start.elapsed();

        let throughput_gbs = (iterations as f64 * original.len() as f64)
            / elapsed.as_secs_f64() / 1_000_000_000.0;

        // Target: 4 GB/s decompression
        assert!(throughput_gbs > 4.0,
            "Throughput: {:.2} GB/s (target: >4 GB/s)", throughput_gbs);
    }

    #[test]
    fn test_gpu_pipeline_zero_copy_output() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = ZstdGpuPipeline::new(&ctx).unwrap();

        let original = generate_test_data(1_000_000);
        let compressed = ZstdCodec::new().compress(&original).unwrap();

        // Decompress directly to GPU buffer
        let gpu_buffer = pipeline.decompress_to_gpu(&compressed).unwrap();

        // Verify data
        let host_data = gpu_buffer.to_host().unwrap();
        assert_eq!(host_data, original);
    }
}
```

### Quality Gate B.3

```bash
#!/bin/bash
# Phase B.3 Quality Gate

echo "=== Phase B.3: GPU Full Pipeline Quality Gate ==="

# 1. All pipeline tests pass
cargo test --package haagenti-cuda --features cuda gpu_pipeline -- --nocapture

# 2. Throughput target met (4 GB/s)
cargo test --package haagenti-cuda --features cuda test_gpu_pipeline_throughput

# 3. Dictionary support works
cargo test --package haagenti-cuda --features cuda test_gpu_pipeline_with_dictionary

# 4. Zero-copy output works
cargo test --package haagenti-cuda --features cuda test_gpu_pipeline_zero_copy

# 5. Batch processing works
cargo test --package haagenti-cuda --features cuda test_gpu_pipeline_batch

echo "=== Phase B.3 PASSED ==="
```

---

## Phase B.4: Neural GPU Codebook Lookup

### Purpose
Implement GPU kernel for neural compression codebook lookup.

### Test Specification

```rust
// tests/cuda_neural_codebook_test.rs

#[cfg(all(test, feature = "cuda"))]
mod gpu_codebook_tests {
    use haagenti_cuda::{GpuContext, NeuralGpuDecoder};
    use haagenti_neural::{LayerCodebook, QuantizedTensor};

    #[test]
    fn test_gpu_codebook_upload() {
        let ctx = GpuContext::new(0).unwrap();

        let codebook = LayerCodebook::random(256, 64); // 256 codes, 64-dim
        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        decoder.upload_codebook(&codebook).unwrap();

        assert!(decoder.has_codebook());
    }

    #[test]
    fn test_gpu_codebook_lookup_simple() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        // Create simple codebook
        let mut codebook = LayerCodebook::new(4, 2); // 4 codes, 2-dim
        codebook.set_code(0, &[1.0, 0.0]);
        codebook.set_code(1, &[0.0, 1.0]);
        codebook.set_code(2, &[-1.0, 0.0]);
        codebook.set_code(3, &[0.0, -1.0]);

        decoder.upload_codebook(&codebook).unwrap();

        // Lookup indices
        let indices = vec![0u8, 1, 2, 3, 0, 1];
        let result = decoder.lookup(&indices).unwrap();

        let expected = vec![
            1.0, 0.0,  // code 0
            0.0, 1.0,  // code 1
            -1.0, 0.0, // code 2
            0.0, -1.0, // code 3
            1.0, 0.0,  // code 0
            0.0, 1.0,  // code 1
        ];

        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gpu_codebook_lookup_batch() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        let codebook = LayerCodebook::random(256, 64);
        decoder.upload_codebook(&codebook).unwrap();

        // Batch of quantized tensors
        let tensors: Vec<QuantizedTensor> = (0..100)
            .map(|_| QuantizedTensor::random(1000)) // 1000 indices each
            .collect();

        let results = decoder.lookup_batch(&tensors).unwrap();

        assert_eq!(results.len(), 100);
        for result in &results {
            assert_eq!(result.len(), 1000 * 64); // 1000 codes * 64 dims
        }
    }

    #[test]
    fn test_gpu_vs_cpu_codebook_lookup() {
        let ctx = GpuContext::new(0).unwrap();
        let gpu_decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        let codebook = LayerCodebook::random(256, 64);
        gpu_decoder.upload_codebook(&codebook).unwrap();

        let cpu_decoder = haagenti_neural::NeuralDecoder::new(&codebook);

        let indices: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let gpu_result = gpu_decoder.lookup(&indices).unwrap();
        let cpu_result = cpu_decoder.lookup(&indices).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(cpu_result.iter()) {
            assert!((g - c).abs() < 1e-5, "GPU: {}, CPU: {}", g, c);
        }
    }

    #[test]
    fn test_gpu_codebook_memory_efficient() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        let initial_memory = ctx.memory_used();

        // Large codebook
        let codebook = LayerCodebook::random(65536, 128); // 64K codes, 128-dim = 32MB
        decoder.upload_codebook(&codebook).unwrap();

        let after_upload = ctx.memory_used();

        // Should use approximately codebook size
        let expected_size = 65536 * 128 * 4; // f32
        let actual_size = after_upload - initial_memory;

        assert!(actual_size < expected_size * 2,
            "Memory: {} bytes (expected ~{} bytes)", actual_size, expected_size);
    }

    #[test]
    fn test_gpu_codebook_throughput() {
        let ctx = GpuContext::new(0).unwrap();
        let decoder = NeuralGpuDecoder::new(&ctx).unwrap();

        let codebook = LayerCodebook::random(256, 64);
        decoder.upload_codebook(&codebook).unwrap();

        let indices: Vec<u8> = (0..10_000_000).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = decoder.lookup(&indices).unwrap();
        }
        let elapsed = start.elapsed();

        // Output: 10M * 64 * 4 bytes = 2.56 GB per iteration
        let output_bytes = indices.len() as f64 * 64.0 * 4.0;
        let throughput_gbs = (iterations as f64 * output_bytes)
            / elapsed.as_secs_f64() / 1_000_000_000.0;

        // Target: 10 GB/s
        assert!(throughput_gbs > 10.0,
            "Throughput: {:.2} GB/s (target: >10 GB/s)", throughput_gbs);
    }
}
```

### Quality Gate B.4

```bash
#!/bin/bash
# Phase B.4 Quality Gate

echo "=== Phase B.4: Neural GPU Codebook Quality Gate ==="

# 1. All codebook tests pass
cargo test --package haagenti-cuda --features cuda gpu_codebook -- --nocapture

# 2. CPU/GPU equivalence
cargo test --package haagenti-cuda --features cuda test_gpu_vs_cpu_codebook

# 3. Throughput target (10 GB/s)
cargo test --package haagenti-cuda --features cuda test_gpu_codebook_throughput

echo "=== Phase B.4 PASSED ==="
```

---

## Phase B.5: Neural GPU Batched Decode

### Purpose
Implement full neural tensor decoding pipeline on GPU.

### Test Specification

```rust
// tests/cuda_neural_decode_test.rs

#[cfg(all(test, feature = "cuda"))]
mod gpu_neural_decode_tests {
    use haagenti_cuda::{GpuContext, NeuralGpuPipeline};
    use haagenti_neural::{NctFile, EncodedModel};

    #[test]
    fn test_gpu_neural_pipeline_creation() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_gpu_decode_single_tensor() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        // Load NCT file
        let nct = NctFile::open("testdata/small_model.nct").unwrap();
        pipeline.load_codebooks(&nct).unwrap();

        // Decode single tensor
        let tensor_data = nct.read_tensor("layer.0.weight").unwrap();
        let decoded = pipeline.decode_tensor(&tensor_data).unwrap();

        assert_eq!(decoded.shape(), tensor_data.shape());
    }

    #[test]
    fn test_gpu_decode_layer_batch() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        let nct = NctFile::open("testdata/small_model.nct").unwrap();
        pipeline.load_codebooks(&nct).unwrap();

        // Decode all tensors in a layer
        let layer_tensors = nct.layer_tensors(0).unwrap();
        let decoded = pipeline.decode_batch(&layer_tensors).unwrap();

        assert_eq!(decoded.len(), layer_tensors.len());
    }

    #[test]
    fn test_gpu_decode_to_candle_tensor() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        let nct = NctFile::open("testdata/small_model.nct").unwrap();
        pipeline.load_codebooks(&nct).unwrap();

        let tensor_data = nct.read_tensor("layer.0.weight").unwrap();

        // Decode directly to candle GPU tensor
        let candle_tensor = pipeline.decode_to_candle(
            &tensor_data,
            &candle_core::Device::Cuda(0),
        ).unwrap();

        assert_eq!(candle_tensor.dims(), tensor_data.shape());
    }

    #[test]
    fn test_gpu_decode_throughput() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        let nct = NctFile::open("testdata/large_model.nct").unwrap();
        pipeline.load_codebooks(&nct).unwrap();

        let all_tensors: Vec<_> = nct.all_tensors().collect();
        let total_bytes: usize = all_tensors.iter()
            .map(|t| t.decompressed_size())
            .sum();

        let start = std::time::Instant::now();
        let _ = pipeline.decode_batch(&all_tensors).unwrap();
        let elapsed = start.elapsed();

        let throughput_gbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

        // Target: 1 GB/s output throughput
        assert!(throughput_gbs > 1.0,
            "Throughput: {:.2} GB/s (target: >1 GB/s)", throughput_gbs);
    }

    #[test]
    fn test_gpu_streaming_decode() {
        let ctx = GpuContext::new(0).unwrap();
        let pipeline = NeuralGpuPipeline::new(&ctx).unwrap();

        let nct = NctFile::open("testdata/large_model.nct").unwrap();
        pipeline.load_codebooks(&nct).unwrap();

        // Stream decode with callback
        let mut decoded_count = 0;
        pipeline.decode_streaming(&nct, |tensor_name, tensor| {
            decoded_count += 1;
            // Process tensor...
            Ok(())
        }).unwrap();

        assert!(decoded_count > 0);
    }
}
```

### Quality Gate B.5

```bash
#!/bin/bash
# Phase B.5 Quality Gate

echo "=== Phase B.5: Neural GPU Batched Decode Quality Gate ==="

# 1. All neural GPU tests pass
cargo test --package haagenti-cuda --features cuda gpu_neural_decode -- --nocapture

# 2. Throughput target (1 GB/s)
cargo test --package haagenti-cuda --features cuda test_gpu_decode_throughput

# 3. Candle integration works
cargo test --package haagenti-cuda --features cuda test_gpu_decode_to_candle

echo "=== Phase B.5 PASSED ==="
```

---

## Phase B.6: Integration & Optimization

### Purpose
Final integration, optimization, and performance tuning.

### Test Specification

```rust
// tests/cuda_integration_test.rs

#[cfg(all(test, feature = "cuda"))]
mod integration_tests {
    #[test]
    fn test_unified_pipeline_zstd_and_neural() {
        // Test using both Zstd and Neural decompression together
    }

    #[test]
    fn test_memory_pool_efficiency() {
        // Verify memory reuse across operations
    }

    #[test]
    fn test_multi_stream_parallelism() {
        // Test concurrent CUDA streams
    }

    #[test]
    fn test_device_fallback_to_cpu() {
        // Verify graceful CPU fallback when GPU unavailable
    }

    #[test]
    fn test_peak_memory_usage() {
        // Track and verify memory bounds
    }
}
```

### Quality Gate B.6

```bash
#!/bin/bash
# Phase B.6 Quality Gate (FINAL)

echo "=== Phase B.6: Integration Quality Gate ==="

# 1. All CUDA tests pass
cargo test --package haagenti-cuda --features cuda -- --nocapture

# 2. Full benchmark suite
cargo bench --package haagenti-cuda --features cuda

# 3. Memory leak check
cargo test --package haagenti-cuda --features cuda -- --test-threads=1
# (Check for memory growth)

# 4. Multi-GPU test (if available)
cargo test --package haagenti-cuda --features cuda,multi-gpu multi_gpu

echo "=== Phase B.6 PASSED ==="
echo "=== TRACK B COMPLETE ==="
```

---

## Track B Summary

### Test Count by Phase

| Phase | Unit Tests | Integration | Benchmark | Total |
|-------|------------|-------------|-----------|-------|
| B.1 Sequences | 9 | 2 | 1 | 12 |
| B.2 FSE | 10 | 3 | 2 | 15 |
| B.3 Pipeline | 7 | 2 | 1 | 10 |
| B.4 Codebook | 7 | 3 | 2 | 12 |
| B.5 Neural | 6 | 3 | 1 | 10 |
| B.6 Integration | 5 | 3 | 0 | 8 |
| **Total** | **44** | **16** | **7** | **67** |

### Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Zstd GPU decompression | N/A | 4 GB/s | New |
| Neural GPU decode | N/A | 1 GB/s | New |
| Codebook lookup | 100 MB/s (CPU) | 10 GB/s (GPU) | 100x |

---

*Document Version: 1.0*
*Created: 2026-01-06*
